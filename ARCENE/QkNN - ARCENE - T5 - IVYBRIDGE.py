import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from mpi4py import MPI

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load dataset on root process
if rank == 0:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'Standardized_ARCENE_Dataset.csv')
    print(f"Loading dataset from {file_path}")
    dataset = pd.read_csv(file_path)
else:
    dataset = None

# Broadcast dataset to all processes
dataset = comm.bcast(dataset, root=0)

# Separate features and labels
features = dataset.iloc[:, :-1]
labels = dataset.iloc[:, -1]

# Normalize dataset on root process
if rank == 0:
    norm_start_time = time.time()
    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    norm_end_time = time.time()
    print(f"Normalization completed in {norm_end_time - norm_start_time:.2f} seconds.")
else:
    normalized_features = None

# Broadcast normalized features and labels
normalized_features = comm.bcast(normalized_features, root=0)
labels = comm.bcast(labels, root=0)

# Split data across processes
split_data = np.array_split(normalized_features, size, axis=0)
split_labels = np.array_split(labels, size, axis=0)
local_features = split_data[rank]
local_labels = split_labels[rank]

# Create feature map
def create_feature_map(num_features, reps=2, entanglement="linear"):
    return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement=entanglement)

# Compute quantum kernel
def compute_kernel(feature_map, X1, X2=None):
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    return quantum_kernel.evaluate(x_vec=X1, y_vec=X2)

# Quantum k-NN function
def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        distances = 1 - test_kernel_matrix[i, :]
        k_nearest_indices = distances.argsort()[:k]
        k_nearest_labels = y_train.iloc[k_nearest_indices].values
        predicted_label = mode(k_nearest_labels, keepdims=True).mode[0]
        predictions.append(predicted_label)
    return np.array(predictions)

# All processes create the same feature map
feature_map = create_feature_map(local_features.shape[1])

# Compute train kernel on local data
local_train_kernel = compute_kernel(feature_map, local_features)

# Gather train kernels on root process
train_kernel_matrix = comm.gather(local_train_kernel, root=0)
if rank == 0:
    train_kernel_matrix = np.vstack(train_kernel_matrix)

# Compute test kernel on root process
if rank == 0:
    test_kernel_matrix = compute_kernel(feature_map, normalized_features, normalized_features)
else:
    test_kernel_matrix = None

# Broadcast test kernel to all processes
test_kernel_matrix = comm.bcast(test_kernel_matrix, root=0)

# Perform k-NN on local data
local_predictions = quantum_knn(test_kernel_matrix, local_train_kernel, local_labels)

# Gather predictions on root process
predictions = comm.gather(local_predictions, root=0)

# Root process saves results
if rank == 0:
    predictions = np.hstack(predictions)
    output_file = os.path.join(base_dir, f"Predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
