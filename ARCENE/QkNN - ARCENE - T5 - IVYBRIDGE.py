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
        distances = 1 - test_kernel_matrix[i, :]  # Calculate "distance" (inverse of similarity)
        k_nearest_indices = distances.argsort()[:k]  # Get indices of k-nearest neighbors
        k_nearest_labels = y_train.iloc[k_nearest_indices].values  # Get labels of nearest neighbors
        predicted_label = mode(k_nearest_labels, keepdims=False).mode[0]  # Fix keepdims
        predictions.append(predicted_label)
    return np.array(predictions)


# All processes create the same feature map
feature_map = create_feature_map(local_features.shape[1])

# Compute train kernel on local data
local_train_kernel = compute_kernel(feature_map, local_features)

# Ensure all processes send non-empty matrices (placeholder if empty)
if local_train_kernel.size == 0:
    local_train_kernel = np.zeros((1, local_features.shape[1]))

# Gather train kernels on root process
train_kernel_matrix = comm.gather(local_train_kernel, root=0)
if rank == 0:
    # Filter out empty matrices before stacking
    train_kernel_matrix = np.vstack([matrix for matrix in train_kernel_matrix if matrix.size > 0])

# Compute test kernel on root process
if rank == 0:
    test_kernel_matrix = compute_kernel(feature_map, normalized_features, normalized_features)
else:
    test_kernel_matrix = None

# Broadcast test kernel to all processes
test_kernel_matrix = comm.bcast(test_kernel_matrix, root=0)

# Perform k-NN on local data using global labels
local_predictions = quantum_knn(test_kernel_matrix, local_train_kernel, labels)

# Gather predictions on root process
predictions = comm.gather(local_predictions, root=0)

# Root process saves results
if rank == 0:
    predictions = np.hstack(predictions)
    output_file = os.path.join(base_dir, f"Predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
