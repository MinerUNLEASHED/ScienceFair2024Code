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

# Set environment variables for thread optimization
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load dataset on root process
if rank == 0:
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    file_path = os.path.join(base_dir, 'Standardized_ARCENE_Dataset.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    print(f"Loading dataset from {file_path}")
    dataset = pd.read_csv(file_path)
else:
    dataset = None

# Broadcast dataset to all processes
dataset = comm.bcast(dataset, root=0)

# Separate features and labels
features = dataset.iloc[:, :-1].to_numpy()  # Ensure data is in NumPy array format
labels = dataset.iloc[:, -1].to_numpy()

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

# Split data into training and testing sets on root process
if rank == 0:
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, labels, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = None, None, None, None

# Broadcast X_train, X_test, y_train, and y_test to all processes
X_train = comm.bcast(X_train, root=0)
X_test = comm.bcast(X_test, root=0)
y_train = comm.bcast(y_train, root=0)
y_test = comm.bcast(y_test, root=0)

# Split X_train across processes
split_data = np.array_split(X_train, size, axis=0)
split_labels = np.array_split(y_train, size, axis=0)
local_X_train = split_data[rank]
local_y_train = split_labels[rank]

# Create feature map
def create_feature_map(num_features, reps=2, entanglement="linear"):
    return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement=entanglement)

# Compute quantum kernel
def compute_kernel(feature_map, X1, X2=None):
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    X2 = X2 if X2 is not None else X1  # Default to self-similarity
    return quantum_kernel.evaluate(x_vec=X1, y_vec=X2)

# Quantum k-NN function
def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        distances = 1 - test_kernel_matrix[i, :]  # Calculate "distance" (inverse of similarity)
        k_nearest_indices = distances.argsort()[:k]  # Get indices of k-nearest neighbors
        k_nearest_labels = np.array(y_train)[k_nearest_indices]  # Get labels of nearest neighbors
        predictions.append(mode(k_nearest_labels).mode[0])
    return np.array(predictions)

# All processes create the same feature map
feature_map = create_feature_map(local_X_train.shape[1])

# Each process computes local train kernel between local_X_train and X_train
local_train_kernel = compute_kernel(feature_map, local_X_train, X_train)

# Each process computes local test kernel between X_test and local_X_train
local_test_kernel = compute_kernel(feature_map, X_test, local_X_train)

# Gather local_train_kernel to root process
train_kernels = comm.gather(local_train_kernel, root=0)

# Gather local_test_kernel to root process
test_kernels = comm.gather(local_test_kernel, root=0)

# Root process assembles full train and test kernel matrices
if rank == 0:
    train_kernel_matrix = np.vstack(train_kernels)
    test_kernel_matrix = np.hstack(test_kernels)

    predictions = quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.2%}")

    output_file = os.path.join(base_dir, f"Predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
