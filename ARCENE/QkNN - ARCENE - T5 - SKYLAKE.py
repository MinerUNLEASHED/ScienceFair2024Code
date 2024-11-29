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
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
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
    features_array = features.values  # Convert to NumPy array
    normalized_features = features_array / np.linalg.norm(features_array, axis=1, keepdims=True)
    norm_end_time = time.time()
    print(f"Normalization completed in {norm_end_time - norm_start_time:.2f} seconds.")
else:
    normalized_features = None

# Broadcast normalized features and labels
normalized_features = comm.bcast(normalized_features, root=0)
labels = comm.bcast(labels.values, root=0)  # Broadcast labels as a NumPy array

# Split data into training and testing sets on root process
if rank == 0:
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, labels, test_size=0.2, random_state=42
    )
    # Convert to NumPy arrays (if not already)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
else:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

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
    X1 = np.array(X1)
    X2 = np.array(X2) if X2 is not None else None
    return quantum_kernel.evaluate(x_vec=X1, y_vec=X2)

# Quantum k-NN function
def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        distances = 1 - test_kernel_matrix[i, :]  # Calculate "distance" (inverse of similarity)
        k_nearest_indices = distances.argsort()[:k]  # Get indices of k-nearest neighbors
        k_nearest_labels = np.array(y_train)[k_nearest_indices]  # Get labels of nearest neighbors
        predicted_label = mode(k_nearest_labels).mode[0]
        predictions.append(predicted_label)
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
    # Stack train_kernels vertically to form train_kernel_matrix
    train_kernel_matrix = np.vstack(train_kernels)  # Shape: (n_train_samples, n_train_samples)

    # Concatenate test_kernels horizontally to form test_kernel_matrix
    test_kernel_matrix = np.hstack(test_kernels)  # Shape: (n_test_samples, n_train_samples)

    # Perform k-NN on root process
    predictions = quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train)

    # Compute accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.2%}")

    # Save predictions
    output_file = os.path.join(base_dir, f"Predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
