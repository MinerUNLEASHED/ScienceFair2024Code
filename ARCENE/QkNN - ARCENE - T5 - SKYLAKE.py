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
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # Add epsilon to zero norms to avoid division by zero
    normalized_features = features / norms
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

# Function to create subcircuits for a subset of features
def create_subcircuits(data, num_features_per_circuit):
    """
    Split data into smaller feature subsets and create circuits.
    """
    num_subcircuits = data.shape[1] // num_features_per_circuit + int(data.shape[1] % num_features_per_circuit > 0)
    subcircuits = []

    for i in range(num_subcircuits):
        start_idx = i * num_features_per_circuit
        end_idx = min((i + 1) * num_features_per_circuit, data.shape[1])
        subset = data[:, start_idx:end_idx]

        feature_map = ZZFeatureMap(feature_dimension=subset.shape[1], reps=2, entanglement="linear")
        subcircuits.append((subset, feature_map))

    return subcircuits

# Function to compute kernel for subcircuits
def compute_subcircuit_kernel(subcircuits_train, subcircuits_test=None):
    """
    Compute quantum kernels for subcircuits and aggregate results.
    """
    total_kernel_train = None
    total_kernel_test = None

    for idx, (subset_train, feature_map) in enumerate(subcircuits_train):
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

        # Compute training kernel
        kernel_train = quantum_kernel.evaluate(x_vec=subset_train, y_vec=subset_train)
        total_kernel_train = kernel_train if total_kernel_train is None else total_kernel_train + kernel_train

        # Compute test kernel if applicable
        if subcircuits_test is not None:
            subset_test = subcircuits_test[idx][0]
            kernel_test = quantum_kernel.evaluate(x_vec=subset_test, y_vec=subset_train)
            total_kernel_test = kernel_test if total_kernel_test is None else total_kernel_test + kernel_test

    return total_kernel_train, total_kernel_test

# Split data into subcircuits
num_features_per_circuit = 30  # Define the number of features per circuit to avoid qubit overflow
subcircuits_train = create_subcircuits(X_train, num_features_per_circuit)
subcircuits_test = create_subcircuits(X_test, num_features_per_circuit)

# Distribute subcircuits among processes
num_subcircuits = len(subcircuits_train)
subcircuits_per_process = num_subcircuits // size
remainder = num_subcircuits % size

if rank < remainder:
    start_idx = rank * (subcircuits_per_process + 1)
    end_idx = start_idx + subcircuits_per_process + 1
else:
    start_idx = rank * subcircuits_per_process + remainder
    end_idx = start_idx + subcircuits_per_process

local_subcircuits_train = subcircuits_train[start_idx:end_idx]
local_subcircuits_test = subcircuits_test[start_idx:end_idx]

# Compute kernels for subcircuits
local_train_kernel, local_test_kernel = compute_subcircuit_kernel(local_subcircuits_train, local_subcircuits_test)

# Sum the kernel matrices across all processes
total_kernel_train = comm.reduce(local_train_kernel, op=MPI.SUM, root=0)
total_kernel_test = comm.reduce(local_test_kernel, op=MPI.SUM, root=0)

# Quantum k-NN function
def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        # Compute distances using kernel trick (always non-negative)
        distances = np.sqrt(
            np.diag(train_kernel_matrix) + test_kernel_matrix[i, i] - 2 * test_kernel_matrix[i, :]
        )
        k_nearest_indices = distances.argsort()[:k]
        k_nearest_labels = np.array(y_train)[k_nearest_indices]
        predictions.append(mode(k_nearest_labels).mode[0])
    return np.array(predictions)

# Root process assembles full train and test kernel matrices
if rank == 0:
    # Proceed with k-NN classification
    predictions = quantum_knn(total_kernel_test, total_kernel_train, y_train)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.2%}")

    # Save predictions
    output_file = os.path.join(base_dir, f"Predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
