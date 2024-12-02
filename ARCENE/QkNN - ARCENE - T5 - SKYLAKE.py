import os
import sys
import time
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    from mpi4py import MPI
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from scipy.stats import mode
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
except Exception as e:
    print(f"Error during imports: {e}", file=sys.stderr)
    sys.exit(1)

# Set environment variables for thread optimization
try:
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
except Exception as e:
    print(f"Error setting environment variables: {e}", file=sys.stderr)
    sys.exit(1)

# Initialize MPI
try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except Exception as e:
    print(f"Error initializing MPI: {e}", file=sys.stderr)
    sys.exit(1)

def abort_on_error(error_message):
    """Abort all processes upon encountering an error."""
    print(f"Rank {rank} encountered an error: {error_message}", file=sys.stderr)
    comm.Abort()

# Load dataset on root process
try:
    if rank == 0:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        file_path = os.path.join(base_dir, 'Standardized_ARCENE_Dataset.csv')
        if not os.path.exists(file_path):
            abort_on_error(f"Dataset file not found at {file_path}")
        print(f"Loading dataset from {file_path}")
        dataset = pd.read_csv(file_path)
    else:
        dataset = None
except Exception as e:
    abort_on_error(f"Error loading dataset: {e}")

# Broadcast dataset to all processes
try:
    dataset = comm.bcast(dataset, root=0)
except Exception as e:
    abort_on_error(f"Failed to broadcast dataset: {e}")

# Separate features and labels
try:
    features = dataset.iloc[:, :-1].to_numpy()
    labels = dataset.iloc[:, -1].to_numpy()
except Exception as e:
    abort_on_error(f"Failed to separate features and labels: {e}")

# Normalize dataset on root process
try:
    if rank == 0:
        norm_start_time = time.time()
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # Avoid division by zero
        normalized_features = features / norms
        norm_end_time = time.time()
        print(f"Normalization completed in {norm_end_time - norm_start_time:.2f} seconds.")
    else:
        normalized_features = None
except Exception as e:
    abort_on_error(f"Error normalizing dataset: {e}")

# Broadcast normalized features and labels
try:
    normalized_features = comm.bcast(normalized_features, root=0)
    labels = comm.bcast(labels, root=0)
except Exception as e:
    abort_on_error(f"Failed to broadcast normalized data: {e}")

# Split data into training and testing sets on root process
try:
    if rank == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            normalized_features, labels, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = None, None, None, None
except Exception as e:
    abort_on_error(f"Failed to split data: {e}")

# Broadcast X_train, X_test, y_train, and y_test to all processes
try:
    X_train = comm.bcast(X_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_train = comm.bcast(y_train, root=0)
    y_test = comm.bcast(y_test, root=0)
except Exception as e:
    abort_on_error(f"Failed to broadcast training/testing data: {e}")

def create_subcircuits(data, num_features_per_circuit):
    """
    Split data into smaller feature subsets and create quantum feature maps.
    """
    try:
        num_subcircuits = data.shape[1] // num_features_per_circuit + int(data.shape[1] % num_features_per_circuit > 0)
        subcircuits = []
        for i in range(num_subcircuits):
            start_idx = i * num_features_per_circuit
            end_idx = min((i + 1) * num_features_per_circuit, data.shape[1])
            subset = data[:, start_idx:end_idx]
            feature_map = ZZFeatureMap(feature_dimension=subset.shape[1], reps=2, entanglement="linear")
            subcircuits.append((subset, feature_map))
        return subcircuits
    except Exception as e:
        abort_on_error(f"Failed to create subcircuits: {e}")

def compute_subcircuit_kernel(subcircuits_train, subcircuits_test=None):
    """
    Compute quantum kernels for subcircuits and aggregate results.
    """
    try:
        total_kernel_train = None
        total_kernel_test = None
        for idx, (subset_train, feature_map) in enumerate(subcircuits_train):
            quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
            kernel_train = quantum_kernel.evaluate(x_vec=subset_train, y_vec=subset_train)
            total_kernel_train = kernel_train if total_kernel_train is None else total_kernel_train + kernel_train
            if subcircuits_test is not None:
                subset_test = subcircuits_test[idx][0]
                kernel_test = quantum_kernel.evaluate(x_vec=subset_test, y_vec=subset_train)
                total_kernel_test = kernel_test if total_kernel_test is None else total_kernel_test + kernel_test
        return total_kernel_train, total_kernel_test
    except Exception as e:
        abort_on_error(f"Failed to compute quantum kernel: {e}")

# Split data into subcircuits
try:
    features_per_circuit = 30  # Default number of features per circuit
    subcircuits_train = create_subcircuits(X_train, features_per_circuit)
    subcircuits_test = create_subcircuits(X_test, features_per_circuit)
except Exception as e:
    abort_on_error(f"Error splitting data into subcircuits: {e}")

# Distribute subcircuits among processes
try:
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
except Exception as e:
    abort_on_error(f"Error distributing subcircuits among processes: {e}")

# Compute kernels for subcircuits
try:
    local_train_kernel, local_test_kernel = compute_subcircuit_kernel(local_subcircuits_train, local_subcircuits_test)
except Exception as e:
    abort_on_error(f"Error computing kernels for subcircuits: {e}")

# Sum the kernel matrices across all processes
try:
    total_kernel_train = comm.reduce(local_train_kernel, op=MPI.SUM, root=0)
    total_kernel_test = comm.reduce(local_test_kernel, op=MPI.SUM, root=0)
except Exception as e:
    abort_on_error(f"Error reducing kernel matrices: {e}")

def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k):
    """
    Perform k-NN classification using quantum kernels.
    """
    try:
        predictions = []
        for i in range(test_kernel_matrix.shape[0]):
            distances = np.sqrt(
                np.diag(train_kernel_matrix) + test_kernel_matrix[i, i] - 2 * test_kernel_matrix[i, :]
            )
            k_nearest_indices = distances.argsort()[:k]
            k_nearest_labels = np.array(y_train)[k_nearest_indices]
            predictions.append(mode(k_nearest_labels).mode[0])
        return np.array(predictions)
    except Exception as e:
        abort_on_error(f"Error in quantum k-NN function: {e}")

# Root process assembles full train and test kernel matrices
if rank == 0:
    try:
        predictions = quantum_knn(total_kernel_test, total_kernel_train, y_train, k=3)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Test accuracy: {accuracy:.2%}")

        # Save predictions
        output_file = os.path.join(base_dir, f"Predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        abort_on_error(f"Error during final prediction and saving: {e}")
