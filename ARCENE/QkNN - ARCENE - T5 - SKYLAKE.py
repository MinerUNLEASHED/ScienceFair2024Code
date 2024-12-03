import os
print("Imported os")

import sys
print("Imported sys")

import time
print("Imported time")

from datetime import datetime
print("Imported datetime")

try:
    import numpy as np
    print("Imported numpy")
    import pandas as pd
    print("Imported pandas")
    from mpi4py import MPI
    print("Imported mpi4py")
    from qiskit.circuit.library import ZZFeatureMap
    print("Imported ZZFeatureMap from qiskit")
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    print("Imported FidelityQuantumKernel from qiskit_machine_learning")
    from scipy.stats import mode
    print("Imported mode from scipy.stats")
    from sklearn.metrics import accuracy_score
    print("Imported accuracy_score from sklearn.metrics")
    from sklearn.model_selection import train_test_split
    print("Imported train_test_split from sklearn.model_selection")
except Exception as e:
    print(f"Error during imports: {e}", file=sys.stderr)
    sys.exit(-2)

# Set environment variables for thread optimization
try:
    os.environ["MKL_NUM_THREADS"] = "1"
    print("Set MKL_NUM_THREADS to 1")
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    print("Set NUMEXPR_NUM_THREADS to 1")
    os.environ["OMP_NUM_THREADS"] = "1"
    print("Set OMP_NUM_THREADS to 1")
except Exception as e:
    print(f"Error setting environment variables: {e}", file=sys.stderr)
    sys.exit(-2)

# Initialize MPI
try:
    comm = MPI.COMM_WORLD
    print("Initialized MPI.COMM_WORLD")
    rank = comm.Get_rank()
    print(f"Obtained rank: {rank}")
    size = comm.Get_size()
    print(f"Obtained size: {size}")
except Exception as e:
    print(f"Error initializing MPI: {e}", file=sys.stderr)
    sys.exit(-2)

def abort_on_error(error_message):
    """Abort all processes upon encountering an error."""
    print(f"Rank {rank} encountered an error: {error_message}", file=sys.stderr)
    comm.Abort()
    sys.exit(-2)

# Load dataset on root process
try:
    if rank == 0:
        print("Rank 0: Attempting to load dataset")
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"Base directory determined: {base_dir}")
        except NameError:
            base_dir = os.getcwd()
            print(f"Base directory (fallback) determined: {base_dir}")
        file_path = os.path.join(base_dir, 'Standardized_ARCENE_Dataset.csv')
        print(f"Dataset path resolved: {file_path}")
        if not os.path.exists(file_path):
            abort_on_error(f"Dataset file not found at {file_path}")
        print(f"Loading dataset from {file_path}")
        dataset = pd.read_csv(file_path)
        print("Dataset loaded successfully")
    else:
        dataset = None
        print(f"Rank {rank}: Dataset set to None")
except Exception as e:
    abort_on_error(f"Error loading dataset: {e}")

# Broadcast dataset to all processes
try:
    print("Broadcasting dataset")
    dataset = comm.bcast(dataset, root=0)
    print("Broadcast completed")
except Exception as e:
    abort_on_error(f"Failed to broadcast dataset: {e}")

# Separate features and labels
try:
    print("Separating features and labels")
    features = dataset.iloc[:, :-1].to_numpy()
    print("Features extracted")
    labels = dataset.iloc[:, -1].to_numpy()
    print("Labels extracted")
except Exception as e:
    abort_on_error(f"Failed to separate features and labels: {e}")

# Normalize dataset on root process
try:
    if rank == 0:
        print("Rank 0: Normalizing dataset")
        norm_start_time = time.time()
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        print("Norms calculated")
        norms[norms == 0] = 1e-10  # Avoid division by zero
        print("Adjusted zero norms")
        normalized_features = features / norms
        print("Features normalized")
        norm_end_time = time.time()
        print(f"Normalization completed in {norm_end_time - norm_start_time:.2f} seconds.")
    else:
        normalized_features = None
        print(f"Rank {rank}: Normalized features set to None")
except Exception as e:
    abort_on_error(f"Error normalizing dataset: {e}")

# Broadcast normalized features and labels
try:
    print("Broadcasting normalized features and labels")
    normalized_features = comm.bcast(normalized_features, root=0)
    print("Normalized features broadcasted")
    labels = comm.bcast(labels, root=0)
    print("Labels broadcasted")
except Exception as e:
    abort_on_error(f"Failed to broadcast normalized data: {e}")

# Split data into training and testing sets on root process
try:
    if rank == 0:
        print("Rank 0: Splitting data into training and testing sets")
        X_train, X_test, y_train, y_test = train_test_split(
            normalized_features, labels, test_size=0.2, random_state=42
        )
        print("Data split into training and testing sets")
    else:
        X_train, X_test, y_train, y_test = None, None, None, None
        print(f"Rank {rank}: Training and testing data set to None")
except Exception as e:
    abort_on_error(f"Failed to split data: {e}")

# Broadcast X_train, X_test, y_train, and y_test to all processes
try:
    print("Broadcasting training and testing data")
    X_train = comm.bcast(X_train, root=0)
    print("Broadcasted X_train")
    X_test = comm.bcast(X_test, root=0)
    print("Broadcasted X_test")
    y_train = comm.bcast(y_train, root=0)
    print("Broadcasted y_train")
    y_test = comm.bcast(y_test, root=0)
    print("Broadcasted y_test")
except Exception as e:
    abort_on_error(f"Failed to broadcast training/testing data: {e}")

def create_subcircuits(data, num_features_per_circuit):
    """
    Split data into smaller feature subsets and create quantum feature maps.
    """
    print(f"Creating subcircuits with {num_features_per_circuit} features per circuit")
    num_subcircuits = data.shape[1] // num_features_per_circuit + int(data.shape[1] % num_features_per_circuit > 0)
    print(f"Number of subcircuits: {num_subcircuits}")
    subcircuits = []
    for i in range(num_subcircuits):
        start_idx = i * num_features_per_circuit
        end_idx = min((i + 1) * num_features_per_circuit, data.shape[1])
        print(f"Processing features {start_idx} to {end_idx}")
        subset = data[:, start_idx:end_idx]
        feature_map = ZZFeatureMap(feature_dimension=subset.shape[1], reps=2, entanglement="linear")
        print("Created feature map")
        subcircuits.append((subset, feature_map))
    print("Subcircuits created successfully")
    return subcircuits

try:
    features_per_circuit = 30  # Default number of features per circuit
    print(f"Setting features_per_circuit to {features_per_circuit}")
    print("Creating training subcircuits")
    subcircuits_train = create_subcircuits(X_train, features_per_circuit)
    print("Training subcircuits created")
    print("Creating testing subcircuits")
    subcircuits_test = create_subcircuits(X_test, features_per_circuit)
    print("Testing subcircuits created")
except Exception as e:
    abort_on_error(f"Error splitting data into subcircuits: {e}")
