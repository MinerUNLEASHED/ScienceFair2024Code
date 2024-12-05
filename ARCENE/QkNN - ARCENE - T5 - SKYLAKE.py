import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import __version__ as scipy_version
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mpi4py import MPI
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

print("Past imports, please work")
# Set a small epsilon value to avoid division by zero
EPSILON = 1e-10

# Set environment variables for Skylake nodes
try:
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    print("Environment variables set for Skylake optimization.", flush=True)
except Exception as e:
    print(f"Error setting environment variables: {e}", file=sys.stderr, flush=True)
    sys.exit(-2)

# Initialize MPI
try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Initialized MPI: Rank {rank}, Size {size}", flush=True)
except Exception as e:
    print(f"Error initializing MPI: {e}", file=sys.stderr, flush=True)
    sys.exit(-2)

# Logging utility
def log(message):
    """
    Logs a message with the MPI rank.

    Parameters:
    message (str): The message to log.
    """
    print(f"[Rank {rank}] {message}", flush=True)

def abort_on_error(error_message):
    """
    Logs an error message and aborts the MPI communication.

    Parameters:
    error_message (str): The error message to log.
    """
    print(f"[Rank {rank}] Error: {error_message}", file=sys.stderr, flush=True)
    comm.Abort()
    sys.exit(-2)

log("Program started...")

# Load dataset
try:
    if rank == 0:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'Standardized_ARCENE_Dataset.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        dataset = pd.read_csv(file_path).values  # Convert to NumPy array
        log("Dataset loaded and converted to NumPy array.")
    else:
        dataset = None
    dataset = comm.bcast(dataset, root=0)
    log("Dataset broadcasted.")
except Exception as e:
    abort_on_error(f"Error loading dataset: {e}")

# Separate features and labels
try:
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    log("Features and labels extracted.")
except Exception as e:
    abort_on_error(f"Error processing dataset: {e}")

# Normalize dataset
try:
    if rank == 0:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = EPSILON  # Avoid division by zero
        normalized_features = features / norms
        log("Dataset normalized.")
    else:
        normalized_features = None
    normalized_features = comm.bcast(normalized_features, root=0)
except Exception as e:
    abort_on_error(f"Error normalizing dataset: {e}")

# Split data into training and testing sets
try:
    if rank == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            normalized_features, labels, test_size=0.2, random_state=42
        )
        log(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    else:
        X_train, X_test, y_train, y_test = None, None, None, None
    X_train = comm.bcast(X_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_train = comm.bcast(y_train, root=0)
    y_test = comm.bcast(y_test, root=0)
except Exception as e:
    abort_on_error(f"Error splitting dataset: {e}")

# Function to create batches of features
def create_batches(data, batch_size):
    try:
        num_features = data.shape[1]
        return [data[:, i:i + batch_size] for i in range(0, num_features, batch_size)]
    except Exception as e:
        abort_on_error(f"Error creating feature batches: {e}")

batch_size = 30
try:
    train_batches = create_batches(X_train, batch_size)
    test_batches = create_batches(X_test, batch_size)
    log(f"Data split into {len(train_batches)} batches of {batch_size} features each.")
except Exception as e:
    abort_on_error(f"Error splitting data into batches: {e}")

# Function to compute quantum kernel matrices
def compute_kernels(train_batches, test_batches=None):
    try:
        total_kernel_train = None
        total_kernel_test = None
        for idx in range(rank, len(train_batches), size):
            batch_train = train_batches[idx]
            log(f"Processing batch {idx + 1}/{len(train_batches)}")
            feature_map = ZZFeatureMap(feature_dimension=batch_train.shape[1], reps=2, entanglement="linear")
            quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
            kernel_train = quantum_kernel.evaluate(x_vec=batch_train)
            if total_kernel_train is None:
                total_kernel_train = kernel_train
            else:
                total_kernel_train += kernel_train
            if test_batches:
                batch_test = test_batches[idx]
                kernel_test = quantum_kernel.evaluate(x_vec=batch_test, y_vec=batch_train)
                if total_kernel_test is None:
                    total_kernel_test = kernel_test
                else:
                    total_kernel_test += kernel_test
        total_kernel_train = comm.reduce(total_kernel_train, op=MPI.SUM, root=0)
        total_kernel_test = comm.reduce(total_kernel_test, op=MPI.SUM, root=0) if test_batches else None
        return total_kernel_train, total_kernel_test
    except Exception as e:
        abort_on_error(f"Error computing quantum kernels: {e}")

try:
    kernel_start_time = time.time()
    total_kernel_train, total_kernel_test = compute_kernels(train_batches, test_batches)
    kernel_end_time = time.time()
    if rank == 0:
        log(f"Kernel computation completed in {kernel_end_time - kernel_start_time:.2f} seconds.")
except Exception as e:
    abort_on_error(f"Error computing kernels: {e}")
