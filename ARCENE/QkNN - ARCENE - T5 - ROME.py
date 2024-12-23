import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mpi4py import MPI
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Constants
EPSILON = 1e-10  # Small epsilon to avoid division by zero

# Set environment variables
try:
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    print("Environment variables set for Milan optimization.", flush=True)
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
    """Logs a message with MPI rank."""
    print(f"[Rank {rank}] {message}", flush=True)

# Abort utility
def abort_on_error(error_message):
    """Logs an error and exits with code -2."""
    print(f"[Rank {rank}] Error: {error_message}", file=sys.stderr, flush=True)
    comm.Abort()

log("Program started...")

# Load dataset
try:
    if rank == 0:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        file_path = os.path.join(base_dir, 'Standardized_ARCENE_Dataset.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        dataset = pd.read_csv(file_path).values.astype(np.float64)  # Convert to NumPy array with specified data type
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
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    zero_norms = (norms == 0).flatten()
    if np.any(zero_norms):
        norms[zero_norms] = EPSILON  # Avoid division by zero
    normalized_features = features / norms
    log("Dataset normalized.")
except Exception as e:
    abort_on_error(f"Error normalizing dataset: {e}")

# Broadcast normalized features and labels
try:
    normalized_features = comm.bcast(normalized_features, root=0)
    labels = comm.bcast(labels, root=0)
except Exception as e:
    abort_on_error(f"Error broadcasting normalized data: {e}")

# Split data into training and testing sets
try:
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, labels, test_size=0.2, random_state=42
    )
    log(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
except Exception as e:
    abort_on_error(f"Error splitting dataset: {e}")

# Broadcast training and testing data
try:
    X_train = comm.bcast(X_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_train = comm.bcast(y_train, root=0)
    y_test = comm.bcast(y_test, root=0)
except Exception as e:
    abort_on_error(f"Error broadcasting training/testing data: {e}")

# Create batches
def create_batches(data, batch_size):
    try:
        num_samples = data.shape[0]
        return [data[i:i + batch_size, :] for i in range(0, num_samples, batch_size)]
    except Exception as e:
        abort_on_error(f"Error creating data batches: {e}")

batch_size = 30
try:
    train_batches = create_batches(X_train, batch_size)
    test_batches = create_batches(X_test, batch_size)
    log(f"Data split into {len(train_batches)} batches of {batch_size} samples each.")
except Exception as e:
    abort_on_error(f"Error splitting data into batches: {e}")

# Compute kernels
def compute_kernels(train_batches, test_batches=None):
    try:
        total_kernel_train = None
        total_kernel_test = None
        batch_count = 0  # For averaging
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
            batch_count += 1
            if test_batches:
                batch_test = test_batches[idx]
                kernel_test = quantum_kernel.evaluate(x_vec=batch_test, y_vec=batch_train)
                if total_kernel_test is None:
                    total_kernel_test = kernel_test
                else:
                    total_kernel_test += kernel_test
        # Reduce total_kernel_train and total_kernel_test across all processes
        total_kernel_train = comm.reduce(total_kernel_train, op=MPI.SUM, root=0)
        total_kernel_test = comm.reduce(total_kernel_test, op=MPI.SUM, root=0) if test_batches else None
        # Reduce batch_count across all processes
        total_batch_count = comm.reduce(batch_count, op=MPI.SUM, root=0)
        if rank == 0:
            # Average the kernels
            total_kernel_train /= total_batch_count
            if total_kernel_test is not None:
                total_kernel_test /= total_batch_count
        return total_kernel_train, total_kernel_test
    except Exception as e:
        abort_on_error(f"Error computing quantum kernels: {e}")

# Perform predictions and save results
try:
    kernel_start_time = time.time()
    total_kernel_train, total_kernel_test = compute_kernels(train_batches, test_batches)
    kernel_end_time = time.time()
    if rank == 0:
        log(f"Kernel computation completed in {kernel_end_time - kernel_start_time:.2f} seconds.")

        # Perform predictions for different k values
        results = []
        test_size = total_kernel_test.shape[0]
        k_values = range(1, test_size + 1)

        run_index = 1  # Initialize run index

        for k in k_values:
            qknn_start_time = time.time()
            predictions = []
            for i in range(test_size):
                distances = 1 - total_kernel_test[i, :]
                k_indices = distances.argsort()[:k]
                k_labels = y_train[k_indices]
                predicted_label = mode(k_labels, axis=None).mode[0]
                predictions.append(predicted_label)
            qknn_end_time = time.time()

            # Evaluate accuracy
            accuracy = accuracy_score(y_test, predictions)

            # Record results
            results.append({
                'Run Index': run_index,
                'k': k,
                'Normalization Time': 'N/A',  # Update if needed
                'Feature Map Time': 'N/A',    # Update if needed
                'Train Kernel Time': 'N/A',   # Update if needed
                'Test Kernel Time': 'N/A',    # Update if needed
                'QkNN Time': qknn_end_time - qknn_start_time,
                'Accuracy': accuracy,
                'Date': datetime.now().strftime("%m/%d/%Y"),
                'Time': datetime.now().strftime("%H:%M:%S")
            })
            run_index += 1  # Increment run index

        # Save results to a CSV file
        try:
            try:
                output_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                output_dir = os.getcwd()
            output_file = os.path.join(output_dir, f"QkNN_Results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            log(f"Results saved to {output_file}.")
        except Exception as e:
            abort_on_error(f"Error saving results: {e}")
    else:
        # Ensure all processes reach the end
        pass
except Exception as e:
    abort_on_error(f"Error during predictions or saving results: {e}")

# Ensure all processes reach the end before finishing
try:
    comm.Barrier()
    if rank == 0:
        log("All processes have reached the end of execution.")
except Exception as e:
    abort_on_error(f"Error during MPI Barrier synchronization: {e}")
