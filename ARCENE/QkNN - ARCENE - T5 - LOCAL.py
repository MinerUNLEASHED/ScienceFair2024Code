import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

try:
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
    from sklearn.svm import SVC
    print("Imported SVC from sklearn.svm")
    from qiskit.primitives import Sampler
    print("Imported Sampler from qiskit.primitives")
    from qiskit_aer import AerSimulator
    print("Imported AerSimulator from qiskit_aer")

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

def abort_on_error(error_message):
    """Abort execution upon encountering an error."""
    print(f"Error encountered: {error_message}", file=sys.stderr)
    sys.exit(-2)

# Load dataset
try:
    print("Attempting to load dataset")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory determined: {base_dir}")
    file_path = os.path.join(base_dir, 'Standardized_ARCENE_Dataset.csv')
    print(f"Dataset path resolved: {file_path}")
    if not os.path.exists(file_path):
        abort_on_error(f"Dataset file not found at {file_path}")
    print(f"Loading dataset from {file_path}")
    dataset = pd.read_csv(file_path)
    print("Dataset loaded successfully")
except Exception as e:
    abort_on_error(f"Error loading dataset: {e}")

# Separate features and labels
try:
    print("Separating features and labels")
    features = dataset.iloc[:, :-1].to_numpy()
    print("Features extracted")
    labels = dataset.iloc[:, -1].to_numpy()
    print("Labels extracted")
except Exception as e:
    abort_on_error(f"Failed to separate features and labels: {e}")

# Normalize dataset
try:
    print("Normalizing dataset")
    norm_start_time = time.time()
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    print("Norms calculated")
    norms[norms == 0] = 1e-10  # Avoid division by zero
    print("Adjusted zero norms")
    normalized_features = features / norms
    print("Features normalized")
    norm_end_time = time.time()
    print(f"Normalization completed in {norm_end_time - norm_start_time:.2f} seconds.")
except Exception as e:
    abort_on_error(f"Error normalizing dataset: {e}")

# Split data into training and testing sets
try:
    print("Splitting data into training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, labels, test_size=0.2, random_state=42
    )
    print("Data split into training and testing sets")
except Exception as e:
    abort_on_error(f"Failed to split data: {e}")

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

# Create subcircuits
try:
    features_per_circuit = 10  # Reduced number of features per circuit
    print(f"Setting features_per_circuit to {features_per_circuit}")
    print("Creating training subcircuits")
    subcircuits_train = create_subcircuits(X_train, features_per_circuit)
    print("Training subcircuits created")
    print("Creating testing subcircuits")
    subcircuits_test = create_subcircuits(X_test, features_per_circuit)
    print("Testing subcircuits created")
except Exception as e:
    abort_on_error(f"Error splitting data into subcircuits: {e}")

# Compute quantum kernels
def compute_subcircuit_kernel(subcircuits_train, subcircuits_test=None):
    """
    Compute quantum kernels for subcircuits and aggregate results.
    """
    print("Computing quantum kernels")
    total_kernel_train = None
    total_kernel_test = None
    backend = AerSimulator(method='statevector')
    # sampler = Sampler()
    try:
        for idx, (subset_train, feature_map) in enumerate(subcircuits_train):
            print(f"Computing kernel for subcircuit {idx}")
            quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
            kernel_train = quantum_kernel.evaluate(x_vec=subset_train, y_vec=subset_train)
            print(f"Training kernel for subcircuit {idx} computed")
            total_kernel_train = kernel_train if total_kernel_train is None else total_kernel_train + kernel_train

            if subcircuits_test is not None:
                subset_test = subcircuits_test[idx][0]
                kernel_test = quantum_kernel.evaluate(x_vec=subset_test, y_vec=subset_train)
                print(f"Testing kernel for subcircuit {idx} computed")
                total_kernel_test = kernel_test if total_kernel_test is None else total_kernel_test + kernel_test
    except Exception as e:
        abort_on_error(f"Failed to compute quantum kernel: {e}")
    return total_kernel_train, total_kernel_test

try:
    print("Computing subcircuit kernels")
    total_kernel_train, total_kernel_test = compute_subcircuit_kernel(subcircuits_train, subcircuits_test)
    print("Kernel computation completed")
except Exception as e:
    abort_on_error(f"Error during kernel computation: {e}")

# Train a classifier using the computed kernel
try:
    print("Training the SVM classifier with precomputed kernel")
    svm_classifier = SVC(kernel='precomputed')
    svm_classifier.fit(total_kernel_train, y_train)
    print("SVM classifier trained successfully")
except Exception as e:
    abort_on_error(f"Failed to train SVM classifier: {e}")

# Evaluate the classifier
try:
    print("Evaluating the classifier")
    y_pred = svm_classifier.predict(total_kernel_test)
    print("Predictions made")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
except Exception as e:
    abort_on_error(f"Failed to evaluate the classifier: {e}")
