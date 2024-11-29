import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed  # Importing Joblib for parallel execution

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the ARCENE dataset
base_dir = os.path.dirname(os.path.abspath(__file__))  # Use script directory for file locations
file_path = os.path.join(base_dir, 'Standardized_ARCENE_Dataset.csv')  # Full path to dataset
print(f"Loading dataset from {file_path}")
dataset = pd.read_csv(file_path)

# Separate features and labels
features = dataset.iloc[:, :-1]  # All columns except the last one
labels = dataset.iloc[:, -1]  # The last column is the label

# Define the number of repetitions
n = 5
print(f"Quantum k-NN experiment will run {n} times")

# Distributed Quantum Feature Map Creation
def create_feature_map(num_features, reps=2, entanglement="linear"):
    return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement=entanglement)

# Distributed Quantum Kernel Computation
def compute_train_kernel(feature_map, X_train):
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    return quantum_kernel.evaluate(x_vec=X_train)

def compute_test_kernel(feature_map, X_test, X_train):
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    return quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

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

# Experiment function
def run_qknn_experiment(run_index, X_train, X_test, y_train, y_test, train_kernel_matrix, test_kernel_matrix, timings):
    results = []

    # Use all k values from 1 to the test size
    test_size = X_test.shape[0]
    k_values = range(1, test_size + 1)

    # Add timing metrics to the first row of results
    results.append({
        'Run Index': run_index,
        'k': 'N/A',
        'Normalization Time': timings['Normalization Time'],
        'Feature Map Time': timings['Feature Map Time'],
        'Train Kernel Time': timings['Train Kernel Time'],
        'Test Kernel Time': timings['Test Kernel Time'],
        'QkNN Time': 'N/A',
        'Accuracy': 'N/A',
        'Date': datetime.now().strftime("%m/%d/%Y"),
        'Time': datetime.now().strftime("%H:%M:%S")
    })

    # Evaluate for each k value
    for k in k_values:
        qknn_start_time = time.time()
        y_pred = quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=k)
        qknn_end_time = time.time()

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Record timing and accuracy
        results.append({
            'Run Index': run_index,
            'k': k,
            'Normalization Time': 'N/A',
            'Feature Map Time': 'N/A',
            'Train Kernel Time': 'N/A',
            'Test Kernel Time': 'N/A',
            'QkNN Time': qknn_end_time - qknn_start_time,
            'Accuracy': accuracy,
            'Date': datetime.now().strftime("%m/%d/%Y"),
        })

    return results

# Perform dataset normalization
print("Normalizing dataset")
norm_start_time = time.time()
normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
norm_end_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(
    normalized_features, labels, test_size=0.2, random_state=22
)

# Distributed feature map creation
print("Creating feature map")
num_features = X_train.shape[1]
feature_map = create_feature_map(num_features)

# Distributed kernel computation using Joblib
print("Computing train kernel matrix")
train_kernel_matrix = Parallel(n_jobs=-1)(
    delayed(compute_train_kernel)(feature_map, X_train) for _ in range(1)
)[0]

print("Computing test kernel matrix")
test_kernel_matrix = Parallel(n_jobs=-1)(
    delayed(compute_test_kernel)(feature_map, X_test, X_train) for _ in range(1)
)[0]

timings = {
    'Normalization Time': norm_end_time - norm_start_time,
    'Feature Map Time': "Distributed",
    'Train Kernel Time': "Distributed",
    'Test Kernel Time': "Distributed"
}

# Execute experiment runs using Joblib
print("Running experiments")
experiment_results = Parallel(n_jobs=-1)(
    delayed(run_qknn_experiment)(
        i, X_train, X_test, y_train, y_test, train_kernel_matrix, test_kernel_matrix, timings
    )
    for i in range(1, n + 1)
)

# Save results to files
for i, run_results in enumerate(experiment_results, 1):
    results_df = pd.DataFrame(run_results)
    output_file = os.path.join(base_dir, f"ARCENE_QkNN_Run_{i}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} completed. Results saved to {output_file}")
