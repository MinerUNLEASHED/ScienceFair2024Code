import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Initialize Dask LocalCluster without the diagnostic dashboard
cluster = LocalCluster(n_workers=20, threads_per_worker=1, memory_limit="50GB", dashboard_address=':0')
client = Client(cluster)

# Load the ARCENE dataset
base_dir = os.getenv("TMPDIR", "/nobackup/")  # Use TMPDIR for distributed file locations
file_path = os.path.join(base_dir, 'Standardized_ARCENE_Dataset.csv')  # Full path to dataset
print(f"Loading dataset from {file_path}")
dataset = pd.read_csv(file_path)

# Separate features and labels
features = dataset.iloc[:, :-1]  # All columns except the last one
labels = dataset.iloc[:, -1]  # The last column is the label

# Define the number of repetitions
n = 5
print(f"Quantum k-NN experiment will run {n} times")

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

# Function to perform a single experiment run and record results
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

# Perform a consistent train-test split and compute kernel matrices
print("Normalizing dataset")
norm_start_time = time.time()
normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
norm_end_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(
    normalized_features, labels, test_size=0.2, random_state=22
)

print("Creating feature map")
feature_map_start_time = time.time()
num_features = X_train.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement="linear")
feature_map_end_time = time.time()

print("Computing train kernel matrix")
train_kernel_start_time = time.time()
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
train_kernel_matrix = quantum_kernel.evaluate(x_vec=X_train)
train_kernel_end_time = time.time()

print("Computing test kernel matrix")
test_kernel_start_time = time.time()
test_kernel_matrix = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
test_kernel_end_time = time.time()

timings = {
    'Normalization Time': norm_end_time - norm_start_time,
    'Feature Map Time': feature_map_end_time - feature_map_start_time,
    'Train Kernel Time': train_kernel_end_time - train_kernel_start_time,
    'Test Kernel Time': test_kernel_end_time - test_kernel_start_time
}

# Execute experiment runs
for i in range(1, n + 1):
    run_results = run_qknn_experiment(i, X_train, X_test, y_train, y_test, train_kernel_matrix, test_kernel_matrix, timings)
    results_df = pd.DataFrame(run_results)
    output_file = os.path.join(base_dir, f"ARCENE_QkNN_Run_{i}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} completed. Results saved to {output_file}")

# Close Dask client
client.close()
