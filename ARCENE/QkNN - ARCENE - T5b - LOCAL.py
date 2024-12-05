import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Load dataset
try:
    print("Loading dataset...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'Standardized_ARCENE_Dataset.csv')
    dataset = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(-2)

# Separate features and labels
print("Separating features and labels...")
features = dataset.iloc[:, :-1].to_numpy()
labels = dataset.iloc[:, -1].to_numpy()
print("Features and labels separated.")

# Normalize dataset
print("Normalizing dataset...")
norms = np.linalg.norm(features, axis=1, keepdims=True)
norms[norms == 0] = 1e-10  # Avoid division by zero
normalized_features = features / norms
print("Dataset normalized.")

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    normalized_features, labels, test_size=0.2, random_state=42
)
print(f"Data split complete. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}.")

# Function to split data into feature batches
def create_feature_batches(data, batch_size):
    print(f"Creating feature batches with batch size {batch_size}...")
    num_features = data.shape[1]
    num_batches = num_features // batch_size + (num_features % batch_size > 0)
    print(f"Number of batches: {num_batches}")
    batches = [
        data[:, i * batch_size : (i + 1) * batch_size]
        for i in range(num_batches)
    ]
    print("Feature batches created.")
    return batches

# Function to compute quantum kernels for feature batches
def compute_batch_kernels(feature_batches_train, feature_batches_test=None):
    print("Computing quantum kernels for feature batches...")
    total_kernel_train = None
    total_kernel_test = None
    for idx, batch_train in enumerate(feature_batches_train):
        print(f"Processing batch {idx + 1}/{len(feature_batches_train)}...")
        feature_map = ZZFeatureMap(feature_dimension=batch_train.shape[1], reps=2, entanglement="linear")
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

        kernel_train = quantum_kernel.evaluate(x_vec=batch_train)
        print(f"Training kernel for batch {idx + 1} computed.")
        total_kernel_train = kernel_train if total_kernel_train is None else total_kernel_train + kernel_train

        if feature_batches_test is not None:
            batch_test = feature_batches_test[idx]
            kernel_test = quantum_kernel.evaluate(x_vec=batch_test, y_vec=batch_train)
            print(f"Testing kernel for batch {idx + 1} computed.")
            total_kernel_test = kernel_test if total_kernel_test is None else total_kernel_test + kernel_test

    print("Quantum kernels computation complete.")
    return total_kernel_train, total_kernel_test

# Create feature batches
batch_size = 30  # Number of features per batch
print("Creating feature batches for training and testing...")
train_batches = create_feature_batches(X_train, batch_size)
test_batches = create_feature_batches(X_test, batch_size)

# Compute kernels for batches
print("Starting kernel computation...")
train_kernel_start_time = time.time()
train_kernel_matrix, test_kernel_matrix = compute_batch_kernels(train_batches, test_batches)
train_kernel_end_time = time.time()
print("Kernel computation complete.")

# Timing Metrics
timings = {
    'Train Kernel Time': train_kernel_end_time - train_kernel_start_time,
}
print(f"Training kernel computation time: {timings['Train Kernel Time']} seconds.")

# Quantum k-NN function
def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    print(f"Running QkNN for k={k}...")
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        distances = 1 - test_kernel_matrix[i, :]
        k_nearest_indices = distances.argsort()[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        predicted_label = mode(k_nearest_labels, keepdims=True).mode[0]
        predictions.append(predicted_label)
    print(f"QkNN completed for k={k}.")
    return np.array(predictions)

# Function to perform a single experiment run and record results for all k values
def run_qknn_experiment(run_index, X_train, X_test, y_train, y_test, train_kernel_matrix, test_kernel_matrix, timings):
    print(f"Starting QkNN experiment run {run_index}...")
    results = []

    # Log the timing information
    results.append({
        'Run Index': run_index,
        'k': 'N/A',
        'Train Kernel Time': timings['Train Kernel Time'],
        'QkNN Time': 'N/A',
        'Accuracy': 'N/A',
        'Date': datetime.now().strftime("%m/%d/%Y"),
        'Time': datetime.now().strftime("%H:%M:%S")
    })

    # Evaluate the model for each k value
    for k in range(1, X_test.shape[0] + 1):
        print(f"Evaluating for k={k}...")
        qknn_start_time = time.time()
        y_pred = quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=k)
        qknn_end_time = time.time()

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for k={k}: {accuracy:.2f}")

        # Log results
        results.append({
            'Run Index': run_index,
            'k': k,
            'Train Kernel Time': 'N/A',
            'QkNN Time': qknn_end_time - qknn_start_time,
            'Accuracy': accuracy,
            'Date': datetime.now().strftime("%m/%d/%Y"),
            'Time': datetime.now().strftime("%H:%M:%S")
        })

    print(f"QkNN experiment run {run_index} completed.")
    return results

# Perform multiple runs and save results to CSV
n = 5  # Number of runs
for i in range(1, n + 1):
    print(f"Starting run {i}...")
    run_results = run_qknn_experiment(i, X_train, X_test, y_train, y_test, train_kernel_matrix, test_kernel_matrix, timings)
    results_df = pd.DataFrame(run_results)

    # Save to CSV
    output_file = f"QkNN_Run_{i}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} completed. Results saved to {output_file}.")
