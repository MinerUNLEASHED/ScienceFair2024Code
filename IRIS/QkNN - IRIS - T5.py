import os
import smtplib
import ssl
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from scipy.stats import mode
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
features = pd.DataFrame(iris.data, columns=iris.feature_names)
labels = pd.Series(iris.target, name="label")

# Define the number of repetitions
n = 5


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


# Function to perform a single experiment run and record results for all k values
def run_qknn_experiment(run_index, X_train, X_test, y_train, y_test, train_kernel_matrix, test_kernel_matrix):
    results = []

    # Use all k values from 1 to the test size
    test_size = X_test.shape[0]
    k_values = range(1, test_size + 1)

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
            'QkNN Time': qknn_end_time - qknn_start_time,
            'Accuracy': accuracy,
            'Date': datetime.now().strftime("%m/%d/%Y"),
            'Time': datetime.now().strftime("%H:%M:%S")
        })

    return results


# Email sending function
def send_mail(run_index):
    sender_mail = "azhmeerjesani@gmail.com"
    receiver_mail = "8323177274@vzwpix.com"
    subject = f"IRIS Quantum k-NN Experiment Run {run_index} Completed"
    body = f"The results for IRIS Quantum k-NN Run {run_index} have been completed. Check the output directory for the CSV file."
    password = "fhxk mrbe hsdp yhgx"

    # Create email
    message = MIMEMultipart()
    message["From"] = sender_mail
    message["To"] = receiver_mail
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_mail, password)
        server.sendmail(sender_mail, receiver_mail, message.as_string())


# Perform a consistent train-test split and compute kernel matrices once
normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
X_train, X_test, y_train, y_test = train_test_split(
    normalized_features, labels, test_size=0.2, random_state=42
)
num_features = X_train.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement="linear")
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

train_kernel_matrix = quantum_kernel.evaluate(x_vec=X_train)
test_kernel_matrix = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

# Repeat the experiment `n` times and save results to CSV files
for i in range(1, n + 1):
    run_results = run_qknn_experiment(i, X_train, X_test, y_train, y_test, train_kernel_matrix, test_kernel_matrix)
    results_df = pd.DataFrame(run_results)
    output_file = f"Iris_QkNN_Run_{i}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} completed. Results saved to {output_file}.")

    # Send email notification with results
    send_mail(i)
    print(f"Email notification sent for Run {i}.")
