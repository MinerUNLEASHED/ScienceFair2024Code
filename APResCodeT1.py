import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Load the IRIS dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

# Define quantum feature map
feature_map = ZZFeatureMap(feature_dimension=4, reps=2)

# Initialize the StatevectorSampler
sampler = StatevectorSampler()

# Function to evaluate model with given QuantumKernel
def evaluate_model(quantum_kernel, label):
    from qiskit_machine_learning.algorithms import QSVC
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(X_train, y_train)
    y_pred = qsvc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{label} Accuracy: {accuracy:.2f}")

# Original quantum kernel (no noise)
quantum_kernel_original = FidelityQuantumKernel(feature_map=feature_map, fidelity=sampler)
evaluate_model(quantum_kernel_original, "Original")

# Note: StatevectorSampler does not support noise models directly.
# To simulate noise, consider using Qiskit Aer simulators with noise models.
