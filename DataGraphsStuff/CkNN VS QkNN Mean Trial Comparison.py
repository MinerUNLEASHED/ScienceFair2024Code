import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets (replace with your file paths)
cknn_file_path = 'custom_CkNN_results.csv'
qknn_file_path = 'consolidated_Qknn_accuracy_corrected.csv'

cknn_data = pd.read_csv(cknn_file_path)
qknn_data = pd.read_csv(qknn_file_path)

# Multiply CkNN values by 100 for percentage accuracy
cknn_data.iloc[:, 1:] = cknn_data.iloc[:, 1:] * 100

# Compute the average performance for each k or index
cknn_avg = cknn_data.iloc[:, 1:].mean(axis=1)  # Average across trials for CkNN
qknn_avg = qknn_data.iloc[:, 1:].mean(axis=1)  # Average across trials for QkNN

# Create the plot
plt.figure(figsize=(16, 10))

# Plot the average performances
plt.plot(cknn_data['Unnamed: 0'], cknn_avg, label='CkNN Average Performance', marker='o', linestyle='--')
plt.plot(qknn_data['k'], qknn_avg, label='QkNN Average Performance', marker='x', linestyle='-')

# Add labels, title, and legend
plt.title('Average Performance of CkNN and QkNN Across Trials', fontsize=20)
plt.xlabel('k-value / Index', fontsize=16)
plt.ylabel('Average Performance (%)', fontsize=16)

# Add legend and grid
plt.legend(fontsize=14)
plt.grid(True)

# Save the plot as a PNG in the same folder
output_path = 'average_performance_cknn_qknn.png'
plt.savefig(output_path, format='png', dpi=300)

# Display the plot
plt.show()

print(f"Graph saved as: {output_path}")
