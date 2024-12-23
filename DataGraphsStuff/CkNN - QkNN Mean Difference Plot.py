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

# Compute the difference (CkNN - QkNN)
difference = cknn_avg - qknn_avg

# Create the plot
plt.figure(figsize=(14, 8))

# Plot the difference
plt.plot(qknn_data['k'], difference, label='CkNN - QkNN Performance', marker='o', linestyle='-', color='purple')

# Add a horizontal line at y=0 for reference
plt.axhline(0, color='black', linestyle='--', linewidth=1)

# Add labels, title, and legend
plt.title('Difference in Average Performance (CkNN - QkNN)', fontsize=20)
plt.xlabel('k-value / Index', fontsize=16)
plt.ylabel('Performance Difference (%)', fontsize=16)

# Add grid and legend
plt.grid(True)
plt.legend(fontsize=14)

# Save the plot as a PNG in the same folder
output_path = 'difference_plot_cknn_qknn.png'
plt.savefig(output_path, format='png', dpi=300)

# Display the plot
plt.show()

print(f"Graph saved as: {output_path}")
