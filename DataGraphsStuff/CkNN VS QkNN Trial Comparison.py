import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets (replace with your file paths)
cknn_file_path = 'custom_CkNN_results.csv'
qknn_file_path = 'consolidated_Qknn_accuracy_corrected.csv'

cknn_data = pd.read_csv(cknn_file_path)
qknn_data = pd.read_csv(qknn_file_path)

# Multiply CkNN values by 100 for percentage accuracy
cknn_data.iloc[:, 1:] = cknn_data.iloc[:, 1:] * 100

# Create a larger figure
plt.figure(figsize=(20, 12))

# Plot each trial for CkNN
for trial in range(1, 8):  # 7 trials
    plt.plot(cknn_data['Unnamed: 0'], cknn_data[f'Trial {trial}'],
             label=f'CkNN Trial {trial}', linestyle='--', marker='o')

# Plot each trial for QkNN
for trial in range(1, 8):  # 7 trials
    plt.plot(qknn_data['k'], qknn_data[f'Trial {trial}'],
             label=f'QkNN Trial {trial}', linestyle='-', marker='x')

# Set axis labels and title
plt.title('Comparison of CkNN and QkNN Performance Across Trials', fontsize=20)
plt.xlabel('k-value / Index', fontsize=16)
plt.ylabel('Performance (%)', fontsize=16)

# Ensure x-axis includes all values from 1 to 50
plt.xticks(range(1, 51), fontsize=12)
plt.yticks(fontsize=12)

# Add legend and grid
plt.legend(fontsize=12, loc='upper right')
plt.grid(True)

# Save the plot as a PNG in the same folder
output_path = 'comparison_cknn_qknn_performance.png'
plt.savefig(output_path, format='png', dpi=300)

# Display the plot
plt.show()

print(f"Graph saved as: {output_path}")
