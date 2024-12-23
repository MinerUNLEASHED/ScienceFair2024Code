import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
raw_dataset_path = 'Cleaned_ARCENE_Dataset.csv'
standardized_dataset_path = 'Standardized_ARCENE_Dataset.csv'

raw_dataset = pd.read_csv(raw_dataset_path)
standardized_dataset = pd.read_csv(standardized_dataset_path)

# Extract features (exclude label columns if present)
raw_features = raw_dataset.iloc[:, :-1].values
standardized_features = standardized_dataset.iloc[:, :-1].values

# Calculate min and max for each dataset
raw_min, raw_max = raw_features.min(axis=0), raw_features.max(axis=0)
standardized_min, standardized_max = standardized_features.min(axis=0), standardized_features.max(axis=0)

# Create the figure
fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=1000)  # High DPI for highest quality

# Plot raw dataset feature range
axes[0].plot(raw_min, label='Min Value', color='blue', linewidth=1.5)
axes[0].plot(raw_max, label='Max Value', color='orange', linewidth=1.5)
axes[0].set_title('Raw Feature Values', fontsize=16)
axes[0].set_xlabel('Features', fontsize=14)
axes[0].set_ylabel('Value', fontsize=14)
axes[0].legend(fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)

# Plot standardized dataset feature range
axes[1].plot(standardized_min, label='Min Value', color='blue', linewidth=1.5)
axes[1].plot(standardized_max, label='Max Value', color='orange', linewidth=1.5)
axes[1].set_title('Standardized Feature Values', fontsize=16)
axes[1].set_xlabel('Features', fontsize=14)
axes[1].set_ylabel('Value', fontsize=14)
axes[1].legend(fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)

# Add a super title for the figure
fig.suptitle("Feature Value Ranges: Before and After Standardization", fontsize=18)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure in the highest resolution as PNG
output_path = "StandarizedFeatureComparisonGraph.png"
plt.savefig(output_path, dpi=1000, format='png')  # Save as PNG with the highest DPI

# plt.show()
