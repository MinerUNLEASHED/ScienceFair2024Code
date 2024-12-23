import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the attached dataset
original_dataset_path = 'Cleaned_ARCENE_Dataset.csv'  # Replace with your file path
original_dataset = pd.read_csv(original_dataset_path)

# Extract features and labels from the dataset
original_features = original_dataset.iloc[:, :-1].values  # All columns except the last one
original_labels = original_dataset['Label'].values  # The last column

# Compute statistics for z-score normalization
original_means = original_features.mean(axis=0)
original_stds = original_features.std(axis=0)

# Create output folder
output_folder = "Feature Normalization Graphs"
os.makedirs(output_folder, exist_ok=True)

# Export all features in one graph
plt.figure(figsize=(25, 20))
num_features = original_features.shape[1]

for i in range(num_features):
    plt.subplot((num_features + 4) // 5, 5, i + 1)  # Grid layout
    plt.hist(original_features[:, i], bins=30, alpha=0.6, label=f'Feature {i+1} Raw Data')
    plt.axvline(original_means[i], color='r', linestyle='dashed', linewidth=1, label=f'Mean')
    plt.axvline(original_means[i] + original_stds[i], color='g', linestyle='dashed', linewidth=1, label=f'STD +1')
    plt.axvline(original_means[i] - original_stds[i], color='g', linestyle='dashed', linewidth=1)
    plt.title(f'Feature {i+1}', fontsize=8)
    plt.xlabel('Value', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.legend(fontsize=6)

plt.suptitle('Z-Score Normalization Visualization for All Features', fontsize=16)
all_features_path = os.path.join(output_folder, "all_features.png")
plt.savefig(all_features_path, dpi=300, bbox_inches='tight')
plt.close()

# Export each feature as a separate graph
for i in range(num_features):
    plt.figure(figsize=(10, 6))
    plt.hist(original_features[:, i], bins=30, alpha=0.6, label=f'Feature {i+1} Raw Data')
    plt.axvline(original_means[i], color='r', linestyle='dashed', linewidth=1, label=f'Mean')
    plt.axvline(original_means[i] + original_stds[i], color='g', linestyle='dashed', linewidth=1, label=f'STD +1')
    plt.axvline(original_means[i] - original_stds[i], color='g', linestyle='dashed', linewidth=1)
    plt.title(f'Z-Score Normalization Visualization for Feature {i+1}', fontsize=12)
    plt.xlabel('Value', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend(fontsize=8)
    feature_path = os.path.join(output_folder, f"feature_{i+1}.png")
    plt.savefig(feature_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"Graphs exported successfully to folder: {output_folder}")
