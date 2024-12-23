import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the normalized dataset
normalized_dataset_path = 'Standardized_ARCENE_Dataset.csv'  # Fully normalized dataset
normalized_dataset = pd.read_csv(normalized_dataset_path)

# Extract features and labels from the normalized dataset
normalized_features = normalized_dataset.iloc[:, :-1].values  # All columns except the last one (features)
normalized_labels = normalized_dataset['Label'].values  # The last column (labels)

# Compute statistics for the normalized dataset
num_features = normalized_features.shape[1]

# Create output folder for normalized dataset
output_folder_normalized = "Feature_Normalization_Graphs/Normalized"
os.makedirs(output_folder_normalized, exist_ok=True)

# Export scatter plot for the first feature (normalized dataset)
plt.figure(figsize=(16, 12))
plt.scatter(range(normalized_features.shape[0]), normalized_features[:, 0], alpha=0.6, label="Feature 1")
plt.title("Scatter Plot of Feature 1 (Normalized Dataset)", fontsize=20)
plt.xlabel("Sample Index", fontsize=18)
plt.ylabel("Feature Value", fontsize=18)
plt.legend(fontsize=16)
scatter_plot_path = os.path.join(output_folder_normalized, "scatter_plot_feature1.png")
plt.savefig(scatter_plot_path, dpi=800, bbox_inches='tight')
plt.close()

# Export heatmap for normalized dataset
plt.figure(figsize=(24, 20))
sns.heatmap(normalized_features, cmap="coolwarm", cbar=True, xticklabels=10, yticklabels=50)
plt.title("Heatmap of Feature Scaling (Normalized Dataset)", fontsize=22)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Samples", fontsize=20)
heatmap_path = os.path.join(output_folder_normalized, "heatmap.png")
plt.savefig(heatmap_path, dpi=800, bbox_inches='tight')
plt.close()

# Export histograms for individual features (normalized dataset)
for i in range(num_features):
    plt.figure(figsize=(16, 12))
    plt.hist(normalized_features[:, i], bins=50, alpha=0.75, label=f'Feature {i+1}')
    plt.title(f"Histogram of Feature {i+1} (Normalized Dataset)", fontsize=20)
    plt.xlabel("Value", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.legend(fontsize=16)
    feature_path = os.path.join(output_folder_normalized, f"feature_{i+1}.png")
    plt.savefig(feature_path, dpi=800, bbox_inches='tight')
    plt.close()

print(f"High-quality graphs for normalized dataset exported successfully to folder: {output_folder_normalized}")
