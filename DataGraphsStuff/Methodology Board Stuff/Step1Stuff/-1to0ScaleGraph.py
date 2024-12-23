import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Cleaned_ARCENE_Dataset.csv'
dataset = pd.read_csv(file_path)

# Analyze the distribution of labels before and after transformation (-1 to 0)
label_counts_before = dataset['Label'].value_counts()
dataset['Label'] = dataset['Label'].replace(-1, 0)
label_counts_after = dataset['Label'].value_counts()

# Plot the schematic for transformation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Before transformation
axes[0].bar(label_counts_before.index, label_counts_before.values, color='skyblue')
axes[0].set_title("Label Distribution: Before Transformation")
axes[0].set_xticks([-1, 1])
axes[0].set_xticklabels(["-1", "1"])
axes[0].set_ylabel("Count")
axes[0].set_xlabel("Label")

# After transformation
axes[1].bar(label_counts_after.index, label_counts_after.values, color='lightgreen')
axes[1].set_title("Label Distribution: After Transformation")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["0", "1"])
axes[1].set_ylabel("Count")
axes[1].set_xlabel("Label")

# Arrows for transformation schematic
fig.suptitle("Transformation of Labels (-1 to 0)", fontsize=16, y=1.05)
plt.tight_layout()

# Save the plot as an image for potential inclusion in the board
output_path = "-1to0ScaleGraphOutput.png"
plt.savefig(output_path)
plt.show()

