import matplotlib.pyplot as plt

# Define the proportions for training and testing split
data_split = [80, 20]
labels = ["Training Set (80%)", "Testing Set (20%)"]

# Create the pie chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(
    data_split,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=['lightgreen', 'lightcoral'],
    explode=[0.1, 0],  # Slightly explode the training slice
    shadow=False,  # Remove the shadow
    textprops={'fontsize': 16}  # Increase font size for all text
)

# Add title with larger font size
ax.set_title("Dataset Split: Training vs Testing", fontsize=20)

# Save the figure with a transparent background
output_path = "dataset_split_pie_chart_transparent.png"
plt.savefig(output_path, dpi=1000, format='png', transparent=True)  # Transparent background
# plt.show()
