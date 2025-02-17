from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._average_line_length_filter import average_line_length_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the average line length filter
min_length = 15  # Set the minimum average line length
max_length = 50  # Set the maximum average line length
dataset = dataset.average_line_length_filter(min_length=min_length, max_length=max_length)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Average line length filtering completed.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_avg_line_length_filtered.json'))