from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._alphanumeric_ratio_filter import alphanumeric_ratio_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the alphanumeric ratio filter
min_ratio = 0.25  # Set the minimum alphanumeric ratio
max_ratio = 0.75  # Set the maximum alphanumeric ratio
dataset = dataset.alphanumeric_ratio_filter(min_ratio=min_ratio, max_ratio=max_ratio)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Alphanumeric ratio filtering completed.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_alnum_ratio_filtered.json'))