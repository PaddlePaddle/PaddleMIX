from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._image_hash_filter import image_hash_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the image hash filter operator
dataset = dataset.image_hash_filter(
    hash_method="phash"  # Use the "phash" method (default)
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Image hash filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_image_hash_filtered.json'))