from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._image_resolution_filter import image_resolution_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the image resolution filter operator
dataset = dataset.image_resolution_filter(
    min_width=112,   # Minimum width
    min_height=112,  # Minimum height
    max_width=1920,  # Maximum width (optional)
    max_height=1080  # Maximum height (optional)
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Image resolution filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_resolution_filtered.json'))