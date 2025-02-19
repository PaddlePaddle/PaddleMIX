from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._image_filesize_filter import image_filesize_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the image file size filter
dataset = dataset.image_filesize_filter(
    min_size_kb=10,  # Minimum file size in KB
    max_size_kb=1024  # Maximum file size in KB
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Image file size filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_filesize_filtered.json'))