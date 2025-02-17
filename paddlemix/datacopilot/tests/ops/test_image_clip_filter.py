from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._image_clip_filter import image_clip_filter
from paddlemix.datacopilot.ops.filter._image_clip_filter import CLIPFilterConfig

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the image_clip_filter operator
config = CLIPFilterConfig(
    threshold=0.1,  # Confidence threshold
    batch_size=8,  # Batch size for processing
    save_images=True,  # Save low-confidence images
    save_dir="./filtered_images"  # Directory to save images
)

dataset = dataset.image_clip_filter(config=config)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Low-confidence filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_clip_filtered.json'))