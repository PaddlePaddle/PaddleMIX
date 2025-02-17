from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._special_characters_filter import special_characters_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the special characters filter
dataset = dataset.special_characters_filter(
    min_ratio=0.0,  # Minimum special character ratio
    max_ratio=0.25  # Maximum special character ratio
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Special character filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_special_char_filtered.json'))