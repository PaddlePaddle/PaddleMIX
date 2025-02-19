from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._stopwords_ratio_filter import stopwords_ratio_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the stopwords ratio filter operator
dataset = dataset.stopwords_ratio_filter(
    min_ratio=0.25  # Minimum stopword ratio
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Stopwords ratio filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_stopwords_filtered.json'))