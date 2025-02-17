from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._conversation_hash_filter import conversation_hash_filter

# Path to the dataset
anno_path = 'random_samples_1w.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the conversation_hash_filter operator
dataset = dataset.conversation_hash_filter(
    method="simhash",  # Use the 'simhash' method (default)
    threshold=0.8,  # Similarity threshold
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Text deduplication complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_conversation_hash_filter.json'))