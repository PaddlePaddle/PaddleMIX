from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._conversation_length_filter import conversation_length_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the conversation length filter operator
max_length = 1024  # Set the maximum allowed conversation length
dataset = dataset.conversation_length_filter(max_length=max_length)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Conversation length filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_length_filtered.json'))