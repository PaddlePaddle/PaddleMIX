from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._token_num_filter import token_num_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the token number filter operator
dataset = dataset.token_num_filter(
    tokenizer_model="Qwen/Qwen2.5-7B",  # Tokenizer model to use
    min_tokens=10,                      # Minimum token count
    max_tokens=512                      # Maximum token count
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Token number filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_token_filtered.json'))