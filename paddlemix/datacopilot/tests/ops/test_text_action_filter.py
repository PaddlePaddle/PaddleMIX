from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._text_action_filter import text_action_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the text action filter operator
dataset = dataset.text_action_filter(
    lang="en",  # Language of the text
    min_action_num=2  # Minimum number of verbs
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Text action filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_action_filtered.json'))