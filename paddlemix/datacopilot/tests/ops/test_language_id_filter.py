from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._language_id_filter import language_id_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the language ID filter operator
dataset = dataset.language_id_filter(
    lang=["en", "fr"],  # Allow only English and French
    min_score=0.9       # Minimum confidence score
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Language ID filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_lang_filtered.json'))