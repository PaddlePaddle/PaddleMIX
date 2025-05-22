from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._text_entity_dependency_filter import text_entity_dependency_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the text entity dependency filter operator
dataset = dataset.text_entity_dependency_filter(
    lang="en",               # Language of the text
    min_dependency_num=2,    # Minimum number of dependency edges per entity
    any_or_all="any"         # Filtering strategy: 'any' or 'all'
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Text entity dependency filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_entity_dependency_filtered.json'))