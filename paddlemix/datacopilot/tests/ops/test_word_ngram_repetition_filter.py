from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._word_ngram_repetition_filter import word_ngram_repetition_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the word n-gram repetition filter
dataset = dataset.word_ngram_repetition_filter(
    rep_len=10,         # Length of the n-gram
    min_ratio=0.0,      # Minimum repetition ratio
    max_ratio=0.2       # Maximum repetition ratio
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Word n-gram repetition filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_ngram_filtered.json'))