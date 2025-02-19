from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._char_ngram_repetition_filter import char_ngram_repetition_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the character n-gram repetition filter
rep_len = 10       # Set the n-gram length
min_ratio = 0.1    # Set the minimum repetition ratio
max_ratio = 0.4    # Set the maximum repetition ratio
dataset = dataset.char_ngram_repetition_filter(rep_len=rep_len, min_ratio=min_ratio, max_ratio=max_ratio)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Character n-gram repetition filtering completed.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_char_ngram_filtered.json'))