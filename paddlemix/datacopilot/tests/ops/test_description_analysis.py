from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.analysis._description_analysis import description_analysis

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the description analysis operator
dataset = dataset.description_analysis(model_name= "Qwen/Qwen2.5-7B", batch_size=1)

# Print the size of the processed dataset
print("Processed dataset size:", len(dataset))
print("Description analysis completed.")

# Export the processed dataset
dataset.export_json(anno_path.replace('.json', '_description_analysis.json'))