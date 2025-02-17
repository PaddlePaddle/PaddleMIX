from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._llm_judge_filter import llm_judge_filter

# Path to the dataset
anno_path = 'datasets/llava/02_val_chatml_filter.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the llm response judgment operator
dataset = dataset.llm_judge_filter(
    model_name="Qwen/Qwen2.5-7B",  # llm model name
    batch_size=1                     # Batch size for processing
)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("llm response judgment complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_llm_judge_filtered.json'))