from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.analysis._base_analysis import base_analysis_pipeline

# Path to the dataset
anno_path = 'datasets/llava/02_val_chatml_filter.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Analysis flags to specify which analyses to run
analysis_flags = {
    "dataset_statistics": False,
    "language_distribution": False,
    "image_path_analysis": True,
    "data_anomalies": False,
    "conversation_tokens": False
}

# Run the base analysis
results = dataset.base_analysis_pipeline(analysis_flags=analysis_flags, output_dir="analysis_results")

# Print the results
print("Analysis results:", results)