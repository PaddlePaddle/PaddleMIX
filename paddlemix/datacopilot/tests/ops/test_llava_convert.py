from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.convert._llava_convert import llava_convert


# Dataset path
anno_path = 'datasets/llava/00_llava_v1_5_mix665k.json'

# Load dataset
print("Loading dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Conversion operator
dataset = dataset.llava_convert()


print("Converted dataset size:", len(dataset))
print("Dataset convert complete.")
dataset.export_json(anno_path.replace('.json', '_convert.json'))