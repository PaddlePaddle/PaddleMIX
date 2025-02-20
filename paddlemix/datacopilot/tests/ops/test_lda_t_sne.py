from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.visualize._lda_t_sne import lda_topic_clustering

# Path to the dataset
json_path = "datasets/llava/01_val_chatml.json"

# Load the dataset
print("Loading dataset...")
dataset = MMDataset.from_json(json_path)
print(f"Dataset size: {len(dataset)} samples")

# Perform LDA topic clustering
results = lda_topic_clustering(
    dataset=dataset,
    num_topics=5,                 # Number of topics to identify
    tsne_perplexity=30,           # Perplexity parameter for T-SNE
    tsne_learning_rate=200,       # Learning rate for T-SNE
    tsne_n_iter=1000,             # Number of T-SNE optimization iterations
    random_state=42,              # Random seed for reproducibility
    output_plot="lda_tsne_plot.png"  # Path to save the visualization
)

# Print results
print("Topic Clustering Results:")
print(f"Topic Distribution per Document (LDA): {results['lda_result']}")
print(f"T-SNE 2D Projection: {results['tsne_result']}")
print(f"Most Likely Topics: {results['topics']}")

# The T-SNE visualization will be saved as 'lda_tsne_plot.png'.
print("T-SNE plot saved as 'lda_tsne_plot.png'")