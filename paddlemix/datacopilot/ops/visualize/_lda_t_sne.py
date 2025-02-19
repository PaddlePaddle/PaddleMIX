import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from ...core import T, MMDataset, register


@register()
def extract_text_for_lda(item: T) -> Optional[str]:
    """
    Extract text from conversations for topic modeling.
    
    Args:
        item (T): A single dataset item containing conversation data.
    
    Returns:
        Optional[str]: A string combining all questions and answers from the conversation, or None if no text is available.
    """
    conversations = item.get("conversations", [])
    text = []
    for convo in conversations:
        # Assuming each conversation is a list containing questions and answers, extract the text
        question, answer = convo
        text.append(question)
        text.append(answer)
    return " ".join(text)


@register()
def lda_topic_clustering(
    dataset: MMDataset,
    num_topics: int = 5,
    tsne_perplexity: int = 30,
    tsne_learning_rate: int = 200,
    tsne_n_iter: int = 1000,
    random_state: int = 42,
    output_plot: str = "lda_tsne_plot.png"
):
    """
    Perform LDA topic clustering on conversation text and visualize the results with T-SNE.
    
    Args:
        dataset (MMDataset): The dataset containing conversation data.
        num_topics (int): The number of topics to identify using LDA.
        tsne_perplexity (int): Perplexity parameter for T-SNE.
        tsne_learning_rate (int): Learning rate for T-SNE.
        tsne_n_iter (int): Number of iterations for T-SNE optimization.
        random_state (int): Random seed for reproducibility.
        output_plot (str): Path to save the T-SNE visualization plot.
    
    Returns:
        Dict: A dictionary containing the following keys:
            - "lda_result": The topic distribution for each document.
            - "tsne_result": The 2D T-SNE projection of the LDA results.
            - "topics": The most likely topic for each document.
    """
    # Extract text data
    texts = dataset.map(extract_text_for_lda)
    texts = [text for text in texts if text.strip()]  # Remove empty texts

    # Text vectorization
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    text_matrix = vectorizer.fit_transform(texts)

    # LDA topic modeling
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=random_state)
    lda_result = lda.fit_transform(text_matrix)

    # Dimensionality reduction using T-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate=tsne_learning_rate,
        n_iter=tsne_n_iter,
        random_state=random_state
    )
    tsne_result = tsne.fit_transform(lda_result)

    # Visualize the results
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        tsne_result[:, 0], tsne_result[:, 1], c=np.argmax(lda_result, axis=1), cmap='tab10', alpha=0.7
    )
    plt.colorbar(scatter, label="Topic Cluster")
    plt.title("LDA Topic Clustering with T-SNE Visualization")
    plt.xlabel("T-SNE Dimension 1")
    plt.ylabel("T-SNE Dimension 2")
    plt.savefig(output_plot)

    # Return results
    return {
        "lda_result": lda_result,
        "tsne_result": tsne_result,
        "topics": np.argmax(lda_result, axis=1).tolist()
    }