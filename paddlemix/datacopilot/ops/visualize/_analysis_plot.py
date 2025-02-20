# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_data_statistics(data_statistics: Dict[str, Any], output_dir: str):
    """
    Plot Data Statistics, displaying the number of conversations divided into bins,
    and add statistical information on the right side.

    Args:
        data_statistics (dict): A dictionary containing dataset statistics.
        output_dir (str): Directory to save the generated plot.
    """
    # Extract statistics
    total_records = data_statistics.get('total_records', 0)
    unique_images = data_statistics.get('unique_images', 0)
    total_conversations = data_statistics.get('total_conversations', 0)
    max_conversations = data_statistics.get('max_conversations', 0)
    min_conversations = data_statistics.get('min_conversations', 0)
    avg_conversations = data_statistics.get('avg_conversations', 0.0)
    valid_items = data_statistics.get("valid_items", [])

    # Extract conversation counts
    conversation_counts = [
        len(item.get("conversations", [])) for item in valid_items
    ]

    # Handle the case when conversation_counts is empty
    if not conversation_counts:
        print("No valid conversations found. Skipping data statistics plot.")
        return

    # Define the range for bins
    min_count = min(conversation_counts)
    max_count = max(conversation_counts)

    # Ensure a valid range for bins
    if min_count == max_count:
        max_count += 1  # Extend the range slightly to create a valid bin

    # Create bins for the number of Q&A pairs
    num_bins = 10
    bins = np.linspace(min_count, max_count, num_bins + 1)

    # Calculate frequencies for each bin
    conversation_freq, _ = np.histogram(conversation_counts, bins=bins)

    # Create bin labels
    bin_labels = [f"{int(bins[i])} to {int(bins[i + 1])}" for i in range(len(bins) - 1)]

    # Create a wider figure split into two parts
    fig = plt.figure(figsize=(15, 6))

    # Left: Bar chart for conversation ranges
    ax1 = fig.add_subplot(121)
    ax1.bar(bin_labels, conversation_freq, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title("Data Statistics: Q&A Pairs by Range")
    ax1.set_xlabel("Number of Q&A Pairs (Range)")
    ax1.set_ylabel("Frequency")
    ax1.tick_params(axis='x', rotation=45)

    # Right: Statistical information
    ax2 = fig.add_subplot(122, facecolor='white')
    ax2.axis('off')

    # Define statistics text
    stats_text = [
        f"Total Records: {total_records}",
        f"Unique Images: {unique_images}",
        f"Total Conversations: {total_conversations}",
        f"Max Conversations: {max_conversations}",
        f"Min Conversations: {min_conversations}",
        f"Avg Conversations: {avg_conversations:.2f}"
    ]

    # Add text to the right-side canvas
    for i, text in enumerate(stats_text):
        ax2.text(0.1, 0.9 - i * 0.1, text, fontsize=12, ha='left', va='center')

    # Adjust layout and save the plot
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/00_data_statistics.png")
    plt.close()
    

    

def plot_field_distribution(field_distribution: Dict[str, Any], output_dir: str):
    """
    Plot Field Distribution chart, including language distribution and basic statistics.

    Args:
        field_distribution (dict): A dictionary containing field distribution data.
        output_dir (str): Directory to save the generated plot.
    """
    # Extract statistics with default values
    human_message_count = field_distribution.get('human_message_count', 0)
    assistant_message_count = field_distribution.get('assistant_message_count', 0)
    mismatched_language_pairs_count = field_distribution.get('mismatched_language_pairs_count', 0)

    # Language distribution, take the top 10
    languages_distribution = field_distribution.get('languages_distribution', {})
    sorted_languages = sorted(languages_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
    if sorted_languages:
        languages, language_counts = zip(*sorted_languages)
    else:
        languages, language_counts = [], []

    # Create a figure split into two parts
    fig = plt.figure(figsize=(15, 6))

    # Left: Language distribution bar chart
    ax1 = fig.add_subplot(121)
    if languages:
        ax1.bar(languages, language_counts, color='lightgreen')
        ax1.set_title("Language Distribution (Top 10)")
        ax1.set_xlabel("Language")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=45)
    else:
        ax1.text(0.5, 0.5, "No Language Data", fontsize=12, ha='center', va='center')
        ax1.set_title("Language Distribution")
        ax1.axis('off')

    # Right: Add statistical information
    ax2 = fig.add_subplot(122, facecolor='white')
    ax2.axis('off')

    # Define statistics text
    stats_text = [
        f"Human Message Count: {human_message_count}",
        f"Assistant Message Count: {assistant_message_count}",
        f"Mismatched Language Pairs: {mismatched_language_pairs_count}"
    ]

    # Add text to the right-side canvas
    for i, text in enumerate(stats_text):
        ax2.text(0.1, 0.9 - i * 0.1, text, fontsize=12, ha='left', va='center')

    # Adjust layout and save the plot
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/01_field_distribution.png")
    plt.close()


def plot_image_path_distribution(validation_result: Dict[str, Any], output_dir: str):
    """
    Plot Image Path Distribution chart, including missing paths and path statistics.

    Args:
        validation_result (dict): A dictionary containing image path validation results.
        output_dir (str): Directory to save the generated plot.
    """
    # Extract statistics with default values
    total_images = validation_result.get('total_images', 0)
    missing_images = validation_result.get('missing_images', 0)
    path_distribution = validation_result.get('path_distribution', {})

    if path_distribution:
        paths, path_counts = zip(*path_distribution.items())
    else:
        paths, path_counts = [], []

    # Create a figure split into two parts
    fig = plt.figure(figsize=(15, 8))

    # Left: Path distribution bar chart
    ax1 = fig.add_subplot(121)
    if paths:
        ax1.bar(paths, path_counts, color='lightblue')
        ax1.set_title("Image Path Distribution", fontsize=14)
        ax1.set_xlabel("Image Path", fontsize=12)
        ax1.set_ylabel("Image Count", fontsize=12)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.tick_params(axis='y', labelsize=10)
        plt.sca(ax1)  # Set current axis to ax1 for xticks adjustment
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels and align to the right
    else:
        ax1.text(0.5, 0.5, "No Path Data", fontsize=14, ha='center', va='center')
        ax1.set_title("Image Path Distribution", fontsize=14)
        ax1.axis('off')

    # Right: Add statistical information
    ax2 = fig.add_subplot(122, facecolor='white')
    ax2.axis('off')

    # Define statistics text
    stats_text = [
        f"Total Images: {total_images}",
        f"Missing Images: {missing_images}",
    ]

    # Add text to the right-side canvas
    for i, text in enumerate(stats_text):
        ax2.text(0.1, 0.9 - i * 0.1, text, fontsize=12, ha='left', va='center')

    # Adjust layout and save the plot
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "02_image_path_distribution.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Image path distribution plot saved at: {output_path}")




def plot_anomaly_statistics(anomaly_results: Dict[str, Any], output_dir: str):
    """
    Plot anomaly statistics, including missing fields, empty conversations, and invalid items.

    Args:
        anomaly_results (dict): A dictionary containing anomaly statistics.
        output_dir (str): Directory to save the generated plot.
    """
    try:
        # 从 anomaly_results 获取各类异常的计数
        missing_field_count = anomaly_results.get('missing_field_count', 0)
        empty_conversation_count = anomaly_results.get('empty_conversation_count', 0)
        invalid_item_count = anomaly_results.get('invalid_item_count', 0)

        # 定义标签和对应的计数
        labels = ['Missing Fields', 'Empty Conversations', 'Invalid Items']
        counts = [missing_field_count, empty_conversation_count, invalid_item_count]

        # 创建画布，分为左右两部分
        fig = plt.figure(figsize=(15, 8))

        # 左边：柱状图
        ax1 = fig.add_subplot(121)
        if any(counts):  # 如果有任何异常数据
            ax1.bar(labels, counts, color=['lightgreen', 'lightblue', 'lightcoral'])
            ax1.set_title("Anomaly Statistics", fontsize=16)
            ax1.set_xlabel("Anomaly Type", fontsize=12)
            ax1.set_ylabel("Count", fontsize=12)
            ax1.tick_params(axis='x', rotation=30, labelsize=10)
            ax1.tick_params(axis='y', labelsize=10)
        else:  # 如果没有异常数据
            ax1.text(0.5, 0.5, "No Anomalies Detected", fontsize=14, ha='center', va='center')
            ax1.set_title("Anomaly Statistics", fontsize=16)
            ax1.axis('off')

        # 右边：关键统计信息文本
        ax2 = fig.add_subplot(122, facecolor='white')
        ax2.axis('off')

        # 准备统计信息文本
        stats_text = [
            f"Missing Fields: {missing_field_count}",
            f"Empty Conversations: {empty_conversation_count}",
            f"Invalid Items: {invalid_item_count}",
            f"Total Anomalies: {sum(counts)}",
        ]

        # 将统计信息绘制在右侧画布上
        for i, text in enumerate(stats_text):
            ax2.text(0.1, 0.9 - i * 0.1, text, fontsize=12, ha='left', va='center', color='black')

        # 保存图像
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "03_anomaly_statistics.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Anomaly statistics plot saved at: {output_path}")

    except Exception as e:
        print(f"Error in plot_anomaly_statistics: {e}")




def plot_token_distribution(token_analysis: Dict[str, Any], role: str, output_dir: str):
    """
    Plot high-frequency and low-frequency tokens for a specific role.

    Args:
        token_analysis (dict): A dictionary containing token analysis results.
        role (str): Role name, e.g., "human" or "assistant".
        output_dir (str): Directory to save the generated plot.
    """
    try:
        high_freq_tokens = token_analysis.get('high_freq_tokens', {})
        low_freq_tokens = token_analysis.get('low_freq_tokens', {})

        # Extract high-frequency and low-frequency tokens
        if high_freq_tokens:
            high_tokens, high_counts = zip(*high_freq_tokens.items())
        else:
            high_tokens, high_counts = [], []

        if low_freq_tokens:
            low_tokens, low_counts = zip(*low_freq_tokens.items())
        else:
            low_tokens, low_counts = [], []

        # Create a figure
        plt.figure(figsize=(12, 6))

        # High-frequency tokens
        plt.subplot(121)
        if high_tokens:
            plt.bar(high_tokens, high_counts, color='green')
            plt.title(f"{role.capitalize()} High Frequency Tokens")
            plt.xlabel("Token")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, "No High Frequency Tokens", fontsize=12, ha='center', va='center')
            plt.title(f"{role.capitalize()} High Frequency Tokens")
            plt.axis('off')

        # Low-frequency tokens
        plt.subplot(122)
        if low_tokens:
            plt.bar(low_tokens, low_counts, color='red')
            plt.title(f"{role.capitalize()} Low Frequency Tokens")
            plt.xlabel("Token")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, "No Low Frequency Tokens", fontsize=12, ha='center', va='center')
            plt.title(f"{role.capitalize()} Low Frequency Tokens")
            plt.axis('off')

        # Adjust layout and save the plot
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/04_{role}_token_analysis.png")
        plt.close()

    except Exception as e:
        print(f"Error in plot_token_distribution for role '{role}': {e}")


def visualize_results(results: Dict[str, Any], output_dir: str, analysis_flags: Dict[str, bool]):
    """
    Unified visualization for all analysis results.

    Args:
        results (dict): A dictionary containing all analysis results.
        output_dir (str): Directory to save the results.
        analysis_flags (dict): Flags to control which analyses to visualize.
    """
    try:
        # Data Statistics
        if analysis_flags.get("dataset_statistics", False):
            plot_data_statistics(results.get("dataset_statistics", {}), output_dir)

        # Field Distribution
        if analysis_flags.get("language_distribution", False):
            plot_field_distribution(results.get("language_distribution", {}), output_dir)

        # Path Validation
        if analysis_flags.get("image_path_analysis", False):
            plot_image_path_distribution(results.get("image_path_analysis", {}), output_dir)

        # Anomaly Detection
        if analysis_flags.get("data_anomalies", False):
            plot_anomaly_statistics(results.get("data_anomalies", {}), output_dir)

        # Token Analysis for Human
        if analysis_flags.get("conversation_tokens", False) and "human" in results.get("conversation_tokens", {}):
            plot_token_distribution(results["conversation_tokens"]["human"], "human", output_dir)

        # Token Analysis for Assistant
        if analysis_flags.get("conversation_tokens", False) and "assistant" in results.get("conversation_tokens", {}):
            plot_token_distribution(results["conversation_tokens"]["assistant"], "assistant", output_dir)

    except Exception as e:
        print(f"Error in visualize_results: {e}")