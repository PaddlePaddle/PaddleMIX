# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


class DataVisualizer:
    def __init__(self, output_dir="outputs"):
        plt.style.use("default")
        sns.set_palette("husl")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_data_distribution(
        self, stats: Dict, title: str = "Data Distribution", filename: str = "distribution.png"
    ):
        """Plot data distribution and save to file"""
        try:
            # Filter numeric values for plotting
            numeric_stats = {k: v for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

            plt.figure(figsize=(12, 6))

            # Plot numeric statistics
            plt.subplot(1, 2, 1)
            df = pd.DataFrame(list(numeric_stats.items()), columns=["Metric", "Value"])
            sns.barplot(x="Value", y="Metric", data=df)
            plt.title("Numeric Metrics")

            # Plot distribution statistics (if available)
            plt.subplot(1, 2, 2)
            dist_data = []
            if "language_dist" in stats:
                dist_data.extend([("Language: " + k, v) for k, v in stats["language_dist"].items()])
            if "format_dist" in stats:
                dist_data.extend([("Format: " + k, v) for k, v in stats["format_dist"].items()])

            if dist_data:
                df_dist = pd.DataFrame(dist_data, columns=["Category", "Count"])
                sns.barplot(x="Count", y="Category", data=df_dist)
                plt.title("Distributions")

            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
        except Exception as e:
            print(f"Error plotting data distribution: {e}")

    def plot_quality_scores(
        self, scores: List[float], title: str = "Quality Scores Distribution", filename: str = "quality_scores.png"
    ):
        """Plot quality scores distribution and save to file"""
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(scores, bins=20, kde=True)
            plt.title(title)
            plt.xlabel("Quality Score")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
        except Exception as e:
            print(f"Error plotting quality scores: {e}")

    def compare_results(
        self, before: Dict, after: Dict, title: str = "Processing Results Comparison", filename: str = "comparison.png"
    ):
        """Compare processing results and save to file"""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(15, 12))

            # Compare numeric metrics
            numeric_before = {
                k: v for k, v in before.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            numeric_after = {k: v for k, v in after.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

            # Create comparison dataframe
            df = pd.DataFrame({"Before": numeric_before, "After": numeric_after}).reset_index()
            df = pd.melt(df, id_vars=["index"], var_name="Stage", value_name="Value")

            sns.barplot(x="Value", y="index", hue="Stage", data=df, ax=axes[0])
            axes[0].set_title("Numeric Metrics Comparison")

            # Compare distributions
            dist_data = []
            if "language_dist" in before and "language_dist" in after:
                all_langs = set(before["language_dist"].keys()) | set(after["language_dist"].keys())
                for lang in all_langs:
                    dist_data.extend(
                        [
                            ("Language: " + lang, "Before", before["language_dist"].get(lang, 0)),
                            ("Language: " + lang, "After", after["language_dist"].get(lang, 0)),
                        ]
                    )

            if dist_data:
                df_dist = pd.DataFrame(dist_data, columns=["Category", "Stage", "Count"])
                sns.barplot(x="Count", y="Category", hue="Stage", data=df_dist, ax=axes[1])
                axes[1].set_title("Distribution Comparison")

            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
        except Exception as e:
            print(f"Error comparing results: {e}")

    def display_sample(self, sample: Dict, filename: str = "sample.png"):
        """Save sample image and conversations to file"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Display image
            if "image" in sample:
                img = Image.open(sample["image"])
                axes[0].imshow(img)
                axes[0].axis("off")
                axes[0].set_title("Image")

            # Display conversations
            if "conversations" in sample:
                conversation_text = ""
                for conv in sample["conversations"]:
                    if "from" in conv and "value" in conv:
                        conversation_text += f"{conv['from']}: {conv['value']}\n\n"

                axes[1].text(0.1, 0.5, conversation_text, fontsize=10, wrap=True)
                axes[1].axis("off")
                axes[1].set_title("Conversations")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
        except Exception as e:
            print(f"Error saving sample: {e}")
