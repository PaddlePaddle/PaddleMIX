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

import logging
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

logger = logging.getLogger(__name__)


class DataVisualizer:
    def __init__(self, output_dir="outputs"):
        """Initialize DataVisualizer with configuration for saving visualizations.

        Args:
            output_dir (str, optional): Directory to save visualization files.
                Defaults to "outputs". Will be created if it doesn't exist.
        """
        plt.style.use("default")
        sns.set_palette("husl")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.style_colors = sns.color_palette("husl", 8)
        self.figure_dpi = 300
        # Add figure counter and max figures limit
        self.max_figures = 20
        self.current_figures = 0

    def _create_figure(self, *args, **kwargs):
        """Safely create a new figure, managing memory by closing old figures.

        Args:
            *args: Positional arguments passed to plt.figure()
            **kwargs: Keyword arguments passed to plt.figure()

        Returns:
            matplotlib.figure.Figure: The created figure object

        Note:
            Automatically closes old figures when reaching max_figures limit
            to prevent memory leaks.
        """
        if self.current_figures >= self.max_figures:
            plt.close("all")
            self.current_figures = 0
        fig = plt.figure(*args, **kwargs)
        self.current_figures += 1
        return fig

    def _create_subplots(self, *args, **kwargs):
        """Safely create subplots, managing memory by closing old figures.

        Args:
            *args: Positional arguments passed to plt.subplots()
            **kwargs: Keyword arguments passed to plt.subplots()

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
                The created figure and axes objects

        Note:
            Automatically closes old figures when reaching max_figures limit
            to prevent memory leaks.
        """
        if self.current_figures >= self.max_figures:
            plt.close("all")
            self.current_figures = 0
        fig, axes = plt.subplots(*args, **kwargs)
        self.current_figures += 1
        return fig, axes

    def plot_data_distribution(
        self, stats: Dict, title: str = "Data Distribution", filename: str = "distribution.png"
    ):
        """Plot and save visualization of dataset statistics.

        Creates a two-panel plot showing:
        1. Numeric metrics (counts, averages, etc.)
        2. Distributions (languages, formats, etc.)

        Args:
            stats (Dict): Dictionary containing dataset statistics
            title (str, optional): Plot title. Defaults to "Data Distribution".
            filename (str, optional): Output filename. Defaults to "distribution.png".

        Returns:
            None: Saves plot to output directory

        Raises:
            ValueError: If stats dictionary is empty or invalid
        """
        try:
            # Filter numeric values for plotting
            numeric_stats = {k: v for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

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
        """Plot and save histogram of quality scores.

        Args:
            scores (List[float]): List of quality scores (0-1)
            title (str, optional): Plot title. Defaults to "Quality Scores Distribution".
            filename (str, optional): Output filename. Defaults to "quality_scores.png".

        Returns:
            None: Saves plot to output directory

        Note:
            Includes KDE (Kernel Density Estimate) overlay on histogram
        """
        try:
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
        """Plot and save comparison of processing results.

        Creates a two-panel plot showing:
        1. Comparison of numeric metrics before and after processing
        2. Comparison of distributions (e.g. languages) before and after

        Args:
            before (Dict): Statistics before processing
            after (Dict): Statistics after processing
            title (str, optional): Plot title. Defaults to "Processing Results Comparison".
            filename (str, optional): Output filename. Defaults to "comparison.png".

        Returns:
            None: Saves plot to output directory

        Raises:
            ValueError: If input dictionaries are empty or invalid
        """
        try:
            fig, axes = self._create_subplots(2, 1, figsize=(15, 12))

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
        """Visualize and save a sample image with its conversations.

        Args:
            sample (Dict): Sample containing:
                - image: Path to image file
                - conversations: List of conversation turns
            filename (str, optional): Output filename. Defaults to "sample.png".

        Returns:
            None: Saves visualization to output directory

        Note:
            If image or conversations are missing, those panels will be empty
        """
        try:
            fig, axes = self._create_subplots(1, 2, figsize=(15, 6))

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

    def plot_image_quality_metrics(self, metrics: Dict[str, float], filename: str = "image_quality.png"):
        """Plot and save radar chart of image quality metrics.

        Args:
            metrics (Dict[str, float]): Dictionary of image quality metrics including:
                - quality_score: Overall quality score
                - blur_score: Blur detection score
                - resolution_score: Resolution quality score
                - brightness_score: Lighting quality score
                - contrast_score: Image contrast score
                - color_score: Color quality score
                - noise_score: Image noise level score
                - saturation_score: Color saturation score
            filename (str, optional): Output filename. Defaults to "image_quality.png".

        Returns:
            None: Saves plot to output directory
        """
        try:
            # Extract metrics
            quality_metrics = {
                "Quality Score": metrics["quality_score"],
                "Blur Score": metrics["blur_score"],
                "Resolution Score": metrics["resolution_score"],
                "Brightness Score": metrics["brightness_score"],
                "Contrast Score": metrics["contrast_score"],
                "Color Score": metrics["color_score"],
                "Noise Score": metrics["noise_score"],
                "Saturation Score": metrics["saturation_score"],
            }

            # Create radar chart
            categories = list(quality_metrics.keys())
            values = list(quality_metrics.values())

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            values = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))

            ax = plt.subplot(111, polar=True)
            ax.plot(angles, values, "o-", linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(angles[:-1] * 180 / np.pi, categories)

            plt.title("Image Quality Metrics")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.figure_dpi)
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting image quality metrics: {e}")

    def plot_text_quality_analysis(self, metrics: Dict[str, Any], filename: str = "text_quality.png"):
        """Plot and save text quality metrics visualization.

        Creates a two-panel plot showing:
        1. Basic text metrics (overall score, vocabulary diversity, readability)
        2. Detailed metrics (sentence length, punctuation ratio)

        Args:
            metrics (Dict[str, Any]): Dictionary of text quality metrics including:
                - text_score: Overall text quality score
                - metrics: Dictionary containing detailed metrics
            filename (str, optional): Output filename. Defaults to "text_quality.png".

        Returns:
            None: Saves plot to output directory
        """
        try:
            fig, (ax1, ax2) = self._create_subplots(1, 2, figsize=(15, 6))

            # Plot basic metrics
            basic_metrics = {
                "Text Score": metrics["text_score"],
                "Vocabulary Diversity": metrics["metrics"]["vocabulary_diversity"],
                "Readability Score": metrics["metrics"]["readability_score"],
            }

            ax1.bar(basic_metrics.keys(), basic_metrics.values(), color=self.style_colors[:3])
            ax1.set_ylim(0, 1)
            ax1.set_title("Basic Text Metrics")

            # Plot detailed metrics
            detailed_metrics = {
                "Avg Sentence Length": metrics["metrics"]["avg_sentence_length"],
                "Punctuation Ratio": metrics["metrics"]["punctuation_ratio"],
            }

            ax2.bar(detailed_metrics.keys(), detailed_metrics.values(), color=self.style_colors[3:5])
            ax2.set_title("Detailed Text Metrics")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.figure_dpi)
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting text quality analysis: {e}")

    def plot_matching_analysis(self, matching_info: Dict[str, Any], filename: str = "matching_analysis.png"):
        """Plot and save gauge chart of image-text matching score.

        Args:
            matching_info (Dict[str, Any]): Dictionary containing:
                - similarity_score: Image-text similarity score (0-1)
            filename (str, optional): Output filename. Defaults to "matching_analysis.png".

        Returns:
            None: Saves plot to output directory
        """
        try:
            fig = self._create_figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="polar")

            similarity_score = matching_info["similarity_score"]

            # Create gauge chart
            gauge = np.linspace(0, 1, 100)
            angle = np.linspace(-np.pi / 2, np.pi / 2, 100)

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="polar")

            # Plot gauge background
            ax.plot(angle, gauge, color="lightgray", linewidth=20, alpha=0.3)

            # Plot similarity score
            score_angle = np.linspace(-np.pi / 2, -np.pi / 2 + np.pi * similarity_score, 100)
            score_gauge = np.linspace(0, similarity_score, 100)
            ax.plot(score_angle, score_gauge, color=self.style_colors[0], linewidth=20)

            # Customize gauge appearance
            ax.set_rticks([])
            ax.set_thetagrids([])
            ax.set_title(f"Image-Text Similarity Score: {similarity_score:.2f}")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.figure_dpi)
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting matching analysis: {e}")

    def plot_language_distribution(self, language_dist: Dict[str, float], filename: str = "language_dist.png"):
        """Plot and save bar chart of language distribution.

        Args:
            language_dist (Dict[str, float]): Dictionary mapping language codes to proportions
            filename (str, optional): Output filename. Defaults to "language_dist.png".

        Returns:
            None: Saves plot to output directory

        Note:
            Languages are sorted by proportion in descending order
        """
        try:
            # Sort languages by proportion
            sorted_langs = sorted(language_dist.items(), key=lambda x: x[1], reverse=True)
            langs, proportions = zip(*sorted_langs)

            # Create bar plot
            plt.bar(langs, proportions, color=self.style_colors)
            plt.xticks(rotation=45, ha="right")
            plt.title("Language Distribution")
            plt.ylabel("Proportion")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.figure_dpi)
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting language distribution: {e}")

    def generate_report(self, stats: Dict, filename: str = "analysis_report.html"):
        """Generate and save comprehensive analysis report in HTML format.

        Args:
            stats (Dict): Dictionary containing analysis statistics
            filename (str, optional): Output filename. Defaults to "analysis_report.html".

        Returns:
            None: Saves HTML report to output directory

        Note:
            Report includes:
            - Dataset overview
            - Quality metrics
            - Links to visualization images
        """
        try:
            # Simplified HTML template with inline CSS
            report_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Data Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}
        .section {{
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .metric {{
            margin: 10px 0;
            color: #333;
        }}
        .chart {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
        }}
        h1, h2 {{
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <h1>Data Analysis Report</h1>

    <div class="section">
        <h2>Dataset Overview</h2>
        <div class="metric">Total Samples: {total_samples:,}</div>
        <div class="metric">Valid Samples: {valid_samples:,}</div>
        <div class="metric">Invalid Samples: {invalid_samples:,}</div>
    </div>

    <div class="section">
        <h2>Quality Metrics</h2>
        <div class="metric">Average Text Length: {avg_text_length:.2f}</div>
        <div class="metric">Average Conversations: {avg_conversations:.2f}</div>
    </div>

    <div class="section">
        <h2>Visualizations</h2>
        <div class="chart">
            <img src="distribution.png" alt="Data Distribution"/>
        </div>
        <div class="chart">
            <img src="quality_scores.png" alt="Quality Scores"/>
        </div>
    </div>
</body>
</html>"""

            # Format numbers in stats
            formatted_stats = stats.copy()
            for key in ["total_samples", "valid_samples", "invalid_samples"]:
                if key in formatted_stats:
                    formatted_stats[key] = int(formatted_stats[key])

            # Create report content
            report_content = report_template.format(**formatted_stats)

            # Save report
            with open(os.path.join(self.output_dir, filename), "w", encoding="utf-8") as f:
                f.write(report_content)

        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
