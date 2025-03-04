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

import argparse
import json
import os

from data_analysis import DataAnalyzer
from data_augmentation import DataAugmentor
from data_cleaning import DataCleaner
from visualization import DataVisualizer


def parse_args():
    """Parse and validate command line arguments for the data processing pipeline.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - annotation_file: Path to annotation JSON file
            - data_dir: Path to dataset directory
            - output_dir: Output directory path
            - clip_model: CLIP model name/path
            - chunk_size: Batch size for processing
            - quality_thresholds: Comma-separated quality thresholds

    Raises:
        argparse.ArgumentError: If required arguments are missing
    """
    parser = argparse.ArgumentParser(description="Data analysis and processing pipeline")

    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the annotation JSON file")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory containing images")

    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save output files (default: outputs)"
    )

    parser.add_argument(
        "--clip_model", type=str, default="openai/clip-vit-base-patch32", help="Name or path of the CLIP model to use"
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of samples to process in each batch (default: 1000)",
    )
    parser.add_argument(
        "--quality_thresholds",
        type=str,
        default="0.7,0.6,0.3",
        help="Comma-separated quality thresholds for image,text,matching (default: 0.7,0.6,0.3)",
    )

    return parser.parse_args()


def pipeline(args):
    """Run the complete data processing pipeline including analysis, cleaning, and augmentation.

    The pipeline performs the following steps:
    1. Initial data analysis and visualization
    2. Quality analysis of images and text
    3. Data cleaning based on quality thresholds
    4. Data augmentation of cleaned samples
    5. Final analysis and report generation

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - annotation_file: Path to annotation JSON file
            - data_dir: Path to dataset directory
            - output_dir: Output directory path
            - clip_model: CLIP model name/path
            - chunk_size: Batch size for processing
            - quality_thresholds: Comma-separated quality thresholds

    Returns:
        None: Results are saved to output directory

    Raises:
        FileNotFoundError: If input files/directories don't exist
        json.JSONDecodeError: If annotation file is invalid JSON
        RuntimeError: If any pipeline step fails
    """
    # Parse quality thresholds
    img_thresh, text_thresh, match_thresh = map(float, args.quality_thresholds.split(","))

    # Initialize modules with updated parameters
    analyzer = DataAnalyzer(clip_model_name=args.clip_model, dataset_dir=args.data_dir)
    cleaner = DataCleaner(
        clip_model_name=args.clip_model,
        dataset_dir=args.data_dir,
        image_quality_threshold=img_thresh,
        text_quality_threshold=text_thresh,
        matching_threshold=match_thresh,
    )
    augmentor = DataAugmentor(dataset_dir=args.data_dir)
    visualizer = DataVisualizer(output_dir=args.output_dir)

    # Validate input paths
    if not os.path.exists(args.annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {args.annotation_file}")

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # 1. Initial Data Analysis
        print("Running initial data analysis...")
        stats = analyzer.analyze(args.annotation_file, chunk_size=args.chunk_size)

        # Visualize initial statistics
        visualizer.plot_data_distribution(stats, title="Initial Data Distribution")
        visualizer.plot_language_distribution(stats["language_dist"], filename="initial_language_dist.png")

        # 2. Quality Analysis
        print("Analyzing data quality...")
        quality_metrics = {
            "image_quality": [],
            "text_quality": [],
            "matching_scores": [],
        }

        with open(args.annotation_file, "r") as f:
            samples = json.load(f)

        # Analyze sample subset
        analysis_samples = samples[: min(100, len(samples))]
        for sample in analysis_samples:
            if "image" in sample and "conversations" in sample:
                image_path = os.path.join(args.data_dir, sample["image"])
                text = " ".join(conv["value"] for conv in sample["conversations"])

                # Analyze image quality
                img_quality = analyzer.analyze_image_quality(image_path)
                quality_metrics["image_quality"].append(img_quality["quality_score"])
                visualizer.plot_image_quality_metrics(
                    img_quality, filename=f"image_quality_{len(quality_metrics['image_quality'])}.png"
                )

                # Analyze text quality
                text_quality = analyzer.analyze_text_quality(text)
                quality_metrics["text_quality"].append(text_quality["text_score"])
                visualizer.plot_text_quality_analysis(
                    text_quality, filename=f"text_quality_{len(quality_metrics['text_quality'])}.png"
                )

                # Analyze matching quality
                matching = analyzer.analyze_image_text_matching(image_path, text)
                if matching:
                    quality_metrics["matching_scores"].append(matching["similarity_score"])
                    visualizer.plot_matching_analysis(
                        matching, filename=f"matching_{len(quality_metrics['matching_scores'])}.png"
                    )

                # Display sample with metrics
                visualizer.display_sample(
                    {
                        "image": image_path,
                        "conversations": sample["conversations"],
                        "metrics": {
                            "image_quality": img_quality["quality_score"],
                            "text_quality": text_quality["text_score"],
                            "matching_score": matching["similarity_score"] if matching else 0.0,
                        },
                    },
                    filename=f"sample_{len(quality_metrics['image_quality'])}.png",
                )

        # Plot quality score distributions
        for metric_name, scores in quality_metrics.items():
            if scores:
                visualizer.plot_quality_scores(
                    scores,
                    title=f"{metric_name.replace('_', ' ').title()} Distribution",
                    filename=f"{metric_name}_distribution.png",
                )

        # 3. Data Cleaning
        print("Running data cleaning...")
        cleaned_data, clean_stats = cleaner.clean(args.annotation_file)

        # 4. Data Augmentation
        print("Running data augmentation...")
        try:
            augmented_data = augmentor.augment(cleaned_data)
        except Exception as e:
            print(f"Error in augmentation pipeline: {e}")
            augmented_data = cleaned_data

        # 5. Final Analysis and Visualization
        print("Generating final analysis...")
        after_stats = analyzer.analyze(augmented_data)

        # Compare and visualize results
        visualizer.compare_results(
            before=stats,
            after=after_stats,
            title="Before vs After Processing Comparison",
            filename="processing_comparison.png",
        )

        # Generate comprehensive report with all metrics
        report_stats = {
            **after_stats,
            "quality_metrics": quality_metrics,
            "cleaning_stats": clean_stats,
            "initial_stats": stats,
        }
        visualizer.generate_report(report_stats)

        # Save processed data
        output_json = os.path.join(args.output_dir, "processed_data.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)

        print(f"Pipeline completed successfully! Results saved to {args.output_dir}")

    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise


def main():
    """Main entry point for the data processing pipeline.

    Parses command line arguments and executes the data processing pipeline.
    Handles any exceptions that occur during pipeline execution.

    Returns:
        None: Results are saved to output directory

    Note:
        The pipeline creates the output directory if it doesn't exist.
        All generated files are saved in the output directory.
    """
    args = parse_args()
    pipeline(args)


if __name__ == "__main__":
    main()
