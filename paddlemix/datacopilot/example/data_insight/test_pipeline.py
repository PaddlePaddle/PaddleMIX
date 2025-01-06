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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data analysis and processing pipeline")

    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the annotation JSON file")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory containing images")

    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save output files (default: outputs)"
    )

    parser.add_argument(
        "--clip_model", type=str, default="openai/clip-vit-base-patch32", help="Name or path of the CLIP model to use"
    )

    return parser.parse_args()


def test_pipeline(args):
    """Run the data processing pipeline.

    Args:
        args: Command line arguments
    """
    # Initialize modules
    analyzer = DataAnalyzer(clip_model_name=args.clip_model, dataset_dir=args.data_dir)
    cleaner = DataCleaner(clip_model_name=args.clip_model, dataset_dir=args.data_dir)
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
        # 1. Data Analysis
        print("Running data analysis...")
        stats = analyzer.analyze(args.annotation_file)
        visualizer.plot_data_distribution(stats)

        # 2. Data Cleaning
        print("Running data cleaning...")
        cleaned_data, clean_stats = cleaner.clean(args.annotation_file)

        # 3. Data Augmentation
        print("Running data augmentation...")
        try:
            augmented_data = augmentor.augment(cleaned_data)
        except Exception as e:
            print(f"Error in augmentation pipeline: {e}")
            augmented_data = cleaned_data

        # 4. Result Visualization
        print("Visualizing results...")
        # Compare before/after stats
        before_stats = stats
        after_stats = analyzer.analyze(augmented_data)

        visualizer.compare_results(before=before_stats, after=after_stats, title="Data Processing Results Comparison")

        # Display sample results if available
        if augmented_data and len(augmented_data) > 0:
            visualizer.display_sample(augmented_data[0])

        # Save processed data
        output_json = os.path.join(args.output_dir, "processed_data.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)

        print(f"Pipeline completed successfully! Results saved to {args.output_dir}")

    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise


def main():
    """Main entry point."""
    args = parse_args()
    test_pipeline(args)


if __name__ == "__main__":
    main()
