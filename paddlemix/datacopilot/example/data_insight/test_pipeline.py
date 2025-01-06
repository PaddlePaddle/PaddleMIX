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

from data_analysis import DataAnalyzer
from data_augmentation import DataAugmentor
from data_cleaning import DataCleaner
from visualization import DataVisualizer


def test_pipeline():
    # Initialize modules
    analyzer = DataAnalyzer()
    cleaner = DataCleaner()
    augmentor = DataAugmentor()
    visualizer = DataVisualizer(output_dir="test_outputs")

    # Test data path
    test_data_path = "../test_data.json"

    # Ensure test data exists
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file not found at {test_data_path}")
        return

    try:
        # 1. Data Analysis
        print("Running data analysis...")
        stats = analyzer.analyze(test_data_path)
        visualizer.plot_data_distribution(stats, filename="initial_distribution.png")

        # 2. Data Cleaning
        print("Running data cleaning...")
        cleaned_data = cleaner.clean(test_data_path)

        # 3. Data Augmentation
        print("Running data augmentation...")
        augmented_data = augmentor.augment(cleaned_data)

        # 4. Result Visualization
        print("Visualizing results...")
        # Compare before/after stats
        before_stats = stats
        after_stats = analyzer.analyze(augmented_data)
        visualizer.compare_results(before=before_stats, after=after_stats, title="Data Processing Results Comparison")

        # Display sample results
        if augmented_data:
            visualizer.display_sample(augmented_data[0], filename="sample_result.png")

        print("Pipeline test completed successfully!")

    except Exception as e:
        print(f"Error in pipeline: {e}")


if __name__ == "__main__":
    test_pipeline()
