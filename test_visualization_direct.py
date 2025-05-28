#!/usr/bin/env python3
"""
Test the visualization directly to ensure we're using the right one
"""

import pickle
from pathlib import Path

# Explicitly import the correct visualization function
from perceptionhd.visualize_v21_fully_generic import generate_visualization_v21_fully_generic

# Load the saved results
results_path = Path("progress_output/analysis_results.pkl")
with open(results_path, 'rb') as f:
    results = pickle.load(f)

print("Loaded results from:", results_path)

# Generate visualization with the correct function
output_path = Path("progress_output/test_generic_viz.html")
print(f"\nGenerating visualization to: {output_path}")

generate_visualization_v21_fully_generic(results, output_path)

print(f"\nDone! Open: file://{output_path.absolute()}")