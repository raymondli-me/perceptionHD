#!/usr/bin/env python3
"""
Simple test of the PerceptionHD pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perceptionhd import PerceptionHDPipeline
from pathlib import Path

# Setup paths
base_dir = Path(__file__).parent
example_dir = base_dir / "examples" / "ai_social_class"
output_dir = base_dir / "test_output"
output_dir.mkdir(exist_ok=True)

print("Running PerceptionHD pipeline test...")

# Initialize and run
pipeline = PerceptionHDPipeline(
    data_path=str(example_dir / "data.csv"),
    embeddings_path=str(example_dir / "embeddings.npy"),
    config_path=str(example_dir / "config.yaml"),
    output_dir=str(output_dir)
)

# Run full analysis
results = pipeline.run_full_analysis()

print(f"\n✅ Test complete! Results saved to: {output_dir}")
print(f"   HTML visualization: {output_dir}/perception_hd_visualization.html")