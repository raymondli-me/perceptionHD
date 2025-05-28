#!/usr/bin/env python3
"""
Run PerceptionHD with detailed progress bars
"""

import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced pipeline
from perceptionhd.pipeline_with_progress import PerceptionHDPipelineWithProgress

def main():
    # Setup paths
    base_dir = Path(__file__).parent
    example_dir = base_dir / "examples" / "ai_social_class"
    output_dir = base_dir / "progress_output"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PerceptionHD Analysis with Progress Tracking")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PerceptionHDPipelineWithProgress(
        data_path=str(example_dir / "data.csv"),
        embeddings_path=str(example_dir / "embeddings.npy"),
        config_path=str(example_dir / "config.yaml"),
        output_dir=str(output_dir)
    )
    
    # Run full analysis
    results = pipeline.run_full_analysis()
    
    print("\nTo view the results, open:")
    print(f"  file://{output_dir}/perception_hd_visualization.html")

if __name__ == "__main__":
    main()