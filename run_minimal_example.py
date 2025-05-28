#!/usr/bin/env python3
"""
Minimal example script demonstrating the PerceptionHD package.
This script runs the complete analysis pipeline from start to finish.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the package to path (for development mode)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perceptionhd import PerceptionHDPipeline

def main():
    """Run the complete PerceptionHD analysis pipeline."""
    
    # Define paths
    base_dir = Path(__file__).parent
    example_dir = base_dir / "examples" / "ai_social_class"
    
    # Input files
    data_path = example_dir / "data.csv"
    embeddings_path = example_dir / "embeddings.npy"
    config_path = example_dir / "config.yaml"
    
    # Output directory
    output_dir = base_dir / "minimal_example_output"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PerceptionHD Minimal Example")
    print("=" * 60)
    
    # Check if files exist
    print("\nChecking input files...")
    for path, name in [(data_path, "Data"), (embeddings_path, "Embeddings"), (config_path, "Config")]:
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} NOT FOUND")
            return
    
    # Initialize pipeline
    print("\nInitializing PerceptionHD pipeline...")
    pipeline = PerceptionHDPipeline(
        data_path=str(data_path),
        embeddings_path=str(embeddings_path),
        config_path=str(config_path),
        output_dir=str(output_dir)
    )
    
    # Run the complete analysis
    print("\nRunning full analysis pipeline...")
    print("This includes:")
    print("  1. Loading data and embeddings")
    print("  2. Computing PCA (200 components)")
    print("  3. Computing UMAP (3D projection)")
    print("  4. Computing XGBoost PC contributions")
    print("  5. Running Double Machine Learning (DML)")
    print("  6. Performing HDBSCAN clustering")
    print("  7. Extracting topic keywords")
    print("  8. Calculating statistics")
    print("  9. Generating interactive visualization")
    
    print("\nStarting analysis...")
    results = pipeline.run_full_analysis()
    
    # Display summary results
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    print(f"\nDataset: {len(results['df'])} essays analyzed")
    print(f"Embeddings: {results['embeddings'].shape}")
    print(f"PCA components: {results['pca_features'].shape[1]}")
    print(f"UMAP projection: {results['umap_coords'].shape}")
    
    print(f"\nVariables analyzed:")
    print(f"  X: {pipeline.config['variables']['X']['name']} (range: {results['df']['X'].min():.2f} - {results['df']['X'].max():.2f})")
    print(f"  Y: {pipeline.config['variables']['Y']['name']} (range: {results['df']['Y'].min():.2f} - {results['df']['Y'].max():.2f})")
    
    print(f"\nClustering results:")
    print(f"  Number of topics: {len(np.unique(results['clusters'])) - 1}")  # -1 for noise cluster
    print(f"  Topic sizes: {dict(zip(*np.unique(results['clusters'], return_counts=True)))}")
    
    print(f"\nDML results:")
    print(f"  Top 5 PCs model R²: {results['dml_results']['top5_r2']:.4f}")
    print(f"  All PCs model R²: {results['dml_results']['all_r2']:.4f}")
    print(f"  Top contributing PCs: {results['dml_results']['top_pcs']}")
    
    print(f"\nOutput files generated:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")
    
    # Path to the generated HTML visualization
    html_path = output_dir / "perception_hd_visualization.html"
    # Create the visualization
    from perceptionhd.visualize_v21_exact_copy import generate_visualization_v21_exact
    html_path = output_dir / "perception_hd_visualization.html"
    generate_visualization_v21_exact(results, html_path)
    
    print(f"\n✨ Interactive visualization created at:")
    print(f"   {html_path}")
    print(f"\nOpen this file in a web browser to explore the results!")
    
    # Optional: Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f"file://{html_path}")
        print("\n(Attempting to open in your default browser...)")
    except:
        pass

if __name__ == "__main__":
    main()