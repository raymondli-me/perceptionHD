#!/usr/bin/env python3
"""
Quick demo of PerceptionHD package with progress tracking.
"""

import os
import sys
import time
from pathlib import Path

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perceptionhd import PerceptionHDPipeline

def main():
    """Run a quick demo of the PerceptionHD pipeline."""
    
    # Define paths
    base_dir = Path(__file__).parent
    example_dir = base_dir / "examples" / "ai_social_class"
    
    # Input files
    data_path = example_dir / "data.csv"
    embeddings_path = example_dir / "embeddings.npy"
    config_path = example_dir / "config.yaml"
    
    # Output directory
    output_dir = base_dir / "quick_demo_output"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PerceptionHD Quick Demo")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    start_time = time.time()
    
    pipeline = PerceptionHDPipeline(
        data_path=str(data_path),
        embeddings_path=str(embeddings_path),
        config_path=str(config_path),
        output_dir=str(output_dir)
    )
    
    print("2. Loading data...")
    pipeline.load_data()
    print(f"   - Loaded {len(pipeline.df)} essays")
    print(f"   - Embeddings shape: {pipeline.embeddings.shape}")
    
    print("\n3. Computing PCA (this may take a moment)...")
    pipeline.compute_pca()
    print(f"   - PCA features shape: {pipeline.pca_features.shape}")
    
    print("\n4. Computing UMAP 3D projection...")
    pipeline.compute_umap()
    print(f"   - UMAP coordinates shape: {pipeline.umap_coords.shape}")
    
    print("\n5. Computing PC contributions with XGBoost...")
    pipeline.compute_contributions()
    print(f"   - Top contributing PCs identified")
    
    print("\n6. Running Double Machine Learning (DML)...")
    pipeline.compute_dml()
    print(f"   - DML analysis complete")
    
    print("\n7. Performing HDBSCAN clustering...")
    pipeline.compute_clustering()
    n_clusters = len(set(pipeline.clusters)) - (1 if -1 in pipeline.clusters else 0)
    print(f"   - Found {n_clusters} topics")
    
    print("\n8. Extracting topic keywords...")
    pipeline.extract_topics()
    print(f"   - Keywords extracted for each topic")
    
    print("\n9. Calculating statistics...")
    pipeline.calculate_statistics()
    
    print("\n10. Saving results...")
    pipeline.save_results()
    
    # Generate visualization
    print("\n11. Generating interactive visualization...")
    from perceptionhd.visualize import create_visualization
    
    # Load the saved results
    import pickle
    results_path = output_dir / "analysis_results.pkl"
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Create visualization
    html_path = create_visualization(results, pipeline.config, output_dir)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Analysis complete in {elapsed:.1f} seconds!")
    
    print(f"\n📊 Results saved to: {output_dir}")
    print(f"🌐 Interactive visualization: {html_path}")
    
    # List output files
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()