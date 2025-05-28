#!/usr/bin/env python3
"""
PerceptionHD Command Line Interface
"""

import argparse
from pathlib import Path
from .pipeline import PerceptionHDPipeline
from .visualize import generate_visualization


def main():
    parser = argparse.ArgumentParser(
        description='PerceptionHD - High-dimensional perception analysis and visualization'
    )
    
    parser.add_argument(
        '--data', 
        required=True,
        help='Path to data CSV file (must have columns: id, text, X, Y)'
    )
    
    parser.add_argument(
        '--embeddings',
        required=True,
        help='Path to embeddings NPY file'
    )
    
    parser.add_argument(
        '--config',
        required=True,
        help='Path to config YAML file'
    )
    
    parser.add_argument(
        '--output',
        default='perceptionhd_output.html',
        help='Output HTML file path (default: perceptionhd_output.html)'
    )
    
    parser.add_argument(
        '--skip-computation',
        action='store_true',
        help='Skip computation and use existing results'
    )
    
    parser.add_argument(
        '--results-file',
        help='Path to existing results pickle file (if skip-computation)'
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("PerceptionHD - High-Dimensional Perception Analysis")
    print("=" * 50)
    
    if args.skip_computation:
        if not args.results_file:
            print("Error: --results-file required when using --skip-computation")
            return 1
            
        print(f"Loading existing results from {args.results_file}")
        import pickle
        with open(args.results_file, 'rb') as f:
            results = pickle.load(f)
    else:
        # Run full pipeline
        pipeline = PerceptionHDPipeline(
            data_path=args.data,
            embeddings_path=args.embeddings,
            config_path=args.config
        )
        
        results = pipeline.run_full_analysis()
    
    # Generate visualization
    print("\nGenerating visualization...")
    generate_visualization(results, args.output)
    
    print(f"\n✓ Visualization saved to: {args.output}")
    print("\nOpen the HTML file in a web browser to explore your data!")
    
    return 0


if __name__ == '__main__':
    exit(main())