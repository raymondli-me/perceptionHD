#!/usr/bin/env python3
"""
PerceptionHD Visualization Generator
Adapts the v21 visualization to use generic X/Y variables
"""

import json
import numpy as np
from pathlib import Path


def generate_visualization(results, output_path):
    """
    Generate the interactive HTML visualization
    
    Args:
        results: Dictionary containing all analysis results
        output_path: Path to save the HTML file
    """
    
    # Extract data from results
    df = results['df']
    X_umap = results['X_umap']
    config = results['config']
    
    # Variable names from config
    X_name = config['variables']['X']['name']
    X_short = config['variables']['X']['short_name']
    Y_name = config['variables']['Y']['name']
    Y_short = config['variables']['Y']['short_name']
    
    # Prepare visualization data
    viz_data = []
    for i in range(len(df)):
        viz_data.append({
            'id': df.iloc[i]['id'],
            'text': df.iloc[i]['text'],
            'X': float(df.iloc[i]['X']),
            'Y': float(df.iloc[i]['Y']),
            'x': float(X_umap[i, 0]),
            'y': float(X_umap[i, 1]),
            'z': float(X_umap[i, 2]) if X_umap.shape[1] > 2 else 0,
            'cluster': int(results['cluster_labels'][i]),
            # Add more fields as needed
        })
    
    # For now, create a simple HTML template
    # In the full implementation, this would use the complete v21 template
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{config['display']['title']}</title>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #000;
            color: #fff;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 20px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h2>{config['display']['title']}</h2>
        <p>Analysis of {X_name} vs {Y_name}</p>
        <p>Samples: {len(df)}</p>
        <p>{X_name} range: {df['X'].min():.2f} - {df['X'].max():.2f}</p>
        <p>{Y_name} distribution: {dict(df['Y'].value_counts().sort_index())}</p>
        
        <h3>DML Results</h3>
        <p>Naive effect: {results['dml_results']['theta_naive']:.3f} (p={results['dml_results']['pval_naive']:.4f})</p>
        <p>With 200 PCs: {results['dml_results']['theta_200']:.3f} (p={results['dml_results']['pval_200']:.4f})</p>
        <p>With Top 5 PCs: {results['dml_results']['theta_top5']:.3f} (p={results['dml_results']['pval_top5']:.4f})</p>
        
        <p><em>Full interactive visualization coming soon...</em></p>
    </div>
    
    <script>
        // Placeholder for Three.js visualization
        const data = {json.dumps(viz_data[:100])};  // First 100 points for demo
        console.log('Data loaded:', data.length, 'points');
        
        // In the full implementation, this would include:
        // - Three.js 3D visualization
        // - All interactive panels
        // - Topic visualization
        // - Gallery mode
        // - PC analysis
        // etc.
    </script>
</body>
</html>
"""
    
    # Save HTML
    with open(output_path, 'w') as f:
        f.write(html_content)
        
    print(f"Visualization saved to {output_path}")