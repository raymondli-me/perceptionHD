#!/usr/bin/env python3
"""
PerceptionHD Visualization - Direct v21 port with label substitution
"""

import json
import numpy as np
from pathlib import Path


def generate_visualization_v21_exact(results, output_path):
    """
    Generate visualization by loading v21 template and substituting labels
    """
    
    # Load the original v21 template
    v21_path = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_dml_pc_analysis/minimal_umap_viz_v21.html")
    
    with open(v21_path, 'r') as f:
        html_content = f.read()
    
    # Get config for variable names
    config = results['config']
    X_name = config['variables']['X']['name']
    X_short = config['variables']['X']['short_name']
    Y_name = config['variables']['Y']['name']
    Y_short = config['variables']['Y']['short_name']
    
    # Prepare the data exactly as v21 expects it
    df = results['df']
    X_umap = results.get('umap_coords', results.get('X_umap'))
    clusters = results.get('clusters', results.get('cluster_labels'))
    
    # Create the essays data structure exactly as v21 expects
    essays_data = []
    for i in range(len(df)):
        essay_dict = {
            'x': float(X_umap[i, 0]),
            'y': float(X_umap[i, 1]),
            'z': float(X_umap[i, 2]) if X_umap.shape[1] > 2 else 0,
            'id': str(df.iloc[i]['id']),
            'text': str(df.iloc[i]['text']),
            'ai_rating': float(df.iloc[i]['X']),  # Keep original field name
            'social_class': float(df.iloc[i]['Y']),  # Keep original field name
            'hdbscan_topic_id': int(clusters[i]),
            'pc_info': []
        }
        
        # Add PC info if available
        if 'contributions_x' in results and 'contributions_y' in results:
            contrib_x = results['contributions_x'][i]
            contrib_y = results['contributions_y'][i]
            total_contrib = np.abs(contrib_x) + np.abs(contrib_y)
            top_5_sample = np.argsort(total_contrib)[-5:][::-1]
            
            for pc_idx in top_5_sample:
                if 'pc_percentiles' in results:
                    percentile = results['pc_percentiles'][i, pc_idx]
                else:
                    pc_values = results['X_pca'][:, pc_idx]
                    percentile = (np.searchsorted(np.sort(pc_values), results['X_pca'][i, pc_idx]) / len(pc_values)) * 100
                
                pc_info = {
                    'pc': f'PC{pc_idx}',
                    'percentile': float(percentile),
                    'contribution_ai': float(contrib_x[pc_idx]),  # Keep original field name
                    'contribution_sc': float(contrib_y[pc_idx]),   # Keep original field name
                    'variance_total': float(results.get('variance_explained', np.zeros(200))[pc_idx] * 100)
                }
                essay_dict['pc_info'].append(pc_info)
        
        essays_data.append(essay_dict)
    
    # Replace the essays data in the HTML
    html_content = html_content.replace(
        "const essays = [",
        f"const essays = {json.dumps(essays_data, indent=2)}\n        // Original array start: ["
    )
    
    # Now do the label replacements throughout the HTML
    replacements = [
        # Variable names in displays
        ("AI Rating", X_name),
        ("Social Class", Y_name),
        ("AI", X_short),
        ("SC", Y_short),
        
        # Threshold labels
        ("AI Rating Low", f"{X_name} Low"),
        ("AI Rating High", f"{X_name} High"),
        ("Social Class Low", f"{Y_name} Low"),
        ("Social Class High", f"{Y_name} High"),
        
        # Legend
        ("High AI + High SC", f"High {X_short} + High {Y_short}"),
        ("High AI + Low SC", f"High {X_short} + Low {Y_short}"),
        ("Low AI + High SC", f"Low {X_short} + High {Y_short}"),
        ("Low AI + Low SC", f"Low {X_short} + Low {Y_short}"),
        
        # Table headers
        (">AI Contrib<", f">{X_short} Contrib<"),
        (">SC Contrib<", f">{Y_short} Contrib<"),
        ("SC → AI", f"{Y_short} → {X_short}"),
        ("AI → SC", f"{X_short} → {Y_short}"),
        
        # Essay metadata display
        ('AI Rating:', f'{X_name}:'),
        ('Social Class:', f'{Y_name}:'),
        
        # Topic stats headers
        ("Top 10% AI", f"Top 10% {X_short}"),
        ("Bottom 10% AI", f"Bottom 10% {X_short}"),
        ("High SC", f"High {Y_short}"),
        ("Low SC", f"Low {Y_short}"),
        
        # Title
        ("AI Rating vs Social Class Analysis", config['display']['title'])
    ]
    
    for old_text, new_text in replacements:
        html_content = html_content.replace(old_text, new_text)
    
    # Handle the essay display JavaScript
    html_content = html_content.replace(
        "essay.ai_rating.toFixed(2)",
        "essay.ai_rating.toFixed(2)"  # Keep the field names the same
    )
    html_content = html_content.replace(
        "essay.social_class.toFixed(2)", 
        "essay.social_class.toFixed(2)"  # Keep the field names the same
    )
    
    # Update the DML results data
    dml_results = results['dml_results']
    
    # Find and replace the DML theta values
    dml_replacements = [
        (r'<td>-?\d+\.\d+</td>(\s*<!--\s*theta_naive\s*-->)', f"<td>{dml_results.get('theta_naive', 0):.3f}</td>\\1"),
        (r'<td>-?\d+\.\d+</td>(\s*<!--\s*theta_200\s*-->)', f"<td>{dml_results.get('theta_200', 0):.3f}</td>\\1"),
        (r'<td>-?\d+\.\d+</td>(\s*<!--\s*theta_top5\s*-->)', f"<td>{dml_results.get('theta_top5', 0):.3f}</td>\\1"),
    ]
    
    # Since we don't have markers, let's do it more directly
    # Look for the DML table section and update values
    if 'theta_naive' in dml_results:
        # This is a bit hacky but will work for now
        html_content = html_content.replace('>0.106<', f">{dml_results['theta_naive']:.3f}<")
        html_content = html_content.replace('>0.055<', f">{dml_results.get('theta_200', dml_results['theta_naive']):.3f}<")
        html_content = html_content.replace('>0.045<', f">{dml_results.get('theta_top5', dml_results['theta_naive']):.3f}<")
    
    # Update topic data
    if 'topics' in results or 'topic_keywords' in results:
        topics = results.get('topics', results.get('topic_keywords', {}))
        topic_data = []
        
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:
                continue
            cluster_points = X_umap[clusters == cluster_id]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                
                # Get keywords
                if cluster_id in topics:
                    if isinstance(topics[cluster_id], list):
                        keywords = ', '.join([w[0] for w in topics[cluster_id][:5]])
                    else:
                        keywords = str(topics[cluster_id])
                else:
                    keywords = f"Topic {cluster_id}"
                
                topic_data.append({
                    'topic_id': int(cluster_id),
                    'keywords': keywords,
                    'centroid': [float(x) for x in centroid],
                    'size': int(np.sum(clusters == cluster_id))
                })
        
        # Replace topics data
        html_content = html_content.replace(
            "const topics = [",
            f"const topics = {json.dumps(topic_data, indent=2)}\n        // Original array start: ["
        )
    
    # Save the modified HTML
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Full v21 visualization saved to {output_path}")
    
    return output_path