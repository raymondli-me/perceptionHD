#!/usr/bin/env python3
"""
PerceptionHD Visualization - v21 with generic X/Y labels and dynamic DML values
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def generate_visualization_v21_generic(results, output_path):
    """
    Generate visualization using v21 template with generic X/Y labels
    """
    
    # Extract data from pipeline results
    essays_df = results['df'].copy()
    X_umap_3d = results.get('umap_coords', results.get('X_umap'))
    X_pca = results.get('pca_features', results.get('X_pca'))
    clusters = results.get('clusters', results.get('cluster_labels'))
    config = results['config']
    
    # Map to expected column names for v21 template
    essays_df['essay_id'] = essays_df['id']
    essays_df['essay'] = essays_df['text']
    essays_df['ai_rating'] = essays_df['Y']  # Keep internal naming for JS
    essays_df['sc11'] = essays_df['X']       # Keep internal naming for JS
    
    # Get top PCs and contributions
    top_pcs = results['dml_results']['top_pcs']
    contributions_ai = results['contributions_y']  # Y contributions
    contributions_sc = results['contributions_x']  # X contributions
    
    # Get variance explained
    variance_explained = results.get('variance_explained', results['dml_results'].get('variance_explained'))
    
    # Calculate percentiles
    pc_percentiles = results.get('pc_percentiles')
    if pc_percentiles is None:
        pc_percentiles = np.zeros((len(essays_df), X_pca.shape[1]))
        for i in range(X_pca.shape[1]):
            pc_values = X_pca[:, i]
            pc_percentiles[:, i] = (np.searchsorted(np.sort(pc_values), pc_values) / len(pc_values)) * 100
    
    # Topic keywords
    topic_keywords = results.get('topics', results.get('topic_keywords', {}))
    
    # Calculate statistics
    ai_ratings_clean = essays_df['ai_rating'].dropna()
    ai_percentiles = {
        10: ai_ratings_clean.quantile(0.10),
        25: ai_ratings_clean.quantile(0.25),
        75: ai_ratings_clean.quantile(0.75),
        90: ai_ratings_clean.quantile(0.90)
    }
    
    # Calculate center of point cloud
    center_x = X_umap_3d[:, 0].mean()
    center_y = X_umap_3d[:, 1].mean()
    center_z = X_umap_3d[:, 2].mean()
    
    # Prepare data for JavaScript
    data_for_js = []
    for i in range(len(essays_df)):
        # Get top 5 PCs for this sample
        total_contrib = np.abs(contributions_ai[i]) + np.abs(contributions_sc[i])
        top_5_indices = np.argsort(total_contrib)[-5:][::-1]
        
        pc_info = []
        for pc_idx in top_5_indices:
            pc_info.append({
                'pc': f'PC{pc_idx}',
                'percentile': float(pc_percentiles[i, pc_idx]),
                'contribution_ai': float(contributions_ai[i, pc_idx]),
                'contribution_sc': float(contributions_sc[i, pc_idx]),
                'variance_total': float(variance_explained[pc_idx] * 100)
            })
        
        data_for_js.append({
            'x': float(X_umap_3d[i, 0]),
            'y': float(X_umap_3d[i, 1]),
            'z': float(X_umap_3d[i, 2]) if X_umap_3d.shape[1] > 2 else 0,
            'essay_id': str(essays_df.iloc[i]['essay_id']),
            'essay': str(essays_df.iloc[i]['essay']),
            'ai_rating': float(essays_df.iloc[i]['ai_rating']),
            'social_class': int(essays_df.iloc[i]['sc11']),
            'hdbscan_topic_id': int(clusters[i]),
            'pc_info': pc_info
        })
    
    # Prepare topic data
    topic_data = []
    unique_topics = np.unique(clusters)
    
    for topic_id in unique_topics:
        if topic_id == -1:
            continue
        
        topic_points = X_umap_3d[clusters == topic_id]
        if len(topic_points) > 0:
            centroid = topic_points.mean(axis=0)
            
            # Get keywords
            if topic_id in topic_keywords:
                if isinstance(topic_keywords[topic_id], list):
                    keywords = ', '.join([w[0] for w in topic_keywords[topic_id][:5]])
                else:
                    keywords = str(topic_keywords[topic_id])
            else:
                keywords = f"Topic {topic_id}"
            
            topic_data.append({
                'topic_id': int(topic_id),
                'keywords': keywords,
                'centroid': [float(x) for x in centroid],
                'size': int(np.sum(clusters == topic_id))
            })
    
    # Topic statistics
    topic_stats = []
    ai_bottom10 = ai_percentiles[10]
    ai_top10 = ai_percentiles[90]
    
    for topic_id in unique_topics:
        if topic_id == -1:
            continue
        
        topic_mask = clusters == topic_id
        topic_essays = essays_df[topic_mask]
        
        if len(topic_essays) > 0:
            ai_ratings = topic_essays['ai_rating'].values
            sc_values = topic_essays['sc11'].values
            
            prob_ai_top10 = np.mean(ai_ratings >= ai_top10)
            prob_ai_bottom10 = np.mean(ai_ratings <= ai_bottom10)
            prob_sc_high = np.mean(sc_values >= 9)  # SC 9-11
            prob_sc_low = np.mean(sc_values <= 3)   # SC 1-3
            
            max_impact = max(prob_ai_top10, prob_ai_bottom10, prob_sc_high, prob_sc_low)
            
            # Get keywords
            if topic_id in topic_keywords:
                if isinstance(topic_keywords[topic_id], list):
                    keywords = ', '.join([w[0] for w in topic_keywords[topic_id][:5]])
                else:
                    keywords = str(topic_keywords[topic_id])
            else:
                keywords = f"Topic {topic_id}"
            
            topic_stats.append({
                'topic_id': int(topic_id),
                'keywords': keywords,
                'size': len(topic_essays),
                'prob_ai_top10': prob_ai_top10,
                'prob_ai_bottom10': prob_ai_bottom10,
                'prob_sc_high': prob_sc_high,
                'prob_sc_low': prob_sc_low,
                'max_impact_prob': max_impact
            })
    
    topic_stats.sort(key=lambda x: x['max_impact_prob'], reverse=True)
    
    # DML results
    dml_results = results['dml_results']
    
    # Load the v21 HTML template
    v21_template_path = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_dml_pc_analysis/minimal_umap_viz_v21.html")
    
    with open(v21_template_path, 'r') as f:
        html_content = f.read()
    
    # Replace the data sections
    # Essays data
    essays_start = html_content.find("const essays = [")
    if essays_start != -1:
        essays_end = html_content.find("];", essays_start) + 2
        old_essays_section = html_content[essays_start:essays_end]
        new_essays_section = f"const essays = {json.dumps(data_for_js, indent=2)};"
        html_content = html_content.replace(old_essays_section, new_essays_section)
    
    # Topics data
    topics_start = html_content.find("const topics = [")
    if topics_start != -1:
        topics_end = html_content.find("];", topics_start) + 2
        old_topics_section = html_content[topics_start:topics_end]
        new_topics_section = f"const topics = {json.dumps(topic_data, indent=2)};"
        html_content = html_content.replace(old_topics_section, new_topics_section)
    
    # Topic stats
    stats_start = html_content.find("const topicStats = [")
    if stats_start != -1:
        stats_end = html_content.find("];", stats_start) + 2
        old_stats_section = html_content[stats_start:stats_end]
        new_stats_section = f"const topicStats = {json.dumps(topic_stats, indent=2)};"
        html_content = html_content.replace(old_stats_section, new_stats_section)
    
    # Center point
    center_start = html_content.find("const centerPoint = [")
    if center_start != -1:
        center_end = html_content.find("];", center_start) + 2
        old_center = html_content[center_start:center_end]
        new_center = f"const centerPoint = [{center_x:.6f}, {center_y:.6f}, {center_z:.6f}];"
        html_content = html_content.replace(old_center, new_center)
    
    # Get X/Y names from config
    X_name = config['variables']['X']['name']
    X_short = config['variables']['X']['short_name']
    Y_name = config['variables']['Y']['name']
    Y_short = config['variables']['Y']['short_name']
    
    # Update threshold values
    html_content = html_content.replace('value="3.41"', f'value="{ai_percentiles[10]:.2f}"')  # Y low
    html_content = html_content.replace('value="7.17"', f'value="{ai_percentiles[90]:.2f}"')  # Y high
    
    # Update Y range in JS
    y_min = ai_ratings_clean.min()
    y_max = ai_ratings_clean.max()
    html_content = html_content.replace('const val = (essays[i].ai_rating - 1) / (10 - 1)', 
                                      f'const val = (essays[i].ai_rating - {y_min}) / ({y_max} - {y_min})')
    
    # Simple label replacements for UI elements only
    ui_replacements = [
        # Title
        ("AI Rating vs Social Class Analysis", config['display']['title']),
        
        # Panel labels
        ("AI Rating Thresholds:", f"{Y_name} Thresholds:"),
        ("Social Class Thresholds:", f"{X_name} Thresholds:"),
        
        # Legend items
        ("High AI + High SC", f"High {Y_short} + High {X_short}"),
        ("High AI + Low SC", f"High {Y_short} + Low {X_short}"),
        ("Low AI + High SC", f"Low {Y_short} + High {X_short}"),
        ("Low AI + Low SC", f"Low {Y_short} + Low {X_short}"),
        
        # Essay viewer width adjustments
        ("left: 60px;", "left: 350px;"),
        ("right: 60px;", "right: 250px;"),
        
        # Z-index fixes for panels
        ("z-index: 1000;", "z-index: 99999;"),
        ("z-index: 1002;", "z-index: 99999;"),
        ("z-index: 1001;", "z-index: 99999;"),
    ]
    
    for old_text, new_text in ui_replacements:
        html_content = html_content.replace(old_text, new_text)
    
    # Update ALL DML values dynamically using more specific patterns
    
    # Find the DML table section
    dml_table_start = html_content.find('<table id="dml-table"')
    dml_table_end = html_content.find('</table>', dml_table_start) + 8
    
    if dml_table_start != -1:
        # Update theta values in the SC->AI column
        html_content = re.sub(
            r'<td>-?\d+\.\d+</td>(\s*<!--\s*theta_naive_sc_to_ai\s*-->)',
            f'<td>{dml_results["theta_naive"]:.3f}</td>\\1',
            html_content
        )
        html_content = re.sub(
            r'<td>-?\d+\.\d+</td>(\s*<!--\s*theta_200_sc_to_ai\s*-->)',
            f'<td>{dml_results["theta_200"]:.3f}</td>\\1',
            html_content
        )
        html_content = re.sub(
            r'<td>-?\d+\.\d+</td>(\s*<!--\s*theta_top5_sc_to_ai\s*-->)',
            f'<td>{dml_results["theta_top5"]:.3f}</td>\\1',
            html_content
        )
        
        # Update p-values
        html_content = re.sub(
            r'<td>[\d.]+</td>(\s*<!--\s*pval_naive_sc_to_ai\s*-->)',
            f'<td>{dml_results["pval_naive"]:.4f}</td>\\1',
            html_content
        )
        html_content = re.sub(
            r'<td>[\d.]+</td>(\s*<!--\s*pval_200_sc_to_ai\s*-->)',
            f'<td>{dml_results["pval_200"]:.4f}</td>\\1',
            html_content
        )
        html_content = re.sub(
            r'<td>[\d.]+</td>(\s*<!--\s*pval_top5_sc_to_ai\s*-->)',
            f'<td>{dml_results["pval_top5"]:.4f}</td>\\1',
            html_content
        )
        
        # Update R² values
        # AI R² (Top 5)
        html_content = re.sub(
            r'AI R² \(Top 5\):</td>\s*<td>[\d.-]+</td>\s*<td>[\d.-]+</td>',
            f'AI R² (Top 5):</td>\\n                <td>{dml_results.get("top5_r2_y", 0):.3f}</td>\\n                <td>{dml_results.get("top5_r2_y", 0):.3f}</td>',
            html_content
        )
        
        # SC R² (Top 5)
        html_content = re.sub(
            r'SC R² \(Top 5\):</td>\s*<td>[\d.-]+</td>\s*<td>[\d.-]+</td>',
            f'SC R² (Top 5):</td>\\n                <td>{dml_results.get("top5_r2_x", 0):.3f}</td>\\n                <td>{dml_results.get("top5_r2_x", 0):.3f}</td>',
            html_content
        )
        
        # Naive R²
        html_content = re.sub(
            r'R² \(variance explained\):</td>\s*<td colspan="2" style="text-align: center;">[\d.]+</td>',
            f'R² (variance explained):</td>\\n                <td colspan="2" style="text-align: center;">{dml_results.get("r2_naive", 0):.3f}</td>',
            html_content
        )
        
        # Update PC list
        pc_list = ', '.join([f'PC{pc}' for pc in dml_results['top_pcs']])
        html_content = re.sub(
            r'PCs Used \(Top 5\):</td>\s*<td colspan="2"[^>]*>[^<]+</td>',
            f'PCs Used (Top 5):</td>\\n                <td colspan="2" style="text-align: center;">{pc_list}</td>',
            html_content
        )
    
    # Save the modified HTML
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Visualization saved to {output_path}")
    
    return output_path