#!/usr/bin/env python3
"""
Generate PerceptionHD visualization using v21 template
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def generate_visualization_html(results, output_path):
    """
    Generate visualization using v21 template with exact styling
    """
    
    # Extract data from pipeline results
    essays_df = results['df'].copy()
    X_umap_3d = results.get('umap_coords', results.get('X_umap'))
    X_pca = results.get('pca_features', results.get('X_pca'))
    clusters = results.get('clusters', results.get('cluster_labels'))
    config = results['config']
    
    # Map to expected column names
    essays_df['essay_id'] = essays_df['id']
    essays_df['essay'] = essays_df['text']
    essays_df['ai_rating'] = essays_df['Y']
    essays_df['sc11'] = essays_df['X']
    
    # Get variable names from config
    X_name = config['variables']['X']['name']
    X_short = config['variables']['X']['short_name']
    Y_name = config['variables']['Y']['name']
    Y_short = config['variables']['Y']['short_name']
    
    # Get data ranges
    x_min = int(config['variables']['X']['min_value'])
    x_max = int(config['variables']['X']['max_value'])
    y_min = essays_df['Y'].min()
    y_max = essays_df['Y'].max()
    
    # Get top PCs and contributions
    top_pcs = results['dml_results']['top_pcs']
    contributions_y = results['contributions_y']
    contributions_x = results['contributions_x']
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
    
    # Calculate percentile thresholds
    ai_p10 = ai_ratings_clean.quantile(0.10)
    ai_p90 = ai_ratings_clean.quantile(0.90)
    sc_p10 = essays_df['sc11'].quantile(0.10)
    sc_p90 = essays_df['sc11'].quantile(0.90)
    
    # Calculate center of point cloud
    center_x = X_umap_3d[:, 0].mean()
    center_y = X_umap_3d[:, 1].mean()
    center_z = X_umap_3d[:, 2].mean() if X_umap_3d.shape[1] > 2 else 0
    
    # Prepare data for JavaScript
    viz_data = []
    for i in range(len(essays_df)):
        # Get top 5 PCs for this sample
        total_contrib = np.abs(contributions_y[i]) + np.abs(contributions_x[i])
        top_5_indices = np.argsort(total_contrib)[-5:][::-1]
        
        pc_info = []
        for pc_idx in top_5_indices:
            pc_info.append({
                'pc': f'PC{pc_idx}',
                'percentile': float(pc_percentiles[i, pc_idx]),
                'contribution_ai': float(contributions_y[i, pc_idx]),
                'contribution_sc': float(contributions_x[i, pc_idx]),
                'variance_total': float(variance_explained[pc_idx] * 100) if pc_idx < len(variance_explained) else 0
            })
        
        viz_data.append({
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
    topic_viz_data = []
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
            
            topic_viz_data.append({
                'topic_id': int(topic_id),
                'keywords': keywords,
                'centroid': [float(x) for x in centroid],
                'size': int(np.sum(clusters == topic_id))
            })
    
    # Topic statistics
    topic_stats_data = []
    
    for topic_id in unique_topics:
        if topic_id == -1:
            continue
        
        topic_mask = clusters == topic_id
        topic_essays = essays_df[topic_mask]
        
        if len(topic_essays) > 0:
            ai_ratings = topic_essays['ai_rating'].values
            sc_values = topic_essays['sc11'].values
            
            prob_ai_top10 = np.mean(ai_ratings >= ai_p90)
            prob_ai_bottom10 = np.mean(ai_ratings <= ai_p10)
            prob_sc_high = np.mean(sc_values >= sc_p90)
            prob_sc_low = np.mean(sc_values <= sc_p10)
            
            max_impact = max(prob_ai_top10, prob_ai_bottom10, prob_sc_high, prob_sc_low)
            
            # Get keywords
            if topic_id in topic_keywords:
                if isinstance(topic_keywords[topic_id], list):
                    keywords = ', '.join([w[0] for w in topic_keywords[topic_id][:5]])
                else:
                    keywords = str(topic_keywords[topic_id])
            else:
                keywords = f"Topic {topic_id}"
            
            topic_stats_data.append({
                'topic_id': int(topic_id),
                'keywords': keywords,
                'size': len(topic_essays),
                'prob_ai_top10': prob_ai_top10,
                'prob_ai_bottom10': prob_ai_bottom10,
                'prob_sc_high': prob_sc_high,
                'prob_sc_low': prob_sc_low,
                'max_impact_prob': max_impact
            })
    
    topic_stats_data.sort(key=lambda x: x['max_impact_prob'], reverse=True)
    
    # DML results
    dml_results = results['dml_results']
    
    # Load template
    template_path = Path(__file__).parent / 'templates' / 'visualization_template_v21.html'
    with open(template_path, 'r') as f:
        html_content = f.read()
    
    # Prepare all replacements
    replacements = {
        # Data arrays
        '{{VIZ_DATA_JSON}}': json.dumps(viz_data, indent=2),
        '{{TOPIC_VIZ_JSON}}': json.dumps(topic_viz_data, indent=2),
        '{{TOPIC_STATS_JSON}}': json.dumps(topic_stats_data, indent=2),
        
        # Center point
        '{{CENTER_X}}': f'{center_x:.6f}',
        '{{CENTER_Y}}': f'{center_y:.6f}',
        '{{CENTER_Z}}': f'{center_z:.6f}',
        
        # Basic stats
        '{{N_ESSAYS}}': str(len(essays_df)),
        '{{AI_RANGE_MIN}}': f'{y_min:.2f}',
        '{{AI_RANGE_MAX}}': f'{y_max:.2f}',
        '{{SC_RANGE_MIN}}': str(x_min),
        '{{SC_RANGE_MAX}}': str(x_max),
        '{{N_TOPICS}}': str(len(topic_viz_data)),
        
        # Thresholds
        '{{AI_THRESHOLD_LOW}}': f'{ai_p10:.2f}',
        '{{AI_THRESHOLD_HIGH}}': f'{ai_p90:.2f}',
        '{{SC_THRESHOLD_LOW}}': str(int(sc_p10)),
        '{{SC_THRESHOLD_HIGH}}': str(int(sc_p90)),
        
        # Selected PCs
        '{{SELECTED_PCS}}': ', '.join([f'PC{pc}' for pc in dml_results['top_pcs']]),
        
        # DML results - Naive model
        '{{NAIVE_THETA}}': f"{dml_results.get('theta_naive', 0):.3f}",
        '{{NAIVE_SE}}': f"{dml_results.get('se_naive', 0):.3f}",
        '{{NAIVE_CI_LOWER}}': f"{dml_results.get('theta_naive', 0) - 1.96 * dml_results.get('se_naive', 0):.3f}",
        '{{NAIVE_CI_UPPER}}': f"{dml_results.get('theta_naive', 0) + 1.96 * dml_results.get('se_naive', 0):.3f}",
        '{{NAIVE_PVAL}}': f"{dml_results.get('pval_naive', 0):.4f}",
        '{{NAIVE_R2}}': f"{dml_results.get('r2_naive', 0):.3f}",
        
        # DML results - 200 PC model
        '{{THETA_200}}': f"{dml_results.get('theta_200', 0):.3f}",
        '{{SE_200}}': f"{dml_results.get('se_200', 0):.3f}",
        '{{CI_200_LOWER}}': f"{dml_results.get('theta_200', 0) - 1.96 * dml_results.get('se_200', 0):.3f}",
        '{{CI_200_UPPER}}': f"{dml_results.get('theta_200', 0) + 1.96 * dml_results.get('se_200', 0):.3f}",
        '{{PVAL_200}}': f"{dml_results.get('pval_200', 0):.4f}",
        '{{REDUCTION_200}}': f"{(1 - abs(dml_results.get('theta_200', 0) / max(dml_results.get('theta_naive', 1), 0.001))) * 100:.1f}",
        '{{ALL_R2_Y}}': f"{dml_results.get('all_r2_y', 0):.3f}",
        '{{ALL_R2_Y_CV}}': f"{dml_results.get('all_r2_y_cv', 0):.3f}",
        '{{ALL_R2_X}}': f"{dml_results.get('all_r2_x', 0):.3f}",
        '{{ALL_R2_X_CV}}': f"{dml_results.get('all_r2_x_cv', 0):.3f}",
        
        # DML results - Top 5 PC model
        '{{THETA_TOP5}}': f"{dml_results.get('theta_top5', 0):.3f}",
        '{{SE_TOP5}}': f"{dml_results.get('se_top5', 0):.3f}",
        '{{CI_TOP5_LOWER}}': f"{dml_results.get('theta_top5', 0) - 1.96 * dml_results.get('se_top5', 0):.3f}",
        '{{CI_TOP5_UPPER}}': f"{dml_results.get('theta_top5', 0) + 1.96 * dml_results.get('se_top5', 0):.3f}",
        '{{PVAL_TOP5}}': f"{dml_results.get('pval_top5', 0):.4f}",
        '{{REDUCTION_TOP5}}': f"{(1 - abs(dml_results.get('theta_top5', 0) / max(dml_results.get('theta_naive', 1), 0.001))) * 100:.1f}",
        '{{TOP5_R2_Y}}': f"{dml_results.get('top5_r2_y', 0):.3f}",
        '{{TOP5_R2_Y_CV}}': f"{dml_results.get('top5_r2_y_cv', 0):.3f}",
        '{{TOP5_R2_X}}': f"{dml_results.get('top5_r2_x', 0):.3f}",
        '{{TOP5_R2_X_CV}}': f"{dml_results.get('top5_r2_x_cv', 0):.3f}",
    }
    
    # Add full embeddings model if available
    if 'theta_full' in dml_results:
        n_embeddings = dml_results.get('n_embeddings', 4096)
        
        # Find where to insert the full embeddings section
        # It should go after the naive model and before the 200 PC model
        full_embeddings_html = f"""
            <tr>
                <td colspan="3" style="padding-top: 15px; font-weight: bold; color: #4CAF50;">Full Embeddings ({n_embeddings} dims)</td>
            </tr>
            <tr>
                <td>DML θ ({{{{X_SHORT}}}}→{{{{Y_SHORT}}}}):</td>
                <td>{dml_results.get('theta_full', 0):.3f}</td>
                <td>{dml_results.get('theta_full', 0):.3f}</td>
            </tr>
            <tr>
                <td>Standard Error:</td>
                <td>{dml_results.get('se_full', 0):.3f}</td>
                <td>{dml_results.get('se_full', 0):.3f}</td>
            </tr>
            <tr>
                <td>95% CI:</td>
                <td>({dml_results.get('theta_full', 0) - 1.96 * dml_results.get('se_full', 0):.3f}, {dml_results.get('theta_full', 0) + 1.96 * dml_results.get('se_full', 0):.3f})</td>
                <td>({dml_results.get('theta_full', 0) - 1.96 * dml_results.get('se_full', 0):.3f}, {dml_results.get('theta_full', 0) + 1.96 * dml_results.get('se_full', 0):.3f})</td>
            </tr>
            <tr>
                <td>p-value:</td>
                <td>{dml_results.get('pval_full', 0):.4f}</td>
                <td>{dml_results.get('pval_full', 0):.4f}</td>
            </tr>
            <tr>
                <td style="color: #ff9800;">Effect Reduction vs Naive:</td>
                <td style="color: #ff9800;">{(1 - abs(dml_results.get('theta_full', 0) / max(dml_results.get('theta_naive', 1), 0.001))) * 100:.1f}%</td>
                <td style="color: #ff9800;">{(1 - abs(dml_results.get('theta_full', 0) / max(dml_results.get('theta_naive', 1), 0.001))) * 100:.1f}%</td>
            </tr>
            <tr>
                <td>{{{{Y_SHORT}}}} R² (Full):</td>
                <td>{dml_results.get('full_r2_y', 0):.3f}</td>
                <td>{dml_results.get('full_r2_y_cv', 0):.3f}</td>
            </tr>
            <tr>
                <td>{{{{X_SHORT}}}} R² (Full):</td>
                <td>{dml_results.get('full_r2_x', 0):.3f}</td>
                <td>{dml_results.get('full_r2_x_cv', 0):.3f}</td>
            </tr>"""
        
        # Insert after naive model section
        insert_marker = '<tr>\n                <td colspan="3" style="padding-top: 15px; font-weight: bold; color: #2196F3;">All 200 PCs Model</td>\n            </tr>'
        html_content = html_content.replace(insert_marker, full_embeddings_html + '\n            ' + insert_marker)
    
    # Apply all replacements
    for placeholder, value in replacements.items():
        html_content = html_content.replace(placeholder, value)
    
    # Now replace all the AI/SC labels with X/Y
    label_replacements = [
        # Title
        ("AI Rating vs Social Class Analysis", config['display']['title']),
        
        # Variable names - all forms
        ("AI Rating", Y_name),
        ("Social Class", X_name),
        
        # Short forms
        ("{{AI_SHORT}}", Y_short),
        ("{{SC_SHORT}}", X_short),
        ("{{X_SHORT}}", X_short),
        ("{{Y_SHORT}}", Y_short),
        
        # In JavaScript and CSS
        ("ai_rating", "y_value"),
        ("social_class", "x_value"),
        ("sc11", "x_value"),
        
        # Button text
        (">AI<", f">{Y_short}<"),
        (">SC<", f">{X_short}<"),
        ("AI/SC", f"{X_short}/{Y_short}"),
        
        # Labels in text
        ("AI:", f"{Y_short}:"),
        ("SC:", f"{X_short}:"),
        ("High AI", f"High {Y_short}"),
        ("Low AI", f"Low {Y_short}"),
        ("High SC", f"High {X_short}"),
        ("Low SC", f"Low {X_short}"),
        
        # Contribution labels
        ("AI Contrib", f"{Y_short} Contrib"),
        ("SC Contrib", f"{X_short} Contrib"),
        ("contribution_ai", "contribution_y"),
        ("contribution_sc", "contribution_x"),
        
        # Topic stats
        ("% High AI", f"% High {Y_short}"),
        ("% Low AI", f"% Low {Y_short}"),
        ("% High SC", f"% High {X_short}"),
        ("% Low SC", f"% Low {X_short}"),
        
        # R² labels
        ("AI R²", f"{Y_short} R²"),
        ("SC R²", f"{X_short} R²"),
        
        # Essay viewer adjustments
        ("left: 60px;", "left: 350px;"),
        ("right: 60px;", "right: 250px;"),
        
        # Z-index (make panels on top)
        ("z-index: 1000;", "z-index: 99999;"),
    ]
    
    # Apply label replacements
    for old_text, new_text in label_replacements:
        html_content = html_content.replace(old_text, new_text)
    
    # Save the HTML
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Visualization generated using v21 template and saved to {output_path}")
    
    return output_path