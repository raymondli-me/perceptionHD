#!/usr/bin/env python3
"""
Generate PerceptionHD visualization HTML from scratch
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_visualization_html(results, output_path):
    """
    Generate complete visualization HTML from scratch
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
    ai_percentiles = {
        10: ai_ratings_clean.quantile(0.10),
        25: ai_ratings_clean.quantile(0.25), 
        75: ai_ratings_clean.quantile(0.75),
        90: ai_ratings_clean.quantile(0.90)
    }
    
    # Calculate center of point cloud
    center_x = X_umap_3d[:, 0].mean()
    center_y = X_umap_3d[:, 1].mean()
    center_z = X_umap_3d[:, 2].mean() if X_umap_3d.shape[1] > 2 else 0
    
    # Prepare data for JavaScript
    data_for_js = []
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
            
            # Calculate percentiles for X variable
            x_percentiles = {
                90: essays_df['sc11'].quantile(0.90),
                10: essays_df['sc11'].quantile(0.10)
            }
            prob_sc_high = np.mean(sc_values >= x_percentiles[90])
            prob_sc_low = np.mean(sc_values <= x_percentiles[10])
            
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
    
    # PC list
    pc_list = ', '.join([f'PC{pc}' for pc in dml_results['top_pcs']])
    
    # Generate the complete HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{config['display']['title']}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            background: #000;
            cursor: none;
            color: #fff;
        }}
        
        /* Text shadow for better readability */
        * {{
            text-shadow: 
                -1px -1px 0 #000,
                 1px -1px 0 #000,
                -1px  1px 0 #000,
                 1px  1px 0 #000;
        }}
        
        #cursor-indicator {{
            position: absolute;
            width: 30px;
            height: 30px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            pointer-events: none;
            z-index: 99999;
            transform: translate(-50%, -50%);
        }}
        
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
            max-width: 400px;
            z-index: 100;
            cursor: move;
            transition: transform 0.3s ease;
            overflow: visible;
        }}
        
        #info.collapsed {{
            transform: translateX(calc(-100% + 40px));
        }}
        
        #info .collapse-btn {{
            position: absolute;
            right: -30px;
            top: 50%;
            transform: translateY(-50%);
            width: 25px;
            height: 60px;
            background: rgba(0,0,0,0.8);
            border: 1px solid rgba(255,255,255,0.2);
            border-left: none;
            border-radius: 0 5px 5px 0;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-size: 16px;
            transition: all 0.2s;
        }}
        
        #info .collapse-btn:hover {{
            background: rgba(255,255,255,0.1);
            color: #fff;
        }}
        
        #controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
            z-index: 100;
            transition: transform 0.3s ease, opacity 0.3s ease;
            overflow: visible;
        }}
        
        #controls.collapsed {{
            transform: translateX(calc(100% - 40px));
        }}
        
        #controls .collapse-btn {{
            position: absolute;
            left: -30px;
            top: 50%;
            transform: translateY(-50%);
            width: 25px;
            height: 60px;
            background: rgba(0,0,0,0.8);
            border: 1px solid rgba(255,255,255,0.2);
            border-right: none;
            border-radius: 5px 0 0 5px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-size: 16px;
            transition: all 0.2s;
        }}
        
        #controls.collapsed .collapse-btn {{
            border-radius: 5px 0 0 5px;
        }}
        
        #controls .collapse-btn:hover {{
            background: rgba(255,255,255,0.1);
            color: #fff;
        }}
        
        /* Essay viewer styles */
        #essay-viewer {{
            position: fixed;
            top: 0;
            left: 350px;
            right: 250px;
            bottom: 0;
            pointer-events: none;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 99999;
        }}
        
        #essay-content {{
            background: rgba(0,0,0,0.95);
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 10px;
            padding: 30px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            display: none;
            pointer-events: auto;
            z-index: 99999;
        }}
        
        /* Filter panel styles */
        .filter-panel {{
            position: absolute;
            top: 60px;
            left: 10px;
            background: rgba(0,0,0,0.9);
            padding: 15px;
            border-radius: 5px;
            border: 1px solid rgba(255,255,255,0.2);
            display: none;
            z-index: 99999;
            width: 250px;
        }}
        
        .filter-group {{
            margin-bottom: 15px;
        }}
        
        .filter-group label {{
            display: block;
            margin-bottom: 5px;
            color: #ccc;
        }}
        
        .filter-group input[type="range"] {{
            width: 100%;
            margin-bottom: 5px;
        }}
        
        .filter-values {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #999;
        }}
        
        /* Topic stats panel */
        #topic-stats {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(0,0,0,0.9);
            padding: 15px;
            border-radius: 5px;
            border: 1px solid rgba(255,255,255,0.2);
            display: none;
            z-index: 99999;
            max-width: 400px;
            max-height: 300px;
            overflow-y: auto;
        }}
        
        .topic-row {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            padding: 5px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .topic-label {{
            flex: 1;
            font-size: 12px;
        }}
        
        .topic-bar {{
            width: 100px;
            height: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
            margin-left: 10px;
        }}
        
        .topic-fill {{
            height: 100%;
            background: #4CAF50;
            transition: width 0.3s;
        }}
        
        /* DML stats modal */
        #dml-modal {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.95);
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 10px;
            padding: 30px;
            display: none;
            z-index: 99999;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        
        #dml-modal h3 {{
            margin-top: 0;
            margin-bottom: 20px;
            color: #4CAF50;
        }}
        
        #dml-modal table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        
        #dml-modal td {{
            padding: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        #dml-modal td:first-child {{
            font-weight: bold;
            color: #ccc;
        }}
        
        .close-btn {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: #fff;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
        }}
        
        .close-btn:hover {{
            background: rgba(255,255,255,0.2);
        }}
        
        /* Button styles */
        button {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: #fff;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            margin: 2px;
            font-size: 12px;
        }}
        
        button:hover {{
            background: rgba(255,255,255,0.2);
        }}
        
        button.active {{
            background: #4CAF50;
            border-color: #4CAF50;
        }}
        
        /* Gallery mode */
        #gallery-container {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.95);
            display: none;
            z-index: 99999;
            overflow-y: auto;
            padding: 20px;
        }}
        
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .essay-card {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 5px;
            padding: 15px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .essay-card:hover {{
            background: rgba(255,255,255,0.1);
            transform: translateY(-2px);
        }}
        
        .essay-card h4 {{
            margin: 0 0 10px 0;
            color: #4CAF50;
        }}
        
        .essay-card .essay-text {{
            font-size: 12px;
            line-height: 1.4;
            max-height: 100px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .essay-card .essay-meta {{
            margin-top: 10px;
            font-size: 11px;
            color: #999;
        }}
        
        /* Keyboard help */
        #keyboard-help {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 10px;
            border-radius: 5px;
            font-size: 11px;
            color: #999;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        /* Loading indicator */
        #loading {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #4CAF50;
            font-size: 24px;
            z-index: 99999;
        }}
    </style>
</head>
<body>
    <div id="cursor-indicator"></div>
    <div id="loading">Loading visualization...</div>
    
    <div id="info">
        <div class="collapse-btn">◀</div>
        <div style="font-weight: bold; margin-bottom: 10px; color: #4CAF50;">{config['display']['title']}</div>
        <div id="stats">
            Total Essays: {len(essays_df)}<br>
            {Y_short} Range: {y_min:.2f} - {y_max:.2f}<br>
            {X_short} Range: {x_min} - {x_max}<br>
            Topics: {len(topic_data)}
        </div>
        <div style="margin-top: 10px;">
            <button onclick="toggleColorMode()">{X_short}/{Y_short}</button>
            <button onclick="toggleColorMode()">Topics</button>
            <button onclick="toggleFilters()">{Y_short} Filter</button>
            <button onclick="toggleTopicStats()">Topic Stats</button>
            <button onclick="showDMLStats()">Show DML Stats</button>
        </div>
        <div style="margin-top: 10px;">
            <button onclick="toggleGalleryMode()">Gallery Mode</button>
            <button onclick="resetView()">Reset View</button>
        </div>
    </div>
    
    <div id="controls">
        <div class="collapse-btn">▶</div>
        <div style="font-weight: bold; margin-bottom: 10px;">Selected Essay</div>
        <div id="selected-info" style="color: #999;">Hover over points to see details</div>
    </div>
    
    <div id="essay-viewer">
        <div id="essay-content">
            <button class="close-btn" onclick="closeEssayViewer()">×</button>
            <h3 id="essay-title"></h3>
            <div id="essay-text" style="margin-bottom: 20px;"></div>
            <div id="essay-stats" style="font-size: 12px; color: #999;"></div>
        </div>
    </div>
    
    <div class="filter-panel" id="ai-filter-panel">
        <div style="font-weight: bold; margin-bottom: 10px;">{Y_name} Filter</div>
        <div class="filter-group">
            <label>{Y_short} Range:</label>
            <input type="range" id="ai-min" min="{y_min}" max="{y_max}" step="0.1" value="{y_min}">
            <input type="range" id="ai-max" min="{y_min}" max="{y_max}" step="0.1" value="{y_max}">
            <div class="filter-values">
                <span id="ai-min-val">{y_min:.1f}</span>
                <span id="ai-max-val">{y_max:.1f}</span>
            </div>
        </div>
        <div class="filter-group">
            <label>Thresholds:</label>
            <button onclick="setAIThreshold({ai_bottom10:.2f}, {ai_top10:.2f})">10th/90th percentile</button>
            <button onclick="setAIThreshold({ai_percentiles[25]:.2f}, {ai_percentiles[75]:.2f})">25th/75th percentile</button>
        </div>
        <button onclick="applyFilters()">Apply</button>
        <button onclick="resetFilters()">Reset</button>
    </div>
    
    <div id="topic-stats">
        <div style="font-weight: bold; margin-bottom: 10px;">Topic Statistics</div>
        <div style="font-size: 11px; color: #999; margin-bottom: 10px;">
            {Y_short} thresholds: Top 10% ≥ {ai_top10:.2f}, Bottom 10% ≤ {ai_bottom10:.2f}<br>
            {X_short} thresholds: Top 10% = {int(essays_df['X'].quantile(0.90))}, Bottom 10% = {int(essays_df['X'].quantile(0.10))}
        </div>
        <div id="topic-stats-content"></div>
    </div>
    
    <div id="dml-modal">
        <button class="close-btn" onclick="closeDMLModal()">×</button>
        <h3>Double Machine Learning Results</h3>
        <div style="margin-bottom: 20px;">
            <strong>Selected Top PCs:</strong> {pc_list}
        </div>
        
        <table>
            <tr>
                <td>Model</td>
                <td>Non-Crossfitted</td>
                <td>Crossfitted (5-fold)</td>
            </tr>
            <tr>
                <td colspan="3" style="padding-top: 15px; font-weight: bold; color: #e91e63;">Naive Model (No Text)</td>
            </tr>
            <tr>
                <td>θ ({X_short}→{Y_short}):</td>
                <td colspan="2" style="text-align: center;">{dml_results.get('theta_naive', 0):.3f}</td>
            </tr>
            <tr>
                <td>Standard Error:</td>
                <td colspan="2" style="text-align: center;">{dml_results.get('se_naive', 0):.3f}</td>
            </tr>
            <tr>
                <td>95% CI:</td>
                <td colspan="2" style="text-align: center;">({dml_results.get('theta_naive', 0) - 1.96 * dml_results.get('se_naive', 0):.3f}, {dml_results.get('theta_naive', 0) + 1.96 * dml_results.get('se_naive', 0):.3f})</td>
            </tr>
            <tr>
                <td>p-value:</td>
                <td colspan="2" style="text-align: center;">{dml_results.get('pval_naive', 0):.4f}</td>
            </tr>
            <tr>
                <td>R² (variance explained):</td>
                <td colspan="2" style="text-align: center;">{dml_results.get('r2_naive', 0):.3f}</td>
            </tr>"""
    
    # Add full embeddings section if available
    if 'theta_full' in dml_results:
        n_embeddings = dml_results.get('n_embeddings', 4096)
        theta_full = dml_results.get('theta_full', 0)
        se_full = dml_results.get('se_full', 0.01)
        pval_full = dml_results.get('pval_full', 0.01)
        ci_full_lower = theta_full - 1.96 * se_full
        ci_full_upper = theta_full + 1.96 * se_full
        
        reduction_full = 0
        if 'theta_naive' in dml_results and dml_results['theta_naive'] != 0:
            reduction_full = (1 - abs(theta_full / dml_results['theta_naive'])) * 100
        
        html_content += f"""
            <tr>
                <td colspan="3" style="padding-top: 15px; font-weight: bold; color: #4CAF50;">Full Embeddings ({n_embeddings} dims)</td>
            </tr>
            <tr>
                <td>DML θ ({X_short}→{Y_short}):</td>
                <td>{theta_full:.3f}</td>
                <td>{theta_full:.3f}</td>
            </tr>
            <tr>
                <td>Standard Error:</td>
                <td>{se_full:.3f}</td>
                <td>{se_full:.3f}</td>
            </tr>
            <tr>
                <td>95% CI:</td>
                <td>({ci_full_lower:.3f}, {ci_full_upper:.3f})</td>
                <td>({ci_full_lower:.3f}, {ci_full_upper:.3f})</td>
            </tr>
            <tr>
                <td>p-value:</td>
                <td>{pval_full:.4f}</td>
                <td>{pval_full:.4f}</td>
            </tr>
            <tr>
                <td style="color: #ff9800;">Effect Reduction vs Naive:</td>
                <td style="color: #ff9800;">{reduction_full:.1f}%</td>
                <td style="color: #ff9800;">{reduction_full:.1f}%</td>
            </tr>
            <tr>
                <td>{Y_short} R² (Full):</td>
                <td>{dml_results.get("full_r2_y", 0):.3f}</td>
                <td>{dml_results.get("full_r2_y_cv", 0):.3f}</td>
            </tr>
            <tr>
                <td>{X_short} R² (Full):</td>
                <td>{dml_results.get("full_r2_x", 0):.3f}</td>
                <td>{dml_results.get("full_r2_x_cv", 0):.3f}</td>
            </tr>"""
    
    # Continue with 200 PCs and Top 5 PCs
    reduction_200 = 0
    if 'theta_naive' in dml_results and dml_results['theta_naive'] != 0:
        reduction_200 = (1 - abs(dml_results.get('theta_200', 0) / dml_results['theta_naive'])) * 100
    
    reduction_top5 = 0
    if 'theta_naive' in dml_results and dml_results['theta_naive'] != 0:
        reduction_top5 = (1 - abs(dml_results.get('theta_top5', 0) / dml_results['theta_naive'])) * 100
    
    html_content += f"""
            <tr>
                <td colspan="3" style="padding-top: 15px; font-weight: bold; color: #2196F3;">All 200 PCs Model</td>
            </tr>
            <tr>
                <td>DML θ ({X_short}→{Y_short}):</td>
                <td>{dml_results.get('theta_200', 0):.3f}</td>
                <td>{dml_results.get('theta_200', 0):.3f}</td>
            </tr>
            <tr>
                <td>Standard Error:</td>
                <td>{dml_results.get('se_200', 0):.3f}</td>
                <td>{dml_results.get('se_200', 0):.3f}</td>
            </tr>
            <tr>
                <td>95% CI:</td>
                <td>({dml_results.get('theta_200', 0) - 1.96 * dml_results.get('se_200', 0):.3f}, {dml_results.get('theta_200', 0) + 1.96 * dml_results.get('se_200', 0):.3f})</td>
                <td>({dml_results.get('theta_200', 0) - 1.96 * dml_results.get('se_200', 0):.3f}, {dml_results.get('theta_200', 0) + 1.96 * dml_results.get('se_200', 0):.3f})</td>
            </tr>
            <tr>
                <td>p-value:</td>
                <td>{dml_results.get('pval_200', 0):.4f}</td>
                <td>{dml_results.get('pval_200', 0):.4f}</td>
            </tr>
            <tr>
                <td style="color: #ff9800;">Effect Reduction vs Naive:</td>
                <td style="color: #ff9800;">{reduction_200:.1f}%</td>
                <td style="color: #ff9800;">{reduction_200:.1f}%</td>
            </tr>
            <tr>
                <td>{Y_short} R² (200 PCs):</td>
                <td>{dml_results.get('all_r2_y', 0):.3f}</td>
                <td>{dml_results.get('all_r2_y_cv', 0):.3f}</td>
            </tr>
            <tr>
                <td>{X_short} R² (200 PCs):</td>
                <td>{dml_results.get('all_r2_x', 0):.3f}</td>
                <td>{dml_results.get('all_r2_x_cv', 0):.3f}</td>
            </tr>
            
            <tr>
                <td colspan="3" style="padding-top: 15px; font-weight: bold; color: #FF9800;">Top 5 PCs Model</td>
            </tr>
            <tr>
                <td>DML θ ({X_short}→{Y_short}):</td>
                <td>{dml_results.get('theta_top5', 0):.3f}</td>
                <td>{dml_results.get('theta_top5', 0):.3f}</td>
            </tr>
            <tr>
                <td>Standard Error:</td>
                <td>{dml_results.get('se_top5', 0):.3f}</td>
                <td>{dml_results.get('se_top5', 0):.3f}</td>
            </tr>
            <tr>
                <td>95% CI:</td>
                <td>({dml_results.get('theta_top5', 0) - 1.96 * dml_results.get('se_top5', 0):.3f}, {dml_results.get('theta_top5', 0) + 1.96 * dml_results.get('se_top5', 0):.3f})</td>
                <td>({dml_results.get('theta_top5', 0) - 1.96 * dml_results.get('se_top5', 0):.3f}, {dml_results.get('theta_top5', 0) + 1.96 * dml_results.get('se_top5', 0):.3f})</td>
            </tr>
            <tr>
                <td>p-value:</td>
                <td>{dml_results.get('pval_top5', 0):.4f}</td>
                <td>{dml_results.get('pval_top5', 0):.4f}</td>
            </tr>
            <tr>
                <td style="color: #ff9800;">Effect Reduction vs Naive:</td>
                <td style="color: #ff9800;">{reduction_top5:.1f}%</td>
                <td style="color: #ff9800;">{reduction_top5:.1f}%</td>
            </tr>
            <tr>
                <td>{Y_short} R² (Top 5):</td>
                <td>{dml_results.get('top5_r2_y', 0):.3f}</td>
                <td>{dml_results.get('top5_r2_y_cv', 0):.3f}</td>
            </tr>
            <tr>
                <td>{X_short} R² (Top 5):</td>
                <td>{dml_results.get('top5_r2_x', 0):.3f}</td>
                <td>{dml_results.get('top5_r2_x_cv', 0):.3f}</td>
            </tr>
        </table>
        
        <div style="margin-top: 20px; font-size: 12px; color: #999;">
            <strong>Note:</strong> The effect reduction shows how much the causal effect is reduced when controlling for text features.
            Higher reduction indicates that text mediates more of the relationship between {X_name} and {Y_name}.
        </div>
    </div>
    
    <div id="gallery-container">
        <button class="close-btn" onclick="closeGalleryMode()">× Close Gallery</button>
        <h2 style="text-align: center; margin-bottom: 30px;">Essay Gallery</h2>
        <div class="gallery-grid" id="gallery-grid"></div>
    </div>
    
    <div id="keyboard-help">
        Click points to view essays | Scroll to zoom | Drag to rotate | R to reset view
    </div>
    
    <script>
        // Data
        const essays = {json.dumps(data_for_js, indent=2)};
        const topics = {json.dumps(topic_data, indent=2)};
        const topicStats = {json.dumps(topic_stats, indent=2)};
        const centerPoint = [{center_x:.6f}, {center_y:.6f}, {center_z:.6f}];
        
        // Three.js setup
        let scene, camera, renderer, controls;
        let pointCloud, pointGeometry;
        let raycaster = new THREE.Raycaster();
        let mouse = new THREE.Vector2();
        let selectedPoint = null;
        let colorMode = 'x_y';
        let hoveredPoint = null;
        
        // UI state
        let isGalleryMode = false;
        let isDMLModalOpen = false;
        
        // Initialize
        init();
        animate();
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            
            // Camera
            camera = new THREE.PerspectiveCamera(
                75,
                window.innerWidth / window.innerHeight,
                0.1,
                1000
            );
            camera.position.set(5, 5, 5);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(...centerPoint);
            controls.update();
            
            // Create point cloud
            createPointCloud();
            
            // Create topic labels
            createTopicLabels();
            
            // Add lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight.position.set(5, 5, 5);
            scene.add(directionalLight);
            
            // Event listeners
            window.addEventListener('resize', onWindowResize);
            renderer.domElement.addEventListener('mousemove', onMouseMove);
            renderer.domElement.addEventListener('click', onMouseClick);
            window.addEventListener('keydown', onKeyDown);
            
            // Custom cursor
            document.addEventListener('mousemove', (e) => {{
                const cursor = document.getElementById('cursor-indicator');
                cursor.style.left = e.clientX + 'px';
                cursor.style.top = e.clientY + 'px';
            }});
            
            // Hide loading
            document.getElementById('loading').style.display = 'none';
            
            // Setup UI
            setupUI();
        }}
        
        function createPointCloud() {{
            // Geometry
            pointGeometry = new THREE.BufferGeometry();
            
            const positions = new Float32Array(essays.length * 3);
            const colors = new Float32Array(essays.length * 3);
            const sizes = new Float32Array(essays.length);
            
            for (let i = 0; i < essays.length; i++) {{
                positions[i * 3] = essays[i].x;
                positions[i * 3 + 1] = essays[i].y;
                positions[i * 3 + 2] = essays[i].z;
                
                const color = getPointColor(i);
                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
                
                sizes[i] = essays[i].hdbscan_topic_id === -1 ? 4.0 : 6.0;
            }}
            
            pointGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            pointGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            pointGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
            
            // Material
            const pointMaterial = new THREE.PointsMaterial({{
                size: 5,
                sizeAttenuation: true,
                vertexColors: true,
                alphaTest: 0.5,
                transparent: true,
                opacity: 0.8
            }});
            
            // Create points
            pointCloud = new THREE.Points(pointGeometry, pointMaterial);
            scene.add(pointCloud);
        }}
        
        function getPointColor(index) {{
            if (colorMode === 'x_y') {{
                // Color by X and Y values
                const x_val = (essays[index].social_class - {x_min}) / ({x_max} - {x_min});
                const y_val = (essays[index].ai_rating - {y_min}) / ({y_max} - {y_min});
                
                return new THREE.Color(
                    0.2 + x_val * 0.8,
                    0.2 + y_val * 0.8,
                    0.4
                );
            }} else if (colorMode === 'topic') {{
                // Color by topic
                const topicId = essays[index].hdbscan_topic_id;
                if (topicId === -1) {{
                    return new THREE.Color(0.3, 0.3, 0.3);
                }}
                
                // Generate consistent color for topic
                const hue = (topicId * 137.5) % 360 / 360;
                return new THREE.Color().setHSL(hue, 0.7, 0.5);
            }}
        }}
        
        function createTopicLabels() {{
            // Add labels for topic centroids
            topics.forEach(topic => {{
                const sprite = createTextSprite(topic.keywords);
                sprite.position.set(...topic.centroid);
                scene.add(sprite);
            }});
        }}
        
        function createTextSprite(text) {{
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 256;
            canvas.height = 64;
            
            context.font = '20px Arial';
            context.fillStyle = 'rgba(255, 255, 255, 0.8)';
            context.textAlign = 'center';
            context.fillText(text, 128, 32);
            
            const texture = new THREE.CanvasTexture(canvas);
            const spriteMaterial = new THREE.SpriteMaterial({{
                map: texture,
                transparent: true
            }});
            
            const sprite = new THREE.Sprite(spriteMaterial);
            sprite.scale.set(2, 0.5, 1);
            
            return sprite;
        }}
        
        function onMouseMove(event) {{
            // Calculate mouse position in normalized device coordinates
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            // Update raycaster
            raycaster.setFromCamera(mouse, camera);
            
            // Check for intersections
            const intersects = raycaster.intersectObject(pointCloud);
            
            if (intersects.length > 0) {{
                const point = intersects[0];
                const index = point.index;
                
                if (hoveredPoint !== index) {{
                    hoveredPoint = index;
                    updateSelectedInfo(index);
                    
                    // Update cursor
                    document.getElementById('cursor-indicator').style.borderColor = 'rgba(76, 175, 80, 0.8)';
                }}
            }} else {{
                if (hoveredPoint !== null) {{
                    hoveredPoint = null;
                    document.getElementById('selected-info').innerHTML = 'Hover over points to see details';
                    document.getElementById('cursor-indicator').style.borderColor = 'rgba(255, 255, 255, 0.3)';
                }}
            }}
        }}
        
        function onMouseClick(event) {{
            if (hoveredPoint !== null) {{
                selectedPoint = hoveredPoint;
                showEssayViewer(selectedPoint);
            }}
        }}
        
        function updateSelectedInfo(index) {{
            const essay = essays[index];
            const pcInfo = essay.pc_info.map(pc => 
                `${{pc.pc}}: ${{pc.percentile.toFixed(1)}}%`
            ).join(', ');
            
            document.getElementById('selected-info').innerHTML = `
                <strong>Essay ${{essay.essay_id}}</strong><br>
                {Y_short}: ${{essay.ai_rating.toFixed(2)}}<br>
                {X_short}: ${{essay.social_class}}<br>
                Topic: ${{essay.hdbscan_topic_id}}<br>
                Top PCs: ${{pcInfo}}
            `;
        }}
        
        function showEssayViewer(index) {{
            const essay = essays[index];
            
            document.getElementById('essay-title').textContent = `Essay ${{essay.essay_id}}`;
            document.getElementById('essay-text').textContent = essay.essay;
            
            // Build detailed stats
            let statsHtml = `
                <div style="margin-bottom: 15px;">
                    <strong>{Y_short}:</strong> ${{essay.ai_rating.toFixed(2)}}<br>
                    <strong>{X_short}:</strong> ${{essay.social_class}}<br>
                    <strong>Topic:</strong> ${{essay.hdbscan_topic_id}}
                </div>
                <div style="margin-bottom: 15px;">
                    <strong>Top 5 Principal Components:</strong><br>
            `;
            
            essay.pc_info.forEach(pc => {{
                const contribColor = pc.contribution_ai > 0 ? '#4CAF50' : '#f44336';
                statsHtml += `
                    <div style="margin: 5px 0; padding: 5px; background: rgba(255,255,255,0.05); border-radius: 3px;">
                        <strong>${{pc.pc}}</strong> (explains ${{pc.variance_total.toFixed(1)}}% variance)<br>
                        Percentile: ${{pc.percentile.toFixed(1)}}%<br>
                        {Y_short} Contrib: <span style="color: ${{contribColor}}">${{pc.contribution_ai.toFixed(3)}}</span><br>
                        {X_short} Contrib: ${{pc.contribution_sc.toFixed(3)}}
                    </div>
                `;
            }});
            
            statsHtml += '</div>';
            
            document.getElementById('essay-stats').innerHTML = statsHtml;
            document.getElementById('essay-content').style.display = 'block';
        }}
        
        function closeEssayViewer() {{
            document.getElementById('essay-content').style.display = 'none';
        }}
        
        function toggleColorMode() {{
            colorMode = colorMode === 'x_y' ? 'topic' : 'x_y';
            updatePointColors();
        }}
        
        function updatePointColors() {{
            const colors = pointGeometry.attributes.color.array;
            
            for (let i = 0; i < essays.length; i++) {{
                const color = getPointColor(i);
                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
            }}
            
            pointGeometry.attributes.color.needsUpdate = true;
        }}
        
        function toggleFilters() {{
            const panel = document.getElementById('ai-filter-panel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        }}
        
        function setAIThreshold(min, max) {{
            document.getElementById('ai-min').value = min;
            document.getElementById('ai-max').value = max;
            document.getElementById('ai-min-val').textContent = min.toFixed(1);
            document.getElementById('ai-max-val').textContent = max.toFixed(1);
        }}
        
        function applyFilters() {{
            const aiMin = parseFloat(document.getElementById('ai-min').value);
            const aiMax = parseFloat(document.getElementById('ai-max').value);
            
            const colors = pointGeometry.attributes.color.array;
            const sizes = pointGeometry.attributes.size.array;
            
            for (let i = 0; i < essays.length; i++) {{
                const inRange = essays[i].ai_rating >= aiMin && essays[i].ai_rating <= aiMax;
                
                if (inRange) {{
                    const color = getPointColor(i);
                    colors[i * 3] = color.r;
                    colors[i * 3 + 1] = color.g;
                    colors[i * 3 + 2] = color.b;
                    sizes[i] = essays[i].hdbscan_topic_id === -1 ? 4.0 : 6.0;
                }} else {{
                    colors[i * 3] = 0.1;
                    colors[i * 3 + 1] = 0.1;
                    colors[i * 3 + 2] = 0.1;
                    sizes[i] = 2.0;
                }}
            }}
            
            pointGeometry.attributes.color.needsUpdate = true;
            pointGeometry.attributes.size.needsUpdate = true;
        }}
        
        function resetFilters() {{
            document.getElementById('ai-min').value = {y_min};
            document.getElementById('ai-max').value = {y_max};
            document.getElementById('ai-min-val').textContent = '{y_min:.1f}';
            document.getElementById('ai-max-val').textContent = '{y_max:.1f}';
            applyFilters();
        }}
        
        function toggleTopicStats() {{
            const stats = document.getElementById('topic-stats');
            if (stats.style.display === 'none') {{
                stats.style.display = 'block';
                updateTopicStats();
            }} else {{
                stats.style.display = 'none';
            }}
        }}
        
        function updateTopicStats() {{
            let html = '';
            
            topicStats.slice(0, 10).forEach(stat => {{
                const maxProb = Math.max(
                    stat.prob_ai_top10,
                    stat.prob_ai_bottom10,
                    stat.prob_sc_high,
                    stat.prob_sc_low
                );
                
                html += `
                    <div class="topic-row">
                        <div class="topic-label">
                            <strong>Topic ${{stat.topic_id}}</strong> (n=${{stat.size}})<br>
                            <span style="font-size: 11px; color: #999;">${{stat.keywords}}</span>
                        </div>
                        <div class="topic-bar">
                            <div class="topic-fill" style="width: ${{maxProb * 100}}%"></div>
                        </div>
                    </div>
                `;
            }});
            
            document.getElementById('topic-stats-content').innerHTML = html;
        }}
        
        function showDMLStats() {{
            document.getElementById('dml-modal').style.display = 'block';
            isDMLModalOpen = true;
        }}
        
        function closeDMLModal() {{
            document.getElementById('dml-modal').style.display = 'none';
            isDMLModalOpen = false;
        }}
        
        function toggleGalleryMode() {{
            if (!isGalleryMode) {{
                showGalleryMode();
            }} else {{
                closeGalleryMode();
            }}
        }}
        
        function showGalleryMode() {{
            isGalleryMode = true;
            document.getElementById('gallery-container').style.display = 'block';
            
            // Build gallery grid
            let html = '';
            essays.forEach((essay, index) => {{
                const topicKeywords = topics.find(t => t.topic_id === essay.hdbscan_topic_id)?.keywords || 'No topic';
                
                html += `
                    <div class="essay-card" onclick="selectFromGallery(${{index}})">
                        <h4>Essay ${{essay.essay_id}}</h4>
                        <div class="essay-text">${{essay.essay.substring(0, 200)}}...</div>
                        <div class="essay-meta">
                            {Y_short}: ${{essay.ai_rating.toFixed(2)}} | {X_short}: ${{essay.social_class}}<br>
                            Topic: ${{topicKeywords}}
                        </div>
                    </div>
                `;
            }});
            
            document.getElementById('gallery-grid').innerHTML = html;
        }}
        
        function closeGalleryMode() {{
            isGalleryMode = false;
            document.getElementById('gallery-container').style.display = 'none';
        }}
        
        function selectFromGallery(index) {{
            closeGalleryMode();
            selectedPoint = index;
            showEssayViewer(index);
            
            // Center camera on selected point
            const essay = essays[index];
            controls.target.set(essay.x, essay.y, essay.z);
            controls.update();
        }}
        
        function resetView() {{
            controls.reset();
            controls.target.set(...centerPoint);
            controls.update();
        }}
        
        function setupUI() {{
            // Range sliders
            document.getElementById('ai-min').addEventListener('input', function() {{
                document.getElementById('ai-min-val').textContent = this.value;
            }});
            
            document.getElementById('ai-max').addEventListener('input', function() {{
                document.getElementById('ai-max-val').textContent = this.value;
            }});
            
            // Collapse buttons
            document.querySelector('#info .collapse-btn').addEventListener('click', function() {{
                document.getElementById('info').classList.toggle('collapsed');
                this.textContent = document.getElementById('info').classList.contains('collapsed') ? '▶' : '◀';
            }});
            
            document.querySelector('#controls .collapse-btn').addEventListener('click', function() {{
                document.getElementById('controls').classList.toggle('collapsed');
                this.textContent = document.getElementById('controls').classList.contains('collapsed') ? '◀' : '▶';
            }});
        }}
        
        function onKeyDown(event) {{
            switch(event.key.toLowerCase()) {{
                case 'r':
                    resetView();
                    break;
                case 'g':
                    toggleGalleryMode();
                    break;
                case 'escape':
                    if (isDMLModalOpen) closeDMLModal();
                    if (isGalleryMode) closeGalleryMode();
                    closeEssayViewer();
                    break;
            }}
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
    </script>
</body>
</html>"""
    
    # Save the HTML
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Visualization generated from scratch and saved to {output_path}")
    
    return output_path