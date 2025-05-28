#!/usr/bin/env python3
"""
PerceptionHD Visualization Generator - Full v21 Implementation
Complete interactive 3D visualization with ALL v21 features adapted for generic X/Y variables
"""

import json
import numpy as np
from pathlib import Path
import re


def generate_visualization_v21(results, output_path):
    """
    Generate the complete interactive HTML visualization with ALL v21 features
    Adapted from minimal_umap_viz_v21.html with generic X/Y variable support
    """
    
    # Extract data
    df = results['df']
    X_umap = results['umap_coords'] if 'umap_coords' in results else results['X_umap']
    X_pca = results['pca_features'] if 'pca_features' in results else results['X_pca']
    clusters = results['clusters'] if 'clusters' in results else results['cluster_labels']
    config = results['config']
    
    # Variable names and configuration
    X_name = config['variables']['X']['name']
    X_short = config['variables']['X']['short_name']
    Y_name = config['variables']['Y']['name'] 
    Y_short = config['variables']['Y']['short_name']
    
    # Get values
    X_values = df['X'].values
    Y_values = df['Y'].values
    
    # Calculate percentiles for thresholds
    x_percentiles = {
        10: np.percentile(X_values, 10),
        25: np.percentile(X_values, 25),
        75: np.percentile(X_values, 75),
        90: np.percentile(X_values, 90)
    }
    
    y_percentiles = {
        10: np.percentile(Y_values, 10),
        25: np.percentile(Y_values, 25),
        75: np.percentile(Y_values, 75),
        90: np.percentile(Y_values, 90)
    }
    
    # Calculate center of point cloud
    center = X_umap.mean(axis=0)
    
    # Prepare data for JavaScript
    viz_data = []
    for i in range(len(df)):
        # Get PC percentiles and contributions
        pc_info = []
        if 'contributions_x' in results and 'contributions_y' in results:
            # Get this sample's contributions
            contrib_x = results['contributions_x'][i]
            contrib_y = results['contributions_y'][i]
            total_contrib = np.abs(contrib_x) + np.abs(contrib_y)
            
            # Find top 5 PCs for this specific sample
            top_5_sample = np.argsort(total_contrib)[-5:][::-1]
            
            for pc_idx in top_5_sample:
                # Calculate percentile for this PC value
                if 'pc_percentiles' in results:
                    percentile = results['pc_percentiles'][i, pc_idx]
                else:
                    pc_values = X_pca[:, pc_idx]
                    percentile = (np.searchsorted(np.sort(pc_values), X_pca[i, pc_idx]) / len(pc_values)) * 100
                
                pc_info.append({
                    'pc': f'PC{pc_idx}',
                    'percentile': float(percentile),
                    'contribution_ai': float(contrib_y[pc_idx]),  # Y contributions -> ai field
                    'contribution_sc': float(contrib_x[pc_idx]),  # X contributions -> sc field
                    'variance_total': float(results['dml_results']['variance_explained'][pc_idx] * 100) if 'variance_explained' in results['dml_results'] else 0
                })
        
        viz_data.append({
            'x': float(X_umap[i, 0]),
            'y': float(X_umap[i, 1]),
            'z': float(X_umap[i, 2]) if X_umap.shape[1] > 2 else 0,
            'id': str(df.iloc[i]['id']),
            'text': str(df.iloc[i]['text']),
            'ai_rating': float(Y_values[i]),  # Y variable -> ai_rating field
            'social_class': float(X_values[i]),  # X variable -> social_class field
            'hdbscan_topic_id': int(clusters[i]),
            'pc_info': pc_info
        })
    
    # Prepare topic data
    topic_viz_data = []
    topic_keywords = results.get('topics', {})
    if not topic_keywords and 'topic_keywords' in results:
        topic_keywords = results['topic_keywords']
    
    unique_clusters = np.unique(clusters)
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        cluster_points = X_umap[clusters == cluster_id]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            
            # Get keywords
            if cluster_id in topic_keywords:
                if isinstance(topic_keywords[cluster_id], list):
                    # List of (word, score) tuples
                    keywords = ', '.join([w[0] for w in topic_keywords[cluster_id][:5]])
                else:
                    keywords = str(topic_keywords[cluster_id])
            else:
                keywords = f"Topic {cluster_id}"
            
            topic_viz_data.append({
                'topic_id': int(cluster_id),
                'keywords': keywords,
                'centroid': [float(x) for x in centroid],
                'size': int(np.sum(clusters == cluster_id))
            })
    
    # Topic statistics for panel
    topic_stats_data = []
    x_top10_threshold = np.percentile(X_values, 90)
    x_bottom10_threshold = np.percentile(X_values, 10)
    
    # Use configured min/max for Y or calculate from data
    y_max = config['variables']['Y'].get('max_value', Y_values.max())
    y_min = config['variables']['Y'].get('min_value', Y_values.min())
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        topic_mask = clusters == cluster_id
        topic_X = X_values[topic_mask]
        topic_Y = Y_values[topic_mask]
        topic_size = len(topic_X)
        
        if topic_size > 0:
            # Calculate probabilities
            prob_x_top10 = np.mean(topic_X >= x_top10_threshold)
            prob_x_bottom10 = np.mean(topic_X <= x_bottom10_threshold)
            prob_y_high = np.mean(topic_Y >= y_max)
            prob_y_low = np.mean(topic_Y <= y_min)
            
            max_impact = max(prob_x_top10, prob_x_bottom10, prob_y_high, prob_y_low)
            
            # Get keywords
            if cluster_id in topic_keywords:
                if isinstance(topic_keywords[cluster_id], list):
                    keywords = ', '.join([w[0] for w in topic_keywords[cluster_id][:5]])
                else:
                    keywords = str(topic_keywords[cluster_id])
            else:
                keywords = f"Topic {cluster_id}"
            
            topic_stats_data.append({
                'topic_id': int(cluster_id),
                'keywords': keywords,
                'size': topic_size,
                'prob_x_top10': prob_x_top10,
                'prob_x_bottom10': prob_x_bottom10,
                'prob_y_high': prob_y_high,
                'prob_y_low': prob_y_low,
                'max_impact_prob': max_impact
            })
    
    # Sort by impact
    topic_stats_data.sort(key=lambda x: x['max_impact_prob'], reverse=True)
    
    # DML results
    dml_results = results['dml_results']
    
    # Get top 5 PCs for DML table
    top_5_pcs = dml_results.get('top_pcs', [])
    if isinstance(top_5_pcs, np.ndarray):
        top_5_pcs = top_5_pcs.tolist()
    
    # PC variance string for display
    if 'variance_explained' in results:
        var_exp = results['variance_explained']
    elif 'variance_explained' in dml_results:
        var_exp = dml_results['variance_explained']
    else:
        var_exp = results.get('pc_stats', {}).get('variance_explained', np.zeros(200))
    
    pc_variance_str = ', '.join([f"PC{i}: {var_exp[i]*100:.1f}%" for i in top_5_pcs[:5]])
    
    # Generate the complete HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{config['display']['title']}</title>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            overflow: hidden;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
        }}
        
        #container {{
            width: 100vw;
            height: 100vh;
        }}
        
        /* Panel base styles with z-index support */
        .panel {{
            position: absolute;
            background: rgba(20, 20, 30, 0.95);
            border: 1px solid rgba(100, 100, 120, 0.3);
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            z-index: 100;
            cursor: move;
        }}
        
        .panel.active {{
            z-index: 1000;
            border-color: rgba(100, 150, 255, 0.5);
            box-shadow: 0 4px 20px rgba(100, 150, 255, 0.3);
        }}
        
        .panel-header {{
            padding: 12px 15px;
            background: rgba(40, 40, 50, 0.5);
            border-bottom: 1px solid rgba(100, 100, 120, 0.2);
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }}
        
        .panel-title {{
            font-weight: 600;
            font-size: 14px;
            color: #b0b0ff;
        }}
        
        .panel-content {{
            padding: 15px;
            max-height: 70vh;
            overflow-y: auto;
        }}
        
        /* Essay viewer with resize handle */
        #essay-display {{
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 800px;
            min-width: 400px;
            max-width: 90vw;
            resize: horizontal;
            overflow: auto;
        }}
        
        #essay-display .panel-content {{
            max-height: 300px;
        }}
        
        .essay-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(100, 100, 120, 0.2);
        }}
        
        .essay-header h3 {{
            margin: 0;
            color: #d0d0ff;
            font-size: 16px;
        }}
        
        .essay-metadata {{
            display: flex;
            gap: 20px;
            font-size: 14px;
            color: #a0a0a0;
        }}
        
        .essay-text {{
            line-height: 1.6;
            font-size: 14px;
            color: #e0e0e0;
            white-space: pre-wrap;
        }}
        
        /* Info panel (left) */
        #info {{
            left: 20px;
            top: 20px;
            width: 300px;
        }}
        
        /* Controls panel (right) */
        #controls {{
            right: 20px;
            top: 20px;
            width: 350px;
        }}
        
        /* DML Results Table (bottom left) */
        #dml-table {{
            left: 20px;
            bottom: 20px;
            width: 600px;
        }}
        
        /* Topic Statistics Panel (bottom right) */
        #topic-stats-panel {{
            right: 20px;
            bottom: 20px;
            width: 650px;
        }}
        
        /* Control elements */
        .control-group {{
            margin: 15px 0;
        }}
        
        .control-group label {{
            display: block;
            margin-bottom: 5px;
            font-size: 13px;
            color: #b0b0b0;
        }}
        
        input[type="range"] {{
            width: 100%;
            height: 4px;
            background: rgba(100, 100, 120, 0.3);
            outline: none;
            -webkit-appearance: none;
        }}
        
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 12px;
            height: 12px;
            background: #6080ff;
            cursor: pointer;
            border-radius: 50%;
        }}
        
        .value-display {{
            display: inline-block;
            margin-left: 10px;
            font-family: monospace;
            color: #80ff80;
        }}
        
        /* Buttons */
        .button-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin: 10px 0;
        }}
        
        button {{
            padding: 6px 12px;
            background: rgba(60, 60, 80, 0.6);
            border: 1px solid rgba(100, 100, 120, 0.3);
            color: #d0d0d0;
            cursor: pointer;
            border-radius: 4px;
            font-size: 12px;
            transition: all 0.2s;
        }}
        
        button:hover {{
            background: rgba(80, 80, 100, 0.6);
            border-color: rgba(120, 120, 140, 0.5);
        }}
        
        button.active {{
            background: rgba(80, 100, 180, 0.6);
            border-color: rgba(100, 150, 255, 0.5);
            color: #ffffff;
        }}
        
        /* Topic buttons */
        .topic-button {{
            padding: 4px 8px;
            font-size: 11px;
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        
        th {{
            text-align: left;
            padding: 8px;
            background: rgba(40, 40, 50, 0.5);
            border-bottom: 2px solid rgba(100, 100, 120, 0.3);
            color: #b0b0ff;
            font-weight: 600;
        }}
        
        td {{
            padding: 8px;
            border-bottom: 1px solid rgba(100, 100, 120, 0.2);
        }}
        
        tr:hover {{
            background: rgba(60, 60, 80, 0.2);
        }}
        
        /* PC cell styling */
        .pc-cell {{
            font-family: monospace;
            color: #ffcc80;
        }}
        
        /* Gallery mode */
        #gallery-panel {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90vw;
            height: 90vh;
            background: rgba(20, 20, 30, 0.98);
            border: 2px solid rgba(100, 100, 120, 0.5);
            border-radius: 10px;
            display: none;
            z-index: 2000;
        }}
        
        #gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            padding: 20px;
            overflow-y: auto;
            height: calc(100% - 60px);
        }}
        
        .gallery-item {{
            background: rgba(40, 40, 50, 0.5);
            border: 1px solid rgba(100, 100, 120, 0.3);
            padding: 15px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .gallery-item:hover {{
            background: rgba(60, 60, 80, 0.5);
            border-color: rgba(100, 150, 255, 0.5);
        }}
        
        /* Collapsible sections */
        .collapsible {{
            cursor: pointer;
            user-select: none;
        }}
        
        .collapsible::before {{
            content: '▼';
            display: inline-block;
            margin-right: 5px;
            transition: transform 0.2s;
        }}
        
        .collapsible.collapsed::before {{
            transform: rotate(-90deg);
        }}
        
        .collapsible-content {{
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        
        /* Close button */
        .close-button {{
            background: none;
            border: none;
            color: #ff6080;
            font-size: 18px;
            cursor: pointer;
            padding: 0;
            width: 20px;
            height: 20px;
        }}
        
        /* PC variance info */
        .pc-variance-info {{
            font-size: 11px;
            color: #888;
            margin-top: 5px;
            font-style: italic;
        }}
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(40, 40, 50, 0.5);
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: rgba(100, 100, 120, 0.5);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(120, 120, 140, 0.5);
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="container"></div>
    
    <!-- Info Panel -->
    <div id="info" class="panel">
        <div class="panel-header">
            <span class="panel-title">Thresholds</span>
        </div>
        <div class="panel-content">
            <div class="control-group">
                <label>{X_name} Low (Bottom 10%):</label>
                <input type="range" id="x-low-val" min="{X_values.min():.1f}" max="{X_values.max():.1f}" 
                       step="0.1" value="{x_percentiles[10]:.1f}">
                <span class="value-display" id="x-low-display">{x_percentiles[10]:.1f}</span>
            </div>
            <div class="control-group">
                <label>{X_name} High (Top 10%):</label>
                <input type="range" id="x-high-val" min="{X_values.min():.1f}" max="{X_values.max():.1f}" 
                       step="0.1" value="{x_percentiles[90]:.1f}">
                <span class="value-display" id="x-high-display">{x_percentiles[90]:.1f}</span>
            </div>
            <div class="control-group">
                <label>{Y_name} Low:</label>
                <input type="range" id="y-low-val" min="{Y_values.min():.1f}" max="{Y_values.max():.1f}" 
                       step="0.1" value="{y_min:.1f}">
                <span class="value-display" id="y-low-display">{y_min:.1f}</span>
            </div>
            <div class="control-group">
                <label>{Y_name} High:</label>
                <input type="range" id="y-high-val" min="{Y_values.min():.1f}" max="{Y_values.max():.1f}" 
                       step="0.1" value="{y_max:.1f}">
                <span class="value-display" id="y-high-display">{y_max:.1f}</span>
            </div>
            <div class="control-group">
                <h4 class="collapsible">Legend</h4>
                <div class="collapsible-content" style="max-height: 200px;">
                    <div style="margin: 10px 0;">
                        <div style="display: inline-block; width: 12px; height: 12px; background: #66ff66; margin-right: 5px;"></div>
                        High {X_short} + High {Y_short}
                    </div>
                    <div style="margin: 10px 0;">
                        <div style="display: inline-block; width: 12px; height: 12px; background: #6666ff; margin-right: 5px;"></div>
                        High {X_short} + Low {Y_short}
                    </div>
                    <div style="margin: 10px 0;">
                        <div style="display: inline-block; width: 12px; height: 12px; background: #ff6666; margin-right: 5px;"></div>
                        Low {X_short} + High {Y_short}
                    </div>
                    <div style="margin: 10px 0;">
                        <div style="display: inline-block; width: 12px; height: 12px; background: #ffff66; margin-right: 5px;"></div>
                        Low {X_short} + Low {Y_short}
                    </div>
                    <div style="margin: 10px 0;">
                        <div style="display: inline-block; width: 12px; height: 12px; background: #aaaaaa; margin-right: 5px;"></div>
                        Neither extreme
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Controls Panel -->
    <div id="controls" class="panel">
        <div class="panel-header">
            <span class="panel-title">Visualization Controls</span>
        </div>
        <div class="panel-content">
            <div class="control-group">
                <label>Color By:</label>
                <select id="color-mode" style="width: 100%; padding: 5px;">
                    <option value="categories">Categories</option>
                    <option value="X">{X_name}</option>
                    <option value="Y">{Y_name}</option>
                    <option value="topics">Topics</option>
                    <option value="pc_gradient">PC Gradient</option>
                </select>
            </div>
            
            <div id="pc-selector-group" class="control-group" style="display: none;">
                <label>Select PC for Gradient:</label>
                <select id="pc-selector" style="width: 100%; padding: 5px;">
                    <!-- Will be populated dynamically -->
                </select>
            </div>
            
            <div class="control-group">
                <h4 class="collapsible">Topics</h4>
                <div class="collapsible-content" style="max-height: 300px;">
                    <button id="show-all-topics">Show All</button>
                    <button id="hide-all-topics">Hide All</button>
                    <button id="toggle-topic-labels">Toggle Labels</button>
                    <div id="topic-buttons" class="button-group" style="margin-top: 10px;">
                        <!-- Topic buttons will be added here -->
                    </div>
                </div>
            </div>
            
            <div class="control-group">
                <label>
                    <input type="checkbox" id="auto-rotate"> Auto-rotate
                </label>
            </div>
            
            <div class="control-group">
                <label>Point Size:</label>
                <input type="range" id="point-size" min="0.1" max="3" step="0.1" value="1">
                <span class="value-display" id="point-size-display">1.0</span>
            </div>
            
            <div class="control-group">
                <label>Point Opacity:</label>
                <input type="range" id="point-opacity" min="0.1" max="1" step="0.05" value="0.8">
                <span class="value-display" id="point-opacity-display">0.8</span>
            </div>
            
            <div class="control-group">
                <button id="reset-camera" style="width: 100%;">Reset Camera</button>
            </div>
            
            <div class="control-group">
                <button id="toggle-gallery" style="width: 100%;">Gallery Mode</button>
            </div>
            
            <div class="control-group">
                <button id="toggle-panels" style="width: 100%;">Toggle All Panels</button>
            </div>
        </div>
    </div>
    
    <!-- Essay Display -->
    <div id="essay-display" class="panel" style="display: none;">
        <div class="panel-header">
            <span class="panel-title">Essay Details</span>
            <button class="close-button" onclick="document.getElementById('essay-display').style.display='none'">×</button>
        </div>
        <div class="panel-content">
            <div class="essay-header">
                <h3 id="essay-id"></h3>
                <div class="essay-metadata">
                    <span>{X_name}: <span id="essay-x" style="color: #80ff80;"></span></span>
                    <span>{Y_name}: <span id="essay-y" style="color: #8080ff;"></span></span>
                    <span>Topic: <span id="essay-topic" style="color: #ff8080;"></span></span>
                </div>
            </div>
            <div class="essay-text" id="essay-text"></div>
            <div style="margin-top: 15px;">
                <h4>Top Contributing PCs:</h4>
                <table id="essay-pc-table">
                    <thead>
                        <tr>
                            <th>PC</th>
                            <th>Percentile</th>
                            <th>{X_short} Contrib</th>
                            <th>{Y_short} Contrib</th>
                            <th>Variance</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
    </div>
    
    <!-- DML Results Table -->
    <div id="dml-table" class="panel">
        <div class="panel-header">
            <span class="panel-title">Double Machine Learning Results</span>
        </div>
        <div class="panel-content">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Direction</th>
                        <th>Effect (θ)</th>
                        <th>Std Error</th>
                        <th>p-value</th>
                        <th>R² (full)</th>
                        <th>R² (CF)</th>
                        <th>PCs</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Naive (No Controls)</td>
                        <td>{Y_short} → {X_short}</td>
                        <td>{dml_results['theta_naive']:.3f}</td>
                        <td>-</td>
                        <td>{dml_results['pval_naive']:.4f}</td>
                        <td>-</td>
                        <td>-</td>
                        <td>None</td>
                    </tr>
                    <tr>
                        <td>DML (200 PCs)</td>
                        <td>{Y_short} → {X_short}</td>
                        <td>{dml_results['theta_200']:.3f}</td>
                        <td>{dml_results['se_200']:.3f}</td>
                        <td>{dml_results['pval_200']:.4f}</td>
                        <td>{dml_results.get('all_r2', 0):.3f}</td>
                        <td>-</td>
                        <td>All 200</td>
                    </tr>
                    <tr style="background: rgba(80, 100, 180, 0.2);">
                        <td><strong>DML (Top 5 PCs)</strong></td>
                        <td>{Y_short} → {X_short}</td>
                        <td><strong>{dml_results['theta_top5']:.3f}</strong></td>
                        <td>{dml_results['se_top5']:.3f}</td>
                        <td>{dml_results['pval_top5']:.4f}</td>
                        <td>{dml_results.get('top5_r2', 0):.3f}</td>
                        <td>-</td>
                        <td class="pc-cell">{', '.join([f'PC{pc}' for pc in top_5_pcs[:5]])}</td>
                    </tr>
                </tbody>
            </table>
            <div class="pc-variance-info">
                Top 5 PCs variance explained: {pc_variance_str}
            </div>
        </div>
    </div>
    
    <!-- Topic Statistics Panel -->
    <div id="topic-stats-panel" class="panel">
        <div class="panel-header">
            <span class="panel-title">Topic Statistics - Extreme Group Probabilities</span>
        </div>
        <div class="panel-content">
            <table>
                <thead>
                    <tr>
                        <th>Topic</th>
                        <th>Size</th>
                        <th>Top 10% {X_short}</th>
                        <th>Bottom 10% {X_short}</th>
                        <th>High {Y_short}</th>
                        <th>Low {Y_short}</th>
                        <th>Max Impact</th>
                    </tr>
                </thead>
                <tbody id="topic-stats-tbody">
                    <!-- Will be populated dynamically -->
                </tbody>
            </table>
        </div>
    </div>
    
    <!-- Gallery Mode Panel -->
    <div id="gallery-panel">
        <div class="panel-header">
            <span class="panel-title">Gallery Mode - Selected Essays</span>
            <button class="close-button" onclick="toggleGallery()">×</button>
        </div>
        <div id="gallery-grid">
            <!-- Gallery items will be added here -->
        </div>
    </div>
    
    <script>
        // Data from Python
        const essays = {json.dumps(viz_data)};
        const topics = {json.dumps(topic_viz_data)};
        const topicStats = {json.dumps(topic_stats_data)};
        const centerPoint = {json.dumps([float(center[0]), float(center[1]), float(center[2]) if len(center) > 2 else 0])};
        
        // Variable names
        const X_name = "{X_name}";
        const Y_name = "{Y_name}";
        const X_short = "{X_short}";
        const Y_short = "{Y_short}";
        
        // Scene setup
        let scene, camera, renderer, controls;
        let points, topicLabels = [];
        let raycaster, mouse;
        let currentEssayIndex = null;
        let galleryMode = false;
        let selectedEssays = [];
        let topicVisibility = {{}};
        let showTopicLabels = true;
        let colorMode = 'categories';
        let selectedPC = 0;
        let panelsVisible = true;
        let currentZIndex = 1000;
        
        // Initialize Three.js
        init();
        animate();
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a0a);
            
            // Camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(30, 30, 30);
            camera.lookAt(centerPoint[0], centerPoint[1], centerPoint[2]);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = false;
            controls.target.set(centerPoint[0], centerPoint[1], centerPoint[2]);
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight.position.set(1, 1, 0.5).normalize();
            scene.add(directionalLight);
            
            // Create points
            createPoints();
            
            // Create topic labels
            createTopicLabels();
            
            // Raycaster for interaction
            raycaster = new THREE.Raycaster();
            raycaster.params.Points.threshold = 0.5;
            mouse = new THREE.Vector2();
            
            // Set up UI
            setupUI();
            
            // Populate topic stats
            populateTopicStats();
            
            // Handle window resize
            window.addEventListener('resize', onWindowResize);
            window.addEventListener('mousemove', onMouseMove);
            window.addEventListener('click', onClick);
        }}
        
        function createPoints() {{
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(essays.length * 3);
            const colors = new Float32Array(essays.length * 3);
            
            for (let i = 0; i < essays.length; i++) {{
                positions[i * 3] = essays[i].x;
                positions[i * 3 + 1] = essays[i].y;
                positions[i * 3 + 2] = essays[i].z;
            }}
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            
            const material = new THREE.PointsMaterial({{
                size: 1,
                vertexColors: true,
                opacity: 0.8,
                transparent: true,
                sizeAttenuation: false
            }});
            
            points = new THREE.Points(geometry, material);
            scene.add(points);
            
            updateColors();
        }}
        
        function createTopicLabels() {{
            const loader = new THREE.FontLoader();
            
            topics.forEach(topic => {{
                // Create sprite for label
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = 512;
                canvas.height = 128;
                
                context.font = 'Bold 40px Arial';
                context.fillStyle = 'rgba(255, 255, 255, 0.8)';
                context.textAlign = 'center';
                context.fillText(`Topic ${{topic.topic_id}}`, 256, 50);
                context.font = '30px Arial';
                context.fillText(topic.keywords.substring(0, 40) + '...', 256, 90);
                
                const texture = new THREE.CanvasTexture(canvas);
                const spriteMaterial = new THREE.SpriteMaterial({{ map: texture, opacity: 0.8 }});
                const sprite = new THREE.Sprite(spriteMaterial);
                
                sprite.position.set(topic.centroid[0], topic.centroid[1] + 5, topic.centroid[2] || 0);
                sprite.scale.set(10, 2.5, 1);
                sprite.userData = {{ topicId: topic.topic_id }};
                
                topicLabels.push(sprite);
                scene.add(sprite);
                
                topicVisibility[topic.topic_id] = true;
            }});
        }}
        
        function updateColors() {{
            const colors = points.geometry.attributes.color.array;
            
            const xLowVal = parseFloat(document.getElementById('x-low-val').value);
            const xHighVal = parseFloat(document.getElementById('x-high-val').value);
            const yLowVal = parseFloat(document.getElementById('y-low-val').value);
            const yHighVal = parseFloat(document.getElementById('y-high-val').value);
            
            for (let i = 0; i < essays.length; i++) {{
                let r, g, b;
                
                if (colorMode === 'categories') {{
                    const essay = essays[i];
                    if (essay.social_class >= xHighVal && essay.ai_rating >= yHighVal) {{
                        r = 0.4; g = 1; b = 0.4; // Green
                    }} else if (essay.social_class >= xHighVal && essay.ai_rating <= yLowVal) {{
                        r = 0.4; g = 0.4; b = 1; // Blue
                    }} else if (essay.social_class <= xLowVal && essay.ai_rating >= yHighVal) {{
                        r = 1; g = 0.4; b = 0.4; // Red
                    }} else if (essay.social_class <= xLowVal && essay.ai_rating <= yLowVal) {{
                        r = 1; g = 1; b = 0.4; // Yellow
                    }} else {{
                        r = 0.7; g = 0.7; b = 0.7; // Gray
                    }}
                }} else if (colorMode === 'X') {{
                    const val = (essays[i].social_class - {X_values.min()}) / ({X_values.max()} - {X_values.min()});
                    r = val; g = 0.5; b = 1 - val;
                }} else if (colorMode === 'Y') {{
                    const val = (essays[i].ai_rating - {Y_values.min()}) / ({Y_values.max()} - {Y_values.min()});
                    r = 1 - val; g = val; b = 0.5;
                }} else if (colorMode === 'topics') {{
                    const topicId = essays[i].hdbscan_topic_id;
                    if (topicId === -1) {{
                        r = 0.5; g = 0.5; b = 0.5;
                    }} else {{
                        const hue = (topicId * 137.5) % 360;
                        const c = hslToRgb(hue / 360, 0.7, 0.5);
                        r = c[0]; g = c[1]; b = c[2];
                    }}
                }} else if (colorMode === 'pc_gradient' && essays[i].pc_info.length > selectedPC) {{
                    const percentile = essays[i].pc_info[selectedPC].percentile / 100;
                    r = percentile; g = 0.5; b = 1 - percentile;
                }}
                
                colors[i * 3] = r;
                colors[i * 3 + 1] = g;
                colors[i * 3 + 2] = b;
            }}
            
            points.geometry.attributes.color.needsUpdate = true;
        }}
        
        function hslToRgb(h, s, l) {{
            let r, g, b;
            if (s === 0) {{
                r = g = b = l;
            }} else {{
                const hue2rgb = (p, q, t) => {{
                    if (t < 0) t += 1;
                    if (t > 1) t -= 1;
                    if (t < 1/6) return p + (q - p) * 6 * t;
                    if (t < 1/2) return q;
                    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                    return p;
                }};
                const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
                const p = 2 * l - q;
                r = hue2rgb(p, q, h + 1/3);
                g = hue2rgb(p, q, h);
                b = hue2rgb(p, q, h - 1/3);
            }}
            return [r, g, b];
        }}
        
        function setupUI() {{
            // Threshold controls
            ['x-low-val', 'x-high-val', 'y-low-val', 'y-high-val'].forEach(id => {{
                const input = document.getElementById(id);
                const display = document.getElementById(id.replace('-val', '-display'));
                input.addEventListener('input', (e) => {{
                    display.textContent = parseFloat(e.target.value).toFixed(1);
                    updateColors();
                }});
            }});
            
            // Color mode
            document.getElementById('color-mode').addEventListener('change', (e) => {{
                colorMode = e.target.value;
                document.getElementById('pc-selector-group').style.display = 
                    colorMode === 'pc_gradient' ? 'block' : 'none';
                updateColors();
            }});
            
            // PC selector
            const pcSelector = document.getElementById('pc-selector');
            for (let i = 0; i < 200; i++) {{
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `PC${{i}}`;
                pcSelector.appendChild(option);
            }}
            pcSelector.addEventListener('change', (e) => {{
                selectedPC = parseInt(e.target.value);
                updateColors();
            }});
            
            // Point controls
            document.getElementById('point-size').addEventListener('input', (e) => {{
                const size = parseFloat(e.target.value);
                points.material.size = size;
                document.getElementById('point-size-display').textContent = size.toFixed(1);
            }});
            
            document.getElementById('point-opacity').addEventListener('input', (e) => {{
                const opacity = parseFloat(e.target.value);
                points.material.opacity = opacity;
                document.getElementById('point-opacity-display').textContent = opacity.toFixed(2);
            }});
            
            // Auto-rotate
            document.getElementById('auto-rotate').addEventListener('change', (e) => {{
                controls.autoRotate = e.target.checked;
                controls.autoRotateSpeed = 0.5;
            }});
            
            // Reset camera
            document.getElementById('reset-camera').addEventListener('click', () => {{
                camera.position.set(30, 30, 30);
                controls.target.set(centerPoint[0], centerPoint[1], centerPoint[2]);
                controls.update();
            }});
            
            // Gallery mode
            document.getElementById('toggle-gallery').addEventListener('click', toggleGallery);
            
            // Toggle panels
            document.getElementById('toggle-panels').addEventListener('click', () => {{
                panelsVisible = !panelsVisible;
                const panels = document.querySelectorAll('.panel');
                panels.forEach(panel => {{
                    if (panel.id !== 'gallery-panel') {{
                        panel.style.display = panelsVisible ? 'block' : 'none';
                    }}
                }});
            }});
            
            // Topic buttons
            createTopicButtons();
            
            // Topic controls
            document.getElementById('show-all-topics').addEventListener('click', () => {{
                Object.keys(topicVisibility).forEach(id => {{
                    topicVisibility[id] = true;
                }});
                updateTopicVisibility();
                updateTopicButtons();
            }});
            
            document.getElementById('hide-all-topics').addEventListener('click', () => {{
                Object.keys(topicVisibility).forEach(id => {{
                    topicVisibility[id] = false;
                }});
                updateTopicVisibility();
                updateTopicButtons();
            }});
            
            document.getElementById('toggle-topic-labels').addEventListener('click', () => {{
                showTopicLabels = !showTopicLabels;
                topicLabels.forEach(label => {{
                    label.visible = showTopicLabels && topicVisibility[label.userData.topicId];
                }});
            }});
            
            // Collapsible sections
            document.querySelectorAll('.collapsible').forEach(elem => {{
                elem.addEventListener('click', () => {{
                    elem.classList.toggle('collapsed');
                    const content = elem.nextElementSibling;
                    if (elem.classList.contains('collapsed')) {{
                        content.style.maxHeight = '0';
                    }} else {{
                        content.style.maxHeight = content.scrollHeight + 'px';
                    }}
                }});
            }});
            
            // Panel z-index management
            document.querySelectorAll('.panel').forEach(panel => {{
                panel.addEventListener('mousedown', () => {{
                    currentZIndex++;
                    panel.style.zIndex = currentZIndex;
                    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
                    panel.classList.add('active');
                }});
            }});
        }}
        
        function createTopicButtons() {{
            const container = document.getElementById('topic-buttons');
            topics.forEach(topic => {{
                const btn = document.createElement('button');
                btn.className = 'topic-button active';
                btn.textContent = `T${{topic.topic_id}}: ${{topic.keywords.substring(0, 20)}}...`;
                btn.title = topic.keywords;
                btn.dataset.topicId = topic.topic_id;
                
                btn.addEventListener('click', () => {{
                    topicVisibility[topic.topic_id] = !topicVisibility[topic.topic_id];
                    updateTopicVisibility();
                    updateTopicButtons();
                }});
                
                container.appendChild(btn);
            }});
        }}
        
        function updateTopicButtons() {{
            document.querySelectorAll('.topic-button').forEach(btn => {{
                const topicId = parseInt(btn.dataset.topicId);
                btn.classList.toggle('active', topicVisibility[topicId]);
            }});
        }}
        
        function updateTopicVisibility() {{
            const positions = points.geometry.attributes.position.array;
            const colors = points.geometry.attributes.color.array;
            
            for (let i = 0; i < essays.length; i++) {{
                const topicId = essays[i].hdbscan_topic_id;
                const visible = topicId === -1 || topicVisibility[topicId];
                
                if (!visible) {{
                    positions[i * 3] = 10000;
                    positions[i * 3 + 1] = 10000;
                    positions[i * 3 + 2] = 10000;
                }} else {{
                    positions[i * 3] = essays[i].x;
                    positions[i * 3 + 1] = essays[i].y;
                    positions[i * 3 + 2] = essays[i].z;
                }}
            }}
            
            points.geometry.attributes.position.needsUpdate = true;
            
            topicLabels.forEach(label => {{
                label.visible = showTopicLabels && topicVisibility[label.userData.topicId];
            }});
            
            updateColors();
        }}
        
        function populateTopicStats() {{
            const tbody = document.getElementById('topic-stats-tbody');
            topicStats.forEach(stat => {{
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>T${{stat.topic_id}}: ${{stat.keywords.substring(0, 30)}}...</td>
                    <td>${{stat.size}}</td>
                    <td>${{(stat.prob_x_top10 * 100).toFixed(1)}}%</td>
                    <td>${{(stat.prob_x_bottom10 * 100).toFixed(1)}}%</td>
                    <td>${{(stat.prob_y_high * 100).toFixed(1)}}%</td>
                    <td>${{(stat.prob_y_low * 100).toFixed(1)}}%</td>
                    <td style="font-weight: bold; color: #ff8080;">${{(stat.max_impact_prob * 100).toFixed(1)}}%</td>
                `;
            }});
        }}
        
        function showEssay(index) {{
            const essay = essays[index];
            currentEssayIndex = index;
            
            document.getElementById('essay-id').textContent = essay.id;
            document.getElementById('essay-x').textContent = essay.social_class.toFixed(2);
            document.getElementById('essay-y').textContent = essay.ai_rating.toFixed(2);
            document.getElementById('essay-topic').textContent = essay.hdbscan_topic_id === -1 ? 'Noise' : `T${{essay.hdbscan_topic_id}}`;
            document.getElementById('essay-text').textContent = essay.text;
            
            // PC table
            const tbody = document.querySelector('#essay-pc-table tbody');
            tbody.innerHTML = '';
            essay.pc_info.forEach(pc => {{
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td class="pc-cell">${{pc.pc}}</td>
                    <td>${{pc.percentile.toFixed(1)}}%</td>
                    <td>${{pc.contribution_x.toFixed(3)}}</td>
                    <td>${{pc.contribution_y.toFixed(3)}}</td>
                    <td>${{pc.variance_total.toFixed(1)}}%</td>
                `;
            }});
            
            document.getElementById('essay-display').style.display = 'block';
        }}
        
        function toggleGallery() {{
            galleryMode = !galleryMode;
            const panel = document.getElementById('gallery-panel');
            
            if (galleryMode) {{
                panel.style.display = 'block';
                updateGalleryGrid();
            }} else {{
                panel.style.display = 'none';
            }}
        }}
        
        function updateGalleryGrid() {{
            const grid = document.getElementById('gallery-grid');
            grid.innerHTML = '';
            
            selectedEssays.forEach(index => {{
                const essay = essays[index];
                const item = document.createElement('div');
                item.className = 'gallery-item';
                item.innerHTML = `
                    <h4>${{essay.id}}</h4>
                    <div style="margin: 5px 0;">
                        <strong>${{X_name}}:</strong> ${{essay.social_class.toFixed(2)}} | 
                        <strong>${{Y_name}}:</strong> ${{essay.ai_rating.toFixed(2)}}
                    </div>
                    <div style="margin-top: 10px; max-height: 150px; overflow-y: auto;">
                        ${{essay.text.substring(0, 200)}}...
                    </div>
                `;
                item.addEventListener('click', () => showEssay(index));
                grid.appendChild(item);
            }});
        }}
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(points);
            
            if (intersects.length > 0 && !galleryMode) {{
                const index = intersects[0].index;
                if (topicVisibility[essays[index].hdbscan_topic_id] !== false) {{
                    showEssay(index);
                }}
            }}
        }}
        
        function onClick(event) {{
            if (galleryMode && currentEssayIndex !== null) {{
                if (!selectedEssays.includes(currentEssayIndex)) {{
                    selectedEssays.push(currentEssayIndex);
                    updateGalleryGrid();
                }}
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
    
    # Save HTML
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Full v21 visualization saved to {output_path}")
    
    return output_path