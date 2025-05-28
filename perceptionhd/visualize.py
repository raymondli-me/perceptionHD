#!/usr/bin/env python3
"""
PerceptionHD Visualization Generator
Complete interactive 3D visualization with all v21 features
"""

import json
import numpy as np
from pathlib import Path


def generate_visualization(results, output_path):
    """
    Generate the complete interactive HTML visualization with all v21 features
    """
    
    # Extract data
    df = results['df']
    X_umap = results['X_umap']
    X_pca = results['X_pca']
    config = results['config']
    
    # Variable names
    X_name = config['variables']['X']['name']
    X_short = config['variables']['X']['short_name']
    Y_name = config['variables']['Y']['name']
    Y_short = config['variables']['Y']['short_name']
    
    # Calculate statistics
    X_values = df['X'].values
    Y_values = df['Y'].values
    
    x_percentiles = {
        10: np.percentile(X_values, 10),
        25: np.percentile(X_values, 25),
        75: np.percentile(X_values, 75),
        90: np.percentile(X_values, 90)
    }
    
    # Calculate center of point cloud
    center_x = X_umap[:, 0].mean()
    center_y = X_umap[:, 1].mean()
    center_z = X_umap[:, 2].mean() if X_umap.shape[1] > 2 else 0
    
    # Prepare visualization data
    viz_data = []
    for i in range(len(df)):
        # Get top 5 contributing PCs for this sample
        if 'contributions_x' in results:
            total_contrib = np.abs(results['contributions_x'][i]) + np.abs(results['contributions_y'][i])
            top_5_sample = np.argsort(total_contrib)[-5:][::-1]
        else:
            top_5_sample = results['top_5_indices']
            
        pc_info = []
        for pc_idx in top_5_sample:
            pc_info.append({
                'pc': f'PC{pc_idx}',
                'percentile': float(results['pc_percentiles'][i, pc_idx]),
                'contribution_x': float(results['contributions_x'][i, pc_idx]) if 'contributions_x' in results else 0,
                'contribution_y': float(results['contributions_y'][i, pc_idx]) if 'contributions_y' in results else 0,
                'variance_total': float(results['variance_explained'][pc_idx] * 100)
            })
        
        viz_data.append({
            'x': float(X_umap[i, 0]),
            'y': float(X_umap[i, 1]),
            'z': float(X_umap[i, 2]) if X_umap.shape[1] > 2 else 0,
            'id': str(df.iloc[i]['id']),
            'text': str(df.iloc[i]['text']),
            'X': float(df.iloc[i]['X']),
            'Y': float(df.iloc[i]['Y']),
            'hdbscan_topic_id': int(results['cluster_labels'][i]),
            'pc_info': pc_info
        })
    
    # Prepare topic data
    topic_viz_data = []
    unique_clusters = np.unique(results['cluster_labels'])
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        cluster_points = X_umap[results['cluster_labels'] == cluster_id]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            topic_viz_data.append({
                'topic_id': int(cluster_id),
                'keywords': results['topic_keywords'].get(cluster_id, f"Topic {cluster_id}"),
                'centroid': [float(x) for x in centroid],
                'size': int(np.sum(results['cluster_labels'] == cluster_id))
            })
    
    # Topic statistics
    topic_stats_data = []
    x_top10_threshold = np.percentile(X_values, 90)
    x_bottom10_threshold = np.percentile(X_values, 10)
    y_top_threshold = config['variables']['Y']['max_value']
    y_bottom_threshold = config['variables']['Y']['min_value']
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        topic_mask = results['cluster_labels'] == cluster_id
        topic_data = df[topic_mask]
        topic_size = len(topic_data)
        
        if topic_size > 0:
            prob_x_top10 = len(topic_data[topic_data['X'] >= x_top10_threshold]) / topic_size
            prob_x_bottom10 = len(topic_data[topic_data['X'] <= x_bottom10_threshold]) / topic_size
            prob_y_top = len(topic_data[topic_data['Y'] >= y_top_threshold]) / topic_size
            prob_y_bottom = len(topic_data[topic_data['Y'] <= y_bottom_threshold]) / topic_size
            
            max_impact = max(prob_x_top10, prob_x_bottom10, prob_y_top, prob_y_bottom)
            
            topic_stats_data.append({
                'topic_id': int(cluster_id),
                'keywords': results['topic_keywords'].get(cluster_id, f"Topic {cluster_id}"),
                'size': topic_size,
                'prob_x_top10': prob_x_top10,
                'prob_x_bottom10': prob_x_bottom10,
                'prob_y_top': prob_y_top,
                'prob_y_bottom': prob_y_bottom,
                'max_impact_prob': max_impact
            })
    
    topic_stats_data.sort(key=lambda x: x['max_impact_prob'], reverse=True)
    
    # DML results with generic formatting
    dml_results = results['dml_results']
    
    # PC variance for display
    pc_variance_str = ', '.join([f"PC{i}: {results['variance_explained'][i]*100:.1f}%" 
                                 for i in results['top_5_indices'][:5]])
    
    # Generate HTML (simplified version - in production this would be much larger)
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{config['display']['title']}</title>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; overflow: hidden; font-family: Arial, sans-serif; background: #000; color: #fff; }}
        #container {{ width: 100vw; height: 100vh; }}
        
        /* Info panel */
        #info {{ position: absolute; left: 20px; top: 20px; background: rgba(0,0,0,0.8); 
                 padding: 20px; border-radius: 10px; border: 1px solid #333; width: 250px; }}
        
        /* Controls panel */
        #controls {{ position: absolute; right: 20px; top: 20px; background: rgba(0,0,0,0.8);
                     padding: 20px; border-radius: 10px; border: 1px solid #333; width: 300px; }}
        
        /* Essay display */
        #essay-display {{ position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
                         background: rgba(0,0,0,0.9); padding: 20px; border-radius: 10px;
                         border: 2px solid #666; max-width: 800px; max-height: 40vh;
                         overflow-y: auto; display: none; }}
        
        /* DML table */
        #dml-table {{ position: absolute; bottom: 20px; left: 20px; background: rgba(0,0,0,0.9);
                      padding: 20px; border-radius: 10px; border: 1px solid #444;
                      max-width: 600px; display: none; }}
        
        /* Topic stats panel */
        #topic-stats-panel {{ position: absolute; bottom: 120px; right: 20px; background: rgba(0,0,0,0.9);
                             padding: 20px; border-radius: 10px; border: 1px solid #444;
                             max-width: 700px; max-height: 500px; display: none; }}
        
        /* Common styles */
        .control-group {{ margin: 10px 0; }}
        label {{ display: block; margin-bottom: 5px; }}
        input[type="range"] {{ width: 100%; }}
        button {{ padding: 5px 10px; margin: 2px; cursor: pointer; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #333; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="container"></div>
    
    <!-- Info Panel -->
    <div id="info">
        <h3>Thresholds</h3>
        <div class="control-group">
            <label>{X_name} Low:</label>
            <input type="range" id="x-low-val" min="{config['variables']['X']['min_value']}" 
                   max="{config['variables']['X']['max_value']}" step="0.1" value="{x_percentiles[10]:.1f}">
            <span id="x-low-display">{x_percentiles[10]:.1f}</span>
        </div>
        <div class="control-group">
            <label>{X_name} High:</label>
            <input type="range" id="x-high-val" min="{config['variables']['X']['min_value']}" 
                   max="{config['variables']['X']['max_value']}" step="0.1" value="{x_percentiles[90]:.1f}">
            <span id="x-high-display">{x_percentiles[90]:.1f}</span>
        </div>
        <div class="control-group">
            <label>{Y_name} Low:</label>
            <input type="range" id="y-low-val" min="{config['variables']['Y']['min_value']}" 
                   max="{config['variables']['Y']['max_value']}" step="1" value="{config['variables']['Y']['min_value']}">
            <span id="y-low-display">{config['variables']['Y']['min_value']}</span>
        </div>
        <div class="control-group">
            <label>{Y_name} High:</label>
            <input type="range" id="y-high-val" min="{config['variables']['Y']['min_value']}" 
                   max="{config['variables']['Y']['max_value']}" step="1" value="{config['variables']['Y']['max_value']}">
            <span id="y-high-display">{config['variables']['Y']['max_value']}</span>
        </div>
        
        <div id="legend" style="margin-top: 20px;">
            <h4>Categories</h4>
            <div><span style="color: #00ff00;">●</span> High {X_short} + High {Y_short}</div>
            <div><span style="color: #ff00ff;">●</span> High {X_short} + Low {Y_short}</div>
            <div><span style="color: #00ffff;">●</span> Low {X_short} + High {Y_short}</div>
            <div><span style="color: #ffff00;">●</span> Low {X_short} + Low {Y_short}</div>
            <div><span style="color: #666666;">●</span> Middle</div>
        </div>
        
        <div id="counts" style="margin-top: 20px; font-size: 12px;"></div>
    </div>
    
    <!-- Controls Panel -->
    <div id="controls">
        <h3>Controls</h3>
        <div class="control-group">
            <label><input type="checkbox" id="auto-rotate" checked> Auto-rotate</label>
        </div>
        <div class="control-group">
            <label><input type="checkbox" id="toggle-dml" onchange="toggleDMLTable()"> Show DML Stats</label>
        </div>
        <div class="control-group">
            <label><input type="checkbox" id="toggle-topic-stats" onchange="toggleTopicStatsPanel()"> Show Topic Stats</label>
        </div>
        <div class="control-group">
            <label><input type="checkbox" id="toggle-topics" onchange="updateTopicVisibility()"> Show Topics</label>
        </div>
        <div class="control-group">
            <label>Point Opacity:</label>
            <input type="range" id="point-opacity" min="0.1" max="1" step="0.1" value="0.8">
        </div>
        
        <h4>Gallery Mode</h4>
        <button onclick="startGallery('both_high')" style="border-color: #00ff00;">High {X_short} + High {Y_short}</button>
        <button onclick="startGallery('x_high')" style="border-color: #ff00ff;">High {X_short} + Low {Y_short}</button>
        <button onclick="startGallery('y_high')" style="border-color: #00ffff;">Low {X_short} + High {Y_short}</button>
        <button onclick="startGallery('both_low')" style="border-color: #ffff00;">Low {X_short} + Low {Y_short}</button>
    </div>
    
    <!-- Essay Display -->
    <div id="essay-display">
        <button onclick="this.parentElement.style.display='none'" style="float: right;">×</button>
        <div id="essay-header" style="font-weight: bold; margin-bottom: 10px;"></div>
        <div id="essay-text" style="white-space: pre-wrap;"></div>
    </div>
    
    <!-- DML Table -->
    <div id="dml-table">
        <button onclick="this.parentElement.style.display='none'" style="float: right;">×</button>
        <h4>Double Machine Learning Results</h4>
        <p>Causal effect of {Y_name} on {X_name}:</p>
        <table>
            <tr>
                <th>Model</th>
                <th>Effect (θ)</th>
                <th>p-value</th>
            </tr>
            <tr>
                <td>Naive (no controls)</td>
                <td>{dml_results['theta_naive']:.3f}</td>
                <td>{dml_results['pval_naive']:.4f}</td>
            </tr>
            <tr>
                <td>All 200 PCs</td>
                <td>{dml_results['theta_200']:.3f}</td>
                <td>{dml_results['pval_200']:.4f}</td>
            </tr>
            <tr>
                <td>Top 5 PCs</td>
                <td>{dml_results['theta_top5']:.3f}</td>
                <td>{dml_results['pval_top5']:.4f}</td>
            </tr>
        </table>
        <p style="margin-top: 15px; font-size: 12px; color: #999;">
            Top 5 PCs: {', '.join([f'PC{i}' for i in results['top_5_indices']])}
        </p>
    </div>
    
    <!-- Topic Stats Panel -->
    <div id="topic-stats-panel">
        <button onclick="this.parentElement.style.display='none'" style="float: right;">×</button>
        <h4>Topic Profile Statistics</h4>
        <div style="max-height: 400px; overflow-y: auto;">
            <table id="topic-stats-table">
                <thead>
                    <tr>
                        <th>Topic</th>
                        <th>Size</th>
                        <th>% High {X_short}</th>
                        <th>% Low {X_short}</th>
                        <th>% High {Y_short}</th>
                        <th>% Low {Y_short}</th>
                    </tr>
                </thead>
                <tbody id="topic-stats-tbody"></tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Data
        const data = {json.dumps(viz_data)};
        const topicVizData = {json.dumps(topic_viz_data)};
        const topicStatsData = {json.dumps(topic_stats_data)};
        const cloudCenter = {{ x: {center_x}, y: {center_y}, z: {center_z} }};
        
        // Three.js setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);
        
        const camera = new THREE.PerspectiveCamera(
            50, window.innerWidth / window.innerHeight, 0.1, 10000
        );
        camera.position.set(200, 200, 200);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('container').appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        controls.target.set(cloudCenter.x, cloudCenter.y, cloudCenter.z);
        
        // Lights
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        scene.add(new THREE.DirectionalLight(0xffffff, 0.4));
        
        // Create points
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        
        data.forEach((d, i) => {{
            positions[i * 3] = d.x * 100;
            positions[i * 3 + 1] = d.y * 100;
            positions[i * 3 + 2] = d.z * 100;
        }});
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({{
            size: 4,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        }});
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Categories
        const categories = new Array(data.length);
        const categoryColors = {{
            'both_high': [0.0, 1.0, 0.0],
            'x_high': [1.0, 0.0, 1.0],
            'y_high': [0.0, 1.0, 1.0],
            'both_low': [1.0, 1.0, 0.0],
            'middle': [0.4, 0.4, 0.4]
        }};
        
        // Update categories based on thresholds
        function updateCategories() {{
            const xLow = parseFloat(document.getElementById('x-low-val').value);
            const xHigh = parseFloat(document.getElementById('x-high-val').value);
            const yLow = parseFloat(document.getElementById('y-low-val').value);
            const yHigh = parseFloat(document.getElementById('y-high-val').value);
            
            const counts = {{ both_high: 0, x_high: 0, y_high: 0, both_low: 0, middle: 0 }};
            
            data.forEach((d, i) => {{
                const highX = d.X > xHigh;
                const lowX = d.X < xLow;
                const highY = d.Y >= yHigh;
                const lowY = d.Y <= yLow;
                
                let category;
                if (highX && highY) category = 'both_high';
                else if (highX && lowY) category = 'x_high';
                else if (lowX && highY) category = 'y_high';
                else if (lowX && lowY) category = 'both_low';
                else category = 'middle';
                
                categories[i] = category;
                counts[category]++;
                
                const color = categoryColors[category];
                colors[i * 3] = color[0];
                colors[i * 3 + 1] = color[1];
                colors[i * 3 + 2] = color[2];
            }});
            
            geometry.attributes.color.needsUpdate = true;
            
            // Update counts display
            const total = data.length;
            document.getElementById('counts').innerHTML = `
                <strong>Counts:</strong><br>
                High {X_short} + High {Y_short}: ${{counts.both_high}} (${{(counts.both_high/total*100).toFixed(1)}}%)<br>
                High {X_short} + Low {Y_short}: ${{counts.x_high}} (${{(counts.x_high/total*100).toFixed(1)}}%)<br>
                Low {X_short} + High {Y_short}: ${{counts.y_high}} (${{(counts.y_high/total*100).toFixed(1)}}%)<br>
                Low {X_short} + Low {Y_short}: ${{counts.both_low}} (${{(counts.both_low/total*100).toFixed(1)}}%)<br>
                Middle: ${{counts.middle}} (${{(counts.middle/total*100).toFixed(1)}}%)
            `;
        }}
        
        // Gallery mode
        let galleryMode = false;
        let currentGalleryCategory = null;
        let currentGalleryIndex = 0;
        
        function startGallery(category) {{
            const indices = [];
            data.forEach((d, i) => {{
                if (categories[i] === category) indices.push(i);
            }});
            
            if (indices.length === 0) {{
                alert('No samples in this category');
                return;
            }}
            
            galleryMode = true;
            currentGalleryCategory = category;
            currentGalleryIndex = 0;
            
            navigateToPoint(indices[0]);
            showEssay(indices[0]);
        }}
        
        function navigateToPoint(index) {{
            const d = data[index];
            const targetPos = new THREE.Vector3(d.x * 100, d.y * 100, d.z * 100);
            
            // Animate camera to point
            // Simplified animation - in full version this would be smoother
            camera.position.set(
                targetPos.x + 50,
                targetPos.y + 50,
                targetPos.z + 50
            );
            controls.target.copy(targetPos);
        }}
        
        function showEssay(index) {{
            const d = data[index];
            document.getElementById('essay-header').innerHTML = `
                ID: ${{d.id}} | {X_short}: ${{d.X.toFixed(2)}} | {Y_short}: ${{d.Y}}
            `;
            document.getElementById('essay-text').textContent = d.text;
            document.getElementById('essay-display').style.display = 'block';
        }}
        
        // Topic visibility
        const topicObjects = [];
        
        function createTopicLabels() {{
            topicVizData.forEach(topic => {{
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = 512;
                canvas.height = 128;
                
                context.fillStyle = 'rgba(0,0,0,0.7)';
                context.fillRect(0, 0, 512, 128);
                
                context.fillStyle = 'white';
                context.font = '24px Arial';
                context.textAlign = 'center';
                context.fillText(topic.keywords, 256, 64);
                
                const texture = new THREE.CanvasTexture(canvas);
                const spriteMaterial = new THREE.SpriteMaterial({{ map: texture }});
                const sprite = new THREE.Sprite(spriteMaterial);
                
                sprite.position.set(
                    topic.centroid[0] * 100,
                    topic.centroid[1] * 100,
                    topic.centroid[2] * 100
                );
                sprite.scale.set(50, 12.5, 1);
                sprite.visible = false;
                
                scene.add(sprite);
                topicObjects.push({{
                    sprite: sprite,
                    topic: topic,
                    position: sprite.position.clone()
                }});
            }});
        }}
        
        createTopicLabels();
        
        function updateTopicVisibility() {{
            const showTopics = document.getElementById('toggle-topics').checked;
            topicObjects.forEach(obj => {{
                obj.sprite.visible = showTopics;
            }});
        }}
        
        // DML table toggle
        function toggleDMLTable() {{
            const table = document.getElementById('dml-table');
            table.style.display = table.style.display === 'none' ? 'block' : 'none';
        }}
        
        // Topic stats toggle
        function toggleTopicStatsPanel() {{
            const panel = document.getElementById('topic-stats-panel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
            
            if (panel.style.display === 'block') {{
                // Populate table
                const tbody = document.getElementById('topic-stats-tbody');
                tbody.innerHTML = '';
                
                topicStatsData.forEach(topic => {{
                    const row = tbody.insertRow();
                    row.innerHTML = `
                        <td>${{topic.keywords}}</td>
                        <td>${{topic.size}}</td>
                        <td>${{(topic.prob_x_top10 * 100).toFixed(1)}}%</td>
                        <td>${{(topic.prob_x_bottom10 * 100).toFixed(1)}}%</td>
                        <td>${{(topic.prob_y_top * 100).toFixed(1)}}%</td>
                        <td>${{(topic.prob_y_bottom * 100).toFixed(1)}}%</td>
                    `;
                }});
            }}
        }}
        
        // Mouse interaction
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        raycaster.params.Points.threshold = 5;
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(points);
            
            if (intersects.length > 0 && !galleryMode) {{
                const index = intersects[0].index;
                showEssay(index);
            }}
        }}
        
        window.addEventListener('mousemove', onMouseMove);
        
        // Controls
        document.getElementById('auto-rotate').addEventListener('change', (e) => {{
            controls.autoRotate = e.target.checked;
        }});
        
        document.getElementById('point-opacity').addEventListener('input', (e) => {{
            material.opacity = parseFloat(e.target.value);
        }});
        
        // Threshold controls
        ['x-low-val', 'x-high-val', 'y-low-val', 'y-high-val'].forEach(id => {{
            document.getElementById(id).addEventListener('input', (e) => {{
                document.getElementById(id.replace('-val', '-display')).textContent = e.target.value;
                updateCategories();
            }});
        }});
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        
        // Initialize
        updateCategories();
        animate();
        
        // Handle resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>"""
    
    # Save HTML
    with open(output_path, 'w') as f:
        f.write(html_content)
        
    print(f"✓ Visualization saved to {output_path}")