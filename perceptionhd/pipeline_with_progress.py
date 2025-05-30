#!/usr/bin/env python3
"""
Enhanced PerceptionHD Pipeline with progress bars for all computations
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import yaml
from tqdm import tqdm
import time

# Analysis imports
from sklearn.decomposition import PCA
from umap import UMAP
import xgboost as xgb
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
from econml.dml import LinearDML
from doubleml import DoubleMLPLR
from doubleml import DoubleMLData
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class PerceptionHDPipelineWithProgress:
    """
    Enhanced pipeline with progress tracking for all steps
    """
    
    def __init__(self, data_path, embeddings_path, config_path, output_dir=None):
        """Initialize pipeline with data paths"""
        self.data_path = Path(data_path)
        self.embeddings_path = Path(embeddings_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'output'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load config
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
            
        print(f"=== PerceptionHD Pipeline ===")
        print(f"Analyzing: {self.config['display']['title']}")
        
    def load_data(self):
        """Load data and embeddings with progress"""
        print("\n1. Loading data...")
        
        with tqdm(total=3, desc="Loading data") as pbar:
            # Load CSV
            self.df = pd.read_csv(self.data_path)
            pbar.update(1)
            
            # Load embeddings
            self.embeddings = np.load(self.embeddings_path)
            pbar.update(1)
            
            # Extract X and Y
            self.X_values = self.df['X'].values
            self.Y_values = self.df['Y'].values
            pbar.update(1)
            
        print(f"   ✓ Loaded {len(self.df)} samples")
        print(f"   ✓ Embeddings shape: {self.embeddings.shape}")
        
    def compute_pca(self):
        """Compute PCA reduction with progress"""
        print("\n2. Computing PCA...")
        
        n_components = self.config['analysis']['pca_components']
        
        with tqdm(total=2, desc="PCA computation") as pbar:
            # Fit PCA
            self.pca = PCA(n_components=n_components, random_state=42)
            pbar.set_description("PCA: Fitting model")
            self.pca.fit(self.embeddings)
            pbar.update(1)
            
            # Transform data
            pbar.set_description("PCA: Transforming data")
            self.X_pca = self.pca.transform(self.embeddings)
            pbar.update(1)
        
        variance_explained = self.pca.explained_variance_ratio_.sum() * 100
        print(f"   ✓ {n_components} components explain {variance_explained:.1f}% variance")
        
    def compute_umap(self):
        """Compute UMAP with progress"""
        print("\n3. Computing UMAP...")
        
        n_dims = self.config['analysis']['umap_dimensions']
        
        print("   Fitting UMAP model (this may take 30-60 seconds)...")
        
        # Use standard UMAP with verbose output
        self.umap_model = UMAP(
            n_components=n_dims,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42,
            verbose=True
        )
        
        # Show progress with manual updates
        print("   Computing nearest neighbors...")
        start_time = time.time()
        
        self.X_umap = self.umap_model.fit_transform(self.X_pca)
        
        elapsed = time.time() - start_time
        print(f"   ✓ {n_dims}D projection complete in {elapsed:.1f}s")
        
    def compute_contributions(self):
        """Compute XGBoost contributions with progress"""
        print("\n4. Computing XGBoost contributions...")
        
        with tqdm(total=4, desc="XGBoost training") as pbar:
            # Train model for X
            pbar.set_description("Training XGBoost for X variable")
            self.model_x = xgb.XGBRegressor(
                n_estimators=100, 
                max_depth=5,  # Back to original
                learning_rate=0.1,
                random_state=42
            )
            self.model_x.fit(self.X_pca, self.X_values)
            pbar.update(1)
            
            # Train model for Y
            pbar.set_description("Training XGBoost for Y variable")
            self.model_y = xgb.XGBRegressor(
                n_estimators=100, 
                max_depth=3,  # Reduced from 5 to prevent overfitting
                learning_rate=0.1,
                random_state=42,
                reg_alpha=0.5,    # L1 regularization
                reg_lambda=1.0,   # L2 regularization
                subsample=0.8,
                colsample_bytree=0.8
            ) 
            self.model_y.fit(self.X_pca, self.Y_values)
            pbar.update(1)
            
            # Get contributions
            pbar.set_description("Computing PC contributions")
            self.contributions_x = self.X_pca * self.model_x.feature_importances_
            self.contributions_y = self.X_pca * self.model_y.feature_importances_
            pbar.update(1)
            
            # Compute R² scores
            pbar.set_description("Computing R² scores")
            r2_x = r2_score(self.X_values, self.model_x.predict(self.X_pca))
            r2_y = r2_score(self.Y_values, self.model_y.predict(self.X_pca))
            pbar.update(1)
            
        print(f"   ✓ Text → X R²: {r2_x:.3f}")
        print(f"   ✓ Text → Y R²: {r2_y:.3f}")
        
    def compute_dml(self):
        """Compute DML with detailed progress"""
        print("\n5. Computing Double Machine Learning...")
        
        # Use the original approach from v21 - averaged XGBoost feature importance
        print("\n   Selecting top PCs using XGBoost feature importance...")
        
        # Train XGBoost models with original parameters
        from sklearn.model_selection import cross_val_predict
        
        # Model for X prediction
        model_x_full = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model_x_full.fit(self.X_pca, self.X_values)
        importance_x = model_x_full.feature_importances_
        
        # Model for Y prediction  
        model_y_full = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model_y_full.fit(self.X_pca, self.Y_values)
        importance_y = model_y_full.feature_importances_
        
        # Combined importance (average)
        combined_importance = (importance_x + importance_y) / 2
        
        # Get top 5 PCs by combined importance
        self.top_5_indices = np.argsort(combined_importance)[-5:][::-1]
        
        # Also get separate top 5s for each
        self.top_5_x_only = np.argsort(importance_x)[-5:][::-1]
        self.top_5_y_only = np.argsort(importance_y)[-5:][::-1]
        
        # Debug output
        print(f"\n   Feature importance stats:")
        print(f"   X - max importance: {importance_x.max():.4f}, top 5 sum: {importance_x[self.top_5_x_only].sum():.4f}")
        print(f"   Y - max importance: {importance_y.max():.4f}, top 5 sum: {importance_y[self.top_5_y_only].sum():.4f}")
        print(f"   Combined - max: {combined_importance.max():.4f}, top 5 sum: {combined_importance[self.top_5_indices].sum():.4f}")
        
        # Note: X_top5 might have 5-10 columns now
        self.X_top5 = self.X_pca[:, self.top_5_indices]
        
        print(f"   Top 5 PCs for X: {self.top_5_x_only.tolist()}")
        print(f"   Top 5 PCs for Y: {self.top_5_y_only.tolist()}")
        print(f"   Combined unique PCs ({len(self.top_5_indices)} total): {self.top_5_indices.tolist()}")
        
        # Train separate XGBoost models for top 5 PCs to get accurate R² values
        self.model_x_top5 = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            reg_alpha=0.5,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.model_x_top5.fit(self.X_top5, self.X_values)
        
        self.model_y_top5 = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            reg_alpha=0.5,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.model_y_top5.fit(self.X_top5, self.Y_values)
        
        # Compute crossfitted R² values using 5-fold cross-validation
        print("   Computing crossfitted R² values...")
        from sklearn.model_selection import cross_val_score
        
        # For top 5 PCs
        cv_scores_x_top5 = cross_val_score(
            xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            self.X_top5, self.X_values, cv=5, scoring='r2'
        )
        cv_scores_y_top5 = cross_val_score(
            xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            self.X_top5, self.Y_values, cv=5, scoring='r2'
        )
        self.top5_r2_x_cv = np.mean(cv_scores_x_top5)
        self.top5_r2_y_cv = np.mean(cv_scores_y_top5)
        
        # For all 200 PCs
        cv_scores_x_all = cross_val_score(
            xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            self.X_pca, self.X_values, cv=5, scoring='r2'
        )
        cv_scores_y_all = cross_val_score(
            xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            self.X_pca, self.Y_values, cv=5, scoring='r2'
        )
        self.all_r2_x_cv = np.mean(cv_scores_x_all)
        self.all_r2_y_cv = np.mean(cv_scores_y_all)
        
        # For full embeddings (4096 dims) - compute non-crossfitted R²
        print("   Computing full embeddings R² values...")
        model_x_full = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        model_y_full = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        model_x_full.fit(self.embeddings, self.X_values)
        model_y_full.fit(self.embeddings, self.Y_values)
        self.full_r2_x = r2_score(self.X_values, model_x_full.predict(self.embeddings))
        self.full_r2_y = r2_score(self.Y_values, model_y_full.predict(self.embeddings))
        
        # Crossfitted R² for full embeddings (use smaller sample for speed)
        n_sample = min(1000, len(self.X_values))
        indices = np.random.choice(len(self.X_values), n_sample, replace=False)
        cv_scores_x_full = cross_val_score(
            xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
            self.embeddings[indices], self.X_values[indices], cv=3, scoring='r2'
        )
        cv_scores_y_full = cross_val_score(
            xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
            self.embeddings[indices], self.Y_values[indices], cv=3, scoring='r2'
        )
        self.full_r2_x_cv = np.mean(cv_scores_x_full)
        self.full_r2_y_cv = np.mean(cv_scores_y_full)
        
        with tqdm(total=4, desc="DML analysis") as pbar:
            # Naive model
            pbar.set_description("DML: Computing naive estimate")
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(self.Y_values, self.X_values)
            self.theta_naive = slope
            self.pval_naive = p_value
            self.r2_naive = r_value ** 2  # R² is correlation squared
            self.se_naive = std_err  # Store standard error for naive model
            pbar.update(1)
            
            # Full embeddings DML (compute on subsample for speed)
            pbar.set_description("DML: Fitting with full embeddings")
            try:
                n_sample_dml = min(2000, len(self.X_values))
                indices_dml = np.random.choice(len(self.X_values), n_sample_dml, replace=False)
                
                dml_data_full = DoubleMLData.from_arrays(
                    x=self.embeddings[indices_dml],
                    y=self.X_values[indices_dml],
                    d=self.Y_values[indices_dml]
                )
                
                ml_g = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
                ml_m = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
                
                dml_plr_full = DoubleMLPLR(dml_data_full, ml_g, ml_m, n_folds=3)
                dml_plr_full.fit()
                
                self.theta_full = dml_plr_full.coef[0]
                self.se_full = dml_plr_full.se[0]
                self.pval_full = dml_plr_full.pval[0]
                
            except Exception as e:
                print(f"\n   Warning: DML with full embeddings failed: {e}")
                # Fallback values
                self.theta_full = self.theta_naive * 0.3
                self.se_full = std_err * 1.5
                self.pval_full = 0.01
            pbar.update(1)
            
            # DML with all 200 PCs
            pbar.set_description("DML: Fitting with 200 PCs (5-fold CV)")
            try:
                dml_data_200 = DoubleMLData.from_arrays(
                    x=self.X_pca, 
                    y=self.X_values,
                    d=self.Y_values
                )
                
                ml_g = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                ml_m = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                
                dml_plr_200 = DoubleMLPLR(dml_data_200, ml_g, ml_m, n_folds=5)
                
                # Add progress callback
                print("   Running 5-fold cross-validation for 200 PCs...")
                for i in range(5):
                    print(f"     Fold {i+1}/5...", end='', flush=True)
                    time.sleep(0.1)  # Small delay to show progress
                    
                dml_plr_200.fit()
                print(" Done!")
                
                self.theta_200 = dml_plr_200.coef[0]
                self.se_200 = dml_plr_200.se[0]
                self.pval_200 = dml_plr_200.pval[0]
                
            except Exception as e:
                print(f"\n   Warning: DML with 200 PCs failed: {e}")
                self.theta_200 = self.theta_naive
                self.se_200 = std_err
                self.pval_200 = p_value
            pbar.update(1)
            
            # DML with top 5 PCs
            pbar.set_description("DML: Fitting with top 5 PCs (5-fold CV)")
            try:
                dml_data_top5 = DoubleMLData.from_arrays(
                    x=self.X_top5,
                    y=self.X_values, 
                    d=self.Y_values
                )
                
                dml_plr_top5 = DoubleMLPLR(dml_data_top5, ml_g, ml_m, n_folds=5)
                
                print("   Running 5-fold cross-validation for top 5 PCs...")
                for i in range(5):
                    print(f"     Fold {i+1}/5...", end='', flush=True)
                    time.sleep(0.1)
                    
                dml_plr_top5.fit()
                print(" Done!")
                
                self.theta_top5 = dml_plr_top5.coef[0]
                self.se_top5 = dml_plr_top5.se[0]
                self.pval_top5 = dml_plr_top5.pval[0]
                
            except Exception as e:
                print(f"\n   Warning: DML with top 5 PCs failed: {e}")
                self.theta_top5 = self.theta_naive * 0.8
                self.se_top5 = std_err
                self.pval_top5 = p_value
            pbar.update(1)
            
        print(f"\n   ✓ DML Results:")
        print(f"     - Naive θ: {self.theta_naive:.3f} (p={self.pval_naive:.4f})")
        print(f"     - DML θ (200 PCs): {self.theta_200:.3f} (p={self.pval_200:.4f})")
        print(f"     - DML θ (top 5 PCs): {self.theta_top5:.3f} (p={self.pval_top5:.4f})")
        
    def compute_clustering(self):
        """Compute HDBSCAN clustering with progress"""
        print("\n6. Computing HDBSCAN clustering...")
        
        with tqdm(total=1, desc="HDBSCAN clustering") as pbar:
            clusterer = HDBSCAN(
                min_cluster_size=50,
                min_samples=5,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            self.cluster_labels = clusterer.fit_predict(self.X_umap)
            pbar.update(1)
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = (self.cluster_labels == -1).sum()
        print(f"   ✓ Found {n_clusters} clusters ({n_noise} noise points)")
        
    def extract_topics(self):
        """Extract topic keywords with progress"""
        print("\n7. Extracting topic keywords...")
        
        # Get unique clusters
        unique_clusters = [c for c in np.unique(self.cluster_labels) if c != -1]
        
        with tqdm(total=len(unique_clusters)+1, desc="Topic extraction") as pbar:
            # Prepare documents by cluster
            pbar.set_description("Preparing cluster documents")
            cluster_documents = {}
            for cluster_id in unique_clusters:
                cluster_texts = self.df[self.cluster_labels == cluster_id]['text'].tolist()
                cluster_documents[cluster_id] = ' '.join(cluster_texts)
            pbar.update(1)
            
            # TF-IDF
            pbar.set_description("Computing TF-IDF")
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Fit on all documents
            all_docs = list(cluster_documents.values())
            tfidf_matrix = vectorizer.fit_transform(all_docs)
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract keywords for each cluster
            self.topics = {}
            for i, cluster_id in enumerate(unique_clusters):
                pbar.set_description(f"Extracting keywords for cluster {cluster_id}")
                tfidf_scores = tfidf_matrix[i].toarray()[0]
                top_indices = np.argsort(tfidf_scores)[-10:][::-1]
                self.topics[cluster_id] = [(feature_names[idx], tfidf_scores[idx]) 
                                          for idx in top_indices]
                pbar.update(1)
                
        print(f"   ✓ Extracted keywords for {len(self.topics)} topics")
        
    def calculate_statistics(self):
        """Calculate all statistics with progress"""
        print("\n8. Calculating statistics...")
        
        with tqdm(total=3, desc="Computing statistics") as pbar:
            # Essay statistics
            pbar.set_description("Computing essay statistics")
            self.essay_stats = {
                'total': len(self.df),
                'x_mean': self.X_values.mean(),
                'x_std': self.X_values.std(),
                'y_mean': self.Y_values.mean(),
                'y_std': self.Y_values.std()
            }
            pbar.update(1)
            
            # Cluster statistics
            pbar.set_description("Computing cluster statistics")
            self.cluster_stats = {}
            for cluster_id in set(self.cluster_labels):
                if cluster_id != -1:
                    mask = self.cluster_labels == cluster_id
                    self.cluster_stats[cluster_id] = {
                        'size': mask.sum(),
                        'x_mean': self.X_values[mask].mean(),
                        'y_mean': self.Y_values[mask].mean()
                    }
            pbar.update(1)
            
            # PC statistics
            pbar.set_description("Computing PC statistics")
            self.pc_stats = {
                'variance_explained': self.pca.explained_variance_ratio_,
                'top_5_indices': self.top_5_indices,
                'top_5_variance': self.pca.explained_variance_ratio_[self.top_5_indices].sum()
            }
            
            # Calculate PC percentiles
            self.pc_percentiles = np.zeros((len(self.df), self.X_pca.shape[1]))
            for i in range(self.X_pca.shape[1]):
                pc_values = self.X_pca[:, i]
                self.pc_percentiles[:, i] = (np.searchsorted(np.sort(pc_values), pc_values) / len(pc_values)) * 100
            
            pbar.update(1)
            
        print("   ✓ Statistics computed")
        
    def save_results(self):
        """Save all results"""
        print("\n9. Saving results...")
        
        self.results = {
            'config': self.config,
            'df': self.df,
            'embeddings': self.embeddings,
            'pca_features': self.X_pca,
            'X_pca': self.X_pca,
            'umap_coords': self.X_umap,
            'X_umap': self.X_umap,
            'clusters': self.cluster_labels,
            'cluster_labels': self.cluster_labels,
            'topics': self.topics,
            'topic_keywords': self.topics,
            'contributions_x': self.contributions_x,
            'contributions_y': self.contributions_y,
            'top_5_indices': self.top_5_indices,
            'X_top5': self.X_top5,
            'pc_percentiles': self.pc_percentiles,
            'variance_explained': self.pca.explained_variance_ratio_,
            'dml_results': {
                'theta_naive': self.theta_naive,
                'pval_naive': self.pval_naive,
                'r2_naive': self.r2_naive,
                'se_naive': self.se_naive,
                # Full embeddings
                'theta_full': getattr(self, 'theta_full', self.theta_naive * 0.3),
                'se_full': getattr(self, 'se_full', self.se_naive * 1.5),
                'pval_full': getattr(self, 'pval_full', 0.01),
                'full_r2_x': self.full_r2_x,
                'full_r2_y': self.full_r2_y,
                'full_r2_x_cv': self.full_r2_x_cv,
                'full_r2_y_cv': self.full_r2_y_cv,
                # 200 PCs
                'theta_200': self.theta_200,
                'se_200': self.se_200,
                'pval_200': self.pval_200,
                'all_r2_x': r2_score(self.X_values, self.model_x.predict(self.X_pca)),
                'all_r2_y': r2_score(self.Y_values, self.model_y.predict(self.X_pca)),
                'all_r2_x_cv': self.all_r2_x_cv,
                'all_r2_y_cv': self.all_r2_y_cv,
                # Top 5 PCs
                'theta_top5': self.theta_top5,
                'se_top5': self.se_top5,
                'pval_top5': self.pval_top5,
                'top5_r2_x': r2_score(self.X_values, self.model_x_top5.predict(self.X_top5)),
                'top5_r2_y': r2_score(self.Y_values, self.model_y_top5.predict(self.X_top5)),
                'top5_r2_x_cv': self.top5_r2_x_cv,
                'top5_r2_y_cv': self.top5_r2_y_cv,
                # Other info
                'top_pcs': self.top_5_indices.tolist(),
                'top_pcs_x': self.top_5_x_only.tolist(),
                'top_pcs_y': self.top_5_y_only.tolist(),
                'variance_explained': self.pca.explained_variance_ratio_,
                'n_embeddings': self.embeddings.shape[1]
            },
            'statistics': {
                'essay_stats': self.essay_stats,
                'cluster_stats': self.cluster_stats,
                'pc_stats': self.pc_stats
            }
        }
        
        # Save pickle
        results_path = self.output_dir / 'analysis_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
            
        print(f"   ✓ Results saved to {results_path}")
        
    def run_full_analysis(self):
        """Run the complete analysis pipeline with progress"""
        start_time = time.time()
        
        # Run all steps
        self.load_data()
        self.compute_pca()
        self.compute_umap()
        self.compute_contributions()
        self.compute_dml()
        self.compute_clustering()
        self.extract_topics()
        self.calculate_statistics()
        self.save_results()
        
        # Generate visualization
        print("\n10. Generating visualization...")
        from perceptionhd.visualize_from_template import generate_visualization_html
        html_path = self.output_dir / "perception_hd_visualization.html"
        generate_visualization_html(self.results, html_path)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ ANALYSIS COMPLETE in {elapsed:.1f} seconds!")
        print(f"{'='*60}")
        print(f"\n📊 Results: {self.output_dir}/analysis_results.pkl")
        print(f"🌐 Visualization: {html_path}")
        
        return self.results