#!/usr/bin/env python3
"""
Quick test to see which PCs get selected using XGBoost feature importance
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import pickle

# Load the same data
data_dir = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
nvembed_dir = data_dir / "nvembed_checkpoints"

# Load PCA features
print("Loading PCA features...")
with open(nvembed_dir / "nvembed_pca_200_features.pkl", 'rb') as f:
    pca_data = pickle.load(f)
    X_pca = pca_data['features']
    essay_ids = pca_data['essay_ids']

# Load essays and ratings
essays_df = pd.read_csv(data_dir / "data" / "asc_9513_essays.csv")
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)

# Load social class
sc_df = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)
essays_df = essays_df.merge(sc_df[['essay_id', 'sc11']], on='essay_id', how='left')

# Load AI ratings (human MacArthur only)
ai_ratings_df = pd.read_csv(data_dir / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv")
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()
essays_df = essays_df.merge(human_mac_ratings[['essay_id', 'rating']], on='essay_id', how='left')
essays_df.rename(columns={'rating': 'ai_rating'}, inplace=True)

# Align data
essays_df = essays_df[essays_df['essay_id'].isin(essay_ids)]
essays_df = essays_df.set_index('essay_id').loc[essay_ids].reset_index()

Y_ai = essays_df['ai_rating'].values
Y_sc = essays_df['sc11'].values

print(f"Data loaded: {X_pca.shape[0]} essays, {X_pca.shape[1]} PCs")

# Train XGBoost models with original parameters
print("\nTraining XGBoost models...")

# Model for AI rating prediction
model_ai = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model_ai.fit(X_pca, Y_ai)
importance_ai = model_ai.feature_importances_

# Model for social class prediction  
model_sc = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model_sc.fit(X_pca, Y_sc)
importance_sc = model_sc.feature_importances_

# Combined importance (average)
combined_importance = (importance_ai + importance_sc) / 2

# Get top 5 PCs by combined importance
top_5_indices = np.argsort(combined_importance)[-5:][::-1]
top_5_ai = np.argsort(importance_ai)[-5:][::-1]
top_5_sc = np.argsort(importance_sc)[-5:][::-1]

print(f"\nResults:")
print(f"Top 5 PCs for AI rating: {top_5_ai.tolist()}")
print(f"Top 5 PCs for Social Class: {top_5_sc.tolist()}")
print(f"Top 5 PCs combined (average importance): {top_5_indices.tolist()}")

print(f"\nDetailed importance for combined top 5:")
for pc in top_5_indices:
    print(f"  PC{pc}: Combined={combined_importance[pc]:.4f} (AI={importance_ai[pc]:.4f}, SC={importance_sc[pc]:.4f})")

# Compare with v21 hardcoded values
v21_pcs = [0, 2, 5, 13, 46]
print(f"\nComparison with v21 hardcoded PCs {v21_pcs}:")
for pc in v21_pcs:
    print(f"  PC{pc}: Combined={combined_importance[pc]:.4f} (AI={importance_ai[pc]:.4f}, SC={importance_sc[pc]:.4f})")