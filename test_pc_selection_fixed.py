#!/usr/bin/env python3
"""
Quick test to verify PC selection with fixed data
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from pathlib import Path

# Load the fixed example data
example_dir = Path("examples/ai_social_class")
df = pd.read_csv(example_dir / "data.csv")
embeddings = np.load(example_dir / "embeddings.npy")

print(f"Loaded {len(df)} samples with {embeddings.shape[1]}-dim embeddings")
print(f"X (Social Class) range: {df['X'].min()}-{df['X'].max()}")
print(f"Y (AI Rating) range: {df['Y'].min():.1f}-{df['Y'].max():.1f}")

# Compute PCA
print("\nComputing PCA...")
pca = PCA(n_components=200, random_state=42)
X_pca = pca.fit_transform(embeddings)
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# Train XGBoost models
print("\nTraining XGBoost models...")
X_values = df['X'].values  # Social Class
Y_values = df['Y'].values  # AI Rating

# Model for X (Social Class) prediction
model_x = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model_x.fit(X_pca, X_values)
importance_x = model_x.feature_importances_

# Model for Y (AI Rating) prediction  
model_y = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model_y.fit(X_pca, Y_values)
importance_y = model_y.feature_importances_

# Combined importance (average)
combined_importance = (importance_x + importance_y) / 2

# Get top 5 PCs
top_5_indices = np.argsort(combined_importance)[-5:][::-1]
top_5_x = np.argsort(importance_x)[-5:][::-1]
top_5_y = np.argsort(importance_y)[-5:][::-1]

print(f"\nResults:")
print(f"Top 5 PCs for X (Social Class): {top_5_x.tolist()}")
print(f"Top 5 PCs for Y (AI Rating): {top_5_y.tolist()}")
print(f"Top 5 PCs combined: {top_5_indices.tolist()}")

print(f"\nDetailed importance for combined top 5:")
for pc in top_5_indices:
    print(f"  PC{pc}: Combined={combined_importance[pc]:.4f} (X={importance_x[pc]:.4f}, Y={importance_y[pc]:.4f})")

# Compare with expected v21 values
v21_pcs = [0, 2, 5, 13, 46]
print(f"\nExpected v21 PCs {v21_pcs}:")
for pc in v21_pcs:
    print(f"  PC{pc}: Combined={combined_importance[pc]:.4f} (X={importance_x[pc]:.4f}, Y={importance_y[pc]:.4f})")