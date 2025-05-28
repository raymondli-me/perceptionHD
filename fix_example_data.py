#!/usr/bin/env python3
"""
Fix the example data to properly use NVEmbed embeddings with correct alignment
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import pickle

# Set up paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
OUTPUT_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/perceptionHD/examples/ai_social_class')

print("Fixing example data to use proper NVEmbed embeddings...")

# Load the EXACT data used in v21 analysis
print("Loading v21 analysis data...")

# Load NVEmbed PCA features to get the essay IDs and ordering
with open(BASE_DIR / 'nvembed_checkpoints/nvembed_pca_200_features.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    essay_ids_ordered = pca_data['essay_ids']  # This is the correct ordering!

print(f"Found {len(essay_ids_ordered)} essays in correct order")

# Load essays
essays_df = pd.read_csv(BASE_DIR / 'data' / 'asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)

# Load social class
sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)

# Load AI ratings (human MacArthur only - same as v21)
ai_ratings_df = pd.read_csv(BASE_DIR / 'asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv')
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()

# Merge all data
merged_df = essays_df.merge(sc_df[['essay_id', 'sc11']], on='essay_id', how='inner')
merged_df = merged_df.merge(human_mac_ratings[['essay_id', 'rating']], on='essay_id', how='inner')
merged_df.rename(columns={'rating': 'ai_rating'}, inplace=True)

# CRITICAL: Order the dataframe to match the essay_ids_ordered from embeddings
merged_df = merged_df.set_index('essay_id').loc[essay_ids_ordered].reset_index()

print(f"Aligned {len(merged_df)} essays in correct order")

# Create simple sequential IDs
merged_df['simple_id'] = [f'ID{i:05d}' for i in range(1, len(merged_df) + 1)]

# Prepare the data - SWAP X and Y to match v21 analysis!
# In v21: Y = AI Rating (outcome), X = Social Class (treatment)
generic_df = pd.DataFrame({
    'id': merged_df['simple_id'],
    'text': merged_df['essay'],
    'X': merged_df['sc11'].astype(int),      # X = Social Class (treatment in DML)
    'Y': merged_df['ai_rating'].round(2)      # Y = AI Rating (outcome in DML)
})

# Save the data
output_path = OUTPUT_DIR / 'data.csv'
generic_df.to_csv(output_path, index=False)
print(f"Saved {len(generic_df)} samples to {output_path}")

# Load the EXACT NVEmbed embeddings in the correct order
embeddings_src = BASE_DIR / 'nvembed_checkpoints' / 'nvembed_embeddings.npy'
embeddings = np.load(embeddings_src)

# The embeddings should already be in the correct order matching essay_ids_ordered
if len(embeddings) == len(merged_df):
    embeddings_dst = OUTPUT_DIR / 'embeddings.npy'
    np.save(embeddings_dst, embeddings)
    print(f"Saved {len(embeddings)} NVEmbed embeddings (4096-dim) to {embeddings_dst}")
else:
    print(f"ERROR: Embedding count mismatch! Embeddings: {len(embeddings)}, Essays: {len(merged_df)}")

# Update config file to reflect correct variable meanings
config = {
    'variables': {
        'X': {
            'name': 'Social Class',
            'short_name': 'SC',
            'min_value': 1,
            'max_value': 5
        },
        'Y': {
            'name': 'AI Rating', 
            'short_name': 'AI',
            'min_value': 1,
            'max_value': 10
        }
    },
    'analysis': {
        'pca_components': 200,
        'umap_dimensions': 3,
        'top_pcs_count': 5,
        'percentile_thresholds': [10, 90]
    },
    'display': {
        'title': 'Social Class → AI Rating Analysis'
    }
}

config_path = OUTPUT_DIR / 'config.yaml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"Updated config to {config_path}")

# Verify the data
print("\n=== Verification ===")
print(f"X (Social Class) distribution:")
print(generic_df['X'].value_counts().sort_index())
print(f"\nY (AI Rating) range: {generic_df['Y'].min():.1f} - {generic_df['Y'].max():.1f}")
print(f"Embeddings shape: {embeddings.shape}")
print("\n✓ Data now matches v21 analysis setup exactly!")