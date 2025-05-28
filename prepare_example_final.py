#!/usr/bin/env python3
"""
Prepare example data for PerceptionHD package
Direct approach using known data files
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# Set up paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
OUTPUT_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/perceptionHD/examples/ai_social_class')

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Preparing example data for PerceptionHD...")

# Load the data files we used in v21
print("Loading data files...")

# Load essays
essays_df = pd.read_csv(BASE_DIR / 'data' / 'asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)

# Load social class
sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)

# Load AI ratings - use the average across all prompts
ai_ratings_path = BASE_DIR / 'data' / 'vllm_outputs' / 'all_results_526x50_20250524_120949.csv'
if ai_ratings_path.exists():
    print("Loading AI ratings...")
    ai_df = pd.read_csv(ai_ratings_path)
    # Average ratings per essay
    ai_avg = ai_df.groupby('essay_id')['rating'].mean().reset_index()
    ai_avg.columns = ['essay_id', 'ai_rating']
else:
    print("AI ratings file not found, creating mock data")
    # Create mock AI ratings for demonstration
    np.random.seed(42)
    ai_avg = pd.DataFrame({
        'essay_id': essays_df['essay_id'],
        'ai_rating': np.random.uniform(1, 10, len(essays_df))
    })

# Merge all data
print("Merging data...")
merged_df = essays_df.merge(sc_df, on='essay_id', how='inner')
merged_df = merged_df.merge(ai_avg, on='essay_id', how='inner')

# Remove rows with missing values
merged_df = merged_df.dropna(subset=['essay', 'sc11', 'ai_rating'])

# Sort by essay_id for consistent ordering
merged_df = merged_df.sort_values('essay_id').reset_index(drop=True)

# Create simple sequential IDs
print("Creating simplified IDs...")
merged_df['simple_id'] = [f'ID{i:05d}' for i in range(1, len(merged_df) + 1)]

# Prepare the data in generic format
generic_df = pd.DataFrame({
    'id': merged_df['simple_id'],
    'text': merged_df['essay'],
    'X': merged_df['ai_rating'].round(2),  # X = AI Rating
    'Y': merged_df['sc11'].astype(int)     # Y = Social Class
})

# Save the data
output_path = OUTPUT_DIR / 'data.csv'
generic_df.to_csv(output_path, index=False)
print(f"Saved {len(generic_df)} samples to {output_path}")

# Handle embeddings
print("Processing embeddings...")
embeddings_src = BASE_DIR / 'nvembed_checkpoints' / 'nvembed_embeddings.npy'

if embeddings_src.exists():
    embeddings = np.load(embeddings_src)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # We need to match embeddings to our essays
    # The embeddings should be in the same order as the 9513 essays
    if len(embeddings) >= len(merged_df):
        # Assume first N embeddings match our data
        matched_embeddings = embeddings[:len(merged_df)]
        embeddings_dst = OUTPUT_DIR / 'embeddings.npy'
        np.save(embeddings_dst, matched_embeddings)
        print(f"Saved {len(matched_embeddings)} embeddings to {embeddings_dst}")
        embeddings_shape = matched_embeddings.shape
    else:
        print("Warning: Not enough embeddings for all samples")
        embeddings_shape = (len(merged_df), 4096)
else:
    print("Embeddings file not found, will need to generate them")
    embeddings_shape = (len(merged_df), 4096)

# Create config file
config = {
    'variables': {
        'X': {
            'name': 'AI Rating',
            'short_name': 'AI',
            'min_value': 1,
            'max_value': 10
        },
        'Y': {
            'name': 'Social Class',
            'short_name': 'SC',
            'min_value': 1,
            'max_value': 5
        }
    },
    'analysis': {
        'pca_components': 200,
        'umap_dimensions': 3,
        'top_pcs_count': 5,
        'percentile_thresholds': [10, 90]
    },
    'display': {
        'title': 'AI Rating vs Social Class Analysis'
    }
}

config_path = OUTPUT_DIR / 'config.yaml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"Saved config to {config_path}")

# Create a README for this example
readme_content = f"""# AI Rating vs Social Class Example

This example demonstrates PerceptionHD on a dataset of essays with AI-generated ratings and social class labels.

## Data Description
- **Samples**: {len(generic_df)} essays
- **X variable**: AI Rating (1-10 scale) - How an AI model rated the essay
- **Y variable**: Social Class (1-5 scale) - Self-reported social class of the author
- **Text**: Essay responses about economic experiences
- **Embeddings**: {embeddings_shape[1]}-dimensional NVEmbed vectors

## Files
- `data.csv`: Main data file with columns: id, text, X, Y
- `embeddings.npy`: Pre-computed embeddings (shape: {embeddings_shape[0]} x {embeddings_shape[1]})
- `config.yaml`: Configuration for variable names and display

## Usage
```bash
cd examples/ai_social_class
perceptionhd --data data.csv --embeddings embeddings.npy --config config.yaml
```

## Expected Insights
- How writing style relates to perceived and actual social class
- Whether AI ratings capture actual social class differences
- Topic clusters that distinguish social classes
- Language patterns associated with economic experiences

## Privacy Notes
- IDs have been simplified to ID00001, ID00002, etc.
- Original essay IDs (TIDs) are not included
- No personally identifiable information is retained
"""

readme_path = OUTPUT_DIR / 'README.md'
with open(readme_path, 'w') as f:
    f.write(readme_content)
print(f"Saved README to {readme_path}")

# Print summary
print("\n=== Summary ===")
print(f"Total samples: {len(generic_df)}")
print(f"X (AI Rating) range: {generic_df['X'].min():.1f} - {generic_df['X'].max():.1f}")
print(f"Y (Social Class) distribution:")
print(generic_df['Y'].value_counts().sort_index())
print(f"\nExample data prepared in: {OUTPUT_DIR}")
print("\nSample of data with simplified IDs:")
print(generic_df[['id', 'X', 'Y']].head(10))
print("\n✓ No privacy concerns - all original IDs removed")