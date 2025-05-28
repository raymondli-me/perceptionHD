#!/usr/bin/env python3
"""
Prepare example data for PerceptionHD package
Converts the AI/SC dataset to generic format with simple IDs
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
OUTPUT_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/perceptionHD/examples/ai_social_class')

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Preparing example data for PerceptionHD...")

# Load the original data
print("Loading original data...")
essays_df = pd.read_csv(BASE_DIR / 'data' / 'asc_9513_essays.csv')

# Also load the AI ratings and social class data
ai_ratings_df = pd.read_csv(BASE_DIR / 'data' / 'vllm_outputs' / 'all_results_526x50_20250524_120949.csv')
sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')

# Rename columns
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)

# Get average AI rating per essay
ai_avg = ai_ratings_df.groupby('essay_id')['rating'].mean().reset_index()
ai_avg.columns = ['essay_id', 'ai_rating']

# Merge all data
merged_df = essays_df.merge(ai_avg, on='essay_id', how='inner')
merged_df = merged_df.merge(sc_df, on='essay_id', how='inner')

# Create simple sequential IDs
print("Creating simplified IDs...")
merged_df = merged_df.sort_values('essay_id').reset_index(drop=True)
merged_df['simple_id'] = [f'ID{i:05d}' for i in range(1, len(merged_df) + 1)]

# Prepare the data in generic format
generic_df = pd.DataFrame({
    'id': merged_df['simple_id'],
    'text': merged_df['essay'],
    'X': merged_df['ai_rating'],  # X = AI Rating
    'Y': merged_df['sc11']        # Y = Social Class
})

# Remove any rows with missing values
original_count = len(generic_df)
generic_df = generic_df.dropna()
final_count = len(generic_df)
print(f"Removed {original_count - final_count} rows with missing values")

# Save the data
output_path = OUTPUT_DIR / 'data.csv'
generic_df.to_csv(output_path, index=False)
print(f"Saved {final_count} samples to {output_path}")

# Handle embeddings
print("Processing embeddings...")
embeddings_src = BASE_DIR / 'nvembed_checkpoints' / 'nvembed_embeddings.npy'
embeddings_dst = OUTPUT_DIR / 'embeddings.npy'

if embeddings_src.exists():
    embeddings = np.load(embeddings_src)
    essay_ids_src = BASE_DIR / 'nvembed_checkpoints' / 'nvembed_essay_ids.npy'
    
    if essay_ids_src.exists():
        essay_ids = np.load(essay_ids_src)
        print(f"Loaded embeddings for {len(essay_ids)} essays")
        
        # Create mapping from essay_id to embedding index
        embedding_map = {eid: i for i, eid in enumerate(essay_ids)}
        
        # Get embeddings in the same order as our data
        ordered_embeddings = []
        for _, row in merged_df.iterrows():
            essay_id = row['essay_id']
            if essay_id in embedding_map:
                ordered_embeddings.append(embeddings[embedding_map[essay_id]])
            else:
                print(f"Warning: No embedding found for {essay_id}")
                ordered_embeddings.append(np.zeros(embeddings.shape[1]))
        
        ordered_embeddings = np.array(ordered_embeddings)
        
        # Filter to match the cleaned data
        kept_indices = merged_df.index[~generic_df.index.isin(merged_df.index[merged_df['ai_rating'].isna() | merged_df['sc11'].isna()])].tolist()
        filtered_embeddings = ordered_embeddings[kept_indices]
        
        np.save(embeddings_dst, filtered_embeddings)
        print(f"Saved embeddings with shape: {filtered_embeddings.shape}")
    else:
        print(f"Warning: Essay IDs not found at {essay_ids_src}")
else:
    print(f"Warning: Embeddings not found at {embeddings_src}")

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

import yaml
config_path = OUTPUT_DIR / 'config.yaml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"Saved config to {config_path}")

# Create a README for this example
readme_content = """# AI Rating vs Social Class Example

This example demonstrates PerceptionHD on a dataset of essays with AI-generated ratings and social class labels.

## Data Description
- **Samples**: {} essays
- **X variable**: AI Rating (1-10 scale) - How an AI model rated the essay
- **Y variable**: Social Class (1-5 scale) - Self-reported social class of the author
- **Text**: Essay responses about economic experiences
- **Embeddings**: 4096-dimensional NVEmbed vectors

## Files
- `data.csv`: Main data file with columns: id, text, X, Y
- `embeddings.npy`: Pre-computed embeddings (shape: {} x {})
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

## Notes
- IDs have been simplified to ID00001, ID00002, etc. for privacy
- Original essay IDs are not included in the public dataset
""".format(final_count, filtered_embeddings.shape[0] if 'filtered_embeddings' in locals() else "N/A", filtered_embeddings.shape[1] if 'filtered_embeddings' in locals() else "N/A")

readme_path = OUTPUT_DIR / 'README.md'
with open(readme_path, 'w') as f:
    f.write(readme_content)
print(f"Saved README to {readme_path}")

# Print summary
print("\n=== Summary ===")
print(f"Total samples: {final_count}")
print(f"X (AI Rating) range: {generic_df['X'].min():.1f} - {generic_df['X'].max():.1f}")
print(f"Y (Social Class) distribution:")
print(generic_df['Y'].value_counts().sort_index())
print(f"\nExample data prepared in: {OUTPUT_DIR}")
print("\nSample of simplified IDs:")
print(generic_df[['id', 'X', 'Y']].head(10))
print("\nNo privacy concerns with simplified IDs - original TIDs removed")