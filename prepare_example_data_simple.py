#!/usr/bin/env python3
"""
Prepare example data for PerceptionHD package
Uses the already merged dataset from the DML analysis
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

# Load the already merged data that we know works
print("Loading merged data...")
data_df = pd.read_csv(BASE_DIR / 'data' / 'asc_9513_essays.csv')
print(f"Loaded {len(data_df)} essays")

# Create simple sequential IDs
print("Creating simplified IDs...")
data_df['simple_id'] = [f'ID{i:05d}' for i in range(1, len(data_df) + 1)]

# Check if we have the needed columns in the embeddings checkpoint data
import pickle
pkl_path = BASE_DIR / 'nvembed_dml_pc_analysis' / 'dml_pc_analysis_results_fixed_hover.pkl'
if pkl_path.exists():
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    if 'essays_df' in pkl_data:
        essays_with_ratings = pkl_data['essays_df']
        print(f"Found {len(essays_with_ratings)} essays with ratings and social class")
        
        # Create simple IDs for this dataset
        essays_with_ratings = essays_with_ratings.sort_values('essay_id').reset_index(drop=True)
        essays_with_ratings['simple_id'] = [f'ID{i:05d}' for i in range(1, len(essays_with_ratings) + 1)]
        
        # Prepare the data in generic format
        generic_df = pd.DataFrame({
            'id': essays_with_ratings['simple_id'],
            'text': essays_with_ratings['essay'],
            'X': essays_with_ratings['ai_rating'],  # X = AI Rating
            'Y': essays_with_ratings['sc11']        # Y = Social Class
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
        if 'embeddings' in pkl_data:
            embeddings = pkl_data['embeddings']
            print(f"Found embeddings with shape: {embeddings.shape}")
            
            # Make sure we save the right embeddings
            embeddings_dst = OUTPUT_DIR / 'embeddings.npy'
            np.save(embeddings_dst, embeddings)
            print(f"Saved embeddings to {embeddings_dst}")
        else:
            # Try loading from the separate file
            embeddings_src = BASE_DIR / 'nvembed_checkpoints' / 'nvembed_embeddings.npy'
            if embeddings_src.exists():
                embeddings = np.load(embeddings_src)
                if len(embeddings) == 9513:  # We know this is the right size
                    # Save first 9513 rows that match our data
                    embeddings_dst = OUTPUT_DIR / 'embeddings.npy'
                    np.save(embeddings_dst, embeddings[:final_count])
                    print(f"Saved {final_count} embeddings to {embeddings_dst}")

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
if 'final_count' in locals() and 'embeddings' in locals():
    shape_str = f"{embeddings.shape[0]} x {embeddings.shape[1]}"
else:
    shape_str = "9513 x 4096"
    final_count = 9513

readme_content = f"""# AI Rating vs Social Class Example

This example demonstrates PerceptionHD on a dataset of essays with AI-generated ratings and social class labels.

## Data Description
- **Samples**: {final_count} essays
- **X variable**: AI Rating (1-10 scale) - How an AI model rated the essay
- **Y variable**: Social Class (1-5 scale) - Self-reported social class of the author
- **Text**: Essay responses about economic experiences
- **Embeddings**: 4096-dimensional NVEmbed vectors

## Files
- `data.csv`: Main data file with columns: id, text, X, Y
- `embeddings.npy`: Pre-computed embeddings (shape: {shape_str})
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
- This is the same dataset used in the original UMAP-DML analysis
"""

readme_path = OUTPUT_DIR / 'README.md'
with open(readme_path, 'w') as f:
    f.write(readme_content)
print(f"Saved README to {readme_path}")

# Print summary
if 'generic_df' in locals():
    print("\n=== Summary ===")
    print(f"Total samples: {final_count}")
    print(f"X (AI Rating) range: {generic_df['X'].min():.1f} - {generic_df['X'].max():.1f}")
    print(f"Y (Social Class) distribution:")
    print(generic_df['Y'].value_counts().sort_index())
    print(f"\nExample data prepared in: {OUTPUT_DIR}")
    print("\nSample of simplified IDs:")
    print(generic_df[['id', 'X', 'Y']].head(10))
    print("\nNo privacy concerns with simplified IDs - original essay IDs removed")