#!/usr/bin/env python3
"""
Generate sample data and embeddings for PerceptionHD
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc


def load_sample_data(data_path=None):
    """
    Load or create sample data with X, Y, id, and text columns
    
    Args:
        data_path: Path to existing data CSV (optional)
        
    Returns:
        pd.DataFrame: DataFrame with columns ['id', 'text', 'X', 'Y']
    """
    print("Loading sample data...")
    
    if data_path and Path(data_path).exists():
        # Load from provided path
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples from {data_path}")
        
        # Ensure required columns exist
        required_cols = ['id', 'text', 'X', 'Y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    
    # Otherwise load the AI/Social Class dataset
    default_path = Path(__file__).parent.parent / 'examples' / 'ai_social_class' / 'data.csv'
    if default_path.exists():
        df = pd.read_csv(default_path)
        print(f"Loaded {len(df)} samples from default dataset")
        return df
    
    # Fallback: Create minimal sample data for testing
    print("Creating sample data for testing...")
    sample_data = {
        'id': [f'essay_{i}' for i in range(100)],
        'text': [
            f"This is a sample essay {i} discussing various topics..."
            for i in range(100)
        ],
        'X': np.random.randint(1, 6, 100),  # e.g., social class 1-5
        'Y': np.random.uniform(1, 10, 100),  # e.g., AI rating 1-10
    }
    
    df = pd.DataFrame(sample_data)
    return df


def generate_nvembed_embeddings(texts, batch_size=8, device='cuda'):
    """
    Generate embeddings using NVIDIA NV-Embed-v2 model
    
    Args:
        texts: List of text strings
        batch_size: Batch size for processing
        device: 'cuda' or 'cpu'
        
    Returns:
        np.ndarray: Embeddings array of shape (n_texts, 4096)
    """
    print(f"Generating NV-Embed-v2 embeddings on {device}...")
    
    # Load the model
    model_name = "nvidia/NV-Embed-v2"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    )
    
    # Add instruction prefix as used in the original analysis
    instruction = "Represent this text for clustering and classification: "
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Process in batches
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Add instruction to each text
        batch_texts_with_instruction = [instruction + text for text in batch_texts]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts_with_instruction,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
            # NV-Embed-v2 uses mean pooling over the sequence
            # Get the last hidden states
            hidden_states = outputs.last_hidden_state
            
            # Apply mean pooling (accounting for padding)
            attention_mask = inputs['attention_mask']
            masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            sum_hidden_states = masked_hidden_states.sum(dim=1)
            sum_attention_mask = attention_mask.sum(dim=1, keepdim=True)
            mean_pooled = sum_hidden_states / sum_attention_mask
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            
            # Move to CPU and convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)
        
        # Clear GPU memory
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    embeddings_array = np.vstack(all_embeddings)
    
    print(f"Generated embeddings shape: {embeddings_array.shape}")
    return embeddings_array


def prepare_perceptionhd_data(output_dir='examples/ai_social_class', 
                            use_gpu=True,
                            sample_size=None,
                            data_path=None):
    """
    Prepare data and embeddings for PerceptionHD analysis
    
    Args:
        output_dir: Directory to save the data
        use_gpu: Whether to use GPU for embedding generation
        sample_size: If provided, only process this many samples (for testing)
        data_path: Path to existing data CSV (optional)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_sample_data(data_path)
    
    # Optionally limit size for testing
    if sample_size:
        df = df.head(sample_size)
        print(f"Using {sample_size} samples for testing")
    
    # Save data CSV
    data_path = output_path / 'data.csv'
    df.to_csv(data_path, index=False)
    print(f"Saved data to {data_path}")
    
    # Generate embeddings
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu' and len(df) > 1000:
        print("WARNING: Using CPU for large dataset will be slow!")
        print("Consider using GPU or reducing sample_size")
    
    embeddings = generate_nvembed_embeddings(
        df['text'].tolist(),
        batch_size=8 if device == 'cuda' else 2,
        device=device
    )
    
    # Save embeddings
    embeddings_path = output_path / 'embeddings.npy'
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")
    
    # Create config file
    config = {
        'data': {
            'essays_path': 'data.csv',
            'embeddings_path': 'embeddings.npy'
        },
        'variables': {
            'X': {
                'name': 'X Variable',
                'short_name': 'X',
                'min_value': int(df['X'].min()),
                'max_value': int(df['X'].max())
            },
            'Y': {
                'name': 'Y Variable', 
                'short_name': 'Y',
                'min_value': float(df['Y'].min()),
                'max_value': float(df['Y'].max())
            }
        },
        'analysis': {
            'pca_components': 200,
            'top_pcs': 5,
            'umap_dimensions': 3
        },
        'display': {
            'title': 'X vs Y Analysis'
        }
    }
    
    # Save config
    import yaml
    config_path = output_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {config_path}")
    
    print("\nData preparation complete!")
    print(f"Files created in {output_path}:")
    print(f"  - data.csv ({len(df)} samples)")
    print(f"  - embeddings.npy ({embeddings.shape})")
    print(f"  - config.yaml")
    
    return df, embeddings


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate data and embeddings for PerceptionHD')
    parser.add_argument('--output-dir', default='examples/ai_social_class',
                       help='Output directory for data files')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of samples to process (default: all)')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to existing data CSV with id, text, X, Y columns')
    
    args = parser.parse_args()
    
    prepare_perceptionhd_data(
        output_dir=args.output_dir,
        use_gpu=not args.cpu,
        sample_size=args.sample_size,
        data_path=args.data_path
    )