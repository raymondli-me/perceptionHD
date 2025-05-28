# PerceptionHD

High-dimensional perception analysis and visualization tool for understanding relationships between text data and outcome variables.

## Features

- **3D UMAP Visualization**: Interactive exploration of high-dimensional embeddings
- **Double Machine Learning**: Causal analysis with text controls
- **Topic Discovery**: HDBSCAN clustering with keyword extraction
- **Principal Component Analysis**: Identify text features that predict outcomes
- **Rich Interactivity**: Gallery mode, filtering, and detailed statistics

## Installation

```bash
pip install perceptionhd
```

## Quick Start

1. Prepare your data with columns: `id`, `text`, `X`, `Y`
2. Generate embeddings (e.g., using sentence-transformers)
3. Create a config file specifying variable names
4. Run: `perceptionhd --data data.csv --embeddings embeddings.npy --config config.yaml`

## Example

See the `examples/ai_social_class` directory for a complete example analyzing the relationship between AI ratings and social class in essay data.

## Documentation

Full documentation available at: https://perceptionhd.readthedocs.io

## License

MIT License
