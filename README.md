# PerceptionHD

A Python package for analyzing high-dimensional perception data using UMAP visualization and Double Machine Learning (DML) for causal inference.

## Features

- **Double Machine Learning (DML)** for causal inference between two variables with text controls
- **3D UMAP visualization** of high-dimensional embeddings with GPU-accelerated rendering
- **Principal Component Analysis (PCA)** for dimensionality reduction
- **XGBoost-based feature importance** for selecting relevant text features
- **HDBSCAN clustering** with automatic topic extraction
- **Interactive web visualization** with real-time filtering and detailed statistics

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/perceptionHD.git
cd perceptionHD

# Install dependencies
pip install -r requirements.txt

# Install the package
python setup.py install
```

## Quick Start

### Using the Example Dataset

The package includes a dataset of 9,513 essays with AI ratings and social class labels.

First, generate the embeddings:

```bash
# Generate embeddings for the example dataset
python -m perceptionhd.data_generator --sample-size 100  # For quick testing with 100 samples

# Or generate full embeddings (requires GPU with 8GB+ VRAM, takes ~30 minutes)
python -m perceptionhd.data_generator  # Full 9,513 samples
```

Then run the analysis:

```bash
# Run the analysis with progress tracking
python run_with_progress.py

# View the results
# Open: file:///path/to/perceptionHD/progress_output/perception_hd_visualization.html
```

### Using Your Own Data

1. **Prepare your data** with columns: `id`, `text`, `X`, `Y`
   - `id`: Unique identifier for each sample
   - `text`: Text content to analyze
   - `X`: Independent variable (e.g., demographic variable)
   - `Y`: Dependent variable (e.g., outcome measure)

2. **Generate embeddings** using NVIDIA NV-Embed-v2:

```python
from perceptionhd.data_generator import prepare_perceptionhd_data

# Generate embeddings for your data
prepare_perceptionhd_data(
    data_path='your_data.csv',
    output_dir='output/your_analysis',
    use_gpu=True,  # Recommended for speed
    sample_size=None  # Use all data
)
```

Or use the command line:

```bash
python -m perceptionhd.data_generator \
    --data-path your_data.csv \
    --output-dir output/your_analysis \
    --sample-size 1000  # Optional: limit samples for testing
```

3. **Create a config file** (`config.yaml`):

```yaml
data:
  essays_path: data.csv
  embeddings_path: embeddings.npy

variables:
  X:
    name: "Your X Variable"
    short_name: "X"
    min_value: 1
    max_value: 10
  Y:
    name: "Your Y Variable"
    short_name: "Y"
    min_value: 1.0
    max_value: 10.0

analysis:
  pca_components: 200
  top_pcs: 5
  umap_dimensions: 3

display:
  title: "Your Analysis Title"
```

4. **Run the analysis**:

```python
from perceptionhd.pipeline_with_progress import PerceptionHDPipelineWithProgress

pipeline = PerceptionHDPipelineWithProgress(
    data_path='output/your_analysis/data.csv',
    embeddings_path='output/your_analysis/embeddings.npy',
    config_path='output/your_analysis/config.yaml',
    output_dir='output/your_analysis'
)

results = pipeline.run_full_analysis()
```

## Example Dataset

The included example analyzes essays from 9,513 participants:
- **X Variable**: Social class (1-5 scale)
- **Y Variable**: AI-generated rating of essay quality (1-10 scale)
- **Text**: Personal essays about life at age 25
- **Embeddings**: 4096-dimensional vectors from NVIDIA NV-Embed-v2

This dataset demonstrates how text mediates the relationship between social background and AI assessments.

## Embedding Generation

The package uses NVIDIA's NV-Embed-v2 model for generating high-quality text embeddings:

```python
from perceptionhd.data_generator import generate_nvembed_embeddings

# Generate embeddings for a list of texts
embeddings = generate_nvembed_embeddings(
    texts=['text1', 'text2', ...],
    batch_size=8,
    device='cuda'  # or 'cpu'
)
```

**Note**: GPU is strongly recommended for embedding generation. The model requires ~8GB VRAM.

## Output

The analysis produces:
1. **Interactive 3D Visualization** (`perception_hd_visualization.html`)
   - 3D UMAP projection of text embeddings
   - Color coding by X/Y variables or clusters
   - Interactive filtering and selection
   - Topic keywords for each cluster

2. **DML Analysis Results**
   - Causal effect estimates with 4 models:
     - Naive (no text controls)
     - Full embeddings (4096 dimensions)
     - PCA reduced (200 components)
     - Top 5 PCs (selected by XGBoost)
   - R² values showing prediction accuracy
   - Effect reduction showing mediation by text

3. **Saved Results** (`analysis_results.pkl`)
   - All computed features and models
   - Can be loaded for further analysis

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for embeddings)
- 16GB RAM minimum
- Modern web browser for visualization

## Citation

If you use PerceptionHD in your research, please cite:

```bibtex
@software{perceptionhd2024,
  title={PerceptionHD: High-Dimensional Perception Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/YOUR_USERNAME/perceptionHD}
}
```

## License

MIT License - see LICENSE file for details.