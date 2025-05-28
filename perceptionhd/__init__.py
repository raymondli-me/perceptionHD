"""
PerceptionHD - High-dimensional perception analysis and visualization

A comprehensive tool for analyzing relationships between text data and outcome variables
using embeddings, UMAP projections, and Double Machine Learning.
"""

__version__ = "0.1.0"

from .pipeline import PerceptionHDPipeline
from .visualize import generate_visualization

__all__ = ['PerceptionHDPipeline', 'generate_visualization']