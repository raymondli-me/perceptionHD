#!/usr/bin/env python3
"""
Set up the PerceptionHD package structure
"""

import os
from pathlib import Path

# Create package directory structure
base_dir = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/perceptionHD')

# Create directories
dirs = [
    'perceptionhd',
    'perceptionhd/templates',
    'examples',
    'examples/ai_social_class',
    'tests',
    'docs'
]

for d in dirs:
    (base_dir / d).mkdir(parents=True, exist_ok=True)
    
# Create __init__.py files
init_files = [
    'perceptionhd/__init__.py',
    'tests/__init__.py'
]

for f in init_files:
    (base_dir / f).touch()
    
# Create setup.py
setup_content = '''from setuptools import setup, find_packages

setup(
    name="perceptionHD",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="High-dimensional perception analysis and visualization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/perceptionHD",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "umap-learn>=0.5.0",
        "xgboost>=1.5.0",
        "hdbscan>=0.8.0",
        "pyyaml>=5.4.0",
        "jinja2>=3.0.0",
        "tqdm>=4.62.0",
        "doubleml>=0.5.0",
    ],
    entry_points={
        "console_scripts": [
            "perceptionhd=perceptionhd.cli:main",
        ],
    },
    package_data={
        "perceptionhd": ["templates/*.html"],
    },
)
'''

with open(base_dir / 'setup.py', 'w') as f:
    f.write(setup_content)

# Create README.md
readme_content = '''# PerceptionHD

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
'''

with open(base_dir / 'README.md', 'w') as f:
    f.write(readme_content)

# Create requirements.txt
requirements = '''numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
umap-learn>=0.5.0
xgboost>=1.5.0
hdbscan>=0.8.0
pyyaml>=5.4.0
jinja2>=3.0.0
tqdm>=4.62.0
doubleml>=0.5.0
sentence-transformers>=2.0.0
'''

with open(base_dir / 'requirements.txt', 'w') as f:
    f.write(requirements)

# Create .gitignore
gitignore = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data files
*.npy
*.pkl
*.csv
!examples/*/data.csv
!examples/*/config.yaml

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Documentation
docs/_build/
'''

with open(base_dir / '.gitignore', 'w') as f:
    f.write(gitignore)

print("✓ Created package structure")
print("\nDirectory structure:")
for root, dirs, files in os.walk(base_dir):
    level = root.replace(str(base_dir), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        if not file.startswith('.'):
            print(f'{subindent}{file}')