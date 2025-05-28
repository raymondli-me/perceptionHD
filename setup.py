from setuptools import setup, find_packages

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
