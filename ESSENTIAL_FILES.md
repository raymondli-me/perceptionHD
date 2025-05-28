# PerceptionHD Essential Files

After archiving non-essential files, here are the core components:

## Package Structure

### Root Directory
- `LICENSE` - MIT license
- `README.md` - Package documentation
- `requirements.txt` - Python dependencies
- `setup.py` - Package installation script
- `run_with_progress.py` - Main entry point for running analysis

### Core Package (`perceptionhd/`)
- `__init__.py` - Package initialization
- `cli.py` - Command-line interface
- `core.py` - Core functionality
- `pipeline.py` - Basic pipeline
- `pipeline_with_progress.py` - Enhanced pipeline with progress tracking
- `visualize.py` - Basic visualization
- `visualize_v21_fully_generic.py` - v21 visualization with full X/Y genericization
- `templates/` - HTML templates (if any)

### Example Data (`examples/ai_social_class/`)
- `config.yaml` - Configuration for X/Y variables
- `data.csv` - Sample data with X, Y, id, and text columns
- `embeddings.npy` - Pre-computed NVEmbed embeddings (4096 dims)
- `README.md` - Example documentation
- `EMBEDDINGS_NOTE.md` - Notes about embeddings

### Output (`progress_output/`)
- `analysis_results.pkl` - Saved analysis results
- `perception_hd_visualization.html` - Generated visualization

### Tests (`tests/`)
- `__init__.py` - Test package initialization

### Documentation (`docs/`)
- Documentation files (if any)

## Key Features Implemented

1. **4-Model DML Structure**:
   - Naive model (no text)
   - Full embeddings (4096 dims)
   - 200 PCs
   - Top 5 PCs (selected via XGBoost feature importance)

2. **Proper R² Calculations**:
   - Non-crossfitted (left column): XGBoost trained on full data
   - Crossfitted (right column): 5-fold cross-validation

3. **Complete Genericization**:
   - All "AI Rating" → Y Variable
   - All "Social Class" → X Variable
   - Dynamic thresholds based on percentiles
   - No hardcoded values

4. **UI Fixes**:
   - Essay viewer: 350px left, 250px right
   - Z-index: 99999 for all panels
   - Proper hover and GPU picking

## Running the Analysis

```bash
cd /media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/perceptionHD
python3 run_with_progress.py
```

This will:
1. Load data from `examples/ai_social_class/`
2. Run the full analysis pipeline
3. Generate visualization at `progress_output/perception_hd_visualization.html`

## Archived Files

All non-essential files have been moved to the `archive/` directory, including:
- Test scripts
- Demo runners
- Preparation scripts
- Old visualization versions
- Temporary outputs
- Build artifacts