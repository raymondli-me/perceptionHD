# AI Rating vs Social Class Example

This example demonstrates PerceptionHD on a dataset of essays with AI-generated ratings and social class labels.

## Data Description
- **Samples**: 9513 essays  
- **X variable**: AI Rating (1-10 scale) - Simulated AI model ratings
- **Y variable**: Social Class (1-5 scale) - Simulated social class distribution
- **Text**: Essay responses about life at age 25
- **Embeddings**: 4096-dimensional NVEmbed vectors

## Files
- `data.csv`: Main data file with columns: id, text, X, Y
- `embeddings.npy`: Pre-computed embeddings (shape: 9513 x 4096)
- `config.yaml`: Configuration for variable names and display

## Usage
```bash
cd examples/ai_social_class
perceptionhd --data data.csv --embeddings embeddings.npy --config config.yaml
```

## Notes
- IDs have been anonymized (ID00001, ID00002, etc.)
- AI ratings are simulated for demonstration purposes
- Social class distribution matches typical UK patterns
- Original essay IDs (TIDs) have been removed

## Expected Insights
- Text patterns associated with different social classes
- How perceived social class (AI rating) relates to actual social class
- Topic clusters that emerge from life narratives
- Language features predictive of socioeconomic status
