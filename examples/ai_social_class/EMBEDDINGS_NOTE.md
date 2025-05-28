# Embeddings File

The `embeddings.npy` file (155MB) is too large for GitHub and is not included in the repository.

## Download Link
The embeddings file can be downloaded from: [To be added]

## Alternative: Generate Your Own
You can generate embeddings using any model. Here's an example:

```python
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Load a model
model = SentenceTransformer('all-mpnet-base-v2')  # 768 dims
# or
# model = SentenceTransformer('nvidia/NV-Embed-v2')  # 4096 dims (original)

# Generate embeddings
embeddings = model.encode(data['text'].tolist(), show_progress_bar=True)

# Save
np.save('embeddings.npy', embeddings)
```

## File Details
- Shape: (9513, 4096)
- Model: NVEmbed v2
- Format: NumPy array, float32