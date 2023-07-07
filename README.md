# ppscore-filter
A small wrapper for the ppscore library that extends its functionality to arrays. PPScore allows for the detection of linear AND nonlinear relationships, so this allows for quick and robust feature selection

# Usage
```python
from ppfilter import get_optimal_features, reduce_features
import numpy as np



arr = np.random.rand(200,58,7)
label = (np.random.rand(200)+0.5).astype(int) #example is binary classification, but can be extended to regression


USE_FULL_DF=True #Takes longer, but had more accurate results. Will use 5000 random samples if marked False
top_k = 30
filter_indices = get_optimal_features(arr,label,USE_FULL_DF, top_k)


n_features = 10
new_arr = reduce_features(filter_indices, arr, n_features)
print(new_arr.shape)

```
