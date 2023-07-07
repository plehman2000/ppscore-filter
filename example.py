from ppfilter import get_optimal_features, reduce_features
import numpy as np



arr = np.random.rand(200,58,7)
label = (np.random.rand(200)+0.5).astype(int) #example is binary classification, but can be extended to regression


USE_FULL_DF=True #? takes longer but more accurate than sampling
top_k = 30
filter_indices = get_optimal_features(arr,label,USE_FULL_DF, top_k)


n_features = 10
new_arr = reduce_features(filter_indices, arr, n_features)
print(new_arr.shape)
