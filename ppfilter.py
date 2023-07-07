import ppscore as pps
import pandas as pd
import numpy as np
from tqdm import tqdm



def get_optimal_features(arr, label, USE_FULL_DF, top_k):
    #! Expects array in order B, D1, D2, so if its a set of N 1-D samples, it has no 3rd dim

    #? If there are multiple channels, it will calculate a ppscore for each channel of features

    #* Check shape
    if len(np.shape(arr)) < 2 or len(np.shape(arr)) > 3:
        raise ValueError(f"Array should have between 2 and 3 dimensions. Your array has shape {np.shape(arr)}")

    if len(np.shape(arr)) == 2:
        arr = np.expand_dims(arr,-1)
        print(np.shape(arr))


    if USE_FULL_DF:
        sample_amt = len(arr)
    else:
        sample_amt=5000
            
    n_channels = np.shape(arr)[-1]
            
    channel_dframes = {}
    for channel in tqdm(range(np.shape(arr)[-1])):
        channel_wise_slice = arr[:,:,channel]
        data_dict = {"label":label}
        for feature in range(np.shape(channel_wise_slice)[-1]):
            temp_feature_slice = channel_wise_slice[:, feature]
            data_dict[f"{feature}"] = temp_feature_slice
            # print(np.shape(temp_feature_slice))
            
        temp_dframe = pd.DataFrame(data_dict)

        predictors_df = pps.predictors(temp_dframe, y="label", sample=sample_amt)
        
        channel_dframes[channel] = predictors_df


    if top_k:
        for x in channel_dframes.keys():
            channel_dframes[x] = channel_dframes[x].head(top_k)
            
    
    for i in range(n_channels):
        channel_dframes[i] = list(channel_dframes[i]['x'])
        
    return channel_dframes



def reduce_features(channel_dframes, arr, num_new_features):
    #! reduce features using calculated feature dframes
    new_arr = np.zeros((np.shape(arr)[0], num_new_features, np.shape(arr)[2]))
    for i in range(np.shape(arr)[-1]):
        temp_ind = channel_dframes[i]
        channel_dframes[i] = [int(x) for x in channel_dframes[i]]
        temp_arr = np.take(arr[:,:,i], channel_dframes[i][:num_new_features], axis=1)
        new_arr[:, :, i] = temp_arr
    return new_arr






