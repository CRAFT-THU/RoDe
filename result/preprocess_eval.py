#%%
import numpy as np
import pandas as pd
import json
import os
from scipy import stats

mat_nnz = pd.read_csv("mat_nnz.csv")
result  = pd.read_csv("result_preprocess.csv")

result['Dataset'] = result['Dataset'].str.strip()
data_all = pd.merge(mat_nnz,result,on=["Dataset"])

data_plot = data_all.sort_values("nnz").values[:,3:].astype(np.float64)


#%%
UN = 5 
idx = 0
avg_data = []
while idx < data_plot.shape[0]:
    avg_data.append(data_plot[idx:idx+UN,:].mean(axis= 0))
    idx += UN

out_array = np.array(avg_data)


ndf = pd.DataFrame(out_array,columns=['nnz','ASpT','Sputnik','RoDe'])

ndf.to_csv('plot_preprocess.csv',index=False)
# %%
