#%%
import numpy as np
import pandas as pd
import json
import os
from scipy import stats

df = pd.read_csv("../result/result_ablation_spmm.csv")

data_array = df.values[:,1:].astype(np.float64)

#%%

no_bs = data_array[:,1] / data_array[:,-1]

print(stats.gmean(no_bs))
# %%

no_pl = data_array[:,3] / data_array[:,-1]
# %%
