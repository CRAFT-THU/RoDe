#%%
import numpy as np
import pandas as pd
import json
import os
from scipy import stats

ASpT_path = 'result_ASpT_'
others_path = 'result__'

items_spmm = ['spmm_f32_n32','spmm_f32_n128','spmm_f64_n32','spmm_f64_n128']
items_sddmm = ['sddmm_f32_n32','sddmm_f32_n128']

# items = ['spmm_f32_n32','spmm_f64_n32','spmm_f64_n128']

results = {}
gm_spmm = { 'Sputnik':[], 'cuSparse':[], 'ASpT':[]}
gm_sddmm ={ 'Sputnik':[], 'ASpT':[]}

mat_nnz = pd.read_csv("mat_nnz.csv")


#%%
ASpT_idx = 1
Sputnik_idx = 3
cuSparse_idx = 5
ours_idx = 7

pers = []

for item in items_spmm:
    data1 = pd.read_csv(ASpT_path + item + '.csv')
    data2 = pd.read_csv(others_path + item + '.csv')

    data_all = pd.merge(data1,data2,on=['Dataset'])
    
    data_all = pd.merge(mat_nnz,data_all,on=['Dataset'])

    # if not os.path.exists(item + '.csv'):
    #     data_all.to_csv(item + '.csv',index = False)

    data_array = data_all.values[:,4:].astype(np.float64)
    
    # deal with sputnik error-run instance: replace with the worse 
    data_array_time = data_array[:,[0,4,6]]
    data_array_gflops = data_array[:,[1,5,7]]
    error_mask = data_array[:,Sputnik_idx] < 0
    print(error_mask.sum())
    data_array[error_mask,Sputnik_idx-1] = data_array_time.max(axis=1)[error_mask]
    data_array[error_mask,Sputnik_idx]   = data_array_gflops.min(axis=1)[error_mask]
    data_all.iloc[:,4:] = data_array

    our_vs_Sputnik_cnt = ( data_array[:,ours_idx] > data_array[:,Sputnik_idx]).sum()
    our_vs_cuSparse_cnt = (data_array[:,ours_idx] > data_array[:,cuSparse_idx]).sum()
    our_vs_ASpT_cnt = (data_array[:,ours_idx] > data_array[:,ASpT_idx]).sum()

    vs_sputnik = data_array[:,ours_idx] / data_array[:,Sputnik_idx]
    vs_cusparse = data_array[:,ours_idx] / data_array[:,cuSparse_idx]
    vs_ASpT = data_array[:,ours_idx] / data_array[:,ASpT_idx]

    gm_spd_vs_sputnik = stats.gmean(vs_sputnik)
    gm_spd_vs_cusparse = stats.gmean(vs_cusparse)
    gm_spd_vs_ASpT = stats.gmean(vs_ASpT)

    max_spd_with_sputnik = vs_sputnik.max()
    max_spd_with_cusparse = vs_cusparse.max()
    max_spd_with_ASpT = vs_ASpT.max()

    results[item] = {}

    results[item]["Number of advantages(with Sputnik)"] = int(our_vs_Sputnik_cnt)
    results[item]["Number of advantages(with cuSparse)"] = int(our_vs_cuSparse_cnt)
    results[item]["Number of advantages(with ASpT)"] = int(our_vs_ASpT_cnt)

    results[item]["Gemetric mean speedup(with Sputnik)"] = gm_spd_vs_sputnik
    results[item]["Gemetric mean speedup(with cuSparse)"] = gm_spd_vs_cusparse
    results[item]["Gemetric mean speedup(with ASpT)"] = gm_spd_vs_ASpT

    results[item]["Max speedup(with Sputnik)"] = max_spd_with_sputnik
    results[item]["Max speedup(with cuSparse)"] = max_spd_with_cusparse
    results[item]["Max speedup(with ASpT)"] = max_spd_with_ASpT
    
    gm_spmm['ASpT'].append(gm_spd_vs_ASpT)
    gm_spmm['cuSparse'].append(gm_spd_vs_cusparse)
    gm_spmm['Sputnik'].append(gm_spd_vs_sputnik)
    
    
    # data_plot = pd.merge(mat_nnz,data_all,on=['Dataset']).sort_values("nnz")
    data_plot = data_all.sort_values("nnz").values[:,3:].astype(np.float64)
    
    UN = 5
    idx = 0
    avg_data = []
    while idx < data_plot.shape[0]:
        avg_data.append(data_plot[idx:idx+UN,:].mean(axis=0))
        idx += UN

    out_array = np.array(avg_data)
    
    ndf = pd.DataFrame(out_array,columns=['nnz','ASpT_time','ASpT','Sputnik_time','Sputnik','cuSPARSE_time','cuSPARSE','RoDe_time','RoDe'])

    ndf.to_csv('plot_'+item+ '.csv',index=False)
    
    
    gflops_array = data_array[:,[ASpT_idx,Sputnik_idx,cuSparse_idx,ours_idx]]
    dspd = gflops_array[:,-1] / gflops_array[:,:-1].max(axis=1)
    
    per = []
    per.append((dspd<0.5).sum())
    per.append((dspd<=0.8).sum() - np.sum(per))
    per.append((dspd <= 1).sum()-np.sum(per))
    per.append((dspd <= 1.2).sum()-np.sum(per))
    per.append((dspd <= 1.5).sum()-np.sum(per))
    per.append((dspd > 1.5).sum())
    
    pers.append([e/np.sum(per) for e in per])
    
    
    
#%%
    

ours_idx = 5
for item in items_sddmm:
    data1 = pd.read_csv(ASpT_path + item + '.csv')
    data2 = pd.read_csv(others_path + item + '.csv')

    data_all = pd.merge(data1,data2,on=['Dataset'])
    
    
    # if not os.path.exists(item + '.csv'):
    #     data_all.to_csv(item + '.csv',index = False)

    data_array = data_all.values[:,1:].astype(np.float64)
    
    # there are also some errror when the row is too large for sputnik, set these items to 1
    error_mask = data_array[:,Sputnik_idx] > 100000
    data_array[error_mask,Sputnik_idx] = 1
    data_array[error_mask,Sputnik_idx - 1] = 1
    
    print(error_mask.sum())
    
    data_all.iloc[:,1:] = data_array

    our_vs_Sputnik_cnt = ( data_array[:,ours_idx] > data_array[:,Sputnik_idx]).sum()
    our_vs_ASpT_cnt = (data_array[:,ours_idx] > data_array[:,ASpT_idx]).sum()

    vs_sputnik = data_array[:,ours_idx] / data_array[:,Sputnik_idx]
    vs_ASpT = data_array[:,ours_idx] / data_array[:,ASpT_idx]

    gm_spd_vs_sputnik = stats.gmean(vs_sputnik)
    gm_spd_vs_ASpT = stats.gmean(vs_ASpT)

    max_spd_with_sputnik = vs_sputnik.max()
    max_spd_with_ASpT = vs_ASpT.max()

    results[item] = {}

    results[item]["Number of advantages(with Sputnik)"] = int(our_vs_Sputnik_cnt)
    results[item]["Number of advantages(with ASpT)"] = int(our_vs_ASpT_cnt)

    results[item]["Gemetric mean speedup(with Sputnik)"] = gm_spd_vs_sputnik
    results[item]["Gemetric mean speedup(with ASpT)"] = gm_spd_vs_ASpT

    results[item]["Max speedup(with Sputnik)"] = max_spd_with_sputnik
    results[item]["Max speedup(with ASpT)"] = max_spd_with_ASpT
    
    
    gm_sddmm['ASpT'].append(gm_spd_vs_ASpT)
    gm_sddmm['Sputnik'].append(gm_spd_vs_sputnik)
    
    
    data_plot = pd.merge(mat_nnz,data_all,on=['Dataset']).sort_values("nnz")
    data_plot = data_plot.values[:,3:].astype(np.float64)
    
    UN = 5
    idx = 0
    avg_data = []
    while idx < data_plot.shape[0]:
        avg_data.append(data_plot[idx:idx+UN,:].mean(axis=0))
        idx += UN

    out_array = np.array(avg_data)
    
    ndf = pd.DataFrame(out_array,columns=['nnz','ASpT_time','ASpT','Sputnik_time','Sputnik','RoDe_time','RoDe'])

    ndf.to_csv('plot_'+item+ '.csv',index=False)
    
    
    gflops_array = data_array[:,[ASpT_idx,Sputnik_idx,ours_idx]]
    dspd = gflops_array[:,-1] / gflops_array[:,:-1].max(axis=1)
    
    per = []
    per.append((dspd<0.5).sum())
    per.append((dspd<=0.8).sum() - np.sum(per))
    per.append((dspd <= 1).sum()-np.sum(per))
    per.append((dspd <= 1.2).sum()-np.sum(per))
    per.append((dspd <= 1.5).sum()-np.sum(per))
    per.append((dspd > 1.5).sum())
    
    pers.append([e/np.sum(per) for e in per])
    
    
results['spmm gmean spd with sputnik'] = stats.gmean(gm_spmm['Sputnik'])
results['spmm gmean spd with cuSPARSE'] = stats.gmean(gm_spmm['cuSparse'])
results['spmm gmean spd with ASpT'] = stats.gmean(gm_spmm['ASpT'])

results['sddmm gmean spd with sputnik'] = stats.gmean(gm_sddmm['Sputnik'])
results['sddmm gmean spd with ASpT'] = stats.gmean(gm_sddmm['ASpT'])

results['speedups distribution(0.5,0.8,1,1.2,1.5)'] = pers


#%%
results_json = json.dumps(results)

with open('all_results.json','w') as f:
    f.write(results_json)
# %%
