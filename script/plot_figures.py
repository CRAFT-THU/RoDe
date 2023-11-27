#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
path = "plot_spmm_"

cases = ['f32_n32','f32_n128','f64_n32','f64_n128']
spmm_benchs = ['ASpT','Sputnik','cuSPARSE','RoDe']

plt.figure(figsize=(16, 12))

i = 1
for e in cases:
    df = pd.read_csv(path + e +'.csv')

    plt.subplot(2,2,i)
    i += 1
    for it in spmm_benchs:
        y_data = df[it]
        plt.plot(y_data, linestyle='-',label=it)

    plt.title('spmm_' + e)
    plt.ylabel('gflops')
    plt.legend()

plt.savefig('spmm_plot.pdf')

# %%
path = "plot_sddmm_"

cases = ['f32_n32','f32_n128']
spmm_benchs = ['ASpT','Sputnik','RoDe']

plt.figure(figsize=(16, 6))

i = 1
for e in cases:
    df = pd.read_csv(path + e +'.csv')

    plt.subplot(1,2,i)
    i += 1
    for it in spmm_benchs:
        y_data = df[it]
        plt.plot(y_data, linestyle='-',label=it)

    plt.title('sddmm_' + e)
    plt.ylabel('gflops')
    plt.legend()

# plt.savefig('spmm_plot.pdf')
plt.savefig('sddmm_plot.pdf')
# %%
path = "plot_preprocess.csv"

spmm_benchs = ['ASpT','Sputnik','RoDe']
plt.figure(figsize=(12, 8))

df = pd.read_csv(path)

for it in spmm_benchs:
    y_data = df[it]
    plt.plot(y_data, linestyle='-',label=it)

plt.title('preprocess_overhead')
plt.ylabel('time(ms)')
plt.yscale('log')
plt.legend()

# plt.savefig('sddmm_plot.pdf')
# plt.show()
plt.savefig('preprocess.pdf')
# %%
