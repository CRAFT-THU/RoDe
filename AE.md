### Download dataset

Download all evaluated matrices (may require several hours to days).
```shell
cd script
./download_data_1.sh
```

---
### Compile

```shell
mkdir build
cd build
cmake ..
make
```
---
### Evaluate

Completing all evaluations requires 2 to 3 days.
```shell
./eval.sh
```

---
The evaluation results have been stored in the 'ae_result' directory, predominantly encompassing the performance outcomes outlined in Sections 7.2, 7.3, and 7.4 of the paper. The summary results are consolidated in the file 'all_results.json' (Table 3, 4, and 6 are populated based on the contents of this file). Additionally, Figures 10, 11, and 12 are generated from the data found in the 'plot_*.csv' tables, resulting in the production of 'spmm_plot.pdf,' 'sddmm_plot.pdf,' and 'preprocess.pdf'.