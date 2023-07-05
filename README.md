# RoDe
A Row Decomposition-based Approach for Sparse Matrix Multiplication on GPUs

---
### Download dataset
Download small dataset, only contains 6 matrices
```shell
cd script
./download_data_small.sh
```

Download all evaluated matrices
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
Evaluate ASpT

```shell
mkdir result
cd script
./ASpT_eval.sh
```

Evaluate others
```shell
cd script
./_eval.sh
```

---
### Profile
```shell
cd script
sudo ./prof_spmm
```
Then, the profiled data is in **result** dir, and it can be opened by **Nvidia Nsight compute**.