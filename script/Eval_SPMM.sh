cp -r eval_ASpT_spmm_*.sh ../build/ASpT_SpMM_GPU/
cp eval_spmm__.sh ../build/eval/
cp eval_spmm_f64_n128.sh ../build/eval/

cd ../build/eval

./eval_spmm__.sh eval_spmm_f32_n32  > _spmm_f32_n32.csv
./eval_spmm__.sh eval_spmm_f32_n128 > _spmm_f32_n128.csv
./eval_spmm__.sh eval_spmm_f64_n32  > _spmm_f64_n32.csv
./eval_spmm_f64_n128.sh             > _spmm_f64_n128.csv

cp -r *.csv ../../results_temp/

cd ../ASpT_SpMM_GPU

./eval_ASpT_spmm_f32_n32.sh  > ASpT_spmm_f32_n32.csv
./eval_ASpT_spmm_f32_n128.sh > ASpT_spmm_f32_n128.csv
./eval_ASpT_spmm_f64_n32.sh  > ASpT_spmm_f64_n32.csv
./eval_ASpT_spmm_f64_n128.sh > ASpT_spmm_f64_n128.csv

cp -r *.csv ../../results_temp/
