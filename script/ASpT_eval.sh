DATA_PATH=$1
OUTPATH=$2

cp eval_ASpT_spmm_*  ../build/ASpT_SpMM_GPU/
cd ../build/ASpT_SpMM_GPU

echo "evaluating ASpT_spmm_f32_n32..."
./eval_ASpT_spmm_f32_n32.sh $DATA_PATH > result_ASpT_spmm_f32_n32.csv

echo "evaluating ASpT_spmm_f32_n128..."
./eval_ASpT_spmm_f32_n128.sh $DATA_PATH > result_ASpT_spmm_f32_n128.csv

echo "evaluating ASpT_spmm_f64_n32..."
./eval_ASpT_spmm_f64_n32.sh $DATA_PATH > result_ASpT_spmm_f64_n32.csv

echo "evaluating ASpT_spmm_f64_n128..."
./eval_ASpT_spmm_f64_n128.sh $DATA_PATH > result_ASpT_spmm_f64_n128.csv

mv result_*  ../../${OUTPATH}/
rm eval_*

cd ../../script
cp eval_ASpT_sddmm_* ../build/ASpT_SDDMM_GPU/
cd ../build/ASpT_SDDMM_GPU

echo "evaluating ASpT_sddmm_f32_n32..."
./eval_ASpT_sddmm_f32_n32.sh $DATA_PATH > result_ASpT_sddmm_f32_n32.csv

echo "evaluating ASpT_sddmm_f32_n128..."
./eval_ASpT_sddmm_f32_n128.sh $DATA_PATH> result_ASpT_sddmm_f32_n128.csv
mv result_*  ../../${OUTPATH}/
rm eval_*