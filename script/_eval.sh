DATA_PATH=$1
OUTPATH=$2

cp eval_spmm_*  ../build/eval
cp eval_sddmm_* ../build/eval
cd ../build/eval

echo "evaluating spmm_f32_n32..."
./eval_spmm_call.sh eval_spmm_f32_n32  $DATA_PATH > result__spmm_f32_n32.csv

echo "evaluating spmm_f32_n128..."
./eval_spmm_call.sh eval_spmm_f32_n128 $DATA_PATH > result__spmm_f32_n128.csv

echo "evaluating spmm_f64_n32..."
./eval_spmm_call.sh eval_spmm_f64_n32  $DATA_PATH > result__spmm_f64_n32.csv

echo "evaluating spmm_f64_n128..."
./eval_spmm_f64_n128.sh                $DATA_PATH > result__spmm_f64_n128.csv

echo "evaluating sddmm_f32_n32..."
./eval_sddmm_call.sh eval_sddmm_f32_n32  $DATA_PATH > result__sddmm_f32_n32.csv

echo "evaluating sddmm_f32_n128..."
./eval_sddmm_call.sh eval_sddmm_f32_n128 $DATA_PATH > result__sddmm_f32_n128.csv

mv result_* ../../${OUTPATH}
rm *.sh