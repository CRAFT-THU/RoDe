base_path=/data/pm/sparse_matrix/data
echo "validating"
for i in $base_path/*
do 
ii=${i:28}
fpath="${i}/${ii}.mtx"
echo -n ${ii}
../build/eval/eval_spmm_f64_n128_p2 $fpath
# echo ">>>>>>>>>>"
done