base_path=$1
echo "Dataset,Spuntik_time,Sputnik_gflops,cuSparse_time,cuSparse_gflops,ours_time,ours_gflops"
for i in $base_path/*
do 
ii=$(basename "$i")
fpath="${i}/${ii}.mtx"
echo -n ${ii}
./eval_spmm_f64_n128_p1 $fpath | sed -Ee "s/CUDA Error: an illegal memory access was encountered//g"
./eval_spmm_f64_n128_p2 $fpath
# echo ">>>>>>>>>>"
done
