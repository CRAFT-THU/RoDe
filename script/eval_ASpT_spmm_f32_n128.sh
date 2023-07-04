echo "Dataset,ASpT_time,ASpT_gflops"
base_path=/data/pm/sparse_matrix/data
for i in $base_path/*
do 
ii=${i:28}
fpath="${i}/${ii}.mtx"
echo -n $ii
./ASpT_spmm_f32_n128 $fpath 128
# echo ">>>>>>>>>>"
done
