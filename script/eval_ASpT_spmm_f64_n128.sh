echo "Dataset,ASpT_time,ASpT_gflops"
base_path=$1
for i in $base_path/*
do 
ii=${i:28}
fpath="${i}/${ii}.mtx"
echo -n $ii
./ASpT_spmm_f64_n128 $fpath 128
# echo ">>>>>>>>>>"
done
