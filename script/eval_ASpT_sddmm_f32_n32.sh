echo "Dataset,ASpT_time,ASpT_gflops"
base_path=$1
for i in $base_path/*
do 
ii=$(basename "$i")
fpath="${i}/${ii}.mtx"
echo -n $ii
./ASpT_sddmm_f32_n32 $fpath 32
# echo ">>>>>>>>>>"
done
