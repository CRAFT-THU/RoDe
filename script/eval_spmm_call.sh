base_path=/data/pm/sparse_matrix/data
echo "Dataset,Sputnik_time,Sputnik_gflops,cuSPARSE_time,cuSPARSE_gflops,ours_time,ours_gflops"
for i in $base_path/*
do 
ii=${i:28}
fpath="${i}/${ii}.mtx"
echo -n ${ii}
./$1 $fpath 
# echo ">>>>>>>>>>"
done
