base_path=$2
echo "Dataset,Sputnik_time,Sputnik_gflops,ours_time,ours_gflops"
for i in $base_path/*
do 
ii=$(basename "$i")
fpath="${i}/${ii}.mtx"
echo -n ${ii}
./$1 $fpath 
# echo ">>>>>>>>>>"
done
