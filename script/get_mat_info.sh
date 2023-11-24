echo "Dataset,nr,nc,nnz"
base_path=$1

for i in $base_path/*
do 
ii=$(basename "$i")
fpath="${i}/${ii}.mtx"
echo -n $ii
./get_matrix_info $fpath
# echo ">>>>>>>>>>"
done
