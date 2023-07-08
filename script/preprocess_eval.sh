cd ../build
echo "Dataset,ASpT,Sputnik,RoDe" > result_preprocess.csv
base_path=/data/pm/sparse_matrix/data
for i in $base_path/*
do 
ii=${i:28}
fpath="${i}/${ii}.mtx"
echo -n $ii >> result_preprocess.csv
./ASpT_SpMM_GPU/pure_preprocess $fpath >> result_preprocess.csv
./Preprocess_opt/preprocess $fpath >> result_preprocess.csv
# echo ">>>>>>>>>>"
done
mv result_preprocess.csv ../result/