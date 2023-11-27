cd ../build
echo "Dataset,ASpT,Sputnik,RoDe" > result_preprocess.csv
base_path=$1
for i in $base_path/*
do 
ii=$(basename "$i")
fpath="${i}/${ii}.mtx"
echo -n $ii >> result_preprocess.csv
./ASpT_SpMM_GPU/pure_preprocess $fpath >> result_preprocess.csv
./Preprocess_opt/preprocess $fpath >> result_preprocess.csv
# echo ">>>>>>>>>>"
done
mv result_preprocess.csv ../result/