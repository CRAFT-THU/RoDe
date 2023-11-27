DATA_PATH=$(pwd)/data
RESULT_PATH=ae_result

mkdir $RESULT_PATH

cd script
./ASpT_eval.sh $DATA_PATH $RESULT_PATH
./_eval.sh $DATA_PATH $RESULT_PATH
cd ..

cp script/get_mat_info.sh build/eval/
cd build/eval
./get_mat_info.sh $DATA_PATH > mat_nnz.csv
mv mat_nnz.csv ../../$RESULT_PATH/
rm get_mat_info.sh
cd ..

# evaluate preprocess
echo "Dataset,ASpT,Sputnik,RoDe" > result_preprocess.csv
base_path=$DATA_PATH
for i in $base_path/*
do 
ii=$(basename "$i")
fpath="${i}/${ii}.mtx"
echo -n $ii >> result_preprocess.csv
./ASpT_SpMM_GPU/pure_preprocess $fpath >> result_preprocess.csv
./Preprocess_opt/preprocess $fpath >> result_preprocess.csv
done
mv result_preprocess.csv ../$RESULT_PATH/
cd ..

cp script/summary.py $RESULT_PATH
cp script/plot_figures.py $RESULT_PATH
cp script/preprocess_eval.py $RESULT_PATH

cd $RESULT_PATH
python summary.py
python preprocess_eval.py
python plot_figures.py