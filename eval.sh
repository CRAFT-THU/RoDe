DATA_PATH=$(pwd)/data
# DATA_PATH=/data/pm/sparse_matrix/data
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
cd ../..

cp script/summary.py $RESULT_PATH
cd $RESULT_PATH
python summary.py