DATA_PATH=$(pwd)/data
# DATA_PATH=/data/pm/sparse_matrix/data
RESULT_PATH=ae_result

mkdir $RESULT_PATH

cd script
./ASpT_eval.sh $DATA_PATH $RESULT_PATH

./_eval.sh $DATA_PATH $RESULT_PATH