echo "Ablation testing(spmm)..."
cd ../build
echo "Dataset,no_blocksplit_time,no_blockSplit_gflops,No_pipeline_time,No_pipeline_time,complete_time,complete_gflops" > result_ablation_spmm.csv
base_path=/data/pm/sparse_matrix/data
for i in $base_path/*
do 
ii=${i:28}
fpath="${i}/${ii}.mtx"
echo -n $ii >> result_ablation_spmm.csv
./ablation_test/spmm_ablation_p1 $fpath >> result_ablation_spmm.csv
./ablation_test/spmm_ablation_p2 $fpath >> result_ablation_spmm.csv
# echo ">>>>>>>>>>"
done
mv result_ablation_spmm.csv ../result/