echo "Ablation testing(sddmm)..."
cd ../build
echo "Dataset,No_pipeline_time,No_pipeline_time,complete_time,complete_gflops" > result_ablation_sddmm.csv
base_path=/data/pm/sparse_matrix/data
for i in $base_path/*
do 
ii=${i:28}
fpath="${i}/${ii}.mtx"
echo -n $ii                           >> result_ablation_sddmm.csv
./ablation_test/sddmm_ablation $fpath >> result_ablation_sddmm.csv
# echo ">>>>>>>>>>"
done
mv result_ablation_sddmm.csv ../result/