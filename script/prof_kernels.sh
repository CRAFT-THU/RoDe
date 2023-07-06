cd ../build/mat_prof
# ncu --set full -o pipeline_mot ./sim_run_prof
# ncu --set full -o prof__spmm ./spmm_prof

ncu --set full -o prof__sddmm ./sddmm_prof

mv *.ncu-rep ../../result/

# cd ../ASpT_SpMM_GPU
# ncu --set full -o prof_ASpT_spmm ./ASpT_spmm_f32_n32 ../../data/mip1/mip1.mtx 32
# mv *.ncu-rep ../../result/

cd ../ASpT_SDDMM_GPU
ncu --set full -o prof_ASpT_sddmm ./ASpT_sddmm_f32_n32 ../../data/mip1/mip1.mtx 32
mv *.ncu-rep ../../result/