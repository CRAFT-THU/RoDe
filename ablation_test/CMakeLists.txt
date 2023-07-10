include_directories(${UTILS_LIBS})

include_directories(${RoDe_SPMM_LIBS})
include_directories(${RoDe_SDDMM_LIBS})

add_executable(spmm_ablation spmm_ablation.cu)
target_link_libraries(spmm_ablation utils RoDe_spmm)

add_executable(spmm_ablation_p1 spmm_ablation_p1.cu)
target_link_libraries(spmm_ablation_p1 utils RoDe_spmm)

add_executable(spmm_ablation_p2 spmm_ablation_p2.cu)
target_link_libraries(spmm_ablation_p2 utils RoDe_spmm)

add_executable(sddmm_ablation sddmm_ablation.cu)
target_link_libraries(sddmm_ablation utils RoDe_sddmm)