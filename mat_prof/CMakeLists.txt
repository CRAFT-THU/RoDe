include_directories(${Sputnik_SPMM_LIBS})
include_directories(${UTILS_LIBS})
include_directories(${CUSPARSE_SPMM_LIBS})
include_directories(${RoDe_SDDMM_LIBS})

add_executable(mat_info_print mat_info_print.cu)
target_link_libraries(mat_info_print utils)

add_executable(sim_run_prof sim_run_prof.cu)
target_link_libraries(sim_run_prof utils sputnik_spmm cuSPARSE_spmm RoDe_spmm)