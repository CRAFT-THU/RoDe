include_directories(${UTILS_LIBS})

include_directories(${Sputnik_SPMM_LIBS})
include_directories(${Sputnik_SDDMM_LIBS})

include_directories(${CUSPARSE_SPMM_LIBS})

include_directories(${RoDe_SPMM_LIBS})
include_directories(${RoDe_SDDMM_LIBS})

add_executable(eval_spmm_f32_n32 eval_spmm_f32_n32.cu)
target_link_libraries(eval_spmm_f32_n32 utils sputnik_spmm cuSPARSE_spmm RoDe_spmm)

add_executable(eval_spmm_f32_n128 eval_spmm_f32_n128.cu)
target_link_libraries(eval_spmm_f32_n128 utils sputnik_spmm cuSPARSE_spmm RoDe_spmm)

add_executable(eval_spmm_f64_n32 eval_spmm_f64_n32.cu)
target_link_libraries(eval_spmm_f64_n32 utils sputnik_spmm cuSPARSE_spmm RoDe_spmm)

add_executable(eval_spmm_f64_n128_p1 eval_spmm_f64_n128_p1.cu)
target_link_libraries(eval_spmm_f64_n128_p1 utils sputnik_spmm cuSPARSE_spmm RoDe_spmm)
add_executable(eval_spmm_f64_n128_p2 eval_spmm_f64_n128_p2.cu)
target_link_libraries(eval_spmm_f64_n128_p2 utils sputnik_spmm cuSPARSE_spmm RoDe_spmm)

add_executable(eval_sddmm_f32_n32 eval_sddmm_f32_n32.cu)
target_link_libraries(eval_sddmm_f32_n32 sputnik_sddmm utils RoDe_sddmm)

add_executable(eval_sddmm_f32_n128 eval_sddmm_f32_n128.cu)
target_link_libraries(eval_sddmm_f32_n128 sputnik_sddmm utils RoDe_sddmm)

add_executable(get_matrix_info get_matrix_info.cu)
target_link_libraries(get_matrix_info utils)