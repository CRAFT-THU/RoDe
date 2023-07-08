add_executable(ASpT_spmm_f32_n32 sspmm_32.cu)

add_executable(ASpT_spmm_f32_n128 sspmm_128.cu)

add_executable(ASpT_spmm_f64_n32 dspmm_32.cu)

add_executable(ASpT_spmm_f64_n128 dspmm_128.cu)

add_executable(pure_preprocess preprocess.cu)
