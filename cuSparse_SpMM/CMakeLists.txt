aux_source_directory(. DIR_CUSPARSE_SPMM_SRCS)

add_library(cuSPARSE_spmm ${DIR_CUSPARSE_SPMM_SRCS})
target_link_libraries(cuSPARSE_spmm ${CUDA_cusparse_LIBRARY})