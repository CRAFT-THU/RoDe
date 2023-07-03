aux_source_directory(. DIR_SPMM_SRCS)

include_directories(${UTILS_LIBS})

add_library(sputnik_spmm ${DIR_SPMM_SRCS})
target_link_libraries(sputnik_spmm utils)