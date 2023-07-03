aux_source_directory(. DIR_OUR_SDDMM_SRCS)

include_directories(${UTILS_LIBS})

add_library(RoDe_sddmm ${DIR_OUR_SDDMM_SRCS})
target_link_libraries(RoDe_sddmm utils)