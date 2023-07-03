aux_source_directory(. DIR_SDDMM_SRCS)

include_directories(${UTILS_LIBS})

add_library(sputnik_sddmm ${DIR_SDDMM_SRCS})
target_link_libraries(sputnik_sddmm utils)