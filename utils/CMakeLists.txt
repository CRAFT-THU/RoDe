aux_source_directory(. DIR_UTILS)

add_library(utils ${DIR_UTILS})
target_link_libraries(utils ${SPC_LIBS})