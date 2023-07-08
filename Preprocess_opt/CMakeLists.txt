include_directories(${UTILS_LIBS})


add_executable(preprocess _preprocess.cu)
target_link_libraries(preprocess utils)