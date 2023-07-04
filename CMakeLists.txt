cmake_minimum_required(VERSION 3.13)

project(SPC CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_definitions(-w)

find_package(CUDA)
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
message("-- nvcc generates code for arch ${CUDA_ARCH_LIST}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
# cmake_policy(SET CMP0104 OLD)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Dependencies.cmake)

set(UTILS_LIBS ${PROJECT_SOURCE_DIR}/utils)
set(Sputnik_SPMM_LIBS ${PROJECT_SOURCE_DIR}/Sputnik_SpMM)
set(Sputnik_SDDMM_LIBS ${PROJECT_SOURCE_DIR}/Sputnik_SDDMM)
set(CUSPARSE_SPMM_LIBS ${PROJECT_SOURCE_DIR}/cuSparse_SpMM)

set(RoDe_SPMM_LIBS ${PROJECT_SOURCE_DIR}/RoDe_SpMM)
set(RoDe_SDDMM_LIBS ${PROJECT_SOURCE_DIR}/RoDe_SDDMM)


include_directories(${PROJECT_SOURCE_DIR})

set(CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80;-G;-g)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xcompiler -fopenmp -gencode arch=compute_80,code=sm_80")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xcompiler -fopenmp -gencode arch=compute_80,code=sm_80")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

MESSAGE( STATUS "this var key = ${CMAKE_CUDA_FLAGS}.")

add_subdirectory(utils)
add_subdirectory(Sputnik_SpMM)
add_subdirectory(Sputnik_SDDMM)

add_subdirectory(cuSparse_SpMM)

add_subdirectory(ASpT_SpMM_GPU)
add_subdirectory(ASpT_SDDMM_GPU)

add_subdirectory(RoDe_SDDMM)
add_subdirectory(RoDe_SpMM)

add_subdirectory(mat_prof)
add_subdirectory(eval)