#ifndef CUDA_SPMM_H_
#define CUDA_SPMM_H_


#include "basic_utils.h"

namespace SPC {
    cudaError_t SputnikSpmm(int m, int k, int n, int nonzeros,
                     const int* __restrict__ row_indices,
                     const float* __restrict__ values,
                     const int* __restrict__ row_offsets,
                     const int* __restrict__ column_indices,
                     const float* __restrict__ dense_matrix,
                     float* __restrict__ output_matrix,
                     cudaStream_t stream);

    cudaError_t SputnikSpmm(int m, int k, int n, int nonzeros,
                     const int* __restrict__ row_indices,
                     const double* __restrict__ values,
                     const int* __restrict__ row_offsets,
                     const int* __restrict__ column_indices,
                     const double* __restrict__ dense_matrix,
                     double* __restrict__ output_matrix,
                     cudaStream_t stream);

    cudaError_t CudaSpmmBiasRelu(int m, int k, int n, int nonzeros,
                             const int* __restrict__ row_indices,
                             const float* __restrict__ values,
                             const int* __restrict__ row_offsets,
                             const int* __restrict__ column_indices,
                             const float* __restrict__ dense_matrix,
                             const float* __restrict__ bias,
                             float* __restrict__ output_matrix,
                             cudaStream_t stream);

    cudaError_t CudaSpmm(int m, int k, int n, int nonzeros,
                     const int* __restrict__ row_indices,
                     const half2* __restrict__ values,
                     const int* __restrict__ row_offsets,
                     const short2* __restrict__ column_indices,
                     const half2* __restrict__ dense_matrix,
                     half2* __restrict__ output_matrix,
                     cudaStream_t stream);


    cudaError_t CudaSpmmBiasRelu(int m, int k, int n, int nonzeros,
                             const int* __restrict__ row_indices,
                             const half2* __restrict__ values,
                             const int* __restrict__ row_offsets,
                             const short2* __restrict__ column_indices,
                             const half2* __restrict__ dense_matrix,
                             const float* __restrict__ bias,
                             half2* __restrict__ output_matrix,
                             cudaStream_t stream);

    template <typename Config>
    cudaError_t CudaSpmmEx(
        int m, int k, int n, int nonzeros,
        const int* __restrict__ row_indices,
        const typename Config::ScalarValue* __restrict__ values,
        const int* __restrict__ row_offsets,
        const typename Config::ScalarIndex* __restrict__ column_indices,
        const typename Config::ScalarValue* __restrict__ dense_matrix,
        const float* __restrict__ bias,
        typename Config::ScalarValue* __restrict__ output_matrix,
        cudaStream_t stream);
}


#endif