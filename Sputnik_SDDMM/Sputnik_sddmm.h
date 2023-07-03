#ifndef SPUTNIK_SDDMM_H_
#define SPUTNIK_SDDMM_H_

cudaError_t SputnikSddmm(int m, int k, int n, int nonzeros,
                      const int* __restrict__ row_indices,
                      const int* __restrict__ row_offsets,
                      const int* __restrict__ column_indices,
                      const float* __restrict__ values,
                      const float* __restrict__ lhs_matrix,
                      const float* __restrict__ rhs_matrix,
                      float* __restrict__ output_values, cudaStream_t stream);

template <typename LoadType, int kBlockItemsY, int kBlockItemsK,
          int kBlockItemsX, int kBlockWidth, int kPredicateK = true>
cudaError_t CudaSddmmEx(int m, int k, int n, int nonzeros,
                        const int* __restrict__ row_indices,
                        const int* __restrict__ row_offsets,
                        const int* __restrict__ column_indices,
                        const float* __restrict__ values,
                        const float* __restrict__ lhs_matrix,
                        const float* __restrict__ rhs_matrix,
                        float* __restrict__ output_values,
                        cudaStream_t stream);

#endif