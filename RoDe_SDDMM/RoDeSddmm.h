#ifndef RODE_SDDMM_H_
#define RODE_SDDMM_H_

void RoDeSDDMM_n32(int m1,int m2,int n,int k,const int* __restrict__ row_indices_block,const int* __restrict__ row_indices_residue,const int* __restrict__ st_offsets,const int* __restrict__ row_offsets,const int* __restrict__ column_indices,\
    const float* __restrict__ values,const float* __restrict__ lhs_matrix,const float* __restrict__ rhs_matrix,float* __restrict__ out,cudaStream_t stream1,cudaStream_t stream2);


void RoDeSDDMM_n128(int m1,int m2,int n,int k,const int* __restrict__ row_indices_block,const int* __restrict__ row_indices_residue,const int* __restrict__ st_offsets,const int* __restrict__ row_offsets,const int* __restrict__ column_indices,\
    const float* __restrict__ values,const float* __restrict__ lhs_matrix,const float* __restrict__ rhs_matrix,float* __restrict__ out,cudaStream_t stream1,cudaStream_t stream2);

#endif