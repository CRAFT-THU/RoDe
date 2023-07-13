#ifndef RODE_SPMM_H_
#define RODE_SPMM_H_

void RoDeSpmm_n32(int m1,int m2,int k,int n,const float* __restrict__ values,const int * __restrict__ column_indices,const int * __restrict__ row_offsets,const int *__restrict__ row_indices1,const int *__restrict__ row_indices2,\
                const int * __restrict__ row_seg_st_offsets,const float *B,float* C,cudaStream_t stream1,cudaStream_t stream2) ;

void RoDeSpmm_n128(int m1,int m2,int k,int n,const float* __restrict__ values,const int * __restrict__ column_indices,const int * __restrict__ row_offsets,const int *__restrict__ row_indices1,const int *__restrict__ row_indices2,\
                const int * __restrict__ row_seg_st_offsets,const float *B,float* C,cudaStream_t stream1,cudaStream_t stream2) ;

void RoDeSpmm_n32(int m1,int m2,int k,int n,const double* __restrict__ values,const int * __restrict__ column_indices,const int * __restrict__ row_offsets,const int *__restrict__ row_indices1,const int *__restrict__ row_indices2,\
                const int * __restrict__ row_seg_st_offsets,const double *B,double* C,cudaStream_t stream1,cudaStream_t stream2);

void RoDeSpmm_n128(int m1,int m2,int k,int n,const double* __restrict__ values,const int * __restrict__ column_indices,const int * __restrict__ row_offsets,const int *__restrict__ row_indices1,const int *__restrict__ row_indices2,\
                const int * __restrict__ row_seg_st_offsets,const double *B,double* C,cudaStream_t stream1,cudaStream_t stream2);

#endif