#include "cuda_runtime.h"
#include <iostream>
#include "common_utils.h"
#include "matrix_utils.h"

// #include "Sputnik_sddmm.h"

#include <cuda/barrier>
#include <cuda/pipeline>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <sys/io.h>

#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>


#include "RoDeSddmm.h"

using namespace std;

using namespace SPC; 

#define BN 32

#define SEG_LENGTH 512

// #define VALIDATE

__global__ void MatrixDiff(int n,float* res,float* A,float* B) {
    if(threadIdx.x == 0 && blockIdx.x == 0)
        res[0] = 0.0f;
    
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    float diff = abs(A[idx] - B[idx]);

    // if(diff > 1e-5) {
    //     printf("[%d] : %f ~ %f\n",idx,A[idx],B[idx]);
    // }

    float r = diff;
    r += __shfl_down_sync(0xffffffff,r,16);
    r += __shfl_down_sync(0xffffffff,r,8);
    r += __shfl_down_sync(0xffffffff,r,4);
    r += __shfl_down_sync(0xffffffff,r,2);
    r += __shfl_down_sync(0xffffffff,r,1);

    if(threadIdx.x == 0)
        atomicAdd(res,r);

    __syncthreads();
    if(threadIdx.x == 0 && blockIdx.x == 0)
        printf("Matrix diff: %f\n",res[0]);
}

__global__ void FillValues(int n,float* array,float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    array[idx] = val;
}

__global__ void StandKernel(int m,int n,int k,const int* __restrict__ row_indices,const int* __restrict__ row_offsets,const int* __restrict__ column_indices,\
    const float* __restrict__ values,const float* __restrict__ lhs_matrix,const float* __restrict__ rhs_matrix,float* __restrict__ out) {

    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(m_idx >= m) return;

    m_idx = row_indices[m_idx];

    int row_offset = row_offsets[m_idx];
    int nonzeros = row_offsets[m_idx + 1] - row_offset;

    for(int n_idx = 0; n_idx < nonzeros; ++ n_idx) {
        int col_idx = column_indices[row_offset + n_idx];
        int sparse_value = values[row_offset + n_idx];

        float res = 0.0f;
        for(int kk = 0; kk < k; ++ kk) {
            res += lhs_matrix[m_idx * k + kk]  * rhs_matrix[col_idx * k + kk];
        }
        out[row_offset + n_idx] = res * sparse_value;
    }

}

void StandCall(int m,int n,int k,const int* __restrict__ row_indices,const int* __restrict__ row_offsets,const int* __restrict__ column_indices,\
    const float* __restrict__ values,const float* __restrict__ lhs_matrix,const float* __restrict__ rhs_matrix,float* __restrict__ out) {
    StandKernel<<<(m + 31)/32 ,32>>>(m,n,k,row_indices,row_offsets,column_indices,values,lhs_matrix,rhs_matrix,out);
}

int main(int argc,char ** argv) {

    int ITER = 10;
    // cudaSetDevice(0); 

    string file_path = "../../data/mip1/mip1.mtx";


    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    double gflops = 0.0f;

    SPC::SparseMatrix sm1(file_path,SPC::SORTED,1);
	
    sm1.RowDivide2Segment(SEG_LENGTH,4,32);

    // printf("NR1 : %d , NR2 : %d\n",sm1.n_segs,sm1.n_segs_residue);

    SPC::CudaSparseMatrix<float> c_sm(sm1);
    int m = c_sm.Rows(), n = c_sm.Columns(), k = BN;

    absl::BitGen bitgen;
    SPC::CudaMatrix<float> d_B1(m, k ,&bitgen);
    SPC::CudaMatrix<float> d_B2(n, k ,&bitgen);

    int size = c_sm.Nonzeros();

    float* d_C;
    cudaMalloc((void**)&d_C,sizeof(float)*size);

    float* d_C1;
    cudaMalloc((void**)&d_C1,sizeof(float)*size);

    float* d_C2;
    cudaMalloc((void**)&d_C2,sizeof(float)*size);

    float* d_C3;
    cudaMalloc((void**)&d_C3,size*sizeof(float));

    float* d_C4;
    cudaMalloc((void**)&d_C4,size*sizeof(float));

    float* d_C5;
    cudaMalloc((void**)&d_C5,size*sizeof(float));

    float* diff;
    cudaMalloc((void**)&diff,sizeof(float)*1);

    FillValues<<<(size+31)/32,32>>>(size,d_C,0.0f);
    FillValues<<<(size+31)/32,32>>>(size,d_C1,0.0f);
    FillValues<<<(size+31)/32,32>>>(size,d_C2,0.0f);
    // FillValues<<<(size+31)/32,32>>>(size,d_C3,0.0f);
    // FillValues<<<(size+31)/32,32>>>(size,d_C4,0.0f);

    #ifdef VALIDATE
        StandCall(m,n,k,
                c_sm.RowIndices(),c_sm.RowOffsets(),c_sm.ColumnIndices(),c_sm.Values(),
                d_B1.Values(),d_B2.Values(),d_C);
    #endif
            
    float tot_ms;
    cudaEvent_t event1,event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    // cudaDeviceSynchronize();
    // cudaEventRecord(event1,0);
    // for(int i=0; i < ITER; ++i)
    //     SputnikSddmm(m,k,n,c_sm.Nonzeros(),
    //                 c_sm.RowIndices(),c_sm.RowOffsets(),c_sm.ColumnIndices(),c_sm.Values(),
    //                 d_B1.Values(),d_B2.Values(),d_C1,stream1);

    // cudaEventRecord(event2,0);

    // cudaEventSynchronize(event1);
    // cudaEventSynchronize(event2);
    // cudaEventElapsedTime(&tot_ms, event1, event2);
    // cudaDeviceSynchronize();

    // gflops = (double)ITER * (double)c_sm.Nonzeros() * 2 * k / tot_ms / 1000000;
    // printf(", %f, %f",tot_ms,gflops);

    cudaDeviceSynchronize();
    cudaEventRecord(event1,0);

    for(int i=0; i < ITER; ++i)
        RoDeSDDMM_n32(c_sm.n_segs,c_sm.n_segs_residue,n,k,
                    c_sm.seg_row_indices,c_sm.seg_row_indices_residue,c_sm.seg_st_offsets,
                    c_sm.RowOffsets(),c_sm.ColumnIndices(),c_sm.Values(),
                    d_B1.Values(),d_B2.Values(),d_C2,stream1,stream2);

    cudaEventRecord(event2,0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();

    gflops = (double)ITER * (double)c_sm.Nonzeros() * 2 * k / tot_ms / 1000000;
    printf("RoDe: %f,%f",tot_ms,gflops);


    printf("\n");


    #ifdef VALIDATE
        MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C1);
        MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C2);
        // MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C3);
        // MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C4);
        // MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C5);
    #endif

    cudaFree(d_C);
    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_C3);
    cudaFree(d_C4);
    cudaFree(d_C5);
    cudaFree(diff);

    return 0;
}