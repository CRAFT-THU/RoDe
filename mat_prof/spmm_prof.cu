// for performance analysis

#include "cuda_runtime.h"
#include "matrix_utils.h"

#include "Sputnik_spmm.h"
#include "cuSPARSE_spmm.h"
#include "RoDeSpmm.h"

#include <sys/io.h>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

using namespace std;
using namespace SPC;

#define SEG_LENGTH 512

#define BN 32

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

__global__ void PrintArray(int n,float* array) {
    for(int i=0; i < n; ++ i)
        printf("Array[%d]:%f\n",i,array[i]);
}

__global__ void PrintArrayInt(int n,int* array) {
    for(int i=0; i < n; ++ i)
        printf("IntArray[%d]:%d\n",i,array[i]);
}

int main(int argc,char **argv) {
    
    // cudaSetDevice(0);

    // string file_path = "../../data/wv2010/wv2010.mtx";
    string file_path = "../../data/mip1/mip1.mtx";


    int ITER = 10;

    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    double gflops = 0.0f;

    // cout<<file_path<<endl;

    SPC::SparseMatrix sm1(file_path,SPC::SORTED,1);

    int * row_offset_h = sm1.RowOffsets();
    int * row_indices_h = sm1.RowIndices();

    for(int i=0; i < 10; ++i)
        cout<<"  row["<<row_indices_h[i]<<"] : "<<row_offset_h[row_indices_h[i]+1] - row_offset_h[row_indices_h[i]]<<endl;


    sm1.RowDivide2Segment(SEG_LENGTH,4,32);
    
    SPC::CudaSparseMatrix<float> c_sm(sm1);

    int m = c_sm.Rows(), k = c_sm.Columns(), n = BN;

    absl::BitGen bitgen;
    SPC::CudaMatrix<float> d_B(k, n ,&bitgen);
    
    float* d_C;
    cudaMalloc((void**)&d_C,sizeof(float)*m*n);

    float* d_C1;
    cudaMalloc((void**)&d_C1,sizeof(float)*m*n);

    float* d_C2;
    cudaMalloc((void**)&d_C2,sizeof(float)*m*n);

    float* diff;
    cudaMalloc((void**)&diff,sizeof(float)*1);

    float tot_ms;
    cudaEvent_t event1,event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    cudaDeviceSynchronize();
    cudaEventRecord(event1,0);
    for(int i=0; i < ITER; ++i)
        SPC::SputnikSpmm(m,c_sm.Columns(),n,c_sm.Nonzeros(),
                    c_sm.RowIndices(),c_sm.Values(),c_sm.RowOffsets(),c_sm.ColumnIndices(),
                    d_B.Values(),
                    d_C1,
                    stream1);

    cudaEventRecord(event2,0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();

    gflops = (double)ITER * (double)c_sm.Nonzeros() * 2 * n / tot_ms / 1000000;

    printf("Sputnik: %f, %f\n",tot_ms,gflops);


    cuSparse_SPMM<float> cu_sp;

    cu_sp.Preprocess(m,c_sm.Columns(),c_sm.Nonzeros(),
                    c_sm.RowOffsets(),c_sm.ColumnIndices(),c_sm.Values());

    cudaDeviceSynchronize();
    cudaEventRecord(event1,0);

    for(int i=0; i < ITER; ++i)
        cu_sp.Process(n,d_B.Values(),d_C);

    cudaEventRecord(event2,0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();

    gflops = (double)ITER * (double)c_sm.Nonzeros() * 2 * n / tot_ms / 1000000;

    printf("cuSparse: %f, %f\n",tot_ms,gflops);


    cudaDeviceSynchronize();
    cudaEventRecord(event1,0);

    for(int i = 0; i < ITER; ++i)
        RoDeSpmm_n32(c_sm.n_segs,c_sm.n_segs_residue,c_sm.Columns(),n,
                        c_sm.Values(),c_sm.ColumnIndices(),c_sm.RowOffsets(),
                       c_sm.seg_row_indices,c_sm.seg_row_indices_residue,c_sm.seg_st_offsets,
                       d_B.Values(),d_C2,stream1,stream2);

    cudaEventRecord(event2,0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();

    gflops = (double)ITER * (double)c_sm.Nonzeros() * 2 * n / tot_ms / 1000000;

    printf("Ours: %f, %f\n",tot_ms,gflops);

    // //    To validate, let ‘ITER’ be 1
    // MatrixDiff<<<(m*n+31)/32,32>>>(m*n,diff,d_C,d_C1);
    // MatrixDiff<<<(m*n+31)/32,32>>>(m*n,diff,d_C,d_C2);

    cudaFree(d_C);
    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(diff);
	
    return 0;
}
// 
