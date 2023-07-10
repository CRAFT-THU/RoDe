#include "cuda_runtime.h"
#include <iostream>
#include "common_utils.h"
#include "matrix_utils.h"

#include "RoDeSpmm.h"

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

using namespace std;


using namespace SPC;

#define BN 128
#define SEG_LENGTH 512

__global__ void MatrixDiff(int n,double* res,double* A,double* B) {
    if(threadIdx.x == 0 && blockIdx.x == 0)
        res[0] = 0.0f;
    
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    double diff = abs(A[idx] - B[idx]);

    // if(diff > 1e-5) {
    //     printf("[%d] : %f ~ %f\n",idx,A[idx],B[idx]);
    // }

    double r = diff;
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

__global__ void PrintArray(int n,double* array) {
    for(int i=0; i < n; ++ i)
        printf("Array[%d]:%f\n",i,array[i]);
}

__global__ void PrintArrayInt(int n,int* array) {
    for(int i=0; i < n; ++ i)
        printf("IntArray[%d]:%d\n",i,array[i]);
}

int main(int argc,char **argv) {
    // cudaSetDevice(1);
    string file_path;
    if(argc < 2) {
        cout<<"No file path, use default file 'mip1' "<<endl;
        // file_path = "../../data/mip1/mip1.mtx";
        file_path = "/data/pm/sparse_matrix/data/144/144.mtx";
    }
    else {
        file_path = argv[1];
    }

    int ITER = 10;

    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    double gflops = 0.0f;

    // cout<<file_path<<endl;

    SPC::SparseMatrix sm1(file_path,SPC::SORTED,1);

    int * row_offset_h = sm1.RowOffsets();
    int * row_indices_h = sm1.RowIndices();


    int NoBlockSplit = sm1.Rows() + 1;
    sm1.RowDivide2Segment(NoBlockSplit,4,32);
    
    SPC::CudaSparseMatrix<double> c_sm(sm1);

    int m = c_sm.Rows(), k = c_sm.Columns(), n = BN;

    absl::BitGen bitgen;
    SPC::CudaMatrix<double> d_B(k, n ,&bitgen);
    
    double* d_C;
    cudaMalloc((void**)&d_C,sizeof(double)*m*n);

    double* d_C1;
    cudaMalloc((void**)&d_C1,sizeof(double)*m*n);

    double* d_C2;
    cudaMalloc((void**)&d_C2,sizeof(double)*m*n);

    double* diff;
    cudaMalloc((void**)&diff,sizeof(double)*1);

    float tot_ms;
    cudaEvent_t event1,event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    /* no block split*/
    cudaDeviceSynchronize();
    cudaEventRecord(event1,0);
    for(int i = 0; i < ITER; ++i)
        RoDeSpmm_n128_nobs(c_sm.n_segs,c_sm.n_segs_residue,c_sm.Columns(),n,
                        c_sm.Values(),c_sm.ColumnIndices(),c_sm.RowOffsets(),
                       c_sm.seg_row_indices,c_sm.seg_row_indices_residue,c_sm.seg_st_offsets,
                       d_B.Values(),d_C1,stream1,stream2);

    cudaEventRecord(event2,0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();
    gflops = (double)ITER * (double)c_sm.Nonzeros() * 2 * n / tot_ms / 1000000;
    printf(", %f, %f",tot_ms,gflops);

    // printf("\n");

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
