#include "cuda_runtime.h"
#include "matrix_utils.h"

// #include "Sputnik_spmm.h"
// #include "cuSPARSE_spmm.h"
// #include "SegSpmm.h"

#include <sys/io.h>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <cub/block/block_scan.cuh>
#include <cub/cub.cuh>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>

#include <chrono>

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

__global__ void PrintArray(int n,int* array) {
    for(int i=0; i < n; ++ i)
        printf("Array[%d]:%d\n",i,array[i]);
}


// <<<?, 32>>>



__global__ void Sputnik_getKeys(int M,const int *row_ptr,int *keys,int *vals) {
    int r_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(r_idx >= M) return;
    keys[r_idx] = - row_ptr[r_idx + 1] + row_ptr[r_idx];
    vals[r_idx] = r_idx;
}
void Sputinik_preprocess(int M,const int *row_ptr, int *row_indices) {

    size_t temp_storage_bytes = 0;
    void* d_temp_storage = nullptr;


    int *keys;
    cudaMalloc((void**)&keys,sizeof(int)*M);
    Sputnik_getKeys<<<(M+1023)/1024,1024>>>(M,row_ptr,keys,row_indices);

    thrust::device_ptr<int> dkeys(keys);
    thrust::device_ptr<int> dvals(row_indices);

    thrust::sort_by_key(dkeys,dkeys + M,dvals);

    cudaFree(keys);
    cudaFree(d_temp_storage);
}

void Sputinik_preprocess_cpu(int M,const int *row_ptr,int *row_indices) {
    std::iota(row_indices,row_indices+M,0);
    std::sort(row_indices,row_indices+M, 
                [&row_ptr](int i,int j) {
                    int nnz_a = row_ptr[i+1] - row_ptr[i];
                    int nnz_b = row_ptr[j+1] - row_ptr[j];
                    return nnz_a > nnz_b;
                });
}
//  RoDe preprocess

// some bug
template <int ThreadBlockX>
__global__ void RoDe_preprocess__(int N_SEG,int blockX,int vecLen,int M,const int *row_ptr, int* block_r_ind, int *st_off,int *residue_r_ind,int *n_blk,int *n_res) {

    using BlockScanT = cub::BlockScan<int, ThreadBlockX>;
    __shared__ typename BlockScanT::TempStorage temp_storage;

    __shared__ typename BlockScanT::TempStorage temp_storage2;

    using BlockReduceT = cub::BlockReduce<int,ThreadBlockX>;
    __shared__ typename BlockReduceT::TempStorage temp_storageR;


    int r_idx = blockDim.x * blockIdx.x + threadIdx.x;

    // if( r_idx >= M) return;

    int nnz = 0;
    int n_padding = 0;
    int row_offset = 0;

    if(r_idx < M)
    {
        row_offset = row_ptr[r_idx];
        n_padding = row_offset % vecLen;
        nnz = row_ptr[r_idx + 1] - row_offset + n_padding;
    }

    int bn = 0;
    int rn = 0;

    if(nnz > N_SEG) {
        bn = nnz / N_SEG;
        int rest = (nnz%N_SEG);
        if(rest >= blockX) bn += 1;
        if(rest % blockX) rn = 1;
    } else {
        if(nnz >= blockX) bn = 1;
        if(nnz % blockX) rn = 1;
    }

    int nReduce = 0;
    if(blockIdx.x > 0) nReduce = (blockIdx.x - 1) / ThreadBlockX + 1;

    int g_bn = 0, g_rn = 0;

    // compute prefix
    int block_prefix_sum = 0;
    BlockScanT(temp_storage).InclusiveSum(bn, g_bn, block_prefix_sum);
    __syncthreads();

    if(threadIdx.x == 0) {
        block_r_ind[blockIdx.x] = block_prefix_sum;
    }
    __threadfence();

    int global_prefix = 0;
    int block_prefix;
    for(int i = 0; i < nReduce; ++i) {
        int e_idx = i * ThreadBlockX + threadIdx.x;
        if(e_idx < blockIdx.x) {
            block_prefix += block_r_ind[e_idx];
        }
    }
    __syncthreads(); __threadfence();
    global_prefix += BlockReduceT(temp_storageR).Sum(block_prefix);
    __syncthreads();

    __shared__ int global_block_sum;
    if(threadIdx.x == 0)
        global_block_sum = global_prefix;
    __syncthreads();

    int block_global_offset = global_block_sum;
    g_bn = g_bn + block_global_offset - bn;

    // residue part
    __syncthreads();

    __threadfence();
    int residue_prefix_sum = 0;
    BlockScanT(temp_storage2).InclusiveSum(rn, g_rn, residue_prefix_sum);
    __syncthreads();

    if(threadIdx.x == 0) {
        residue_r_ind[blockIdx.x] = residue_prefix_sum;
    }

    __threadfence();

    int global_prefix2 = 0;
    for(int i = 0; i < nReduce; ++i) {
        int block_prefix = 0;

        int e_idx = i * ThreadBlockX + threadIdx.x;
        if(e_idx < blockIdx.x) {
            block_prefix = residue_r_ind[e_idx];
        }

        // if(threadIdx.x == 0)
        //     printf("sum[%d]:%d\n",blockIdx.x,sum);
        __syncthreads(); __threadfence();
        global_prefix2 += BlockReduceT(temp_storageR).Sum(block_prefix);
        __syncthreads();
    }

    __shared__ int global_block_sum2;
    if(threadIdx.x == 0)
        global_block_sum2 = global_prefix2;
    __syncthreads();


    // if(threadIdx.x == 0) {
    //     printf("gbprf[%d]:%d\n",global_block_sum);
    // }

    int residue_global_offset = global_block_sum2;

    g_rn = g_rn + residue_global_offset - rn;

    if(r_idx >= M) return;

    for(int i = 0; i < bn; ++i) {
        block_r_ind[g_bn + i] = r_idx;
        st_off[g_bn + i] = row_offset;
        row_offset += N_SEG;
    }
    if(rn > 0)
        residue_r_ind[g_rn] = r_idx;

    if(r_idx == M - 1) {
        n_blk[0] = g_bn + bn;
        n_res[0] = g_rn + rn;
    }

    residue_r_ind[r_idx] = rn;
}
__global__ void RoDe_get_bn_rn(int N_SEG,int blockX,int vecLen,int M,const int *row_ptr,int *bns,int *rns) {
   
    int r_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(r_idx >= M) return;

    int nnz = 0;
    int n_padding = 0;
    int row_offset = 0;

    row_offset = row_ptr[r_idx];
    n_padding = row_offset % vecLen;
    nnz = row_ptr[r_idx + 1] - row_offset + n_padding;
    

    int bn = 0;
    int rn = 0;

    if(nnz > N_SEG) {
        bn = nnz / N_SEG;
        int rest = (nnz%N_SEG);
        if(rest >= blockX) bn += 1;
        if(rest % blockX) rn = 1;
    } else {
        if(nnz >= blockX) bn = 1;
        if(nnz % blockX) rn = 1;
    }

    bns[r_idx] = bn;
    rns[r_idx] = rn;
}
__global__ void RoDe_fill_data(int N_SEG,int blockX,int vecLen,int M,const int *row_ptr, int* block_r_ind, int *st_off,int *residue_r_ind,int *n_blk,int *n_res){
    int r_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(r_idx >= M) return;

    int nnz = 0;
    int n_padding = 0;
    int row_offset = 0;

    row_offset = row_ptr[r_idx];
    n_padding = row_offset % vecLen;
    nnz = row_ptr[r_idx + 1] - row_offset + n_padding;

    int g_bn = block_r_ind[r_idx];
    int g_rn = residue_r_ind[r_idx];

    __syncthreads();

    if(nnz > N_SEG) {
        block_r_ind[g_bn] = r_idx;
        st_off[g_bn++] = row_offset;
        row_offset = (row_offset + N_SEG) - n_padding;
        nnz -= N_SEG;
    }
    while(nnz > N_SEG) {
        block_r_ind[g_bn] = r_idx;
        st_off[g_bn++] = row_offset;

        row_offset += N_SEG;
        nnz -= N_SEG;
    }

    if(nnz > 0) {
      if(nnz >= blockX){
        block_r_ind[g_bn] = r_idx;
        st_off[g_bn++] = row_offset;
      }
      if(nnz % blockX) {
        residue_r_ind[g_rn++] = r_idx;
      }
    }

    if(r_idx == M-1) {
        n_blk[0] = g_bn;
        n_res[0] = g_rn;
    }

}
void RoDe_preprocess(int N_SEG,int blockX,int vecLen,int M,const int *row_ptr, int* block_r_ind, int *st_off,int *residue_r_ind,int *n_blk,int *n_res) {

    size_t temp_storage_bytes = 0;
    void* d_temp_storage = nullptr;

    RoDe_get_bn_rn<<<(M+1023)/1024,1024>>>(N_SEG,blockX,vecLen,M,row_ptr,residue_r_ind,st_off);

    cub::DeviceScan::ExclusiveSum(d_temp_storage,temp_storage_bytes,residue_r_ind,block_r_ind,M);
    cudaMalloc(&d_temp_storage,temp_storage_bytes);

    cub::DeviceScan::ExclusiveSum(d_temp_storage,temp_storage_bytes,residue_r_ind,block_r_ind,M);
    cub::DeviceScan::ExclusiveSum(d_temp_storage,temp_storage_bytes,st_off,residue_r_ind,M);
    cudaFree(d_temp_storage);

    RoDe_fill_data<<<(M+1023)/1024,1024>>>(N_SEG,blockX,vecLen,M,row_ptr,block_r_ind,st_off,residue_r_ind,n_blk,n_res);

}
void RoDe_preprocess_cpu(int N_SEG,int blockX,int vecLen,int M,const int *row_ptr, int* block_r_ind, int *st_off,int *residue_r_ind,int &n_blk,int &n_res) {
    n_blk = 0; n_res = 0;
    for(int i=0; i < M; ++i) {
        int row_offset = row_ptr[i];
        int n_padding = row_offset % vecLen;
        int nnz = row_ptr[i+1] - row_offset + n_padding;
        
        if(nnz > N_SEG) {
            block_r_ind[n_blk] = i;
            st_off[n_blk ++] = row_offset;

            row_offset = (row_offset + N_SEG) - n_padding;
            nnz -= N_SEG;
        }

    while(nnz > N_SEG) {
        block_r_ind[n_blk] = i;
        st_off[n_blk ++] = row_offset;

        row_offset += N_SEG;
        nnz -= N_SEG;
    }

    if(nnz > 0) {
        if(nnz >= blockX){
            block_r_ind[n_blk] = i;
            st_off[n_blk ++] = row_offset;
        }
        if( nnz % blockX) {
            residue_r_ind[n_res++] = i;
        }
    }
  }
}   

__global__ void sum(int* array,int N) {
    int sum = 0;
    for(int i = 0; i < N; ++i)
        sum += array[i];

    printf("sum: %d\n",sum);
}

int main(int argc,char **argv) {
    
    // cudaSetDevice(0);

    string file_path;
    if(argc < 2) {
        cout<<"No file path, use default file 'mip1' "<<endl;
        file_path = "../../data/mip1/mip1.mtx";
    }
    else {
        file_path = argv[1];
    }

    int ITER = 1;

    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    double gflops = 0.0f;

    // cout<<file_path<<endl;

    SPC::SparseMatrix sm1(file_path,SPC::SORTED,1);

    int * row_offset_h = sm1.RowOffsets();
    int * row_indices_h = sm1.RowIndices();

    sm1.RowDivide2Segment(SEG_LENGTH,4,32);
    
    SPC::CudaSparseMatrix<float> c_sm(sm1);

    int m = c_sm.Rows(), k = c_sm.Columns(), n = BN;

    // printf("row_info: %d\n",m);
    // for(int i = 0; i < 10; ++i)
    //     printf("nnz[%d]:%d\n",i,row_offset_h[i+1]-row_offset_h[i]);

    // // RoDe meta-data
    int *d_blk_rinds, *d_res_rinds;
    int *d_st_offs;
    int *n_blk, *n_res;

    int n_ub = sm1.Nonzeros() / SEG_LENGTH + m;

    cudaMalloc((void**)&d_res_rinds,sizeof(int) * m);
    cudaMalloc((void**)&d_blk_rinds,sizeof(int)*n_ub);
    cudaMalloc((void**)&d_st_offs,sizeof(int)*n_ub);
    cudaMalloc((void**)&n_blk,sizeof(int));
    cudaMalloc((void**)&n_res,sizeof(int));

    // RoDe_preprocess<1024><<<(m+1023)/1024,1024>>>(SEG_LENGTH,32,4,m,c_sm.RowOffsets(),d_blk_rinds,d_st_offs,d_res_rinds,n_blk,n_res);

    int *d_row_indices;
    cudaMalloc((void**)&d_row_indices,sizeof(int)* m);


    float tot_ms;
    cudaEvent_t event1,event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    cudaDeviceSynchronize();
    cudaEventRecord(event1,0);
    for(int i=0; i < ITER; ++i)
       Sputinik_preprocess(m,c_sm.RowOffsets(),d_row_indices);

    cudaEventRecord(event2,0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();
    printf(",%f",tot_ms);

    cudaDeviceSynchronize();
    cudaEventRecord(event1,0);
    for(int i=0; i < ITER; ++i)
        RoDe_preprocess(SEG_LENGTH,32,4,m,c_sm.RowOffsets(),d_blk_rinds,d_st_offs,d_res_rinds,n_blk,n_res);
    cudaEventRecord(event2,0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();

    printf(",%f\n",tot_ms);

    cudaFree(d_res_rinds);
    cudaFree(d_blk_rinds);
    cudaFree(d_st_offs);
    cudaFree(n_res);
    cudaFree(n_blk);

    cudaFree(d_row_indices);


    int *h_row_indices = new int[m];

    int *h_block_row_inds = new int[n_ub];
    int *h_st_offs = new int[n_ub];
    int *h_residue_row_inds = new int[m];
    int h_n_blk,h_n_res;

    auto start_time = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < ITER; ++i) 
        Sputinik_preprocess_cpu(m,sm1.RowOffsets(),h_row_indices);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // std::cout << "Sputnik preprocess time(CPU): " << duration.count() << " microseconds" << std::endl;


    start_time = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITER; ++i) 
        RoDe_preprocess_cpu(SEG_LENGTH,32,4,m,sm1.RowOffsets(),h_block_row_inds,h_st_offs,h_residue_row_inds,h_n_blk,h_n_res);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // std::cout << "RoDe preprocess time: " << duration.count() << " microseconds" << std::endl;


    delete[] h_row_indices;
    delete[] h_block_row_inds;
    delete[] h_st_offs;
    delete[] h_residue_row_inds;

    return 0;
}
// 
