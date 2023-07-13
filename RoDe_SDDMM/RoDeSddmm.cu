#include "RoDeSddmm.h"

#include "cuda_runtime.h"
#include <iostream>
#include "common_utils.h"
#include "matrix_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace SPC;

#define STAGE 4

// for Blcok-part
template <typename LoadType ,int kBlockItemsY, int kBlockItemsK, int kBlockItemsX, int kBlockWidth,int K>
__global__ void __launch_bounds__(kBlockItemsY* kBlockWidth)
SDDMMKernel4Block(int m,int n,int k,const int* __restrict__ row_indices,const int* __restrict__ row_offsets,const int* __restrict__ column_indices,const int* __restrict__ st_offsets,\
    const float* __restrict__ values,const float* __restrict__ lhs_matrix,const float* __restrict__ rhs_matrix,float* __restrict__ out) {
    
    static constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(float);
    static constexpr int kThreadItemsX = kBlockItemsX / kBlockWidth / kValuesPerLoad;
    static constexpr int kThreadItemsK = kBlockItemsK / kBlockWidth / kValuesPerLoad;
    static constexpr int kScalarThreadItemsK = kBlockItemsK / kBlockWidth;
    static constexpr int kScalarThreadItemsX = kBlockItemsX / kBlockWidth;
    typedef typename Value2Index<LoadType>::Index IndexType;


    int m_idx = blockIdx.x * kBlockItemsY + threadIdx.y;
    int n_idx = blockIdx.y * kBlockItemsX;

    if(m_idx >= m) return;

    int r_idx = Load(row_indices + m_idx);

    int row_offset = Load(st_offsets + m_idx);
    int nonzeros = min(Load(row_offsets + r_idx + 1),Load(st_offsets + m_idx + 1)) - row_offset;

    cooperative_groups::thread_block threadblock = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<kBlockWidth> subwarp = cooperative_groups::tiled_partition<kBlockWidth>(threadblock);
        
    // memory aligner:
    static constexpr int kValueAligment = sizeof(LoadType) / sizeof(float);
    static constexpr uint32_t kAlignmentMask = ~(kValueAligment - 1);
    static constexpr int kMaxValuesToMask = kValueAligment - 1;
    static constexpr int kMaskSteps = (kMaxValuesToMask + kBlockWidth - 1) / kBlockWidth;

    int values_to_mask_ = row_offset & (kValueAligment - 1);
    int aligned_nonzeros = nonzeros + values_to_mask_;

    nonzeros = aligned_nonzeros / kBlockItemsX * kBlockItemsX;

    if(n_idx >= nonzeros) return;

    row_offset = row_offset & kAlignmentMask ;

    if(n_idx > 0) {
        values_to_mask_ = 0;
    }

    __shared__ int column_indices_tile_array[kBlockItemsX * kBlockItemsY];
    float values_tile[kScalarThreadItemsX] = {};

    int* column_indices_tile = column_indices_tile_array + kBlockItemsK * threadIdx.y;

    __align__(16) float lhs_fragment[kBlockItemsK / kBlockWidth];
    __align__(16) float rhs_fragment[2* kBlockItemsK * STAGE / kBlockWidth];

    float accumulator_fragment[kBlockItemsX] = {};
    float output_fragment[kBlockItemsX / kBlockWidth];

    Barrier<kBlockItemsY,kBlockWidth> barrier(threadIdx.y);

    // load column indices 
    IndexType* column_indices_tile_ = reinterpret_cast<IndexType*>(column_indices_tile);
    const IndexType* column_indices_= reinterpret_cast<const IndexType*>(column_indices + row_offset + n_idx);
    cooperative_groups::memcpy_async(subwarp,column_indices_tile_,kBlockItemsX / kValuesPerLoad, column_indices_,kBlockItemsX/kValuesPerLoad );
    // cooperative_groups::wait(subwarp);

    LoadType* values_tile_ = reinterpret_cast<LoadType *>(values_tile);
    const LoadType* sparse_values_ = reinterpret_cast<const LoadType*>(values + row_offset + n_idx) + threadIdx.x;
    #pragma unroll
    for(int x_item_idx = 0; x_item_idx < kThreadItemsX; ++ x_item_idx) {
        Store(Load(sparse_values_),values_tile_);
        sparse_values_ += kBlockWidth;
        values_tile_   += kBlockWidth;
    }

    const LoadType *lhs_matrix_ = reinterpret_cast<const LoadType*>(lhs_matrix + r_idx * k) + threadIdx.x;
    LoadType *lhs_fragment_ = reinterpret_cast<LoadType*>(lhs_fragment);

    const LoadType *rhs_matrix_base = reinterpret_cast<const LoadType*>(rhs_matrix) + threadIdx.x;
    LoadType* rhs_fragment_ = reinterpret_cast<LoadType*>(rhs_fragment);

    cooperative_groups::wait(subwarp);

    constexpr int PipeStage = kBlockItemsX / STAGE;
    #pragma unroll
    // for(int kk = k ; kk >= kBlockItemsK; kk -= kBlockItemsK) {
    for(int kk = 0; kk < K / kBlockItemsK; ++kk){
        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++ k_item_idx) {
            Store(Load(lhs_matrix_), lhs_fragment_ + k_item_idx);
            lhs_matrix_ += kBlockWidth;
        }

        int *column_idxs_tile = reinterpret_cast<int*>(column_indices_tile);
        #pragma unroll
        for(int kStage = 0; kStage < STAGE; ++ kStage) {
            const LoadType *rhs_matrix_ = reinterpret_cast<const LoadType*>(rhs_matrix_base) + K / kValuesPerLoad * (*column_idxs_tile);
            
            #pragma unroll
            for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
                int fragment_offset = kStage * kThreadItemsK + k_item_idx;
                Store(Load(rhs_matrix_),rhs_fragment_ + fragment_offset);
                rhs_matrix_ += kBlockWidth;
            }
            ++column_idxs_tile;
        }    

        // asm volatile("":::"memory");
        int write_idx = 1;
        #pragma unroll
        for(int x_item_idx = 0; x_item_idx < PipeStage - 1; ++ x_item_idx) {
            #pragma unroll
            for(int kStage = 0; kStage < STAGE; ++ kStage) {
                const LoadType *rhs_matrix_ = reinterpret_cast<const LoadType*>(rhs_matrix_base) + K / kValuesPerLoad * (*column_idxs_tile);
                
                #pragma unroll
                for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
                    int fragment_offset = (write_idx * STAGE + kStage) * kThreadItemsK + k_item_idx;
                    Store(Load(rhs_matrix_),rhs_fragment_ + fragment_offset);
                    rhs_matrix_ += kBlockWidth;
                }
                ++column_idxs_tile;
            } 
            barrier.Sync();
            // asm volatile("":::"memory");
            // cooperative_groups::wait(subwarp);

            #pragma unroll
            for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++ k_item_idx) {
                const float lhs_value = lhs_fragment[k_item_idx];
                #pragma unroll
                for(int kStage = 0; kStage < STAGE; ++ kStage) {
                    int x_idx = (x_item_idx * STAGE + kStage);
                    const float rhs_value = rhs_fragment[k_item_idx + kScalarThreadItemsK * (kStage + (write_idx^1)*STAGE)];
                    accumulator_fragment[x_idx] += lhs_value * rhs_value;
                }
            }
            write_idx ^= 1;
        } 

        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++ k_item_idx) {
            const float lhs_value = lhs_fragment[k_item_idx];
            #pragma unroll
            for(int kStage = 0; kStage < STAGE; ++ kStage) {
                int x_idx = ((PipeStage - 1) * STAGE + kStage);
                const float rhs_value = rhs_fragment[k_item_idx + kScalarThreadItemsK * (kStage + (write_idx^1)*STAGE)];
                accumulator_fragment[x_idx] += lhs_value * rhs_value;
            }
        } 
        rhs_matrix_base = OffsetCast<const LoadType>(rhs_matrix_base,kBlockItemsK * sizeof(float));
    }

    const uint32_t kShflMask = barrier.ThreadMask();
    #pragma unroll 
    for(int base_idx = 0; base_idx < kScalarThreadItemsX; ++ base_idx) {
        #pragma unroll 
        for(int k_item_idx = 1; k_item_idx < kBlockWidth; k_item_idx *= 2) {
            const int kBoundX = kBlockWidth / ( k_item_idx * 2);
            #pragma unroll 
            for(int x_item_idx = 0; x_item_idx < kBoundX; ++ x_item_idx) {
                const int idx_a = x_item_idx * 2 * kValuesPerLoad * k_item_idx;
                const int idx_b = (x_item_idx *2  + 1) * kValuesPerLoad * k_item_idx;
                const int kStep = Log2(k_item_idx);
                if((threadIdx.x >> kStep)&1){
                    float t = accumulator_fragment[base_idx + idx_a];
                    accumulator_fragment[base_idx + idx_a] = accumulator_fragment[base_idx + idx_b];
                    accumulator_fragment[base_idx + idx_b] = t;
                }
                accumulator_fragment[base_idx + idx_a] += __shfl_xor_sync(kShflMask,accumulator_fragment[base_idx + idx_b],k_item_idx,kBlockWidth);
            }
        }
    }

    #pragma unroll
    for(int out_idx = 0; out_idx < kScalarThreadItemsX; ++ out_idx) {
        output_fragment[out_idx] = accumulator_fragment[out_idx] * values_tile[out_idx];
    }

    float* output_values_ = reinterpret_cast<float*>(out + row_offset + n_idx ) + threadIdx.x * kValuesPerLoad;

    int sp_idx = threadIdx.x * kValuesPerLoad;// + n_idx;
    #pragma unroll
    for(int x_item_idx = 0; x_item_idx < kScalarThreadItemsX; ++ x_item_idx) {
        if(sp_idx >= values_to_mask_) {
            Store(output_fragment[x_item_idx],output_values_);
        }
        ++ output_values_;
        ++ sp_idx;
    }
}

// for Residue-part
template <typename LoadType ,int kBlockItemsY, int kBlockItemsK, int kBlockItemsX, int kBlockWidth,int K>
__global__ void __launch_bounds__(kBlockItemsY* kBlockWidth)
SDDMMKernel4Residue(int m,int n,int k,const int* __restrict__ row_indices,const int* __restrict__ row_offsets,const int* __restrict__ column_indices,\
    const float* __restrict__ values,const float* __restrict__ lhs_matrix,const float* __restrict__ rhs_matrix,float* __restrict__ out) {

    static constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(float);
    static constexpr int kThreadItemsX = kBlockItemsX / kBlockWidth / kValuesPerLoad;
    static constexpr int kThreadItemsK = kBlockItemsK / kBlockWidth / kValuesPerLoad;

    static constexpr int kScalarThreadItemsK = kBlockItemsK / kBlockWidth;
    static constexpr int kScalarThreadItemsX = kBlockItemsX / kBlockWidth;

    static constexpr int kResidueUnroll = 4;
    static constexpr int kResidueOuterLimit = kBlockItemsX / kResidueUnroll;

    int m_idx = blockIdx.x * kBlockItemsY + threadIdx.y;
    int n_idx = blockIdx.y * kBlockItemsX;

    if(m_idx >= m) return;

    m_idx = Load(row_indices + m_idx);

    int row_offset = Load(row_offsets + m_idx);
    int nonzeros = Load(row_offsets + m_idx + 1) - row_offset;

    // memory aligner:
    static constexpr int kValueAligment = sizeof(LoadType) / sizeof(float);
    static constexpr uint32_t kAlignmentMask = ~(kValueAligment - 1);
    static constexpr int kMaxValuesToMask = kValueAligment - 1;
    static constexpr int kMaskSteps = (kMaxValuesToMask + kBlockWidth - 1) / kBlockWidth;

    int values_to_mask_ = row_offset & (kValueAligment - 1);
    int aligned_nonzeros = nonzeros + values_to_mask_;

    if(aligned_nonzeros >= kBlockItemsX) {
        nonzeros = aligned_nonzeros % kBlockItemsX;
        row_offset = (row_offset & kAlignmentMask) + aligned_nonzeros / kBlockItemsX * kBlockItemsX; 
    }

    int Nonzeros = nonzeros;

    nonzeros = nonzeros - threadIdx.x;

    __shared__ int column_indices_tile_array[kBlockItemsX * kBlockItemsY];
    float values_tile[kScalarThreadItemsX] = {};

    int* column_indices_tile = column_indices_tile_array + kBlockItemsX * threadIdx.y;

    float lhs_fragment[kBlockItemsK / kBlockWidth];
    float rhs_fragment[kBlockItemsK / kBlockWidth * kResidueUnroll];

    float output_fragment[kBlockItemsX / kBlockWidth];

    Barrier<kBlockItemsY,kBlockWidth> barrier(threadIdx.y);

    const float* sparse_values_ = values        + row_offset + threadIdx.x;
    const int* column_indices_ = column_indices + row_offset + threadIdx.x;
    const int kDimK_ = k * sizeof(float);

    int* column_idxs_tile = column_indices_tile + threadIdx.x;
    int nonzeros_ = nonzeros;

    #pragma unroll
    for(int x_item_idx = 0; x_item_idx < kScalarThreadItemsX; ++ x_item_idx) {
        if(nonzeros_ <= 0) break;
        // Store(kDimK_ * Load(column_indices_), column_idxs_tile);
        Store(Load(column_indices_), column_idxs_tile);

        nonzeros_ -= kBlockWidth;
        column_idxs_tile += kBlockWidth;
        column_indices_  += kBlockWidth;
    }
    barrier.Sync();

    nonzeros_ = nonzeros;

    #pragma unroll
    for(int x_item_idx = 0; x_item_idx < kScalarThreadItemsX; ++ x_item_idx) {
        if(nonzeros_ <= 0) break;
        Store( Load(sparse_values_), values_tile + x_item_idx);

        nonzeros_ -= kBlockWidth;
        sparse_values_ += kBlockWidth;
    }
    // asm("");

    const LoadType *lhs_matrix_ = reinterpret_cast<const LoadType*>(lhs_matrix + m_idx * k) + threadIdx.x;
    LoadType *lhs_fragment_ = reinterpret_cast<LoadType*>(lhs_fragment);

    const LoadType *rhs_matrix_base = reinterpret_cast<const LoadType*>(rhs_matrix) + threadIdx.x;
    LoadType *rhs_fragment_ = reinterpret_cast<LoadType*>(rhs_fragment);

    const uint32_t kShflMask = barrier.ThreadMask();

    int x_idx = 0, t_idx = 0;
    #pragma unroll
    // for(int kk = k; kk >= kBlockItemsK; kk -= kBlockItemsK) {
    for(int kk = 0; kk < K / kBlockItemsK; ++kk) {
        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++ k_item_idx) {
            Store(Load(lhs_matrix_), lhs_fragment_ + k_item_idx);
            lhs_matrix_ += kBlockWidth;
        }

        //  shfl or smem?
        int *column_idxs_tile = reinterpret_cast<int*>(column_indices_tile);
        nonzeros_ = Nonzeros;

        for(int i = 0; i < Nonzeros / kResidueUnroll; ++i) {
            float accumulator_fragment[kResidueUnroll] = {};

            #pragma unroll
            for(int j = 0; j < kResidueUnroll; ++j) {
                // const LoadType *rhs_matrix_ = OffsetCast<const LoadType>(rhs_matrix_base,*column_idxs_tile);
                // const LoadType *rhs_matrix_ = OffsetCast<const LoadType>(rhs_matrix_base,*column_idxs_tile);
                const LoadType *rhs_matrix_ = reinterpret_cast<const LoadType*>(rhs_matrix_base) + K / kValuesPerLoad * (*column_idxs_tile);

                #pragma unroll
                for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
                    int fragment_offset = j * kThreadItemsK + k_item_idx;
                    Store(Load(rhs_matrix_),rhs_fragment_ + fragment_offset);
                    rhs_matrix_ += kBlockWidth;
                }
                ++ column_idxs_tile;
            }
            #pragma unroll
            for(int j=0; j < kResidueUnroll; ++j) {
                #pragma unroll
                for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++k_item_idx) {
                    accumulator_fragment[j] += lhs_fragment[k_item_idx] * rhs_fragment[j*kScalarThreadItemsK + k_item_idx];
                }
            }


            int tag = threadIdx.x & 3;

            if(tag == 1) {
                float temp = accumulator_fragment[1];
                accumulator_fragment[1] = accumulator_fragment[0];
                accumulator_fragment[0] = temp;
                temp = accumulator_fragment[2];
                accumulator_fragment[2] = accumulator_fragment[3];
                accumulator_fragment[3] = temp;
            }else if(tag == 2){
                float temp = accumulator_fragment[2];
                accumulator_fragment[2] = accumulator_fragment[0];
                accumulator_fragment[0] = temp;
                temp = accumulator_fragment[3];
                accumulator_fragment[3] = accumulator_fragment[1];
                accumulator_fragment[1] = temp;
            }else if(tag == 3) {
                float temp = accumulator_fragment[3];
                accumulator_fragment[3] = accumulator_fragment[0];
                accumulator_fragment[0] = temp;
                temp = accumulator_fragment[2];
                accumulator_fragment[2] = accumulator_fragment[1];
                accumulator_fragment[1] = temp;
            }

            // accumulator_fragment[1] = __shfl_xor_sync(kShflMask,accumulator_fragment[1],1);
            // accumulator_fragment[2] = __shfl_xor_sync(kShflMask,accumulator_fragment[2],2);
            // accumulator_fragment[3] = __shfl_xor_sync(kShflMask,accumulator_fragment[3],3);

            // accumulator_fragment[0] += accumulator_fragment[1] + accumulator_fragment[2] + accumulator_fragment[3];

            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],16);
            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],8);
            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],4);

            #pragma unroll
            for(int j = 1; j < kResidueUnroll; ++j) {
                accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[j],j);
            }

            #pragma unroll
            for(int j = kBlockWidth/2; j >= kResidueUnroll; j /= 2){
                accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],j);
            }


            if(threadIdx.x >= t_idx && threadIdx.x < t_idx + kResidueUnroll){
                output_fragment[x_idx] = accumulator_fragment[0] * values_tile[x_idx];
                ++x_idx;
            }

            t_idx = (t_idx + kResidueUnroll) % kBlockWidth;
        }

        if( (Nonzeros & 3) >= 2) {
            float accumulator_fragment[2] = {};

            #pragma unroll
            for(int j = 0; j < 2; ++j) {
                // const LoadType *rhs_matrix_ = OffsetCast<const LoadType>(rhs_matrix_base,*column_idxs_tile);
                const LoadType *rhs_matrix_ = reinterpret_cast<const LoadType*>(rhs_matrix_base) + K / kValuesPerLoad * (*column_idxs_tile);

                #pragma unroll
                for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
                    int fragment_offset = j * kThreadItemsK + k_item_idx;
                    Store(Load(rhs_matrix_),rhs_fragment_ + fragment_offset);
                    rhs_matrix_ += kBlockWidth;
                }
                ++ column_idxs_tile;
            }
            #pragma unroll
            for(int j=0; j < 2; ++j) {
                #pragma unroll
                for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++k_item_idx) {
                    accumulator_fragment[j] += lhs_fragment[k_item_idx] * rhs_fragment[j*kScalarThreadItemsK + k_item_idx];
                }
            }

            int tag = threadIdx.x & 1;
            if(tag == 1) {
                float temp = accumulator_fragment[1];
                accumulator_fragment[1] = accumulator_fragment[0];
                accumulator_fragment[0] = temp;
            }

            accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[1],1);

            #pragma unroll
            for(int j = kBlockWidth/2 ; j >= 2; j /= 2) {
                accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],j);
            }

            // accumulator_fragment[1] = __shfl_xor_sync(kShflMask,accumulator_fragment[1],1);
            // accumulator_fragment[0] += accumulator_fragment[1];

            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],16);
            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],8);
            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],4);
            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],2);

            if( threadIdx.x >= t_idx && threadIdx.x < t_idx + 2) {
                output_fragment[x_idx] = accumulator_fragment[0] * values_tile[x_idx];
                ++x_idx;
            }
            t_idx += 2;
        }

        if(Nonzeros & 1) {
            float accumulator_fragment[1] = {};

            // const LoadType *rhs_matrix_ = OffsetCast<const LoadType>(rhs_matrix_base,*column_idxs_tile);
            const LoadType *rhs_matrix_ = reinterpret_cast<const LoadType*>(rhs_matrix_base) + K / kValuesPerLoad * (*column_idxs_tile);

            #pragma unroll
            for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
                Store(Load(rhs_matrix_),rhs_fragment_ + k_item_idx);
                rhs_matrix_ += kBlockWidth;
            }

            #pragma unroll
            for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++k_item_idx) {
                accumulator_fragment[0] += lhs_fragment[k_item_idx] * rhs_fragment[k_item_idx];
            }

            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],16);
            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],8);
            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],4);
            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],2);
            // accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],1);

            #pragma unroll
            for(int j = kBlockWidth/2 ; j >= 1; j /= 2) {
                accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],j);
            }

            if(threadIdx.x == t_idx) {
                output_fragment[x_idx] = accumulator_fragment[0] * values_tile[x_idx];
                ++x_idx;
            }
        }
        rhs_matrix_base = OffsetCast<const LoadType>(rhs_matrix_base, kBlockItemsK * sizeof(float));
    }  


    float* output_values_ = out + row_offset + threadIdx.x;
    #pragma unroll
    for(int x_item_idx = 0; x_item_idx < kScalarThreadItemsX; ++ x_item_idx) {
        if(nonzeros > 0) {
            Store(output_fragment[x_item_idx],output_values_);
        }
        nonzeros       -= kBlockWidth;
        output_values_ += kBlockWidth;
    }

}

template <typename LoadType ,int kBlockItemsY, int kBlockItemsK, int kBlockItemsX, int kBlockWidth,int K,int SEG_LENGTH>
void RoDeSDDMMKernel_n32(int m1,int m2,int n,int k,const int* __restrict__ row_indices_block,const int* __restrict__ row_indices_residue,const int* __restrict__ st_offsets,const int* __restrict__ row_offsets,const int* __restrict__ column_indices,\
    const float* __restrict__ values,const float* __restrict__ lhs_matrix,const float* __restrict__ rhs_matrix,float* __restrict__ out,cudaStream_t stream1,cudaStream_t stream2) {
    
    dim3 grid_dim1((m1 + kBlockItemsY - 1)/kBlockItemsY, SEG_LENGTH/kBlockItemsX, 1);
    dim3 grid_dim2((m2 + kBlockItemsY - 1)/kBlockItemsY, 1, 1);

    dim3 block_dim(kBlockWidth,kBlockItemsY,1);

    SDDMMKernel4Block<LoadType,kBlockItemsY,kBlockItemsK,kBlockItemsX,kBlockWidth,K><<<grid_dim1,block_dim,0,stream1>>>(m1,n,k,row_indices_block,row_offsets,column_indices,st_offsets,values,lhs_matrix,rhs_matrix,out);
    SDDMMKernel4Residue<LoadType,kBlockItemsY,kBlockItemsK,kBlockItemsX,kBlockWidth,K><<<grid_dim2,block_dim,0,stream2>>>(m2,n,k,row_indices_residue,row_offsets,column_indices,values,lhs_matrix,rhs_matrix,out);

}

void RoDeSDDMM_n32(int m1,int m2,int n,int k,const int* __restrict__ row_indices_block,const int* __restrict__ row_indices_residue,const int* __restrict__ st_offsets,const int* __restrict__ row_offsets,const int* __restrict__ column_indices,\
    const float* __restrict__ values,const float* __restrict__ lhs_matrix,const float* __restrict__ rhs_matrix,float* __restrict__ out,cudaStream_t stream1,cudaStream_t stream2)  {
    RoDeSDDMMKernel_n32<float4,4,32,32,8,32,512>(m1,m2,n,k,row_indices_block,row_indices_residue,st_offsets,row_offsets,column_indices,values,lhs_matrix,rhs_matrix,out,stream1,stream2);
}


template <typename LoadType ,int kBlockItemsY, int kBlockItemsK, int kBlockItemsX, int kBlockWidth,int K,int SEG_LENGTH>
void RoDeSDDMMKernel_n128(int m1,int m2,int n,int k,const int* __restrict__ row_indices_block,const int* __restrict__ row_indices_residue,const int* __restrict__ st_offsets,const int* __restrict__ row_offsets,const int* __restrict__ column_indices,\
    const float* __restrict__ values,const float* __restrict__ lhs_matrix,const float* __restrict__ rhs_matrix,float* __restrict__ out,cudaStream_t stream1,cudaStream_t stream2) {
    
    dim3 grid_dim1((m1 + kBlockItemsY - 1)/kBlockItemsY, SEG_LENGTH/kBlockItemsX, 1);
    dim3 grid_dim2((m2 + kBlockItemsY - 1)/kBlockItemsY, 1, 1);

    dim3 block_dim1(kBlockWidth,kBlockItemsY,1);
    dim3 block_dim2(kBlockItemsX,kBlockItemsY,1);

    SDDMMKernel4Block<LoadType,kBlockItemsY,kBlockItemsK,kBlockItemsX,kBlockWidth,K><<<grid_dim1,block_dim1,0,stream1>>>(m1,n,k,row_indices_block,row_offsets,column_indices,st_offsets,values,lhs_matrix,rhs_matrix,out);
    SDDMMKernel4Residue<LoadType,kBlockItemsY,K,kBlockItemsX,kBlockItemsX,K><<<grid_dim2,block_dim2,0,stream2>>>(m2,n,k,row_indices_residue,row_offsets,column_indices,values,lhs_matrix,rhs_matrix,out);

}

void RoDeSDDMM_n128(int m1,int m2,int n,int k,const int* __restrict__ row_indices_block,const int* __restrict__ row_indices_residue,const int* __restrict__ st_offsets,const int* __restrict__ row_offsets,const int* __restrict__ column_indices,\
    const float* __restrict__ values,const float* __restrict__ lhs_matrix,const float* __restrict__ rhs_matrix,float* __restrict__ out,cudaStream_t stream1,cudaStream_t stream2)  {

    RoDeSDDMMKernel_n128<float4,4,32,32,8,128,32>(m1,m2,n,k,row_indices_block,row_indices_residue,st_offsets,row_offsets,column_indices,values,lhs_matrix,rhs_matrix,out,stream1,stream2);
}
