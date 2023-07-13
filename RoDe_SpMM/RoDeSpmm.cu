#include "RoDeSpmm.h"

#include "basic_utils.h"
#include "cuda_runtime.h"
#include <iostream>
#include "common_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace SPC;

template <typename ScalarValue,typename SparseValue,typename DenseValue, int kBlockItemsY,int kBlockItemsK,int kBlockItemsX,int kBlockWidth,int kResidueUnroll,int STAGE = 8> 
struct SparseKernel {

    typedef int ScalarIndex;

    static constexpr int kSparseValuesPerLoad = sizeof(SparseValue) / sizeof(ScalarValue);
    
    static constexpr int kThreadItemsX = kBlockItemsX / kBlockWidth ;
    static constexpr int kThreadItemsK = kBlockItemsK / kBlockWidth / kSparseValuesPerLoad ;

    static constexpr int kDenseValuesPerLoad = sizeof(DenseValue) / sizeof(ScalarValue);
    static constexpr int kDenseThreadItemsX = kBlockItemsX / kDenseValuesPerLoad / kBlockWidth;

    static constexpr int kElementsPerScalar = 1;
    
    typedef typename Value2Index<SparseValue>::Index SparseIndex;

    typedef SPC::Barrier<kBlockItemsY,kBlockWidth> Barrier;

    typedef typename SPC::TypeUtils<DenseValue>::Accumulator Accumulator;

    static constexpr int kResidueOuterLimit = kBlockItemsK / kResidueUnroll;
    static constexpr int kResidueInnerLimit = kResidueUnroll;

    static constexpr int kDenseFragmentSize = kElementsPerScalar * kBlockItemsK * kBlockItemsX / kBlockWidth;
    static constexpr int kOutputFragmentSize = kBlockItemsX * kElementsPerScalar / kBlockWidth;
    static constexpr int kTileSize = kBlockItemsY * kBlockItemsK ;
    
    static __device__ __forceinline__
    void Kernel4Residue(int m,int k,int n,const ScalarValue* __restrict__ values,const int * __restrict__ column_indices,const int * __restrict__ row_offsets,const int* __restrict__ row_indices,const ScalarValue * B,ScalarValue *C) {
       
        #ifdef THREADBLOCK_SWIZZLE
            int m_idx = blockIdx.y * kBlockItemsY + threadIdx.y;
            int n_idx = blockIdx.x * kBlockItemsX;
        #else
            int m_idx = blockIdx.x * kBlockItemsY + threadIdx.y;
            int n_idx = blockIdx.y * kBlockItemsX ;
        #endif

        if(m_idx >= m) return;
        
        m_idx = Load(row_indices + m_idx);

        int row_offset = Load(row_offsets + m_idx);
        int nonzeros = Load(row_offsets + m_idx + 1) - row_offset;

        __shared__  ScalarValue values_tile_array[kTileSize];
        __shared__  ScalarIndex column_indices_tile_array[kTileSize];

        ScalarValue * values_tile = values_tile_array + kBlockItemsK * threadIdx.y;
        ScalarIndex * column_indices_tile = column_indices_tile_array + kBlockItemsK * threadIdx.y;

        // memory aligner:
        static constexpr int kValueAligment = sizeof(SparseValue) / sizeof(ScalarValue);
        static constexpr uint32_t kAlignmentMask = ~(kValueAligment - 1);
        static constexpr int kMaxValuesToMask = kValueAligment - 1;
        static constexpr int kMaskSteps = (kMaxValuesToMask + kBlockWidth - 1) / kBlockWidth;

        int values_to_mask_ = row_offset & (kValueAligment - 1);

        int aligned_nonzeros = nonzeros + values_to_mask_;
        int blocked_nonzeros = nonzeros / kBlockItemsK * kBlockItemsK;

        bool atomicFlag = false;

        if(aligned_nonzeros >= kBlockItemsK) {
            nonzeros = aligned_nonzeros % kBlockItemsK;
            row_offset = (row_offset & kAlignmentMask) + aligned_nonzeros / kBlockItemsK * kBlockItemsK;
            atomicFlag = true;
        }

        Barrier barrier(threadIdx.y);

        __align__(16) ScalarValue dense_matrix_fragment[kDenseFragmentSize] = {};
        __align__(16) ScalarValue output_fragment[kOutputFragmentSize] = {};

        const DenseValue *dense_matrix = reinterpret_cast<const DenseValue*>(B + n_idx) + threadIdx.x;
        DenseValue *dense_fragment = reinterpret_cast<DenseValue*>(dense_matrix_fragment);  

        SparseValue* sparse_values_tile = reinterpret_cast<SparseValue*>(values_tile) + threadIdx.x;
        SparseIndex* sparse_columns_tile = reinterpret_cast<SparseIndex*>(column_indices_tile) + threadIdx.x;

        SparseValue* sparse_values_tile_ = sparse_values_tile;
        SparseIndex* sparse_columns_tile_ = sparse_columns_tile;
        if(kResidueUnroll > 1) {
            const ScalarValue kZeroValues[kSparseValuesPerLoad] = {};
            const ScalarIndex kZeroIndices[kSparseValuesPerLoad] = {};
            #pragma unroll
            for(int i=0; i < kThreadItemsK; ++i) {
                Store(*reinterpret_cast<const SparseIndex*>(kZeroIndices),sparse_columns_tile_);
                Store(*reinterpret_cast<const SparseValue*>(kZeroValues), sparse_values_tile_);

                sparse_values_tile_ += kBlockWidth;
                sparse_columns_tile_ += kBlockWidth;
            }

            barrier.Sync();
        }

        constexpr int kResidueUpdateStrideValue = -1 * static_cast<int>(sizeof(ScalarValue)) * (kSparseValuesPerLoad - 1);
        const int kResidueUpdateValue = static_cast<int>(threadIdx.x) * kResidueUpdateStrideValue;

        constexpr int kResidueUpdateStrideIndex = -1 * static_cast<int>(sizeof(ScalarIndex)) * (kSparseValuesPerLoad - 1);
        const int kResidueUpdateIndex = static_cast<int>(threadIdx.x) * kResidueUpdateStrideIndex;

        const ScalarValue *sparse_values = reinterpret_cast<const ScalarValue*>(values + row_offset + threadIdx.x);
        const ScalarIndex *sparse_columns = reinterpret_cast<const ScalarIndex*>(column_indices + row_offset + threadIdx.x);

        ScalarIndex* sparse_columns_tile__ = OffsetCast<ScalarIndex>(sparse_columns_tile,kResidueUpdateIndex);
        ScalarValue* sparse_values_tile__ = OffsetCast<ScalarValue>(sparse_values_tile,kResidueUpdateValue);

        constexpr int kScalarThreadItemsK = kBlockItemsK / kBlockWidth;

        int nonzeros_ = nonzeros;
        #pragma unroll
        for(int i = 0; i < kScalarThreadItemsK; ++i) {
            if(nonzeros_ <= static_cast<int>(threadIdx.x)) break;

            Store(Load(sparse_values),sparse_values_tile__);
            Store(Load(sparse_columns),sparse_columns_tile__);

            sparse_values += kBlockWidth;
            sparse_columns += kBlockWidth;
            sparse_values_tile__ += kBlockWidth;
            sparse_columns_tile__ += kBlockWidth;

            nonzeros_ -= kBlockWidth;
        }
        asm("");
        barrier.Sync();

        const ScalarIndex *dense_row_offsets = column_indices_tile;
        #pragma unroll
        for(int i = 0; i < kResidueOuterLimit; ++i) {
            if(nonzeros <= 0) break;

            #pragma unroll
            for(int j = 0; j < kResidueInnerLimit; ++j){
                const int k_item_idx = i * kResidueInnerLimit + j;

                ScalarIndex scaled_indices = dense_row_offsets[0] * n * sizeof(ScalarValue);
                ScalarValue lhs_values  = values_tile[k_item_idx];

                const DenseValue* matrix__ = SPC::OffsetCast<const DenseValue>(dense_matrix,scaled_indices);

                #pragma unroll
                for(int l = 0; l < kDenseThreadItemsX; ++l) {
                    ScalarValue *outputs = output_fragment +
                                    l * kDenseValuesPerLoad ;
                    SPC::VectorCompute<DenseValue>::FMA(lhs_values,Load(matrix__),reinterpret_cast<Accumulator*>(outputs));

                    matrix__ += kBlockWidth;
                }
                
                ++dense_row_offsets;
            }
            nonzeros -= kResidueInnerLimit;
        }
        asm("");

        const int output_offset = m_idx * n + n_idx;
        if(atomicFlag) {
            ScalarValue* output_matrix = C + output_offset + threadIdx.x * kDenseValuesPerLoad;
            #pragma unroll
            for(int i = 0; i < kDenseThreadItemsX; ++i) {
                #pragma unroll
                for(int j=0; j < kDenseValuesPerLoad; ++j) {
                    atomicAdd(output_matrix + j, output_fragment[i * kDenseValuesPerLoad +j]);
                }
                output_matrix += kBlockWidth * kDenseValuesPerLoad;
            }
        }
        else {
            DenseValue* output_matrix = reinterpret_cast<DenseValue*>(C + output_offset) + threadIdx.x;
            #pragma unroll
            for(int i = 0; i < kDenseThreadItemsX; ++i) {
                const DenseValue* output_fragment_ = reinterpret_cast<const DenseValue*>(output_fragment);
                *output_matrix = output_fragment_[i];

                output_matrix += kBlockWidth;
            }
        }
    }

    static __device__ __forceinline__
    void Kernel4Block(int m,int k,int n,const ScalarValue* __restrict__ values,const int * __restrict__ column_indices,const int * __restrict__ row_offsets,const int* __restrict__ row_indices,const int* __restrict__ st_offsets,const ScalarValue * B,ScalarValue *C) {
       
        #ifdef THREADBLOCK_SWIZZLE
            int m_idx = blockIdx.y * kBlockItemsY + threadIdx.y;
            int n_idx = blockIdx.x * kBlockItemsX;
        #else
            int m_idx = blockIdx.x * kBlockItemsY + threadIdx.y;
            int n_idx = blockIdx.y * kBlockItemsX ;
        #endif

        if(m_idx >= m) return;
        
        int r_idx = Load(row_indices + m_idx);

        int row_offset = Load(st_offsets + m_idx);
        int nonzeros = min(Load(row_offsets + r_idx+1) ,Load(st_offsets + m_idx + 1)) - row_offset;

        constexpr int kTileSize = kBlockItemsY * kBlockItemsK ;
        __shared__  ScalarValue values_tile_array[2*kTileSize];
        __shared__  ScalarIndex column_indices_tile_array[kTileSize];

        ScalarValue * values_tile = values_tile_array + kBlockItemsK * threadIdx.y;
        ScalarIndex * column_indices_tile = column_indices_tile_array + kBlockItemsK * threadIdx.y;

        // memory aligner:
        static constexpr int kValueAligment = sizeof(SparseValue) / sizeof(ScalarValue);
        static constexpr uint32_t kAlignmentMask = ~(kValueAligment - 1);
        static constexpr int kMaxValuesToMask = kValueAligment - 1;
        static constexpr int kMaskSteps = (kMaxValuesToMask + kBlockWidth - 1) / kBlockWidth;

        int values_to_mask_ = row_offset & (kValueAligment - 1);

        int aligned_nonzeros = nonzeros + values_to_mask_;

        nonzeros = aligned_nonzeros / kBlockItemsK * kBlockItemsK;

        row_offset = row_offset & kAlignmentMask;

        Barrier barrier(threadIdx.y);

        cooperative_groups::thread_block threadblock = cooperative_groups::this_thread_block();
        cooperative_groups::thread_block_tile<kBlockWidth> subwarp = cooperative_groups::tiled_partition<kBlockWidth>(threadblock);
        
        __align__(16) ScalarValue dense_matrix_fragment[kDenseFragmentSize] = {};
        __align__(16) ScalarValue output_fragment[kOutputFragmentSize] = {};

        const DenseValue *dense_matrix = reinterpret_cast<const DenseValue*>(B + n_idx) + threadIdx.x;
        DenseValue *dense_fragment = reinterpret_cast<DenseValue*>(dense_matrix_fragment);  

        const SparseValue *sparse_values = reinterpret_cast<const SparseValue*>(values + row_offset) ;// + threadIdx.x;
        const SparseIndex *sparse_columns = reinterpret_cast<const SparseIndex*>(column_indices + row_offset);// + threadIdx.x;

        SparseValue* sparse_values_tile = reinterpret_cast<SparseValue*>(values_tile); // + threadIdx.x;
        SparseIndex* sparse_columns_tile = reinterpret_cast<SparseIndex*>(column_indices_tile); // + threadIdx.x;

        constexpr int Pipeline_steps = kBlockItemsK / STAGE;

        if(nonzeros >= kBlockItemsK) {
            // Load sparse tile...
            cooperative_groups::memcpy_async(subwarp,sparse_columns_tile,kBlockItemsK / kSparseValuesPerLoad,sparse_columns,kBlockItemsK / kSparseValuesPerLoad);
            cooperative_groups::memcpy_async(subwarp,sparse_values_tile, kBlockItemsK / kSparseValuesPerLoad,sparse_values ,kBlockItemsK / kSparseValuesPerLoad);

            sparse_columns += kBlockItemsK / kSparseValuesPerLoad;
            sparse_values +=  kBlockItemsK / kSparseValuesPerLoad;

            cooperative_groups::wait(subwarp);

            // mask padding part:
            ScalarValue *values_tile_sv = reinterpret_cast<ScalarValue*>(values_tile);
            ScalarIndex *column_indices_tile_si = reinterpret_cast<ScalarIndex*>(column_indices_tile);
            int mask_idx = threadIdx.x;
            #pragma unroll
            for(int i=0; i < kMaskSteps; ++i) {
                if(mask_idx < values_to_mask_) {
                    values_tile_sv[mask_idx] = 0.0f;
                    column_indices_tile_si[mask_idx] = 0;
                    mask_idx += kBlockWidth;
                }
            }
            cooperative_groups::wait(subwarp);

            nonzeros -= kBlockItemsK;

            ScalarValue * values_tile = values_tile_array + kBlockItemsK * threadIdx.y;
            const ScalarIndex *dense_row_offsets = column_indices_tile;
            DenseValue dense_fragment_regs[2][STAGE*kDenseThreadItemsX];

            #pragma unroll
            for(int k_st = 0; k_st < STAGE; ++ k_st) {
                ScalarIndex row_idx = dense_row_offsets[k_st] * n * sizeof(ScalarValue);
                const DenseValue *dense_values = SPC::OffsetCast<const DenseValue>(dense_matrix,row_idx);
                #pragma unroll
                for(int ele_idx = 0; ele_idx < kDenseThreadItemsX; ++ ele_idx) {
                    dense_fragment_regs[0][k_st * kDenseThreadItemsX + ele_idx] = dense_values[ele_idx * kBlockWidth];
                }
            }
            dense_row_offsets += STAGE;

            int row_idxs[STAGE];
            #pragma unroll
            for(int k_st =0; k_st < STAGE ; ++ k_st) {
                row_idxs[k_st] = dense_row_offsets[k_st] * n * sizeof(ScalarValue) ;
            }

            #pragma unroll
            for(int i = 1; i < Pipeline_steps ; ++i) {

                #pragma unroll
                for(int k_st = 0; k_st < STAGE; ++ k_st) {
                    ScalarIndex row_idx = row_idxs[k_st];
                    // ScalarIndex row_idx = dense_row_offsets[k_st] * n;
                    const DenseValue *dense_values = SPC::OffsetCast<const DenseValue>(dense_matrix,row_idx);
                    #pragma unroll
                    for(int ele_idx = 0; ele_idx < kDenseThreadItemsX; ++ ele_idx) {
                        dense_fragment_regs[(i&1)][k_st * kDenseThreadItemsX + ele_idx] = dense_values[ele_idx * kBlockWidth];
                    }
                }
                dense_row_offsets += STAGE;

                // cooperative_groups::wait_prior<STAGE2>(subwarp);
                // cooperative_groups::wait(subwarp);
                barrier.Sync();

                ScalarValue lhs[STAGE];
                int d_idx = (i-1)*STAGE;
                #pragma unroll
                for(int k_st =0; k_st < STAGE; ++ k_st) {
                    lhs[k_st] = values_tile[d_idx + k_st];
                    row_idxs[k_st] = dense_row_offsets[k_st] * n * sizeof(ScalarValue);

                    #pragma unroll
                    for(int j=0; j < kDenseThreadItemsX; ++j)
                    {
                        DenseValue rhs = dense_fragment_regs[(i-1)&1][k_st * kDenseThreadItemsX + j];
                        ScalarValue *outputs = output_fragment + 
                                            j * kDenseValuesPerLoad;
                        SPC::VectorCompute<DenseValue>::FMA(lhs[k_st],rhs,reinterpret_cast<Accumulator*>(outputs));
                    }
                }

            }
            
            // cooperative_groups::wait(subwarp);
            barrier.Sync();

            if(nonzeros >= kBlockItemsK)
            {
                cooperative_groups::memcpy_async(subwarp,sparse_columns_tile,kBlockItemsK / kSparseValuesPerLoad,sparse_columns,kBlockItemsK / kSparseValuesPerLoad);
                sparse_columns += kBlockItemsK / kSparseValuesPerLoad;
                
                cooperative_groups::memcpy_async(subwarp,sparse_values_tile + kTileSize / kSparseValuesPerLoad, kBlockItemsK / kSparseValuesPerLoad,sparse_values, kBlockItemsK / kSparseValuesPerLoad);
                sparse_values += kBlockItemsK / kSparseValuesPerLoad;
            }

            ScalarValue lhs[STAGE];
            int d_idx = (Pipeline_steps - 1)*(STAGE);
            #pragma unroll
            for(int k_st =0; k_st < STAGE; ++ k_st) {
                lhs[k_st] = values_tile[d_idx + k_st];

                #pragma unroll
                for(int j=0; j < kDenseThreadItemsX; ++j)
                {
                    DenseValue rhs = dense_fragment_regs[(Pipeline_steps-1)&1][k_st * kDenseThreadItemsX + j];
                    ScalarValue *outputs = output_fragment + 
                                        j * kDenseValuesPerLoad;
                    SPC::VectorCompute<DenseValue>::FMA(lhs[k_st],rhs,reinterpret_cast<Accumulator*>(outputs));
                }
            }

        }

        int col_off = 1;
        for(;nonzeros >= kBlockItemsK; nonzeros -= kBlockItemsK, col_off ^= 1) {

            cooperative_groups::wait(subwarp);

            //  load B and compute pipeline...
            ScalarValue * values_tile = values_tile_array + kBlockItemsK * threadIdx.y + col_off * kTileSize;
            const ScalarIndex *dense_row_offsets = column_indices_tile;
            
            DenseValue dense_fragment_regs[2][STAGE*kDenseThreadItemsX];

            #pragma unroll
            for(int k_st = 0; k_st < STAGE; ++ k_st) {
                ScalarIndex row_idx = dense_row_offsets[k_st] * n * sizeof(ScalarValue);
                const DenseValue *dense_values = SPC::OffsetCast<const DenseValue>(dense_matrix,row_idx);
                #pragma unroll
                for(int ele_idx = 0; ele_idx < kDenseThreadItemsX; ++ ele_idx) {
                    dense_fragment_regs[0][k_st * kDenseThreadItemsX + ele_idx] = dense_values[ele_idx * kBlockWidth];
                }
            }
            dense_row_offsets += STAGE;

            int row_idxs[STAGE];
            #pragma unroll
            for(int k_st =0; k_st < STAGE ; ++ k_st) {
                row_idxs[k_st] = dense_row_offsets[k_st] * n * sizeof(ScalarValue);
            }

            #pragma unroll
            for(int i = 1; i < Pipeline_steps ; ++i) {

                #pragma unroll
                for(int k_st = 0; k_st < STAGE; ++ k_st) {
                    ScalarIndex row_idx = row_idxs[k_st];
                    // ScalarIndex row_idx = dense_row_offsets[k_st] * n;
                    const DenseValue *dense_values = SPC::OffsetCast<const DenseValue>(dense_matrix,row_idx);
                    #pragma unroll
                    for(int ele_idx = 0; ele_idx < kDenseThreadItemsX; ++ ele_idx) {
                        dense_fragment_regs[(i&1)][k_st * kDenseThreadItemsX + ele_idx] = dense_values[ele_idx * kBlockWidth];
                    }
                }
                dense_row_offsets += STAGE;

                // cooperative_groups::wait_prior<STAGE2>(subwarp);
                // cooperative_groups::wait(subwarp);
                barrier.Sync();

                ScalarValue lhs[STAGE];
                int d_idx = (i-1)*(STAGE);
                #pragma unroll
                for(int k_st =0; k_st < STAGE; ++ k_st) {
                    lhs[k_st] = values_tile[d_idx + k_st];
                    row_idxs[k_st] = dense_row_offsets[k_st] * n * sizeof(ScalarValue);

                    #pragma unroll
                    for(int j=0; j < kDenseThreadItemsX; ++j)
                    {
                        DenseValue rhs = dense_fragment_regs[(i-1)&1][k_st * kDenseThreadItemsX + j];
                        ScalarValue *outputs = output_fragment + 
                                            j * kDenseValuesPerLoad;
                        SPC::VectorCompute<DenseValue>::FMA(lhs[k_st],rhs,reinterpret_cast<Accumulator*>(outputs));
                    }
                }
            }
            
            // cooperative_groups::wait(subwarp);
            barrier.Sync();

            if(nonzeros >= 2 * kBlockItemsK)
            {
                cooperative_groups::memcpy_async(subwarp,sparse_columns_tile,kBlockItemsK / kSparseValuesPerLoad,sparse_columns,kBlockItemsK / kSparseValuesPerLoad);
                sparse_columns += kBlockItemsK / kSparseValuesPerLoad;
                
                cooperative_groups::memcpy_async(subwarp,sparse_values_tile + (col_off^1)*kTileSize / kSparseValuesPerLoad, kBlockItemsK / kSparseValuesPerLoad,sparse_values, kBlockItemsK / kSparseValuesPerLoad);
                sparse_values += kBlockItemsK / kSparseValuesPerLoad;
            }

            ScalarValue lhs[STAGE];
            int d_idx = (Pipeline_steps - 1)*(STAGE);
            #pragma unroll
            for(int k_st =0; k_st < STAGE; ++ k_st) {
                lhs[k_st] = values_tile[d_idx + k_st];

                #pragma unroll
                for(int j=0; j < kDenseThreadItemsX; ++j)
                {
                    DenseValue rhs = dense_fragment_regs[(Pipeline_steps-1)&1][k_st * kDenseThreadItemsX + j];
                    ScalarValue *outputs = output_fragment + 
                                        j * kDenseValuesPerLoad;
                    SPC::VectorCompute<DenseValue>::FMA(lhs[k_st],rhs,reinterpret_cast<Accumulator*>(outputs));
                }
            }
        }

        // Store to C
        const int output_offset = r_idx * n + n_idx;
        ScalarValue* output_matrix = C + output_offset + threadIdx.x * kDenseValuesPerLoad;
        #pragma unroll
        for(int i = 0; i < kDenseThreadItemsX; ++i) {
            #pragma unroll
            for(int j=0; j < kDenseValuesPerLoad; ++j) {
                atomicAdd(output_matrix + j, output_fragment[i * kDenseValuesPerLoad +j]);
            }
            output_matrix += kBlockWidth * kDenseValuesPerLoad;
        }
        
    }

};

template <typename ScalarValue,typename SparseValue,typename DenseValue, int kBlockItemsY,int kBlockItemsK,int kBlockItemsX,int kBlockWidth,int kResidueUnroll,int STAGE> 
__global__ void __launch_bounds__(kBlockItemsY*kBlockWidth)
RoDeComputeKernel1(int m,int k,int n,const ScalarValue* __restrict__ values,const int * __restrict__ column_indices,const int *__restrict__ row_offsets,const int *__restrict__ row_indices,const int* __restrict__ row_seg_st_offsets,const ScalarValue * B,ScalarValue* C) {
    SparseKernel<ScalarValue,SparseValue,DenseValue,kBlockItemsY,kBlockItemsK,kBlockItemsX,kBlockWidth,kResidueUnroll,STAGE>::Kernel4Block(m,k,n,values,column_indices,row_offsets,row_indices,row_seg_st_offsets,B,C);
}

template <typename ScalarValue,typename SparseValue,typename DenseValue, int kBlockItemsY,int kBlockItemsK,int kBlockItemsX,int kBlockWidth,int kResidueUnroll,int STAGE> 
__global__ void __launch_bounds__(kBlockItemsY*kBlockWidth)
RoDeComputeKernel2(int m,int k,int n,const ScalarValue* __restrict__ values,const int * __restrict__ column_indices,const int *__restrict__ row_offsets,const int *__restrict__ row_indices,const ScalarValue * B,ScalarValue* C) {
    SparseKernel<ScalarValue,SparseValue,DenseValue,kBlockItemsY,kBlockItemsK,kBlockItemsX,kBlockWidth,kResidueUnroll,STAGE>::Kernel4Residue(m,k,n,values,column_indices,row_offsets,row_indices,B,C);
}

template <typename ScalarValue,typename SparseValue,typename DenseValue, int kBlockItemsY,int kBlockItemsK,int kBlockItemsX1,int kBlockItemsX2,int kBlockWidth,int kResidueUnroll,int STAGE = 8> 
void RoDeSpmmKernel(int m1,int m2,int k,int n,const ScalarValue* __restrict__ values,const int * __restrict__ column_indices,const int * __restrict__ row_offsets,const int *__restrict__ row_indices1,const int *__restrict__ row_indices2,const int* __restrict__ row_seg_st_offsets,const ScalarValue *B,ScalarValue* C,cudaStream_t stream1,cudaStream_t stream2) {

    #ifdef THREADBLOCK_SWIZZLE
        dim3 grid_dim1((n + kBlockItemsX1 - 1) / kBlockItemsX1,(m1 + kBlockItemsY - 1) / kBlockItemsY,1);
        dim3 grid_dim2((n + kBlockItemsX2 - 1) / kBlockItemsX2,(m2 + kBlockItemsY - 1) / kBlockItemsY,1);
    #else
        dim3 grid_dim1( (m1 + kBlockItemsY - 1) / kBlockItemsY, (n + kBlockItemsX1 - 1)/kBlockItemsX1,1);
        dim3 grid_dim2( (m2 + kBlockItemsY - 1) / kBlockItemsY, (n + kBlockItemsX2 - 1)/kBlockItemsX2,1);
    #endif

    dim3 block_dim( kBlockWidth, kBlockItemsY, 1); 

    RoDeComputeKernel1<ScalarValue,SparseValue,DenseValue,kBlockItemsY,kBlockItemsK,kBlockItemsX1,kBlockWidth,kResidueUnroll,STAGE><<<grid_dim1,block_dim,0,stream1>>>(m1,k,n,
                        values,column_indices,row_offsets,row_indices1,row_seg_st_offsets,B,C);
    RoDeComputeKernel2<ScalarValue,SparseValue,DenseValue,kBlockItemsY,kBlockItemsK,kBlockItemsX2,kBlockWidth,kResidueUnroll,STAGE><<<grid_dim2,block_dim,0,stream2>>>(m2,k,n,
                        values,column_indices,row_offsets,row_indices2,B,C);
}

void RoDeSpmm_n32(int m1,int m2,int k,int n,const float* __restrict__ values,const int * __restrict__ column_indices,const int * __restrict__ row_offsets,const int *__restrict__ row_indices1,const int *__restrict__ row_indices2,\
                const int * __restrict__ row_seg_st_offsets,const float *B,float* C,cudaStream_t stream1,cudaStream_t stream2) {
    RoDeSpmmKernel<float,float4,float4,4,32,32,32,8,4>(m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}

void RoDeSpmm_n128(int m1,int m2,int k,int n,const float* __restrict__ values,const int * __restrict__ column_indices,const int * __restrict__ row_offsets,const int *__restrict__ row_indices1,const int *__restrict__ row_indices2,\
                const int * __restrict__ row_seg_st_offsets,const float *B,float* C,cudaStream_t stream1,cudaStream_t stream2) {
    RoDeSpmmKernel<float,float4,float4,4,32,64,64,8,4>(m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}

void RoDeSpmm_n32(int m1,int m2,int k,int n,const double* __restrict__ values,const int * __restrict__ column_indices,const int * __restrict__ row_offsets,const int *__restrict__ row_indices1,const int *__restrict__ row_indices2,\
                const int * __restrict__ row_seg_st_offsets,const double *B,double* C,cudaStream_t stream1,cudaStream_t stream2) {
    RoDeSpmmKernel<double,double4,double4,4,32,32,32,8,4>(m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}

void RoDeSpmm_n128(int m1,int m2,int k,int n,const double* __restrict__ values,const int * __restrict__ column_indices,const int * __restrict__ row_offsets,const int *__restrict__ row_indices1,const int *__restrict__ row_indices2,\
                const int * __restrict__ row_seg_st_offsets,const double *B,double* C,cudaStream_t stream1,cudaStream_t stream2) {
    RoDeSpmmKernel<double,double4,double4,4,32,64,64,8,4,4>(m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}
