#ifndef SPC_COMMON_UTILS_H_
#define SPC_COMMON_UTILS_H_

#include "basic_utils.h"
namespace SPC {

// barrier *******************************************************

__device__ constexpr uint32_t StaticPow(uint32_t base, uint32_t exponent) {
  return exponent == 0 ? 1 : base * StaticPow(base, exponent - 1);
}

template <int kBlockItemsY, int kBlockWidth>
struct Barrier {
  static constexpr int kThreadsPerBlock = kBlockItemsY * kBlockWidth;

  static_assert(kThreadsPerBlock % 32 == 0,
                "The thread-block size must be divisible by the warp size.");
  static_assert(kThreadsPerBlock > 0, "The thread-block size must be nonzero.");

  static constexpr int kThreadsPerOutputTile = kBlockWidth;

  static_assert((kThreadsPerOutputTile % 2) == 0 ||
                    (kThreadsPerOutputTile == 1),
                "The number of threads collaborating on a tile must be "
                "a multiple of two or all threads must be independent.");

  static_assert((kThreadsPerBlock == 32) || (kThreadsPerOutputTile >= 32) ||
                    (kThreadsPerOutputTile == 1),
                "Independent warps must be in separate thread-blocks "
                "when using a subwarp tiling.");

  #if __CUDA_ARCH__ >= 700
    uint32_t thread_mask = 0xffffffff;
  #endif

  __device__ __forceinline__ Barrier(int thread_idx_y) {
    #if __CUDA_ARCH__ >= 700

        if ((kThreadsPerOutputTile < 32) && (kThreadsPerOutputTile > 1)) {
          constexpr uint32_t kBaseSubwarpMask =
              StaticPow(2, kThreadsPerOutputTile) - 1;
          thread_mask = kBaseSubwarpMask << (thread_idx_y * kThreadsPerOutputTile);
        }
    #endif
  }

  __device__ __forceinline__ void Sync() {
    #if __CUDA_ARCH__ >= 700
        if (kThreadsPerOutputTile > 32) {
          __syncthreads();
        } else if (kThreadsPerOutputTile > 1) {
          __syncwarp(thread_mask);
        }
    #else
        if (kThreadsPerOutputTile > 32) {
          __syncthreads();
        }
    #endif
  }
  __device__ __forceinline__ uint32_t ThreadMask() const {
    #if __CUDA_ARCH__ >= 700
        return thread_mask;
    #else
        return 0xffffffff;
    #endif
  }
};

// vector compute ****************************************************
template <typename Value>
struct VectorCompute {
  typedef typename TypeUtils<Value>::Accumulator Accumulator;

  static __device__ __forceinline__ void FMA(float x1, Value x2,
                                             Accumulator *out);

  // Complementary index type to our load type.
  typedef typename Value2Index<Value>::Index Index;

  static __device__ __forceinline__ void Mul(int, Index x2, Index *out);

  static __device__ __forceinline__ void Dot(Value x1, Value x2,
                                             Accumulator *out);
};

template <>
struct VectorCompute<float> {
  static __device__ __forceinline__ void FMA(float x1, float x2, float *out) {
    out[0] += x1 * x2;
  }

  static __device__ __forceinline__ void Mul(int x1, int x2, int *out) {
    out[0] = x1 * x2;
  }

  static __device__ __forceinline__ void Dot(float x1, float x2, float *out) {
    out[0] += x1 * x2;
  }
};

template <>
struct VectorCompute<float2> {
  static __device__ __forceinline__ void FMA(float x1, float2 x2, float2 *out) {
    out[0].x += x1 * x2.x;
    out[0].y += x1 * x2.y;
  }

  static __device__ __forceinline__ void Mul(int x1, int2 x2, int2 *out) {
    out[0].x = x1 * x2.x;
    out[0].y = x1 * x2.y;
  }

  static __device__ __forceinline__ void Dot(float2 x1, float2 x2, float *out) {
    out[0] += x1.x * x2.x;
    out[0] += x1.y * x2.y;
  }
};

template <>
struct VectorCompute<float4> {
  static __device__ __forceinline__ void FMA(float x1, float4 x2, float4 *out) {
    out[0].x += x1 * x2.x;
    out[0].y += x1 * x2.y;
    out[0].z += x1 * x2.z;
    out[0].w += x1 * x2.w;
  }

  static __device__ __forceinline__ void Mul(int x1, int4 x2, int4 *out) {
    out[0].x = x1 * x2.x;
    out[0].y = x1 * x2.y;
    out[0].z = x1 * x2.z;
    out[0].w = x1 * x2.w;
  }

  static __device__ __forceinline__ void Dot(float4 x1, float4 x2, float *out) {
    out[0] += x1.x * x2.x;
    out[0] += x1.y * x2.y;
    out[0] += x1.z * x2.z;
    out[0] += x1.w * x2.w;
  }
};

template <>
struct VectorCompute<double4> {
  static __device__ __forceinline__ void FMA(double x1, double4 x2, double4 *out) {
    out[0].x += x1 * x2.x;
    out[0].y += x1 * x2.y;
    out[0].z += x1 * x2.z;
    out[0].w += x1 * x2.w;
  }

  static __device__ __forceinline__ void Mul(int x1, int4 x2, int4 *out) {
    out[0].x = x1 * x2.x;
    out[0].y = x1 * x2.y;
    out[0].z = x1 * x2.z;
    out[0].w = x1 * x2.w;
  }

  static __device__ __forceinline__ void Dot(double4 x1, double4 x2, double *out) {
    out[0] += x1.x * x2.x;
    out[0] += x1.y * x2.y;
    out[0] += x1.z * x2.z;
    out[0] += x1.w * x2.w;
  }
};

template <>
struct VectorCompute<double> {
  static __device__ __forceinline__ void FMA(double x1, double x2, double *out) {
    out[0] += x1 * x2;
  }

  static __device__ __forceinline__ void Mul(int x1, int x2, int *out) {
    out[0] = x1 * x2;
  }

  static __device__ __forceinline__ void Dot(double x1, double x2, double *out) {
    out[0] += x1 * x2;
  }
};

template <>
struct VectorCompute<half2> {
  static __device__ __forceinline__ void FMA(float x1, half2 x2, float2 *out) {
    float2 x2_f2 = __half22float2(x2);
    VectorCompute<float2>::FMA(x1, x2_f2, out);
  }

  static __device__ __forceinline__ void Mul(int x1, short2 x2, short2 *out) {
    out[0].x = static_cast<short>(x1 * x2.x);
    out[0].y = static_cast<short>(x1 * x2.y);
  }
};

template <>
struct VectorCompute<half4> {
  static __device__ __forceinline__ void FMA(float x1, half4 x2, float4 *out) {
    float2 x2x_f2 = __half22float2(x2.x);
    float2 x2y_f2 = __half22float2(x2.y);
    float4 x2_f4 = make_float4(x2x_f2.x, x2x_f2.y, x2y_f2.x, x2y_f2.y);
    VectorCompute<float4>::FMA(x1, x2_f4, out);
  }

  static __device__ __forceinline__ void Mul(int x1, short4 x2, short4 *out) {
    VectorCompute<half2>::Mul(x1, x2.x, &out[0].x);
    VectorCompute<half2>::Mul(x1, x2.y, &out[0].y);
  }
};

template <>
struct VectorCompute<half8> {
  static __device__ __forceinline__ void FMA(float x1, half8 x2, float4 *out) {
    half4 x2x_h4;
    x2x_h4.x = x2.x;
    x2x_h4.y = x2.y;
    VectorCompute<half4>::FMA(x1, x2x_h4, out);
    half4 x2y_h4;
    x2y_h4.x = x2.z;
    x2y_h4.y = x2.w;
    VectorCompute<half4>::FMA(x1, x2y_h4, out + 1);
  }

  static __device__ __forceinline__ void Mul(int x1, short8 x2, short8 *out) {
    VectorCompute<half2>::Mul(x1, x2.x, &out[0].x);
    VectorCompute<half2>::Mul(x1, x2.y, &out[0].y);
    VectorCompute<half2>::Mul(x1, x2.z, &out[0].z);
    VectorCompute<half2>::Mul(x1, x2.w, &out[0].w);
  }
};

//  tiling **********************************
template <typename OutType, typename InType>
__device__ __forceinline__ OutType* OffsetCast(InType* ptr, int offset) {
  return reinterpret_cast<OutType*>(
      const_cast<char*>(reinterpret_cast<const char*>(ptr)) + offset);
}

template <int kBlockItemsY, int kBlockItemsK, int kBlockItemsX>
struct TilingUtils {
  static __device__ __forceinline__ int IndexM() {
    return blockIdx.x * kBlockItemsY + threadIdx.y;
  }

  static __device__ __forceinline__ int IndexN() {
    return blockIdx.y * kBlockItemsX;
  }

  template <typename T>
  static __device__ __forceinline__ T* MaybeOffset(T* ptr, int off) {
    return ptr + off;
  }
};

template <int kBlockItemsK, int kBlockItemsX>
struct TilingUtils<1, kBlockItemsK, kBlockItemsX> {
  static __device__ __forceinline__ int IndexM() { return blockIdx.x; }

  static __device__ __forceinline__ int IndexN() {
    return blockIdx.y * kBlockItemsX;
  }

  template <typename T>
  static __device__ __forceinline__ T* MaybeOffset(T* ptr, int /* unused */) {
    return ptr;
  }
};

// memory aligner *********************************
template <typename Value,int kBlockWidth> 
struct MemoryAligner {
  typedef typename TypeUtils<Value>::ScalarValue ScalarValue;

  typedef typename Value2Index<ScalarValue>::Index ScalarIndex;

  static constexpr int kValueAlignment = sizeof(Value) / sizeof(ScalarValue);

  static constexpr uint32_t kAlignmentMask = ~(kValueAlignment - 1);

  static constexpr int kMaxValuesToMask = kValueAlignment - 1;

  static constexpr int kMaskSteps =
      (kMaxValuesToMask + kBlockWidth - 1) / kBlockWidth;

  typedef typename std::conditional<std::is_same<ScalarValue,double>::value,double,float>::type Ftype;
  
  int row_offset_;


  int nonzeros_;


  int values_to_mask_;


  __device__ __forceinline__ MemoryAligner(int row_offset, int nonzeros) {
    row_offset_ = row_offset;
    nonzeros_ = nonzeros;
    values_to_mask_ = row_offset & (kValueAlignment - 1);
  }
  __device__ __forceinline__ int AlignedRowOffset() {
    return row_offset_ & kAlignmentMask;
  }

  __device__ __forceinline__ int AlignedNonzeros() {
    return nonzeros_ + values_to_mask_;
  }

  __device__ __forceinline__ void MaskPrefix(
      ScalarValue* values_tile_sv, ScalarIndex* column_indices_tile_si) {
    // NOTE: The below masking code is data type agnostic. Cast input pointers
    // to float/int so that we efficiently operate on 4-byte words.
    Ftype* values_tile = reinterpret_cast<Ftype*>(values_tile_sv);
    int* column_indices_tile = reinterpret_cast<int*>(column_indices_tile_si);

    int mask_idx = threadIdx.x;
    #pragma unroll
    for (int mask_step = 0; mask_step < kMaskSteps; ++mask_step) {
      if (mask_idx < values_to_mask_) {
        values_tile[mask_idx] = 0.0f;
        column_indices_tile[mask_idx] = 0;
        mask_idx += kBlockWidth;
      }
    }
  }
};

template <int kBlockWidth>
struct MemoryAligner<float,kBlockWidth> {
  int row_offset_;
  int nonzeros_;
  __device__ __forceinline__ MemoryAligner(int row_offset, int nonzeros) {
    row_offset_ = row_offset;
    nonzeros_ = nonzeros;
  }

  __device__ __forceinline__ int AlignedRowOffset() { return row_offset_; }

  __device__ __forceinline__ int AlignedNonzeros() { return nonzeros_; }

  __device__ __forceinline__ void MaskPrefix(float*, int*) { /* noop */}

};

template <int kBlockWidth>
struct MemoryAligner<half2,kBlockWidth>{
  int row_offset_;
  int nonzeros_;

  __device__ __forceinline__ MemoryAligner(int row_offset, int nonzeros) {
    row_offset_ = row_offset;
    nonzeros_ = nonzeros;
  }

  __device__ __forceinline__ int AlignedRowOffset() { return row_offset_; }

  __device__ __forceinline__ int AlignedNonzeros() { return nonzeros_; }

  __device__ __forceinline__ void MaskPrefix(half2*, short2*) { /* noop */
  }
};

__host__ __device__ __forceinline__ int Log2(int x) {
  if (x >>= 1) return Log2(x) + 1;
  return 0;
}

constexpr __host__ __device__ __forceinline__ int Min(int a, int b) {
  return a < b ? a : b;
}
}

#endif