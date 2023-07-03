#ifndef SPC_BASIC_UTILS_H_
#define SPC_BASIC_UTILS_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>

namespace SPC {

typedef __half half;
typedef __half2 half2;

struct __align__(8) half4 {
    half2 x, y;
};

struct __align__(16) half8 {
    half2 x, y, z, w;
};

struct __align__(8) short4 {
    short2 x, y;
};

struct __align__(16) short8 {
    short2 x, y, z, w;
};

template <typename Value>
struct TypeUtils {
  static constexpr int kElementsPerScalar = 1;

  static constexpr __device__ __forceinline__ bool IsMixed() { return false; }

  typedef Value Accumulator;

  typedef float ScalarValue;
};

template <>
struct TypeUtils<half2> {
  static constexpr int kElementsPerScalar = 2;
  static constexpr __device__ __forceinline__ bool IsMixed() { return true; }

  typedef float2 Accumulator;
  typedef half2 ScalarValue;
};

template <>
struct TypeUtils<half4> {
  static constexpr int kElementsPerScalar = 2;
  static constexpr __device__ __forceinline__ bool IsMixed() { return true; }

  typedef float4 Accumulator;
  typedef half2 ScalarValue;
};

template <>
struct TypeUtils<half8> {
  static constexpr int kElementsPerScalar = 2;
  static constexpr __device__ __forceinline__ bool IsMixed() { return true; }

  typedef float4 Accumulator;
  typedef half2 ScalarValue;
};

template <>
struct TypeUtils<double4> {
  static constexpr int kElementsPerScalar = 1;
  static constexpr __device__ __forceinline__ bool IsMixed() { return false; }

  typedef double4 Accumulator;
  typedef double ScalarValue;
};

template <>
struct TypeUtils<double> {
  static constexpr int kElementsPerScalar = 1;
  static constexpr __device__ __forceinline__ bool IsMixed() { return false; }

  typedef double Accumulator;
  typedef double ScalarValue;
};

template <>
struct TypeUtils<double2> {
  static constexpr int kElementsPerScalar = 1;
  static constexpr __device__ __forceinline__ bool IsMixed() { return false; }

  typedef double2 Accumulator;
  typedef double ScalarValue;
};

template <typename Value>
struct Value2Index {
  typedef int Index;
};

template <>
struct Value2Index<float2> {
  typedef int2 Index;
};

template <>
struct Value2Index<float4> {
  typedef int4 Index;
};

template <>
struct Value2Index<double2> {
  typedef int2 Index;
};

template <>
struct Value2Index<double4> {
  typedef int4 Index;
};

template <>
struct Value2Index<half2> {
  typedef short2 Index;
};

template <>
struct Value2Index<half4> {
  typedef short4 Index;
};

template <>
struct Value2Index<half8> {
  typedef short8 Index;
};

template <typename To, typename From>
__device__ __forceinline__ void Convert(const From *in, To *out) {
  *out = *reinterpret_cast<const To *>(in);
}

__device__ __forceinline__ void Convert(const float *in, half2 *out) {
  // Convert two 32-bit floats into 16-bit floats and pack into
  // a single half2.
  *out = __float22half2_rn(*reinterpret_cast<const float2 *>(in));
}

__device__ __forceinline__ void Convert(const float *in, half4 *out) {
  // Convert four 32-bit floats into 16-bit floats and pack into
  // a single half4.
  const float2 *in_f2 = reinterpret_cast<const float2 *>(in);
  out->x = __float22half2_rn(in_f2[0]);
  out->y = __float22half2_rn(in_f2[1]);
}

__device__ __forceinline__ void Convert(const float *in, half8 *out) {
  // Convert 8 32-bit floats into 16-bits floats and pack into
  // a single half8
  const float2 *in_f2 = reinterpret_cast<const float2 *>(in);
  out->x = __float22half2_rn(in_f2[0]);
  out->y = __float22half2_rn(in_f2[1]);
  out->z = __float22half2_rn(in_f2[2]);
  out->w = __float22half2_rn(in_f2[3]);
}

__device__ __forceinline__ void Convert(const short2 *x, int *out) {
  // Extract two 16-bit integers into 2 32-bit integers. Useful for
  // all variants of the kernels with low precision inputs. To
  // support a wide enough range of input matrix sizes, we need to
  // use 32-bits for all offsets derived from 16-bit indices.
  out[0] = static_cast<int>(x->x);
  out[1] = static_cast<int>(x->y);
}

__device__ __forceinline__ void Convert(const short4 *x, int *out) {
  Convert(&x->x, out);
  Convert(&x->y, out + 2);
}

__device__ __forceinline__ void Convert(const short2 x, int *out) {
  Convert(&x, out);
}

__device__ __forceinline__ void Convert(short4 x, int *out) {
  Convert(&x.x, out);
  Convert(&x.y, out + 2);
}

__device__ __forceinline__ void Convert(const half2 *x, float *out) {
  // Extract two 16-bit IEEE floating-point values into two 32-bit
  // IEEE floating-point values. Useful for pseudo-fp16 kernels.
  float2 tmp = __half22float2(*x);
  out[0] = tmp.x;
  out[1] = tmp.y;
}

__device__ __forceinline__ void Convert(const float *in, double *out) {
  *out = *reinterpret_cast<const double *>(in);
}

template <class To, class From>
__device__ __forceinline__ To BitCast(const From& src) noexcept {
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
__device__ __forceinline__ void Store(const T& value, T* ptr) {
  *ptr = value;
}

__device__ __forceinline__ void Store(const half8& value, half8* ptr) {
  *reinterpret_cast<float4*>(ptr) = BitCast<float4>(value);
}

__device__ __forceinline__ void Store(const half4& value, half4* ptr) {
  *reinterpret_cast<float2*>(ptr) = BitCast<float2>(value);
}

__device__ __forceinline__ void Store(const short8& value, short8* ptr) {
  *reinterpret_cast<int4*>(ptr) = BitCast<int4>(value);
}

__device__ __forceinline__ void Store(const short4& value, short4* ptr) {
  *reinterpret_cast<int2*>(ptr) = BitCast<int2>(value);
}

template <typename T>
__device__ __forceinline__ T Load(const T* address) {
  return __ldg(address);
}

__device__ __forceinline__ double4 Load(const double4* address) {
  return *address;
}

__device__ __forceinline__ half4 Load(const half4* address) {
  float2 x = __ldg(reinterpret_cast<const float2*>(address));
  return BitCast<half4>(x);
}

__device__ __forceinline__ half8 Load(const half8* address) {
  float4 x = __ldg(reinterpret_cast<const float4*>(address));
  return BitCast<half8>(x);
}

__device__ __forceinline__ short4 Load(const short4* address) {
  int2 x = __ldg(reinterpret_cast<const int2*>(address));
  return BitCast<short4>(x);
}

__device__ __forceinline__ short8 Load(const short8* address) {
  int4 x = __ldg(reinterpret_cast<const int4*>(address));
  return BitCast<short8>(x);
}
}

#define CUDA_CALL(code)                                     \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    if(status!=cudaSuccess) std::cout<<"CUDA Error: " << err; \
  } while (0)


#endif