
#ifndef SPUTNIK_SDDMM_ALL_REDUCE_H_
#define SPUTNIK_SDDMM_ALL_REDUCE_H_


#include "common_utils.h"

template <typename LoadType, int kBlockItemsX, int kBlockWidth>
struct AllReduce {
  //
  /// Static members.
  //

  // The number of values that will be loaded per-thread, per-load.
  static constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(float);

  // The number of outputs each thread is responsible for.
  static constexpr int kThreadItemsX = kBlockItemsX / kBlockWidth;

  //
  /// Member variables.
  //

  // Thread mask used for warp shuffle operations.
  const uint32_t kShflMask;

  // Register file fragment storing the thread local partial results.
  float* inputs;

  // Registe file fragment for storing each threads results.
  float* outputs;

  __device__ __forceinline__ AllReduce(const uint32_t thread_mask,
                                       float* inputs_, float* outputs_)
      : kShflMask(thread_mask), inputs(inputs_), outputs(outputs_) {}
  __device__ __forceinline__ void Swap(int i, int j, float* x) {
    float t = x[i];
    x[i] = x[j];
    x[j] = t;
  }

  __device__ __forceinline__ void ReduceStep(int lane, int i, int j) {
    const int kStep = SPC::Log2(lane);
    if ((threadIdx.x >> kStep) & 1) Swap(i, j, inputs);
    inputs[i] += __shfl_xor_sync(kShflMask, inputs[j], lane, kBlockWidth);
  }

  __device__ __forceinline__ void Reduce() {
#pragma unroll
    for (int base_idx = 0; base_idx < kThreadItemsX; ++base_idx) {
#pragma unroll
      for (int k_item_idx = 1; k_item_idx < kBlockWidth; k_item_idx *= 2) {
        const int kBoundX = kBlockWidth / (k_item_idx * 2);
#pragma unroll
        for (int x_item_idx = 0; x_item_idx < kBoundX; ++x_item_idx) {
          const int idx_a = x_item_idx * 2 * kValuesPerLoad * k_item_idx;
          const int idx_b = (x_item_idx * 2 + 1) * kValuesPerLoad * k_item_idx;
          ReduceStep(k_item_idx, base_idx + idx_a, base_idx + idx_b);
        }
      }
    }

    // Move the last four values to the first four of the output. This
    // should get cleaned up during register allocation.
#pragma unroll
    for (int out_idx = 0; out_idx < kThreadItemsX; ++out_idx) {
      outputs[out_idx] = inputs[out_idx];
    }
  }
};

#endif  // SPUTNIK_SDDMM_ALL_REDUCE_H_
