#ifndef SPUTNIK_SDDMM_COMPUTE_UTILS_H_
#define SPUTNIK_SDDMM_COMPUTE_UTILS_H_


template <int kBlockItemsK, int kBlockItemsX, int kBlockWidth>
struct ComputeUtils {
  //
  /// Static members.
  //

  // The number of values in the k-dimension each thread is responsible for.
  static constexpr int kThreadItemsK = kBlockItemsK / kBlockWidth;

  // The number of outputs each thread is responsbile for.
  static constexpr int kThreadItemsX = kBlockItemsX;

  //
  /// Member variables.
  //

  // Register file fragment storing the lhs values.
  const float* lhs_fragment;

  // Register file fragment storing the rhs values.
  const float* rhs_fragment;

  // Register file fragment to accumulate results into.
  float* output_fragment;

  __device__ __forceinline__ ComputeUtils(const float* lhs_fragment_,
                                          const float* rhs_fragment_,
                                          float* output_fragment_)
      : lhs_fragment(lhs_fragment_),
        rhs_fragment(rhs_fragment_),
        output_fragment(output_fragment_) {}

  /**
   * @brief Compute a tile-level matrix product.
   */
  __device__ __forceinline__ void TileMAC() {
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
      const float lhs_value = lhs_fragment[k_item_idx];
#pragma unroll
      for (int x_item_idx = 0; x_item_idx < kThreadItemsX; ++x_item_idx) {
        const float rhs_value =
            rhs_fragment[k_item_idx + x_item_idx * kThreadItemsK];
        output_fragment[x_item_idx] += lhs_value * rhs_value;
      }
    }
  }
};

#endif  // SPUTNIK_SDDMM_COMPUTE_UTILS_H_
