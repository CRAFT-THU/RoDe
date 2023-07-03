#ifndef SPMM_CONFIG_H_
#define SPMM_CONFIG_H_

#include "basic_utils.h"

namespace SPC {

template <typename ScalarValue_,    // Scalar data type for all operands.
          typename SparseValue_,    // Vector data type for the sparse matrix.
          typename DenseValue_,     // Vector data type for the dense operands.
          int kBlockItemsY_,        // Tile size in the m-dimension.
          int kBlockItemsK_,        // Tile size in the k-dimension.
          int kBlockItemsX_,        // Tile size in the n-dimension.
          int kBlockWidth_,         // Threadblock width.
          int kResidueUnroll_ = 4,  // Number of unroll steps in the residue.
          bool kPredicateLoads_ = true,  // Whether to predicate loads or not.
          bool kLaunchBounds_ = false,  // Whether or not to set launch bounds.
          int kMinOccupancy_ = 8>       // Minimum occupancy to target.

struct SpmmConfig {

    typedef ScalarValue_ ScalarValue;
    typedef SparseValue_ SparseValue;
    typedef DenseValue_ DenseValue;
    typedef typename Value2Index<SparseValue>::Index Index;
    typedef typename Value2Index<ScalarValue>::Index ScalarIndex;

    static constexpr int kBlockItemsY = kBlockItemsY_;
    static constexpr int kBlockItemsK = kBlockItemsK_;
    static constexpr int kBlockItemsX = kBlockItemsX_;
    static constexpr int kBlockWidth = kBlockWidth_;
    static constexpr int kResidueUnroll = kResidueUnroll_;
    static constexpr int kPredicateLoads = kPredicateLoads_;
    static constexpr bool kLaunchBounds = kLaunchBounds_;
    static constexpr int kMinOccupancy = kMinOccupancy_;
    static constexpr int kElementsPerScalar =
        TypeUtils<ScalarValue_>::kElementsPerScalar;

    // Sanity checks on the template arguments.
    static_assert((kBlockItemsY * kBlockWidth) % 32 == 0,
                "The thread-block size must be divisible by the warp size.");
    static_assert((kBlockItemsY * kBlockWidth) > 0,
                "The thread-block size must be nonzero.");
    static_assert(kBlockItemsK >= kBlockWidth,
                "k-dimension tile must be >= block width.");
    static_assert(kBlockItemsK % kBlockWidth == 0,
                "k-dimension tile size must be divisible by block width.");
    static_assert(kBlockItemsX >= kBlockWidth,
                "n-dimension tile size must be >= block width.");
    static_assert(kBlockItemsX % kBlockWidth == 0,
                "n-dimension tile size must be divisible by block width.");

    // The number of values in every load/store of type DenseValue.
    static constexpr int kValuesPerItemX =
        sizeof(DenseValue) / sizeof(ScalarValue);

    // The number of items in the n-dimension each thread is responsbile for.
    static constexpr int kThreadItemsX =
        kBlockItemsX / kBlockWidth / kValuesPerItemX;

    // The number of threads per threadblock.
    static constexpr int kThreadsPerBlock = kBlockItemsY * kBlockWidth;
};


}


#endif