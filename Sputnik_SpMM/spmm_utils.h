#ifndef SPC_SPMM_UTILS_H_
#define SPC_SPMM_UTILS_H_

#include "common_utils.h"

namespace SPC {

//  Predicate *********************************************************
template <int kPredicates, int kPredicatesPerByte = 4>
class PredicateVector {
 public:
  static_assert(kPredicatesPerByte <= 8,
                "Can't pack more than 8 predicates into a byte.");
  // Use 32-bit unsigned integers to store predicates.
  typedef uint32_t PredicateStorage;

  // The number of bytes we need to store the predicates.
  static constexpr int kBytes_ =
      (kPredicates + kPredicatesPerByte - 1) / kPredicatesPerByte;

  // The number of words we need to store the predicates.
  static constexpr int kWords_ =
      (kBytes_ + sizeof(PredicateStorage) - 1) / sizeof(PredicateStorage);

  //
  /// Member variables.
  //

  // Register storage for the predicates.
  PredicateStorage predicates_[kWords_];

  /**
   * @brief Constructor. Initialize all predicate bits to 1.
   */
  __device__ __forceinline__ PredicateVector() {
    #pragma unroll
    for (int i = 0; i < kWords_; ++i) {
      predicates_[i] = 0xffffffff;
    }
  }

  /**
   * @brief Set the bit at the specified location to zero.
   */
  __device__ __forceinline__ void DisableBit(int idx) {
    int word, bit;
    GetWordAndBitOffsets(idx, &word, &bit);
    // NOTE: It could be worth looking into using bit-field insert
    // inline assembly for these operations.
    predicates_[word] &= ~(1 << bit);
  }

  /**
   * @brief Get the bit at the specified location.
   */
  __device__ __forceinline__ bool GetBit(int idx) const {
    int word, bit;
    GetWordAndBitOffsets(idx, &word, &bit);
    // NOTE: It could be worth looking into using bit-field extract
    // inline assembly for these operations.
    return (predicates_[word] >> bit) & 1;
  }

 private:
  /**
   * @brief Convert an index to word and byte offsets for setting and
   * extracting the underlying predicate.
   */
  __device__ __forceinline__ void GetWordAndBitOffsets(int idx, int *word,
                                                       int *bit) const {
    // NOTE: Indices to this function should be statically known s.t.
    // the following indexing math can be evaluated during compilation.
    //
    // TODO(tgale): Figure out a way to force the compiler to enforce
    // that these are statically known. Using constexpr here causes the
    // compiler to complain, even though all inputs are statically known
    // indices of unrolled loops.
    const int kWordOffset =
        (idx / kPredicatesPerByte) / sizeof(PredicateStorage);
    const int kByteOffset =
        (idx / kPredicatesPerByte) % sizeof(PredicateStorage);
    const int kBitOffset =
        (idx % kPredicatesPerByte) % sizeof(PredicateStorage);

    // TODO(tgale): Following cutlass, we store predicates in the first four
    // bits of each byte. It's not totally clear why we do this versus using
    // all the bits or spread out the predicates every-other bit.
    *word = kWordOffset;
    *bit = kByteOffset * 8 + kBitOffset;
  }
};

template <typename LoadType, int kBlockItemsX, int kBlockWidth>
struct PredicatesN {
  //
  /// Static members.
  //

  typedef typename TypeUtils<LoadType>::ScalarValue ScalarValue;

  // The number of values in every load/store with of LoadType.
  static constexpr int kValuesPerItem_ = sizeof(LoadType) / sizeof(ScalarValue);

  // The number of items in the n-dimension each thread is responsbile for.
  static constexpr int kThreadItemsX_ =
      kBlockItemsX / kBlockWidth / kValuesPerItem_;

  // The number of values we increment by after each load.
  static constexpr int increment_x_ = kBlockWidth * kValuesPerItem_;

  //
  /// Member functions.
  //

  // Shorthand for n-dim predicate vector of the appropriate size.
  typedef PredicateVector<kThreadItemsX_> Predicates;

  // Default constructor.
  __device__ __forceinline__ PredicatesN() {}

  /**
   * @brief Set predicates for this threads loads in the n-dimension.
   *
   * When loading/storing along the n-dimension of the problem we need
   * to avoid going out of bounds if the problem dimensions don't divide
   * evenly by the tile dimensions. This function sets the appropriate
   * predicates to avoid out-of-bounds memory accesses.
   *
   * @param n_idx The column index marking the start of the 1-dimensional
   * tile that this thread collaborates to compute.
   * @param n The number of columns in the dense rhs and output matrices.
   * @param predicates Pointer to a vector of predicates that we'll store
   * the computed predicates in.
   */
  static __device__ __forceinline__ void Set(int n_idx, int n,
                                             Predicates *predicates) {
    int index = n_idx + threadIdx.x * kValuesPerItem_;

    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
      if (index >= n) {
        predicates->DisableBit(x_item_idx);
      }
      index += increment_x_;
    }
  }
};

template <typename LoadType, int kBlockItemsK, int kBlockWidth>
struct PredicatesK {
  //
  /// Static members.
  //

  // The number of values in every load/store with of LoadType.
  static constexpr int kValuesPerItem_ = sizeof(LoadType) / sizeof(float);

  // The number of items in the n-dimension each thread is responsbile for.
  static constexpr int kThreadItemsK_ =
      kBlockItemsK / kBlockWidth / kValuesPerItem_;

  // The number of values we increment by after each load.
  static constexpr int increment_k_ = kBlockWidth * kValuesPerItem_;

  //
  /// Member functions.
  //

  // Shorthand for a predicate vector of the appropriate size.
  typedef PredicateVector<kThreadItemsK_> Predicates;

  // Default constructor.
  __device__ __forceinline__ PredicatesK() {}

  /**
   * @brief Set predicates for this threads loads in the k-dimension.
   *
   * When loading along the k-dimension of the problem we need to avoid
   * going out of bounds if the problem dimensions don't divide evenly
   * by the tile dimensions. This function sets the appropriate predicates
   * to avoid out-of-bounds memory accesses.
   *
   * @param residue The number of values left to load along the k-dimension
   * after we've computed the maximum number of full tiles possible.
   * @param predicates Pointer to a vector of predicates that we'll store
   * the computed predicates in.
   */
  static __device__ __forceinline__ void Set(int residue,
                                             Predicates *predicates) {
    int index = threadIdx.x * kValuesPerItem_;
    #pragma unroll
    for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; ++k_item_idx) {
      if (index >= residue) {
        predicates->DisableBit(k_item_idx);
      }
      index += increment_k_;
    }
  }
};

//  compute utils ******************************************
template <typename Value, int kBlockItemsK, int kBlockItemsX, int kBlockWidth>
struct ComputeUtils {
    typedef typename TypeUtils<Value>::ScalarValue ScalarValue;
    static constexpr int kValuesPerItem_ = sizeof(Value) / sizeof(ScalarValue);

    static constexpr int kThreadItemsX_ =
      kBlockItemsX / kBlockWidth / kValuesPerItem_;

    static constexpr int kElementsPerScalar_ =
      TypeUtils<Value>::kElementsPerScalar;

    typedef typename TypeUtils<Value>::Accumulator Accumulator;

    typedef typename std::conditional<std::is_same<ScalarValue,double>::value,double,float>::type Ftype;

    const ScalarValue* lhs_tile_;
    const Value* rhs_fragment_;
    Ftype* output_fragment_;

    __device__ __forceinline__ ComputeUtils(const ScalarValue* lhs_tile,
                                          const ScalarValue* rhs_fragment,
                                          Ftype* output_fragment)
      : lhs_tile_(lhs_tile),
        rhs_fragment_(reinterpret_cast<const Value*>(rhs_fragment)),
        output_fragment_(output_fragment) {}

    __device__ __forceinline__ void TileMAC() {
        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kBlockItemsK; ++k_item_idx) {
            Ftype lhs_values[kElementsPerScalar_];
            Convert(lhs_tile_ + k_item_idx, lhs_values);

            #pragma unroll
            for(int elt_idx = 0; elt_idx < kElementsPerScalar_; ++elt_idx) {

                #pragma unroll
                for(int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++ x_item_idx) {
                    Ftype* outputs = output_fragment_ + x_item_idx * kValuesPerItem_ * kElementsPerScalar_;
                    int rhs_offset = k_item_idx * kThreadItemsX_ * kElementsPerScalar_ + elt_idx * kThreadItemsX_ + x_item_idx;

                    VectorCompute<Value>::FMA(lhs_values[elt_idx],rhs_fragment_[rhs_offset],reinterpret_cast<Accumulator*>(outputs));
                }
            }
        }
    }

};


//  Sparse Tile ********************************************
//  Value -> Scalar(32bits) -> Elements
template <typename Value, int kBlockItemsK, int kBlockWidth>
struct SparseTile {
    typedef typename TypeUtils<Value>::ScalarValue ScalarValue;

    static_assert(kBlockItemsK >= kBlockWidth,
                "Sparse tile K-items must be >= thread block width.");
    static_assert(kBlockItemsK % kBlockWidth == 0,
                "Sparse tile K-items must be divisible by block width.");
    static_assert((sizeof(Value) / sizeof(ScalarValue)) <=
                    (kBlockItemsK / kBlockWidth),
                "The number of values per load must be <= values per thread.");

    typedef typename Value2Index<ScalarValue>::Index ScalarIndex;
    typedef typename Value2Index<Value>::Index Index;

    static constexpr int kValuesPerLoad_ = sizeof(Value) / sizeof(ScalarValue);

    static constexpr int kThreadItemsK_ = kBlockItemsK / kBlockWidth / kValuesPerLoad_;

    static constexpr int kElementsPerScalar_ = TypeUtils<Value>::kElementsPerScalar;

    const int rhs_columns_;

    const Value *values_;

    const Index *column_idxs_;

    Value *values_tile_base_;
    Index *column_idxs_tile_base_;

    typedef typename std::conditional<std::is_same<ScalarValue,double>::value,double,float>::type Ftype;

    __device__ __forceinline__
    SparseTile(int rhs_columns,int offset,int thread_idx_x,
                const ScalarValue *__restrict__ values,
                const ScalarIndex *__restrict__ column_idxs,
                ScalarValue *values_tile,ScalarIndex * column_idxs_tile)
        : rhs_columns_( rhs_columns * sizeof(ScalarValue)),
        values_(reinterpret_cast<const Value*>(values + offset) + thread_idx_x),
        column_idxs_(reinterpret_cast<const Index*>(column_idxs+offset) + thread_idx_x),
        values_tile_base_(reinterpret_cast<Value *>(values_tile) + thread_idx_x),
        column_idxs_tile_base_(reinterpret_cast<Index *>(column_idxs_tile)+ thread_idx_x) {}


    __device__ __forceinline__ void Load_() {
        Value * values_tile = values_tile_base_;
        Index * column_idxs_tile = column_idxs_tile_base_;

        #pragma unroll 
        for(int k_item_idx = 0; k_item_idx < kThreadItemsK_; ++k_item_idx) {
            Store(SPC::Load(values_),values_tile);

            if(TypeUtils<ScalarValue>::IsMixed()) {
                // 防止 16-bit 索引溢出
                Store(SPC::Load(column_idxs_),column_idxs_tile);
            } else {
                VectorCompute<Value>::Mul(rhs_columns_,SPC::Load(column_idxs_),column_idxs_tile);
            }

            values_ += kBlockWidth;
            column_idxs_ += kBlockWidth;

            values_tile += kBlockWidth;
            column_idxs_tile += kBlockWidth;
            
        }
    }

    __device__ __forceinline__ void ZeroTiles() {
        Value *values_tile = values_tile_base_;
        Index *column_idxs_tile = column_idxs_tile_base_;

        const Ftype kZeroValues[kValuesPerLoad_] = {};
        const int kZeroIndices[kValuesPerLoad_] = {};

        #pragma unroll 
        for(int k_item_idx = 0; k_item_idx < kThreadItemsK_; ++k_item_idx) {
            Store(*reinterpret_cast<const Value *>(kZeroValues), values_tile);
            Store(*reinterpret_cast<const Index *>(kZeroIndices), column_idxs_tile);
            values_tile += kBlockWidth;
            column_idxs_tile += kBlockWidth;
        }
    }

    __device__ __forceinline__ void Residue(int residue) {
        // 调整offset ： 将 value offset 调整为 scalar offset 
        constexpr int kResidueUpdateStrideValue = -1 * static_cast<int>(sizeof(ScalarValue) * (kValuesPerLoad_ - 1));
        const int kResidueUpdateValue = static_cast<int>(threadIdx.x) * kResidueUpdateStrideValue;

        constexpr int kResidueUpdateStrideIndex = -1 * static_cast<int>(sizeof(ScalarIndex) * (kValuesPerLoad_ - 1));
        const int kResidueUpdateIndex = static_cast<int>(threadIdx.x) * kResidueUpdateStrideIndex;

        const ScalarValue *values =
            OffsetCast<const ScalarValue>(values_, kResidueUpdateValue);
        const ScalarIndex *column_idxs =
            OffsetCast<const ScalarIndex>(column_idxs_, kResidueUpdateIndex);

        ScalarValue *values_tile = OffsetCast<ScalarValue>(values_tile_base_, kResidueUpdateValue);
        ScalarIndex *column_idxs_tile = OffsetCast<ScalarIndex>(column_idxs_tile_base_, kResidueUpdateIndex);

        constexpr int kScalarThreadItemsK = kBlockItemsK / kBlockWidth;

        #pragma unroll 
        for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++k_item_idx) {
            if( residue <= static_cast<int>(threadIdx.x)) return ;

            Store(SPC::Load(values),values_tile);

            if (TypeUtils<ScalarValue>::IsMixed()) {
                Store(SPC::Load(column_idxs), column_idxs_tile);
            } else {
                VectorCompute<ScalarValue>::Mul(rhs_columns_, SPC::Load(column_idxs), column_idxs_tile);
            }
            values += kBlockWidth;
            column_idxs += kBlockWidth;
            values_tile += kBlockWidth;
            column_idxs_tile += kBlockWidth;
            residue -= kBlockWidth;
        }
        asm("");

    }

};

//  Dense Tile **********************************************
template <typename Value, int kBlockItemsK, int kBlockItemsX, int kBlockWidth,
          int kResidueUnroll>
struct DenseTile {
  typedef typename TypeUtils<Value>::ScalarValue ScalarValue;

  static_assert(kBlockItemsK * kBlockItemsX >= kBlockWidth,
                "Dense tile size must be >= thread block width.");
  static_assert(kBlockItemsK * kBlockItemsX % kBlockWidth == 0,
                "Dense tile size must be divisible by thread block width.");
  static_assert(sizeof(Value) >= sizeof(ScalarValue),
                "Value size must be >= data type size.");
  static_assert(sizeof(Value) % sizeof(ScalarValue) == 0,
                "Value size must be divisbile by data type size.");

  typedef typename Value2Index<ScalarValue>::Index ScalarIndex;

  static constexpr int kValuesPerLoad_ = sizeof(Value) / sizeof(ScalarValue);

  static_assert(kValuesPerLoad_ * kBlockWidth <= kBlockItemsX,
                "The number of values loaded from a row of rhs "
                "at once must not exceed kBlockItemsX.");

  static constexpr int kThreadItemsX_ =
      kBlockItemsX / kBlockWidth / kValuesPerLoad_;

  // Compile time check on the residue unrolling parameter.
  static_assert(kBlockItemsK % kResidueUnroll == 0,
                "k-dimension tile size must be divisible by the residue"
                " unrolling factor.");

  // The number of outer loop iterations for the residue handling.
  static constexpr int kResidueOuterLimit_ = kBlockItemsK / kResidueUnroll;

  // The number of inner loop iterations for the residue handling.
  static constexpr int kResidueInnerLimit_ = kResidueUnroll;

  // The number of elements to compute on per-scalar rhs element.
  static constexpr int kElementsPerScalar_ =
      TypeUtils<Value>::kElementsPerScalar;

  // Data type of our accumulator registers.
  typedef typename TypeUtils<Value>::Accumulator Accumulator;

  // Shorthand for n-dim a predicate vector of the appropriate size.
  typedef PredicateVector<kThreadItemsX_> Predicates;

  const int rhs_columns_;

  const Value *matrix_base_;

  const ScalarIndex *row_offsets_base_;

  Value *matrix_fragment_;

  typedef typename std::conditional<std::is_same<ScalarValue,double>::value,double,float>::type Ftype;

  // Constructor. Set the initial pointer offsets.
  __device__ __forceinline__ DenseTile(int rhs_columns, int offset,
                                       int thread_idx_x,
                                       const ScalarValue *__restrict__ matrix,
                                       const ScalarIndex *row_offsets,
                                       ScalarValue *matrix_fragment)
      : rhs_columns_(rhs_columns * sizeof(ScalarValue)) {
    matrix_base_ =
        reinterpret_cast<const Value *>(matrix + offset) + thread_idx_x;
    row_offsets_base_ = row_offsets;
    matrix_fragment_ = reinterpret_cast<Value *>(matrix_fragment);
  }

  __device__ __forceinline__ void Load_(const Predicates &predicates_n) {
    const ScalarIndex *row_offsets = row_offsets_base_;

    #pragma unroll
    for (int k_item_idx = 0; k_item_idx < kBlockItemsK; ++k_item_idx) {
      // Load the row offsets and extract into 32-bit integer values.
        int scaled_indices[kElementsPerScalar_];
        Convert(row_offsets, scaled_indices);
        #pragma unroll
        for (int elt_idx = 0; elt_idx < kElementsPerScalar_; ++elt_idx) {
            // Possibly scale the indices s.t. they properly index into the
            // right-hand size dense matrix.
            if (TypeUtils<ScalarValue>::IsMixed()) {
                scaled_indices[elt_idx] *= rhs_columns_;
            }

            // Increment the matrix pointer.
            const Value *matrix =
                OffsetCast<const Value>(matrix_base_, scaled_indices[elt_idx]);
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
            // NOTE: There are a few different ways we could have expressed
            // this loop while avoiding out-of-bounds memory accesses. See
            // the documentation for PredicateVector for more info.
                if (predicates_n.GetBit(x_item_idx)) {
                    int fragment_offset =
                        k_item_idx * kThreadItemsX_ * kElementsPerScalar_ +
                        elt_idx * kThreadItemsX_ + x_item_idx;
                    matrix_fragment_[fragment_offset] = SPC::Load(matrix);

                    // Increment our matrix pointer for the next iteration.
                    matrix += kBlockWidth;
                }
            }
        }
        ++row_offsets;
    }
  }

  __device__ __forceinline__ void ResidueLoadAndCompute(
      int residue, const Predicates &predicates_n, const ScalarValue *lhs_tile,
      Ftype *output_fragment) {
    const ScalarIndex *row_offsets = row_offsets_base_;

    // If we're only going to perform a single iteration of the inner loop,
    // pull the predicate check out of the loop.
    if ((kThreadItemsX_ == 1) && !predicates_n.GetBit(0)) return;

    #pragma unroll
    for (int k_outer_idx = 0; k_outer_idx < kResidueOuterLimit_;
         ++k_outer_idx) {
        // The compiler doesn't like unrolling this loop with this bail-out,
        // but for some reason if we use "return" instead of "break", and we
        // have an asm block following the loop the compiler does it just fine.
        //
        // TODO(tgale): The empty asm block at the end of this loop is very
        // weird. Explore ways to unroll this loop without this block.
        if (residue <= 0) return;

        #pragma unroll
        for (int k_inner_idx = 0; k_inner_idx < kResidueInnerLimit_;
            ++k_inner_idx) {
            const int k_item_idx = k_inner_idx + k_outer_idx * kResidueInnerLimit_;

            // Load the row offsets and extract into 32-bit integer values.
            int scaled_indices[kElementsPerScalar_];
            Convert(row_offsets, scaled_indices);

            // Load the weight from smem and extract into 32-bit float values.
            Ftype lhs_values[kElementsPerScalar_];
            Convert(lhs_tile + k_item_idx, lhs_values);
            #pragma unroll
            for (int elt_idx = 0; elt_idx < kElementsPerScalar_; ++elt_idx) {
            // Possibly scale the indices s.t. they properly index into the
            // right-hand size dense matrix.
                if (TypeUtils<ScalarValue>::IsMixed()) {
                    scaled_indices[elt_idx] *= rhs_columns_;
                }

            // Increment hte matrix pointer.
            const Value *matrix =
                OffsetCast<const Value>(matrix_base_, scaled_indices[elt_idx]);
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
                // NOTE: We special-case kThreadItemsX_ == 1 to generate cleaner
                // branching code in this unrolled loop.
                if ((kThreadItemsX_ == 1) || predicates_n.GetBit(x_item_idx)) {
                    Ftype *outputs = output_fragment + x_item_idx * kValuesPerLoad_ *
                                                        kElementsPerScalar_;

                    // Load the rhs & lhs and compute immediately.
                    VectorCompute<Value>::FMA(
                        lhs_values[elt_idx], SPC::Load(matrix),
                        reinterpret_cast<Accumulator *>(outputs));

                    // Increment our matrix pointer for the next iteration.
                    matrix += kBlockWidth;
                }
            }
            }
            // Increment our row offsets pointer for the next iteration.
            ++row_offsets;
        }
        // Update the number of items left to process.
        residue -= kResidueInnerLimit_;
    }
    asm("");
  }
};

//  output tile ***************************************
template <typename Value,int kBlockItemsX,int kBlockWidth> 
struct OutputTile {
    typedef typename TypeUtils<Value>::ScalarValue ScalarValue;
    static constexpr int kValuesPerStore_ = sizeof(Value) / sizeof(ScalarValue);
    static constexpr int kThreadItemsX_ =
        kBlockItemsX / kBlockWidth / kValuesPerStore_;
    typedef PredicateVector<kThreadItemsX_> Predicates;

    static constexpr int kElementsPerScalar_ =
        TypeUtils<ScalarValue>::kElementsPerScalar;

    typedef typename std::conditional<std::is_same<ScalarValue,double>::value,double,float>::type Ftype;

    const Ftype* output_fragment_;

    Value* output_matrix_;

    __device__ __forceinline__ OutputTile(int row_offset, int column_offset,
                                        int cols, int thread_idx_x,
                                        const Ftype* output_fragment,
                                        ScalarValue* output_matrix) {

        output_fragment_ = output_fragment;
        const int output_offset = row_offset * cols + column_offset;
        output_matrix_ =
            reinterpret_cast<Value*>(output_matrix + output_offset) + thread_idx_x;
    }

    __device__ __forceinline__ void Store_(const Predicates& predicates_n) {
        #pragma unroll 
        for(int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++ x_item_idx) {
            if(predicates_n.GetBit(x_item_idx)) {
                if(TypeUtils<Value>::IsMixed()) {
                    Value out;
                    const int fragment_offset = x_item_idx * kElementsPerScalar_ * kValuesPerStore_;
                    Convert(output_fragment_ + fragment_offset, &out);
                    SPC::Store(out,output_matrix_);
                }else {
                    const Value* output_fragment = reinterpret_cast<const Value*>(output_fragment_);
                    *output_matrix_ = output_fragment[x_item_idx];
                }
                output_matrix_ += kBlockWidth;
            }
        }
    }
};


}

#endif