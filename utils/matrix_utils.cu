// Copyright 2020 The Sputnik Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>

#include "glog/logging.h"
#include "matrix_utils.h"


#define METHOD_V1

namespace SPC {

namespace {

/**
 * @brief Helper to convert float data to half precision data.
 */
__global__ void ConvertKernel(const float *in_f, half2 *out, int n) {
  const float2 *in = reinterpret_cast<const float2 *>(in_f);
  n /= 2;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  out[idx] = __float22half2_rn(in[idx]);
}

__global__ void ConvertKernel(const int *in_i, short2 *out, int n) {
  const int2 *in = reinterpret_cast<const int2 *>(in_i);
  n /= 2;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  int2 a = in[idx];
  short2 b;
  b.x = static_cast<short>(a.x);
  b.y = static_cast<short>(a.y);
  out[idx] = b;
}

__global__ void ConvertKernel(const half2 *in, float *out_f, int n) {
  float2 *out = reinterpret_cast<float2 *>(out_f);
  n /= 2;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  out[idx] = __half22float2(in[idx]);
}

__global__ void ConvertKernel(const short2 *in, int *out_i, int n) {
  int2 *out = reinterpret_cast<int2 *>(out_i);
  n /= 2;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  short2 a = in[idx];
  int2 b;
  b.x = static_cast<int>(a.x);
  b.y = static_cast<int>(a.y);
  out[idx] = b;
}

__global__ void ConvertKernel(const float *in, double *out_i, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  out_i[idx] = (double) in[idx];
}

__global__ void ConvertKernel(const double *in, float *out_i, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  out_i[idx] = (float) in[idx];
}

/**
 * @brief Create a dense matrix with randomly sampled values.
 *
 * @param rows The number of rows in the matrix.
 * @param columns The number of columns in the matrix.
 * @param Buffer allocated to store the dense matirx.
 */
template <typename ValueType>
void MakeDenseMatrix(int rows, int columns, ValueType *matrix,
                     absl::BitGen *generator) {
  // Generate random values for the matrix.
  for (int64_t i = 0; i < static_cast<int64_t>(rows) * columns; ++i) {
    matrix[i] = absl::Uniform<ValueType>(*generator, -1, 1);
  }
}

void PadSparseMatrix(const std::vector<int> &row_offsets,
                     const std::vector<float> &values,
                     const std::vector<int> &column_indices, int row_padding,
                     std::vector<int> *row_offsets_out,
                     std::vector<float> *values_out,
                     std::vector<int> *column_indices_out) {
  CHECK_GE(row_padding, 0) << "Row padding factor must be greater than zero.";
  if (row_padding < 2) {
    // For row padding to the nearest 1 element, copy the input to the
    // output and return early. We also execute this code path for
    // `row_padding` == 0, which indicates no padding is to be added.
    row_offsets_out->assign(row_offsets.begin(), row_offsets.end());
    values_out->assign(values.begin(), values.end());
    column_indices_out->assign(column_indices.begin(), column_indices.end());
    return;
  }
  row_offsets_out->push_back(0);

  int offset = 0;
  for (int i = 0; i < row_offsets.size() - 1; ++i) {
    // Copy the existing values and column indices for this row to
    // the output.
    int row_length = row_offsets[i + 1] - row_offsets[i];
    values_out->resize(values_out->size() + row_length);
    column_indices_out->resize(column_indices_out->size() + row_length);
    std::copy(values.begin() + row_offsets[i],
              values.begin() + row_offsets[i + 1],
              values_out->begin() + offset);
    std::copy(column_indices.begin() + row_offsets[i],
              column_indices.begin() + row_offsets[i + 1],
              column_indices_out->begin() + offset);
    offset += row_length;

    // Calculate the number of zeros that need to be inserted in
    // this row to reach the desired padding factor.
    int residue = offset % row_padding;
    int to_add = (row_padding - residue) % row_padding;
    for (; to_add > 0; --to_add) {
      values_out->push_back(0.0);

      // NOTE: When we pad with zeros the column index that we assign
      // the phantom zero needs to be a valid column index s.t. we
      // don't index out-of-range into the dense rhs matrix when
      // computing spmm. Here we set all padding column-offsets to
      // the same column as the final non-padding weight in the row.
      column_indices_out->push_back(column_indices_out->back());
      ++offset;
    }
    row_offsets_out->push_back(offset);
  }
}

}  // namespace

template <typename In, typename Out>
cudaError_t Convert(const In *in, Out *out, int n) {
  if (n == 0) return cudaSuccess;
  CHECK_EQ(n % 2, 0) << "Number of elements must be multiple of 2.";

  int threads_per_block = 64;
  int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
  ConvertKernel<<<blocks_per_grid, threads_per_block, 0, 0>>>(in, out, n);
  return cudaGetLastError();
}

template<>
cudaError_t Convert(const float *in, float *out, int n) {
  return cudaMemcpy(out, in, n * sizeof(float), cudaMemcpyDeviceToDevice);
}

template<>
cudaError_t Convert(const int *in, int *out, int n) {
  return cudaMemcpy(out, in, n * sizeof(int), cudaMemcpyDeviceToDevice);
}

template<>
cudaError_t Convert(const float *in, double *out, int n) {
  if (n == 0) return cudaSuccess;

  int threads_per_block = 64;
  int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
  ConvertKernel<<<blocks_per_grid, threads_per_block, 0, 0>>>(in, out, n);
  return cudaGetLastError();
}

/**
 * @brief Create a sparse matrix with uniformly sampled non-zeros.
 *
 * @param rows The number of rows in the matrix.
 * @param columns The number of columns in the matrix.
 * @param nonzeros The number of non-zero values in the sparse matrix.
 * @param values Host-side buffer for the sparse matrix values.
 * Size `nonzeros + (row_padding - 1) * rows`.
 * @param row_offsets Host-side buffer for the sparse matrix row offsets.
 * Size `rows + 1`.
 * @param column_indices Host-side buffer for the sparse matrix column
 * offsets.
 * Size `nonzeros + (row_padding - 1) * rows`.
 * @param row_padding Each row in the sparse matrix will be padded to a
 * multiple of this value. Defaults to 4, which enables the user of
 * 4-element vector loads and stores. For best performance, pad to
 * `kBlockItemsK`.
 */
template <typename ValueType, typename IndexType>
void MakeSparseMatrixRandomUniform(int rows, int columns, int nonzeros,
                                   ValueType *values, IndexType *row_offsets,
                                   IndexType *column_indices,
                                   absl::BitGen *generator,
                                   int row_padding) {
  // The number of elements in the dense version of the matrix.
  int64_t num_elements = static_cast<int64_t>(rows) * columns;

  CHECK_LE(nonzeros, num_elements) << "The number of non-zero elements "
                                   << "must be <= the number of elements.";
  CHECK_GT(nonzeros, 0)
      << "The sparse matrix must have at least 1 non-zero value.";
  CHECK_GE(row_padding, 0) << "Row padding factor must be greater than zero.";

  // Generate random values for the matrix.
  std::vector<ValueType> nonzero_values(nonzeros);
  for (auto &v : nonzero_values) {
    v = absl::Uniform<ValueType>(*generator, -1, 1);
  }

  // Create a uniformly distributed random sparsity mask. We randomly
  // select which values to make zero and then mask them to create the
  // sparse matrix.
  std::vector<int64_t> indices(num_elements);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), *generator);

  // Create the compressed sparse row indices and offsets.
  int64_t offset = 0;
  row_offsets[0] = 0;
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < columns; ++j) {
      int64_t idx = i * columns + j;
      if (indices[idx] < nonzeros) {
        values[offset] = nonzero_values[indices[idx]];
        column_indices[offset] = j;
        ++offset;
      }
    }

    // If row_padding is zero, skip this code s.t. we don't mod zero.
    if (row_padding > 0) {
      // Pad the row with zeros s.t. every row contains a multiple of
      // `row_padding` elements.
      int residue = (offset - row_offsets[i]) % row_padding;
      int to_add = (row_padding - residue) % row_padding;
      for (; to_add > 0; --to_add) {
        values[offset] = 0.0;

        // NOTE: When we pad with zeros the column index that we assign
        // the phantom zero needs to be a valid column index s.t. we
        // don't index out-of-range into the dense rhs matrix when
        // computing spmm. Here we set all padding column-offsets to
        // the same column as the final non-padding weight in the row.
        column_indices[offset] = column_indices[offset - 1];
        ++offset;
      }

      // Set the row offset and sanity check the offset to make sure
      // we padded the row correctly.
      CHECK_EQ((offset - row_offsets[i]) % row_padding, 0);
    }
    row_offsets[i + 1] = offset;
  }
}

/**
 * @brief Create a sparse matrix with uniformly sampled non-zeros.
 *
 * @param rows The number of rows in the matrix.
 * @param columnss The number of columns in the matrix.
 * @param nonzeros_per_row The number of non-zero values in each row of
 * the sparse matrix.
 * @param values Host-side buffer for the sparse matrix values.
 * Size `nonzeros_per_row * rows`.
 * @param row_offsets Host-side buffer for the sparse matrix row offsets.
 * Size `rows + 1`.
 * @param column_indices Host-side buffer for the sparse matrix column
 * offsets.
 * Size `nonzeros_per_row * rows`.
 */
template <typename ValueType, typename IndexType>
void MakeSparseMatrixPerfectUniform(int rows, int columns, int nonzeros_per_row,
                                    ValueType *values, IndexType *row_offsets,
                                    IndexType *column_indices,
                                    absl::BitGen *generator) {
  // Generate random values for the matrix.
  int nonzeros = nonzeros_per_row * rows;
  for (int64_t i = 0; i < nonzeros; ++i) {
    values[i] = absl::Uniform<ValueType>(*generator, -1, 1);
  }

  // Select indices to make zero.
  int64_t offset = 0;
  std::vector<int64_t> indices(columns);
  std::iota(indices.begin(), indices.end(), 0);
  for (int64_t i = 0; i < rows; ++i) {
    std::shuffle(indices.begin(), indices.end(), *generator);
    std::vector<int64_t> sorted(nonzeros_per_row, 0);
    for (int64_t j = 0; j < nonzeros_per_row; ++j) {
      sorted[j] = indices[j];
    }
    std::sort(sorted.begin(), sorted.end());
    for (int64_t j = 0; j < nonzeros_per_row; ++j) {
      column_indices[offset + j] = sorted[j];
    }

    offset += nonzeros_per_row;
  }

  // Create the row offsets.
  offset = 0;
  for (int64_t i = 0; i < rows + 1; ++i) {
    row_offsets[i] = offset;
    offset += nonzeros_per_row;
  }
}

void IdentityRowSwizzle(int rows, const int * /* unused */, int *row_indices) {
  std::iota(row_indices, row_indices + rows, 0);
}

void SortedRowSwizzle(int rows, const int *row_offsets, int *row_indices) {
  // Create our unsorted row indices.
  std::vector<int> swizzle_staging(rows);
  std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

  // Argsort the row indices based on their length.
  std::sort(swizzle_staging.begin(), swizzle_staging.end(),
            [&row_offsets](int idx_a, int idx_b) {
              int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
              int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
              return length_a > length_b;
            });

  // Copy the ordered row indices to the output.
  std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
}

void LoadBalanceSort(int rows, const int *row_offsets, int *row_indices,int BLOCK) {
  std::vector<int> swizzle_staging(rows);
  std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);
  
  int head_idx = 0, tail_idx = rows - 1, r_idx = 0;
  while(head_idx < rows) {
    swizzle_staging[head_idx++] = row_indices[r_idx ++];

    for(int bidx = 1; bidx < BLOCK; ++bidx){
      if(head_idx >= rows) break;
      swizzle_staging[head_idx++] = row_indices[tail_idx--];
    }
  }
  std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
}

SparseMatrix::SparseMatrix(int rows, int columns, int nonzeros,
                           ElementDistribution weight_distribution,
                           absl::BitGen *generator, Swizzle row_swizzle,
                           int pad_rows_to) {
  // Save the matrix meta-data.
  rows_ = rows;
  columns_ = columns;
  nonzeros_ = nonzeros;
  weight_distribution_ = weight_distribution;
  row_swizzle_ = row_swizzle;
  pad_rows_to_ = pad_rows_to;

  CHECK_LE(pad_rows_to_, columns)
      << "Rows cannot be padded to more values than there are columns.";

  // Create some temporary host-side buffers to build the matrix in.
  // Note that we have to pad these buffers to account for potential
  // extra storage requirements for row padding.
  int padding_elements = std::max((pad_rows_to_ - 1) * rows_, 0);
  std::vector<float> values_staging(nonzeros_ + padding_elements);
  std::vector<int> row_offsets_staging(rows_ + 1);
  std::vector<int> column_indices_staging(nonzeros_ + padding_elements);

  if (weight_distribution == RANDOM_UNIFORM) {
    MakeSparseMatrixRandomUniform(
        rows_, columns_, nonzeros_, values_staging.data(),
        row_offsets_staging.data(), column_indices_staging.data(), generator,
        pad_rows_to_);
  } else {
    // Verify that the number of nonzeros divides evenly into the
    // number of rows.
    CHECK_EQ(nonzeros_ % rows_, 0) << "The number of nonzeros must divide "
                                   << "evenly by the number of rows to "
                                   << "construct a PERFECT_UNIFORM matrix.";

    MakeSparseMatrixPerfectUniform(
        rows_, columns_, nonzeros_ / rows_, values_staging.data(),
        row_offsets_staging.data(), column_indices_staging.data(), generator);
  }

  // Figure out exactly how much storage we need for the padded matrices,
  // allocate the storage, and copy the matrices into our storage.
  num_elements_with_padding_ = row_offsets_staging[rows_];

  values_ = new float[num_elements_with_padding_];
  column_indices_ = new int[num_elements_with_padding_];
  row_offsets_ = new int[rows_ + 1];

  // Copy the data into our allocated buffers.
  std::memcpy(values_, values_staging.data(),
              num_elements_with_padding_ * sizeof(float));
  std::memcpy(column_indices_, column_indices_staging.data(),
              num_elements_with_padding_ * sizeof(int));
  std::memcpy(row_offsets_, row_offsets_staging.data(),
              (rows_ + 1) * sizeof(int));

  // Allocate storage for our swizzled row indices and set the values.
  row_indices_ = new int[rows_];
  if (row_swizzle_ == IDENTITY) {
    IdentityRowSwizzle(rows_, row_offsets_, row_indices_);
  } else {
    SortedRowSwizzle(rows_, row_offsets_, row_indices_);
  }

  // row_indices1 = new int[rows_];
  // row_indices2 = new int[rows_];
}

void SparseMatrix::InterLD(int BLOCK) {
  SortedRowSwizzle(rows_, row_offsets_, row_indices_);
  LoadBalanceSort(rows_,row_offsets_,row_indices_,BLOCK);
}

void SparseMatrix::InterLD_part(int BLOCK) {
  std::vector<int> swizzle_staging(nr1);

  int head_idx = 0, tail_idx = nr1 - 1, r_idx = 0;
  while(head_idx < nr1) {
    swizzle_staging[head_idx++] = row_indices_1[r_idx++];

    for(int bidx=1; bidx < BLOCK; ++ bidx) {
      if(head_idx >= nr1) break;
      swizzle_staging[head_idx++] = row_indices_1[tail_idx--];
    }
  }

  std::memcpy(row_indices_1,swizzle_staging.data(),sizeof(int)*nr1);
}

SparseMatrix::SparseMatrix(const CudaSparseMatrix<float> &sparse_matrix) {
  InitFromCudaSparseMatrix(sparse_matrix);
}

SparseMatrix::SparseMatrix(int rows, int columns, int nonzeros,
                           const std::vector<int> &row_offsets,
                           const std::vector<int> &column_indices,
                           absl::BitGen *generator, Swizzle row_swizzle,
                           int pad_rows_to)
    : rows_(rows),
      columns_(columns),
      nonzeros_(nonzeros),
      pad_rows_to_(pad_rows_to),
      weight_distribution_(RANDOM_UNIFORM),
      row_swizzle_(row_swizzle) {
  CHECK_LE(pad_rows_to_, columns)
      << "Rows cannot be padded to more values than there are columns.";

  // Generate random values for the sparse matrix parameters.
  std::vector<float> values(nonzeros_);
  for (auto &v : values) v = absl::Uniform<float>(*generator, -1, 1);

  // Pad the rows to the desired length.
  std::vector<int> row_offsets_staging, column_indices_staging;
  std::vector<float> values_staging;
  PadSparseMatrix(row_offsets, values, column_indices, pad_rows_to,
                  &row_offsets_staging, &values_staging,
                  &column_indices_staging);

  // Figure out exactly how much storage we need for the padded matrices,
  // allocate the storage, and copy the matrices into our storage.
  num_elements_with_padding_ = row_offsets_staging[rows_];

  values_ = new float[num_elements_with_padding_];
  column_indices_ = new int[num_elements_with_padding_];
  row_offsets_ = new int[rows_ + 1];

  // Copy the data into our allocated buffers.
  std::memcpy(values_, values_staging.data(),
              num_elements_with_padding_ * sizeof(float));
  std::memcpy(column_indices_, column_indices_staging.data(),
              num_elements_with_padding_ * sizeof(int));
  std::memcpy(row_offsets_, row_offsets_staging.data(),
              (rows_ + 1) * sizeof(int));

  // Allocate storage for our swizzled row indices and set the values.
  row_indices_ = new int[rows_];
  if (row_swizzle_ == IDENTITY) {
    IdentityRowSwizzle(rows_, row_offsets_, row_indices_);
  } else {
    SortedRowSwizzle(rows_, row_offsets_, row_indices_);
  }
}

void SparseMatrix::InitFromCudaSparseMatrix(
    const CudaSparseMatrix<float> &sparse_matrix) {
  // Copy the sparse matrix meta-data.
  rows_ = sparse_matrix.Rows();
  columns_ = sparse_matrix.Columns();
  nonzeros_ = sparse_matrix.Nonzeros();
  pad_rows_to_ = sparse_matrix.PadRowsTo();
  num_elements_with_padding_ = sparse_matrix.NumElementsWithPadding();
  weight_distribution_ = sparse_matrix.WeightDistribution();
  row_swizzle_ = sparse_matrix.RowSwizzle();

  // Allocate memory on the CPU for our matrix.
  values_ = new float[num_elements_with_padding_];
  column_indices_ = new int[num_elements_with_padding_];
  row_offsets_ = new int[rows_ + 1];
  row_indices_ = new int[rows_];

  // Copy the results to the CPU.
  CUDA_CALL(cudaMemcpy(values_, sparse_matrix.Values(),
                       sizeof(float) * num_elements_with_padding_,
                       cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(column_indices_, sparse_matrix.ColumnIndices(),
                       sizeof(int) * num_elements_with_padding_,
                       cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(row_offsets_, sparse_matrix.RowOffsets(),
                       sizeof(int) * (rows_ + 1), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(row_indices_, sparse_matrix.RowIndices(),
                       sizeof(int) * rows_, cudaMemcpyDeviceToHost));
}

template <typename Value>
CudaSparseMatrix<Value>::CudaSparseMatrix(
    int rows, int columns, int nonzeros,
    ElementDistribution weight_distribution, absl::BitGen *generator,
    Swizzle row_swizzle, int pad_rows_to) {
  CHECK_EQ(pad_rows_to % TypeUtils<Value>::kElementsPerScalar, 0)
      << "The number of elements in each row must be divisible by "
      << "the number of elements per scalar value for the specified "
      << "data type.";
  // Create a sparse matrix on the host.
  SparseMatrix sparse_matrix(rows, columns, nonzeros, weight_distribution,
                             generator, row_swizzle, pad_rows_to);
  InitFromSparseMatrix(sparse_matrix);
}

template <typename Value>
CudaSparseMatrix<Value>::CudaSparseMatrix(const SparseMatrix &sparse_matrix) {
  // The number of nonzeros in each row must be divisible by the number of
  // elements per scalar for the specified data type.
  for (int i = 0; i < sparse_matrix.Rows(); ++i) {
    int nnz = sparse_matrix.RowOffsets()[i + 1] - sparse_matrix.RowOffsets()[i];
    CHECK_EQ(nnz % TypeUtils<Value>::kElementsPerScalar, 0)
        << "The number of elements in each row must be divisible by "
        << "the number of elements per scalar value for the specified "
        << "data type.";
  }
  InitFromSparseMatrix(sparse_matrix);
}

template <typename Value>
void CudaSparseMatrix<Value>::InitFromSparseMatrix(
    const SparseMatrix &sparse_matrix) {
  // Copy the sparse matrix meta-data.
  rows_ = sparse_matrix.Rows();
  columns_ = sparse_matrix.Columns();
  nonzeros_ = sparse_matrix.Nonzeros();
  pad_rows_to_ = sparse_matrix.PadRowsTo();
  num_elements_with_padding_ = sparse_matrix.NumElementsWithPadding();
  weight_distribution_ = sparse_matrix.WeightDistribution();
  row_swizzle_ = sparse_matrix.RowSwizzle();

  // Allocate memory on the GPU for our matrix.
  float *values_float = nullptr;
  int *column_indices_int = nullptr;
  CUDA_CALL(
      cudaMalloc(&values_float, sizeof(float) * num_elements_with_padding_));
  CUDA_CALL(cudaMalloc(&column_indices_int,
                       sizeof(int) * num_elements_with_padding_));
  CUDA_CALL(cudaMalloc(&row_offsets_, sizeof(int) * (rows_ + 1)));
  CUDA_CALL(cudaMalloc(&row_indices_, sizeof(int) * rows_));

  // Copy the results to the GPU.
  CUDA_CALL(cudaMemcpy(values_float, sparse_matrix.Values(),
                       sizeof(float) * num_elements_with_padding_,
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(column_indices_int, sparse_matrix.ColumnIndices(),
                       sizeof(int) * num_elements_with_padding_,
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(row_offsets_, sparse_matrix.RowOffsets(),
                       sizeof(int) * (rows_ + 1), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(row_indices_, sparse_matrix.RowIndices(),
                       sizeof(int) * rows_, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Allocate memory for the values and indices in the target datatype.
  int elements =
      num_elements_with_padding_ / TypeUtils<Value>::kElementsPerScalar;
  CUDA_CALL(cudaMalloc(&values_, sizeof(Value) * elements));
  CUDA_CALL(cudaMalloc(&column_indices_, sizeof(Index) * elements));

  // Convert to the target datatype.
  CUDA_CALL(Convert(values_float, values_, num_elements_with_padding_));
  CUDA_CALL(
      Convert(column_indices_int, column_indices_, num_elements_with_padding_));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Free the temporary memory.
  CUDA_CALL(cudaFree(values_float));
  CUDA_CALL(cudaFree(column_indices_int));


  nr1 = sparse_matrix.nr1;
  nr2 = sparse_matrix.nr2;

  row_indices_1 = nullptr;
  row_indices_2 = nullptr;
  if(nr1 > 0){
    CUDA_CALL(cudaMalloc((void**)&row_indices_1,sizeof(int)*nr1));
    CUDA_CALL(cudaMemcpy(row_indices_1,sparse_matrix.row_indices_1,sizeof(int)*nr1,cudaMemcpyHostToDevice));
  }
  if(nr2 > 0) {
    CUDA_CALL(cudaMalloc((void**)&row_indices_2,sizeof(int)*nr2));
    CUDA_CALL(cudaMemcpy(row_indices_2,sparse_matrix.row_indices_2,sizeof(int)*nr2,cudaMemcpyHostToDevice));
  }

  seg_row_indices = nullptr;
  seg_st_offsets = nullptr;
  seg_row_indices_residue = nullptr;

  n_segs = sparse_matrix.n_segs;
  
  if(n_segs > 0) {
    CUDA_CALL(cudaMalloc((void**)&seg_row_indices,sizeof(int)*n_segs));
    CUDA_CALL(cudaMemcpy(seg_row_indices,sparse_matrix.seg_row_indices,sizeof(int)*n_segs,cudaMemcpyHostToDevice));
  
    CUDA_CALL(cudaMalloc((void**)&seg_st_offsets,sizeof(int)*(n_segs+1)));
    CUDA_CALL(cudaMemcpy(seg_st_offsets,sparse_matrix.seg_st_offsets,sizeof(int)*(n_segs+1),cudaMemcpyHostToDevice));
  }

  n_segs_residue = sparse_matrix.n_segs_residue;
  if(n_segs_residue > 0) {
    CUDA_CALL(cudaMalloc((void**)&seg_row_indices_residue,sizeof(int)*n_segs_residue));
    CUDA_CALL(cudaMemcpy(seg_row_indices_residue,sparse_matrix.seg_row_indices_residue,sizeof(int)*n_segs_residue,cudaMemcpyHostToDevice));
  }
  
}

Matrix::Matrix(int rows, int columns, absl::BitGen *generator) {
  rows_ = rows;
  columns_ = columns;

  // Allocate storage for the matrix
  values_ = new float[rows_ * columns_];
  MakeDenseMatrix(rows_, columns_, values_, generator);
}

template <typename Value>
void Matrix::InitFromCudaMatrix(const CudaMatrix<Value> &matrix) {
  // Copy the matrix meta-data.
  rows_ = matrix.Rows();
  columns_ = matrix.Columns();

  // Allocate memory for our matrix.
  values_ = new float[rows_ * columns_];

  // Allocate a temporary buffer on GPU to convert the values into.
  float *matrix_values_float = nullptr;
  CUDA_CALL(cudaMalloc(&matrix_values_float, sizeof(float) * rows_ * columns_));
  CUDA_CALL(Convert(matrix.Values(), matrix_values_float, rows_ * columns_));

  // Copy the results from the GPU.
  CUDA_CALL(cudaMemcpy(values_, matrix_values_float,
                       sizeof(float) * rows_ * columns_,
                       cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(matrix_values_float));
}

template <typename Value>
CudaMatrix<Value>::CudaMatrix(int rows, int columns, absl::BitGen *generator) {
  // Create a dense matrix on the host and copy to gpu.
  Matrix matrix(rows, columns, generator);
  InitFromMatrix(matrix);
}

template <typename Value>
CudaMatrix<Value>::CudaMatrix(const Matrix &matrix) {
  InitFromMatrix(matrix);
}

template <typename Value>
void CudaMatrix<Value>::InitFromMatrix(const Matrix &matrix) {
  // Copy the matrix meta-data.
  rows_ = matrix.Rows();
  columns_ = matrix.Columns();

  // Allocate memory on the GPU for our matrix.
  float *values_float = nullptr;
  CUDA_CALL(cudaMalloc(&values_float, sizeof(float) * rows_ * columns_));

  // Copy the results to the GPU.
  CUDA_CALL(cudaMemcpy(values_float, matrix.Values(),
                       sizeof(float) * rows_ * columns_,
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Allocate memory for the values in the target data type.
  size_t elements = rows_ * columns_ / TypeUtils<Value>::kElementsPerScalar;
  CUDA_CALL(cudaMalloc(&values_, sizeof(Value) * elements));

  // Convert to the target data type.
  CUDA_CALL(Convert(values_float, values_, rows_ * columns_));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Free the temporary memory.
  CUDA_CALL(cudaFree(values_float));
}

// Explicit instantiations for template functions and classes.
template class CudaSparseMatrix<float>;
template class CudaSparseMatrix<half2>;
template class CudaSparseMatrix<double>;

template class CudaMatrix<float>;
template class CudaMatrix<half2>;
template class CudaMatrix<double>;

template void Matrix::InitFromCudaMatrix(const CudaMatrix<float> &);
template void Matrix::InitFromCudaMatrix(const CudaMatrix<half2> &);
template void Matrix::InitFromCudaMatrix(const CudaMatrix<double> &);


int compare0(const void *a, const void *b)
{
  if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
  if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
  if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col > 0) return 1;
  if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col < 0) return -1;
  return 0;
}

SparseMatrix::SparseMatrix(const std::string& file_path, Swizzle row_swizzle,int pad_rows_to) {

  // printf("begin reading...\n");

  srand(0);

  row_swizzle_ = row_swizzle;
  pad_rows_to_ = pad_rows_to;

  char buf[300];
  int nflag, sflag;

  FILE *fp = fopen(file_path.c_str(),"r");
  fgets(buf, 300, fp);
  
  if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
  else sflag = 0;

  if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
  else if(strstr(buf, "complex") != NULL) nflag = -1;
  else nflag = 1;

  #ifdef SYM
      sflag = 1;
  #endif

  int pre_count = 0;
  while(1) {
      pre_count++;
      fgets(buf, 300, fp);
      if(strstr(buf, "%") == NULL) break;
  }
  fclose(fp);

  int i;
  fp = fopen(file_path.c_str(), "r");
  for(i=0;i<pre_count;i++)
    fgets(buf, 300, fp);


	fscanf(fp, "%d %d %d", &rows_, &columns_, &nonzeros_);

  // printf("row: %d column:%d nnz: %d\n",rows_,columns_,nonzeros_);
	nonzeros_ *= (sflag+1);

	struct v_struct *temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(nonzeros_+1));
	// gold_temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(nonzeros_+1));
	for(i=0;i<nonzeros_;i++) {

		fscanf(fp, "%d %d", &temp_v[i].row, &temp_v[i].col);
		// temp_v[i].grp = INIT_GRP;
		temp_v[i].row--; temp_v[i].col--;

		if(temp_v[i].row < 0 || temp_v[i].row >= rows_ || temp_v[i].col < 0 || temp_v[i].col >= columns_) {
			fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].row, temp_v[i].col);
			exit(0);
		}
		if(nflag == 0) temp_v[i].val = (float)(rand()%1048576)/1048576;
		else if(nflag == 1) {
				float ftemp;
				fscanf(fp, " %f ", &ftemp);
				temp_v[i].val = ftemp;
		} else { // complex
				float ftemp1, ftemp2;
				fscanf(fp, " %f %f ", &ftemp1, &ftemp2);
				temp_v[i].val = ftemp1;
		}
    // ??
		// temp_v[i].val = (float)(rand()%1048576)/1048576;
		#ifdef SIM_VALUE
		  temp_v[i].val = 1.0f;
		#endif

		if(sflag == 1) {
			i++;
			temp_v[i].row = temp_v[i-1].col;
			temp_v[i].col = temp_v[i-1].row;
			temp_v[i].val = temp_v[i-1].val;
			// temp_v[i].grp = INIT_GRP;
		}
	}
  fclose(fp);
	qsort(temp_v, nonzeros_, sizeof(struct v_struct), compare0);

  // remove repeated-elements
	int *loc = (int *)malloc(sizeof(int)*(nonzeros_+1));
	memset(loc, 0, sizeof(int)*(nonzeros_+1));
	loc[0]=1;
	for(i=1;i<nonzeros_;i++) {
		if(temp_v[i].row == temp_v[i-1].row && temp_v[i].col == temp_v[i-1].col)
			loc[i] = 0;
		else loc[i] = 1;
	}
	for(i=1;i<=nonzeros_;i++)
    loc[i] += loc[i-1];
	for(i=nonzeros_; i>=1; i--)
    loc[i] = loc[i-1];
	loc[0] = 0;
	for(i=0;i<nonzeros_;i++) {
		temp_v[loc[i]].row = temp_v[i].row;
		temp_v[loc[i]].col = temp_v[i].col;
		temp_v[loc[i]].val = temp_v[i].val;
		// temp_v[loc[i]].grp = temp_v[i].grp;
	}
	nonzeros_ = loc[nonzeros_];
	temp_v[nonzeros_].row = rows_;
	free(loc);

  std::vector<float> values_staging(nonzeros_);
  std::vector<int> row_offsets_staging(rows_ + 1);
  std::vector<int> column_indices_staging(nonzeros_);

  // std::iota(row_offsets_staging.begin(), row_offsets_staging.end(), 0);
  std::fill(row_offsets_staging.begin(),row_offsets_staging.end(),0);
	
	for(i=0; i< nonzeros_ ;i++) {
		column_indices_staging[i] = temp_v[i].col;
		values_staging[i] = temp_v[i].val;
		row_offsets_staging[1+temp_v[i].row] = i+1;
	}

	for(i=1;i< rows_ ;i++) {
    if(row_offsets_staging[i] == 0) row_offsets_staging[i] = row_offsets_staging[i-1];
	}
	row_offsets_staging[rows_] = nonzeros_;
  free(temp_v);

  std::vector<int> row_offsets_staging1, column_indices_staging1;
  std::vector<float> values_staging1;
  PadSparseMatrix(row_offsets_staging, values_staging, column_indices_staging, pad_rows_to,
                  &row_offsets_staging1, &values_staging1,
                  &column_indices_staging1);

  num_elements_with_padding_ = row_offsets_staging1[rows_];

  values_ = new float[num_elements_with_padding_];
  column_indices_ = new int[num_elements_with_padding_];
  row_offsets_ = new int[rows_ + 1];

  std::memcpy(values_, values_staging1.data(),
              num_elements_with_padding_ * sizeof(float));
  std::memcpy(column_indices_, column_indices_staging1.data(),
              num_elements_with_padding_ * sizeof(int));
  std::memcpy(row_offsets_, row_offsets_staging1.data(),
              (rows_ + 1) * sizeof(int));

  row_indices_ = new int[rows_];
  if (row_swizzle_ == IDENTITY) {
    IdentityRowSwizzle(rows_, row_offsets_, row_indices_);
  } else {
    SortedRowSwizzle(rows_, row_offsets_, row_indices_);
  }

  row_indices_1 = new int[rows_];
  row_indices_2 = new int[rows_];

  seg_row_indices = nullptr;
  seg_st_offsets  = nullptr;
  n_segs = 0;

  seg_row_indices_residue = nullptr;
  n_segs_residue = 0;
}

template <typename T>
__global__ void square_sum(int n,const T* v1,const T* v2,T* res) {
  T r = 0.0f;
  for(int i = threadIdx.x; i < n; i += blockDim.x) {
    r += (v1[i] - v2[i]) * (v1[i] - v2[i]);
  }

  r += __shfl_down_sync(0xffffffff,r,16);
  r += __shfl_down_sync(0xffffffff,r,8);
  r += __shfl_down_sync(0xffffffff,r,4);
  r += __shfl_down_sync(0xffffffff,r,2);
  r += __shfl_down_sync(0xffffffff,r,1);

  if((threadIdx.x & 31) == 0) 
    atomicAdd(&res[0],r);

}

template <typename T>
void Matrix_Diff(const CudaMatrix<T>& matrix1,const CudaMatrix<T>& matrix2) {
  int n = matrix1.Rows() * matrix1.Columns();

  T * res, *_res;

  cudaMalloc((void**)&_res,sizeof(T));
  res = new T[1];

  res[0] = 0.0f;
  cudaMemcpy(_res,res,sizeof(T),cudaMemcpyHostToDevice);

  square_sum<T><<<1,1024>>>(n,matrix1.Values(),matrix2.Values(),_res);

  cudaMemcpy(res,_res,sizeof(T),cudaMemcpyDeviceToHost);

  cudaFree(_res);

  printf("Diff : %f",res[0] / n);

  delete[] res;
  return;
}

#ifdef METHOD_V1
  void SparseMatrix::RowDivide(int vectorLen,int K) {
    nr1 = 0;
    nr2 = 0;

    for(int i=0; i < rows_; ++ i) {
      int r_idx = row_indices_[i];
      int n_padding = row_offsets_[r_idx] % vectorLen;
      int nnz = row_offsets_[r_idx + 1] - row_offsets_[r_idx] + n_padding;
      if(nnz >= K) {
        row_indices_1[nr1++] = row_indices_[i];
      }
      if(nnz % K) {
        row_indices_2[nr2++] = row_indices_[i];
      }

    }

  }
#else
  void SparseMatrix::RowDivide(int vectorLen,int K) {
    nr1 = 0;
    nr2 = 0;

    for(int i=0; i < rows_; ++ i) {
      int r_idx = row_indices_[i];
      int n_padding = row_offsets_[r_idx] % vectorLen;
      int nnz = row_offsets_[r_idx + 1] - row_offsets_[r_idx] + n_padding;
      if(nnz >= K) {
        row_indices_1[nr1++] = row_indices_[i];
      }
      else {
        row_indices_2[nr2++] = row_indices_[i];
      }

    }
  }
#endif

void SparseMatrix::RowDivide2Segment(int SegmentLength,int vectorLen,int KBLOCK) {
  int nr = 0;
  std::vector<int> row_indices_staging;
  std::vector<int> st_offsets_staging;
  std::vector<int> row_indices_residue_staging;
  
  for(int i=0; i < rows_; ++i) {
    int row_offset = row_offsets_[i];
    int n_padding = row_offset % vectorLen;
    int nnz = row_offsets_[i+1] - row_offset + n_padding;

    if(nnz > SegmentLength) {
      row_indices_staging.push_back(i);
      st_offsets_staging.push_back(row_offset);
      row_offset = (row_offset + SegmentLength) - n_padding;

      nnz -= SegmentLength;
    }

    while(nnz > SegmentLength) {
      row_indices_staging.push_back(i);
      st_offsets_staging.push_back(row_offset);

      row_offset += SegmentLength;
      nnz -= SegmentLength;
    }

    if(nnz > 0) {
      if(nnz >= KBLOCK){
        row_indices_staging.push_back(i);
        st_offsets_staging.push_back(row_offset);
      }
      if( nnz % KBLOCK) {
        row_indices_residue_staging.push_back(i);
      }
    }
  }

  st_offsets_staging.push_back(row_offsets_[rows_]);

  if(n_segs > 0) {
    delete[] seg_row_indices;
    delete[] seg_st_offsets;
  }
  if(n_segs_residue > 0) {
    delete[] seg_row_indices_residue;
  }

  n_segs = row_indices_staging.size();
  seg_row_indices = new int[n_segs];
  seg_st_offsets = new int[n_segs+1];

  n_segs_residue = row_indices_residue_staging.size();
  seg_row_indices_residue = new int[n_segs_residue];

  std::memcpy(seg_row_indices,row_indices_staging.data(),sizeof(int)*n_segs);
  std::memcpy(seg_st_offsets,st_offsets_staging.data(),sizeof(int)*(n_segs+1));
  std::memcpy(seg_row_indices_residue,row_indices_residue_staging.data(),sizeof(int)*n_segs_residue);
}

void SparseMatrix::RowDivide2Segment_Nopadding(int SegmentLength,int vectorLen,int KBLOCK) {
  int nr = 0;
  std::vector<int> row_indices_staging;
  std::vector<int> st_offsets_staging;
  std::vector<int> row_indices_residue_staging;
  
  for(int i=0; i < rows_; ++i) {
    int row_offset = row_offsets_[i];
    int nnz = row_offsets_[i+1] - row_offset ;

    while(nnz > SegmentLength) {
      row_indices_staging.push_back(i);
      st_offsets_staging.push_back(row_offset);

      row_offset += SegmentLength;
      nnz -= SegmentLength;
    }

    if(nnz > 0) {
      if(nnz >= KBLOCK){
        row_indices_staging.push_back(i);
        st_offsets_staging.push_back(row_offset);
      }
      if( nnz % KBLOCK) {
        row_indices_residue_staging.push_back(i);
      }
    }
  }

  st_offsets_staging.push_back(row_offsets_[rows_]);

  n_segs = row_indices_staging.size();
  seg_row_indices = new int[n_segs];
  seg_st_offsets = new int[n_segs+1];

  n_segs_residue = row_indices_residue_staging.size();
  seg_row_indices_residue = new int[n_segs_residue];

  std::memcpy(seg_row_indices,row_indices_staging.data(),sizeof(int)*n_segs);
  std::memcpy(seg_st_offsets,st_offsets_staging.data(),sizeof(int)*(n_segs+1));
  std::memcpy(seg_row_indices_residue,row_indices_residue_staging.data(),sizeof(int)*n_segs_residue);
}

template void Matrix_Diff(const CudaMatrix<float>& matrix1,const CudaMatrix<float>& matrix2);

}  


