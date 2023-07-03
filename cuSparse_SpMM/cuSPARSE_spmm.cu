#include "cuSPARSE_spmm.h"

// template <typename Value>
void cuSparse_SPMM<float>::Preprocess(int m,int k,int nonzeros,
                    int *row_offsets,int* column_indices,float* values) {
        
    m_ = m; k_ = k;
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return ;
    }
    cusparseCreateCsr(&matA, m, k, nonzeros,
                    row_offsets, column_indices, values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
}

// template <>
void cuSparse_SPMM<double>::Preprocess(int m,int k,int nonzeros,
                    int *row_offsets,int* column_indices,double* values){

    m_ = m; k_ = k;
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return ;
    }
    cusparseCreateCsr(&matA, m, k, nonzeros,
                    row_offsets, column_indices, values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
}

// template <typename Value>
void cuSparse_SPMM<float>::Process(int n,float * B,float * C) {
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    // Create dense matrix B
    int ldb = n;
    int ldc = n;

    cusparseCreateDnMat(&matB, k_, n, ldb, B,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, m_, n, ldc, C,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW);
    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);


    cusparseSpMM(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
}

// template <>
void cuSparse_SPMM<double>::Process(int n,double * B,double * C) {
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    // Create dense matrix B
    int ldb = n;
    int ldc = n;

    cusparseCreateDnMat(&matB, k_, n, ldb, B,
                                       CUDA_R_64F, CUSPARSE_ORDER_ROW);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, m_, n, ldc, C,
                                       CUDA_R_64F, CUSPARSE_ORDER_ROW);
    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);


    cusparseSpMM(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
}