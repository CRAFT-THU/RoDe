#ifndef _H_CUSPARSE_SPMM_H_
#define _H_CUSPARSE_SPMM_H_


#include <cuda_runtime.h>
#include "cusparse.h"
#include "stdio.h"

template <typename Value> 
class cuSparse_SPMM {
public:
    void Preprocess(int m,int k,int nonzeros,
                    int *row_offsets,int* column_indices,Value* values);

    void Process(int n,Value * B,Value * C); 

private:
    cusparseStatus_t status;
    cusparseHandle_t handle=0;   
    cusparseSpMatDescr_t matA;  
    cusparseDnMatDescr_t matB, matC; 
    Value alpha = 1.0f,beta = 0.0f;    

    int m_,k_;    
};

template <> 
class cuSparse_SPMM<float> {
public:
    void Preprocess(int m,int k,int nonzeros,
                    int *row_offsets,int* column_indices,float* values);

    void Process(int n,float * B,float * C); 

private:
    cusparseStatus_t status;
    cusparseHandle_t handle=0;   
    cusparseSpMatDescr_t matA;  
    cusparseDnMatDescr_t matB, matC; 
    float alpha = 1.0f,beta = 0.0f;    

    int m_,k_;    
};


template<>
class cuSparse_SPMM<double> {
public:
    void Preprocess(int m,int k,int nonzeros,
                    int *row_offsets,int* column_indices,double* values);
    void Process(int n,double * B,double * C);

private:
    cusparseStatus_t status;
    cusparseHandle_t handle=0;   
    cusparseSpMatDescr_t matA;  
    cusparseDnMatDescr_t matB, matC; 
    double alpha = 1.0f,beta = 0.0f;    

    int m_,k_; 
};

#endif