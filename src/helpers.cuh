/**
 * @file helpers.cuh
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse_v2.h" 

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <chrono>
#include <assert.h>

#define reel float
#define reel_eps  std::numeric_limits<reel>::epsilon()

#define matrix std::vector<std::vector<reel>>
#define tensor std::vector<std::vector<std::vector<reel>>>



/****************************************************
 *              CUDA Check for error
 ****************************************************/

  #define CHECK_CUDA(func)                                                       \
  {                                                                              \
      cudaError_t status = (func);                                               \
      if (status != cudaSuccess) {                                               \
          printf("CUDA API failed at line %d with error: %s (%d)\n",             \
                __LINE__, cudaGetErrorString(status), status);                   \
      }                                                                          \
  }

  #define CHECK_CUBLAS(func)                                                     \
  {                                                                              \
      cublasStatus_t status = (func);                                            \
      if (status != CUBLAS_STATUS_SUCCESS) {                                     \
          printf("CUBLAS API failed at line %d with error: %s (%d)\n",           \
                __LINE__, cublasGetStatusString(status), status);                \
      }                                                                          \
  }

  #define CHECK_CUSPARSE(func)                                                   \
  {                                                                              \
      cusparseStatus_t status = (func);                                          \
      if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
          printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
                __LINE__, cusparseGetErrorString(status), status);               \
      }                                                                          \
  }                                                                             



/****************************************************
 *              COO Matrix
 ****************************************************/
  
  struct COOMatrix{
    COOMatrix() {};
    COOMatrix(std::vector< matrix > & denseMatrix, std::vector< matrix > & scaleMatrix);
    ~COOMatrix();

    uint ExtendTheSystem(uint nTimes);
    void AllocateOnGPU(cusparseHandle_t & handle, cusparseDnVecDescr_t & vecX, cusparseDnVecDescr_t & vecY);
    size_t memFootprint();
    
    std::ostream& print(std::ostream& out) const;


   // Host-side data
    uint nzz;
    uint n;
    std::vector<reel> val;
    std::vector<uint> row;
    std::vector<uint> col;

   // Device-side data
    reel *d_val;
    uint *d_row;
    uint *d_col;

    cusparseSpMatDescr_t sparseMat_desc;
    void *d_buffer;
    size_t bufferSize;

    reel alpha;
    reel beta;
    reel *d_alpha;
    reel *d_beta;
  };

  std::ostream& operator<<(std::ostream& out, COOMatrix const& mat);



/****************************************************
 *              COO Tensor
 ****************************************************/

  struct COOTensor{
    COOTensor() {};
    COOTensor(std::vector< tensor > & denseTensor, std::vector< matrix > & scaleMatrix);
    ~COOTensor();

    uint ExtendTheSystem(uint nTimes);
    void AllocateOnGPU();
    size_t memFootprint();

    std::ostream& print(std::ostream& out) const;


   // Host-side data
    uint nzz;
    uint n;
    std::vector<reel> val;
    std::vector<uint> row;
    std::vector<uint> col;
    std::vector<uint> slice;

   // Device-side data
    reel *d_val;
    uint *d_row;
    uint *d_col;
    uint *d_slice;
  };

  std::ostream& operator<<(std::ostream& out, COOTensor const& tensor_);



/****************************************************
 *              COO Vector
 ****************************************************/
  
  struct COOVector{
    COOVector() {};
    COOVector(std::vector< std::vector<reel> > & denseVector);
    ~COOVector();

    uint ExtendTheSystem(uint nTimes);
    void AllocateOnGPU();
    size_t memFootprint();

    std::ostream& print(std::ostream& out) const;

    // host-side data
    uint nzz;
    uint n;
    std::vector<reel> val;
    std::vector<uint> indice;

    // device-side data
    reel *d_val;
    uint *d_indice;

    cusparseSpVecDescr_t sparseVec_desc;
  };

  std::ostream& operator<<(std::ostream& out, COOVector const& vector_);



/****************************************************
 *              Utilities
 ****************************************************/
  
  void invertMatrix(std::vector< matrix > & vectMat, float scaleFactor);

  std::ostream& operator<<(std::ostream& out, matrix const& mat);

  template <typename T>
  std::ostream& operator<<(std::ostream& out, std::vector<T> const& vec);

  void printVector(std::vector<reel> & vec);
  uint extendTheVector(std::vector<reel> & vec, uint nTimes);



