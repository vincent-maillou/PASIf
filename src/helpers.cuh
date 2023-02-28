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
#define tensor3d std::vector<std::vector<std::vector<reel>>>
#define tensor4d std::vector<std::vector<std::vector<std::vector<reel>>>>

enum problemType {forward, backward};


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
    COOMatrix(std::array<uint,2> n_,
              std::vector<reel>  values_,
              std::vector<uint>  row_,
              std::vector<uint>  col_);
    ~COOMatrix();

    uint   extendTheSystem(uint nTimes);
    void   allocateOnGPU(cusparseHandle_t     & handle, 
                         cusparseDnVecDescr_t & vecX, 
                         cusparseDnVecDescr_t & vecY);
    size_t memFootprint();

    
    std::ostream& print(std::ostream& out) const;


   // Host-side data
    uint                nzz;
    std::array<uint, 2> n;
    std::vector<reel>   val;
    std::vector<uint>   row;
    std::vector<uint>   col;

   // Device-side data
    reel *d_val;
    uint *d_row;
    uint *d_col;

    cusparseSpMatDescr_t sparseMat_desc;
    void*  d_buffer;
    size_t bufferSize;

    reel  alpha; reel *d_alpha;
    reel  beta;  reel *d_beta;
  };

  std::ostream& operator<<(std::ostream& out, COOMatrix const& mat);



/****************************************************
 *              COO Tensor 3D
 ****************************************************/

  struct COOTensor3D{
    COOTensor3D() {};
    COOTensor3D(std::array<uint, 3> n_,
                std::vector<reel>   values_,
                std::vector<uint>   indices_);
    ~COOTensor3D();

    uint   extendTheSystem(uint nTimes);
    void   allocateOnGPU();
    size_t memFootprint();

    std::ostream& print(std::ostream& out) const;


   // Host-side data
    uint nzz;
    std::array<uint, 3> n;
    std::vector<reel>   val;
    std::vector<uint>   slice;
    std::vector<uint>   row;
    std::vector<uint>   col;
    

   // Device-side data
    reel *d_val;
    uint *d_slice;
    uint *d_row;
    uint *d_col;
  };

  std::ostream& operator<<(std::ostream& out, COOTensor3D const& tensor_);



/****************************************************
 *              COO Tensor 4D
 ****************************************************/

  struct COOTensor4D{
    COOTensor4D() {};
    COOTensor4D(std::array<uint, 4> n_,
                std::vector<reel>   values_,
                std::vector<uint>   indices_);
    ~COOTensor4D();

    uint   extendTheSystem(uint nTimes);
    void   allocateOnGPU();
    size_t memFootprint();

    std::ostream& print(std::ostream& out) const;


   // Host-side data
    uint nzz;
    std::array<uint, 4> n;
    std::vector<reel>   val;
    std::vector<uint>   hyperslice;
    std::vector<uint>   slice;
    std::vector<uint>   row;
    std::vector<uint>   col;
    

   // Device-side data
    reel *d_val;
    uint *d_hyperslice;
    uint *d_slice;
    uint *d_row;
    uint *d_col;
    
  };

  std::ostream& operator<<(std::ostream& out, COOTensor4D const& tensor_);



/****************************************************
 *              COO Vector
 ****************************************************/
  
  struct COOVector{
    COOVector() {};
    COOVector(std::vector<reel> & denseVector_);
    ~COOVector();

    uint   extendTheSystem(uint nTimes);
    void   allocateOnGPU();
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
  

  std::ostream& operator<<(std::ostream& out, matrix const& mat);

  template <typename T>
  std::ostream& operator<<(std::ostream& out, std::vector<T> const& vec);

  void printVector(std::vector<reel> & vec);
  uint extendTheVector(std::vector<reel> & vec, uint nTimes);



