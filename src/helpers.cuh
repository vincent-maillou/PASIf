#pragma once

#include <cuda_runtime.h>
#include "cusparse_v2.h" 

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <chrono>
#include <assert.h>

#define reel float

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
          return EXIT_FAILURE;                                                   \
      }                                                                          \
  }

  #define CHECK_CUSPARSE(func)                                                   \
  {                                                                              \
      cusparseStatus_t status = (func);                                          \
      if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
          printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
                __LINE__, cusparseGetErrorString(status), status);               \
          return EXIT_FAILURE;                                                   \
      }                                                                          \
  }                                                                             



/****************************************************
 *              Data structures
 ****************************************************/

  struct COOMatrix{
    COOMatrix() {};
    COOMatrix(std::vector< matrix > & denseMatrix, std::vector< matrix > & scaleMatrix);
    ~COOMatrix();

   // Host-side data
    uint nzz;
    std::vector<reel> val;
    std::vector<uint> row;
    std::vector<uint> col;

   // Device-side data
    reel *d_val;
    uint *d_row;
    uint *d_col;
  };

  struct COOTensor{
    COOTensor() {};
    COOTensor(std::vector< tensor > & denseTensor, std::vector< matrix > & scaleMatrix);
    ~COOTensor();

   // Host-side data
    uint nzz;
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

