/**
 * @file kernels.cu
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "kernels.cuh"




/** customSpMV3()
 * @brief Perform the coo sparse matrix - dense vector cube multiplication
 * 
 * @return __global__ 
 */
 __global__
 void customSpMV3(reel *d_alpha, reel* d_val, uint* d_row, uint* d_col, 
                  reel* X, reel *d_beta, reel* Y){
  //do something
  ;
 }



/** customSpTV2()
 * @brief Perform the coo sparse tensor - dense vector square multiplication
 * 
 */
 __global__
 void customSpTV2(reel *d_alpha, reel *d_val, uint *d_row, uint *d_col, uint *d_slice, uint nzz,
                  reel* X, reel *d_beta, reel* Y){
  

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;  

  for(int k = index; k < nzz; k += stride){
    atomicAdd(&Y[d_slice[k]], d_val[k] * X[d_row[k]] * X[d_col[k]]);
  }
 }



/** customAxpbyMultiForces()
 * @brief Performe a custom Axpby operation on the forces vector to accomodate multi excitation file
 * parallelisme acrose a single system
 * 
 */
 __global__
 void customAxpbyMultiForces(reel* alpha, reel* d_val, uint* d_indice, reel* excitationsSet,
                             reel* beta, reel* Y, uint n, uint t, uint intraSystParal){
  //do something
  ;
 }



/** updateSlope()
 * @brief Compute the next estimation vectors
 * 
 */
 __global__
 void updateSlope(reel* rki, reel* q, reel* rk, reel dt, uint n){

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;  

  for(int k = index; k < n; k += stride){
    rki[k] = q[k] + dt*rk[k];
  }
 }



/** integrate()
 * @brief Compute the next state vector based on the rk4 estimations
 * 
 */
 __global__
 void integrate(reel* q, reel* rk1, reel* rk2, reel* rk3, reel* rk4, reel h6, uint n){

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k < n; k += stride){
    q[k] += h6*(rk1[k] + 2*rk2[k] + 2*rk3[k] + rk4[k]);
  }
 }