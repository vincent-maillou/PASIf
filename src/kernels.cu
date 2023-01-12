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



/** customSpTd2V()
 * @brief Perform the coo sparse tensor - dense vector square multiplication
 * 
 */
 __global__
 void customSpTd2V(reel *d_val, 
                   uint *d_row, 
                   uint *d_col, 
                   uint *d_slice, 
                   uint nzz,
                   reel* X, 
                   reel* Y){

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k < nzz; k += stride){
    atomicAdd(&Y[d_slice[k]], d_val[k] * X[d_row[k]] * X[d_col[k]]);
  }
 }



 /** customSpTd3V()
 * @brief Perform the coo sparse tensor 4d - dense vector multiplication (order 3)
 * 
 */
 __global__
 void customSpTd3V(reel *d_val, 
                   uint *d_row, 
                   uint *d_col, 
                   uint *d_slice, 
                   uint *d_hyperslice,
                   uint nzz,
                   reel* X, 
                   reel* Y){

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k < nzz; k += stride){
    atomicAdd(&Y[d_hyperslice[k]], d_val[k] * X[d_row[k]] * X[d_col[k]] * X[d_slice[k]]);
  }
 }



/** customAxpbyMultiForces()
 * @brief Performe a custom Axpby operation on the forces vector to accomodate multi excitation file
 * parallelisme acrose a single system
 * 
 */
 __global__
 void customAxpbyMultiForces(reel* d_val, 
                             uint* d_indice, 
                             uint nzz, 
                             reel* excitationsSet,
                             uint lengthOfeachExcitation, 
                             uint kSim, 
                             reel* Y, 
                             uint n, 
                             uint t,
                             uint intraStrmParallelism){

  uint dofStride = n/intraStrmParallelism;
  uint selectedExcitation = kSim*intraStrmParallelism;

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k<nzz; k += stride){
    // Y[d_indice[k]] += 0.0;
    Y[d_indice[k]] += d_val[k]*excitationsSet[(selectedExcitation+d_indice[k]/dofStride)*lengthOfeachExcitation + t];

  }
 }



/** updateSlope()
 * @brief Compute the next estimation vectors
 * 
 */
 __global__
 void updateSlope(reel* rki, 
                  reel* q, 
                  reel* rk, 
                  reel dt, 
                  uint n){

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k < n; k += stride){
    rki[k] = q[k] + dt*rk[k];
  }
 }



/** integrate()
 * @brief Compute the next state vector based on the rk4 estimations
 * 
 */
 __global__
 void integrate(reel* q, 
                reel* rk1, 
                reel* rk2, 
                reel* rk3, 
                reel* rk4, 
                reel h6, 
                uint n){

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k < n; k += stride){
    q[k] += h6*(rk1[k] + 2*rk2[k] + 2*rk3[k] + rk4[k]);
  }
 }
