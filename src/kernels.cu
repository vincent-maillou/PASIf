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



/** SpTd2V()
 * @brief Perform the coo sparse tensor - dense vector square multiplication
 * 
 */
 __global__
 void SpTd2V(reel *d_val, 
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



 /** SpTd3V()
 * @brief Perform the coo sparse tensor 4d - dense vector multiplication (order 3)
 * 
 */
 __global__
 void SpTd3V(reel *d_val, 
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



/** applyExcitationFiles()
 * @brief Performe a custom Axpby operation on the forces vector to accomodate 
 * multi excitation file parallelisme acrose a single system
 * 
 */
 __global__
 void applyExcitationFiles(reel* d_val, 
                           uint* d_indice, 
                           uint  nzz, 
                           reel* excitationsSet,
                           uint  lengthOfeachExcitation, 
                           uint  currentSimulation,
                           uint  systemStride,
                           reel* Y, 
                           uint  t){

  uint selectedExcitation = currentSimulation;

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k<nzz; k += stride){
    selectedExcitation += d_indice[k]/systemStride;
    Y[d_indice[k]] += d_val[k] * excitationsSet[selectedExcitation*lengthOfeachExcitation + t];
  }
 }



/** interpolateExcitationFiles()
 * @brief Performe a custom Axpby operation on the forces vector, it interpolate
 * it regarding the interpolation matrix, to accomodate multi excitation file
 * parallelisme acrose a single system
 * 
 */
 __global__
 void interpolateExcitationFiles(reel* d_val, 
                                 uint* d_indice, 
                                 uint  nzz, 
                                 reel* excitationsSet,
                                 uint  lengthOfeachExcitation, 
                                 uint  currentSimulation,
                                 uint  systemStride,
                                 reel* Y, 
                                 uint  t,
                                 reel* interpolationMatrix,
                                 uint  interpolationWindowSize,
                                 int   i){

  uint selectedExcitation = currentSimulation;
  int  iws2m1 = interpolationWindowSize/2 - 1;

  int startInterpolate = -iws2m1;
  int endInterpolate   = iws2m1+1;

  // Prevent out of bound interpolation
  if((int)t+startInterpolate < 0){
    startInterpolate = -t;
  }
  if((int)t+endInterpolate > lengthOfeachExcitation){
    endInterpolate = lengthOfeachExcitation - t;
  }

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  
  for(uint k = index; k<nzz; k += stride){
    selectedExcitation += d_indice[k]/systemStride;

    // Interpolate the excitations
    reel tmp = 0.;
    for(int j=startInterpolate; j<=endInterpolate; ++j){
      reel a = interpolationMatrix[interpolationWindowSize*i + j + iws2m1];
      reel b = excitationsSet[(selectedExcitation)*lengthOfeachExcitation + t + j];

      tmp += a * b;
    }

    Y[d_indice[k]] += d_val[k] * tmp;
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



