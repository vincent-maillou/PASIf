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



/** SpT3dV()
 * @brief Perform the coo sparse tensor - dense vector square multiplication
 * 
 */
 __global__
 void SpT3dV(reel *d_val, 
             uint *d_row, 
             uint *d_col, 
             uint *d_slice, 
             uint  nzz,
             reel* X1,
             reel* X2, 
             reel* Y){

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k < nzz; k += stride){
    atomicAdd(&Y[d_slice[k]], d_val[k] * X1[d_row[k]] * X2[d_col[k]]);
  }
 }



 /** SpT4dV()
 * @brief Perform the coo sparse tensor 4d - dense vector multiplication (order 3)
 * 
 */
 __global__
 void SpT4dV(reel *d_val, 
             uint *d_row, 
             uint *d_col, 
             uint *d_slice, 
             uint *d_hyperslice,
             uint  nzz,
             reel* X1,
             reel* X2,
             reel* X3, 
             reel* Y){

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k < nzz; k += stride){
    atomicAdd(&Y[d_hyperslice[k]], d_val[k] * X1[d_slice[k]] * X2[d_col[k]] * X3[d_row[k]]);
  }
 }



/** applyForces()
 * @brief Performe a custom AXPY operation on the state vector using the targeted 
 * excitation.
 * 
 */
 __global__
 void applyForces(reel* d_val, 
                  uint* d_indice, 
                  uint  nzz, 
                  reel* excitationsSet,
                  uint  lengthOfeachExcitation, 
                  uint  currentSimulation,
                  uint  systemStride,
                  reel* Y, 
                  uint  t,
                  reel* modulationBuffer,
                  uint  m){

  uint selectedExcitation = currentSimulation;

  reel modulation = modulate(modulationBuffer, m);

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  
  for(uint k = index; k<nzz; k += stride){
    selectedExcitation += d_indice[k]/systemStride;
    Y[d_indice[k]] += modulation * d_val[k] * excitationsSet[selectedExcitation*lengthOfeachExcitation + t];
  }
 }



/** interpolateForces()
 * @brief Apply the interpolation matrix to the targeted excitation and add it to the
 * state vector.
 * 
 */
 __global__
 void interpolateForces(reel* d_val, 
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
                        int   i,
                        reel* modulationBuffer,
                        uint  m){

  uint selectedExcitation = currentSimulation;
  int  iws2m1 = interpolationWindowSize/2 - 1;

  int startInterpolate = -iws2m1;
  int endInterpolate   = iws2m1+1;

  // Prevent out of bound interpolation, 0 value (no force) will be used
  // in case of out of bound
  if((int)t+startInterpolate < 0){
    startInterpolate = -t;
  }
  if((int)t+endInterpolate > lengthOfeachExcitation){
    endInterpolate = lengthOfeachExcitation - t;
  }

  reel modulation = modulate(modulationBuffer, m);

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  
  for(uint k = index; k<nzz; k += stride){
    selectedExcitation += d_indice[k]/systemStride;

    // Interpolate the excitations
    reel tmp = 0.;
    for(int j=startInterpolate; j<=endInterpolate; ++j){
      reel a = interpolationMatrix[interpolationWindowSize*(i-1) + j + iws2m1];
      reel b = excitationsSet[(selectedExcitation)*lengthOfeachExcitation + t + j];

      tmp += modulation * a * b;
    }

    Y[d_indice[k]] += d_val[k] * tmp;
  }
 }



 /** modulate()
 * @brief If the modulation buffer is not null, return the value at the given index
 * 
 */
 __device__
 reel modulate(reel* modulationBuffer, 
               uint  m){

  if(modulationBuffer != NULL)
    return modulationBuffer[m];

  return 1.;
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



