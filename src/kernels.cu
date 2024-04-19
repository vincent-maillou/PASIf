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
 * @brief Perform the coo sparse tensor - dense vector quadratic contraction
 * 
 */
 __global__
 void SpTdV(reel *d_val_G, 
            uint *d_slice_G, 
            uint *d_row_G, 
            uint *d_col_G, 
            uint  nzz_G,
            reel *d_val_L, 
            uint *d_hyperslice_L, 
            uint *d_slice_L, 
            uint *d_row_L, 
            uint *d_col_L,
            uint nzz_L,
            reel *d_val_P,
            uint *d_hyperhyperslice_P,
            uint *d_hyperslice_P, 
            uint *d_slice_P, 
            uint *d_row_P, 
            uint *d_col_P,
            uint  nzz_P,
            uint ntimes,
            uint n_source,
            uint n_setpoint,
            reel* X_source,
            reel* X_setpoint,
            reel* Y){
  
  uint stride = blockDim.x;
  uint sp_offset = blockIdx.x*n_setpoint;
  uint src_offset = blockIdx.x*n_source;

  uint k=threadIdx.x;
  while(k<nzz_G){      
    atomicAdd(&Y[d_slice_G[k]+src_offset],  d_val_G[k]*X_source[d_row_G[k]+src_offset]*X_setpoint[d_col_G[k]+sp_offset]);
    k += stride;
  }

  k=blockDim.x-threadIdx.x;
  while(k<nzz_L){
    atomicAdd(&Y[d_hyperslice_L[k]+src_offset], d_val_L[k]*X_source[d_slice_L[k]+src_offset]*X_setpoint[d_row_L[k]+sp_offset]*X_setpoint[d_col_L[k]+sp_offset]);
    k += stride;
  }

  k=threadIdx.x-nzz_G;
  while(k<nzz_P && k>0){
    atomicAdd(&Y[d_hyperhyperslice_P[k]+src_offset], d_val_P[k]*X_source[d_hyperslice_P[k]+src_offset]*X_setpoint[d_slice_P[k]+sp_offset]*X_setpoint[d_row_P[k]+sp_offset]*X_setpoint[d_col_P[k]+sp_offset]);
    k += stride;
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
                  uint  systemStride,
                  reel* Y, 
                  uint* d_step,
                  int offset){

  uint selectedExcitation = 0;
  uint k = blockIdx.x;
  uint step = *d_step+offset;
  if(step<lengthOfeachExcitation){
    selectedExcitation += d_indice[k]/systemStride;
    atomicAdd(&Y[d_indice[k]], __fmul_rn(d_val[k], excitationsSet[selectedExcitation*lengthOfeachExcitation + step]));
  }
 }



/** interpolateForces()
 * @brief Apply the interpolation matrix to the targeted excitation and add 
 * it to the state vector.
 * 
 */
 __global__
 void interpolateForces(reel* d_val, 
                        uint* d_indice, 
                        uint  nzz, 
                        reel* excitationsSet,
                        uint  lengthOfeachExcitation, 
                        uint  systemStride,
                        reel* Y, 
                        reel* interpolationMatrix,
                        uint  interpolationWindowSize,
                        uint* d_step,
                        int offset,
                        bool halfStep,
                        bool backward){

  // Prevent out of bound interpolation, 0 value (no force) will be used
  // in case of out of bound
  uint selectedExcitation(0);

  uint step = *d_step+offset;
  uint interpidx((((step << 1) + (halfStep?1:0)) & (interpolationWindowSize-1)));
  uint excoff((((step << 1) + (halfStep?1:0))>>2));


  uint k = blockIdx.x;
  // if((excoff<lengthOfeachExcitation && excoff>1 && backward)){
  //   selectedExcitation += d_indice[k]/systemStride;
  //   uint sweepStep((selectedExcitation)*lengthOfeachExcitation);
  //   // Interpolate the excitations
  //   atomicAdd(&Y[d_indice[k]], __fmul_rn(d_val[k], __fadd_rn(__fmul_rn( interpolationMatrix[interpolationWindowSize-interpidx], excitationsSet[sweepStep+excoff]), __fmul_rn(interpolationMatrix[interpolationWindowSize*2-interpidx], excitationsSet[sweepStep+(excoff-1)]))));
  // }else if (excoff<lengthOfeachExcitation && excoff>0 && !backward){
  if (excoff<lengthOfeachExcitation){
      selectedExcitation += d_indice[k]/systemStride;
      uint sweepStep((selectedExcitation)*lengthOfeachExcitation);
      // Interpolate the excitations
      atomicAdd(&Y[d_indice[k]], d_val[k]*(interpolationMatrix[interpolationWindowSize+interpidx]*excitationsSet[sweepStep+excoff]+interpolationMatrix[interpidx]*excitationsSet[sweepStep+(excoff+1)]));
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
                  reel  dt, 
                  uint  n){

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k < n; k += stride){
    rki[k] = q[k]+ dt*rk[k];
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
                reel  h6, 
                uint  n){

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k < n; k += stride){
    atomicAdd(&q[k], h6*(rk1[k] +2*rk2[k]+2*rk3[k]+rk4[k]));
    // __syncthreads();
  }
 }

 __global__
 void stepfwd(uint* d_step){
  *d_step += 1;
 }

