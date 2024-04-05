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
      
    int k = threadIdx.x;
    //K is the index of the non-linear element
    uint l = blockIdx.x;
    //l is the index of the execitation

    //index variables
    uint hyperhyperslice_idx = 0;
    uint hyperslice_idx = 0;
    uint slice_idx = 0;
    reel val = 0;
    uint row_idx = 0;
    uint col_idx = 0;
    int shift_src = l*n_source;
    int shift_spt = l*n_setpoint;

    //Tensor 3D element
    while (k <nzz_G){//While pointing to a tensor elemnt
      slice_idx = d_slice_G[k]+shift_spt;
      row_idx = d_row_G[k]+shift_src;
      col_idx = d_col_G[k]+shift_spt;
      val = d_val_G[k];
      //getting C00 indexes
      atomicAdd(&Y[slice_idx], __fmul_rn(__fmul_rn(val, X_source[row_idx]), X_setpoint[col_idx]));//Tensor contraction
      k += blockDim.x;//in case there are more non zero elements than thread possibles
    }

  k = blockDim.x - threadIdx.x-1;
  //For the 4D tensor we start from the back
  //This avoids the first few threads to take care of all non linear elements

  //Tensor 4D element
  while (k <nzz_L){
    hyperslice_idx = d_hyperslice_L[k]+shift_spt;
    slice_idx = d_slice_L[k]+shift_src;
    row_idx = d_row_L[k]+shift_spt;
    col_idx = d_col_L[k]+shift_spt;
    val = d_val_L[k];
    atomicAdd(&Y[hyperslice_idx], __fmul_rn(__fmul_rn(__fmul_rn(val, X_source[slice_idx]), X_setpoint[row_idx]), X_setpoint[col_idx]));
    k += blockDim.x;
  }

  k = threadIdx.x-nzz_G;
  //For Psi we start in the middle,w ith an offset

  //Tensor 5D element
  while (k <nzz_P && k>=0){
    hyperhyperslice_idx = d_hyperhyperslice_P[k]+shift_spt;
    hyperslice_idx = d_hyperslice_P[k]+shift_src;
    slice_idx = d_slice_P[k]+shift_spt;
    row_idx = d_row_P[k]+shift_spt;
    col_idx = d_col_P[k]+shift_spt;
    val = d_val_P[k];
    if(l<ntimes){
      atomicAdd(&Y[hyperhyperslice_idx], __fmul_rn(X_source[hyperslice_idx], __fmul_rn(__fmul_rn(__fmul_rn(val, X_setpoint[slice_idx]), X_setpoint[row_idx]), X_setpoint[col_idx])));
    }
    k+= blockDim.x;
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
  if((excoff<lengthOfeachExcitation && excoff>1 && backward)){
    selectedExcitation += d_indice[k]/systemStride;
    uint sweepStep((selectedExcitation)*lengthOfeachExcitation);
    // Interpolate the excitations
    atomicAdd(&Y[d_indice[k]], __fmul_rn(d_val[k], __fadd_rn(__fmul_rn( interpolationMatrix[interpolationWindowSize-interpidx], excitationsSet[sweepStep+excoff]), __fmul_rn(interpolationMatrix[interpolationWindowSize*2-interpidx], excitationsSet[sweepStep+(excoff-1)]))));
  }else if (excoff<lengthOfeachExcitation && excoff>0 && !backward){
      selectedExcitation += d_indice[k]/systemStride;
      uint sweepStep((selectedExcitation)*lengthOfeachExcitation);
      // Interpolate the excitations
      atomicAdd(&Y[d_indice[k]], __fmul_rn(d_val[k], __fadd_rn(__fmul_rn( interpolationMatrix[interpolationWindowSize+interpidx], excitationsSet[sweepStep+excoff]), __fmul_rn(interpolationMatrix[interpidx], excitationsSet[sweepStep+(excoff+1)]))));
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
                reel  h6, 
                uint  n){

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  for(uint k = index; k < n; k += stride){
    q[k] += h6*(rk1[k] + 2*rk2[k] + 2*rk3[k] + rk4[k]);
  }
 }

 __global__
 void stepfwd(uint* d_step){
  *d_step += 1;
 }