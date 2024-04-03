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
            uint n0,
            uint nlast,
            reel* X0,
            reel* Xlast,
            reel* Y){
      
    uint k = threadIdx.x;
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
    int shift = l*n0;

    //Tensor 3D element
    while (k <nzz_G){//While pointing to a tensor elemnt
      slice_idx = d_slice_G[k];
      val = d_val_G[k];
      row_idx = d_row_G[k];
      col_idx = d_col_G[k];
      //getting C00 indexes
      atomicAdd(&Y[slice_idx+shift], __fmul_rn(__fmul_rn(val, X0[row_idx+shift]), Xlast[col_idx+l*nlast]));//Tensor contraction
      k += blockDim.x;//in case there are more non zero elements than thread possibles
    }

  k = blockDim.x - threadIdx.x;
  k = threadIdx.x;

  //For the 4D tensor we start from the back
  //This avoids the first few threads to take care of all non linear elements

  //Tensor 4D element
  while (k <nzz_L){
    hyperslice_idx = d_hyperslice_L[k];
    slice_idx = d_slice_L[k];
    val = d_val_L[k];
    row_idx = d_row_L[k];
    col_idx = d_col_L[k];
    atomicAdd(&Y[hyperslice_idx+shift], __fmul_rn(__fmul_rn(__fmul_rn(val, X0[slice_idx+shift]), X0[row_idx+shift]), Xlast[col_idx+l*nlast]));
    k+= blockDim.x;
  }

  k = threadIdx.x+nzz_G;
  k = threadIdx.x;

  //For Psi we start in the middle, with an offset

  //Tensor 5D element
  while (k <nzz_P){
    hyperhyperslice_idx = d_hyperhyperslice_P[k];
    hyperslice_idx = d_hyperslice_P[k];
    slice_idx = d_slice_P[k];
    val = d_val_P[k];
    row_idx = d_row_P[k];
    col_idx = d_col_P[k];
    if(l<ntimes){
      atomicAdd(&Y[hyperhyperslice_idx+shift], __fmul_rn(X0[hyperhyperslice_idx+shift], __fmul_rn(__fmul_rn(__fmul_rn(val, X0[slice_idx+shift]), X0[row_idx+shift]), Xlast[col_idx+l*nlast])));
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
                  uint  t){

  uint selectedExcitation = 0;

  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  
  for(uint k = index; k<nzz; k += stride){
    selectedExcitation += d_indice[k]/systemStride;
    Y[d_indice[k]] += d_val[k] * excitationsSet[selectedExcitation*lengthOfeachExcitation + t];
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
                        uint excoff,
                        uint interpidx,
                        bool backward){

  // Prevent out of bound interpolation, 0 value (no force) will be used
  // in case of out of bound
  uint selectedExcitation(0);
  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;  

  if(backward){
    for(uint k = index; k<nzz; k += stride){
      selectedExcitation += d_indice[k]/systemStride;
      uint sweepStep((selectedExcitation)*lengthOfeachExcitation);
      // Interpolate the excitations
      Y[d_indice[k]] += __fmul_rn(d_val[k], __fadd_rn(__fmul_rn( interpolationMatrix[interpolationWindowSize-interpidx], excitationsSet[sweepStep+excoff]), __fmul_rn(interpolationMatrix[interpolationWindowSize*2-interpidx], excitationsSet[sweepStep+(excoff-1)])));
    }
  }else{
    for(uint k = index; k<nzz; k += stride){
      selectedExcitation += d_indice[k]/systemStride;
      uint sweepStep((selectedExcitation)*lengthOfeachExcitation);
      // Interpolate the excitations
      Y[d_indice[k]] += __fmul_rn(d_val[k], __fadd_rn(__fmul_rn( interpolationMatrix[interpolationWindowSize+interpidx], excitationsSet[sweepStep+excoff]), __fmul_rn(interpolationMatrix[interpidx], excitationsSet[sweepStep+(excoff+1)])));
    }
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



