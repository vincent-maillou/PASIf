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
  
  uint sp_offset = blockIdx.x*n_setpoint;
  uint src_offset = blockIdx.x*n_source;

  int k=threadIdx.x + blockIdx.y*blockDim.x;

  if(k<nzz_G) atomicAdd(&Y[d_slice_G[k]+src_offset],  d_val_G[k]*X_source[d_row_G[k]+src_offset]*X_setpoint[d_col_G[k]+sp_offset]);


  k = blockDim.x*gridDim.y-threadIdx.x-1;
  if(k<nzz_L && k>=0) atomicAdd(&Y[d_hyperslice_L[k]+src_offset], d_val_L[k]*X_source[d_slice_L[k]+src_offset]*X_setpoint[d_row_L[k]+sp_offset]*X_setpoint[d_col_L[k]+sp_offset]);


  k= blockIdx.y*blockDim.x + threadIdx.x-nzz_G;
  
  if(k<nzz_P && k>=0) atomicAdd(&Y[d_hyperhyperslice_P[k]+src_offset], d_val_P[k]*X_source[d_hyperslice_P[k]+src_offset]*X_setpoint[d_slice_P[k]+sp_offset]*X_setpoint[d_row_P[k]+sp_offset]*X_setpoint[d_col_P[k]+sp_offset]);

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

  uint k = blockIdx.x*blockDim.x + threadIdx.x;
  uint step = *d_step+offset;
  if(step<lengthOfeachExcitation && step>0 && k<nzz){
    uint selectedExcitation = d_indice[k]/systemStride;
    atomicAdd(&Y[d_indice[k]], d_val[k]*excitationsSet[selectedExcitation*lengthOfeachExcitation + step]);
  }
 }


/** interpolateForces()
 * @brief Apply the interpolation matrix to the targeted excitation and add 
 * it to the state vector.
 * 
 */
 __global__
 void interpolateForces_fwd(reel* d_val, 
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
                        bool halfStep){

  // Prevent out of bound interpolation, 0 value (no force) will be used
  // in case of out of bound
  uint selectedExcitation(0);

  uint step = *d_step+offset;
  uint log2interp = log2f(interpolationWindowSize);
  uint interpidx((((step << 1) + (halfStep?1:0)) & (interpolationWindowSize-1)));
  uint excoff((((step << 1) + (halfStep?1:0))>>log2interp));

  uint k = threadIdx.x + blockIdx.x*blockDim.x;
  if (excoff<lengthOfeachExcitation && excoff>0 && k<nzz){
      selectedExcitation += d_indice[k]/systemStride;
      uint sweepStep((selectedExcitation)*lengthOfeachExcitation);
      // Interpolate the excitations
      atomicAdd(&Y[d_indice[k]], d_val[k]*(interpolationMatrix[interpolationWindowSize+interpidx]*excitationsSet[sweepStep+excoff]+interpolationMatrix[interpidx]*excitationsSet[sweepStep+(excoff+1)]));
  }
 }

 __global__
 void interpolateForces_bwd(reel* d_val, 
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
                        bool halfStep){

  // Prevent out of bound interpolation, 0 value (no force) will be used
  // in case of out of bound
  uint selectedExcitation(0);

  uint step = *d_step+offset;
  uint log2interp = log2f(interpolationWindowSize);
  uint interpidx((((step << 1) + (halfStep?1:0)) & (interpolationWindowSize-1)));
  uint excoff((((step << 1) + (halfStep?1:0))>>log2interp));


  uint k = threadIdx.x + blockIdx.x*blockDim.x;
  if((excoff<lengthOfeachExcitation && excoff>1 && k<nzz)){
    selectedExcitation += d_indice[k]/systemStride;
      uint sweepStep((selectedExcitation)*lengthOfeachExcitation);
      // Interpolate the excitations
      atomicAdd(&Y[d_indice[k]], d_val[k]*(interpolationMatrix[interpolationWindowSize-interpidx]*excitationsSet[sweepStep+excoff]+ interpolationMatrix[interpolationWindowSize*2-interpidx]*excitationsSet[sweepStep+(excoff-1)]));
  }
 }


 __global__
 void request_setpoint(reel* d_traj,
                    uint target_pos,
                    uint n_dofs,
                    uint* setpoint){
  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint offset_target = target_pos*n_dofs;
  uint offset_rqst = *setpoint*n_dofs;
  if (index<n_dofs) d_traj[offset_target+index] = d_traj[offset_rqst+index];              
}

 __global__
 void half_setpoint(reel* d_traj,
                    uint target_pos,
                    uint n_dofs,
                    uint* setpoint){
  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint offset_target = target_pos*n_dofs;
  uint offset_0 = *setpoint*n_dofs;
  uint offset_1 = (*setpoint-1)*n_dofs;

  if (index<n_dofs) d_traj[offset_target+index] = .5*(d_traj[offset_0+index]+d_traj[offset_1+index]);              
}

 __global__
 void previous_setpoint(reel* d_traj,
                    uint target_pos,
                    uint n_dofs,
                    uint* setpoint){
  uint index  = threadIdx.x + blockIdx.x * blockDim.x;
  uint offset_target = target_pos*n_dofs;
  uint offset_rqst = (*setpoint-1)*n_dofs;
  if (index<n_dofs) d_traj[offset_target+index] = d_traj[offset_rqst+index];              
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
  if (index<n) rki[index] = q[index]+ dt*rk[index];
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

  uint k  = threadIdx.x + blockIdx.x * blockDim.x;
  if(k<n)  q[k] += h6*(rk1[k] +2*rk2[k]+2*rk3[k]+rk4[k]);
 }

 __global__
 void stepfwd(uint* d_step){
  *d_step += 1;
 }

  __global__
 void stepbwd(uint* d_step, uint* d_setpoint){
  *d_step -= 1;
  *d_setpoint -= 1;
 }
