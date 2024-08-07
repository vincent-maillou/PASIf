/**
 * @file kernels.cuh
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once

#include "helpers.cuh"



/** SpT3dV()
 * @brief Perform the coo sparse tensor 3d - dense vector multiplication (order 2)
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
            uint *hyperhyperslice_P,
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
            reel* Y);




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
                  int offset);



/** interpolateForces()
 * @brief Apply the interpolation matrix to the targeted excitation and add it to the
 * state vector.
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
                        bool halfStep);

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
                        bool halfStep);

 __global__
 void request_setpoint(reel* d_traj,
                    uint target_pos,
                    uint n_dofs,
                    uint* setpoint);

 __global__
 void half_setpoint(reel* d_traj,
                    uint target_pos,
                    uint n_dofs,
                    uint* setpoint);

 __global__
 void previous_setpoint(reel* d_traj,
                    uint target_pos,
                    uint n_dofs,
                    uint* setpoint);

/** updateSlope()
 * @brief Compute the next estimation vectors
 * 
 */
 __global__
 void updateSlope(reel* rki, 
                  reel* q, 
                  reel* rk, 
                  reel  dt, 
                  uint  n);



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
                uint  n);


 __global__
 void stepfwd(uint* d_step);


 __global__
 void stepbwd(uint* d_step, uint* d_setpoint);