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
 void SpT3dV(reel *d_val, 
             uint *d_row, 
             uint *d_col, 
             uint *d_slice, 
             uint  nzz,
             reel* X, 
             reel* Y);

            
          

/** SpT4dV()
 * @brief Perform the coo sparse tensor 4d - dense vector multiplication (order 3).
 * 
 */
 __global__
 void SpT4dV(reel *d_val, 
             uint *d_row, 
             uint *d_col, 
             uint *d_slice, 
             uint *d_hyperslice,
             uint  nzz,
             reel* X, 
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
                  uint  currentSimulation,
                  uint  systemStride,
                  reel* Y, 
                  uint  t,
                  reel* modulationBuffer,
                  uint  m);



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
                        uint  m);



/** modulate()
 * @brief If the modulation buffer is not null, return the value at the given index
 * 
 */
 __device__
 reel modulate(reel* modulationBuffer, 
               uint  m);



/** updateSlope()
 * @brief Compute the next estimation vectors
 * 
 */
 __global__
 void updateSlope(reel* rki, 
                  reel* q, 
                  reel* rk, 
                  reel dt, 
                  uint n);



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
                uint n);


