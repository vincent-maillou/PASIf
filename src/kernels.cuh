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



/** SpTV2()
 * @brief Perform the coo sparse tensor 3d - dense vector multiplication (order 2)
 * 
 */
 __global__
 void SpTd2V(reel *d_val, 
             uint *d_row, 
             uint *d_col, 
             uint *d_slice, 
             uint  nzz,
             reel* X, 
             reel* Y);

            
          

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
             uint  nzz,
             reel* X, 
             reel* Y);



/** applyExcitationFiles()
 * @brief Performe a custom Axpby operation on the forces vector to accomodate multi excitation file
 * parallelisme acrose a single system
 * 
 */
/*  __global__
 void applyExcitationFiles(reel* d_val, 
                           uint* d_indice, 
                           uint  nzz, 
                           reel* excitationsSet,
                           uint  lengthOfeachExcitation, 
                           uint  currentSimulation,
                           uint  systemStride,
                           reel* Y, 
                           uint  t); */



/** interpolateExcitationFiles()
 * @brief Performe a custom Axpby operation on the forces vector, it interpolate
 * it regarding the interpolation matrix, to accomodate multi excitation file
 * parallelisme acrose a single system
 * 
 */
/*  __global__
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
                                 int   i); */



/** modterpolator()
 * @brief Performe a custom Axpby operation on the forces vector, if needed
 * it interpolate the forces w.r.t the interpolation matrix otherwise it just
 * apply the forces.
 * 
 */
 __global__
 void modterpolator(reel* d_val, 
                    uint* d_indice, 
                    uint  nzz, 
                    reel* excitationsSet,
                    uint  lengthOfeachExcitation, 
                    uint  currentSimulation,
                    uint  systemStride,
                    reel* Y, 
                    uint  t,
                    reel* interpolationMatrix,
                    uint  interpolationNumberOfPoints,
                    uint  interpolationWindowSize,
                    int   i);

__device__
void applyForces(reel* d_val, 
                 uint* d_indice, 
                 uint  nzz, 
                 reel* excitationsSet,
                 uint  lengthOfeachExcitation, 
                 uint  currentSimulation,
                 uint  systemStride,
                 reel* Y, 
                 uint  t);

__device__
void interpolate(reel* d_val, 
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
                 int   i);



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


