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



/** customSpTV2()
 * @brief Perform the coo sparse tensor 3d - dense vector multiplication (order 2)
 * 
 */
 __global__
 void customSpTd2V(reel *d_val, 
                   uint *d_row, 
                   uint *d_col, 
                   uint *d_slice, 
                   uint nzz,
                   reel* X, 
                   reel* Y);

            
          

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
                   reel* Y);



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
                             uint intraStrmParallelism);



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


