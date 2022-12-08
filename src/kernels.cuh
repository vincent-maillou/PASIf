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



/** customSpMV3()
 * @brief Perform the coo sparse matrix - dense vector cube multiplication
 * 
 * @return __global__ 
 */
 __global__
 void customSpMV3(reel* d_val, uint* d_row, uint* d_col, uint nzz, reel* X, reel* Y);



/** customSpTV2()
 * @brief Perform the coo sparse tensor - dense vector square multiplication
 * 
 */
 __global__
 void customSpTV2(reel *d_val, uint *d_row, uint *d_col, uint *d_slice, uint nzz,
                  reel* X, reel* Y);



/** customAxpbyMultiForces()
 * @brief Performe a custom Axpby operation on the forces vector to accomodate multi excitation file
 * parallelisme acrose a single system
 * 
 */
 __global__
 void customAxpbyMultiForces(reel* d_val, uint* d_indice, uint nzz, reel* excitationsSet,
                             uint lengthOfeachExcitation, uint kSim, reel* Y, uint n, uint t,
                             uint intraStrmParallelism);



/** updateSlope()
 * @brief Compute the next estimation vectors
 * 
 */
 __global__
 void updateSlope(reel* rki, reel* q, reel* rk, reel dt, uint n);



/** integrate()
 * @brief Compute the next state vector based on the rk4 estimations
 * 
 */
 __global__
 void integrate(reel* q, reel* rk1, reel* rk2, reel* rk3, reel* rk4, reel h6, uint n);