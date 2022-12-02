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



/**
 * @brief Perform the coo sparse matrix - dense vector cube multiplication
 * 
 * @return __global__ 
 */
__global__
void SpMV3(reel *d_alpha, cusparseSpMatDescr_t sparseMat, cusparseDnVecDescr_t X,
           reel *d_beta, cusparseDnVecDescr_t Y) {};



/**
 * @brief Perform the coo sparse tensor - dense vector square multiplication
 * 
 */
__global__
void SpTV2(reel *d_alpha, reel *d_val, uint *d_row, uint *d_col, uint *d_slice, uint nzz,
           cusparseDnVecDescr_t X, reel *d_beta, cusparseDnVecDescr_t Y) {};




/**
 * @brief Performe the intra-step rk4 state update
 * 
 */
/* __global__
void updateState(reel *d_state, reel *d_k1, reel *d_k2, reel *d_k3, reel *d_k4, reel *d_dt, uint n); */



/**
 * @brief Performe the final rk4 integration
 * 
 */
/* __global__
void integrate(reel *d_state, reel *d_k1, reel *d_k2, reel *d_k3, reel *d_k4, reel *d_dt, uint n); */