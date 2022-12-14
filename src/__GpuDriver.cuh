/**
 * @file __GpuDriver.cuh
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-11-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "helpers.cuh"
#include "kernels.cuh"



class __GpuDriver{
 public:
  __GpuDriver(std::vector< std::vector<double> > excitationSet_, uint sampleRate_);
  ~__GpuDriver();

  int __setSystems(std::vector< matrix > & M_,
                   std::vector< matrix > & B_,
                   std::vector< matrix > & K_,
                   std::vector< tensor > & Gamma_,
                   std::vector< matrix > & Lambda_,
                   std::vector< std::vector<reel> > & ForcePattern_,
                   std::vector< std::vector<reel> > & InitialConditions_);
                   
  std::array<std::vector<reel>, 2> __getAmplitudes();


 private:
  int  loadExcitationsSet(std::vector<std::vector<double>> ExcitationsSet_);
  int  setCUDA(uint nStreams_);
  void derivatives(cusparseDnVecDescr_t m_desc, 
                   cusparseDnVecDescr_t k_desc,
                   cusparseDnVecDescr_t q1_desc, 
                   cusparseDnVecDescr_t q2_desc,
                   uint k, 
                   uint t);
  void rkStep(uint k, 
              uint t);
  void checkAndDestroy();
  void optimizeIntraStrmParallelisme();


  //            Host-wise data
  std::vector<reel> excitationSet;
  uint sampleRate;
  uint numberOfExcitations;
  uint lengthOfeachExcitation;

  uint numberOfDOFs;


  //            Device-wise data
  reel* d_ExcitationsSet;

  // System description
  COOMatrix* B;
  COOMatrix* K;
  COOTensor* Gamma;
  COOMatrix* Lambda;
  COOVector* ForcePattern;

  // RK4 related vectors
  std::vector<reel> QinitCond; reel* d_QinitCond; 
  reel* d_Q1; cusparseDnVecDescr_t d_Q1_desc;
  reel* d_Q2; cusparseDnVecDescr_t d_Q2_desc;

  reel* d_mi; cusparseDnVecDescr_t d_mi_desc;
  reel* d_ki; cusparseDnVecDescr_t d_ki_desc;

  reel* d_m1; cusparseDnVecDescr_t d_m1_desc;
  reel* d_m2; cusparseDnVecDescr_t d_m2_desc;
  reel* d_m3; cusparseDnVecDescr_t d_m3_desc;
  reel* d_m4; cusparseDnVecDescr_t d_m4_desc;

  reel* d_k1; cusparseDnVecDescr_t d_k1_desc;
  reel* d_k2; cusparseDnVecDescr_t d_k2_desc;
  reel* d_k3; cusparseDnVecDescr_t d_k3_desc;
  reel* d_k4; cusparseDnVecDescr_t d_k4_desc;

  reel h; 
  reel h2; 
  reel h6;

  reel alpha; reel* d_alpha;
  reel beta1; reel* d_beta1;
  reel beta0; reel* d_beta0;


  // For debug purpose
  /* reel* d_trajectory; */


  //        Computation parameters
  uint nStreams;
  cudaStream_t *streams;

  uint IntraStrmParallelism;
  uint numberOfSimulationToPerform;
  uint exceedingSimulations;

  int  deviceId;
  int  numberOfSMs;
  uint nBlocks;
  uint nThreadsPerBlock;

  cublasHandle_t   h_cublas;
  cusparseHandle_t h_cuSPARSE;
};


/**
 * @brief Ideas of further optimization
 * 
 */

/*

1. cudaGraph capture of the kernels
2. cudaGraph capture of the kernels + cudaGraph capture of the data transfers
3. CUBLAS_COMPUTE_32F_FAST_TF32
4. Interleaved excitations files





*/



