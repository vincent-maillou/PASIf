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
  __GpuDriver(std::vector< std::vector<reel> > excitationSet_, uint sampleRate_);
  ~__GpuDriver();


  int __setSystems(std::vector< matrix > & M_,
                   std::vector< matrix > & B_,
                   std::vector< matrix > & K_,
                   std::vector< tensor > & Gamma_,
                   std::vector< matrix > & Lambda_,
                   std::vector< std::vector<reel> > & ForcePattern_);

  int __getAmplitudes();


 private:
  int loadExcitationsSet(std::vector<std::vector<reel>> ExcitationsSet_);
  int setCUDA(uint nStreams_);


  // Host-wise data
  std::vector<reel> excitationSet;
  uint sampleRate;
  uint numberOfExcitations;
  uint lengthOfeachExcitation;

  uint numberOfDOFs;


  // Device-wise data
  reel *d_ExcitationsSet;

  // System description
  COOMatrix *B;
  COOMatrix *K;
  COOTensor *Gamma;
  COOMatrix *Lambda;
  COOVector *ForcePattern;


  // Computation parameters
  uint nStreams;
  cudaStream_t *streams;

  uint IntraStrmParallelism;

  int  deviceId;
  int  numberOfSMs;
  uint numberOfBlocks;
  uint numberOfThreadsPerBlock;

  cusparseHandle_t h_cuSPARSE;
};






