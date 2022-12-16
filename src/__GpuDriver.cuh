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

  int __loadExcitationsSet(std::vector< std::vector<double> > excitationSet_, uint sampleRate_);

  int __setSystems(std::vector< matrix > & M_,
                   std::vector< matrix > & B_,
                   std::vector< matrix > & K_,
                   std::vector< tensor > & Gamma_,
                   std::vector< matrix > & Lambda_,
                   std::vector< std::vector<reel> > & ForcePattern_,
                   std::vector< std::vector<reel> > & InitialConditions_);
                   
  std::array<std::vector<reel>, 2> __getAmplitudes();


 private:
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



IDRIS application:
Background:
We're a team of 3:
- Vincent Maillou, doing his master thesis at AMOLF, graduating from Ecole Centrale Nantes in 2023. MSc computer science, specialization
 in High Peformance Computing. Wrote the first version of the targeted code for this hackathon in C++ and CUDA.
- Theophile louvet, PhD student at AMOLF. MSc Transport engineering from ESTACA and a data science specialization 
degree at CentraleSupelec. He contributed to the gradient part of the code.
- Parisa Omidvar, a PhD student in the field of computational mechanics at AMOLF. MSc in mechanical engineering. User of the
code, working on a different project than Theophile and Vincent but benefit from the code. New to GPU programming but looking
forward to improve her skills.

Vincent will leave the team in Mai for a PhD at the ETH Zurich and therefore one of the side goal of this hackathon 
for us is to ensure that both Theophile and Parisa will be able to maintain the code after Vincent's departure.


Skillsets:
- Vincent: C++/CUDA/MPI/Cluster
- Theophile: Python/C++/CUDA/Optimization
- Parisa: Python/Computational mechanics/GPU programming enthousiast



Brought together by our supervisor: Dr Marc Serra-Garcia, tenure track group leader (Hypermart Matter-group at AMOLF)
Marc got his MSc in Aerospace, Aeronautical and Astronautical Engineering from Caltech in 2012. He got his PhD
from ETH Zurich in 2017 in the field of non-linear and stochastic physics. He is a computational scientist with 
a strong background in numerical methods.
Refactor:
  "Our research supervisor is Dr. Marc Serra-Garcia, a tenure track group leader at the Hypermart 
  Matter-group at AMOLF. Dr. Serra-Garcia holds a MSc in Aerospace, Aeronautical and Astronautical 
  Engineering from Caltech and a PhD in non-linear and stochastic physics from ETH Zurich. He is a 
  highly skilled computational scientist with a strong background in numerical methods."




What make us unique:
- We're a team of 3, with complementary skills, working on connected but different projects that used
the same code. 
- We're higly motivated to improve the code and learn from this event. Some of us are already familiar with
GPU programming and high performance computing, some of us are discovering this field. 
We're all eager to learn and improve our skills!
- We'll be 100% dedicated to this event during the all duration of the hackathon, looking forward to it.
Refactor:
  "Our team of 3 brings a range of complementary skills and diverse projects that utilize the same codebase. 
  While some of us have prior experience with GPU programming and high performance computing, others are 
  excited to dive into this field for the first time. We are highly motivated to improve our skills and the 
  code through this hackathon, and are fully dedicated to the event for its duration. We look forward to the 
  opportunity to learn and grow together."


*/



