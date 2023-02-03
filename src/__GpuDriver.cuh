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
  __GpuDriver(std::vector<std::vector<double> > excitationSet_,
              uint sampleRate_,
              uint numsteps_);

  ~__GpuDriver();

  int  _loadExcitationsSet(std::vector< std::vector<double> > excitationSet_,
                           uint sampleRate_);

  void _setB(std::vector< matrix > & B_); 
  void _setK(std::vector< matrix > & K_);
  void _setGamma(std::vector< tensor3d > & Gamma_);
  void _setLambda(std::vector< tensor4d > & Lambda_);
  void _setForcePattern(std::vector< std::vector<reel> > & ForcePattern_);
  void _setInitialConditions(std::vector< std::vector<reel> > & InitialConditions_);
                   
  void _setInterpolationMatrix(std::vector<reel> & interpolationMatrix_,
                               uint interpolationWindowSize_);

  void _setModulationBuffer(std::vector<reel> & modulationBuffer_);
                  
  std::vector<reel> _getAmplitudes(bool dCompute_ = false,
                                   bool dSystem_  = false);

  std::vector<std::vector<reel>> _getTrajectory(std::vector<uint> relevantsTrajs_);

 private:
  // Initialization functions
  int  setCUDA(uint nStreams_);
  void setTimesteps();
  void allocateDeviceStatesVector();
  void allocateDeviceSystems();
  void resetStatesVectors();

  // Simulation related functions
  void forwardRungeKutta(uint tStart_, 
                         uint tEnd_,
                         uint k);

  void rkStep(uint k, 
              uint t,
              uint i,
              uint m);

  void derivatives(cusparseDnVecDescr_t m_desc, 
                   cusparseDnVecDescr_t q_desc, 
                   uint k, 
                   uint t,
                   uint i,
                   uint m);

  void modterpolator(reel* Y,
                     uint  k,
                     uint  t,
                     uint  i,
                     uint  m);

  // GPU work distribution functions
  void optimizeIntraStrmParallelisme();

  // Display functions
  void displaySimuInfos(bool dCompute_=false,
                        bool dSystem_=false,
                        bool dSolver_=false);

  // Memory cleaning functions
  void clearDeviceStatesVector();
  void clearB();
  void clearK();
  void clearGamma();
  void clearLambda();
  void clearForcePattern();
  void clearInitialConditions();
  void clearInterpolationMatrix();
  void clearModulationBuffer();




  //            Simulation related data
  std::vector<reel> excitationSet;
  uint sampleRate;
  uint numberOfExcitations;
  uint lengthOfeachExcitation;

  uint n_dofs;
  uint numsteps; 

  //            Interpolation related data
  uint interpolationNumberOfPoints; // = Height of the interpolation matrix
  uint interpolationWindowSize;     // = Width of the interpolation matrix
  std::vector<reel> interpolationMatrix;
  reel*             d_interpolationMatrix;

  //            Modulation related data
  uint modulationBufferSize;
  std::vector<reel> modulationBuffer;
  reel*             d_modulationBuffer;

  //            Device-wise data
  reel* d_ExcitationsSet;

  // System matrix description
  COOMatrix*   B;
  COOMatrix*   K;
  COOTensor3D* Gamma;
  COOTensor4D* Lambda;
  COOVector*   ForcePattern;

  // RK4 States vectors
  std::vector<reel> QinitCond; reel* d_QinitCond; 
  reel* d_Q;  cusparseDnVecDescr_t d_Q_desc;

  reel* d_mi; cusparseDnVecDescr_t d_mi_desc;

  reel* d_m1; cusparseDnVecDescr_t d_m1_desc;
  reel* d_m2; cusparseDnVecDescr_t d_m2_desc;
  reel* d_m3; cusparseDnVecDescr_t d_m3_desc;
  reel* d_m4; cusparseDnVecDescr_t d_m4_desc;

  reel h; 
  reel h2; 
  reel h6;

  reel alpha; reel* d_alpha;
  reel beta1; reel* d_beta1;
  reel beta0; reel* d_beta0;

  // getTrajectory() related attributes
  std::vector<uint> relevantsTrajs;
  std::vector<std::vector<reel>> h_trajectories; reel* d_trajectories;
  

  //        Computation parameters
  uint nStreams;
  cudaStream_t *streams;

  uint intraStrmParallelism;
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
- [ ] Interpola/Mod matrix in cst memory
- [ ] Try COO conversion to CSR (bench speedup)
- [ ] Adapted thread/block for sparse Tensors and forces kernels 
- [ ] CUBLAS_COMPUTE_32F_FAST_TF32
- [ ] Interleaved excitations files
*/


