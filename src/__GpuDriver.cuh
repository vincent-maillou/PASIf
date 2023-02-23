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
  __GpuDriver(std::vector<std::vector<double>> excitationSet_,
              uint sampleRate_,
              uint numsteps_,
              bool dCompute_ = false,
              bool dSystem_  = false,
              bool dSolver_  = false);

  ~__GpuDriver();


  //            Forward system interface 
  void _setFwdB(std::vector<reel> values_,
                std::vector<uint> row_,
                std::vector<uint> col_,
                uint n_); 
  void _setFwdK(std::vector<reel> values_,
                std::vector<uint> row_,
                std::vector<uint> col_,
                uint n_);
  void _setFwdGamma(std::vector<uint> dimensions_,
                    std::vector<reel> values_,
                    std::vector<uint> indices_);
  void _setFwdLambda(std::vector<uint> dimensions_,
                     std::vector<reel> values_,
                     std::vector<uint> indices_);
  void _setFwdForcePattern(std::vector<reel> & forcePattern_);
  void _setFwdInitialConditions(std::vector<reel> & initialConditions_);
  
  void _allocateSystemOnDevice();


  //            Backward system interface 




  //            Compute options interface 
  int  _loadExcitationsSet(std::vector< std::vector<double> > excitationSet_,
                           uint sampleRate_);
  void _setInterpolationMatrix(std::vector<reel> & interpolationMatrix_,
                               uint interpolationWindowSize_);
  void _setModulationBuffer(std::vector<reel> & modulationBuffer_);

  void _displaySimuInfos();


  //            Solvers interface
  std::vector<reel> _getAmplitudes();
  std::vector<reel> _getTrajectory(uint saveSteps_ = 1);
  std::vector<reel> _getGradient(uint adjointSize_,
                                 uint save);

 private:
  // Initialization functions
  int  setCUDA(uint nStreams_);
  void setTimesteps();
  void resetStatesVectors();


  //            Solver functions
  void forwardRungeKutta(uint tStart_, 
                         uint tEnd_,
                         uint k,
                         uint saveSteps  = 1,
                         uint saveOffset = 0);

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

  void   parallelizeThroughExcitations();
  size_t getSystemMemFootprint();
  size_t getAdjointMemFootprint(); 

  // Memory cleaning functions
  void clearTrajectories();
  void clearInterpolationMatrix();
  void clearModulationBuffer();


  //            Simulation related data
  std::vector<reel> excitationSet; reel* d_ExcitationsSet;
  uint sampleRate;
  uint numberOfExcitations;
  uint lengthOfeachExcitation;

  uint numsteps; 
  uint totalNumsteps; // Take into acount the interpolated steps

  reel h; 
  reel h2; 
  reel h6;

  reel alpha; reel* d_alpha;
  reel beta1; reel* d_beta1;
  reel beta0; reel* d_beta0;

  bool dCompute;
  bool dSystem;
  bool dSolver;


  //            Compute system related data
  uint n_dofs;

  COOMatrix*   B;
  COOMatrix*   K;
  COOTensor3D* Gamma;
  COOTensor4D* Lambda;
  COOVector*   ForcePattern;

  reel* d_QinitCond; 
  
  reel* d_Q;  cusparseDnVecDescr_t d_Q_desc;
  reel* d_mi; cusparseDnVecDescr_t d_mi_desc;
  reel* d_m1; cusparseDnVecDescr_t d_m1_desc;
  reel* d_m2; cusparseDnVecDescr_t d_m2_desc;
  reel* d_m3; cusparseDnVecDescr_t d_m3_desc;
  reel* d_m4; cusparseDnVecDescr_t d_m4_desc;

  void setComputeSystem(problemType type_ = forward);


  //            Forward system related data
  uint n_dofs_fwd;

  COOMatrix*   fwd_B;
  COOMatrix*   fwd_K;
  COOTensor3D* fwd_Gamma;
  COOTensor4D* fwd_Lambda;
  COOVector*   fwd_ForcePattern;

  std::vector<reel> h_fwd_QinitCond; 
  reel*             d_fwd_QinitCond; 
  
  reel* d_fwd_Q;  cusparseDnVecDescr_t d_fwd_Q_desc;
  reel* d_fwd_mi; cusparseDnVecDescr_t d_fwd_mi_desc;
  reel* d_fwd_m1; cusparseDnVecDescr_t d_fwd_m1_desc;
  reel* d_fwd_m2; cusparseDnVecDescr_t d_fwd_m2_desc;
  reel* d_fwd_m3; cusparseDnVecDescr_t d_fwd_m3_desc;
  reel* d_fwd_m4; cusparseDnVecDescr_t d_fwd_m4_desc;

  void  allocateDeviceSystem();
  void  allocateDeviceSystemStatesVector();
  void  extendSystem();

  void  clearFwdB();
  void  clearFwdK();
  void  clearFwdGamma();
  void  clearFwdLambda();
  void  clearFwdForcePattern();
  void  clearFwdInitialConditions();
  void  clearSystemStatesVector();


  //            Adjoint system related data
  uint n_dofs_bwd;

  COOMatrix*   bwd_B;
  COOMatrix*   bwd_K;
  COOTensor3D* bwd_Gamma;
  COOTensor4D* bwd_Lambda;
  COOVector*   bwd_ForcePattern;

  std::vector<reel> h_bwd_QinitCond; 
  reel*             d_bwd_QinitCond; 
  
  reel* d_bwd_Q;  cusparseDnVecDescr_t d_bwd_Q_desc;
  reel* d_bwd_mi; cusparseDnVecDescr_t d_bwd_mi_desc;
  reel* d_bwd_m1; cusparseDnVecDescr_t d_bwd_m1_desc;
  reel* d_bwd_m2; cusparseDnVecDescr_t d_bwd_m2_desc;
  reel* d_bwd_m3; cusparseDnVecDescr_t d_bwd_m3_desc;
  reel* d_bwd_m4; cusparseDnVecDescr_t d_bwd_m4_desc;

  // void  allocateDeviceSystem();
  // void  allocateDeviceSystemStatesVector();
  void  extendAdjoint();

  // void  clearFwdB();
  // void  clearFwdK();
  // void  clearFwdGamma();
  // void  clearFwdLambda();
  // void  clearFwdForcePattern();
  // void  clearFwdInitialConditions();
  // void  clearSystemStatesVector();


  //            Interpolation related data
  uint interpolationNumberOfPoints; // = Height of the interpolation matrix
  uint interpolationWindowSize;     // = Width of the interpolation matrix
  std::vector<reel> interpolationMatrix;
  reel*             d_interpolationMatrix;


  //            Modulation related data
  uint modulationBufferSize;
  std::vector<reel> modulationBuffer;
  reel*             d_modulationBuffer;


  //            getTrajectory related data
  std::vector<reel> h_trajectories; reel* d_trajectories;

  uint numSetpoints;
  uint chunkSize;
  uint lastChunkSize;
  

  //            GPU Devices parameters
  uint nStreams;
  cudaStream_t *streams;

  uint parallelismThroughExcitations;
  uint numberOfSimulationToPerform;
  uint exceedingSimulations;

  int  deviceId;
  int  numberOfSMs;
  uint nBlocks;
  uint nThreadsPerBlock;

  cublasHandle_t   h_cublas;
  cusparseHandle_t h_cusparse;
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


