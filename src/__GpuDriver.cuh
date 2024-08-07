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
              bool dSolver_  = false,
              uint GPUId_ = 0);

  ~__GpuDriver();


  //            Forward system interface 
  void _setFwdK(std::array<uint, 2> n_,
                std::vector<reel>   values_,
                std::vector<uint>   indices_,
                std::vector<uint>   indptr_);
  void _setFwdGamma(std::array<uint, 3> n_,
                    std::vector<reel>   values_,
                    std::vector<uint>   indices_);
  void _setFwdLambda(std::array<uint, 4> n_,
                     std::vector<reel>   values_,
                     std::vector<uint>   indices_);
  void _setFwdPsi(std::array<uint, 5> n_,
                     std::vector<reel>   values_,
                     std::vector<uint>   indices_);
  void _setFwdForcePattern(std::vector<reel> & forcePattern_);
  void _setFwdInitialConditions(std::vector<reel> & initialConditions_);
  
  void _allocateSystemOnDevice();


  //            Backward system interface 
  void _setBwdK(std::array<uint, 2> n_,
                std::vector<reel>   values_,
                std::vector<uint>   indices_,
                std::vector<uint>   indptr_);
  void _setBwdGamma(std::array<uint, 3> n_,
                    std::vector<reel>   values_,
                    std::vector<uint>   indices_);
  void _setBwdLambda(std::array<uint, 4> n_,
                     std::vector<reel>   values_,
                     std::vector<uint>   indices_);
  void _setBwdPsi(std::array<uint, 5> n_,
                  std::vector<reel>   values_,
                  std::vector<uint>   indices_);
  void _setBwdForcePattern(std::vector<reel> & forcePattern_);
  void _setBwdInitialConditions(std::vector<reel> & initialConditions_);

  void _allocateAdjointOnDevice();


  //            Compute options interface 
  int  _loadExcitationsSet(std::vector< std::vector<double> > excitationSet_,
                           uint sampleRate_);
  void _setInterpolationMatrix(std::vector<reel> & interpolationMatrix_,
                               uint interpolationWindowSize_);
  void _setModulationBuffer(std::vector<reel> & modulationBuffer_);


  //            Solvers interface
  std::vector<reel> _getAmplitudes();
  std::vector<reel> _getTrajectory(uint saveSteps_ = 1);
  std::vector<reel> _getGradient(uint   save);


 private:
  // Initialization functions
  int  setCUDA(uint nStreams_);
  void setTimesteps();


  //            Solver functions
  void forwardRungeKutta(uint tStart_, 
                         uint tEnd_,
                         uint k,
                         uint saveSteps  = 1,
                         uint saveOffset = 0);

  void fwdStep();

  void backwardRungeKutta(uint  tStart_, 
                          uint  tEnd_,
                          uint  k,
                          uint  startSetpoint);

  void bwdStep();                        

  void derivatives(reel* pm, 
                   reel* pq,
                   reel* pq_fwd_state);

  void modterpolator(reel* Y,
                     int  offset,
                     bool  halfStep,
                     bool backward);


  //            GPU work distribution
  void   parallelizeThroughExcitations(bool bwd_setting);
  size_t getSystemMemFootprint();
  size_t getAdjointMemFootprint(); 


  //            Compute option cleaners
  void clearExcitationsSet();
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
  reel beta0; reel* d_beta0;

  bool dCompute;
  bool dSystem;
  bool dSolver;


  //            Compute system related data
  uint n_dofs;

  CSRMatrix*   K;
  COOTensor3D* Gamma;
  COOTensor4D* Lambda;
  COOTensor5D* Psi;
  COOVector*   ForcePattern;

  std::vector<reel>* h_QinitCond; 
  reel*              d_QinitCond; 
  
  reel* d_Q;
  reel* d_mi;
  reel* d_m1;
  reel* d_m2;
  reel* d_m3;
  reel* d_m4;

  void setComputeSystem(problemType type_);
  void resetStatesVectors();
  void displaySimuInfos(problemType type_);


  //            Forward system related data
  uint n_dofs_fwd;

  CSRMatrix*   fwd_K;
  COOTensor3D* fwd_Gamma;
  COOTensor4D* fwd_Lambda;
  COOTensor5D* fwd_Psi;

  COOVector*   fwd_ForcePattern;

  std::vector<reel> h_fwd_QinitCond; 
  reel*             d_fwd_QinitCond; 
  
  reel* d_fwd_Q;
  reel* d_fwd_mi;
  reel* d_fwd_m1;
  reel* d_fwd_m2;
  reel* d_fwd_m3;
  reel* d_fwd_m4;

  void  allocateDeviceSystem();
  void  allocateDeviceSystemStatesVector();
  void  extendSystem();

  void  clearFwdK();
  void  clearFwdGamma();
  void  clearFwdLambda();
  void  clearFwdPsi();
  void  clearFwdForcePattern();
  void  clearFwdInitialConditions();
  void  clearSystemStatesVector();


  //            Adjoint system related data
  uint n_dofs_bwd;

  CSRMatrix*   bwd_K;
  COOTensor3D* bwd_Gamma;
  COOTensor4D* bwd_Lambda;
  COOTensor5D* bwd_Psi;
  COOVector*   bwd_ForcePattern;

  std::vector<reel> h_bwd_QinitCond; 
  reel*             d_bwd_QinitCond; 
  
  reel* d_bwd_Q;
  reel* d_bwd_mi;
  reel* d_bwd_m1;
  reel* d_bwd_m2;
  reel* d_bwd_m3;
  reel* d_bwd_m4;

  void  allocateDeviceAdjointSystem();
  void  allocateDeviceAdjointStatesVector();
  void  extendAdjoint();

  void  clearBwdK();
  void  clearBwdGamma();
  void  clearBwdLambda();
  void  clearBwdPsi();
  void  clearBwdForcePattern();
  void  clearBwdInitialConditions();
  void  clearAdjointStatesVector();


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
  uint maxThreads;
  uint gridShapeY;

  cublasHandle_t   h_cublas;
  cusparseHandle_t h_cusparse;

  uint fwd_step; uint* d_step;
  cudaGraph_t fwd_graph;
  cudaGraphExec_t fwd_instance;
  bool fwd_graphs_created;

  bool bwd_graphs_created;
  uint bwd_setpoint; 
  uint* d_setpoint;
  cudaGraph_t bwd_graph;
  cudaGraphExec_t bwd_instance;

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



/**
 * @brief Issues
 * 
 */

/*
- [ ] cuda API crash when using more than 1.6M DOF
*/


