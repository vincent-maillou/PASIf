/**
 * @file __GpuDriver.cu
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-11-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "__GpuDriver.cuh"



/****************************************************
 *              Public functions                   *
 ****************************************************/
  /** __GpuDriver::__GpuDriver()
    * @brief Construct a new gpudriver::  gpudriver object
    * 
    * @param excitationSet_ 
    * @param sampleRate_ 
    */
    __GpuDriver::__GpuDriver(std::vector<std::vector<double>> excitationSet_, 
                             uint sampleRate_,
                             uint numsteps_,
                             bool dCompute_,
                             bool dSystem_,
                             bool dSolver_,
                             uint GPUId_) : 
        //            Simulation related data
        d_ExcitationsSet(nullptr),
        numsteps(numsteps_),
        totalNumsteps(numsteps),

        alpha(1.0), d_alpha(nullptr),
        beta0(0.0), d_beta0(nullptr),

        dCompute(dCompute_),
        dSystem(dSystem_),
        dSolver(dSolver_),


        //            Compute system related data
        n_dofs(0),

        K(nullptr), 
        Gamma(nullptr), 
        Lambda(nullptr), 
        Psi(nullptr),
        ForcePattern(nullptr),

        h_QinitCond(nullptr),
        d_QinitCond(nullptr),

        d_Q(nullptr),
        d_mi(nullptr),
        d_m1(nullptr),
        d_m2(nullptr),
        d_m3(nullptr),
        d_m4(nullptr),


        //            Forward system related data
        n_dofs_fwd(0),

        fwd_K(nullptr),
        fwd_Gamma(nullptr),
        fwd_Lambda(nullptr),
        fwd_Psi(nullptr),
        fwd_ForcePattern(nullptr),

        d_fwd_QinitCond(nullptr),

        d_fwd_Q(nullptr),
        d_fwd_mi(nullptr),
        d_fwd_m1(nullptr),
        d_fwd_m2(nullptr),
        d_fwd_m3(nullptr),
        d_fwd_m4(nullptr),


        //            Adjoint system related data
        n_dofs_bwd(0),

        bwd_K(nullptr),
        bwd_Gamma(nullptr),
        bwd_Lambda(nullptr),
        bwd_Psi(nullptr),
        bwd_ForcePattern(nullptr),

        d_bwd_QinitCond(nullptr),

        d_bwd_Q(nullptr),
        d_bwd_mi(nullptr),
        d_bwd_m1(nullptr),
        d_bwd_m2(nullptr),
        d_bwd_m3(nullptr),
        d_bwd_m4(nullptr),


        //            Interpolation related data
        interpolationNumberOfPoints(0),
        interpolationWindowSize(0),
        d_interpolationMatrix(nullptr),


        //            Modulation related data
        modulationBufferSize(0),
        d_modulationBuffer(nullptr),


        //            getTrajectory related data
        d_trajectories(nullptr),
        numSetpoints(0),
        chunkSize(0),
        lastChunkSize(0),

        //            GPU Devices parameters
        nStreams(1),
        streams(nullptr),

        parallelismThroughExcitations(1),
        numberOfSimulationToPerform(0),
        exceedingSimulations(0),

        deviceId(GPUId_),
        numberOfSMs(0),
        nBlocks(0),
        nThreadsPerBlock(0),
        maxThreads(512),

        h_cublas(nullptr),
        h_cusparse(nullptr)
        {
    
      setCUDA(nStreams);

      // Put on the device alpha and beta values for the cuSPARSE API
      CHECK_CUDA( cudaMalloc((void**)&d_alpha, sizeof(reel)) )
      CHECK_CUDA( cudaMalloc((void**)&d_beta0, sizeof(reel)) )
      CHECK_CUDA( cudaMemcpy(d_alpha, &alpha,  sizeof(reel), cudaMemcpyHostToDevice) )
      CHECK_CUDA( cudaMemcpy(d_beta0, &beta0,  sizeof(reel), cudaMemcpyHostToDevice) )

      _loadExcitationsSet(excitationSet_,
                          sampleRate_);

    }

  __GpuDriver::~__GpuDriver(){

    //            Compute option cleaners
    clearExcitationsSet();
    clearTrajectories();
    clearInterpolationMatrix();
    clearModulationBuffer();


    //            Forward system destructor
    clearFwdK();
    clearFwdGamma();
    clearFwdLambda();
    clearFwdPsi();
    clearFwdForcePattern();
    clearFwdInitialConditions();
    clearSystemStatesVector();


    //            Backward system destructor
    clearBwdK();
    clearBwdGamma();
    clearBwdLambda();
    clearBwdPsi();
    clearBwdForcePattern();
    clearBwdInitialConditions();
    clearAdjointStatesVector();
    

    if(streams != nullptr){
      for(uint i = 0; i < nStreams; i++){
        CHECK_CUDA( cudaStreamDestroy(streams[i]) );
      }
      delete[] streams;
      streams = nullptr;
    }
  }



/*                    Forward system interface                    */

  void __GpuDriver::_setFwdK(std::array<uint, 2> n_,
                             std::vector<reel>   values_,
                             std::vector<uint>   indices_,
                             std::vector<uint>   indptr_){
    clearFwdK();

    fwd_K = new CSRMatrix(n_,
                          values_,
                          indices_,
                          indptr_);
  }

  void __GpuDriver::_setFwdGamma(std::array<uint, 3> n_,
                                 std::vector<reel>   values_,
                                 std::vector<uint>   indices_){
    clearFwdGamma();

    fwd_Gamma = new COOTensor3D(n_,
                                values_,
                                indices_);
  }

  void __GpuDriver::_setFwdLambda(std::array<uint, 4> n_,
                                  std::vector<reel>   values_,
                                  std::vector<uint>   indices_){
    clearFwdLambda();

    fwd_Lambda = new COOTensor4D(n_,
                                 values_,
                                 indices_);
  }

  void __GpuDriver::_setFwdPsi(std::array<uint, 5> n_,
                                  std::vector<reel>   values_,
                                  std::vector<uint>   indices_){
    clearFwdPsi();

    fwd_Psi = new COOTensor5D(n_,
                                 values_,
                                 indices_);
  }


  void __GpuDriver::_setFwdForcePattern(std::vector<reel> & forcePattern_){
    clearFwdForcePattern();

    fwd_ForcePattern = new COOVector(forcePattern_);
  }

  void __GpuDriver::_setFwdInitialConditions(std::vector<reel> & initialConditions_){
    clearFwdInitialConditions();

    // Set the number of degrees of freedom of the forward problem
    n_dofs_fwd      = initialConditions_.size();
    h_fwd_QinitCond = initialConditions_;
  }

  void __GpuDriver::_allocateSystemOnDevice(){
    // Extend the systems to exploit the GPU memory
    parallelizeThroughExcitations(0);

    // Allocate the rk4 vectors on the GPU
    allocateDeviceSystemStatesVector();

    // Allocate the matrix on the GPU
    allocateDeviceSystem();
  }



/*                    Backward system interface                    */


  void __GpuDriver::_setBwdK(std::array<uint, 2> n_,
                             std::vector<reel>   values_,
                             std::vector<uint>   indices_,
                             std::vector<uint>   indptr_){
    clearBwdK();

    bwd_K = new CSRMatrix(n_,
                          values_,
                          indices_,
                          indptr_);
  }

  void __GpuDriver::_setBwdGamma(std::array<uint, 3> n_,
                                 std::vector<reel>   values_,
                                 std::vector<uint>   indices_){
    clearBwdGamma();

    bwd_Gamma = new COOTensor3D(n_,
                                values_,
                                indices_);
  }

  void __GpuDriver::_setBwdLambda(std::array<uint, 4> n_,
                                  std::vector<reel>   values_,
                                  std::vector<uint>   indices_){
    clearBwdLambda();

    bwd_Lambda = new COOTensor4D(n_,
                                 values_,
                                 indices_);
  }

  void __GpuDriver::_setBwdPsi(std::array<uint, 5> n_,
                               std::vector<reel>   values_,
                               std::vector<uint>   indices_){
    clearBwdPsi();

    bwd_Psi = new COOTensor5D(n_,
                              values_,
                              indices_);
  }

  void __GpuDriver::_setBwdForcePattern(std::vector<reel> & forcePattern_){
    clearBwdForcePattern();

    bwd_ForcePattern = new COOVector(forcePattern_);
  }

  void __GpuDriver::_setBwdInitialConditions(std::vector<reel> & initialConditions_){
    clearBwdInitialConditions();

    // Set the number of degrees of freedom of the forward problem
    n_dofs_bwd      = initialConditions_.size();
    h_bwd_QinitCond = initialConditions_;
  }

  void __GpuDriver::_allocateAdjointOnDevice(){
    // Extend the systems to exploit the GPU memory
    parallelizeThroughExcitations(1);

    // Allocate the rk4 vectors on the GPU
    allocateDeviceAdjointStatesVector();

    // Allocate the matrix on the GPU
    allocateDeviceAdjointSystem();
  }



/*                    Compute options interface                    */
 
  int __GpuDriver::_loadExcitationsSet(std::vector< std::vector<double> > excitationSet_, 
                                        uint sampleRate_){

    sampleRate = sampleRate_;
    setTimesteps();


    // Check if the ExcitationsSet is already loaded
    excitationSet.clear();
    if(d_ExcitationsSet != nullptr){
      CHECK_CUDA( cudaFree(d_ExcitationsSet) )
      d_ExcitationsSet = nullptr;
    }

    // Check the size of all the excitation vectors
    for(auto &excitation : excitationSet_){
      if(excitation.size() != excitationSet_[0].size()){
        std::cout << "[Error] __GpuDriver: Excitations vectors are not of the same size" << std::endl;
        return 1;
      }
    }

    numberOfExcitations    = excitationSet_.size();
    lengthOfeachExcitation = excitationSet_[0].size();
    // Parse the input excitationSet_ to a 1D array
    for(auto &excitation : excitationSet_){
      for(auto &sample : excitation){
        excitationSet.push_back((reel)sample);
      }
    }

    // Allocate memory on the GPU
    CHECK_CUDA( cudaMalloc((void**)&d_ExcitationsSet, excitationSet.size()*sizeof(reel)) )
    // Copy the ExcitationsSet to the GPU
    CHECK_CUDA( cudaMemcpy(d_ExcitationsSet, excitationSet.data(), excitationSet.size()*sizeof(reel), cudaMemcpyHostToDevice) )
    std::cout << "[Info] __GpuDriver: Loaded " << numberOfExcitations << " excitations of length " << lengthOfeachExcitation << " each." << std::endl;
  
    return 0;
  }

  void __GpuDriver::_setInterpolationMatrix(std::vector<reel> & interpolationMatrix_,
                                            uint interpolationWindowSize_){
    interpolationMatrix = interpolationMatrix_;
    interpolationWindowSize  = interpolationWindowSize_;
    interpolationNumberOfPoints = interpolationMatrix_.size()/interpolationWindowSize_;

    setTimesteps();
    totalNumsteps = numsteps*(1+interpolationNumberOfPoints);

    // Allocate the interpolation matrix on the GPU
    CHECK_CUDA( cudaMalloc((void**)&d_interpolationMatrix, interpolationMatrix.size()*sizeof(reel)) )
    // Copy the interpolation matrix to the GPU
    CHECK_CUDA( cudaMemcpy(d_interpolationMatrix, interpolationMatrix.data(), interpolationMatrix.size()*sizeof(reel), cudaMemcpyHostToDevice) )
    
    std::cout << "[Info] __GpuDriver: Loaded interpolation matrix, " << interpolationNumberOfPoints << " points, windows length " << interpolationWindowSize << std::endl;
  }

  void __GpuDriver::_setModulationBuffer(std::vector<reel> & modulationBuffer_){
    modulationBuffer = modulationBuffer_;
    modulationBufferSize = modulationBuffer_.size();

    // Allocate the modulation buffer on the GPU
    CHECK_CUDA( cudaMalloc((void**)&d_modulationBuffer, modulationBufferSize*sizeof(reel)) )
    // Copy the modulation buffer to the GPU
    CHECK_CUDA( cudaMemcpy(d_modulationBuffer, modulationBuffer.data(), modulationBufferSize*sizeof(reel), cudaMemcpyHostToDevice) )
  
    std::cout << "[Info] __GpuDriver: Loaded modulation buffer of " << modulationBufferSize << " points." << std::endl;
  }



/*                    Solvers interface                    */

  std::vector<reel> __GpuDriver::_getAmplitudes(){
    setComputeSystem(forward);
    displaySimuInfos(forward);
    std::vector<reel> resultsQ;
    resultsQ.resize(n_dofs*numberOfSimulationToPerform);



    auto begin = std::chrono::high_resolution_clock::now();
    // Perform the simulations
    for(size_t k(0); k < numberOfSimulationToPerform; ++k){

      forwardRungeKutta(0, numsteps, k);

      // std::cout << "k = " << k << std::endl;
      // std::cout << "n_doofs = " << n_dofs << std::endl;
      // std::cout << "numsteps = " << numsteps << std::endl;
      // std::cout << "d_Q = " << d_Q << std::endl;
      // std::cout << "d_fwd_Q = " << d_fwd_Q << std::endl;
      // std::cout << "sizeof(reel) = " << sizeof(reel) << std::endl;
      // std::cout << "resultsQ.max_size() = " << resultsQ.max_size() << std::endl;
      // std::cout << "numberOfSimulationToPerform = " << numberOfSimulationToPerform << std::endl;

      std::cout << "resultsQ.size() = " << resultsQ.size() << std::endl;

      // Copy the results of the performed simulation from the GPU to the CPU
      CHECK_CUDA( cudaMemcpy(resultsQ.data()+k*n_dofs, d_Q, n_dofs*sizeof(reel), cudaMemcpyDeviceToHost) )
      CHECK_CUDA( cudaDeviceSynchronize() )

      resetStatesVectors();

    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-begin;
    std::cout << "CUDA getAmplitudes() execution time: " << elapsed_seconds.count() << "s" << std::endl;

    // Cut the results vector to the correct size
    if(exceedingSimulations != 0){
      resultsQ.resize(n_dofs*(numberOfSimulationToPerform-1)+exceedingSimulations);
    }

    return std::vector<reel>{resultsQ};
  }

  std::vector<reel> __GpuDriver::_getTrajectory(uint saveSteps_){
    setComputeSystem(forward);
    displaySimuInfos(forward);

    // Reserve the size of the trajectories vector on the CPU
    h_trajectories.clear();
    size_t reservedTrajSize = (n_dofs*numberOfSimulationToPerform*(interpolationNumberOfPoints+1)*numsteps)/saveSteps_;
    h_trajectories.resize(reservedTrajSize);

    // Allocate the memory on the GPU
    CHECK_CUDA( cudaMalloc((void**)&d_trajectories, h_trajectories.size()*sizeof(reel)) )
    CHECK_CUDA( cudaMemset(d_trajectories, 0, h_trajectories.size()*sizeof(reel)) )



    auto begin = std::chrono::high_resolution_clock::now();
    // Perform the simulations
    for(size_t k(0); k<numberOfSimulationToPerform; ++k){

      forwardRungeKutta(0, numsteps, k, saveSteps_);

      resetStatesVectors();

    }
    auto end = std::chrono::high_resolution_clock::now();



    std::chrono::duration<double> elapsed_seconds = end-begin;
    std::cout << "CUDA getTrajectory() execution time: " << elapsed_seconds.count() << "s" << std::endl;


    // Copy the results of the performed simulation from the GPU to the CPU
    CHECK_CUDA( cudaMemcpy(h_trajectories.data(), d_trajectories, h_trajectories.size()*sizeof(reel), cudaMemcpyDeviceToHost) )
    
    clearTrajectories();

    return h_trajectories;
  }

  std::vector<reel> __GpuDriver::_getGradient(uint save){
    displaySimuInfos(forward);
    displaySimuInfos(backward);

    numSetpoints  = std::sqrt(numsteps);
    chunkSize     = std::ceil((reel)numsteps/(reel)numSetpoints);
    lastChunkSize = chunkSize - (numSetpoints*chunkSize)%numsteps;


    // 1. Compute the setPoints (chunks first step)
    size_t reservedTrajSize = (numSetpoints+chunkSize-2)*(interpolationNumberOfPoints+1)*n_dofs_fwd;

    h_trajectories.clear();
    h_trajectories.resize(reservedTrajSize);

    // Chunk and setpoints are stored inside of the trajectories vector
    CHECK_CUDA( cudaMalloc((void**)&d_trajectories, reservedTrajSize*sizeof(reel)) )
    CHECK_CUDA( cudaMemset(d_trajectories, 0, reservedTrajSize*sizeof(reel)) )

    // Compute the setpoints
    setComputeSystem(forward);
    forwardRungeKutta(0, numsteps, 0, chunkSize);

    // DEBUG 
    std::cout << "numsteps: " << numsteps << std::endl;
    std::cout << "totalNumsteps: " << totalNumsteps << std::endl;
    std::cout << "numSetpoints: " << numSetpoints << std::endl;
    std::cout << "chunkSize: " << chunkSize << std::endl;
    std::cout << "lastChunkSize: " << lastChunkSize << std::endl;
    std::cout << "reservedTrajSize: " << reservedTrajSize << std::endl << std::endl;

    reel* h_fwd_setpoints = new reel[n_dofs_fwd];
    reel* h_bwd_state     = new reel[n_dofs_bwd];
    std::cout << std::setprecision(4);

    size_t startStep = 0;
    size_t endStep   = 0;
    size_t startBwdSetpoint = 0;

    for(size_t setpoint(numSetpoints-1); setpoint>0; --setpoint){
      startStep = setpoint*chunkSize;

      if(setpoint == numSetpoints-1){
        // Avoid to overflow the allocated traj memory
        endStep          = numsteps;
        startBwdSetpoint = setpoint + (lastChunkSize-1)*(interpolationNumberOfPoints+1)+interpolationNumberOfPoints;
      }
      else{
        endStep          = startStep + chunkSize;
        startBwdSetpoint = setpoint + (chunkSize-1)*(interpolationNumberOfPoints+1)+interpolationNumberOfPoints;
      }


      // 2. Compute the current chunk
      CHECK_CUBLAS( cublasScopy(h_cublas,
                                n_dofs_fwd, 
                                d_trajectories + (setpoint)*n_dofs_fwd, 
                                1, 
                                d_fwd_Q, 
                                1) )
      

      setComputeSystem(forward);
      forwardRungeKutta(startStep, endStep, 0, 1, setpoint);

      // 3. Compute backward the gradient on the current chunk
      setComputeSystem(backward);
      backwardRungeKutta(endStep, startStep, 0, startBwdSetpoint);

      if(setpoint == save){
        break;
      }
    }

    delete[] h_fwd_setpoints;
    delete[] h_bwd_state;

    clearTrajectories();

    std::vector<reel> h_gradient;
    h_gradient.resize(n_dofs_bwd);

    CHECK_CUDA( cudaMemcpy(h_gradient.data(), d_bwd_Q, n_dofs_bwd*sizeof(reel), cudaMemcpyDeviceToHost) )

    return h_gradient;
  }
  


/****************************************************
 *              Private functions                   *
 ****************************************************/

  int __GpuDriver::setCUDA(uint nStreams_){
    nStreams = nStreams_;

    /* // Query the number of available devices
    int nDevices;
    CHECK_CUDA( cudaGetDeviceCount(&nDevices) ) */

    // Query the device parameters
    CHECK_CUDA( cudaSetDevice(deviceId))
    CHECK_CUDA( cudaGetDevice(&deviceId) )
    CHECK_CUDA( cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId) )

    nThreadsPerBlock = 128;

    nBlocks          = numberOfSMs * 32;

    // Spawn the streams
    streams = new cudaStream_t[nStreams];
    for(uint i = 0; i < nStreams; i++){
      CHECK_CUDA( cudaStreamCreate(&streams[i]) )
    }

    // Create the cuBLAS handle
    CHECK_CUBLAS( cublasCreate(&h_cublas) )
    CHECK_CUBLAS( cublasSetPointerMode(h_cublas, CUBLAS_POINTER_MODE_DEVICE) )

    CHECK_CUBLAS( cublasSetStream(h_cublas, streams[0]) )

    // Create the cuSPARSE handle
    CHECK_CUSPARSE( cusparseCreate(&h_cusparse) )
    CHECK_CUSPARSE( cusparseSetPointerMode(h_cusparse, CUSPARSE_POINTER_MODE_DEVICE) )

    CHECK_CUSPARSE( cusparseSetStream(h_cusparse, streams[0]) )

    return 0;
  }

  void __GpuDriver::setTimesteps(){
    if(!interpolationMatrix.empty()){
      h  = 1.0/(sampleRate*(interpolationNumberOfPoints+1));
    }
    else{
      h  = 1.0/sampleRate;
    }

    h2 = h/2.0;
    h6 = h/6.0;
  }



/*                    Solver private methods                    */

  void __GpuDriver::forwardRungeKutta(uint tStart_, 
                                      uint tEnd_,
                                      uint k,
                                      uint saveSteps,
                                      uint saveOffset){
                          
    uint   m(0); // Modulation index
    size_t trajSaveIndex(0);

    // Performe the rk4 steps
    for(uint t(tStart_); t<tEnd_ ; ++t){
      // Always performe one step without interpolation, and then performe the
      // interpolation steps

      for(uint i(0); i<=interpolationNumberOfPoints; ++i){
        fwdStep(k, t, i, m);

        if(d_trajectories != nullptr && (t%saveSteps==0) && (i==0 || saveSteps==1)){
          //only save non interpolated steps  
          CHECK_CUBLAS( cublasScopy(h_cublas,
                                    n_dofs, 
                                    d_Q, 
                                    1, 
                                    d_trajectories + (trajSaveIndex+saveOffset)*n_dofs, 
                                    1) )
          ++trajSaveIndex;
        }
        
        ++m;
        if(m == modulationBufferSize){
          m = 0;
        }
      }
    }                       
  }

  void __GpuDriver::fwdStep(uint k, 
                            uint t,
                            uint i,
                            uint m){


    // Compute the derivatives
    derivatives(d_m1, d_Q, nullptr, k, t, i, m);

      updateSlope<<<Gamma->ntimes, maxThreads, 0, streams[0]>>>(d_mi, d_Q, d_m1, h2, n_dofs);

    derivatives(d_m2, d_mi, nullptr, k, t, i+1, m);

      updateSlope<<<Gamma->ntimes, maxThreads, 0, streams[0]>>>(d_mi, d_Q, d_m2, h2, n_dofs);

    derivatives(d_m3, d_mi, nullptr, k, t, i+1, m);

      updateSlope<<<Gamma->ntimes, maxThreads, 0, streams[0]>>>(d_mi, d_Q, d_m3, h, n_dofs);

    derivatives(d_m4, d_mi, nullptr, k, t, i+2, m);

    // Compute next state vector Q
    integrate<<<Gamma->ntimes, maxThreads, 0, streams[0]>>>(d_Q, d_m1, d_m2, d_m3, d_m4, h6, n_dofs);
  }

  void __GpuDriver::backwardRungeKutta(uint  tStart_, 
                                       uint  tEnd_,
                                       uint  k,
                                       uint  startSetpoint){

    uint   m(0); // Modulation index
    uint   currentSetpoint(startSetpoint);
    reel* h_bwd_state     = new reel[n_dofs_bwd];
    reel* h_fwd_setpoints = new reel[n_dofs_fwd];

    // Performe the rk4 steps
    for(uint t(tStart_-1); t>=tEnd_ ; --t){

      for(int i(interpolationNumberOfPoints); i>=0; --i){
      
          // CHECK_CUDA( cudaMemcpy(h_fwd_setpoints, 
          //                        d_trajectories + currentSetpoint*n_dofs_fwd,
          //                        n_dofs_fwd*sizeof(reel),
          //                        cudaMemcpyDeviceToHost) );
          // CHECK_CUDA( cudaMemcpy(h_bwd_state, d_Q, n_dofs_bwd*sizeof(reel), cudaMemcpyDeviceToHost) );
          // std::cout << "in backward RK step: " << t << " interpolation: "<<i<<" cSetP: " << currentSetpoint << " at "<< (size_t)d_trajectories+currentSetpoint*n_dofs_fwd<<" / ";

          // std::cout << " h_fwd_setpoints: ";
          // for(size_t j(0); j<n_dofs_fwd; ++j){
          //   std::cout << h_fwd_setpoints[j] << " ";
          // }

          // std::cout << "   h_bwd_state: ";
          // for(size_t j(0); j<n_dofs_bwd; ++j){
          //   std::cout << h_bwd_state[j] << " ";
          // }
          // std::cout << std::endl;
      
        bwdStep(k, t, i, m, currentSetpoint);
        // Point to the previous state vector stored in the trajectory buffer
        --currentSetpoint;
      }
      if(t==0){break;}
    }  

    delete[] h_fwd_setpoints;
    delete[] h_bwd_state;

    // Save the computed adjoint state in what will be the initial condition for the next backward run
    CHECK_CUDA( cudaMemcpy(d_QinitCond, d_Q, n_dofs*sizeof(reel), cudaMemcpyDeviceToDevice) )
  }

  void __GpuDriver::bwdStep(uint k, 
                            uint t,
                            uint i,
                            uint m,
                            uint currentSetpoint){

    // Compute the derivatives
    derivatives(d_m1, d_Q, d_trajectories+currentSetpoint*n_dofs_fwd, k, t, i, m);

      updateSlope<<<Gamma->ntimes, maxThreads, 0, streams[0]>>>(d_mi, d_Q, d_m1, h2, n_dofs);

    derivatives(d_m2, d_mi, d_trajectories+currentSetpoint*n_dofs_fwd, k, t, i-1, m);

      updateSlope<<<Gamma->ntimes, maxThreads, 0, streams[0]>>>(d_mi, d_Q, d_m2, h2, n_dofs);

    derivatives(d_m3, d_mi, d_trajectories+currentSetpoint*n_dofs_fwd, k, t, i-1, m);

      updateSlope<<<Gamma->ntimes, maxThreads, 0, streams[0]>>>(d_mi, d_Q, d_m3, h, n_dofs);

    derivatives(d_m4, d_mi, d_trajectories+currentSetpoint*n_dofs_fwd, k, t, i-2, m);

    // Compute next state vector Q
    integrate<<<nBlocks, maxThreads, 0, streams[0]>>>(d_Q, d_m1, d_m2, d_m3, d_m4, h6, n_dofs);

  }

  inline void __GpuDriver::derivatives(reel* pm, 
                                       reel* pq,
                                       reel* pq_fwd_state, 
                                       uint k, 
                                       uint t,
                                       uint i,
                                       uint m){

    if(pq_fwd_state == nullptr){
      pq_fwd_state = pq;
    }
    
    // k = K.d_mi + Gamma.d_mi² + Lambda.d_mi³ + ForcePattern.d_ExcitationsSet
    // k += K.d_mi
    CHECK_CUSPARSE(cusparseDnMatSetValues(K->denseMat_desc, pq))
    CHECK_CUSPARSE(cusparseDnMatSetValues(K->resMat_desc, pm))
    CHECK_CUSPARSE(cusparseSpMM(h_cusparse, 
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                d_alpha, 
                                K->sparseMat_desc,
                                K->denseMat_desc,
                                d_beta0,
                                K->resMat_desc,
                                CUDA_R_32F, 
                                CUSPARSE_SPMM_CSR_ALG1, 
                                K->d_buffer));
    
    uint nThreads = min(Lambda->nzz+Gamma->nzz+Psi->nzz, maxThreads);
    //Each excitation is one block. We allocate one thread per non-linear element, with a limit of 512
    //Then each thread is made for one file and one (or more) non linear element
    SpTdV<<<Gamma->ntimes, nThreads, 0, streams[0]>>>(Gamma->d_val,
                                                        Gamma->d_slice,
                                                        Gamma->d_row, 
                                                        Gamma->d_col,
                                                        Gamma->nzz,          Lambda->d_val,
                                                        Lambda->d_hyperslice,
                                                        Lambda->d_slice, 
                                                        Lambda->d_row, 
                                                        Lambda->d_col,
                                                        Lambda->nzz,
                                                        Psi->d_val,
                                                        Psi->d_hyperhyperslice,
                                                        Psi->d_hyperslice,
                                                        Psi->d_slice,
                                                        Psi->d_row,
                                                        Psi->d_col,
                                                        Psi->nzz,
                                                        Gamma->ntimes,
                                                        Gamma->n[0],
                                                        Gamma->n[2],
                                                        pq,
                                                        pq_fwd_state,
                                                        pm);

    // Conditional release of the excitation in the case of a simulation longer 
    // than the excitation length
    if(t < lengthOfeachExcitation && (pq == pq_fwd_state /* WOP: Just to ensure the fwd only for now*/ )){
      // k += ForcePattern.d_ExcitationsSet
      modterpolator(pm, k, t, i, m);
    }
  } 

  inline void __GpuDriver::modterpolator(reel* Y,
                                         uint  k,
                                         uint  t,
                                         uint  i,
                                         uint  m){

    // "currentSimulation" refers to the simulation number in the case of
    // wich multiple simulation are needed to compute the system against all
    // of the excitation file
    uint currentSimulation = k/parallelismThroughExcitations;
    uint systemStride      = n_dofs/parallelismThroughExcitations;
    uint adjustedTime          = t;
    uint adjustedInterpolation = i;
    uint adjustedModulation    = m;

    uint useCase = 0;

    if(interpolationNumberOfPoints == 0){
      adjustedTime         += i;
      adjustedModulation   += i;
      adjustedInterpolation = 0;

      useCase = 0;
    }
    else{
      if(i > interpolationNumberOfPoints){
        adjustedTime          += 1;
        adjustedModulation    += 1;
        adjustedInterpolation -= (interpolationNumberOfPoints+1);
      }
      
      if(adjustedInterpolation == 0){
        useCase = 0;
      }
      else{
        useCase = 1;
        adjustedModulation    += adjustedInterpolation;
      }
    }

    if(adjustedModulation >= modulationBufferSize){
      adjustedModulation -= modulationBufferSize;
    }

    switch(useCase){
      case 0: // Just apply the force
        applyForces<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>
                                              (ForcePattern->d_val, 
                                              ForcePattern->d_indice, 
                                              ForcePattern->nzz, 
                                              d_ExcitationsSet,
                                              lengthOfeachExcitation, 
                                              currentSimulation,
                                              systemStride,
                                              Y, 
                                              adjustedTime,
                                              d_modulationBuffer,
                                              adjustedModulation);
        break;
      case 1: // Interpolate the force
        interpolateForces<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>
                                                    (ForcePattern->d_val, 
                                                    ForcePattern->d_indice, 
                                                    ForcePattern->nzz, 
                                                    d_ExcitationsSet,
                                                    lengthOfeachExcitation, 
                                                    currentSimulation,
                                                    systemStride,
                                                    Y, 
                                                    adjustedTime,
                                                    d_interpolationMatrix,
                                                    interpolationWindowSize,
                                                    adjustedInterpolation,
                                                    d_modulationBuffer,
                                                    adjustedModulation);
        break;
    }
  }



/*                    GPU work distribution                    */

  void __GpuDriver::parallelizeThroughExcitations(bool bwd_setting){
    size_t freeGpuSpace(0); 
    size_t totalGpuSpace(0);
    CHECK_CUDA( cudaMemGetInfo(&freeGpuSpace, &totalGpuSpace) )

    // std::cout << "free space = " << freeGpuSpace << " total space = " << totalGpuSpace << std::endl;

    size_t memFootprint(0);
    memFootprint  = getSystemMemFootprint();
    memFootprint += getAdjointMemFootprint();

    // std::cout << "memFootprint = " << memFootprint << std::endl;

    //TODO: Look at the new memory footprint of the system
    //Do multi GPU parallelization here?

    parallelismThroughExcitations = (0.8*freeGpuSpace) / memFootprint;
    if(parallelismThroughExcitations >= 1){
      parallelismThroughExcitations = numberOfExcitations;
      numberOfSimulationToPerform   = 1;
      exceedingSimulations          = 0;
    }
    else{
      numberOfSimulationToPerform = numberOfExcitations / parallelismThroughExcitations;
      exceedingSimulations        = numberOfExcitations % parallelismThroughExcitations;
      if(exceedingSimulations != 0){
        numberOfSimulationToPerform++;
      }
    }

    if(n_dofs_fwd > 0 && !bwd_setting){
      extendSystem();
    }
    if(n_dofs_bwd > 0 && bwd_setting){
      extendAdjoint();
    }

    /* memFootprint  = getSystemMemFootprint(); // LM
    memFootprint += getAdjointMemFootprint();

    std::cout << "memFootprint = " << memFootprint << std::endl; */

  }

  size_t __GpuDriver::getSystemMemFootprint(){
    if(n_dofs_fwd == 0){
      return 0;
    }

    size_t memFootprint(0);

    memFootprint += fwd_K->memFootprint();
    memFootprint += fwd_Gamma->memFootprint();
    memFootprint += fwd_Lambda->memFootprint();
    memFootprint += fwd_ForcePattern->memFootprint();
    memFootprint += sizeof(reel)*h_fwd_QinitCond.size();
    // Memory footprint of the RK4 states vectors
    memFootprint += sizeof(reel)*6*n_dofs_fwd;

    return memFootprint;
  }

  size_t __GpuDriver::getAdjointMemFootprint(){
    if(n_dofs_bwd == 0){
      return 0;
    }

    size_t memFootprint(0);

    memFootprint += bwd_K->memFootprint();
    memFootprint += bwd_Gamma->memFootprint();
    memFootprint += bwd_Lambda->memFootprint();
    memFootprint += bwd_ForcePattern->memFootprint();
    memFootprint += sizeof(reel)*h_bwd_QinitCond.size();
    // Memory footprint of the RK4 states vectors
    memFootprint += sizeof(reel)*6*n_dofs_bwd;

    return memFootprint;
  }

  

/*                    Compute option cleaners                    */

  void __GpuDriver::clearExcitationsSet(){
    if(d_ExcitationsSet != nullptr){
      CHECK_CUDA( cudaFree(d_ExcitationsSet) );
      d_ExcitationsSet = nullptr;
    }
  }

  void __GpuDriver::clearTrajectories(){
    if(d_trajectories != nullptr){
      CHECK_CUDA( cudaFree(d_trajectories) )
      d_trajectories = nullptr;
    }
  }

  void __GpuDriver::clearInterpolationMatrix(){
    if(interpolationMatrix.size() != 0){
      interpolationMatrix.clear();
    }
    if(d_interpolationMatrix != nullptr){
      CHECK_CUDA( cudaFree(d_interpolationMatrix) )
      d_interpolationMatrix = nullptr;
    }
  }

  void __GpuDriver::clearModulationBuffer(){
    if(modulationBuffer.size() != 0){
      modulationBuffer.clear();
    }
    if(d_modulationBuffer != nullptr){
      CHECK_CUDA( cudaFree(d_modulationBuffer) )
      d_modulationBuffer = nullptr;
    }
  }



/*                    Compute system private methods                    */

  void __GpuDriver::setComputeSystem(problemType type_){
    if(type_ == forward){
      n_dofs       = n_dofs_fwd;

      K            = fwd_K;
      Gamma        = fwd_Gamma;
      Lambda       = fwd_Lambda;
      Psi          = fwd_Psi;
      ForcePattern = fwd_ForcePattern;

      h_QinitCond  = &h_fwd_QinitCond;
      d_QinitCond  = d_fwd_QinitCond;

      d_Q  = d_fwd_Q;
      d_mi = d_fwd_mi;
      d_m1 = d_fwd_m1;
      d_m2 = d_fwd_m2;
      d_m3 = d_fwd_m3;
      d_m4 = d_fwd_m4;

      // Set the initial conditions to the states vectors
      // resetStatesVectors();
    }
    else if(type_ == backward){
      n_dofs       = n_dofs_bwd;

      K            = bwd_K;
      Gamma        = bwd_Gamma;
      Lambda       = bwd_Lambda;
      Psi          = bwd_Psi;

      ForcePattern = bwd_ForcePattern;

      h_QinitCond  = &h_bwd_QinitCond;
      d_QinitCond  = d_bwd_QinitCond;

      d_Q  = d_bwd_Q;
      d_mi = d_bwd_mi;
      d_m1 = d_bwd_m1;
      d_m2 = d_bwd_m2;
      d_m3 = d_bwd_m3;
      d_m4 = d_bwd_m4;

      // Set the initial conditions to the states vectors
      resetStatesVectors();
    }
  }

  void __GpuDriver::resetStatesVectors(){
    // Reset Q1 and Q2 to initials conditions
    CHECK_CUDA( cudaMemcpy(d_Q, d_QinitCond, n_dofs*sizeof(reel), cudaMemcpyDeviceToDevice) )

    // Reset all of the other vectors to 0
    CHECK_CUDA( cudaMemset(d_mi, 0, n_dofs*sizeof(reel)) )

    CHECK_CUDA( cudaMemset(d_m1, 0, n_dofs*sizeof(reel)) )
    CHECK_CUDA( cudaMemset(d_m2, 0, n_dofs*sizeof(reel)) )
    CHECK_CUDA( cudaMemset(d_m3, 0, n_dofs*sizeof(reel)) )
    CHECK_CUDA( cudaMemset(d_m4, 0, n_dofs*sizeof(reel)) )
  }

  void __GpuDriver::displaySimuInfos(problemType type_){
    if(type_ == forward){
      setComputeSystem(forward);
    }
    else{
      setComputeSystem(backward);
    }

    if(dCompute){
      std::cout << "A system with " << n_dofs << " DOFs has been assembled" << std::endl;
      std::cout << "  This system is composed of " << parallelismThroughExcitations << " parallelized simulations of " << n_dofs/parallelismThroughExcitations << " DOF each." << std::endl;
      std::cout << "  The total number of excitation files is " << numberOfExcitations;
      std::cout << "  hence the number of simulation to perform is " << numberOfSimulationToPerform << std::endl;
      if(numsteps < lengthOfeachExcitation){
        std::cout << "  Warning: The number of steps to perform is inferior to the length of the excitation files" << std::endl;
      }
    }
    
    if(dSystem){
      if(type_ == forward){
        setComputeSystem(forward);
        std::cout << std::endl << "Forward system:" << std::endl;
      }
      else{
        setComputeSystem(backward);
        std::cout << std::endl << "Backward system:" << std::endl;
      }

      std::cout << "Here is the assembled system" << std::endl;
      std::cout << "K:" << std::endl << *K << std::endl;
      std::cout << "Gamma:" << std::endl << *Gamma << std::endl;
      std::cout << "Lambda:" << std::endl << *Lambda << std::endl;
      if(type_ == backward && Psi != nullptr){
        std::cout << "Psi:" << std::endl << *Psi << std::endl;
      }
      else{
        std::cout << "Psi:" << std::endl << "  No Psi tensor has been provided" << std::endl << std::endl;
      }
      std::cout << "ForcePattern:" << std::endl << *ForcePattern << std::endl;
      std::cout << "QinitCond:" << std::endl << "  "; printVector(*h_QinitCond);

      std::cout << "InterpolationMatrix:" << std::endl;
      if(!interpolationMatrix.empty()){
        for(uint i=0; i<interpolationNumberOfPoints; ++i){
          for(uint j=0; j<interpolationWindowSize; ++j){
            std::cout << interpolationMatrix[i*interpolationWindowSize + j] << " ";
          }
          std::cout << std::endl;
        }
      }
      else{
        std::cout << "  No interpolation matrix has been provided" << std::endl;
      }
      std::cout << std::endl;

      std::cout << "Modulation buffer:" <<  std::endl;
      if(!modulationBuffer.empty()){
        printVector(modulationBuffer);
      }
      else{
        std::cout << "  No modulation buffer has been provided" << std::endl << std::endl;
      }
    }

    if(dSolver){
      std::cout << "Solver info:" << std::endl;
      std::cout << "  Number of steps to perform: " << numsteps << std::endl;
      reel duration = numsteps*(interpolationNumberOfPoints+1)*h;
      std::cout << "  Duration length: " << duration << "s" << std::endl;
      std::cout << "  Time step: h=" << h << "s / h2=" << h2 << "s / h6=" << h6 << "s" << std::endl;
    }
  }



/*                    Forward system private methods                    */
  void __GpuDriver::allocateDeviceSystem(){

    fwd_K->allocateOnGPU(h_cusparse);

    fwd_Gamma->allocateOnGPU();

    fwd_Lambda->allocateOnGPU();

    fwd_Psi->allocateOnGPU();

    fwd_ForcePattern->allocateOnGPU();
  }

  void __GpuDriver::allocateDeviceSystemStatesVector(){
    // Allocate the GPU memory and create the cuSPARSE descriptors
    // associated with the forward system.
    CHECK_CUDA( cudaMalloc((void**)&d_fwd_QinitCond, n_dofs_fwd*sizeof(reel)) )
    CHECK_CUDA( cudaMalloc((void**)&d_fwd_Q, n_dofs_fwd*sizeof(reel)) )
    CHECK_CUDA( cudaMemcpy(d_fwd_QinitCond, h_fwd_QinitCond.data(), n_dofs_fwd*sizeof(reel), cudaMemcpyHostToDevice) )
    
    // Copy the device QinitCond initial conditions vector to Q device vector
    CHECK_CUDA( cudaMemcpy(d_fwd_Q, d_fwd_QinitCond, n_dofs_fwd*sizeof(reel), cudaMemcpyDeviceToDevice) )

    CHECK_CUDA( cudaMalloc((void**)&d_fwd_mi, n_dofs_fwd*sizeof(reel)) )
    
    CHECK_CUDA( cudaMalloc((void**)&d_fwd_m1, n_dofs_fwd*sizeof(reel)) )
    CHECK_CUDA( cudaMalloc((void**)&d_fwd_m2, n_dofs_fwd*sizeof(reel)) )
    CHECK_CUDA( cudaMalloc((void**)&d_fwd_m3, n_dofs_fwd*sizeof(reel)) )
    CHECK_CUDA( cudaMalloc((void**)&d_fwd_m4, n_dofs_fwd*sizeof(reel)) )
  }

  void __GpuDriver::extendSystem(){
    // Extend each system to match the parallelismThroughExcitations to achieve
    std::array<uint, 5> dofChecking = {
                                       fwd_K->extendTheSystem(parallelismThroughExcitations), 
                                       fwd_Gamma->extendTheSystem(parallelismThroughExcitations), 
                                       fwd_Lambda->extendTheSystem(parallelismThroughExcitations), 
                                       fwd_ForcePattern->extendTheSystem(parallelismThroughExcitations-1),
                                       extendTheVector(h_fwd_QinitCond, parallelismThroughExcitations-1)};

    // Checking that each system is of the same size
    for(uint i = 0; i < dofChecking.size(); i++){
      // std::cout<<"Dof checking ["<<i<<"]: "<<dofChecking[i]<<std::endl;
      if(dofChecking[i] != dofChecking[0]){
        std::cout << "[Error] __GpuDriver: The number of DOFs is not the same for all System matrix after system extension." << std::endl;
      }
    }

    n_dofs_fwd = dofChecking[0];
  }

  void __GpuDriver::clearFwdK(){
    if(K == fwd_K){
      K = nullptr;
    }

    if(fwd_K != nullptr){
      delete fwd_K;
      fwd_K = nullptr;
    }
  }

  void __GpuDriver::clearFwdPsi(){

    if(Psi == fwd_Psi){
      Psi = nullptr;
    }
    if(fwd_Psi != nullptr){
      delete fwd_Psi;
      fwd_Psi = nullptr;
    }

  }

  void __GpuDriver::clearFwdGamma(){
    if(Gamma == fwd_Gamma){
      Gamma = nullptr;
    }

    if(fwd_Gamma != nullptr){
      delete fwd_Gamma;
      fwd_Gamma = nullptr;
    }
  }

  void __GpuDriver::clearFwdLambda(){
    if(Lambda == fwd_Lambda){
      Lambda = nullptr;
    }

    if(fwd_Lambda != nullptr){
      delete fwd_Lambda;
      fwd_Lambda = nullptr;
    }
  }

  void __GpuDriver::clearFwdForcePattern(){
    if(ForcePattern == fwd_ForcePattern){
      ForcePattern = nullptr;
    }

    if(fwd_ForcePattern != nullptr){
      delete fwd_ForcePattern;
      fwd_ForcePattern = nullptr;
    }
  }

  void __GpuDriver::clearFwdInitialConditions(){
    if(d_QinitCond == d_fwd_QinitCond){
      d_QinitCond = nullptr;
    }

    if(d_fwd_QinitCond != nullptr){
      CHECK_CUDA( cudaFree(d_fwd_QinitCond) )
      d_fwd_QinitCond = nullptr;
    }

    if(h_fwd_QinitCond.size() != 0){
      h_fwd_QinitCond.clear();
    }
  }

  void __GpuDriver::clearSystemStatesVector(){
    if(d_fwd_Q != nullptr){
      if(d_Q == d_fwd_Q){
        d_Q = nullptr;
      }

      CHECK_CUDA( cudaFree(d_fwd_Q) )
      d_fwd_Q = nullptr;
    }

    if(d_fwd_mi != nullptr){
      if(d_mi == d_fwd_mi){
        d_mi = nullptr;
      }

      CHECK_CUDA( cudaFree(d_fwd_mi) )
      d_fwd_mi = nullptr;
    }

    if(d_fwd_m1 != nullptr){
      if(d_m1 == d_fwd_m1){
        d_m1 = nullptr;
      }

      CHECK_CUDA( cudaFree(d_fwd_m1) )
      d_fwd_m1 = nullptr;
    }
    if(d_fwd_m2 != nullptr){
      if(d_m2 == d_fwd_m2){
        d_m2 = nullptr;
      }

      CHECK_CUDA( cudaFree(d_fwd_m2) )
      d_fwd_m2 = nullptr;
    }
    if(d_fwd_m3 != nullptr){
      if(d_m3 == d_fwd_m3){
        d_m3 = nullptr;
      }

      CHECK_CUDA( cudaFree(d_fwd_m3) )
      d_fwd_m3 = nullptr;
    }
    if(d_fwd_m4 != nullptr){
      if(d_m4 == d_fwd_m4){
        d_m4 = nullptr;
      }

      CHECK_CUDA( cudaFree(d_fwd_m4) )
      d_fwd_m4 = nullptr;
    }
  }



/*                    Adjoint system private methods                    */

  void __GpuDriver::allocateDeviceAdjointSystem(){
    bwd_K->allocateOnGPU(h_cusparse);

    bwd_Gamma->allocateOnGPU();

    bwd_Lambda->allocateOnGPU();

    bwd_Psi->allocateOnGPU();

    bwd_ForcePattern->allocateOnGPU();
  }

  void __GpuDriver::allocateDeviceAdjointStatesVector(){
    // Allocate the GPU memory and create the cuSPARSE descriptors
    // associated with the backward system.
    CHECK_CUDA( cudaMalloc((void**)&d_bwd_QinitCond, n_dofs_bwd*sizeof(reel)) )
    CHECK_CUDA( cudaMalloc((void**)&d_bwd_Q, n_dofs_bwd*sizeof(reel)) )
    CHECK_CUDA( cudaMemcpy(d_bwd_QinitCond, h_bwd_QinitCond.data(), n_dofs_bwd*sizeof(reel), cudaMemcpyHostToDevice) )
    
    // Copy the device QinitCond initial conditions vector to Q device vector
    CHECK_CUDA( cudaMemcpy(d_bwd_Q, d_bwd_QinitCond, n_dofs_bwd*sizeof(reel), cudaMemcpyDeviceToDevice) )

    CHECK_CUDA( cudaMalloc((void**)&d_bwd_mi, n_dofs_bwd*sizeof(reel)) )
  
    CHECK_CUDA( cudaMalloc((void**)&d_bwd_m1, n_dofs_bwd*sizeof(reel)) )
    CHECK_CUDA( cudaMalloc((void**)&d_bwd_m2, n_dofs_bwd*sizeof(reel)) )
    CHECK_CUDA( cudaMalloc((void**)&d_bwd_m3, n_dofs_bwd*sizeof(reel)) )
    CHECK_CUDA( cudaMalloc((void**)&d_bwd_m4, n_dofs_bwd*sizeof(reel)) )
  }

  void __GpuDriver::extendAdjoint(){
    // Extend each system to match the parallelismThroughExcitations to achieve
    std::array<uint, 6> dofChecking = {
                                       bwd_K->extendTheSystem(parallelismThroughExcitations), 
                                       bwd_Gamma->extendTheSystem(parallelismThroughExcitations), 
                                       bwd_Lambda->extendTheSystem(parallelismThroughExcitations),
                                       bwd_Psi->extendTheSystem(parallelismThroughExcitations), 
                                       bwd_ForcePattern->extendTheSystem(parallelismThroughExcitations-1),
                                       extendTheVector(h_bwd_QinitCond, parallelismThroughExcitations-1)};

    // Checking that each system is of the same size
    for(uint i = 0; i < dofChecking.size(); i++){
      // std::cout<<"Dof checking ["<<i<<"]: "<<dofChecking[i]<<std::endl;
      if(dofChecking[i] != dofChecking[0]){
        std::cout << "[Error] __GpuDriver: The number of DOFs is not the same for all System matrix after system extension." << std::endl;
      }
    }

    n_dofs_bwd = dofChecking[0];
  }

  void __GpuDriver::clearBwdK(){
    if(K == bwd_K){
      K = nullptr;
    }

    if(bwd_K != nullptr){
      delete bwd_K;
      bwd_K = nullptr;
    }
  }

  void __GpuDriver::clearBwdGamma(){
    if(Gamma == bwd_Gamma){
      Gamma = nullptr;
    }

    if(bwd_Gamma != nullptr){
      delete bwd_Gamma;
      bwd_Gamma = nullptr;
    }
  }

  void __GpuDriver::clearBwdLambda(){
    if(Lambda == bwd_Lambda){
      Lambda = nullptr;
    }

    if(bwd_Lambda != nullptr){
      delete bwd_Lambda;
      bwd_Lambda = nullptr;
    }
  }

  void __GpuDriver::clearBwdPsi(){
    if(Psi == bwd_Psi){
      Psi = nullptr;
    }

    if(bwd_Psi != nullptr){
      delete bwd_Psi;
      bwd_Psi = nullptr;
    }
  }

  void __GpuDriver::clearBwdForcePattern(){
    if(ForcePattern == bwd_ForcePattern){
      ForcePattern = nullptr;
    }

    if(bwd_ForcePattern != nullptr){
      delete bwd_ForcePattern;
      bwd_ForcePattern = nullptr;
    }
  }

  void __GpuDriver::clearBwdInitialConditions(){
    if(d_QinitCond == d_bwd_QinitCond){
      d_QinitCond = nullptr;
    }

    if(d_bwd_QinitCond != nullptr){
      CHECK_CUDA( cudaFree(d_bwd_QinitCond) )
      d_bwd_QinitCond = nullptr;
    }

    if(h_bwd_QinitCond.size() != 0){
      h_bwd_QinitCond.clear();
    }
  }

  void __GpuDriver::clearAdjointStatesVector(){
    if(d_bwd_Q != nullptr){
      if(d_Q == d_bwd_Q){
        d_Q = nullptr;
      }

      CHECK_CUDA( cudaFree(d_bwd_Q) )
      d_bwd_Q = nullptr;
    }

    if(d_bwd_mi != nullptr){
      if(d_mi == d_bwd_mi){
        d_mi = nullptr;
      }

      CHECK_CUDA( cudaFree(d_bwd_mi) )
      d_bwd_mi = nullptr;
    }

    if(d_bwd_m1 != nullptr){
      if(d_m1 == d_bwd_m1){
        d_m1 = nullptr;
      }

      CHECK_CUDA( cudaFree(d_bwd_m1) )
      d_bwd_m1 = nullptr;
    }
    if(d_bwd_m2 != nullptr){
      if(d_m2 == d_bwd_m2){
        d_m2 = nullptr;
      }

      CHECK_CUDA( cudaFree(d_bwd_m2) )
      d_bwd_m2 = nullptr;
    }
    if(d_bwd_m3 != nullptr){
      if(d_m3 == d_bwd_m3){
        d_m3 = nullptr;
      }

      CHECK_CUDA( cudaFree(d_bwd_m3) )
      d_bwd_m3 = nullptr;
    }
    if(d_bwd_m4 != nullptr){
      if(d_m4 == d_bwd_m4){
        d_m4 = nullptr;
      }

      CHECK_CUDA( cudaFree(d_bwd_m4) )
      d_bwd_m4 = nullptr;
    }
  }