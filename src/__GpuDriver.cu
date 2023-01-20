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
                             uint numsteps_) : 
        // Simulation
        n_dofs(0),
        baseNumsteps(numsteps_),
        //simNumsteps(numsteps_),

        // Interpolation
        interpolationNumberOfPoints(0),
        interpolationWindowSize(0),

        // Computation parameters
        nStreams(1),
        intraStrmParallelism(1),
        numberOfSimulationToPerform(0),

        // Kernel parameters
        alpha(1.0),
        beta1(1.0),
        beta0(0.0){

      // Interpolation
      d_interpolationMatrix = nullptr;

      // System
      B      = nullptr;
      K      = nullptr;
      Gamma  = nullptr;
      Lambda = nullptr;
      ForcePattern     = nullptr;
      d_ExcitationsSet = nullptr;

      // RK4
      d_QinitCond = nullptr;
      d_Q = nullptr;

      d_mi = nullptr;

      d_m1 = nullptr;
      d_m2 = nullptr;
      d_m3 = nullptr;
      d_m4 = nullptr;

      // CUDA
      streams    = nullptr;
      h_cuSPARSE = NULL;
    
      // Put on the device alpha and beta values for the cuSPARSE API
      CHECK_CUDA( cudaMalloc((void**)&d_alpha, sizeof(reel)) )
      CHECK_CUDA( cudaMalloc((void**)&d_beta1, sizeof(reel)) )
      CHECK_CUDA( cudaMalloc((void**)&d_beta0, sizeof(reel)) )
      CHECK_CUDA( cudaMemcpy(d_alpha, &alpha, sizeof(reel), cudaMemcpyHostToDevice) )
      CHECK_CUDA( cudaMemcpy(d_beta1, &beta1, sizeof(reel), cudaMemcpyHostToDevice) )
      CHECK_CUDA( cudaMemcpy(d_beta0, &beta0, sizeof(reel), cudaMemcpyHostToDevice) )

      _loadExcitationsSet(excitationSet_, sampleRate_);
      setCUDA(nStreams);
    }



  /**
    * @brief Destroy the gpudriver::  gpudriver object
    * 
    */
    __GpuDriver::~__GpuDriver(){
      // Free memory      
      if(d_ExcitationsSet != nullptr){
        CHECK_CUDA( cudaFree(d_ExcitationsSet) );
        d_ExcitationsSet = nullptr;
      }

      clearB();
      clearK();
      clearGamma();
      clearLambda();
      clearForcePattern();
      clearInitialConditions();
      clearInterpolationMatrix();
      clearModulationBuffer();
      clearDeviceStatesVector();

      if(streams != nullptr){
        for(uint i = 0; i < nStreams; i++){
          CHECK_CUDA( cudaStreamDestroy(streams[i]) );
        }
        delete[] streams;
        streams = nullptr;
      }
    }



  /** __GpuDriver::loadExcitationsSet()
    * @brief Load the excitation set in the GPU memory
    * 
    * @param excitationSet_ 
    */
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
          std::cout << "Error : Excitations vectors are not of the same size" << std::endl;
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
      std::cout << "Loaded " << numberOfExcitations << " excitations of length " << lengthOfeachExcitation << " each." << std::endl;
    
      return 0;
    }



  void __GpuDriver::_setB(std::vector< matrix > & B_){
    clearB();

    B = new COOMatrix(B_);
  }

  void __GpuDriver::_setK(std::vector< matrix > & K_){
    clearK();

    K = new COOMatrix(K_);
  }

  void __GpuDriver::_setGamma(std::vector< tensor3d > & Gamma_){
    clearGamma();

    Gamma = new COOTensor3D(Gamma_);
  }

  void __GpuDriver::_setLambda(std::vector< tensor4d > & Lambda_){
    clearLambda();

    Lambda = new COOTensor4D(Lambda_);
  }

  void __GpuDriver::_setForcePattern(std::vector< std::vector<reel> > & ForcePattern_){
    clearForcePattern();

    ForcePattern = new COOVector(ForcePattern_);
  }

  void __GpuDriver::_setInitialConditions(std::vector< std::vector<reel> > & InitialConditions_){
    clearInitialConditions();

    // Initialize the number of DOF at the original size of the system
    n_dofs = InitialConditions_[0].size();

    // Allocate the QinitCond vector with the set of initials conditions
    for(size_t k(0); k<InitialConditions_.size(); k++){
      for(size_t i(0); i<InitialConditions_[k].size(); i++){
        QinitCond.push_back(InitialConditions_[k][i]);
      }
    }
  }

  void __GpuDriver::_setInterpolationMatrix(std::vector<reel> & interpolationMatrix_,
                                            uint interpolationWindowSize_){
    interpolationMatrix = interpolationMatrix_;
    interpolationWindowSize  = interpolationWindowSize_;
    interpolationNumberOfPoints = interpolationMatrix_.size()/interpolationWindowSize_;

    setTimesteps();

    // Allocate the interpolation matrix on the GPU
    CHECK_CUDA( cudaMalloc((void**)&d_interpolationMatrix, interpolationMatrix.size()*sizeof(reel)) )
    // Copy the interpolation matrix to the GPU
    CHECK_CUDA( cudaMemcpy(d_interpolationMatrix, interpolationMatrix.data(), interpolationMatrix.size()*sizeof(reel), cudaMemcpyHostToDevice) )
  }



  void __GpuDriver::_setModulationBuffer(std::vector<reel> & modulationBuffer_){
    modulationBuffer = modulationBuffer_;
    modulationBufferSize = modulationBuffer_.size();

    // Allocate the modulation buffer on the GPU
    CHECK_CUDA( cudaMalloc((void**)&d_modulationBuffer, modulationBufferSize*sizeof(reel)) )
    // Copy the modulation buffer to the GPU
    CHECK_CUDA( cudaMemcpy(d_modulationBuffer, modulationBuffer.data(), modulationBufferSize*sizeof(reel), cudaMemcpyHostToDevice) )
  }



  /** __GpuDriver::driver_getAmplitudes()
   * @brief 
   * 
   * @return std::vector<reel>
   */
   std::vector<reel> __GpuDriver::_getAmplitudes(bool displayComputeInfos_, bool displaySystem_){
      
    optimizeIntraStrmParallelisme();

    if(displaySystem_){
      displayAssembledSystem();
    }
    if(displayComputeInfos_){
      displayComputationInfos();
    }
    if(true){
      std::cout << "The number of steps of the simulations are" << std::endl;
      reel duration = baseNumsteps*(interpolationNumberOfPoints+1)*h;
      std::cout << "  duration = " << duration << "s" << std::endl;
      /* std::cout << "  baseNumsteps = " << baseNumsteps << std::endl;
      std::cout << "  interpolationNumberOfPoints = " << interpolationNumberOfPoints << std::endl;
      std::cout << "  total number of steps = " << baseNumsteps*(interpolationNumberOfPoints+1) << std::endl; */
      std::cout << "The timestep of the simulations are" << std::endl;
      std::cout << "  h = " << h << std::endl;
      std::cout << "  h2 = " << h2 << std::endl;
      std::cout << "  h6 = " << h6 << std::endl;
    }
    

    // Allocate the memory for the states and RK4 vectors coefficients,
    // and create the dense vector descriptors
    CHECK_CUDA( cudaMalloc((void**)&d_QinitCond, n_dofs*sizeof(reel)) )
    CHECK_CUDA( cudaMalloc((void**)&d_Q, n_dofs*sizeof(reel)) )
    CHECK_CUDA( cudaMemcpy(d_QinitCond, QinitCond.data(), n_dofs*sizeof(reel), cudaMemcpyHostToDevice) )
    // Copy the device QinitCond initial conditions vector to Q device vector
    CHECK_CUDA( cudaMemcpy(d_Q, d_QinitCond, n_dofs*sizeof(reel), cudaMemcpyDeviceToDevice) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_Q_desc, n_dofs, d_Q, CUDA_R_32F) )

    CHECK_CUDA( cudaMalloc((void**)&d_mi, n_dofs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_mi_desc, n_dofs, d_mi, CUDA_R_32F) )
    
    CHECK_CUDA( cudaMalloc((void**)&d_m1, n_dofs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_m1_desc, n_dofs, d_m1, CUDA_R_32F) )
    CHECK_CUDA( cudaMalloc((void**)&d_m2, n_dofs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_m2_desc, n_dofs, d_m2, CUDA_R_32F) )
    CHECK_CUDA( cudaMalloc((void**)&d_m3, n_dofs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_m3_desc, n_dofs, d_m3, CUDA_R_32F) )
    CHECK_CUDA( cudaMalloc((void**)&d_m4, n_dofs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_m4_desc, n_dofs, d_m4, CUDA_R_32F) )



    // Allocate the matrices and vectors on the GPU
    B->AllocateOnGPU(h_cuSPARSE, d_mi_desc, d_Q_desc);
    K->AllocateOnGPU(h_cuSPARSE, d_mi_desc, d_Q_desc);
    Gamma->AllocateOnGPU();
    Lambda->AllocateOnGPU();
    ForcePattern->AllocateOnGPU();


    std::vector<reel> resultsQ;
    resultsQ.resize(n_dofs*numberOfSimulationToPerform);


    auto begin = std::chrono::high_resolution_clock::now();

    // Perform the simulations
    for(size_t k(0); k<numberOfSimulationToPerform; ++k){

      uint m(0); // Modulation index

      // Performe the rk4 steps
      for(uint t(0); t<baseNumsteps ; ++t){
        // Always performe one step without interpolation, and then performe the
        // interpolation steps
        for(uint i(0); i<=interpolationNumberOfPoints; ++i){
          rkStep(k, t, i, m);
          
          ++m;
          if(m == modulationBufferSize){
            m = 0;
          }
        }

      }



      // Copy the results of the performed simulation from the GPU to the CPU
      CHECK_CUDA( cudaMemcpy(resultsQ.data()+k*n_dofs, d_Q, n_dofs*sizeof(reel), cudaMemcpyDeviceToHost) )
      CHECK_CUDA( cudaDeviceSynchronize() )

      // Reset Q1 and Q2 to initials conditions
      CHECK_CUDA( cudaMemcpy(d_Q, d_QinitCond, n_dofs*sizeof(reel), cudaMemcpyDeviceToDevice) )

      // Reset all of the other vectors to 0
      CHECK_CUDA( cudaMemset(d_mi, 0, n_dofs*sizeof(reel)) )

      CHECK_CUDA( cudaMemset(d_m1, 0, n_dofs*sizeof(reel)) )
      CHECK_CUDA( cudaMemset(d_m2, 0, n_dofs*sizeof(reel)) )
      CHECK_CUDA( cudaMemset(d_m3, 0, n_dofs*sizeof(reel)) )
      CHECK_CUDA( cudaMemset(d_m4, 0, n_dofs*sizeof(reel)) )
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-begin;
    std::cout << "CUDA solver execution time: " << elapsed_seconds.count() << "s" << std::endl;

    // Cut the results vector to the correct size
    if(exceedingSimulations != 0){
      resultsQ.resize(n_dofs*(numberOfSimulationToPerform-1)+exceedingSimulations);
    }

    return std::vector<reel>{resultsQ};
   }



/****************************************************
 *              Private functions                   *
 ****************************************************/
  /** __GpuDriver::setCUDA()
    * @brief Set the parameters of the computation
    * 
    * @param nStreams_ 
    * @param nIntraStrmParallelism_ 
    * @return int 
    */
    int __GpuDriver::setCUDA(uint nStreams_){
      nStreams = nStreams_;

      /* // Query the number of available devices
      int nDevices;
      CHECK_CUDA( cudaGetDeviceCount(&nDevices) ) */

      // Query the device parameters
      CHECK_CUDA( cudaGetDevice(&deviceId) )
      CHECK_CUDA( cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId) )

      nThreadsPerBlock = 128;
      nBlocks = numberOfSMs * 32;

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
      CHECK_CUSPARSE( cusparseCreate(&h_cuSPARSE) )
      CHECK_CUSPARSE( cusparseSetPointerMode(h_cuSPARSE, CUSPARSE_POINTER_MODE_DEVICE) )

      CHECK_CUSPARSE( cusparseSetStream(h_cuSPARSE, streams[0]) )

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



  /**
   * @brief Compute the derivatives of the system
   * 
   */
   inline void __GpuDriver::derivatives(cusparseDnVecDescr_t m_desc, 
                                        cusparseDnVecDescr_t q_desc, 
                                        uint k, 
                                        uint t,
                                        uint i,
                                        uint m){

    // Get the pointers from the descriptors
    reel *pm; reel *pq;
    CHECK_CUSPARSE( cusparseDnVecGetValues(m_desc, (void**)&pm) )
    CHECK_CUSPARSE( cusparseDnVecGetValues(q_desc, (void**)&pq) )
    
    // k = B.d_ki + K.d_mi + Gamma.d_mi² + Lambda.d_mi³ + ForcePattern.d_ExcitationsSet
    // k = B.d_ki
    cusparseSpMV(h_cuSPARSE, 
                 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                 d_alpha, 
                 B->sparseMat_desc, 
                 q_desc,
                 d_beta0, 
                 m_desc, 
                 CUDA_R_32F, 
                 CUSPARSE_SPMV_ALG_DEFAULT, 
                 B->d_buffer);
    
    // k += K.d_mi
    cusparseSpMV(h_cuSPARSE, 
                 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                 d_alpha, 
                 K->sparseMat_desc, 
                 q_desc, 
                 d_beta1, 
                 m_desc, 
                 CUDA_R_32F, 
                 CUSPARSE_SPMV_ALG_DEFAULT, 
                 K->d_buffer);
    
    // k += Gamma.d_mi²
    SpT3dV<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(Gamma->d_val,
                                                         Gamma->d_row, 
                                                         Gamma->d_col,
                                                         Gamma->d_slice, 
                                                         Gamma->nzz, 
                                                         pq, 
                                                         pm);
    
    // k += Lambda.d_mi³
    SpT4dV<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(Lambda->d_val,
                                                         Lambda->d_row, 
                                                         Lambda->d_col,
                                                         Lambda->d_slice, 
                                                         Lambda->d_hyperslice,
                                                         Lambda->nzz, 
                                                         pq, 
                                                         pm);
    
    // Conditional release of the excitation in the case of a simulation longer 
    // than the excitation length
    if(t < lengthOfeachExcitation){
      // k += ForcePattern.d_ExcitationsSet
      modterpolator(pm, k, t, i, m);
    }
   }

  

  /**
   * @brief Performe a single Runge-Kutta step
   * 
   */
   void __GpuDriver::rkStep(uint k, 
                            uint t,
                            uint i,
                            uint m){

    // Compute the derivatives
    derivatives(d_m1_desc, d_Q_desc, k, t, i, m);

      updateSlope<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_mi, d_Q, d_m1, h2, n_dofs);

    derivatives(d_m2_desc, d_mi_desc, k, t, i+1, m);

      updateSlope<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_mi, d_Q, d_m2, h2, n_dofs);

    derivatives(d_m3_desc, d_mi_desc, k, t, i+1, m);

      updateSlope<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_mi, d_Q, d_m3, h, n_dofs);

    derivatives(d_m4_desc, d_mi_desc, k, t, i+2, m);

    // Compute next state vector Q
    integrate<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_Q, d_m1, d_m2, d_m3, d_m4, h6, n_dofs);
   

    // Store the result in the trajectory
    reel temp = 0.;
    cudaMemcpy(&temp, d_Q, sizeof(reel), cudaMemcpyDeviceToHost);

    h_trajectory.push_back(temp);
   }



  inline void __GpuDriver::modterpolator(reel* Y,
                                         uint  k,
                                         uint  t,
                                         uint  i,
                                         uint  m){

    // "currentSimulation" refers to the simulation number in the case of
    // wich multiple simulation are needed to compute the system against all
    // of the excitation file
    uint currentSimulation = k/intraStrmParallelism;
    uint systemStride      = n_dofs/intraStrmParallelism;
    uint adjustedTime          = t;
    uint adjustedInterpolation = i;

    uint useCase = 0;

    if(interpolationNumberOfPoints == 0){
      adjustedTime         += i;
      adjustedInterpolation = 0;

      useCase = 0;
    }
    else{
      if(i > interpolationNumberOfPoints){
        adjustedTime          += 1;
        adjustedInterpolation -= (interpolationNumberOfPoints+1);
      }
      
      if(adjustedInterpolation == 0){
        useCase = 0;
      }
      else{
        useCase = 1;
      }
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
                                              m);
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
                                                    m);
        break;
    }
  }



  inline void __GpuDriver::displayAssembledSystem(){
    std::cout << "Here is the assembled system" << std::endl;
    std::cout << "B:" << std::endl << *B << std::endl;
    std::cout << "K:" << std::endl << *K << std::endl;
    std::cout << "Gamma:" << std::endl << *Gamma << std::endl;
    std::cout << "Lambda:" << std::endl << *Lambda << std::endl;
    std::cout << "ForcePattern:" << std::endl << *ForcePattern << std::endl;
    std::cout << "QinitCond:" << std::endl; printVector(QinitCond);

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
      std::cout << "No interpolation matrix has been provided" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Modulation buffer:" <<  std::endl;
    printVector(modulationBuffer);

  }


  
  inline void __GpuDriver::displayComputationInfos(){
    std::cout << "A system with " << n_dofs << " DOFs has been assembled" << std::endl;
    std::cout << "  This system is composed of " << intraStrmParallelism << " parallelized simulations of " << n_dofs/intraStrmParallelism << " DOF each." << std::endl;
    std::cout << "  The total number of excitation files is " << numberOfExcitations << std::endl;
    std::cout << "  Hence the number of simulation to perform is " << numberOfSimulationToPerform << std::endl;
    if(baseNumsteps < lengthOfeachExcitation){
      std::cout << "  Warning: The number of steps to perform is inferior to the length of the excitation files" << std::endl;
    }
  }



  /** __GpuDriver::checkAndDestroy()
   * @brief Check the device pointer array and destroy them if they are not null
   * 
   */
   void __GpuDriver::clearDeviceStatesVector(){
    
    if(d_Q != nullptr){
      CHECK_CUDA( cudaFree(d_Q) )
      d_Q = nullptr;
    }

    if(d_mi != nullptr){
      CHECK_CUDA( cudaFree(d_mi) )
      d_mi = nullptr;
    }

    if(d_m1 != nullptr){
      CHECK_CUDA( cudaFree(d_m1) )
      d_m1 = nullptr;
    }
    if(d_m2 != nullptr){
      CHECK_CUDA( cudaFree(d_m2) )
      d_m2 = nullptr;
    }
    if(d_m3 != nullptr){
      CHECK_CUDA( cudaFree(d_m3) )
      d_m3 = nullptr;
    }
    if(d_m4 != nullptr){
      CHECK_CUDA( cudaFree(d_m4) )
      d_m4 = nullptr;
    }
   }

  void __GpuDriver::clearB(){
    if(B != nullptr){
      delete B;
      B = nullptr;
    }
  }

  void __GpuDriver::clearK(){
    if(K != nullptr){
      delete K;
      K = nullptr;
    }
  }

  void __GpuDriver::clearGamma(){
    if(Gamma != nullptr){
      delete Gamma;
      Gamma = nullptr;
    }
  }

  void __GpuDriver::clearLambda(){
    if(Lambda != nullptr){
      delete Lambda;
      Lambda = nullptr;
    }
  }

  void __GpuDriver::clearForcePattern(){
    if(ForcePattern != nullptr){
      delete ForcePattern;
      ForcePattern = nullptr;
    }
  }

  void __GpuDriver::clearInitialConditions(){
    if(QinitCond.size() != 0){
      QinitCond.clear();
    }
    if(d_QinitCond != nullptr){
      CHECK_CUDA( cudaFree(d_QinitCond) )
      d_QinitCond = nullptr;
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



/**
 * @brief Optimize the parallelism of the kernel
 * 
 */
  void __GpuDriver::optimizeIntraStrmParallelisme(){
    
    // 1. Get free storage on the GPU
    size_t freeSpace, totalSpace;
    CHECK_CUDA( cudaMemGetInfo(&freeSpace, &totalSpace) )

    /* std::cout << "Free space on the GPU: " << freeSpace << " bytes" << std::endl;
    std::cout << "Total space on the GPU: " << totalSpace << " bytes" << std::endl;
    std::cout << "Used space " << totalSpace-freeSpace << " bytes" << std::endl; */

    // 2. Compute the size required by 1 instance of the system

    // .1 Size of the matrix of the system
    size_t sizeOfSystem(0);
    sizeOfSystem += B->memFootprint();
    sizeOfSystem += K->memFootprint();
    sizeOfSystem += Gamma->memFootprint();
    sizeOfSystem += Lambda->memFootprint();
    sizeOfSystem += ForcePattern->memFootprint();

    // std::cout << "Size of 1 system: " << sizeOfSystem << " bytes" << std::endl;

    // .2 Size of the rk4 and states vector needed for the computation
    size_t sizeOfStates(0);
    sizeOfStates += 13*sizeof(reel)*n_dofs;

    // std::cout << "Size of the states: " << sizeOfStates << " bytes" << std::endl;

    size_t totalSize = sizeOfSystem + sizeOfStates;

    // 3. Compute the max number of system that we can fit in the gpu memory

    size_t maxNumberOfSystem = (0.8*freeSpace) / totalSize;

    if(maxNumberOfSystem > numberOfExcitations){
      maxNumberOfSystem = numberOfExcitations;
    }

    intraStrmParallelism = maxNumberOfSystem;

    numberOfSimulationToPerform = numberOfExcitations / intraStrmParallelism;
    exceedingSimulations = numberOfExcitations % intraStrmParallelism;
    if(exceedingSimulations != 0){
      numberOfSimulationToPerform++;
    }

    // Extend each system by the number of intra-stream parallelization wanted
    std::array<uint, 6> dofChecking = {B->ExtendTheSystem(intraStrmParallelism-1), 
                                       K->ExtendTheSystem(intraStrmParallelism-1), 
                                       Gamma->ExtendTheSystem(intraStrmParallelism-1), 
                                       Lambda->ExtendTheSystem(intraStrmParallelism-1), 
                                       ForcePattern->ExtendTheSystem(intraStrmParallelism-1),
                                       extendTheVector(QinitCond, intraStrmParallelism-1)};

    // Checking that each system is of the same size
    for(uint i = 0; i < dofChecking.size(); i++){
      if(dofChecking[i] != dofChecking[0]){
        std::cout << "Error : The number of DOFs is not the same for all the Matrix describing the system" << std::endl;
      }
    }

    // Modify if needed the number of DOFs
    if(n_dofs != dofChecking[0]){
      n_dofs = dofChecking[0];
    }
  }




