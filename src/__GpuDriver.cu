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
    __GpuDriver::__GpuDriver(std::vector<std::vector<double>> excitationSet_, uint sampleRate_) : 
        numberOfDOFs(0),
        nStreams(1),
        IntraStrmParallelism(1),
        numberOfSimulationToPerform(0),
        alpha(1.0),
        beta1(1.0),
        beta0(0.0){
      // System
      B      = nullptr;
      K      = nullptr;
      Gamma  = nullptr;
      Lambda = nullptr;
      ForcePattern     = nullptr;
      d_ExcitationsSet = nullptr;

      // RK4
      d_QinitCond = nullptr;
      d_Q1 = nullptr;
      d_Q2 = nullptr;

      d_mi = nullptr;
      d_ki = nullptr;

      d_m1 = nullptr;
      d_m2 = nullptr;
      d_m3 = nullptr;
      d_m4 = nullptr;

      d_k1 = nullptr;
      d_k2 = nullptr;
      d_k3 = nullptr;
      d_k4 = nullptr;

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
      // Set the RK4 timesteps
      h = 1.0/sampleRate;
      h2 = h/2.0;
      h6 = h/6.0;

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
    numberOfDOFs = InitialConditions_[0].size();

    // Allocate the QinitCond vector with the set of initials conditions
    for(size_t k(0); k<InitialConditions_.size(); k++){
      for(size_t i(0); i<InitialConditions_[k].size(); i++){
        QinitCond.push_back(InitialConditions_[k][i]);
      }
    }
  }



  /** __GpuDriver::driver_getAmplitudes()
   * @brief 
   * 
   * @return std::array<std::vector<reel>, 2>
   */
   std::array<std::vector<reel>, 2> __GpuDriver::_getAmplitudes(){
      
    optimizeIntraStrmParallelisme();

    if(true){
      std::cout << "Checking the system assembly" << std::endl;
      std::cout << "B:" << std::endl << *B << std::endl;
      std::cout << "K:" << std::endl << *K << std::endl;
      std::cout << "Gamma:" << std::endl << *Gamma << std::endl;
      std::cout << "Lambda:" << std::endl << *Lambda << std::endl;
      std::cout << "ForcePattern:" << std::endl << *ForcePattern << std::endl;
      std::cout << "QinitCond:" << std::endl; printVector(QinitCond);
    }
    if(true){
      std::cout << "A system with " << numberOfDOFs << " DOFs has been assembled" << std::endl;
      std::cout << "  This system is composed of " << IntraStrmParallelism << " parallelized simulations of " << numberOfDOFs/IntraStrmParallelism << " DOF each." << std::endl;
      std::cout << "  The total number of excitation files is " << numberOfExcitations << std::endl;
      std::cout << "  Hence the number of simulation to perform is " << numberOfSimulationToPerform << std::endl;
    }
    if(false){
      std::cout << "The timestep of the simulations are" << std::endl;
      std::cout << "h = " << h << std::endl;
      std::cout << "h2 = " << h2 << std::endl;
      std::cout << "h6 = " << h6 << std::endl;
    }
    

    // Allocate the memory for the states and RK4 vectors coefficients,
    // and create the dense vector descriptors
    CHECK_CUDA( cudaMalloc((void**)&d_QinitCond, numberOfDOFs*sizeof(reel)) )
    CHECK_CUDA( cudaMalloc((void**)&d_Q1, numberOfDOFs*sizeof(reel)) )
    CHECK_CUDA( cudaMemcpy(d_QinitCond, QinitCond.data(), numberOfDOFs*sizeof(reel), cudaMemcpyHostToDevice) )
    // Copy the device QinitCond initial conditions vector to Q1 device vector
    CHECK_CUDA( cudaMemcpy(d_Q1, d_QinitCond, numberOfDOFs*sizeof(reel), cudaMemcpyDeviceToDevice) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_Q1_desc, numberOfDOFs, d_Q1, CUDA_R_32F) )
    CHECK_CUDA( cudaMalloc((void**)&d_Q2, numberOfDOFs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_Q2_desc, numberOfDOFs, d_Q2, CUDA_R_32F) )

    CHECK_CUDA( cudaMalloc((void**)&d_mi, numberOfDOFs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_mi_desc, numberOfDOFs, d_mi, CUDA_R_32F) )
    CHECK_CUDA( cudaMalloc((void**)&d_ki, numberOfDOFs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_ki_desc, numberOfDOFs, d_ki, CUDA_R_32F) )
    
    CHECK_CUDA( cudaMalloc((void**)&d_m1, numberOfDOFs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_m1_desc, numberOfDOFs, d_m1, CUDA_R_32F) )
    CHECK_CUDA( cudaMalloc((void**)&d_m2, numberOfDOFs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_m2_desc, numberOfDOFs, d_m2, CUDA_R_32F) )
    CHECK_CUDA( cudaMalloc((void**)&d_m3, numberOfDOFs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_m3_desc, numberOfDOFs, d_m3, CUDA_R_32F) )
    CHECK_CUDA( cudaMalloc((void**)&d_m4, numberOfDOFs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_m4_desc, numberOfDOFs, d_m4, CUDA_R_32F) )

    CHECK_CUDA( cudaMalloc((void**)&d_k1, numberOfDOFs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_k1_desc, numberOfDOFs, d_k1, CUDA_R_32F) )
    CHECK_CUDA( cudaMalloc((void**)&d_k2, numberOfDOFs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_k2_desc, numberOfDOFs, d_k2, CUDA_R_32F) )
    CHECK_CUDA( cudaMalloc((void**)&d_k3, numberOfDOFs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_k3_desc, numberOfDOFs, d_k3, CUDA_R_32F) )
    CHECK_CUDA( cudaMalloc((void**)&d_k4, numberOfDOFs*sizeof(reel)) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_k4_desc, numberOfDOFs, d_k4, CUDA_R_32F) )



    // Allocate the matrices and vectors on the GPU
    B->AllocateOnGPU(h_cuSPARSE, d_mi_desc, d_ki_desc);
    K->AllocateOnGPU(h_cuSPARSE, d_mi_desc, d_ki_desc);
    Gamma->AllocateOnGPU();
    Lambda->AllocateOnGPU();
    ForcePattern->AllocateOnGPU();


    std::vector<reel> resultsQ1;
    std::vector<reel> resultsQ2;
    resultsQ1.resize(numberOfDOFs*numberOfSimulationToPerform);
    resultsQ2.resize(numberOfDOFs*numberOfSimulationToPerform);


    auto begin = std::chrono::high_resolution_clock::now();


    // Perform the simulations
    for(size_t k(0); k<numberOfSimulationToPerform; k++){

      std::cout << "  " << k+1 << " / " << numberOfSimulationToPerform  << std::endl;

      // Performe the rk4 steps
      for(uint t(0); t<lengthOfeachExcitation; ++t){

        rkStep(k, t);

      }



      // Copy the results of the performed simulation from the GPU to the CPU
      CHECK_CUDA( cudaMemcpy(resultsQ1.data()+k*numberOfDOFs, d_Q1, numberOfDOFs*sizeof(reel), cudaMemcpyDeviceToHost) )
      CHECK_CUDA( cudaMemcpy(resultsQ2.data()+k*numberOfDOFs, d_Q2, numberOfDOFs*sizeof(reel), cudaMemcpyDeviceToHost) )
      CHECK_CUDA( cudaDeviceSynchronize() )

      // Reset Q1 and Q2 to initials conditions
      CHECK_CUDA( cudaMemcpy(d_Q1, d_QinitCond, numberOfDOFs*sizeof(reel), cudaMemcpyDeviceToDevice) )
      CHECK_CUDA( cudaMemset(d_Q2, 0, numberOfDOFs*sizeof(reel)) )

      // Reset all of the other vectors to 0
      CHECK_CUDA( cudaMemset(d_mi, 0, numberOfDOFs*sizeof(reel)) )
      CHECK_CUDA( cudaMemset(d_ki, 0, numberOfDOFs*sizeof(reel)) )

      CHECK_CUDA( cudaMemset(d_m1, 0, numberOfDOFs*sizeof(reel)) )
      CHECK_CUDA( cudaMemset(d_m2, 0, numberOfDOFs*sizeof(reel)) )
      CHECK_CUDA( cudaMemset(d_m3, 0, numberOfDOFs*sizeof(reel)) )
      CHECK_CUDA( cudaMemset(d_m4, 0, numberOfDOFs*sizeof(reel)) )

      CHECK_CUDA( cudaMemset(d_k1, 0, numberOfDOFs*sizeof(reel)) )
      CHECK_CUDA( cudaMemset(d_k2, 0, numberOfDOFs*sizeof(reel)) )
      CHECK_CUDA( cudaMemset(d_k3, 0, numberOfDOFs*sizeof(reel)) )
      CHECK_CUDA( cudaMemset(d_k4, 0, numberOfDOFs*sizeof(reel)) )
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-begin;
    std::cout << "CUDA solver execution time: " << elapsed_seconds.count() << "s" << std::endl;


    // Cut the results vector to the correct size
    if(exceedingSimulations != 0){
      resultsQ1.resize(numberOfDOFs*(numberOfSimulationToPerform-1)+exceedingSimulations);
      resultsQ2.resize(numberOfDOFs*(numberOfSimulationToPerform-1)+exceedingSimulations);
    }



    return std::array<std::vector<reel>, 2>{resultsQ1, resultsQ2};
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



  /**
   * @brief Compute the derivatives of the system
   * 
   */
   void __GpuDriver::derivatives(cusparseDnVecDescr_t m_desc, 
                                 cusparseDnVecDescr_t k_desc,
                                 cusparseDnVecDescr_t q1_desc, 
                                 cusparseDnVecDescr_t q2_desc,
                                 uint k, 
                                 uint t){

    // Get the pointers from the descriptors
    reel *pm; reel *pk; 
    reel *pq1; reel *pq2;
    CHECK_CUSPARSE( cusparseDnVecGetValues(m_desc, (void**)&pm) )
    CHECK_CUSPARSE( cusparseDnVecGetValues(k_desc, (void**)&pk) )
    CHECK_CUSPARSE( cusparseDnVecGetValues(q1_desc, (void**)&pq1) )
    CHECK_CUSPARSE( cusparseDnVecGetValues(q2_desc, (void**)&pq2) )

    // m = k
    cublasScopy(h_cublas, 
                numberOfDOFs, 
                pq2, 
                1, 
                pm, 
                1);


    // k = B.d_ki + K.d_mi + Gamma.d_mi² + Lambda.d_mi³ + ForcePattern.d_ExcitationsSet
    // k = B.d_ki
    cusparseSpMV(h_cuSPARSE, 
                 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                 d_alpha, 
                 B->sparseMat_desc, 
                 q2_desc,
                 d_beta0, 
                 k_desc, 
                 CUDA_R_32F, 
                 CUSPARSE_SPMV_ALG_DEFAULT, 
                 B->d_buffer);
    
    // k += K.d_mi
    cusparseSpMV(h_cuSPARSE, 
                 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                 d_alpha, 
                 K->sparseMat_desc, 
                 q1_desc, 
                 d_beta1, 
                 k_desc, 
                 CUDA_R_32F, 
                 CUSPARSE_SPMV_ALG_DEFAULT, 
                 K->d_buffer);
    
    // k += Gamma.d_mi²
    customSpTd2V<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(Gamma->d_val,
                                                               Gamma->d_row, 
                                                               Gamma->d_col,
                                                               Gamma->d_slice, 
                                                               Gamma->nzz, 
                                                               pq1, 
                                                               pk);
    
    // k += Lambda.d_mi³
    customSpTd3V<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(Lambda->d_val,
                                                               Lambda->d_row, 
                                                               Lambda->d_col,
                                                               Lambda->d_slice, 
                                                               Lambda->d_hyperslice,
                                                               Lambda->nzz, 
                                                               pq1, 
                                                               pk);
    
    // k += ForcePattern.d_ExcitationsSet
    customAxpbyMultiForces<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(ForcePattern->d_val,
                                                                        ForcePattern->d_indice,
                                                                        ForcePattern->nzz,
                                                                        d_ExcitationsSet,
                                                                        lengthOfeachExcitation,
                                                                        k,
                                                                        pk,
                                                                        numberOfDOFs,
                                                                        t,
                                                                        IntraStrmParallelism);
    
   }

  

  /**
   * @brief Performe a single Runge-Kutta step
   * 
   */
   void __GpuDriver::rkStep(uint k, 
                            uint t){

    // Compute the derivatives
    derivatives(d_m1_desc, d_k1_desc, d_Q1_desc, d_Q2_desc, k, t);

      updateSlope<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_mi, d_Q1, d_m1, h2, numberOfDOFs);
      updateSlope<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_ki, d_Q2, d_k1, h2, numberOfDOFs);

    derivatives(d_m2_desc, d_k2_desc, d_mi_desc, d_ki_desc, k, t+1);

      updateSlope<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_mi, d_Q1, d_m2, h2, numberOfDOFs);
      updateSlope<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_ki, d_Q2, d_k2, h2, numberOfDOFs);

    derivatives(d_m3_desc, d_k3_desc, d_mi_desc, d_ki_desc, k, t+1);

      updateSlope<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_mi, d_Q1, d_m3, h, numberOfDOFs);
      updateSlope<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_ki, d_Q2, d_k3, h, numberOfDOFs);

    derivatives(d_m4_desc, d_k4_desc, d_mi_desc, d_ki_desc, k, t+2);

    // Compute next Q1 and Q2 vectors
    integrate<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_Q1, d_m1, d_m2, d_m3, d_m4, h6, numberOfDOFs);
    integrate<<<nBlocks, nThreadsPerBlock, 0, streams[0]>>>(d_Q2, d_k1, d_k2, d_k3, d_k4, h6, numberOfDOFs);

   }



  /** __GpuDriver::checkAndDestroy()
   * @brief Check the device pointer array and destroy them if they are not null
   * 
   */
   void __GpuDriver::clearDeviceStatesVector(){
    
    if(d_Q1 != nullptr){
      CHECK_CUDA( cudaFree(d_Q1) )
      d_Q1 = nullptr;
    }
    if(d_Q2 != nullptr){
      CHECK_CUDA( cudaFree(d_Q2) )
      d_Q2 = nullptr;
    }

    if(d_mi != nullptr){
      CHECK_CUDA( cudaFree(d_mi) )
      d_mi = nullptr;
    }
    if(d_ki != nullptr){
      CHECK_CUDA( cudaFree(d_ki) )
      d_ki = nullptr;
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

    if(d_k1 != nullptr){
      CHECK_CUDA( cudaFree(d_k1) )
      d_k1 = nullptr;
    }
    if(d_k2 != nullptr){
      CHECK_CUDA( cudaFree(d_k2) )
      d_k2 = nullptr;
    }
    if(d_k3 != nullptr){
      CHECK_CUDA( cudaFree(d_k3) )
      d_k3 = nullptr;
    }
    if(d_k4 != nullptr){
      CHECK_CUDA( cudaFree(d_k4) )
      d_k4 = nullptr;
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
    sizeOfStates += 13*sizeof(reel)*numberOfDOFs;

    // std::cout << "Size of the states: " << sizeOfStates << " bytes" << std::endl;

    size_t totalSize = sizeOfSystem + sizeOfStates;

    // 3. Compute the max number of system that we can fit in the gpu memory

    size_t maxNumberOfSystem = (0.8*freeSpace) / totalSize;

    if(maxNumberOfSystem > numberOfExcitations){
      maxNumberOfSystem = numberOfExcitations;
    }

    IntraStrmParallelism = maxNumberOfSystem;

    numberOfSimulationToPerform = numberOfExcitations / IntraStrmParallelism;
    exceedingSimulations = numberOfExcitations % IntraStrmParallelism;
    if(exceedingSimulations != 0){
      numberOfSimulationToPerform++;
    }

    // Extend each system by the number of intra-stream parallelization wanted
    std::array<uint, 6> dofChecking = {B->ExtendTheSystem(IntraStrmParallelism-1), 
                                      K->ExtendTheSystem(IntraStrmParallelism-1), 
                                      Gamma->ExtendTheSystem(IntraStrmParallelism-1), 
                                      Lambda->ExtendTheSystem(IntraStrmParallelism-1), 
                                      ForcePattern->ExtendTheSystem(IntraStrmParallelism-1),
                                      extendTheVector(QinitCond, IntraStrmParallelism-1)};

    // Checking that each system is of the same size
    for(uint i = 0; i < dofChecking.size(); i++){
      if(dofChecking[i] != dofChecking[0]){
        std::cout << "Error : The number of DOFs is not the same for all the Matrix describing the system" << std::endl;
      }
    }

    // Modify if needed the number of DOFs
    if(numberOfDOFs != dofChecking[0]){
      numberOfDOFs = dofChecking[0];
    }
  }




