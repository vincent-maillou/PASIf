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


/**
 * @brief Construct a new gpudriver::  gpudriver object
 * 
 * @param excitationSet_ 
 * @param sampleRate_ 
 */
  __GpuDriver::__GpuDriver(std::vector<std::vector<reel>> excitationSet_, uint sampleRate_) : 
      sampleRate(sampleRate_),
      numberOfDOFs(0),
      nStreams(1),
      IntraStrmParallelism(2){

    B = nullptr;
    K = nullptr;
    Gamma = nullptr;
    Lambda = nullptr;
    ForcePattern = nullptr;

    d_ExcitationsSet = nullptr;
    loadExcitationsSet(excitationSet_);

    streams = nullptr;
    h_cuSPARSE = NULL;

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

    if(B != nullptr){
      delete B;
      B = nullptr;
    }

    if(K != nullptr){
      delete K;
      K = nullptr;
    }

    if(Gamma != nullptr){
      delete Gamma;
      Gamma = nullptr;
    }

    if(Lambda != nullptr){
      delete Lambda;
      Lambda = nullptr;
    }

    if(ForcePattern != nullptr){
      delete ForcePattern;
      ForcePattern = nullptr;
    }

    if(streams != nullptr){
      for(uint i = 0; i < nStreams; i++){
        CHECK_CUDA( cudaStreamDestroy(streams[i]) );
      }
      delete[] streams;
      streams = nullptr;
    }

  }



/**
 * @brief Load the excitation set in the GPU memory
 * 
 * @param excitationSet_ 
 * @return int 
 */
  int __GpuDriver::loadExcitationsSet(std::vector<std::vector<reel>> excitationSet_){
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
        return -1;
      }
    }

    numberOfExcitations    = excitationSet_.size();
    lengthOfeachExcitation = excitationSet_[0].size();

    // Parse the input excitationSet_ to a 1D array
    for(auto &excitation : excitationSet_){
      for(auto &sample : excitation){
        excitationSet.push_back(sample);
      }
    }

    // Allocate memory on the GPU
    CHECK_CUDA( cudaMalloc((void**)&d_ExcitationsSet, excitationSet.size()*sizeof(reel)) )

    // Copy the ExcitationsSet to the GPU
    CHECK_CUDA( cudaMemcpy(d_ExcitationsSet, excitationSet.data(), excitationSet.size()*sizeof(reel), cudaMemcpyHostToDevice) )

    std::cout << "Loaded " << numberOfExcitations << " excitations of length " << lengthOfeachExcitation << " each." << std::endl;

    return 0;
  }



/**
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

    numberOfThreadsPerBlock = 128;
    numberOfBlocks = numberOfSMs * 32;

    // Spawn the streams
    streams = new cudaStream_t[nStreams];
    for(uint i = 0; i < nStreams; i++){
      CHECK_CUDA( cudaStreamCreate(&streams[i]) )
    }

    // Create the cuSPARSE handle
    CHECK_CUSPARSE( cusparseCreate(&h_cuSPARSE) )
    CHECK_CUSPARSE( cusparseSetPointerMode(h_cuSPARSE, CUSPARSE_POINTER_MODE_DEVICE) )


    return 0;
  }



/**
 * @brief Set the parameters of the system
 * 
 * @param M_ 
 * @param B_ 
 * @param K_ 
 * @param Gamma_ 
 * @param Lambda_ 
 * @param ForcePattern_ 
 * @return int 
 */
  int __GpuDriver::__setSystems(std::vector< matrix > & M_,
                   std::vector< matrix > & B_,
                   std::vector< matrix > & K_,
                   std::vector< tensor >  & Gamma_,
                   std::vector< matrix > & Lambda_,
                   std::vector< std::vector<reel> > & ForcePattern_){

    // Check the number of systems in all of the input vectors
    if(M_.size() != B_.size() || M_.size() != K_.size() || M_.size() != Gamma_.size() || M_.size() != Lambda_.size() || M_.size() != ForcePattern_.size()){
      std::cout << "Error : The number of systems is not the same for all the input parameters" << std::endl;
      return -1;
    }

    // Check that the matrix of each system are of the same size
    for(uint i = 0; i < M_.size(); i++){
      if(M_[i].size() != B_[i].size() || M_[i].size() != K_[i].size() || M_[i].size() != Gamma_[i].size() || M_[i].size() != Lambda_[i].size() || M_[i].size() != ForcePattern_[i].size()){
        std::cout << "Error : The size of the matrix of the system " << i << " is not the same for all the input parameters" << std::endl;
        return -1;
      }
    }



    // Check if the system matrix have already been loaded, if so delete them and free the memory
    if(B != nullptr){
      delete B;
      B = nullptr;
    }
    if(K != nullptr){
      delete K;
      K = nullptr;
    }
    if(Gamma != nullptr){
      delete Gamma;
      Gamma = nullptr;
    }
    if(Lambda != nullptr){
      delete Lambda;
      Lambda = nullptr;
    }
    if(ForcePattern != nullptr){
      delete ForcePattern;
      ForcePattern = nullptr;
    }



    // Set the matrix defining the system
    std::vector< matrix > invertedScaledM = M_;
    invertMatrix(invertedScaledM, 1.0);

    B      = new COOMatrix(B_, invertedScaledM);
    K      = new COOMatrix(K_, invertedScaledM);
    Gamma  = new COOTensor(Gamma_, invertedScaledM);
    Lambda = new COOMatrix(Lambda_, invertedScaledM);
    ForcePattern = new COOVector(ForcePattern_);

    std::cout << "test 1" << std::endl;
    // Extend each system by the number of intra-stream parallelization wanted
    std::array<uint, 5> dofChecking = {B->ExtendTheSystem(IntraStrmParallelism), 
                                       K->ExtendTheSystem(IntraStrmParallelism), 
                                       Gamma->ExtendTheSystem(IntraStrmParallelism), 
                                       Lambda->ExtendTheSystem(IntraStrmParallelism), 
                                       ForcePattern->ExtendTheSystem(IntraStrmParallelism)};

    // Checking that each system is of the same size
    for(uint i = 0; i < dofChecking.size(); i++){
      if(dofChecking[i] != dofChecking[0]){
        std::cout << "Error : The number of DOFs is not the same for all the Matrix describing the system" << std::endl;
        return -1;
      }
    }

    // Modify if needed the number of DOFs
    if(numberOfDOFs != dofChecking[0]){
      numberOfDOFs = dofChecking[0];
    }

    return 0;
  }



/**
 * @brief 
 * 
 * @return int 
 */
  int __GpuDriver::__getAmplitudes(){

    std::cout << "Checking the system assembly" << std::endl;
    std::cout << "B:" << std::endl << *B << std::endl;
    std::cout << "K:" << std::endl << *K << std::endl;
    std::cout << "Gamma:" << std::endl << *Gamma << std::endl;
    std::cout << "Lambda:" << std::endl << *Lambda << std::endl;
    std::cout << "ForcePattern:" << std::endl << *ForcePattern << std::endl;

    return 1;
  }


