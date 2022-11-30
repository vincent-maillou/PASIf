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



  __GpuDriver::__GpuDriver(std::vector<std::vector<reel>> excitationSet_, uint sampleRate_) : sampleRate(sampleRate_){
    d_ExcitationsSet = nullptr;
    loadExcitationsSet(excitationSet_);

  }


  __GpuDriver::~__GpuDriver(){

    // Free memory      
    if(d_ExcitationsSet != nullptr){
      cudaFree(d_ExcitationsSet);
      d_ExcitationsSet = nullptr;
    }


  }




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

    return 0;
  }



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

    numberOfDOFs = M_[0].size();

    // Set the matrix defining the system
    B      = COOMatrix(B_, M_);
    K      = COOMatrix(K_, M_);
    Gamma  = COOTensor(Gamma_, M_);
    Lambda = COOMatrix(Lambda_, M_);

    // Set the force pattern
    for(size_t i = 0; i < ForcePattern_.size(); i++){
      for(size_t j = 0; j < ForcePattern_[i].size(); j++){
        ForcePattern.push_back(ForcePattern_[i][j]);
      }
    }

    // Set the force pattern on the GPU
    CHECK_CUDA( cudaMalloc((void**)&d_ForcePattern, ForcePattern.size()*sizeof(reel)) )
    CHECK_CUDA( cudaMemcpy(d_ForcePattern, ForcePattern.data(), ForcePattern.size()*sizeof(reel), cudaMemcpyHostToDevice) )

    return 0;
  }

