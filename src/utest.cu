#include "helpers.cuh"
#include "kernels.cuh"

int main(int argc, char **argv) {
  std::cout << "Unitary tests for the kernels of the project" << std::endl;

  int  deviceId;
  int  numberOfSMs;
  uint nBlocks;
  uint nThreadsPerBlock;

  CHECK_CUDA( cudaGetDevice(&deviceId) )
  CHECK_CUDA( cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId) )

  nThreadsPerBlock = 128;
  nBlocks = numberOfSMs * 32;



  // Test of the customSpMV3 kernel
  {
  std::cout << std::endl << "Test of the customSpMV3 kernel" << std::endl;

  std::vector<reel> h_val = {1, 2};
  std::vector<uint> h_col = {0, 2};
  std::vector<uint> h_row = {0, 2};
  uint nzz = h_val.size();

  std::vector<reel> h_x = {2, 2, 2};
  std::vector<reel> h_y = {0, 0, 0};

  reel* d_val;
  uint* d_col;
  uint* d_row;
  reel* d_x;
  reel* d_y;

  CHECK_CUDA( cudaMalloc((void**)&d_val, nzz*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_col, nzz*sizeof(uint)) )
  CHECK_CUDA( cudaMalloc((void**)&d_row, (h_row.size())*sizeof(uint)) )
  CHECK_CUDA( cudaMalloc((void**)&d_x, h_x.size()*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_y, h_y.size()*sizeof(reel)) )

  CHECK_CUDA( cudaMemcpy(d_val, h_val.data(), nzz*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_col, h_col.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_row, h_row.data(), (h_row.size())*sizeof(uint), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_x, h_x.data(), h_x.size()*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_y, h_y.data(), h_y.size()*sizeof(reel), cudaMemcpyHostToDevice) )

  customSpMV3<<<nBlocks, nThreadsPerBlock>>>(d_val, d_row, d_col, nzz, d_x, d_y);
  CHECK_CUDA( cudaDeviceSynchronize() )

  CHECK_CUDA( cudaMemcpy(h_y.data(), d_y, h_y.size()*sizeof(reel), cudaMemcpyDeviceToHost) )
  CHECK_CUDA( cudaDeviceSynchronize() )

  std::cout << "  h_y = " << h_y[0] << " " << h_y[1] << " " << h_y[2] << std::endl;

  CHECK_CUDA( cudaFree(d_val) )
  CHECK_CUDA( cudaFree(d_col) )
  CHECK_CUDA( cudaFree(d_row) )
  CHECK_CUDA( cudaFree(d_x) )
  CHECK_CUDA( cudaFree(d_y) )
  }


  // Test of the customSpTV2 kernel
  {
  std::cout << std::endl << "Test of the customSpTV2 kernel" << std::endl;

  std::vector<reel> h_val = {1, 1, 1};
  std::vector<uint> h_col = {0, 1, 2};
  std::vector<uint> h_row = {0, 1, 2};
  std::vector<uint> h_slice = {0, 1, 2};
  uint nzz = h_val.size();

  std::vector<reel> h_x = {1, 2, 3};
  std::vector<reel> h_y = {0, 0, 0};

  reel* d_val;
  uint* d_col;
  uint* d_row;
  uint* d_slice;
  reel* d_x;
  reel* d_y;

  CHECK_CUDA( cudaMalloc((void**)&d_val, nzz*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_col, nzz*sizeof(uint)) )
  CHECK_CUDA( cudaMalloc((void**)&d_row, (h_row.size())*sizeof(uint)) )
  CHECK_CUDA( cudaMalloc((void**)&d_slice, h_slice.size()*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_x, h_x.size()*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_y, h_y.size()*sizeof(reel)) )

  CHECK_CUDA( cudaMemcpy(d_val, h_val.data(), nzz*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_col, h_col.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_row, h_row.data(), (h_row.size())*sizeof(uint), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_slice, h_slice.data(), h_slice.size()*sizeof(uint), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_x, h_x.data(), h_x.size()*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_y, h_y.data(), h_y.size()*sizeof(reel), cudaMemcpyHostToDevice) )

  customSpTV2<<<nBlocks, nThreadsPerBlock>>>(d_val, d_row, d_col, d_slice, nzz, d_x, d_y);
  CHECK_CUDA( cudaDeviceSynchronize() )

  CHECK_CUDA( cudaMemcpy(h_y.data(), d_y, h_y.size()*sizeof(reel), cudaMemcpyDeviceToHost) )
  CHECK_CUDA( cudaDeviceSynchronize() )


  std::cout << "  h_y = " << h_y[0] << " " << h_y[1] << " " << h_y[2] << std::endl;
  }


  // Test of the customAxpbyMultiForces kernel
  {
  std::cout << std::endl << "Test of the customAxpbyMultiForces kernel" << std::endl;
  uint lengthOfeachExcitation = 10;
  uint nExcitations = 8;
  std::vector<reel> excitationsSet;
  for(size_t i(0); i<nExcitations; ++i){
    for(size_t j(0); j<lengthOfeachExcitation; ++j)
      excitationsSet.push_back(i);
  }
    

  uint n = 2;
  std::vector<reel> h_val = {1, 1};
  std::vector<uint> h_indice = {0, 1};
  uint nzz = h_val.size();

  std::vector<reel> h_y = {0, 0};



  uint intraStrmParallelism = 2;

  uint numberOfSimulationToPerform = nExcitations / intraStrmParallelism;
  uint exceedingSimulations = nExcitations % intraStrmParallelism;
  if(exceedingSimulations != 0){
    numberOfSimulationToPerform++;
  }

  // Extend the systems regarding the intraStrmParallelism 
  for(size_t i(0); i<intraStrmParallelism-1; ++i){
    for(size_t j(0); j<nzz; ++j){
      h_val.push_back(h_val[j]);
      h_y.push_back(h_y[j]);
      h_indice.push_back(h_indice[j]+(i+1)*n);
    }
  }

  nzz = h_val.size();
  n = h_y.size();

  // Print the modified vector for checking
  std::cout << "  h_val = ";
  for(size_t i(0); i<nzz; ++i){
    std::cout << h_val[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  h_indice = ";
  for(size_t i(0); i<nzz; ++i){
    std::cout << h_indice[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  h_y = ";
  for(size_t i(0); i<nzz; ++i){
    std::cout << h_y[i] << " ";
  }
  std::cout << std::endl;
  

  reel* d_excitations;
  reel* d_val;
  uint* d_indice;
  reel* d_y;

  CHECK_CUDA( cudaMalloc((void**)&d_excitations, nExcitations*lengthOfeachExcitation*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_val, nzz*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_indice, nzz*sizeof(uint)) )
  CHECK_CUDA( cudaMalloc((void**)&d_y, h_y.size()*sizeof(reel)) )

  CHECK_CUDA( cudaMemcpy(d_excitations, excitationsSet.data(), nExcitations*lengthOfeachExcitation*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_val, h_val.data(), nzz*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_indice, h_indice.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_y, h_y.data(), h_y.size()*sizeof(reel), cudaMemcpyHostToDevice) )



  std::vector<reel> results;
  results.resize(numberOfSimulationToPerform*n);

  for(size_t k(0); k<numberOfSimulationToPerform; k++){

    std::cout << "  Simulation " << k+1 << " over " << numberOfSimulationToPerform << std::endl;
    
    for(size_t t = 0; t < lengthOfeachExcitation; t++){
      customAxpbyMultiForces<<<nBlocks, nThreadsPerBlock>>>
        (d_val, d_indice, nzz, d_excitations, lengthOfeachExcitation, k,
         d_y, n, t, intraStrmParallelism);
    }
    

    // Copy the results of the performed simulation from the GPU to the CPU
    CHECK_CUDA( cudaMemcpy(results.data()+k*n, d_y, n*sizeof(reel), cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaDeviceSynchronize() )

    // Reset the y vector
    CHECK_CUDA( cudaMemset(d_y, 0, n*sizeof(reel)) )
    }

  // Print the result vector
  std::cout << "  results = ";
  for (uint i = 0; i < numberOfSimulationToPerform*n; i++){
    std::cout << results[i] << " ";
  }
  std::cout << std::endl;



  }

  // Test of the updateSlope kernel
  {
  std::cout << std::endl << "Test of the updateSlope kernel" << std::endl;
  uint n = 3;
  std::vector<reel> h_rki = {0, 0, 0};
  std::vector<reel> h_q   = {1, 1, 1};
  std::vector<reel> h_rk  = {1, 2, 3};
  reel dt = 2;

  reel* d_rki;
  reel* d_q;
  reel* d_rk;

  CHECK_CUDA( cudaMalloc((void**)&d_rki, n*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_q, n*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_rk, n*sizeof(reel)) )

  CHECK_CUDA( cudaMemcpy(d_rki, h_rki.data(), n*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_q, h_q.data(), n*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_rk, h_rk.data(), n*sizeof(reel), cudaMemcpyHostToDevice) )

  updateSlope<<<nBlocks, nThreadsPerBlock>>>(d_rki, d_q, d_rk, dt, n);

  CHECK_CUDA( cudaMemcpy(h_rki.data(), d_rki, n*sizeof(reel), cudaMemcpyDeviceToHost) )
  CHECK_CUDA( cudaDeviceSynchronize() )

  std::cout << "  h_rki = ";
  for (uint i = 0; i < n; i++)
    {
      std::cout << h_rki[i] << " ";
    }


  }

  // Test of the integrate kernel
  {
  std::cout << std::endl << "Test of the integrate kernel" << std::endl;
  uint n = 3;
  std::vector<reel> h_q   = {1, 1, 1};
  std::vector<reel> h_rk1 = {1, 2, 3};
  std::vector<reel> h_rk2 = {1, 2, 3};
  std::vector<reel> h_rk3 = {1, 2, 3};
  std::vector<reel> h_rk4 = {1, 2, 3};
  reel h6 = 2;

  reel* d_q;
  reel* d_rk1;
  reel* d_rk2;
  reel* d_rk3;
  reel* d_rk4;

  CHECK_CUDA( cudaMalloc((void**)&d_q, n*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_rk1, n*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_rk2, n*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_rk3, n*sizeof(reel)) )
  CHECK_CUDA( cudaMalloc((void**)&d_rk4, n*sizeof(reel)) )

  CHECK_CUDA( cudaMemcpy(d_q, h_q.data(), n*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_rk1, h_rk1.data(), n*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_rk2, h_rk2.data(), n*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_rk3, h_rk3.data(), n*sizeof(reel), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_rk4, h_rk4.data(), n*sizeof(reel), cudaMemcpyHostToDevice) )

  integrate<<<nBlocks, nThreadsPerBlock>>>(d_q, d_rk1, d_rk2, d_rk3, d_rk4, h6, n);

  CHECK_CUDA( cudaMemcpy(h_q.data(), d_q, n*sizeof(reel), cudaMemcpyDeviceToHost) )
  CHECK_CUDA( cudaDeviceSynchronize() )

  std::cout << "  h_q = ";
  for (uint i = 0; i < n; i++){
    std::cout << h_q[i] << " ";
  }
  std::cout << std::endl;


  }


}