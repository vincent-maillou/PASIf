/**
 * @file helpers.cu
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-11-30
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "helpers.cuh"



/****************************************************
 *              COO Matrix
 ****************************************************/
  /**
  * @brief Construct a new COOMatrix::COOMatrix object
  * 
  * @param denseMatrix 
  * @param scaleMatrix 
  */
    COOMatrix::COOMatrix(std::vector< matrix > & denseMatrix) :
        n(0),
        alpha(1),
        beta(1){
      // Set device pointer to nullprt
      d_val = nullptr;
      d_row = nullptr;
      d_col = nullptr;

      d_buffer = nullptr;
      bufferSize = 0;

      d_alpha = nullptr;
      d_beta = nullptr;

      // Fill the COO matrix with the values of the vector of dense matrix
      for(size_t k(0); k<denseMatrix.size(); ++k){

        for(size_t i(0); i<denseMatrix[k].size(); ++i){
          for(size_t j(0); j<denseMatrix[k][i].size(); ++j){
            if(std::abs(denseMatrix[k][i][j]) > reel_eps){
              row.push_back(i+n);
              col.push_back(j+n);
              val.push_back(denseMatrix[k][i][j]);
            }
          }
        }
        n += denseMatrix[k].size();

      }
      nzz = val.size();

    }



  /**
  * @brief Destroy the COOMatrix::COOMatrix object
  * 
  */
    COOMatrix::~COOMatrix(){
      if(d_val != nullptr){
        CHECK_CUDA( cudaFree(d_val) );
      }
      if(d_row != nullptr){
        CHECK_CUDA( cudaFree(d_row) );
      }
      if(d_col != nullptr){
        CHECK_CUDA( cudaFree(d_col) );
      }
      if(d_buffer != nullptr){
        CHECK_CUDA( cudaFree(d_buffer) );
      }
      if(d_alpha != nullptr){
        CHECK_CUDA( cudaFree(d_alpha) );
      }
      if(d_beta != nullptr){
        CHECK_CUDA( cudaFree(d_beta) );
      }
    }



  /**
   * @brief Extend the COO Matrix by appending n times the same matrix
   * 
   * @param n 
   */
    uint COOMatrix::ExtendTheSystem(uint nTimes){
      if(nTimes == 0){
        return n;
      }
      
      for(uint i(0); i<nTimes; ++i){
        for(uint j(0); j<nzz; ++j){
          row.push_back(row[j]+(i+1)*n);
          col.push_back(col[j]+(i+1)*n);
          val.push_back(val[j]);
        }
      }
      n += nTimes*n;
      nzz = val.size();

      return n;
    }



  /**
   * @brief Construct a new COOMatrix::AllocateOnGPU object
   * 
   */
    void COOMatrix::AllocateOnGPU(cusparseHandle_t & handle, 
                                  cusparseDnVecDescr_t & vecX, 
                                  cusparseDnVecDescr_t & vecY){

      // Allocate memory on the device
      CHECK_CUDA( cudaMalloc((void**)&d_row, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_col, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_val, nzz*sizeof(reel)) );

      // Copy the data to the device
      CHECK_CUDA( cudaMemcpy(d_row, row.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_col, col.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_val, val.data(), nzz*sizeof(reel), cudaMemcpyHostToDevice) );

      // Create the sparse matrix descriptor and allocate the needed buffer
      CHECK_CUSPARSE( cusparseCreateCoo(&sparseMat_desc, n, n, 
                                        nzz, d_row, d_col, d_val, 
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
      
      CHECK_CUDA( cudaMalloc((void**)&d_alpha, sizeof(reel)) );
      CHECK_CUDA( cudaMalloc((void**)&d_beta, sizeof(reel)) );

      CHECK_CUSPARSE( cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &d_alpha, sparseMat_desc, vecX, &d_beta, vecY, CUDA_R_32F, 
                                              CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )

      CHECK_CUDA( cudaMalloc((void**)&d_buffer, bufferSize) );
    }

    size_t COOMatrix::memFootprint(){
      // Return the number of bytes needed to store this element on the GPU
      size_t memFootprint;

      memFootprint = bufferSize + 2*nzz*sizeof(uint) + nzz*sizeof(reel); 

      return memFootprint;
    }
  
    std::ostream& COOMatrix::print(std::ostream& out) const{
      // Print the sparse COO matrix in a readable format
      if(nzz == 0){
        out << "Empty matrix" << std::endl;
        return out;
      }

      out << "  ";
      size_t k(0);
      for(size_t i(0); i<n; ++i){
        for(size_t j(0); j<n; ++j){
          if(col[k] == j && row[k] == i){
            out << val[k] << " ";
            ++k;
          }
          else{
            out << "_ ";
          }
        } 
        out << std::endl << "  ";
      }
      out << std::endl;
      
      return out;
    }

    std::ostream& operator<<(std::ostream& out, COOMatrix const& mat){
      return mat.print(out);
    }   



/****************************************************
 *              COO Tensor 3D
 ****************************************************/
  /**
  * @brief Construct a new COOTensor3D::COOTensor3D object
  * 
  * @param denseTensor 
  * @param scaleMatrix 
  */
    COOTensor3D::COOTensor3D(std::vector< tensor3d > & denseTensor) : n(0){
      // Set device pointer to nullprt
      d_val = nullptr;
      d_row = nullptr;
      d_col = nullptr;
      d_slice = nullptr;


      // Fill the COO Tensor with the values of the vector of dense tensor
      for(size_t l(0); l<denseTensor.size(); ++l){

        for(size_t k(0); k<denseTensor[l].size(); ++k){
          for(size_t i(0); i<denseTensor[l][k].size(); ++i){
            for(size_t j(0); j<denseTensor[l][k][i].size(); ++j){
              if(std::abs(denseTensor[l][k][i][j]) > reel_eps){
                row.push_back(i+n);
                col.push_back(j+n);
                slice.push_back(k+n);
                val.push_back(denseTensor[l][k][i][j]);
              }
            }
          }
        }
        n += denseTensor[l].size();
      
      }
      nzz = val.size();

    }

  /**
  * @brief Destroy the COOTensor3D::COOTensor3D object
  * 
  */
    COOTensor3D::~COOTensor3D(){
      if(d_val != nullptr){
        CHECK_CUDA( cudaFree(d_val) );
      }
      if(d_row != nullptr){
        CHECK_CUDA( cudaFree(d_row) );
      }
      if(d_col != nullptr){
        CHECK_CUDA( cudaFree(d_col) );
      }
      if(d_slice != nullptr){
        CHECK_CUDA( cudaFree(d_slice) );
      }
    }

  /**
   * @brief Extend the COO Tensor by appending n times the same tensor
   * 
   * @param nTimes 
   */
    uint COOTensor3D::ExtendTheSystem(uint nTimes){
      if(nTimes == 0){
        return n;
      }

      for(uint i(0); i<nTimes; ++i){
        for(uint j(0); j<nzz; ++j){
          row.push_back(row[j]+(i+1)*n);
          col.push_back(col[j]+(i+1)*n);
          slice.push_back(slice[j]+(i+1)*n);
          val.push_back(val[j]);
        }
      }
      n += nTimes*n;
      nzz = val.size();

      return n;
    }

  /**
   * @brief Construct a new COOTensor3D::AllocateOnGPU object
   * 
   */
    void COOTensor3D::AllocateOnGPU(){
      // Allocate memory on the device
      CHECK_CUDA( cudaMalloc((void**)&d_row, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_col, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_slice, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_val, nzz*sizeof(reel)) );

      // Copy the data to the device
      CHECK_CUDA( cudaMemcpy(d_row, row.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_col, col.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_slice, slice.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_val, val.data(), nzz*sizeof(reel), cudaMemcpyHostToDevice) );
    }

    size_t COOTensor3D::memFootprint(){
      // Return the number of bytes needed to store this element on the GPU
      size_t memFootprint;

      memFootprint = 3*nzz*sizeof(uint) + nzz*sizeof(reel); 

      return memFootprint;
    }

    std::ostream& COOTensor3D::print(std::ostream& out) const{
      if(nzz == 0){
        out << "Empty COO Tensor" << std::endl;
        return out;
      }

      out << "  ";
      size_t p(0);
      for(size_t m(0); m<n; ++m){
        size_t k(p);
        for(size_t j(0); j<n; ++j){
          for(size_t i(0); i<n; ++i){
            if(row[k] == j && col[k] == i && slice[k] == m){
              out << val[k] << " ";
              ++k;
              ++p;
            }
            else{
              out << "_ ";
            }
          }
          out << std::endl << "  ";
        }
        out << std::endl << "  ";
      }
      out << std::endl;
      
      return out;
    }

    std::ostream& operator<<(std::ostream& out, COOTensor3D const& tensor_){
      return tensor_.print(out);
    }



/****************************************************
 *              COO Tensor 4D
 ****************************************************/
  /**
    * @brief Construct a new COOTensor4D::COOTensor4D object
    * 
    * @param denseTensor 
    * @param scaleMatrix 
    */
    COOTensor4D::COOTensor4D(std::vector< tensor4d > & denseTensor) : n(0){
      // Set device pointer to nullprt
      d_val = nullptr;
      d_row = nullptr;
      d_col = nullptr;
      d_slice = nullptr;
      d_hyperslice = nullptr;


      // Fill the COO Tensor with the values of the vector of the 4D dense tensor
      for(size_t l(0); l<denseTensor.size(); ++l){

        for(size_t hs(0); hs<denseTensor[l].size(); ++hs){
          for(size_t s(0); s<denseTensor[l][hs].size(); ++s){
            for(size_t i(0); i<denseTensor[l][hs][s].size(); ++i){
              for(size_t j(0); j<denseTensor[l][hs][s][i].size(); ++j){
                if(std::abs(denseTensor[l][hs][s][i][j]) > reel_eps){
                  row.push_back(i+n);
                  col.push_back(j+n);
                  slice.push_back(s+n);
                  hyperslice.push_back(hs+n);
                  val.push_back(denseTensor[l][hs][s][i][j]);
                }
              }
            }
          }
        }
        n += denseTensor[l].size();
      
      }
      nzz = val.size();

    }



  /**
  * @brief Destroy the COOTensor3D::COOTensor3D object
  * 
  */
    COOTensor4D::~COOTensor4D(){
      if(d_val != nullptr){
        CHECK_CUDA( cudaFree(d_val) );
      }
      if(d_row != nullptr){
        CHECK_CUDA( cudaFree(d_row) );
      }
      if(d_col != nullptr){
        CHECK_CUDA( cudaFree(d_col) );
      }
      if(d_slice != nullptr){
        CHECK_CUDA( cudaFree(d_slice) );
      }
      if(d_hyperslice != nullptr){
        CHECK_CUDA( cudaFree(d_hyperslice) );
      }
    }



  /**
   * @brief Extend the COO Tensor by appending n times the same tensor
   * 
   * @param nTimes 
   */
    uint COOTensor4D::ExtendTheSystem(uint nTimes){
      if(nTimes == 0){
        return n;
      }

      for(uint i(0); i<nTimes; ++i){
        for(uint j(0); j<nzz; ++j){
          row.push_back(row[j]+(i+1)*n);
          col.push_back(col[j]+(i+1)*n);
          slice.push_back(slice[j]+(i+1)*n);
          hyperslice.push_back(hyperslice[j]+(i+1)*n);
          val.push_back(val[j]);
        }
      }
      n += nTimes*n;
      nzz = val.size();

      return n;
    }



  /**
   * @brief Construct a new COOTensor3D::AllocateOnGPU object
   * 
   */
    void COOTensor4D::AllocateOnGPU(){
      // Allocate memory on the device
      CHECK_CUDA( cudaMalloc((void**)&d_row, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_col, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_slice, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_hyperslice, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_val, nzz*sizeof(reel)) );

      // Copy the data to the device
      CHECK_CUDA( cudaMemcpy(d_row, row.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_col, col.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_slice, slice.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_hyperslice, hyperslice.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_val, val.data(), nzz*sizeof(reel), cudaMemcpyHostToDevice) );
    }



    size_t COOTensor4D::memFootprint(){
      // Return the number of bytes needed to store this element on the GPU
      size_t memFootprint;

      memFootprint = 4*nzz*sizeof(uint) + nzz*sizeof(reel); 

      return memFootprint;
    }



    std::ostream& COOTensor4D::print(std::ostream& out) const{
      if(nzz == 0){
        out << "Empty COO Tensor" << std::endl;
        return out;
      }

      std::cout << "  val: ";
      for(size_t i(0); i<nzz; ++i){
        std::cout << val[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "  row: ";
      for(size_t i(0); i<nzz; ++i){
        std::cout << row[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "  col: ";
      for(size_t i(0); i<nzz; ++i){
        std::cout << col[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "  slice: ";
      for(size_t i(0); i<nzz; ++i){
        std::cout << slice[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "  hyperslice: ";
      for(size_t i(0); i<nzz; ++i){
        std::cout << hyperslice[i] << " ";
      }
      std::cout << std::endl;
      
      return out;
    }

    std::ostream& operator<<(std::ostream& out, COOTensor4D const& tensor_){
      return tensor_.print(out);
    }



/****************************************************
 *              COO Vector
 ****************************************************/
  /** COOVector::COOVector()
    * @brief Construct a new COOVector::COOVector object
    * 
    * @param denseVector 
    */  
    COOVector::COOVector(std::vector< std::vector<reel> > & denseVector) : n(0) {
      d_val = nullptr;
      d_indice = nullptr;

      for(size_t i(0); i<denseVector.size(); ++i){
        for(size_t j(0); j<denseVector[i].size(); ++j){
          if(std::abs(denseVector[i][j]) > reel_eps){
            indice.push_back(j+n);
            val.push_back(denseVector[i][j]);
          }
        }
        n += denseVector[i].size();

      }
      nzz = val.size();

    }


  /** COOVector::COOVector()
    * @brief Construct a new COOVector::COOVector object
    * 
    * @param denseVector 
    */  
    COOVector::~COOVector(){
      if(d_val != nullptr){
        CHECK_CUDA( cudaFree(d_val) );
      }
      if(d_indice != nullptr){
        CHECK_CUDA( cudaFree(d_indice) );
      }
    }

  uint COOVector::ExtendTheSystem(uint nTimes){
    if(nTimes == 0){
      return n;
    }

    for(uint i(0); i<nTimes; ++i){
      for(uint j(0); j<nzz; ++j){
        indice.push_back(indice[j]+(i+1)*n);
        val.push_back(val[j]);
      }
    }
    n += nTimes*n;
    nzz = val.size();

    return n;
  }



  void COOVector::AllocateOnGPU(){
    // Allocate memory on the device
    CHECK_CUDA( cudaMalloc((void**)&d_indice, nzz*sizeof(uint)) );
    CHECK_CUDA( cudaMalloc((void**)&d_val, nzz*sizeof(reel)) );

    // Copy the data to the device
    CHECK_CUDA( cudaMemcpy(d_indice, indice.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_val, val.data(), nzz*sizeof(reel), cudaMemcpyHostToDevice) );

    // Create the sparse vector descriptor
    CHECK_CUSPARSE( cusparseCreateSpVec(&sparseVec_desc, n, nzz, &d_indice, &d_val,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
  }



  size_t COOVector::memFootprint(){
    // Return the number of bytes needed to store this element on the GPU
    size_t memFootprint;

    memFootprint = nzz*sizeof(uint) + nzz*sizeof(reel); 

    return memFootprint;
  }



  std::ostream& COOVector::print(std::ostream& out) const{
    if(nzz == 0){
      out << "Empty COO Vector" << std::endl;
      return out;
    }

    out << "  val: ";
    size_t p(0);
    for(size_t i(0); i<n; ++i){
      if(indice[p] == i){
        out << val[p] << " ";
        ++p;
      }
      else{
        out << "_ ";
      }
    }
    out << std::endl;

    out << "  ind: ";
    p = 0;
    for(size_t i(0); i<n; ++i){
      if(indice[p] == i){
        out << indice[p] << " ";
        ++p;
      }
      else{
        out << "_ ";
      }
    }
    out << std::endl;

    return out;
  }



  std::ostream& operator<<(std::ostream& out, COOVector const& vector_){
    return vector_.print(out);
  }



/****************************************************
 *              Utilities
 ****************************************************/
  std::ostream& operator<<(std::ostream& out, matrix const& mat){
    // Print the row-major dense matrix in the output stream
    for(size_t i(0); i<mat.size(); ++i){
      for(size_t j(0); j<mat[i].size(); ++j){
        out << mat[i][j] << " ";
      }
      out << std::endl;
    }

    return out;
  }



  void printVector(std::vector<reel> & vec){
    std::cout << vec << std::endl;
  }



  template <typename T>
  std::ostream& operator<<(std::ostream& out, std::vector<T> const& vec){
    for(size_t i(0); i<vec.size(); ++i){
      out << vec[i] << " ";
    }
    out << std::endl;
    return out;
  }      



  uint extendTheVector(std::vector<reel> & vec, uint nTimes){
    uint n(vec.size());
    for(uint i(0); i<nTimes; ++i){
      for(uint j(0); j<n; ++j){
        vec.push_back(vec[j]);
      }
    }
    return vec.size();
  }



  