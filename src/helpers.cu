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
 *              CSR Matrix
 ****************************************************/
  /**
  * @brief Construct a new CSRMatrix::CSRMatrix object
  * 
  * @param denseMatrix 
  * @param scaleMatrix 
  */
    CSRMatrix::CSRMatrix(std::array<uint,2> n_,
                         std::vector<reel>  values_,
                         std::vector<uint>  indices_,
                         std::vector<uint>  indptr_):
      n(n_),
      alpha(1),
      beta(0){

      // Set device pointer to nullprt
      d_val = nullptr;
      d_indices = nullptr;
      d_indptr = nullptr;
      d_vec = nullptr;
      d_res = nullptr;

      d_buffer = nullptr;
      bufferSize = 0;

      d_alpha = nullptr;
      d_beta = nullptr;

      for(size_t i(0); i<values_.size(); ++i){
        val.push_back(values_[i]);
        indices.push_back(indices_[i]);
      }
      for(size_t i(0); i<n[0]+1; ++i){
        indptr.push_back(indptr_[i]);
        vec.push_back(0);
      }
      nzz = val.size();

      /* std::cout << "val" << std::endl;
      for(size_t i(0); i<val.size(); ++i){
        std::cout << val[i] << std::endl;
      }

      std::cout << "row" << std::endl;
      for(size_t i(0); i<row.size(); ++i){
        std::cout << row[i] << std::endl;
      }

      std::cout << "col" << std::endl;
      for(size_t i(0); i<col.size(); ++i){
        std::cout << col[i] << std::endl;
      } */
    }



  /**
  * @brief Destroy the CSRMatrix::CSRMatrix object
  * 
  */
    CSRMatrix::~CSRMatrix(){
      if(d_val != nullptr){
        CHECK_CUDA( cudaFree(d_val) );
      }
      if(d_indices != nullptr){
        CHECK_CUDA( cudaFree(d_indices) );
      }
      if(d_indptr != nullptr){
        CHECK_CUDA( cudaFree(d_indptr) );
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
      if(d_vec != nullptr){
        CHECK_CUDA( cudaFree(d_vec) );
      }
      if(d_res != nullptr){
        CHECK_CUDA( cudaFree(d_res) );
      }
    }

  /**
   * @brief Extend the COO Matrix by appending n times the same matrix
   * 
   * @param n 
   */
    uint CSRMatrix::extendTheSystem(uint nTimes){
      // Return the highest dimmention of the matrix
      // after the extension
      ntimes = nTimes;

      for(uint l=1; l<ntimes; l+=1){
          for(uint i=0; i<n[0]; i+=1){
            vec.push_back(0);
          }
      }

      return n[0]*ntimes;
    }



  /**
   * @brief Construct a new CSRMatrix::allocateOnGPU object
   * 
   */
    void CSRMatrix::allocateOnGPU(cusparseHandle_t     & handle){
      // Allocate memory on the device
      CHECK_CUDA( cudaMalloc((void**)&d_indices, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_indptr, (n[0]+1)*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_val, nzz*sizeof(reel)) );
      CHECK_CUDA( cudaMalloc((void**)&d_vec, ntimes*n[0]*sizeof(reel)) );
      CHECK_CUDA( cudaMalloc((void**)&d_res, ntimes*n[0]*sizeof(reel)) );

      // Copy the data to the device
      CHECK_CUDA( cudaMemcpy(d_indices, indices.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_indptr, indptr.data(), (n[0]+1)*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_val, val.data(), nzz*sizeof(reel), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_vec, vec.data(), ntimes*n[0]*sizeof(reel), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_res, vec.data(), ntimes*n[0]*sizeof(reel), cudaMemcpyHostToDevice) );      

      // Create the sparse matrix descriptor and allocate the needed buffer
      CHECK_CUSPARSE( cusparseCreateConstCsr(&sparseMat_desc, 
                                        n[0], 
                                        n[1], 
                                        nzz, 
                                        d_indptr, 
                                        d_indices, 
                                        d_val, 
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_32I, 
                                        CUSPARSE_INDEX_BASE_ZERO, 
                                        CUDA_R_32F) )

      CHECK_CUSPARSE( cusparseCreateDnMat(&denseMat_desc, 
                                        n[0], 
                                        ntimes, 
                                        n[0], 
                                        d_vec, 
                                        CUDA_R_32F, 
                                        CUSPARSE_ORDER_COL) )

      CHECK_CUSPARSE( cusparseCreateDnMat(&resMat_desc, 
                                        n[0],
                                        ntimes, 
                                        n[0], 
                                        d_vec, 
                                        CUDA_R_32F,
                                        CUSPARSE_ORDER_COL) );
      
      CHECK_CUDA( cudaMalloc((void**)&d_alpha, sizeof(reel)) );
      CHECK_CUDA( cudaMalloc((void**)&d_beta,  sizeof(reel)) );

      CHECK_CUSPARSE( cusparseSpMM_bufferSize(handle, 
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &d_alpha, 
                                              sparseMat_desc, 
                                              denseMat_desc, 
                                              &d_beta, 
                                              resMat_desc, 
                                              CUDA_R_32F, 
                                              CUSPARSE_SPMM_CSR_ALG1, 
                                              &bufferSize) )

      CHECK_CUDA( cudaMalloc((void**)&d_buffer, bufferSize) );

      CHECK_CUSPARSE(cusparseSpMM_preprocess(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &d_alpha,
                                            sparseMat_desc,
                                            denseMat_desc,
                                            &d_beta,
                                            resMat_desc,
                                            CUDA_R_32F,
                                            CUSPARSE_SPMM_CSR_ALG1,
                                            d_buffer));
    }

    size_t CSRMatrix::memFootprint(){
      // Return the number of bytes needed to store this element on the GPU
      size_t memFootprint;

      memFootprint = bufferSize + (nzz + n[0]+1)*sizeof(uint) + nzz*sizeof(reel) + n[0]*ntimes*sizeof(reel)*2;

      return memFootprint;
    }
  
    std::ostream& CSRMatrix::print(std::ostream& out) const{
      // Print the sparse COO matrix in a readable format
      if(nzz == 0){
        out << "Empty matrix" << std::endl;
        return out;
      }

      size_t index(0); // Keep track of the column index
      // Loop through each row
      for (size_t row(0); row < n[0]; ++row) {
          size_t row_start(indptr[row]);
          size_t row_end(indptr[row + 1]);

          // Loop through each column of the current row
          for (size_t col(0); col < n[1]; ++col) {
              if (index>=row_start && index < row_end && indices[index] == col) {
                  // The current position has a non-zero element
                  out << val[index] << " ";
                  index++; // Move to the next non-zero element
              } else {
                  // The current position has a zero element
                  out << "- ";
              }
          }
          out << "\n"; // Move to the next row
      }
      out << std::endl;
      
      return out;
    }

    std::ostream& operator<<(std::ostream& out, CSRMatrix const& mat){
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
    COOTensor3D::COOTensor3D(std::array<uint, 3> n_,
                             std::vector<reel>   values_,
                             std::vector<uint>   indices_) : 
      n(n_){
      // Set device pointer to nullptr
      d_val   = nullptr;
      d_row   = nullptr;
      d_col   = nullptr;
      d_slice = nullptr;

      nzz = values_.size();
      for(size_t i(0); i<nzz; ++i){
        val.push_back(values_[i]);
        slice.push_back(indices_[3*i]);
        row.push_back(indices_[3*i+1]);
        col.push_back(indices_[3*i+2]);
      }
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
    uint COOTensor3D::extendTheSystem(uint nTimes){
      ntimes = nTimes;

      return ntimes*n[0];
    }

  /**
   * @brief Construct a new COOTensor3D::allocateOnGPU object
   * 
   */
    void COOTensor3D::allocateOnGPU(){
      // Allocate memory on the device
      CHECK_CUDA( cudaMalloc((void**)&d_row,   nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_col,   nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_slice, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_val,   nzz*sizeof(reel)) );

      // Copy the data to the device
      CHECK_CUDA( cudaMemcpy(d_val,   val.data(),   nzz*sizeof(reel), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_slice, slice.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_row,   row.data(),   nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_col,   col.data(),   nzz*sizeof(uint), cudaMemcpyHostToDevice) );

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
      for(size_t m(0); m<n[0]; ++m){
        size_t k(p);
        for(size_t j(0); j<n[1]; ++j){
          for(size_t i(0); i<n[2]; ++i){
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
    COOTensor4D::COOTensor4D(std::array<uint, 4> n_,
                             std::vector<reel> values_,
                             std::vector<uint> indices_) : 
      n(n_){
      // Set device pointer to nullprt
      d_val        = nullptr;
      d_hyperslice = nullptr;
      d_slice      = nullptr;
      d_row        = nullptr;
      d_col        = nullptr;
      
      nzz = values_.size();
      for(size_t i(0); i<nzz; ++i){
        val.push_back(values_[i]);
        hyperslice.push_back(indices_[4*i]);
        slice.push_back(indices_[4*i+1]);
        row.push_back(indices_[4*i+2]);
        col.push_back(indices_[4*i+3]);
      }
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
    uint COOTensor4D::extendTheSystem(uint nTimes){
      ntimes = nTimes;
      return ntimes*n[0];
    }

  /**
   * @brief Construct a new COOTensor3D::allocateOnGPU object
   * 
   */
    void COOTensor4D::allocateOnGPU(){
      // Allocate memory on the device
      CHECK_CUDA( cudaMalloc((void**)&d_row,        nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_col,        nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_slice,      nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_hyperslice, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_val,        nzz*sizeof(reel)) );

      // Copy the data to the device
      CHECK_CUDA( cudaMemcpy(d_val,        val.data(),        nzz*sizeof(reel), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_hyperslice, hyperslice.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_slice,      slice.data(),      nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_row,        row.data(),        nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_col,        col.data(),        nzz*sizeof(uint), cudaMemcpyHostToDevice) );
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
      std::cout << "  hyperslice: ";
      for(size_t i(0); i<nzz; ++i){
        std::cout << hyperslice[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "  slice: ";
      for(size_t i(0); i<nzz; ++i){
        std::cout << slice[i] << " ";
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
      
      return out;
    }

    std::ostream& operator<<(std::ostream& out, COOTensor4D const& tensor_){
      return tensor_.print(out);
    }



/****************************************************
 *              COO Tensor 5D
 ****************************************************/
  /**
    * @brief Construct a new COOTensor5D::COOTensor5D object
    * 
    * @param denseTensor 
    * @param scaleMatrix 
    */
    COOTensor5D::COOTensor5D(std::array<uint, 5> n_,
                             std::vector<reel> values_,
                             std::vector<uint> indices_) : 
      n(n_){
      // Set device pointer to nullprt
      d_val             = nullptr;
      d_hyperhyperslice = nullptr;
      d_hyperslice      = nullptr;
      d_slice           = nullptr;
      d_row             = nullptr;
      d_col             = nullptr;
      
      nzz = values_.size();
      for(size_t i(0); i<nzz; ++i){
        val.push_back(values_[i]);
        hyperhyperslice.push_back(indices_[5*i]);
        hyperslice.push_back(indices_[5*i+1]);
        slice.push_back(indices_[5*i+2]);
        row.push_back(indices_[5*i+3]);
        col.push_back(indices_[5*i+4]);
      }
    }

  /**
  * @brief Destroy the COOTensor3D::COOTensor3D object
  * 
  */
    COOTensor5D::~COOTensor5D(){
      if(d_val != nullptr){
        CHECK_CUDA( cudaFree(d_val) );
      }
      if(d_hyperhyperslice != nullptr){
        CHECK_CUDA( cudaFree(d_hyperhyperslice) );
      }
      if(d_hyperslice != nullptr){
        CHECK_CUDA( cudaFree(d_hyperslice) );
      }
      if(d_slice != nullptr){
        CHECK_CUDA( cudaFree(d_slice) );
      }
      if(d_row != nullptr){
        CHECK_CUDA( cudaFree(d_row) );
      }
      if(d_col != nullptr){
        CHECK_CUDA( cudaFree(d_col) );
      }
    }

  /**
   * @brief Extend the COO Tensor by appending n times the same tensor
   * 
   * @param nTimes 
   */
    uint COOTensor5D::extendTheSystem(uint nTimes){
      ntimes = nTimes;
      return ntimes*n[0];
    }

  /**
   * @brief Construct a new COOTensor5D::allocateOnGPU object
   * 
   */
    void COOTensor5D::allocateOnGPU(){
      // Allocate memory on the device
      CHECK_CUDA( cudaMalloc((void**)&d_hyperhyperslice, nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_hyperslice,      nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_slice,           nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_row,             nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_col,             nzz*sizeof(uint)) );
      CHECK_CUDA( cudaMalloc((void**)&d_val,             nzz*sizeof(reel)) );

      // Copy the data to the device
      CHECK_CUDA( cudaMemcpy(d_val,             val.data(),             nzz*sizeof(reel), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_hyperhyperslice, hyperhyperslice.data(), nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_hyperslice,      hyperslice.data(),      nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_slice,           slice.data(),           nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_row,             row.data(),             nzz*sizeof(uint), cudaMemcpyHostToDevice) );
      CHECK_CUDA( cudaMemcpy(d_col,             col.data(),             nzz*sizeof(uint), cudaMemcpyHostToDevice) );
    }

    size_t COOTensor5D::memFootprint(){
      // Return the number of bytes needed to store this element on the GPU
      size_t memFootprint;

      memFootprint = 5*nzz*sizeof(uint) + nzz*sizeof(reel); 

      return memFootprint;
    }

    std::ostream& COOTensor5D::print(std::ostream& out) const{
      if(nzz == 0){
        out << "Empty COO Tensor" << std::endl;
        return out;
      }

      std::cout << "  val: ";
      for(size_t i(0); i<nzz; ++i){
        std::cout << val[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "  hyperhyperslice: ";
      for(size_t i(0); i<nzz; ++i){
        std::cout << hyperhyperslice[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "  hyperslice: ";
      for(size_t i(0); i<nzz; ++i){
        std::cout << hyperslice[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "  slice: ";
      for(size_t i(0); i<nzz; ++i){
        std::cout << slice[i] << " ";
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
      
      return out;
    }

    std::ostream& operator<<(std::ostream& out, COOTensor5D const& tensor_){
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
    COOVector::COOVector(std::vector<reel> & denseVector_) : n(0) {
      d_val = nullptr;
      d_indice = nullptr;

      n = denseVector_.size();
      for(size_t i(0); i<denseVector_.size(); ++i){
        if(std::abs(denseVector_[i]) > reel_eps){
          indice.push_back(i);
          val.push_back(denseVector_[i]);
        }
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

  uint COOVector::extendTheSystem(uint nTimes){
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

  void COOVector::allocateOnGPU(){
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