""" This file is the interface between the Python code 
and the C++/CUDA side. Mainly pre-processing is done here 
as well as wrapping somes functions in a more Pythonic way. """

from __future__  import annotations

# Ensure that the code has been compiled withe the
# latest version of the CUDA module. Lunch make.
import os
#os.system("make")

# Path to the compiled CUDA module
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/build')

from PASIfgpu import __GpuDriver

# Standard libraries
import numpy as np
import copy  as cp
from   dataclasses         import dataclass
from   typing              import Union
from   scipy.sparse        import coo_matrix
from   scipy.sparse        import dia_matrix
from   scipy.sparse.linalg import inv
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

@dataclass
class coo_tensor:
    """_summary_ : Describe a sparse tensor in a COO format
    
       _description_ : The coo_tensor is described as higher-dimmension major 
    format with the following ordering: N-dims, N-dims-1, ..., slices, rows, columns.
    - For example a 2D tensor (matrices) is described as: rows-major format.
    - For example a 3D tensor is described as: slices-major format.. and so on.
    """
    def __init__(self, dimensions_: list[int] = [2, 2]):
        self.dimensions = dimensions_
        self.val        = []
        self.indices    = []
        
    def __str__(self) -> str:
        # Print the sparse tensor
        tensorString = "\n"
        currentValue = 0
        if len(self.dimensions) == 3:
            # Explicit print for 3D tensors
            for k in range(self.dimensions[0]):
                for i in range(self.dimensions[1]):
                    for j in range(self.dimensions[2]):
                        if currentValue < len(self.val) and self.indices[3*currentValue] == i and self.indices[3*currentValue+1] == j and self.indices[3*currentValue+2] == k:
                            tensorString += str(self.val[currentValue]) + " "
                            currentValue += 1
                        else:
                            tensorString += "_" + " "
                    tensorString += "\n "
                tensorString += "\n "
        else:
            # Else print the tensor in COO format
            for i in range(len(self.val)):
                tensorString += "val: "
                tensorString += str(self.val[i]) + " "
            tensorString += "\n"
            for i in range(len(self.dimensions)):
                for j in range(len(self.val)):
                    tensorString += "dim" + str(i+1) + ": "
                    tensorString += str(self.indices[j*len(self.dimensions)+i]) + " "
                tensorString += "\n"
        return tensorString
        
        
    # Describe any dimension tensor in COO format
    #   - List the size of each dimension of the tensor. 
    #   - Length of the list is the number of dimensions
    dimensions: list[int]
    #   - List of values of the tensor in a row major order (smaller dimension first)
    val       : list[float]
    #   - Indices are stored sequentially in the higher-dimmension major order
    indices   : list[int]
    
    
    def concatenateTensor(self, tensor_ : coo_tensor):
        #                   ----- TODO -----
        if(len(self.dimensions) != len(tensor_.dimensions)):
            raise Exception("The number of dimensions of the two tensors are not the same")

        # Concatenate the values of the two tensors
        self.val = self.val + tensor_.val
        
        # Concatenate the indices and extend the indices of the second tensor
        for i in range(len(tensor_.indices)):
            self.indices.append(tensor_.indices[i] + self.dimensions[i%len(self.dimensions)])
            
        # Update the dimensions of the tensor
        for i in range(len(self.dimensions)):
            self.dimensions[i] += tensor_.dimensions[i]    
        
    def multiplyByDiagMatrix(self, diagMatrix_: list):
        # Check input tensor dimensions
        
        """ print()
        
        if len(diagMatrix_) != self.dimensions[-2] or len(diagMatrix_) != self.dimensions[-1]:
            raise Exception("The input matrix dimensions must match the two smallest (row/col) dimensions of the tensor")
         """
        if len(self.dimensions) == 2:
            for i in range(len(self.val)):
                self.val[i] = self.val[i] * diagMatrix_[self.indices[i*len(self.dimensions)]]
        else:
            for i in range(len(self.val)):
                self.val[i] = self.val[i] * diagMatrix_[self.indices[i*len(self.dimensions)]]

    def getIndicesDim(self, dim_: int) -> list[int]:
        # Unfold the indices list of the tensor and return the indices  
        # of the dimension dim_ in a list format.
        unfoldedIndices = []
        
        nDim = len(self.dimensions)
        for i in range(len(self.val)):
            unfoldedIndices.append(self.indices[dim_+i*nDim])
        
        return unfoldedIndices
    
    def isSquare(self) -> bool:
        for i in range(len(self.dimensions)):
            if(self.dimensions[i] != self.dimensions[0]):
                return False
        return True
    
    def extendDimmensions(self, extension_: int):
        # Extend the dimensions of the tensor by extension_
        for i in range(len(self.dimensions)):
            self.dimensions[i] += extension_
            
    def offsetDimmension(self,
                          targetDim_ : int,
                          offset_    : int):
        # Add to the indices of the targetDim_ dimension the offset_ value
        # - Used to offset a tensor in a specific dimension
        for i in range(len(self.indices)):
            if i % len(self.dimensions) == targetDim_:
                self.indices[i] += offset_

class PASIf(__GpuDriver):
    def __init__(self, 
                 excitationSet: list[np.ndarray], 
                 sampleRate   : int, 
                 numsteps_    : int  = 0, 
                 dCompute_    : bool = False,
                 dSystem_     : bool = False,
                 dSolver_     : bool = False):
        if(numsteps_ == 0):
            self.numsteps = len(excitationSet[0])
        else:
            self.numsteps = numsteps_
            
        super().__init__(excitationSet, 
                         sampleRate, 
                         self.numsteps,
                         dCompute_,
                         dSystem_,
                         dSolver_,
                         RANK)

        self.sampleRate   = sampleRate
        self.globalSystemSize   = 0 # Number of DOFs
        self.globalAdjointSize  = 0 # Number of adjoint DOFs
        self.interpolSize = 0
        self.saveSteps    = 1
        self.n_excitations = len(excitationSet)
        
        self.systemSet       : bool = False
        self.numberOfSystems : int  = 0
        self.system_M                   : dia_matrix = None 
        self.system_B                   : coo_matrix = None 
        self.system_K                   : coo_matrix = None 
        self.system_Gamma               : coo_tensor = None # 3D tensor    
        self.system_Lambda              : coo_tensor = None # 4D tensor
        self.system_forcePattern        : np.ndarray = None 
        self.system_initialConditions   : np.ndarray = None
        
        self.jacobianSet : bool = False
        self.jacobian_M                 : dia_matrix = None
        self.jacobian_B                 : coo_matrix = None 
        self.jacobian_K                 : coo_matrix = None 
        self.jacobian_Gamma             : coo_tensor = None # 3D tensor    
        self.jacobian_Lambda            : coo_tensor = None # 4D tensor
        self.jacobian_forcePattern      : np.ndarray = None 
        self.jacobian_initialConditions : np.ndarray = None
        self.jacobian_Psi               : coo_tensor = None # 5D tensor

    def setExcitations(self, 
                       excitationSet: list[np.ndarray], 
                       sampleRate   : int):
        self.sampleRate = sampleRate
        self._loadExcitationsSet(excitationSet, self.sampleRate)
  
    def setSystems(self,
                   vecM                 : list[dia_matrix],
                   vecB                 : list[coo_matrix],
                   vecK                 : list[coo_matrix],
                   vecGamma             : list[coo_tensor],
                   vecLambda            : list[coo_tensor],
                   vecForcePattern      : list[np.ndarray],
                   vecInitialConditions : list[np.ndarray]):
        
        self.systemSet = False
        
        assert len(vecM) > 0
        assert len(vecM) == len(vecB) == len(vecK) == len(vecGamma) == len(vecLambda) == len(vecForcePattern) == len(vecInitialConditions), "The number of Systems in the input vectors must be the same."
        assert (type(vecM[0]) == dia_matrix and 
                type(vecB[0]) == coo_matrix and 
                type(vecK[0]) == coo_matrix and 
                type(vecGamma[0])  == coo_tensor and 
                type(vecLambda[0]) == coo_tensor and 
                type(vecForcePattern[0])      == np.ndarray and 
                type(vecInitialConditions[0]) == np.ndarray), "Inputs vectors types error."
        
        # Check the validity of the input system (ie. size consistency)
        self.globalSystemSize = 0
        self.__checkInputs(vecM, 
                           vecB, 
                           vecK, 
                           vecGamma, 
                           vecLambda, 
                           vecForcePattern, 
                           vecInitialConditions)

        
        # Unfold each of the input vectors of COO systems into a single COO system
        self.__unfoldSystems(vecM,
                             vecB,
                             vecK,
                             vecGamma,
                             vecLambda,
                             vecForcePattern,
                             vecInitialConditions)
        
        
        # Pre-process the system (ie. multiply per -1*M^-1)
        self.__systemPreprocessing()


        # Convert the Column-major COO system to Row-major COO system

        csr_K = (self.system_B + self.system_K).tocsr()
        csr_K.sort_indices()
        self._setFwdK(self.system_K.shape,
                      csr_K.data, 
                      csr_K.indices, 
                      csr_K.indptr)

        # for i, val in enumerate(self.system_Gamma.val):
            # print(f'Val {val}, indices: {self.system_Gamma.indices[i*3:(i+1)*3]}')

        self._setFwdGamma(self.system_Gamma.dimensions, 
                          self.system_Gamma.val, 
                          self.system_Gamma.indices)

        # for i, val in enumerate(self.system_Lambda.val):
        #     print(f'Val {val}, indices: {self.system_Lambda.indices[i*4:(i+1)*4]}')

        self._setFwdLambda(self.system_Lambda.dimensions,
                           self.system_Lambda.val,
                           self.system_Lambda.indices)

        self._setFwdPsi([1]*5, [], [])
        self._setFwdForcePattern(self.system_forcePattern)
        self._setFwdInitialConditions(self.system_initialConditions)
        self.systemSet = True
        
        # Load the system on the GPU
        self._allocateSystemOnDevice()
        
    def setJacobian(self,
                    vecM                : list[dia_matrix],
                    vecB                : list[coo_matrix],
                    vecK                : list[coo_matrix],
                    vecGamma            : list[coo_tensor],
                    vecLambda           : list[coo_tensor],
                    vecForcePattern     : list[np.ndarray],
                    vecInitialConditions: list[np.ndarray],
                    vecPsi              : list[coo_tensor]): 
        
        self.jacobianSet       = False
        self.globalAdjointSize = 0
        
        if self.systemSet == False:
            raise ValueError("The system must be set before the jacobian")
          
        assert len(vecM) > 0
        assert len(vecM) == self.numberOfSystems, "The number of jacobian must be the same as the number of systems."
        assert len(vecM) == len(vecB) == len(vecK) == len(vecGamma) == len(vecLambda) == len(vecForcePattern) == len(vecInitialConditions), "The number of Systems in the input vectors must be the same."
        assert (type(vecM[0]) == dia_matrix and 
                type(vecB[0]) == coo_matrix and 
                type(vecK[0]) == coo_matrix and 
                type(vecGamma[0])  == coo_tensor and 
                type(vecLambda[0]) == coo_tensor and 
                type(vecForcePattern[0])      == np.ndarray and 
                type(vecInitialConditions[0]) == np.ndarray and
                type(vecPsi[0]) == coo_tensor), "Inputs vectors types error."  


        self.__checkInputs(vecM, 
                           vecB, 
                           vecK, 
                           vecGamma, 
                           vecLambda, 
                           vecForcePattern, 
                           vecInitialConditions,
                           vecPsi)
        
        self.__unfoldJacobians(vecM, 
                               vecB, 
                               vecK, 
                               vecGamma, 
                               vecLambda, 
                               vecForcePattern, 
                               vecInitialConditions,
                               vecPsi)

        self.__jacobianPreprocessing() 
        self.jacobianSet = True     
        jac_csr_K = (self.jacobian_B+self.jacobian_K).tocsr()
        jac_csr_K.sort_indices()
        
        # print()
        # print()
        # print("K")
        # print(jac_csr_K)
        self._setBwdK(self.jacobian_K.shape,
                      jac_csr_K.data, 
                      jac_csr_K.indices, 
                      jac_csr_K.indptr)

        # print()
        # print()
        # print("Gamma")
        # for i, v in enumerate(self.jacobian_Gamma.val):
        #     print(f'Val {v} at {self.jacobian_Gamma.indices[3*i:3*(i+1)]}')

        self._setBwdGamma(self.jacobian_Gamma.dimensions, 
                          self.jacobian_Gamma.val, 
                          self.jacobian_Gamma.indices)

        # print()
        # print()
        # print("Lambda")
        # for i, v in enumerate(self.jacobian_Lambda.val):
        #     print(f'Val {v} at {self.jacobian_Lambda.indices[4*i:4*(i+1)]}')

        self._setBwdLambda(self.jacobian_Lambda.dimensions,
                           self.jacobian_Lambda.val,
                           self.jacobian_Lambda.indices)

        self._setBwdForcePattern(self.jacobian_forcePattern)
        self._setBwdInitialConditions(self.jacobian_initialConditions)

        self._setBwdPsi(self.jacobian_Psi.dimensions,
                    self.jacobian_Psi.val,
                    self.jacobian_Psi.indices)
        self.jacobianSet = True
        
        # Load the system on the GPU
        self._allocateAdjointOnDevice()

    def setInterpolationMatrix(self, 
                               interpolationMatrix_: list[np.ndarray]):
        # Verify that each row of the interpolation matrix are even and of the same size
        for i in range(len(interpolationMatrix_)):
            if(len(interpolationMatrix_[i])%2 != 0):
                raise ValueError("The windows size must be even.")
            elif(len(interpolationMatrix_[i]) != len(interpolationMatrix_[0])):
                raise ValueError("The windows size must be the same for all the rows.")  

        self.interpolSize = len(interpolationMatrix_)

        # Modify the matrix into a single vector
        self.interpolationMatrix = np.array(interpolationMatrix_).flatten()  

        self._setInterpolationMatrix(self.interpolationMatrix, len(interpolationMatrix_[0]))

    def setModulationBuffer(self, 
                            modulationBuffer_: np.ndarray):
        if(len(modulationBuffer_) == 0):
            raise ValueError("The modulation buffer must be non-empty.")
        elif(len(modulationBuffer_) > 32000):
            raise ValueError("The modulation buffer don't fit in the GPU cst memory.")

        self.modulationBuffer = modulationBuffer_

        self._setModulationBuffer(self.modulationBuffer)

    def getAmplitudes(self):
        if self.systemSet == False:
            raise ValueError("The system must be set before computing the amplitudes.")
        
        return self._getAmplitudes()

    def getTrajectory(self, 
                      saveSteps: int = 1):
        if self.systemSet == False:
            raise ValueError("The system must be set before computing the trajectory.")
        
        self.saveSteps = saveSteps
        trajectory = self._getTrajectory(saveSteps)
        # re-arrange the computed trajectory in a plotable way.
        numOfSavedSteps = int(self.numsteps/self.saveSteps)
        unwrappedTrajectory = np.array([np.zeros(numOfSavedSteps) for i in range(self.globalSystemSize*self.n_excitations + 1)])
    
        for t in range(numOfSavedSteps):
            # first row always contain time
            unwrappedTrajectory[0][t] = t*self.saveSteps/self.sampleRate
            for i in range(self.globalSystemSize*self.n_excitations):
                unwrappedTrajectory[i+1][t] = trajectory[t*self.globalSystemSize*self.n_excitations + i]
    
        return unwrappedTrajectory
    
    def getGradient(self,):
        if self.jacobianSet == False:
            raise ValueError("The jacobian must be set before computing the gradient.")

        res =  self._getGradient(0)
        # print(res)
        return res

    def getMatrices(self):
        if not self.systemSet: raise ValueError("System is not set yet")
        K_assembled = (self.system_B + self.system_K)
        return K_assembled, self.system_Gamma, self.system_Lambda, self.system_forcePattern, self.system_initialConditions

    ############################################################# 
    #                      Private methods
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def __checkInputs(self,
                      inputVecM                 : list[dia_matrix], 
                      inputVecB                 : list[coo_matrix], 
                      inputVecK                 : list[coo_matrix], 
                      inputVecGamma             : list[coo_tensor], 
                      inputVecLambda            : list[coo_tensor], 
                      inputVecForcePattern      : list[np.ndarray],
                      inputVecInitialConditions : list[np.ndarray],
                      inputVecPsi               : list[coo_tensor] = None):
        """
        Check for size consistency of the input matrices and vectors.
        """
        
        for i in range(len(inputVecM)):
            ndofs = inputVecM[i].shape[0]
            
            assert inputVecM[i].shape[0] == inputVecM[i].shape[1], "The matrix M must be squared."
            assert inputVecB[i].shape[0] == inputVecB[i].shape[1], "The matrix B must be squared."
            assert inputVecK[i].shape[0] == inputVecK[i].shape[1], "The matrix K must be squared."
            if inputVecPsi is None:
                assert inputVecGamma[i].dimensions[0]  == inputVecGamma[i].dimensions[1]  == inputVecGamma[i].dimensions[2], "The System Gamma tensor must be squared."
                assert inputVecLambda[i].dimensions[0] == inputVecLambda[i].dimensions[1] == inputVecLambda[i].dimensions[2] == inputVecLambda[i].dimensions[3], "The System Lambda tensor must be squared."
            
            assert inputVecB[i].shape[0]                 == ndofs, "The matrix B must have the same size as the matrix M."
            assert inputVecK[i].shape[0]                 == ndofs, "The matrix K must have the same size as the matrix M."
            assert inputVecGamma[i].dimensions[0]        == ndofs, "The Gamma tensor must have the same size as the matrix M."
            assert inputVecLambda[i].dimensions[0]       == ndofs, "The Lambda tensor must have the same size as the matrix M."
            assert inputVecForcePattern[i].shape[0]      == ndofs, "The force pattern vector must have the same size as the matrix M."
            assert inputVecInitialConditions[i].shape[0] == ndofs, "The initial conditions vector must have the same size as the matrix M."
            
            if inputVecPsi is not None:
                assert inputVecPsi[i].dimensions[0] == ndofs, "The Psi tensor must have the same size as the matrix M."
                assert inputVecPsi[i].dimensions[0] == inputVecPsi[i].dimensions[1], "The two highest dim of the Psi tensor should be of the size of the adjoint system."
                assert inputVecPsi[i].dimensions[2] == inputVecPsi[i].dimensions[3]  == inputVecPsi[i].dimensions[4], "The tree smallest dim of the Psi tensor should be of the size of the forward system."


    def __unfoldSystems(self,
                        inputVecM                 : list[dia_matrix], 
                        inputVecB                 : list[coo_matrix], 
                        inputVecK                 : list[coo_matrix], 
                        inputVecGamma             : list[coo_tensor], 
                        inputVecLambda            : list[coo_tensor], 
                        inputVecForcePattern      : list[np.ndarray],
                        inputVecInitialConditions : list[np.ndarray]):
        """
        This method is used to unfold the systems parsed as a list of systems 
        into a single system. The resulting system will be stored in the class
        to be later concatenated with the adjoint system.
        """
        
        self.numberOfSystems = len(inputVecM)
    
        self.globalSystemSize = inputVecM[0].shape[0]

        if self.numberOfSystems > 1:
            diagM : np.ndarray = inputVecM[0].data
            
            valB  : np.ndarray = inputVecB[0].data
            rowB  : np.ndarray = inputVecB[0].row
            colB  : np.ndarray = inputVecB[0].col
            
            valK  : np.ndarray = inputVecK[0].data
            rowK  : np.ndarray = inputVecK[0].row
            colK  : np.ndarray = inputVecK[0].col
            
            self.system_Gamma  = cp.deepcopy(inputVecGamma[0])
            self.system_Lambda = cp.deepcopy(inputVecLambda[0])
            
            for i in range(1, self.numberOfSystems):
                diagM = np.concatenate((diagM, inputVecM[i].data))
                
                valB = np.concatenate((valB, inputVecB[i].data))
                rowB = np.concatenate((rowB, inputVecB[i].row + self.globalSystemSize))
                colB = np.concatenate((colB, inputVecB[i].col + self.globalSystemSize))
                
                valK = np.concatenate((valK, inputVecK[i].data))
                rowK = np.concatenate((rowK, inputVecK[i].row + self.globalSystemSize))
                colK = np.concatenate((colK, inputVecK[i].col + self.globalSystemSize))
                
                self.system_Gamma.concatenateTensor(inputVecGamma[i])
                self.system_Lambda.concatenateTensor(inputVecLambda[i])
                inputVecForcePattern[0]      = np.concatenate((inputVecForcePattern[0], inputVecForcePattern[i]), axis = 0)
                inputVecInitialConditions[0] = np.concatenate((inputVecInitialConditions[0], inputVecInitialConditions[i]), axis = 0)
                
                self.globalSystemSize += inputVecM[i].shape[0]
            

            diagM         = diagM.reshape((self.globalSystemSize))
            self.system_M = dia_matrix((diagM, [0]),         shape = (self.globalSystemSize, self.globalSystemSize))
            self.system_B = coo_matrix((valB, (rowB, colB)), shape = (self.globalSystemSize, self.globalSystemSize))
            self.system_K = coo_matrix((valK, (rowK, colK)), shape = (self.globalSystemSize, self.globalSystemSize))
            
        else:
            self.system_M = cp.deepcopy(inputVecM[0])
            self.system_B = cp.deepcopy(inputVecB[0])
            self.system_K = cp.deepcopy(inputVecK[0])
            
            self.system_Gamma  = cp.deepcopy(inputVecGamma[0])
            self.system_Lambda = cp.deepcopy(inputVecLambda[0])

        #self.globalSystemSize *= self.n_excitations
        #TODO: check here

        self.system_forcePattern      = cp.deepcopy(inputVecForcePattern[0])
        self.system_initialConditions = cp.deepcopy(inputVecInitialConditions[0])
 
    def __unfoldJacobians(self,
                          inputVecM                 : list[dia_matrix], 
                          inputVecB                 : list[coo_matrix], 
                          inputVecK                 : list[coo_matrix], 
                          inputVecGamma             : list[coo_tensor], 
                          inputVecLambda            : list[coo_tensor], 
                          inputVecForcePattern      : list[np.ndarray],
                          inputVecInitialConditions : list[np.ndarray],
                          inputVecPsi               : list[coo_tensor]):
        """
        This method is used to unfold the jacobians system parsed as a lists
        into a single system. Each of the jacobians correspond to their respective 
        system that has been stored previously.
        """
        
        self.globalAdjointSize = inputVecM[0].shape[0]

        if self.numberOfSystems > 1:
            diagM : np.ndarray = inputVecM[0].data
            
            valB  : np.ndarray = inputVecB[0].data
            rowB  : np.ndarray = inputVecB[0].row
            colB  : np.ndarray = inputVecB[0].col
            
            valK  : np.ndarray = inputVecK[0].data
            rowK  : np.ndarray = inputVecK[0].row
            colK  : np.ndarray = inputVecK[0].col
            
            for i in range(1, self.numberOfSystems):
                diagM = np.concatenate((diagM, inputVecM[i].data)).flatten()
                
                valB = np.concatenate((valB, inputVecB[i].data))
                rowB = np.concatenate((rowB, inputVecB[i].row + self.globalAdjointSize))
                colB = np.concatenate((colB, inputVecB[i].col + self.globalAdjointSize))
                
                valK = np.concatenate((valK, inputVecK[i].data))
                rowK = np.concatenate((rowK, inputVecK[i].row + self.globalAdjointSize))
                colK = np.concatenate((colK, inputVecK[i].col + self.globalAdjointSize))
                
                inputVecGamma[0].concatenateTensor(inputVecGamma[i])
                inputVecLambda[0].concatenateTensor(inputVecLambda[i])
                inputVecPsi[0].concatenateTensor(inputVecPsi[i])
                inputVecForcePattern[0]      = np.concatenate((inputVecForcePattern[0], inputVecForcePattern[i]), axis = 0)
                inputVecInitialConditions[0] = np.concatenate((inputVecInitialConditions[0], inputVecInitialConditions[i]), axis = 0)
                
                self.globalAdjointSize += inputVecM[i].shape[0]
            
            self.jacobian_M = dia_matrix((diagM, [0]),         shape = (self.globalAdjointSize, self.globalAdjointSize))
            self.jacobian_B = coo_matrix((valB, (rowB, colB)), shape = (self.globalAdjointSize, self.globalAdjointSize))
            self.jacobian_K = coo_matrix((valK, (rowK, colK)), shape = (self.globalAdjointSize, self.globalAdjointSize))
            
        else:
            self.jacobian_M = cp.deepcopy(inputVecM[0])
            self.jacobian_B = cp.deepcopy(inputVecB[0])
            self.jacobian_K = cp.deepcopy(inputVecK[0])
            
        self.jacobian_Gamma             = cp.deepcopy(inputVecGamma[0])
        self.jacobian_Lambda            = cp.deepcopy(inputVecLambda[0])
        self.jacobian_Psi               = cp.deepcopy(inputVecPsi[0])
        self.jacobian_forcePattern      = cp.deepcopy(inputVecForcePattern[0])
        self.jacobian_initialConditions = cp.deepcopy(inputVecInitialConditions[0])
    
    def __systemPreprocessing(self):
        
        self.system_M = -1.*inv(self.system_M)
        self.system_B = coo_matrix(self.system_M.dot(self.system_B))
        self.system_K = coo_matrix(self.system_M.dot(self.system_K))
        
        self.system_Gamma.multiplyByDiagMatrix(self.system_M.data)
        self.system_Lambda.multiplyByDiagMatrix(self.system_M.data)
        
        # print(self.system_M.data)
        
        self.system_forcePattern *= -1.*self.system_M.data
        
    def __jacobianPreprocessing(self):

        # print()
        # print()
        # print("M")
        # print(self.jacobian_M.data)

        self.jacobian_M = -1*inv(self.jacobian_M)
        self.jacobian_B = coo_matrix(self.jacobian_M.dot(self.jacobian_B))
        self.jacobian_K = coo_matrix(self.jacobian_M.dot(self.jacobian_K)) 
        self.jacobian_Gamma.multiplyByDiagMatrix(self.jacobian_M.data)
        self.jacobian_Lambda.multiplyByDiagMatrix(self.jacobian_M.data)
        self.jacobian_Psi.multiplyByDiagMatrix(self.jacobian_M.data)
        #self.jacobian_forcePattern *= -1*self.jacobian_M.data