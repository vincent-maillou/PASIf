""" This file is the interface between the Python code 
and the C++/CUDA side. Mainly pre-processing is done here 
as well as wrapping somes functions in a more Pythonic way. """

from __future__  import annotations

# Ensure that the code has been compiled withe the
# latest version of the CUDA module. Lunch make.
import os
os.system("make")

# Path to the compiled CUDA module
import sys
sys.path.append('./build')

from PASIfgpu import __GpuDriver

# Standard libraries
import numpy as np
import copy  as cp
from   dataclasses import dataclass
from   typing      import Union
from scipy.sparse        import coo_matrix
from scipy.sparse        import dia_matrix
from scipy.sparse.linalg import inv


@dataclass
class coo_tensor:
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
    #   - Indices are stored sequentially in the following order (0-indexing):
    #       -> pt1_dim1, pt1_dim2, ..., pt1_dimN, pt2_dim1, pt2_dim2, ..., pt2_dimN, ...
    indices   : list[int]
    
    
    def concatenateTensor(self, tensor_ : coo_tensor):
        #                   ----- TODO -----
        if(self.dimensions != tensor_.dimensions):
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
        if len(diagMatrix_) != self.dimensions[0] or len(diagMatrix_) != self.dimensions[1]:
            raise Exception("The input matrix dimensions must match the row/col dimensions of the tensor")
        
        if len(self.dimensions) == 2:
            for i in range(len(self.val)):
                self.val[i] = self.val[i] * diagMatrix_[self.indices[i*len(self.dimensions)]]
        else:
            for i in range(len(self.val)):
                self.val[i] = self.val[i] * diagMatrix_[self.indices[i*len(self.dimensions)+len(self.dimensions)-1]]
        
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
                         dSolver_)

        self.sampleRate   = sampleRate
        self.globalSystemSize   = 0 # Number of DOFs
        self.globalAdjointSize  = 0 # Number of adjoint DOFs
        self.interpolSize = 0
        self.saveSteps    = 1
        
        self.systemSet       : bool = False
        self.numberOfSystems : int  = 0
        self.system_M                 : dia_matrix 
        self.system_B                 : coo_matrix 
        self.system_K                 : coo_matrix 
        self.system_Gamma             : coo_tensor # 3D tensor    
        self.system_Lambda            : coo_tensor # 4D tensor
        self.system_forcePattern      : np.ndarray 
        self.system_initialConditions : np.ndarray
        
        self.jacobianSet : bool = False
        self.jacobian_M                 : dia_matrix 
        self.jacobian_B                 : coo_matrix 
        self.jacobian_K                 : coo_matrix 
        self.jacobian_Gamma             : coo_tensor # 3D tensor    
        self.jacobian_Lambda            : coo_tensor # 4D tensor
        self.jacobian_Psi               : coo_tensor # 5D tensor
        self.jacobian_forcePattern      : np.ndarray 
        self.jacobian_initialConditions : np.ndarray

    def setExcitations(self, 
                       excitationSet: list[np.ndarray], 
                       sampleRate   : int):
        self.sampleRate = sampleRate
        self._loadExcitationsSet(excitationSet, self.sampleRate)
  
    def setSystems(self,
                   vecM                : list[dia_matrix],
                   vecB                : list[coo_matrix],
                   vecK                : list[coo_matrix],
                   vecGamma            : list[coo_tensor],
                   vecLambda           : list[coo_tensor],
                   vecForcePattern     : list[np.ndarray],
                   vecInitialConditions: list[np.ndarray]):
        
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
        dataB = [x for _, x in sorted(zip(self.system_B.row, self.system_B.data))]
        rowB  = [x for x, _ in sorted(zip(self.system_B.row, self.system_B.col))]
        colB  = [x for _, x in sorted(zip(self.system_B.row, self.system_B.col))]
        
        dataK = [x for _, x in sorted(zip(self.system_K.row, self.system_K.data))]
        rowK  = [x for x, _ in sorted(zip(self.system_K.row, self.system_K.col))]
        colK  = [x for _, x in sorted(zip(self.system_K.row, self.system_K.col))]
        
        self._setB(dataB, 
                   rowB, 
                   colB,
                   self.system_B.shape[0])
        self._setK(dataK, 
                   rowK, 
                   colK,
                   self.system_K.shape[0])
        self._setGamma(self.system_Gamma.dimensions, 
                       self.system_Gamma.val, 
                       self.system_Gamma.indices)
        self._setLambda(self.system_Lambda.dimensions,
                        self.system_Lambda.val,
                        self.system_Lambda.indices)
        self._setForcePattern(self.system_forcePattern)
        self._setInitialConditions(self.system_initialConditions)
        self.systemSet = True
        
        # Load the system on the GPU
        self._allocateOnDevice()
        
    def setJacobian(self,
                    vecM                : list[dia_matrix],
                    vecB                : list[coo_matrix],
                    vecK                : list[coo_matrix],
                    vecGamma            : list[coo_tensor],
                    vecLambda           : list[coo_tensor],
                    vecForcePattern     : list[np.ndarray],
                    vecInitialConditions: list[np.ndarray],
                    vecPsi              : list[coo_tensor]): 
        if self.systemSet == False:
            raise ValueError("The system must be set before the jacobian")
          
        """ # Check if the system is valid
        self.globalAdjointSize = 0  
        self.__checkSystemInput(vecM, 
                                vecB, 
                                vecK, 
                                vecGamma, 
                                vecLambda, 
                                vecForcePattern, 
                                vecInitialConditions, 
                                vecPsi)

        self.vecJacobM                  = cp.deepcopy(vecM)
        self.vecJacobB                  = cp.deepcopy(vecB)
        self.vecJacobK                  = cp.deepcopy(vecK)
        self.vecJacobGamma              = cp.deepcopy(vecGamma)
        self.vecJacobLambda             = cp.deepcopy(vecLambda)
        self.vecJacobPsi                = cp.deepcopy(vecPsi)
        self.vecJacobForcePattern       = cp.deepcopy(vecForcePattern)
        self.vecJacobInitialConditions  = cp.deepcopy(vecInitialConditions)

        # Pre-process the system
        self.__systemPreprocessing(self.vecJacobM, 
                                   self.vecJacobB, 
                                   self.vecJacobK, 
                                   self.vecJacobGamma, 
                                   self.vecJacobLambda, 
                                   self.vecJacobForcePattern, 
                                   self.vecJacobPsi)
        
        self.jacobianSet = True
        
        # Assemble the system and the jacobian in a single representation
        #self.__assembleSystemAndJacobian()
        
        # Load the system in the CPP side
        self._setB(self.vecJacobB)
        self._setK(self.vecJacobK)
        self._setGamma(self.vecJacobGamma)
        self._setLambda(self.vecJacobLambda)
        #self._setPsi(self.vecJacobPsi)
        self._setForcePattern(self.vecJacobForcePattern)
        self._setInitialConditions(self.vecJacobInitialConditions)
        
        # Load the system on the GPU
        self._allocateOnDevice() """
 
    def setInterpolationMatrix(self, 
                               interpolationMatrix_: list[vector]):
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
                            modulationBuffer_: vector):
        if(len(modulationBuffer_) == 0):
            raise ValueError("The modulation buffer must be non-empty.")
        elif(len(modulationBuffer_) > 32000):
            raise ValueError("The modulation buffer don't fit in the GPU cst memory.")

        self.modulationBuffer = modulationBuffer_

        self._setModulationBuffer(self.modulationBuffer)

    def getAmplitudes(self):
        if self.systemSet == False:
            raise ValueError("The system must be set before computing the amplitudes.")
        else:
            self._displaySimuInfos()
        
        return self._getAmplitudes()

    def getTrajectory(self, 
                      saveSteps: int = 1):
        if self.systemSet == False:
            raise ValueError("The system must be set before computing the trajectory.")
        else:
            self._displaySimuInfos()
        
        self.saveSteps = saveSteps
        trajectory = self._getTrajectory(saveSteps)
    
        # re-arrange the computed trajectory in a plotable way.
        numOfSavedSteps = int(self.numsteps*(self.interpolSize+1)/self.saveSteps)
        unwrappedTrajectory = np.array([np.zeros(numOfSavedSteps) for i in range(self.globalSystemSize + 1)])
    
        for t in range(numOfSavedSteps):
            # first row always contain time
            unwrappedTrajectory[0][t] = t*self.saveSteps/(self.sampleRate*(self.interpolSize+1))
            for i in range(self.globalSystemSize):
                unwrappedTrajectory[i+1][t] = trajectory[t*self.globalSystemSize + i]
    
        return unwrappedTrajectory
    
    def getGradient(self, save = 0):
        if self.jacobianSet == False:
            raise ValueError("The jacobian must be set before computing the gradient.")
        else:
            self._displaySimuInfos()
        
        gradient = self._getGradient(self.globalAdjointSize, save)
        
        
        
        chunkSize = 280
        numSetpoints = 279
        
        # re-arrange the computed trajectory in a plotable way.
        #numOfSavedSteps = int(self.numsteps*(self.interpolSize+1)/chunkSize)
        numOfSavedSteps = numSetpoints + chunkSize - 2 
        unwrappedGradient = np.array([np.zeros(numOfSavedSteps) for i in range(self.globalAdjointSize + 1)])
    
        for t in range(numOfSavedSteps):
            # first row always contain time
            unwrappedGradient[0][t] = t*numSetpoints/(self.sampleRate*(self.interpolSize+1))
            for i in range(self.globalAdjointSize):
                unwrappedGradient[i+1][t] = gradient[t*self.globalAdjointSize + i]
    
        return unwrappedGradient
        
        return gradient



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
        
        # Case of System setting
        for i in range(len(inputVecM)):
            ndofs = inputVecM[i].shape[0]
            
            assert inputVecM[i].shape[0] == inputVecM[i].shape[1], "The matrix M must be squared."
            assert inputVecB[i].shape[0] == inputVecB[i].shape[1], "The matrix B must be squared."
            assert inputVecK[i].shape[0] == inputVecK[i].shape[1], "The matrix K must be squared."
            assert inputVecGamma[i].dimensions[0]  == inputVecGamma[i].dimensions[1]  == inputVecGamma[i].dimensions[2], "The Gamma tensor must be squared."
            assert inputVecLambda[i].dimensions[0] == inputVecLambda[i].dimensions[1] == inputVecLambda[i].dimensions[2] == inputVecLambda[i].dimensions[3], "The Lambda tensor must be squared."
            
            assert inputVecB[i].shape[0]                 == ndofs, "The matrix B must have the same size as the matrix M."
            assert inputVecK[i].shape[0]                 == ndofs, "The matrix K must have the same size as the matrix M."
            assert inputVecGamma[i].dimensions[0]        == ndofs, "The Gamma tensor must have the same size as the matrix M."
            assert inputVecLambda[i].dimensions[0]       == ndofs, "The Lambda tensor must have the same size as the matrix M."
            assert inputVecForcePattern[i].shape[0]      == ndofs, "The force pattern vector must have the same size as the matrix M."
            assert inputVecInitialConditions[i].shape[0] == ndofs, "The initial conditions vector must have the same size as the matrix M."
            
            if inputVecPsi is not None:
                assert inputVecPsi[i].dimensions[0]    == ndofs, "The Psi tensor must have the same size as the matrix M."
                assert inputVecPsi[i].dimensions[0] == inputVecPsi[i].dimensions[1] == inputVecPsi[i].dimensions[2] == inputVecPsi[i].dimensions[3]  == inputVecPsi[i].dimensions[4], "The Psi tensor must be squared."


    def __unfoldSystems(self,
                        inputVecM                 : list[dia_matrix], 
                        inputVecB                 : list[coo_matrix], 
                        inputVecK                 : list[coo_matrix], 
                        inputVecGamma             : list[coo_tensor], 
                        inputVecLambda            : list[coo_tensor], 
                        inputVecForcePattern      : list[np.ndarray],
                        inputVecInitialConditions : list[np.ndarray]):
        
        numberOfSystems = len(inputVecM)
    
        self.globalSystemSize = inputVecM[0].shape[0]

        if numberOfSystems > 1:
            diagM : np.ndarray = inputVecM[0].data
            
            valB  : np.ndarray = inputVecB[0].data
            rowB  : np.ndarray = inputVecB[0].row
            colB  : np.ndarray = inputVecB[0].col
            
            valK  : np.ndarray = inputVecK[0].data
            rowK  : np.ndarray = inputVecK[0].row
            colK  : np.ndarray = inputVecK[0].col
            
            for i in range(1, numberOfSystems):
                diagM = np.concatenate((diagM, inputVecM[i].data)).flatten()
                
                valB = np.concatenate((valB, inputVecB[i].data))
                rowB = np.concatenate((rowB, inputVecB[i].row + self.globalSystemSize))
                colB = np.concatenate((colB, inputVecB[i].col + self.globalSystemSize))
                
                valK = np.concatenate((valK, inputVecK[i].data))
                rowK = np.concatenate((rowK, inputVecK[i].row + self.globalSystemSize))
                colK = np.concatenate((colK, inputVecK[i].col + self.globalSystemSize))
                
                inputVecGamma[0].concatenateTensor(inputVecGamma[i])
                inputVecLambda[0].concatenateTensor(inputVecLambda[i])
                inputVecForcePattern[0]      = np.concatenate((inputVecForcePattern[0], inputVecForcePattern[i]), axis = 0)
                inputVecInitialConditions[0] = np.concatenate((inputVecInitialConditions[0], inputVecInitialConditions[i]), axis = 0)
                
                self.globalSystemSize += inputVecM[i].shape[0]
            
            self.system_M = dia_matrix((diagM, [0]),         shape = (self.globalSystemSize, self.globalSystemSize))
            self.system_B = coo_matrix((valB, (rowB, colB)), shape = (self.globalSystemSize, self.globalSystemSize))
            self.system_K = coo_matrix((valK, (rowK, colK)), shape = (self.globalSystemSize, self.globalSystemSize))
            
        else:
            self.system_M = cp.deepcopy(inputVecM[0])
            self.system_B = cp.deepcopy(inputVecB[0])
            self.system_K = cp.deepcopy(inputVecK[0])
            
        self.system_Gamma             = cp.deepcopy(inputVecGamma[0])
        self.system_Lambda            = cp.deepcopy(inputVecLambda[0])
        self.system_forcePattern      = cp.deepcopy(inputVecForcePattern[0])
        self.system_initialConditions = cp.deepcopy(inputVecInitialConditions[0])
        
        
    def __unfoldJacobians(self,
                          input_cooVecM              : list[coo_tensor],
                          input_cooVecB              : list[coo_tensor],
                          input_cooVecK              : list[coo_tensor],
                          input_cooVecGamma          : list[coo_tensor],
                          input_cooVecLambda         : list[coo_tensor],
                          input_cooVecPsi            : list[coo_tensor],
                          input_vecForcePattern      : list[vector],
                          input_vecInitialConditions : list[vector]):
        
        if len(input_cooVecM) > 1:
            for i in range(1, len(input_cooVecM)):
                input_cooVecM[0].concatenateTensor(input_cooVecM[i])
                input_cooVecB[0].concatenateTensor(input_cooVecB[i])
                input_cooVecK[0].concatenateTensor(input_cooVecK[i])
                input_cooVecGamma[0].concatenateTensor(input_cooVecGamma[i])
                input_cooVecLambda[0].concatenateTensor(input_cooVecLambda[i])
                input_cooVecPsi[0].concatenateTensor(input_cooVecPsi[i])
                input_vecForcePattern[0]      += input_vecForcePattern[i]
                input_vecInitialConditions[0] += input_vecInitialConditions[i]
        
        self.jacobian_cooM              = cp.deepcopy(input_cooVecM[0])
        self.jacobian_cooB              = cp.deepcopy(input_cooVecB[0])
        self.jacobian_cooK              = cp.deepcopy(input_cooVecK[0])
        self.jacobian_cooGamma          = cp.deepcopy(input_cooVecGamma[0])  
        self.jacobian_cooLambda         = cp.deepcopy(input_cooVecLambda[0])
        self.jacobian_cooPsi            = cp.deepcopy(input_cooVecPsi[0])
        self.jacobian_forcePattern      = cp.deepcopy(input_vecForcePattern[0])
        self.jacobian_initialConditions = cp.deepcopy(input_vecInitialConditions[0])
    
    
    def __systemPreprocessing(self):
        
        self.system_M = -1*inv(self.system_M)
        self.system_B = coo_matrix(self.system_M.dot(self.system_B))
        self.system_K = coo_matrix(self.system_M.dot(self.system_K))
        
        self.system_Gamma.multiplyByDiagMatrix(self.system_M.data)
        self.system_Lambda.multiplyByDiagMatrix(self.system_M.data)
        
        self.system_forcePattern *= -1*self.system_M.data
        

                
                
                
                
    def __assembleSystemAndJacobian(self):
        # Work in progress
        """ for i in range(len(self.vecJacobM)):
            localSystemSize  = len(self.vecSystemM[i])
            localAdjointSize = len(self.vecJacobM[i])
            
            sysDim0 = np.zeros((localAdjointSize, localSystemSize))
            
            # Assemble the 2D matrix
            self.vecJacobB[i] = np.block([[self.vecSystemB[i]   , sysDim0],
                                          [np.transpose(sysDim0), self.vecJacobB[i]]])
            self.vecJacobK[i] = np.block([[self.vecSystemK[i]   , sysDim0],
                                          [np.transpose(sysDim0), self.vecJacobK[i]]])
            
            # Assemble the 3D tensor
            sysDim = localSystemSize + localAdjointSize
            sysDim20 = np.zeros((sysDim, sysDim))
            
            tempGamma = []
            for j in range(sysDim):
                tempGamma.append(np.block([[self.vecSystemGamma[i][j], sysDim0],
                                           [sysDim0                  , sysDim0]]))
            for j in range(sysDim):
                tempGamma.append(np.block([[sysDim0, sysDim0],
                                           [sysDim0, self.vecJacobGamma[i][j]]]))
            
            # Assemble the 4D tensor
            
            
            # Assemble the 5D tensor
            
            
            # Assemble the Force Pattern
            self.vecJacobForcePattern[i]      = np.concatenate((self.vecSystemForcePattern[i], self.vecJacobForcePattern[i]))
            
            # Assemble the Initial Conditions
            self.vecJacobInitialConditions[i] = np.concatenate((self.vecSystemInitialConditions[i], self.vecJacobInitialConditions[i])) """
            
            