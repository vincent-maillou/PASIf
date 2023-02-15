""" This file is the interface between the Python code 
and the C++/CUDA side. Mainly pre-processing is done here 
as well as wrapping somes functions in a more Pythonic way. """

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
from dataclasses import dataclass
from typing      import Union



@dataclass
class cooTensor:
    def __init__(self, dimensions_: list[int] = [2, 2]):
        self.dimensions = dimensions_
        self.val        = []
        self.indices    = []
        
    # Describe any dimension tensor in COO format
    #   - List the size of each dimension of the tensor. 
    #   - Length of the list is the number of dimensions
    dimensions: list[int]
    #   - List of values of the tensor in a row major order (smaller dimension first)
    val       : list[float]
    #   - Indices are stored sequentially in the following order (0-indexing):
    #       -> pt1_dim1, pt1_dim2, ..., pt1_dimN, pt2_dim1, pt2_dim2, ..., pt2_dimN, ...
    indices   : list[int]
    
    def isSquare(self) -> bool:
        for i in range(len(self.dimensions)):
            if(self.dimensions[i] != self.dimensions[0]):
                return False
        return True



# Type aliases for dense representation of parameters tensors
vector   = list[float]
matrix   = list[vector]
tensor3d = list[matrix]
tensor4d = list[tensor3d]
tensor5d = list[tensor4d]



class PASIf(__GpuDriver):
    def __init__(self, 
                 excitationSet: list[vector], 
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
        self.system_cooM                 : cooTensor # 2D tensor
        self.system_cooB                 : cooTensor # 2D tensor
        self.system_cooK                 : cooTensor # 2D tensor
        self.system_cooGamma             : cooTensor # 3D tensor    
        self.system_cooLambda            : cooTensor # 4D tensor
        self.system_forcePattern         : vector 
        self.system_initialConditions    : vector
        
        self.jacobianSet : bool = False
        self.jacobian_cooM                 : cooTensor # 2D tensor
        self.jacobian_cooB                 : cooTensor # 2D tensor
        self.jacobian_cooK                 : cooTensor # 2D tensor
        self.jacobian_cooGamma             : cooTensor # 3D tensor    
        self.jacobian_cooLambda            : cooTensor # 4D tensor
        self.jacobian_cooPsi               : cooTensor # 5D tensor   
        self.jacobian_forcePattern         : vector 
        self.jacobian_initialConditions    : vector

    def setExcitations(self, 
                       excitationSet: list[vector], 
                       sampleRate   : int):
        self.sampleRate = sampleRate
        self._loadExcitationsSet(excitationSet, self.sampleRate)
  
    def setSystems(self,
                   vecM                : Union[list[matrix],   list[cooTensor]],
                   vecB                : Union[list[matrix],   list[cooTensor]],
                   vecK                : Union[list[matrix],   list[cooTensor]],
                   vecGamma            : Union[list[tensor3d], list[cooTensor]],
                   vecLambda           : Union[list[tensor4d], list[cooTensor]],
                   vecForcePattern     : list[vector],
                   vecInitialConditions: list[vector]):
        # Check non-empty input
        if(len(vecM) == 0 or len(vecB) == 0 or len(vecK) == 0 or len(vecGamma) == 0 or len(vecLambda) == 0 or len(vecForcePattern) == 0 or len(vecInitialConditions) == 0):
            raise ValueError("At least one of the input system is empty.")
        
        
        # Check and convert the input system to COO format if needed,
        # then check the validity of the system.
        cooInputVecM      : list[cooTensor] = []
        cooInputVecB      : list[cooTensor] = []
        cooInputVecK      : list[cooTensor] = []
        cooInputVecGamma  : list[cooTensor] = []
        cooInputVecLambda : list[cooTensor] = []
        
        if(type(vecM[0]) == matrix):
            cooInputVecM = self.__convertToCoo(vecM)
        elif(type(vecM[0]) == cooTensor):
            cooInputVecM = vecM
        else:
            raise TypeError("Input system M is not a list of matrices or a list of COO tensors.")    
        
        if(type(vecB[0]) == matrix):
            cooInputVecB = self.__convertToCoo(vecB)
        elif(type(vecB[0]) == cooTensor):
            cooInputVecB = vecB
        else:
            raise TypeError("Input system B is not a list of matrices or a list of COO tensors.")    
            
        if(type(vecK[0]) == matrix):
            cooInputVecK = self.__convertToCoo(vecK)
        elif(type(vecK[0]) == cooTensor):
            cooInputVecK = vecK
        else:
            raise TypeError("Input system K is not a list of matrices or a list of COO tensors.")    
            
        if(type(vecGamma[0]) == tensor3d):
            cooInputVecGamma = self.__convertToCoo(vecGamma)
        elif(type(vecGamma[0]) == cooTensor):
            cooInputVecGamma = vecGamma
        else:
            raise TypeError("Input system Gamma is not a list of matrices or a list of COO tensors.")    
            
        if(type(vecLambda[0]) == tensor4d):
            cooInputVecLambda = self.__convertToCoo(vecLambda)
        elif(type(vecLambda[0]) == cooTensor):
            cooInputVecLambda = vecLambda
        else:
            raise TypeError("Input system Lambda is not a list of matrices or a list of COO tensors.")    
                
        
        # Check the validity of the input system (ie. size consistency)
        self.globalSystemSize = 0
        self.__checkSystemInput(cooInputVecM, 
                                cooInputVecB, 
                                cooInputVecK, 
                                cooInputVecGamma, 
                                cooInputVecLambda, 
                                vecForcePattern, 
                                vecInitialConditions)
        
        
        # Unfold the input systems into a single coo descriptor
        self.__unfoldSystems(cooInputVecM        , self.system_cooM,
                             cooInputVecB        , self.system_vecB,
                             cooInputVecK        , self.system_vecK,
                             cooInputVecGamma    , self.system_cooGamma, 
                             cooInputVecLambda   , self.system_cooLambda,
                             vecForcePattern     , self.system_forcePattern,
                             vecInitialConditions, self.system_initialConditions)
        
        
        # Pre-process the system (ie. multiply per -1*M^-1)
        self.__systemPreprocessing(self.system_cooM,
                                   self.system_cooB,
                                   self.system_cooK,
                                   self.system_cooGamma, 
                                   self.system_cooLambda,
                                   self.system_forcePattern)
        self.systemSet = True
        
        # WORK IN PROGRESS ------------------------------------------------------------
        cooB = cooTensor(dimensions_ = [6, 6])
        cooB.val     = [1, 1, 1, -1, -1]
        cooB.indices = [0,3 , 1,4 , 2,5 , 3,3 , 4,4]

        cooK = cooTensor(dimensions_ = [6, 6])
        cooK.val     = [-6, -1]
        cooK.indices = [3,0 , 4,1]
        
        cooGamma = cooTensor(dimensions_ = [6, 6, 6])
        cooGamma.val      = [10, 1, 1]
        cooGamma.indices  = [0,1,3 , 0,0,4 , 1,1,5]

        cooLambda = cooTensor(dimensions_ = [6, 6, 6, 6])
        cooLambda.val     = [-4000]
        cooLambda.indices = [1,1,1,4]
        
        unfoldedForcePattern = []
        for i in range(len(self.vecSystemForcePattern)):
            for j in range(len(self.vecSystemForcePattern[i])):
                unfoldedForcePattern.append(self.vecSystemForcePattern[i][j])
        
        unfoldedInitialConditions = []
        for i in range(len(self.vecSystemInitialConditions)):
            for j in range(len(self.vecSystemInitialConditions[i])):
                unfoldedInitialConditions.append(self.vecSystemInitialConditions[i][j])
        # WORK IN PROGRESS ------------------------------------------------------------
        
        """ self._setB(cooB.dimensions, 
                   cooB.val, 
                   cooB.indices)
        self._setK(cooK.dimensions, 
                   cooK.val, 
                   cooK.indices)
        self._setGamma(cooGamma.dimensions, 
                       cooGamma.val, 
                       cooGamma.indices)
        self._setLambda(cooLambda.dimensions,
                        cooLambda.val,
                        cooLambda.indices)
        self._setForcePattern(unfoldedForcePattern)
        self._setInitialConditions(unfoldedInitialConditions)
        
        # Load the system on the GPU
        self._allocateOnDevice() """
        
    def setJacobian(self,
                    vecM                : list[matrix],
                    vecB                : list[matrix],
                    vecK                : list[matrix],
                    vecGamma            : list[tensor3d],
                    vecLambda           : list[tensor4d],
                    vecForcePattern     : list[vector],
                    vecInitialConditions: list[vector],
                    vecPsi              : list[tensor5d]): 
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
    def __checkSystemInput(self,
                           cooInputVecM         : list[cooTensor], 
                           cooInputVecB         : list[cooTensor], 
                           cooInputVecK         : list[cooTensor], 
                           cooInputVecGamma     : list[cooTensor], 
                           cooInputVecLambda    : list[cooTensor], 
                           vecForcePattern      : list[vector],
                           vecInitialConditions : list[vector],
                           cooInputVecPsi       : list[cooTensor] = None):
        # Case of System setting
        if type(cooInputVecPsi) != list:
            # Check the number of System in all of the inputs vectors
            if(len(cooInputVecM) != len(cooInputVecB)      or  
               len(cooInputVecM) != len(cooInputVecK)      or 
               len(cooInputVecM) != len(cooInputVecGamma)  or 
               len(cooInputVecM) != len(cooInputVecLambda) or 
               len(cooInputVecM) != len(vecForcePattern)   or 
               len(cooInputVecM) != len(vecInitialConditions)):
                raise ValueError("The number of Systems in the input vectors must be the same.")

            # Check that the matrix of each System are squared and of the same size
            for i in range(len(cooInputVecM)):
                if(cooInputVecM[i].isSquare()      == False or
                   cooInputVecB[i].isSquare()      == False or
                   cooInputVecK[i].isSquare()      == False or
                   cooInputVecGamma[i].isSquare()  == False or
                   cooInputVecLambda[i].isSquare() == False or
                   cooInputVecM[i].dimensions[0]   != cooInputVecB[i].dimensions[0]      or
                   cooInputVecM[i].dimensions[0]   != cooInputVecK[i].dimensions[0]      or
                   cooInputVecM[i].dimensions[0]   != cooInputVecGamma[i].dimensions[0]  or
                   cooInputVecM[i].dimensions[0]   != cooInputVecLambda[i].dimensions[0] or
                   cooInputVecM[i].dimensions[0]   != len(vecForcePattern[i])            or
                   cooInputVecM[i].dimensions[0]   != len(vecInitialConditions[i])):
                    raise ValueError("The dimension of each System must be the same.")
                self.globalSystemSize += len(cooInputVecM[i])    
            self.numberOfSystems = len(cooInputVecM) 
        
        else:
            # Check the number of Jacobian in all of the inputs vectors and against the number of setted Systems
            if(len(self.numberOfSystems) != len(cooInputVecM)      or  
               len(self.numberOfSystems) != len(cooInputVecB)      or 
               len(self.numberOfSystems) != len(cooInputVecK)      or 
               len(self.numberOfSystems) != len(cooInputVecGamma)  or 
               len(self.numberOfSystems) != len(cooInputVecLambda) or 
               len(self.numberOfSystems) != len(vecForcePattern)   or 
               len(self.numberOfSystems) != len(vecInitialConditions)):
                raise ValueError("The number of Jacobian in the input vectors must be the same and match the number of setted Systems.")

            # Check that the matrix of each System are squared and of the same size
            for i in range(len(cooInputVecM)):
                if(cooInputVecM[i].isSquare()      == False or
                   cooInputVecB[i].isSquare()      == False or
                   cooInputVecK[i].isSquare()      == False or
                   cooInputVecM[i].dimensions[0]   != cooInputVecB[i].dimensions[0]      or
                   cooInputVecM[i].dimensions[0]   != cooInputVecK[i].dimensions[0]      or
                   cooInputVecM[i].dimensions[0]   != cooInputVecGamma[i].dimensions[0]  or
                   cooInputVecM[i].dimensions[0]   != cooInputVecLambda[i].dimensions[0] or
                   cooInputVecM[i].dimensions[0]   != cooInputVecPsi[i].dimensions[0]    or
                   cooInputVecM[i].dimensions[0]   != len(vecForcePattern[i])            or
                   cooInputVecM[i].dimensions[0]   != len(vecInitialConditions[i])):
                    raise ValueError("The dimension of each Jacobian must be the same.")
                self.globalAdjointSize += len(cooInputVecM[i])   


    
    def __unfoldSystems(self,
                        input_cooVecM              : list[cooTensor]       , output_cooM              : cooTensor,
                        input_cooVecB              : list[cooTensor]       , output_cooB              : cooTensor,
                        input_cooVecK              : list[cooTensor]       , output_cooK              : cooTensor,
                        input_cooVecGamma          : list[cooTensor]       , output_cooGamma          : cooTensor,
                        input_cooVecLambda         : list[cooTensor]       , output_cooLambda         : cooTensor,
                        input_vecForcePattern      : list[vector]          , output_forcePattern      : vector,
                        input_vecInitialConditions : list[vector]          , output_initialConditions : vector,
                        input_cooVecPsi            : list[cooTensor] = None, output_cooPsi            : cooTensor = None):
        
        
        
        """ self.system_cooM                 : cooTensor # 2D tensor
        self.system_cooB                 : cooTensor # 2D tensor
        self.system_cooK                 : cooTensor # 2D tensor
        self.system_cooGamma             : cooTensor # 3D tensor    
        self.system_cooLambda            : cooTensor # 4D tensor
        self.system_forcePattern         : vector 
        self.system_initialConditions    : vector """
    
    
    
    
    
    
    
    
    
    def __systemPreprocessing(self,
                              cooM           : cooTensor,
                              cooB           : cooTensor,
                              cooK           : cooTensor,
                              cooGamma       : cooTensor,
                              cooLambda      : cooTensor,
                              forcePattern   : vector,
                              cooPsi         : cooTensor = None):
        
        
        
        
        
        print("vecM[0]: ", vecM[0])
        print("vecB[0]: ", vecB[0])
        print("vecK[0]: ", vecK[0])
        print("vecGamma[0]: ", vecGamma[0])
        print("vecLambda[0]: ", vecLambda[0])
        print("vecForcePattern[0]: ", vecForcePattern[0])
        
        # Invert the M matrix and then pre-multiply the others
        for i in range(len(vecM)):
            vecM[i] = np.linalg.inv(vecM[i])
            vecB[i] = -1 * np.matmul(vecM[i], vecB[i])
            vecK[i] = -1 * np.matmul(vecM[i], vecK[i])
            vecGamma[i]  = np.einsum('ij, jkl -> ikl', vecM[i], vecGamma[i])
            vecLambda[i] = -1 * np.einsum('ij, jklm -> iklm', vecM[i], vecLambda[i])
            vecForcePattern[i] = np.diag(vecM[i]) * vecForcePattern[i]
            
            if type(vecPsi) == np.ndarray:
                vecPsi[i] = -1 * np.einsum('ij, jklmn -> iklmn', vecM[i], vecPsi[i])
                
                
                
                
                
                
                
                
                
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
            
            