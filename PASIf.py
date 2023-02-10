""" This file is the interface between the Python code 
and the C++/CUDA side. Mainly pre-processing is done here 
as well as wrapping somes functions in a more Pythonic way. """



# Import modules
import numpy as np

# Ensure that the code has been compiled withe the
# latest version of the CUDA module. Lunch make.
import os
os.system("make")

# Path to the compiled CUDA module
import sys
sys.path.append('./build')

from PASIfgpu import __GpuDriver



class PASIf(__GpuDriver):
    def __init__(self, 
                 excitationSet: list[ list ], 
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
        
        self.systemSet    = False
        self.vecSystemM = []
        self.vecSystemB = []
        self.vecSystemK = []
        self.vecSystemGamma  = [] # Order 3 tensors     
        self.vecSystemLambda = [] # Order 4 tensors
        self.vecSystemForcePattern      = []
        self.vecSystemInitialConditions = []
        
        self.jacobianSet  = False
        self.vecJacobM = []
        self.vecJacobB = []
        self.vecJacobK = []
        self.vecJacobGamma  = [] # Order 3 tensors  
        self.vecJacobLambda = [] # Order 4 tensors
        self.vecJacobPsi    = [] # Order 5 tensors
        self.vecJacobForcePattern      = []
        self.vecJacobInitialConditions = []

    def setExcitations(self, 
                       excitationSet: list[list], 
                       sampleRate   : int):
        self.sampleRate = sampleRate
        self._loadExcitationsSet(excitationSet, self.sampleRate)
  
    def setSystems(self,
                   vecM                : list[ list[list] ],
                   vecB                : list[ list[list] ],
                   vecK                : list[ list[list] ],
                   vecGamma            : list[ list[list[list]] ],
                   vecLambda           : list[ list[list[list[list]]] ],
                   vecForcePattern     : list[ list ],
                   vecInitialConditions: list[ list ]):
        # Check if the system is valid
        self.__checkSystemInput(vecM, vecB, vecK, vecGamma, vecLambda, vecForcePattern, vecInitialConditions)

        # Pre-process the system
        self.__systemPreprocessing(vecM, vecB, vecK, vecGamma, vecLambda, vecForcePattern)

        # Store localy the setted system to later define the jacobian
        self.vecSystemB                   = vecB
        self.vecSystemK                   = vecK
        self.vecSystemGamma               = vecGamma
        self.vecSystemLambda              = vecLambda
        self.vecSystemForcePattern        = vecForcePattern
        self.vecSystemInitialConditions   = vecInitialConditions
        
        self.systemSet = True

        # Load the system in the GPU
        self._setB(self.vecSystemB)
        self._setK(self.vecSystemK)
        self._setGamma(self.vecSystemGamma)
        self._setLambda(self.vecSystemLambda)
        self._setForcePattern(self.vecSystemForcePattern)
        self._setInitialConditions(self.vecSystemInitialConditions)
        
    def setJacobian(self,
                    vecM                : list[ list[list] ],
                    vecB                : list[ list[list] ],
                    vecK                : list[ list[list] ],
                    vecGamma            : list[ list[list[list]] ],
                    vecLambda           : list[ list[list[list[list]]] ],
                    vecPsi              : list[ list[list[list[list[list]]]] ],
                    vecForcePattern     : list[ list ],
                    vecInitialConditions: list[ list ]): 
        if self.systemSet == False:
            raise ValueError("The system must be set before the jacobian")
          
        # Check if the system is valid
        self.__checkSystemInput(vecM, vecB, vecK, vecGamma, vecLambda, vecForcePattern, vecInitialConditions, vecPsi)

        # Pre-process the system
        self.__systemPreprocessing(vecM, vecB, vecK, vecGamma, vecLambda, vecForcePattern, vecPsi)
        
        self.vecJacobB                  = vecB
        self.vecJacobK                  = vecK
        self.vecJacobGamma              = vecGamma
        self.vecJacobLambda             = vecLambda
        self.vecJacobPsi                = vecPsi
        self.vecJacobForcePattern       = vecForcePattern
        self.vecJacobInitialConditions  = vecInitialConditions
        
        self.jacobianSet = True
        
        # Assemble the system and the jacobian in a single representation
        self.__assembleSystemAndJacobian()
        
        # Load the system in the GPU
        self._setB(self.vecJacobB)
        self._setK(self.vecJacobK)
        self._setGamma(self.vecJacobGamma)
        self._setLambda(self.vecJacobLambda)
        """ self._setPsi(self.vecJacobPsi) """
        self._setForcePattern(self.vecJacobForcePattern)
        self._setInitialConditions(self.vecJacobInitialConditions)
 
    def setInterpolationMatrix(self, 
                               interpolationMatrix_: list[ list ]):
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
                            modulationBuffer_: list):
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
    
        numOfSavedSteps = int(self.numsteps*(self.interpolSize+1)/self.saveSteps)
        unwrappedTrajectory = np.array([np.zeros(numOfSavedSteps) for i in range(self.globalSystemSize + 1)])
    
        for t in range(numOfSavedSteps):
            unwrappedTrajectory[0][t] = t*self.saveSteps/(self.sampleRate*(self.interpolSize+1))
            for i in range(self.globalSystemSize):
                unwrappedTrajectory[i+1][t] = trajectory[t*self.globalSystemSize + i]
    
        return unwrappedTrajectory
    
    """ def getGradient(self):
        if self.jacobianSet == False:
            raise ValueError("The jacobian must be set before computing the gradient.")
        
        return self._getGradient() """


    
    ############################################################# 
    #                      Private methods
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def __checkSystemInput(self,
                           vecM                : list[ list[list] ],
                           vecB                : list[ list[list] ],
                           vecK                : list[ list[list] ],
                           vecGamma            : list[ list[list[list]] ],
                           vecLambda           : list[ list[list[list[list]]] ],
                           vecForcePattern     : list[ list ],
                           vecInitialConditions: list[ list ],
                           vecPsi              : list[ list[list[list[list[list]]]] ] = None):
        # Check the number of system in all of the inputs vectors
        if vecPsi == None:
            # Case of System setting
            if len(vecM) != len(vecB) or len(vecM) != len(vecK) or len(vecM) != len(vecGamma) or len(vecM) != len(vecLambda) or len(vecM) != len(vecForcePattern) or len(vecM) != len(vecInitialConditions):
                raise ValueError("The number of Systems in the input vectors must be the same.")
        else:
            # Case of Jacobian setting, also check the size of the Jacobian against the size of the System
            if len(self.vecSystemM) != len(vecM) or len(self.vecSystemM) != len(vecB) or len(self.vecSystemM) != len(vecK) or len(self.vecSystemM) != len(vecGamma) or len(self.vecSystemM) != len(vecLambda) or len(self.vecSystemM) != len(vecForcePattern) or len(self.vecSystemM) != len(vecInitialConditions):
                raise ValueError("The number of Jacobian in the input vectors must be the same and match the number of setted Systems.")

        # Check that the matrix of each system are of the same size
        if vecPsi == None:
            for i in range(len(vecM)):
                if len(vecM[i]) != len(vecB[i]) or len(vecM[i]) != len(vecK[i]) or len(vecM[i]) != len(vecGamma[i]) or len(vecM[i]) != len(vecLambda[i]) or len(vecM[i]) != len(vecForcePattern[i]) or len(vecM[i]) != len(vecInitialConditions[i]):
                    raise ValueError("The dimension of each System must be the same.")
                self.globalSystemSize += len(vecM[i])
        else:
            for i in range(len(vecM)):
                if len(vecM[i]) != len(vecB[i]) or len(vecM[i]) != len(vecK[i]) or len(vecM[i]) != len(vecGamma[i]) or len(vecM[i]) != len(vecLambda[i]) or len(vecM[i]) != len(vecForcePattern[i]) or len(vecM[i]) != len(vecInitialConditions[i] or len(vecM[i]) != len(vecPsi[i])):
                    raise ValueError("The dimension of each Jacobian must be the same.")
                self.globalAdjointSize += len(vecM[i])   
    
    def __systemPreprocessing(self,
                              vecM           : list[ list[list] ],
                              vecB           : list[ list[list] ],
                              vecK           : list[ list[list] ],
                              vecGamma       : list[ list[list[list]] ],
                              vecLambda      : list[ list[list[list[list]]] ],
                              vecForcePattern: list[ list ],
                              vecPsi         : list[ list[list[list[list[list]]]] ] = None):
        # Invert the M matrix and then pre-multiply the others
        for i in range(len(vecM)):
            vecM[i] = np.linalg.inv(vecM[i])
            vecM[i] = vecM[i]
            vecB[i] = -1 * np.matmul(vecM[i], vecB[i])
            vecK[i] = -1 * np.matmul(vecM[i], vecK[i])
            vecGamma[i] = np.einsum('ij, jkl -> ikl', vecM[i], vecGamma[i])
            vecLambda[i] = -1 * np.einsum('ij, jklm -> iklm', vecM[i], vecLambda[i])
            vecForcePattern[i] = np.diag(vecM[i]) * vecForcePattern[i]
            
            if vecPsi != None:
                vecPsi[i] = -1 * np.einsum('ij, jklmn -> iklmn', vecM[i], vecPsi[i])
                
    def __assembleSystemAndJacobian(self):
        for i in range(len(self.vecJacobM)):
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
            self.vecJacobInitialConditions[i] = np.concatenate((self.vecSystemInitialConditions[i], self.vecJacobInitialConditions[i]))
            
            