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
  def __init__(self, excitationSet, sampleRate):
    super().__init__(excitationSet, sampleRate)



  def setExcitations(self, excitationSet, sampleRate):
    self._loadExcitationsSet(excitationSet, sampleRate)



  def setSystems(self,
                 vecM,
                 vecB,
                 vecK,
                 vecGamma,
                 vecLambda,
                 vecForcePattern,
                 vecInitialConditions):

    # Check if the system is valid
    # Check the number of system in all of the inputs vectors
    if len(vecM) != len(vecB) or len(vecM) != len(vecK) or len(vecM) != len(vecGamma) or len(vecM) != len(vecLambda) or len(vecM) != len(vecForcePattern) or len(vecM) != len(vecInitialConditions):
      raise ValueError("The number of systems in the input vectors must be the same.")

    # Check that the matrix of each system are of the same size
    for i in range(len(vecM)):
      if len(vecM[i]) != len(vecB[i]) or len(vecM[i]) != len(vecK[i]) or len(vecM[i]) != len(vecGamma[i]) or len(vecM[i]) != len(vecLambda[i]) or len(vecM[i]) != len(vecForcePattern[i]) or len(vecM[i]) != len(vecInitialConditions[i]):
        raise ValueError("The size of the matrix of each system must be the same.")



    # Invert the M matrix and then pre-multiply the other matrices
    for i in range(len(vecM)):
      vecM[i] = np.linalg.inv(vecM[i])
      vecM[i] = vecM[i]
      vecB[i] = -1 * np.matmul(vecM[i], vecB[i])
      vecK[i] = -1 * np.matmul(vecM[i], vecK[i])
      vecGamma[i] = np.einsum('ij, jkl -> ikl', vecM[i], vecGamma[i])
      vecLambda[i] = np.einsum('ij, jklm -> iklm', vecM[i], vecLambda[i])
      vecForcePattern[i] = np.diag(vecM[i]) * vecForcePattern[i]

    # For debugging purpose print the matrices
    """ print("vecM: \n", vecM)
    print("vecB: \n", vecB)
    print("vecK: \n", vecK) """
    # print("vecGamma: \n", vecGamma)
    print("vecLambda: \n", vecLambda)
    # print("vecForcePattern: \n", vecForcePattern)

    """ Lambda3 = [[0.0, 0.0, 0.0],
               [0.0, -4000.0, 0.0],
               [0.0, 0.0, 0.0]] """

    Lambda3 = [[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]
    
    vecLambda3 = np.array([Lambda3])

    # Load the system in the GPU
    self._setSystems(vecB, vecK, vecGamma, vecLambda3, vecForcePattern, vecInitialConditions)



  def getAmplitudes(self):
    return self._getAmplitudes()


