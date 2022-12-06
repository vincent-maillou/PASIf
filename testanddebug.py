import sys
sys.path.append('./build')

import os
import numpy as np

# print(dir(pasif))

pathToDataset = "data/training/"
sampleRate = 128000

excitationSet = []

listOfFiles = os.listdir(pathToDataset)

for file in listOfFiles:
  # Reading float value from file
  excitation = np.fromfile(pathToDataset + file)
  excitationSet.append(excitation)    

# print("Number of excitation signals: ", len(excitationSet))



import PASIf as pasif
""" print( dir(pasif) ) """
gpudriver = pasif.__GpuDriver(excitationSet, sampleRate)
""" print( dir(gpudriver) ) """

# Define M, B, K, Gamma, Lambda and ForcePattern
M2 = [[1.0, 0.0],
     [0.0, 1.0]]
    
M3 = [[1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]]


B2 = [[1.0, 0.0],
     [0.0, 1.0]]

B3 = [[3.0, 0.0, 0.0],
      [0.0, 3.0, 0.0],
      [0.0, 0.0, 3.0]]


K2 = [[10.0, 1.0],
     [1.0, 10.0]]

K3 = [[30.0, 3.0, 0.0],
      [3.0, 30.0, 3.0],
      [0.0, 3.0, 30.0]]


Gamma2 = [[[1.0, 0.0],
          [0.0, 0.0]], [[0.0, 0.0],
                        [0.0, 1.0]]]

Gamma3 = [[[3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                                [0.0, 3.0, 0.0],
                                [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 3.0]]]

Lambda2 = [[0.5, 0.0],
          [0.0, 0.5]]

Lambda3 = [[0.3, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.3]]  


ForcePattern2 = [1.0, 0.0]

ForcePattern3 = [0.0, 3.0, 0.0]

InitialCondition2 = [0.0, 2.0]

InitialCondition3 = [0.0, 0.0, 3.0]


vecM = [M2, M2, M3]
vecB = [B2, B2, B3]
vecK = [K2, K2, K3]
vecGamma = [Gamma2, Gamma2, Gamma3]
vecLambda = [Lambda2, Lambda2, Lambda3]
vecForcePattern = [ForcePattern2, ForcePattern2, ForcePattern3]
vecInitialCondition = [InitialCondition2, InitialCondition2, InitialCondition3]

gpudriver.__setSystems(vecM, vecB, vecK, vecGamma, vecLambda, vecForcePattern, vecInitialCondition)
gpudriver.__getAmplitudes()

vecM = [M2, M3]
vecB = [B2, B3]
vecK = [K2, K3]
vecGamma = [Gamma2, Gamma3]
vecLambda = [Lambda2, Lambda3]
vecForcePattern = [ForcePattern2, ForcePattern3]
vecInitialCondition = [InitialCondition2, InitialCondition3]

gpudriver.__setSystems(vecM, vecB, vecK, vecGamma, vecLambda, vecForcePattern, vecInitialCondition)
gpudriver.__getAmplitudes()

""" import time
time.sleep(10) """

