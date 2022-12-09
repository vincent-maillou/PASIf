import sys
sys.path.append('./build')

import os
import numpy as np

# print(dir(pasif))

excitationLength = 78001
sampleRate = 16000

excitationSet = []

# Generate the dummy excitation signals
for i in range(1):
      excitation = np.zeros(excitationLength)
      for t in range(excitationLength):
            excitation[t] = 1.0

      """ if(i%2 == 0):
            for t in range(excitationLength):
                  excitation[t] = 1.0
      else:
            for t in range(excitationLength):
                  excitation[t] = 2.0 """

      excitationSet.append(excitation)


print("Number of excitation signals: ", len(excitationSet))



import PASIf as pasif
""" print( dir(pasif) ) """
gpudriver = pasif.__GpuDriver(excitationSet, sampleRate)
""" print( dir(gpudriver) ) """

# Define M, B, K, Gamma, Lambda and ForcePattern
M2 = [[1.0, 0.0],
     [0.0, 1.5]]

M3 = [[1.0, 0.0, 0.0],
      [0.0, 1.5, 0.0],
      [0.0, 0.0, 0.0]]


B2 = [[2.0, 0.0],
     [0.0, 2.0]]

B3 = [[2.0, 0.0, 0.0],
      [0.0, 2.0, 0.0],
      [0.0, 0.0, 0.0]]


K2 = [[6.0, 0.6],
     [0.6, 6.0]]

K3 = [[6.0, 0.6, 0.0],
      [0.6, 6.0, 0.0],
      [0.0, 0.0, 0.0]]


Gamma2 = [[[0.0, 0.0],
          [0.0, 0.0]], [[0.0, 0.0],
                        [0.0, 0.0]]]

Gamma3 = [[[0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0]]]



Lambda2 = [[0.0, 0.0],
          [0.0, 0.0]] 

Lambda3 = [[0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0]]


ForcePattern2 = [1.0, 1.0]

ForcePattern3 = [1.0, 1.0, 0.0]


InitialCondition2 = [0.0, 0.0]

InitialCondition3 = [0.0, 0.0, 0.0]



vecM = [M2]

vecB = [B2]
vecK = [K2]
vecGamma = [Gamma2]
vecLambda = [Lambda2]
vecForcePattern = [ForcePattern2]
vecInitialCondition = [InitialCondition2]

vecM3 = [M3]
vecB3 = [B3]
vecK3 = [K3]
vecGamma3 = [Gamma3]
vecLambda3 = [Lambda3]
vecForcePattern3 = [ForcePattern3]
vecInitialCondition3 = [InitialCondition3]

n = 1

# gpudriver.__setSystems(n*vecM, n*vecB, n*vecK, n*vecGamma, n*vecLambda, n*vecForcePattern, n*vecInitialCondition)
gpudriver.__setSystems(n*vecM3, n*vecB3, n*vecK3, n*vecGamma3, n*vecLambda3, n*vecForcePattern3, n*vecInitialCondition3)
amplitudes = gpudriver.__getAmplitudes()

print("Size of amplitudes vector: ", len(amplitudes))
print("Amplitudes: ", amplitudes)