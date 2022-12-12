import sys
sys.path.append('./build')

import os
import numpy as np

# print(dir(pasif))

excitationLength = 78001
#excitationLength = 1
sampleRate = 16000
# sampleRate = 1

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
M1 = [[1.0]]

M2 = [[1.0, 0.0],
     [0.0,  0.0]]

M3 = [[1.0, 0.0, 0.0],
      [0.0, 1.5, 0.0],
      [0.0, 0.0, 0.0]]

B1 = [[2.0]]

B2 = [[2.0, 0.0],
     [0.0, 0.0]]

B3 = [[2.0, 0.0, 0.0],
      [0.0, 2.0, 0.0],
      [0.0, 0.0, 0.0]]

K1 = [[12.0]]

K2 = [[12.0, 0.0],
      [0.0, 0.0]]

K3 = [[12.0, 1.2, 0.0],
      [1.2, 12.0, 0.0],
      [0.0, 0.0, 0.0]]

Gamma1 = [[[0.0]]]

Gamma2 = [[[0.0, 0.0],
          [0.0, 0.0]], [[1.0, 0.0],
                        [0.0, 0.0]]]

Gamma3 = [[[0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0]]]


Lambda1 = [[0.0]]

Lambda2 = [[0.0, 0.0],
          [0.0, 0.0]] 

Lambda3 = [[0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0]]

ForcePattern1 = [1.0]

ForcePattern2 = [1.0, 0.0]

ForcePattern3 = [1.0, 1.0, 0.0]

InitialCondition1 = [0.0]

InitialCondition2 = [0.0, 0.0]

InitialCondition3 = [0.0, 0.0, 0.0]


# Define the vectors of M, B, K, Gamma, Lambda and ForcePattern
vecM1 = [M1]
vecB1 = [B1]
vecK1 = [K1]
vecGamma1 = [Gamma1]
vecLambda1 = [Lambda1]
vecForcePattern1 = [ForcePattern1]
vecInitialCondition1 = [InitialCondition1]

vecM2 = [M2]
vecB2 = [B2]
vecK2 = [K2]
vecGamma2 = [Gamma2]
vecLambda2 = [Lambda2]
vecForcePattern2 = [ForcePattern2]
vecInitialCondition2 = [InitialCondition2]

vecM3 = [M3]
vecB3 = [B3]
vecK3 = [K3]
vecGamma3 = [Gamma3]
vecLambda3 = [Lambda3]
vecForcePattern3 = [ForcePattern3]
vecInitialCondition3 = [InitialCondition3]

n = 1

# gpudriver.__setSystems(n*vecM1, n*vecB1, n*vecK1, n*vecGamma1, n*vecLambda1, n*vecForcePattern1, n*vecInitialCondition1)
gpudriver.__setSystems(n*vecM2, n*vecB2, n*vecK2, n*vecGamma2, n*vecLambda2, n*vecForcePattern2, n*vecInitialCondition2)
# gpudriver.__setSystems(n*vecM3, n*vecB3, n*vecK3, n*vecGamma3, n*vecLambda3, n*vecForcePattern3, n*vecInitialCondition3)
amplitudes = gpudriver.__getAmplitudes()

print("Size of amplitudes vector: ", len(amplitudes))
print("Amplitudes: ", amplitudes)






""" # Generate the time vector with the sample rate
time = np.arange(0, len(amplitudes)/sampleRate, 1/sampleRate)

# plot the amplitude regarding one time over 2
import matplotlib.pyplot as plt
plt.plot(time, amplitudes)

# Compute the integrale of the amplitude
plt.title(np.trapz(np.array(amplitudes)**2, x=time))

plt.show() """