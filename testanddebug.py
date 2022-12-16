import sys
sys.path.append('./build')

import os
import numpy as np

# print(dir(pasif))

pathToDataset = "data/training/"
sampleRate = 16000

excitationSet = []

""" listOfFiles = os.listdir(pathToDataset)

for file in listOfFiles:
  # Reading float value from file
  excitation = np.fromfile(pathToDataset + file)
  excitationSet.append(excitation)     """

# print("Number of excitation signals: ", len(excitationSet))


# excitation = np.fromfile("data/training/soundfile_4")
excitation = np.ones(78001)
# excitation[0:3] = 0
# excitation = excitation[excitation!=0]
excitationSet.append(excitation)  

""" import matplotlib.pyplot as plt
plt.plot(excitationSet[0])
plt.show() """

import PASIf as pasif
""" print( dir(pasif) ) """
gpudriver = pasif.__GpuDriver(excitationSet, sampleRate)
""" print( dir(gpudriver) ) """

# Define M, B, K, Gamma, Lambda and ForcePattern
M1 = [[1.0]]


M3 = [[1.0, 0.0, 0.0],
      [0.0, 10, 0.0],
      [0.0, 0.0, 1.0]]

B1 = [[2.0]]

B3 = [[2.0, 0.0, 0.0],
      [0.0, 10.0, 0.0],
      [0.0, 0.0, 0.0]]


K1 = [[12.0]]

K3 = [[12.0, 0.0, 0.0],
      [0.0, 10.0, 0.0],
      [0.0, 0.0, 0.0]]

Gamma1 = [[[0.0]]]

Gamma3 = [[[0.0, 20.0, 0.0],
           [0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0],
                                                 [0.0, 0.0, 0.0]]]
Lambda1 = [[0.0]]

Lambda3 = [[0.0, 0.0, 0.0],
           [0.0, 40000.0, 0.0],
           [0.0, 0.0, 0.0]]

ForcePattern1 = [1.0]
ForcePattern3 = [1.0, 0.0, 0.0]

InitialCondition1 = [0.0]
InitialCondition3 = [0.0, 0.0, 0.0]

vecM1 = [M1]
vecB1 = [B1]
vecK1 = [K1]
vecGamma1 = [Gamma1]
vecLambda1 = [Lambda1]
vecForcePattern1 = [ForcePattern1]
vecInitialCondition1 = [InitialCondition1]

vecM3 = [M3]
vecB3 = [B3]
vecK3 = [K3]
vecGamma3 = [Gamma3]
vecLambda3 = [Lambda3]
vecForcePattern3 = [ForcePattern3]
vecInitialCondition3 = [InitialCondition3]

n = 1
gpudriver.__setSystems(n*vecM1, n*vecB1, n*vecK1, n*vecGamma1, n*vecLambda1, n*vecForcePattern1, n*vecInitialCondition1)

#gpudriver.__setSystems(n*vecM3, n*vecB3, n*vecK3, n*vecGamma3, n*vecLambda3, n*vecForcePattern3, n*vecInitialCondition3)
results = gpudriver.__getAmplitudes()

print("Size of results vector: ", len(results[0]))
print("Results Q1: ", results[0])
print("Results Q2: ", results[1])


# Multiply the excitation signal by 2
excitationSet[0] = 2*excitationSet[0]

gpudriver.__loadExcitationsSet(excitationSet, sampleRate)

results = gpudriver.__getAmplitudes()

print("Size of results vector: ", len(results[0]))
print("Results Q1: ", results[0])
print("Results Q2: ", results[1])