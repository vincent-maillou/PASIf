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

print("Number of excitation signals: ", len(excitationSet))



import PASIf as pasif
print( dir(pasif) )
gpudriver = pasif.__GpuDriver(excitationSet, sampleRate)
print( dir(gpudriver) )
