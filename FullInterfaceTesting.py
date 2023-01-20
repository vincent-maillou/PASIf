from PASIf import *

import numpy as np


# Generate the excitation set
excitationSet = []
excitation = np.ones(78001)
# Fill the excitation vector with a ramp
""" excitation = []
for i in range(78001):
      excitation.append(i/78001) """

excitationSet.append(excitation)
sampleRate = 16000


pasif = PASIf(excitationSet, sampleRate)

# pasif.setExcitations(excitationSet, sampleRate)


M = [[1.0, 0.0, 0.0],
      [0.0, 10, 0.0],
      [0.0, 0.0, 1.0]]

B = [[2.0, 0.0, 0.0],
      [0.0, 10.0, 0.0],
      [0.0, 0.0, 0.0]]

K = [[12.0, 0.0, 0.0],
      [0.0, 10.0, 0.0],
      [0.0, 0.0, 0.0]]

Gamma = [[[0.0, 20.0, 0.0],
           [0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0]], [[10, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                                                 [0.0, 1, 0.0],
                                                 [0.0, 0.0, 0.0]]]

Lambda = [[[[0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0]]],
           [[[0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                              [0.0, 40000.0, 0.0],
                              [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0]]],
           [[[0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0]]]]

ForcePattern = [1.0, 0.0, 0.0]

InitialCondition = [0.0, 0.0, 0.0]

#n=8*1000*2
n=1

vecM = np.array(n*[M])
vecB = np.array(n*[B])
vecK = np.array(n*[K])
vecGamma = np.array(n*[Gamma])
vecLambda = np.array(n*[Lambda])
vecForcePattern = np.array(n*[ForcePattern])
vecInitialCondition = np.array(n*[InitialCondition])



# Start python timer
import time
start = time.time()

### THIS PART WILL BE IN THE CUDA ENV AS PREPROCESSING ###

# Make the second-order ode reduction.
# To do so we incorporate an identity matrix in the B matrix.
# We then extend the K, Gamma, Lambda and vectors to the new size

extendedVecM = []
extendedVecB = []
extendedVecK = []
extendedVecGamma = []
extendedVecLambda = []
extendedForcePattern = []
extendedInitialCondition = []

numberOfSystems = len(vecM)

for i in range(numberOfSystems):
      sysDim = vecM[i].shape[0]
      sysDimI = np.eye(sysDim)
      sysDim0 = np.zeros((sysDim, sysDim))

      #Extending M, B, K
      extendedVecM.append(np.block([[sysDimI, sysDim0],
                                    [sysDim0, vecM[i]]]))

      extendedVecB.append(np.block([[sysDim0, -1*sysDimI],
                                    [sysDim0,    vecB[i]]]))

      extendedVecK.append(np.block([[sysDim0, sysDim0],
                                    [vecK[i], sysDim0]]))

      # Extending Gamma 3D Tensor
      sysDim2 = 2*sysDim
      tempGamma = []
      zerosDim2 = np.zeros((sysDim2, sysDim2))

      for j in range(sysDim):
            tempGamma.append(zerosDim2)
      for j in range(sysDim):
            tempGamma.append(np.block([[vecGamma[i][j], sysDim0],
                                       [sysDim0,        sysDim0]]))

      tempGamma = np.array(tempGamma)
      extendedVecGamma.append(tempGamma)

      # Extending Lambda 4D Tensor
      tempLambda = []
      for j in range(sysDim):
            tempLambda2 = []
            for k in range(sysDim):
                  tempLambda2.append(zerosDim2)
            tempLambda2 = np.array(tempLambda2)
            tempLambda.append(tempLambda2)

      for j in range(sysDim):
            tempLambda2 = []
            for k in range(sysDim):
                  tempLambda2.append(np.block([[vecLambda[i][j][k], sysDim0],
                                               [sysDim0,            sysDim0]]))
            tempLambda2 = np.array(tempLambda2)
            tempLambda.append(tempLambda2)

      tempLambda = np.array(tempLambda)
      extendedVecLambda.append(tempLambda)

      # Extending Force Pattern
      extendedForcePattern.append(np.concatenate((np.zeros(sysDim), vecForcePattern[i])))

      # Extending Initial Condition 
      extendedInitialCondition.append(np.concatenate((vecInitialCondition[i], np.zeros(sysDim))))


extendedVecM = np.array(extendedVecM)
extendedVecB = np.array(extendedVecB)
extendedVecK = np.array(extendedVecK)
extendedVecGamma = np.array(extendedVecGamma)
extendedVecLambda = np.array(extendedVecLambda)
extendedForcePattern = np.array(extendedForcePattern)
extendedInitialCondition = np.array(extendedInitialCondition)


pasif.setSystems(extendedVecM, extendedVecB, extendedVecK, extendedVecGamma, extendedVecLambda, extendedForcePattern, extendedInitialCondition)



# Interpolation matrix
""" intMat = np.array([[2/10, 4/10, 3/10, 1/10], 
                   [1/10, 3/10, 4/10, 2/10]]) """

intMat = np.array([[2/10, 3/10, 3/10, 2/10]])

pasif.setInterpolationMatrix(intMat)

# Modulation buffer
modulationBuffer = np.array([1.0, 1.0, 1.0, 1.0])
#modulationBuffer = np.array([0.5, 0.5, 0.5, 0.5])
#modulationBuffer = np.array([0, 1, 0, 1])

# Fill the modulation buffer with a sine wave of period 1Hz and amplitude 1
#modulationBuffer = np.sin(2*np.pi*1*np.linspace(0, 1, 1000))


pasif.setModulationBuffer(modulationBuffer)


results = pasif.getAmplitudes(verbose_ = True, debug_ = True)

print("Amplitudes: ", results)

trajectory = pasif.getTrajectory()

print("Number of steps: ", len(trajectory))

# Plot the trajectory
import matplotlib.pyplot as plt
plt.plot(trajectory)
plt.show()

# End python timer
end = time.time()
print("Overlall elapsed time: ", end - start)
