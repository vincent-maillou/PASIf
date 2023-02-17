from PASIf import *

import numpy as np
import time
import matplotlib.pyplot as plt



# Generate the excitation set
excitationSet = []
excitation = np.ones(78001)
# Fill the excitation vector with a ramp
""" excitation = []
for i in range(78001):
      excitation.append(i/78001) """

excitationSet.append(excitation)

sampleRate = 16000

displayCompute = True
displaySystem  = True
displaySolver  = True

pasif = PASIf(excitationSet, sampleRate, 0, displayCompute, displaySystem, displaySolver)

# pasif.setExcitations(excitationSet, sampleRate)


M = [[1, 0.0, 0.0],
      [0.0, 10, 0.0],
      [0.0, 0.0, 1.0]]

B = [[1.0, 0.0, 0.0],
      [0.0, 10.0, 0.0],
      [0.0, 0.0, 0.0]]

K = [[6.0, 0.0, 0.0],
      [0.0, 10.0, 0.0],
      [0.0, 0.0, 0.0]]

Gamma = [[[0.0, 10.0, 0.0],
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

ForcePattern = [0.5, 0.0, 0.0]

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




cooM = cooTensor(dimensions_ = [6, 6])
cooM.val     = [1, 1, 1, 1, 10, 1]
cooM.indices = [0,0 , 1,1 , 2,2 , 3,3 , 4,4 , 5,5]

cooB = cooTensor(dimensions_ = [6, 6])
cooB.val     = [-1, -1, -1, 1, 10]
cooB.indices = [0,3 , 1,4 , 2,5 , 3,3 , 4,4]

cooK = cooTensor(dimensions_ = [6, 6])
cooK.val     = [6, 10]
cooK.indices = [3,0 , 4,1]

cooGamma = cooTensor(dimensions_ = [6, 6, 6])
cooGamma.val     = [20, 10, 1]
cooGamma.indices = [0,1,3 , 0,0,4 , 1,1,5]

cooLambda = cooTensor(dimensions_ = [6, 6, 6, 6])
cooLambda.val     = [40000]
cooLambda.indices = [1,1,1,4]


""" cooB = cooTensor(dimensions_ = [6, 6])
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
cooLambda.indices = [1,1,1,4] """

forcePattern = [1, 0, 0, 0, 0, 0]
initialCondition = [0, 0, 0, 0, 0, 0]

""" print("cooM: ", cooM)
print("cooB: ", cooB)
print("cooK: ", cooK)
print("cooGamma: ", cooGamma)
print("cooLambda: ", cooLambda) """

n = 1
cooVecM = [cooM]
cooVecM *= n
cooVecB = [cooB]
cooVecB *= n
cooVecK = [cooK]
cooVecK *= n
cooVecGamma = [cooGamma]
cooVecGamma *= n
cooVecLambda = [cooLambda]
cooVecLambda *= n
vecForcePattern = [forcePattern]
vecForcePattern *= n
vecInitialCondition = [initialCondition]
vecInitialCondition *= n

start = time.time()
pasif.setSystems(cooVecM, cooVecB, cooVecK, cooVecGamma, cooVecLambda, vecForcePattern, vecInitialCondition)
end = time.time()

print("setSystems() overall time: ", end - start)


# Interpolation matrix
""" intMat = np.array([[2/10, 4/10, 3/10, 1/10], 
                   [1/10, 3/10, 4/10, 2/10]]) """

""" intMat = np.array([[2/10, 3/10, 3/10, 2/10]])

pasif.setInterpolationMatrix(intMat) """

# Modulation buffer
modulationBuffer = np.array([1.0, 1.0, 1.0, 1.0])
#modulationBuffer = np.array([0.5, 0.5, 0.5, 0.5])
#modulationBuffer = np.array([0, 1, 0, 1])

# Fill the modulation buffer with a sine wave of period 1Hz and amplitude 1
#modulationBuffer = np.sin(2*np.pi*1*np.linspace(0, 1, 1000))


#pasif.setModulationBuffer(modulationBuffer)

# Start python timer

""" start   = time.time()
results = pasif.getAmplitudes()
end     = time.time()

print("setMatrix() + getAmplitude() overall time: ", end - start)
print("Amplitudes: ", results) """




""" start      = time.time()
trajectory = pasif.getTrajectory(saveSteps = 1)
end        = time.time()

print("getTrajectories() overall time: ", end - start)
plt.plot(trajectory[0], trajectory[1])
plt.show()




# Initialize the Psi 5 dimmensional tensor and fill it with 0
Psi = np.zeros((sysDim2, sysDim2, sysDim2, sysDim2, sysDim2))
extendedVecPsi = np.array(n*[Psi])

start      = time.time()
pasif.setJacobian(extendedVecM, extendedVecB, extendedVecK, extendedVecGamma, extendedVecLambda, extendedForcePattern, extendedInitialCondition, extendedVecPsi)
end        = time.time()

print("setJacobian() overall time: ", end - start)

start      = time.time()
gradient = pasif.getGradient()
end        = time.time()

print("getGradient() overall time: ", end - start)

plt.plot(gradient[0], gradient[1], 'x')
plt.show() """



""" for save in range(279):
      gradient = pasif.getGradient(279-save)
      pasif.setJacobian(extendedVecM, extendedVecB, extendedVecK, extendedVecGamma, extendedVecLambda, extendedForcePattern, extendedInitialCondition, extendedVecPsi)
      

      fig, axs = plt.subplots(2, constrained_layout=True)
      fig.suptitle('Chunk computation process', fontsize=16)
      axs[0].set_title('Complete Trajectory')
      axs[0].plot(trajectory[0], trajectory[1])
      axs[0].set_xlabel('time (s)')
      axs[0].set_ylabel('amplitude')

      axs[1].set_title('Reverse forward chunk computation')
      axs[1].plot(gradient[1])
      axs[1].set_xlabel(f"current setpoint: {save}")
      axs[1].set_ylabel('amplitude')

      i = 0
      fig.savefig(f"fig/plt_{save}.png") """


