"""
This file provide a way of testing the entire interface using hand-made 
users inputs.

A single DOF-probed system only is tested here.

It's not connected to Springtronics.
"""

import sys
sys.path.append('../')
sys.path.append('../build')

from PASIf import *

import numpy as np
import time
import matplotlib.pyplot as plt



# Generate the excitation set
excitation : np.ndarray = np.ones(78001)
# Fill the excitation vector with a ramp
""" excitation = []
for i in range(78001):
      excitation.append(i/78001) """
      
n_excitations = 1
excitationSet : list[np.ndarray] = [excitation] * n_excitations 
      
sampleRate = 16000

displayCompute = False
displaySystem  = False
displaySolver  = False

pasif = PASIf(excitationSet, sampleRate, 0, displayCompute, displaySystem, displaySolver)

# pasif.setExcitations(excitationSet, sampleRate)


n = 1

systemSize = 4

from scipy.sparse import dia_matrix
M    : dia_matrix       = dia_matrix(([1., 1., 1., 1.], [0]), shape=(4, 4)) 
vecM : list[dia_matrix] = [M] * n


""" To follow "test_1DOFProbe.ipynb"
k = 12.
b = 2. 
"""

# But here follow "example copy.ipynb"
k = 1.
b = 1.

from scipy.sparse import coo_matrix
B    : coo_matrix       = coo_matrix(([-1., -1.], ([0, 1], [2, 3])), shape=(systemSize, systemSize))
vecB : list[coo_matrix] = [B] * n

K    : coo_matrix       = coo_matrix(([k, b], ([2, 2], [0, 2])), shape=(systemSize, systemSize))
vecK : list[coo_matrix] = [K] * n

Gamma : coo_tensor = coo_tensor(dimensions_ = [systemSize, systemSize, systemSize])
Gamma.val          = [-1.]
Gamma.indices      = [3, 0, 0]
vecGamma : list[coo_tensor] = [Gamma] * n

Lambda : coo_tensor = coo_tensor(dimensions_ = [systemSize, systemSize, systemSize, systemSize])
Lambda.val          = [0]
Lambda.indices      = [0,0,0,0]
vecLambda : list[coo_tensor] = [Lambda] * n

forcePattern    : np.ndarray       = np.array([0., 0., 1., 0.])
vecForcePattern : list[np.ndarray] = [forcePattern] * n

initialCondition    : np.ndarray       = np.zeros(systemSize)
vecInitialCondition : list[np.ndarray] = [initialCondition] * n


""" print("M: \n", M.todense())
print("B: \n", B.todense())
print("K: \n", K.todense())
print("Gamma: ", Gamma)
print("Lambda: ", Lambda)
print("Force Pattern: ", forcePattern)
print("Initial Condition: ", initialCondition) """


start = time.time()
pasif.setSystems(vecM, vecB, vecK, vecGamma, vecLambda, vecForcePattern, vecInitialCondition)
end = time.time()

print("setSystems() overall time: ", end - start)


# Interpolation matrix
""" intMat = np.array([[2/10, 4/10, 3/10, 1/10], 
                   [1/10, 3/10, 4/10, 2/10]]) """

intMat = np.array([[2/10, 3/10, 3/10, 2/10]])

# pasif.setInterpolationMatrix(intMat)

# Modulation buffer
modulationBuffer = np.array([1.0, 1.0, 1.0, 1.0])
#modulationBuffer = np.array([0.5, 0.5, 0.5, 0.5])
#modulationBuffer = np.array([0, 1, 0, 1])

# Fill the modulation buffer with a sine wave of period 1Hz and amplitude 1
#modulationBuffer = np.sin(2*np.pi*1*np.linspace(0, 1, 1000))


#pasif.setModulationBuffer(modulationBuffer)


start   = time.time()
results = pasif.getAmplitudes()
end     = time.time()

print("setMatrix() + getAmplitude() overall time: ", end - start)
print("Amplitudes: ", results)




start      = time.time()
trajectory = pasif.getTrajectory(saveSteps = 1)
end        = time.time()

print("getTrajectories() overall time: ", end - start)
plt.plot(trajectory[0], trajectory[1])
plt.show() 



from scipy.sparse import coo_matrix

# From Theophile's example
adjointSize = 6

M_values  = np.ones(adjointSize)
M    : dia_matrix       = dia_matrix((M_values, [0]), shape=(adjointSize, adjointSize)) 
vecM : list[dia_matrix] = [M] * n


B_dims    = [adjointSize, adjointSize]
""" B_values  = [-1, -1]
B_indexes = [2, 0,
             3, 1] """
B_values  = [-1, -1]
B_rows    = [1, 2]
B_cols    = [4, 5]

B    : coo_matrix       = coo_matrix((B_values, (B_rows, B_cols)), shape=(B_dims[0], B_dims[1]))
vecB : list[coo_matrix] = [B] * n


K_dims    = [adjointSize, adjointSize]
K_values  = []
K_rows    = []
K_cols    = []

K    : coo_matrix       = coo_matrix((K_values, (K_rows, K_cols)), shape=(K_dims[0], K_dims[1]))
vecK : list[coo_matrix] = [K] * n


Gamma_dims     = [adjointSize, adjointSize, systemSize]
Gamma_values   = []
Gamma_indexes  = []

Gamma : coo_tensor = coo_tensor(dimensions_ = Gamma_dims)
Gamma.val          = Gamma_values
Gamma.indices      = Gamma_indexes
vecGamma : list[coo_tensor] = [Gamma] * n


Lambda_dims    = [adjointSize, adjointSize, systemSize, systemSize]
Lambda_values  = [-1, -1]
Lambda_indexes = [4, 0, 0, 0, 5, 0, 1, 1]

Lambda : coo_tensor = coo_tensor(dimensions_ = Lambda_dims)
Lambda.val          = Lambda_values
Lambda.indices      = Lambda_indexes
vecLambda : list[coo_tensor] = [Lambda] * n

forcePattern    : np.ndarray = np.zeros(adjointSize)
vecForcePattern : list[np.ndarray] = [forcePattern] * n

initialCondition    : np.ndarray = np.zeros(adjointSize)
initialCondition[0] = 1.
#initialCondition[1] = 1.
#initialCondition[2] = 1.
#initialCondition[3] = 1.
vecInitialCondition : list[np.ndarray] = [initialCondition] * n

Psi_dims         = [adjointSize, adjointSize, systemSize, systemSize, systemSize]
Psi : coo_tensor = coo_tensor(dimensions_ = Psi_dims)
Psi.val          = [0]
Psi.indices      = [1,1,1,1,1]
vecPsi : list[coo_tensor] = [Psi] * n


start      = time.time()
pasif.setJacobian(vecM, vecB, vecK, vecGamma, vecLambda, vecForcePattern, vecInitialCondition, vecPsi)
end        = time.time()

print("setJacobian() overall time: ", end - start)

start      = time.time()
gradient = pasif.getGradient()
end        = time.time()

print("getGradient() overall time: ", end - start)
print("Full gradient: ", gradient)
print(f'Amplitude" {results}')

# plt.plot(gradient[0], gradient[1], 'r')
# plt.plot(gradient[0], gradient[2], 'b')
# plt.show() 

# for i in range(len(gradient)-1):
#     plt.plot(gradient[0], gradient[i+1])
# plt.show() 

pos= [] 
from tqdm import trange

for save in trange(5):
    gradient = pasif.getGradient(279-save)

    pasif.setJacobian(vecM, vecB, vecK, vecGamma, vecLambda, vecForcePattern, [np.array(gradient)], vecPsi)
    pos.append(gradient[0])

fig, axs = plt.subplots(2, constrained_layout=True)
fig.suptitle('Chunk computation process', fontsize=16)
axs[0].set_title('Complete Trajectory')
axs[0].plot(trajectory[0], trajectory[1])
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('amplitude')

axs[1].set_title('Reverse forward chunk computation')
axs[1].plot(np.flip(pos))
axs[1].set_xlabel(f"current setpoint: {save}")
axs[1].set_ylabel('amplitude')
plt.show()
