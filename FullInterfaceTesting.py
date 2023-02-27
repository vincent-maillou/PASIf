from PASIf import *

import numpy as np
import time
import matplotlib.pyplot as plt



# Generate the excitation set
excitation : np.ndarray = np.ones(2*78001)
# Fill the excitation vector with a ramp
""" excitation = []
for i in range(78001):
      excitation.append(i/78001) """
      
n_excitations = 1
excitationSet : list[np.ndarray] = [excitation] * n_excitations 
      
sampleRate = 16000

displayCompute = True
displaySystem  = True
displaySolver  = True

pasif = PASIf(excitationSet, sampleRate, 0, displayCompute, displaySystem, displaySolver)

# pasif.setExcitations(excitationSet, sampleRate)


n = 1

systemSize = 8

from scipy.sparse import dia_matrix
M    : dia_matrix       = dia_matrix(([1., 1., 1., 1., 1., 10., 1., 1.], [0]), shape=(systemSize, systemSize)) 
vecM : list[dia_matrix] = [M] * n


from scipy.sparse import coo_matrix
B    : coo_matrix       = coo_matrix(([-1, -1, -1, -1, 1, 10], ([0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 4, 5])), shape=(systemSize, systemSize))
vecB : list[coo_matrix] = [B] * n

K    : coo_matrix       = coo_matrix(([6, 10], ([4, 5], [0, 1])), shape=(systemSize, systemSize))
vecK : list[coo_matrix] = [K] * n

Gamma : coo_tensor = coo_tensor(dimensions_ = [systemSize, systemSize, systemSize])
Gamma.val          = [-10, -10, -1, -1]
Gamma.indices      = [0,1,4 , 0,0,5 , 0,0,6, 1,1,7]
vecGamma : list[coo_tensor] = [Gamma] * n

Lambda : coo_tensor = coo_tensor(dimensions_ = [systemSize, systemSize, systemSize, systemSize])
Lambda.val          = [40000]
Lambda.indices      = [1,1,1,5]
vecLambda : list[coo_tensor] = [Lambda] * n

forcePattern    : np.ndarray = np.array([0, 0, 0, 0, 0.5, 0, 0, 0])
vecForcePattern : list[np.ndarray] = [forcePattern] * n

initialCondition    : np.ndarray = np.zeros(systemSize)
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




""" start      = time.time()
trajectory = pasif.getTrajectory(saveSteps = 1)
end        = time.time()

print("getTrajectories() overall time: ", end - start)
plt.plot(trajectory[0], trajectory[1])
plt.show() """


""" jac_M : dia_matrix = dia_matrix(([1., 1., 1., 1., 1., 1.], [0]), shape=(6, 6))
vecJacM : list[dia_matrix] = [jac_M] * n

jac_B : coo_matrix = coo_matrix(([1], ([2], [0])), shape=(6, 6))
vecJacB : list[coo_matrix] = [jac_B] * n

jac_K : coo_matrix = coo_matrix(([-6, -2], ([0, 2], [1, 1])), shape=(6, 6))
vecJacK : list[coo_matrix] = [jac_K] * n

jac_Gamma : coo_tensor = coo_tensor(dimensions_ = [6, 6, 4])
jac_Gamma.val          = [2, -1, -1]
jac_Gamma.indices      = [0, 2, 0, 4, 1, 2, 5, 1, 0]
vecJac_Gamma : list[coo_tensor] = [jac_Gamma] * n """




from scipy.sparse import coo_matrix

# From Theophile's example
adjointSize = 12

M_values  = np.ones(adjointSize)
M    : dia_matrix       = dia_matrix((M_values, [0]), shape=(adjointSize, adjointSize)) 
vecM : list[dia_matrix] = [M] * n


B_dims    = [12, 12]
B_values  = [1, 1, 1, 1]
B_indexes = [4, 0, 5, 1, 6, 2, 7, 3]
B_rows    = B_indexes[::2]
B_cols    = B_indexes[1::2]

B    : coo_matrix       = coo_matrix((B_values, (B_rows, B_cols)), shape=(B_dims[0], B_dims[1]))
vecB : list[coo_matrix] = [B] * n


K_dims    = [12, 12]
K_values  = [-12.0, -2.0, -0.5, -1.0]
K_indexes = [0, 4, 4, 4, 1, 5, 5, 5]
K_rows    = K_indexes[::2]
K_cols    = K_indexes[1::2]
K    : coo_matrix       = coo_matrix((K_values, (K_rows, K_cols)), shape=(K_dims[0], K_dims[1]))
vecK : list[coo_matrix] = [K] * n


Gamma_dims     = [12, 12, 8]
Gamma_values   = [-20.0, -20.0, -1.0, -1.0, -2.0, -0.1, -0.1, 1, 1]
Gamma_indexes  = [0, 4, 1, 1, 4, 0, 8, 4, 0, 9, 4, 4, 0, 5, 0, 10, 5, 1, 11, 5, 5, 0, 6, 0, 1, 7, 1]

Gamma_values   = [-2.0, 1, -20, -1.0, -20.0, 1, -0.1, -1.0, -0.1]
Gamma_indexes  = [0, 5, 0,
                  0, 6, 0,
                  1, 4, 0,
                  8, 4, 0,
                  0, 4, 1,
                  1, 7, 1,
                  10, 5, 1,
                  9, 4, 4,
                  11, 5, 5]

Gamma_dims     = [12, 12, 12]
Gamma_values   = [-20.0, -20, -1.0, -1.0, -2.0, -0.1, -0.1, 1, 1]
Gamma_indexes  = [0, 1, 4, 
                  1, 0, 4, 
                  8, 0, 4,
                  9, 4, 4, 
                  0, 0, 5,
                  10, 1, 5, 
                  11, 5, 5,
                  0, 0, 6, 
                  1, 1, 7]

Gamma : coo_tensor = coo_tensor(dimensions_ = Gamma_dims)
Gamma.val          = Gamma_values
Gamma.indices      = Gamma_indexes
vecGamma : list[coo_tensor] = [Gamma] * n


Lambda_dims    = [12, 12, 8, 8]
Lambda_values  = [120000.0]
Lambda_indexes = [1, 5, 1, 1]
Lambda : coo_tensor = coo_tensor(dimensions_ = Lambda_dims)
Lambda.val          = Lambda_values
Lambda.indices      = Lambda_indexes
vecLambda : list[coo_tensor] = [Lambda] * n

forcePattern    : np.ndarray = np.zeros(adjointSize)
vecForcePattern : list[np.ndarray] = [forcePattern] * n

initialCondition    : np.ndarray = np.zeros(adjointSize)
initialCondition[2] = 1.
initialCondition[3] = 1.
vecInitialCondition : list[np.ndarray] = [initialCondition] * n

Psi_dims    = [12, 12, 12, 12, 12]
Psi : coo_tensor = coo_tensor(dimensions_ = Psi_dims)
Psi.val          = [1]
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
print("Gradient: ", gradient)

""" plt.plot(gradient[0], gradient[1], 'r')
plt.plot(gradient[0], gradient[2], 'b')
plt.show() """

""" for i in range(len(gradient)-1):
    plt.plot(gradient[0], gradient[i+1])
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


