import sys
sys.path.append('../')

import Springtronics as spr
import matplotlib.pyplot as plt
import numpy as np

system = spr.MechanicalSystem()
m = 2
b= 2
k= 6
force = 1

filelength = 78001
sr = 16000
x = np.linspace(0, filelength/sr, filelength)

dofName = 'oscillator'
system.degreesOfFreedom[f'{dofName}'] = spr.ParametricVariable(m)
system.interactionPotentials[f'{dofName}_K'] =  spr.IntegerPotential(k)
system.interactionPotentials[f'{dofName}_K'].degreesOfFreedom[dofName] = 2
system.interactionPotentials[f'{dofName}_B'] =  spr.LocalDamping(dofName, b)
m_cant = 10
b_cant = 10
k_cant = 5
gamma = 10
duffing  = 10000 #lambda

dofName = 'cantilever'
system.degreesOfFreedom[f'{dofName}'] = spr.ParametricVariable(m_cant)
system.interactionPotentials[f'{dofName}_K'] =  spr.IntegerPotential(k_cant)
system.interactionPotentials[f'{dofName}_K'].degreesOfFreedom[dofName] = 2
#spring
system.interactionPotentials[f'{dofName}_L'] =  spr.IntegerPotential(duffing)
system.interactionPotentials[f'{dofName}_L'].degreesOfFreedom[dofName] = 4
#duffing potential (lambda)
system.interactionPotentials[f'{dofName}_B'] =  spr.LocalDamping(dofName, b_cant)

opticalDOF = 'oscillator'
mechanicalDOF = 'cantilever'

system.interactionPotentials[f'OptoCoup_{opticalDOF}_{mechanicalDOF}'] =  spr.IntegerPotential(-gamma)
system.interactionPotentials[f'OptoCoup_{opticalDOF}_{mechanicalDOF}'].degreesOfFreedom[opticalDOF] = 2
system.interactionPotentials[f'OptoCoup_{opticalDOF}_{mechanicalDOF}'].degreesOfFreedom[mechanicalDOF] = 1
#optomechanical coupling (gamma)

system.excitationSources['step'] = spr.DirectCInjectionSource(force)
system.interactionPotentials[f'excitation'] = spr.Excitation(opticalDOF, 'step', 1.0)




def buildSets(trainingDataFolder, numTrainingFiles, numTestFiles):
    # Building the training set
    trainingSet = []
    for i in range(numTrainingFiles):
        trainingSet.append(f'{trainingDataFolder}soundfile_{i*4+1}')
        trainingSet.append(f'{trainingDataFolder}soundfile_{i*4+2}')
        
    # Building the test set
    testSet = []
    for i in range(numTestFiles):
        testSet.append(f'{trainingDataFolder}soundfile_{(i+numTrainingFiles)*4+1}')
        testSet.append(f'{trainingDataFolder}soundfile_{(i+numTrainingFiles)*4+2}')
        
    return trainingSet, testSet


trainingDataFolder = '/home/vincent-maillou/Documents/4_Travail/AMOLF_Internship/1_Code/Developpement/SpeechRecognition/Data/selection_v0.01/training/'
numTrainingFiles = 256 
numTestFiles = 128 

trainingSet, testSet = buildSets(trainingDataFolder, numTrainingFiles, numTestFiles)
excitationFileLength = 76800

system.excitationSources['soundData2'] = spr.BinaryFileSource(fileList=trainingSet,
                                                  fileLength=excitationFileLength,
                                                  fileDataType='double',
                                                  modulationFrequency=10500.0,
                                                  log2Upsampling=2)
system.interactionPotentials[f'excitation2'] = spr.Excitation(opticalDOF, 'soundData2', 1.0)









# Probe string
system.probes['system_output'] = spr.WindowedA2Probe(opticalDOF,
                                                    startIndex=0,
                                                    endIndex=filelength)

# Define an adjoint source (used to compute the gradient efficiently: https://en.wikipedia.org/wiki/Adjoint_state_method)
system.interactionPotentials['adjoint_source'] = system.probes['system_output'].makeAdjointSource()

# Probe cantilever
system.probes['cant_output'] = spr.WindowedA2Probe(mechanicalDOF,
                                                    startIndex=0,
                                                    endIndex=filelength)

# Define an adjoint source (used to compute the gradient efficiently: https://en.wikipedia.org/wiki/Adjoint_state_method)
system.interactionPotentials['adjoint_source'] = system.probes['cant_output'].makeAdjointSource()

env = spr.CPPEnvironment(numSteps = filelength,
                           timeStep = 1.0/sr,
                           numSweepSteps = 1,
                           numThreads=1)
                      
traj = env.getTrajectories(system, initialConditions=np.zeros(2), deleteTemp=False)

""" plt.plot(x, traj[:,0])
plt.title(f'Final energy: {round(traj[-1, -2], 8)}\nFinal amplitude: {round(traj[-1,0], 8)}')
plt.show() """



# From now CUDA Env testing
vecSystem = [system]
excitation = []
for i in range(filelength):
    excitation.append(1)
excitationSet = [excitation]

gpuenv = spr.CUDAEnvironment(vecSystem, 
                             numSteps = filelength, 
                             timeStep = 1.0/sr)


# gpuenv.setModulationBuffer(8, 195)

gpuenv.setExcitations(excitationSet, timeStep = 1.0/sr)


import time
start = time.time()
amplitudes = gpuenv.getAmplitudes(vecSystem)
stop = time.time()
print(f'Total getAmplitude() time: {stop-start} s')

print("Probes amplitudes: ", amplitudes)