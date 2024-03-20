import sys
sys.path.append('/home/louvet/Documents/02_code/Springtronics/')

import Springtronics as spr

# Standard library imports
import matplotlib.pyplot as plt
import numpy as np
import time


USE_SOUND_FILE=True

system = spr.MechanicalSystem()
if USE_SOUND_FILE:
    m = .1
    k = (2*500*np.pi)**2*m
    b = 10*2*np.sqrt(k*m)
    m_cant = 10
    k_cant = (2*50*np.pi)**2*m_cant
    b_cant = .75*2*np.sqrt(k_cant*m_cant)

else:    
    m = 200
    b = 200
    k = 600
    m_cant = 1000
    b_cant = 1000
    k_cant = 500

gamma  = 10000
duffing  = 1000 #lambda

force = 1
filelength = 5000
numSteps=filelength
sr = 16000
x  = np.linspace(0, filelength/sr, filelength)

dofName = 'oscillator'
system.degreesOfFreedom[f'{dofName}'] = spr.ParametricVariable(m)
system.interactionPotentials[f'{dofName}_K'] =  spr.IntegerPotential(k)
system.interactionPotentials[f'{dofName}_K'].degreesOfFreedom[dofName] = 2
system.interactionPotentials[f'{dofName}_K'].strength.parameterized = True
system.interactionPotentials[f'{dofName}_B'] =  spr.LocalDamping(dofName, b)
system.interactionPotentials[f'{dofName}_B'].strength.parameterized = True


dofName = 'cantilever'
system.degreesOfFreedom[f'{dofName}'] = spr.ParametricVariable(m_cant)
system.interactionPotentials[f'{dofName}_K'] =  spr.IntegerPotential(k_cant)
system.interactionPotentials[f'{dofName}_K'].degreesOfFreedom[dofName] = 2
system.interactionPotentials[f'{dofName}_K'].strength.parameterized = True
system.interactionPotentials[f'{dofName}_B'] =  spr.LocalDamping(dofName, b_cant)
system.interactionPotentials[f'{dofName}_B'].strength.parameterized = True

# spring
system.interactionPotentials[f'{dofName}_L'] =  spr.IntegerPotential(duffing)
system.interactionPotentials[f'{dofName}_L'].degreesOfFreedom[dofName] = 4
system.interactionPotentials[f'{dofName}_L'].strength.parameterized = True


opticalDOF = 'oscillator'
mechanicalDOF = 'cantilever'

system.interactionPotentials[f'OptoCoup_{opticalDOF}_{mechanicalDOF}'] =  spr.IntegerPotential(-gamma)
system.interactionPotentials[f'OptoCoup_{opticalDOF}_{mechanicalDOF}'].degreesOfFreedom[opticalDOF] = 2
system.interactionPotentials[f'OptoCoup_{opticalDOF}_{mechanicalDOF}'].degreesOfFreedom[mechanicalDOF] = 1
system.interactionPotentials[f'OptoCoup_{opticalDOF}_{mechanicalDOF}'].strength.parameterized = True

#optomechanical coupling (gamma)

if not USE_SOUND_FILE:
    system.excitationSources['step'] = spr.DirectCInjectionSource(force)
    system.interactionPotentials[f'excitation'] = spr.Excitation(opticalDOF, 'step', 1e5)

else:
    trainingSet = ['/home/louvet/Documents/01_data/00_one_to_four/training/soundfile_2500']

    sr = 16000*16
    excitationFileLength = 78001
    numSteps= 153600

    system.excitationSources['soundData2'] = spr.BinaryFileSource(fileList=trainingSet,
                                                    fileLength=excitationFileLength,
                                                    fileDataType='double',
                                                    #modulationFrequency=10502.564103,
                                                    log2Upsampling=2)
    system.interactionPotentials[f'excitation2'] = spr.Excitation(opticalDOF, 'soundData2', 1)

# # Probe string
# system.probes['system_output'] = spr.WindowedA2Probe(opticalDOF,
#                                                      startIndex=0,
#                                                      endIndex=numSteps)

# # Define an adjoint source (used to compute the gradient efficiently: https://en.wikipedia.org/wiki/Adjoint_state_method)
# system.interactionPotentials['adjoint_source'] = system.probes['system_output'].makeAdjointSource()

# Probe cantilever
system.probes['cant_output'] = spr.WindowedA2Probe(mechanicalDOF,
                                                   startIndex=0,
                                                   endIndex=numSteps)

# Define an adjoint source (used to compute the gradient efficiently: https://en.wikipedia.org/wiki/Adjoint_state_method)
system.interactionPotentials['adjoint_source'] = system.probes['cant_output'].makeAdjointSource()

cppEnvironment = spr.CPPEnvironment(numSteps = int(numSteps),
                         timeStep = 1.0/sr,
                         numSweepSteps = 1,
                         numThreads=1)
start  = time.time()
cppTrajectory = cppEnvironment.getTrajectories(system)
print(f'CPP trajectory time is {time.time()-start}')
start  = time.time()
cppAmplitude = cppEnvironment.getAmplitudes(system)
print(f'CPP amplitude time is {time.time()-start}')
start  = time.time()
cppGrad = np.array(cppEnvironment.getGradients(system, deleteTemp=False))
print(f'CPP grad time is {time.time()-start}')

# From now CUDA Env testing
vecSystem = [system]
displayCompute = False
displaySystem  = False
displaySolver  = False
cudaEnvironment = spr.CUDAEnvironment(vecSystem, 
                                      numSteps  = int(numSteps/2) if USE_SOUND_FILE else numSteps, 
                                      timeStep  = 2/sr if USE_SOUND_FILE else 1/sr,
                                      dCompute_ = displayCompute,
                                      dSystem_  = displaySystem,
                                      dSolver_  = displaySolver)

if not USE_SOUND_FILE:
    excitation = []
    for i in range(filelength):
        excitation.append(1*force)
    excitationSet = [excitation]
    cudaEnvironment.setExcitations(excitationSet, timeStep = 1.0/sr)
 

print("---------------------------------------------------")

start = time.time()
amplitudes = cudaEnvironment.getAmplitudes(vecSystem)
stop = time.time()
print(f'Total getAmplitude() time: {stop-start} s')
print("Probes amplitudes: ", amplitudes)

start = time.time()
gpuTrajectory = cudaEnvironment.getTrajectories(vecSystem, saveSteps_=1)
stop = time.time()
print(f'Total getTrajectory() time: {stop-start} s')

start = time.time()
gpuGrad = cudaEnvironment.getGradients([system])
stop = time.time()
print(f'Total getGradients() time: {stop-start} s')

cpp_stringCantileverProbeEnergy = round(cppTrajectory[-1, -1], 20)
gpu_stringCantileverProbeEnergy = round(amplitudes[0][0], 20)


print(cppAmplitude)
print(amplitudes)
print("CPU cantilever probe energy: ", cpp_stringCantileverProbeEnergy)
print("GPU cantilever probe energy: ", gpu_stringCantileverProbeEnergy)
print(f"Cantilever probe energy relative error: {100*np.abs(cpp_stringCantileverProbeEnergy-gpu_stringCantileverProbeEnergy)/cpp_stringCantileverProbeEnergy}")
print()
print(f"CPU gradient: {cppGrad}")
print(f"GPU gradient: {gpuGrad}")
print(f"Relative error {np.abs(cppGrad-gpuGrad)/np.abs(cppGrad)*100} %")

#   ----- Plotting -----   #
if gpuTrajectory.shape[1]!=cppTrajectory.shape[0]:
    new_gpu_traj = []
    new_x = np.linspace(0, gpuTrajectory[0,-1], cppTrajectory.shape[0])
    for i in range(gpuTrajectory.shape[0]):
        new_gpu_traj.append(np.interp(new_x, gpuTrajectory[0,:], gpuTrajectory[i, :]))

    gpuTrajectory = np.array(new_gpu_traj)




fig, axs = plt.subplots(3, constrained_layout=True)
fig.suptitle('Oscilator trajectory', fontsize=16)
axs[0].set_title('CPP Trajectory')
axs[0].plot(gpuTrajectory[0], cppTrajectory[:,0])
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('amplitude')

axs[1].set_title('CUDA Trajectory')
axs[1].plot(gpuTrajectory[0], gpuTrajectory[1])
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel('amplitude')

axs[2].set_title('Error')
axs[2].plot(gpuTrajectory[0], abs(gpuTrajectory[1]-cppTrajectory[:,0]))
axs[2].set_xlabel('time (s)')
axs[2].set_ylabel('amplitude')

fig, axs = plt.subplots(3, constrained_layout=True)
fig.suptitle('String cantilever trajectory', fontsize=16)
axs[0].set_title('CPP Trajectory')
axs[0].plot(gpuTrajectory[0], cppTrajectory[:,1])
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('amplitude')

axs[1].set_title('CUDA Trajectory')
axs[1].plot(gpuTrajectory[0], gpuTrajectory[2])
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel('amplitude')

axs[2].set_title('Error')
axs[2].plot(gpuTrajectory[0], abs(gpuTrajectory[2]-cppTrajectory[:,1]))
axs[2].set_xlabel('time (s)')
axs[2].set_ylabel('amplitude')
plt.show()   
