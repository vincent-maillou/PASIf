# Import Springtronics
import sys
sys.path.append('/home/louvet/Documents/02_code')


import Springtronics as spr

# Standard library imports
import matplotlib.pyplot as plt
import numpy as np
import time


USE_SOUND_FILE=True

system = spr.MechanicalSystem()
if USE_SOUND_FILE:
    m = .1
    k = (2*100*np.pi)**2*m
    b = 10*2*np.sqrt(k*m)
    m_cant = 10
    k_cant = (2*50*np.pi)**2*m_cant
    b_cant = .75*2*np.sqrt(k_cant*m_cant)

else:    
    m = 2
    b = 2
    k = 6
    m_cant = 10
    b_cant = 10
    k_cant = 5

gamma  = 1000
duffing  = 100000 #lambda

force = 1
filelength = 78001
numSteps=filelength
sr = 16000
x  = np.linspace(0, filelength/sr, filelength)

dofName = 'oscillator'
system.degreesOfFreedom[f'{dofName}'] = spr.ParametricVariable(m)
system.interactionPotentials[f'{dofName}_K'] =  spr.IntegerPotential(k)
system.interactionPotentials[f'{dofName}_K'].degreesOfFreedom[dofName] = 2
system.interactionPotentials[f'{dofName}_B'] =  spr.LocalDamping(dofName, b)


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

if not USE_SOUND_FILE:
    system.excitationSources['step'] = spr.DirectCInjectionSource(force)
    system.interactionPotentials[f'excitation'] = spr.Excitation(opticalDOF, 'step', 1.0)
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

# Probe string
system.probes['system_output'] = spr.WindowedA2Probe(opticalDOF,
                                                     startIndex=0,
                                                     endIndex=numSteps)

# Define an adjoint source (used to compute the gradient efficiently: https://en.wikipedia.org/wiki/Adjoint_state_method)
system.interactionPotentials['adjoint_source'] = system.probes['system_output'].makeAdjointSource()

# Probe cantilever
system.probes['cant_output'] = spr.WindowedA2Probe(mechanicalDOF,
                                                   startIndex=0,
                                                   endIndex=numSteps)

# Define an adjoint source (used to compute the gradient efficiently: https://en.wikipedia.org/wiki/Adjoint_state_method)
system.interactionPotentials['adjoint_source'] = system.probes['cant_output'].makeAdjointSource()

cppEnvironment = spr.CPPEnvironment(numSteps = int(numSteps),
                         timeStep = 1.0/sr,
                         numSweepSteps = 2,
                         numThreads=1)
                      
cppTrajectory = cppEnvironment.getTrajectories(system, initialConditions=np.zeros(2), deleteTemp=False)

# From now CUDA Env testing
vecSystem = [system]
displayCompute = True
displaySystem  = True
displaySolver  = True
cudaEnvironment = spr.CUDAEnvironment(vecSystem, 
                                      numSteps  = int(numSteps/2), 
                                      timeStep  = 2/sr,
                                      dCompute_ = displayCompute,
                                      dSystem_  = displaySystem,
                                      dSolver_  = displaySolver)

if not USE_SOUND_FILE:
    excitation = []
    for i in range(filelength):
        excitation.append(1*force)
    excitationSet = [excitation]

    cudaEnvironment.setExcitations(excitationSet, timeStep = 1.0/sr)
else:
    f=20
    #cudaEnvironment.setModulationBuffer(8*2*f, 195*f) 

print("---------------------------------------------------")

start = time.time()
amplitudes = cudaEnvironment.getAmplitudes(vecSystem)
stop = time.time()
print(f'Total getAmplitude() time: {stop-start} s')
print("Probes amplitudes: ", amplitudes)

start = time.time()
gpuTrajectory = cudaEnvironment.getTrajectory(vecSystem, saveSteps_=1)
stop = time.time()
print(f'Total getTrajectory() time: {stop-start} s')

cpp_oscilatorProbeEnergy        = round(cppTrajectory[-1, -2], 20)
cpp_stringCantileverProbeEnergy = round(cppTrajectory[-1, -1], 20)
gpu_oscilatorProbeEnergy        = round(amplitudes[0][0], 20)
gpu_oscilatorProbeEnergy        = round(gpuTrajectory[7, -1], 20)

gpu_stringCantileverProbeEnergy = round(amplitudes[0][1], 20)

print("CPU Oscilator probe energy: ", cpp_oscilatorProbeEnergy)
print("GPU Oscilator probe energy: ", gpu_oscilatorProbeEnergy)
print(f"Oscilator probe energy relative error: {100*np.abs(cpp_oscilatorProbeEnergy-gpu_oscilatorProbeEnergy)/cpp_oscilatorProbeEnergy}")

print("CPU cantilever probe energy: ", cpp_stringCantileverProbeEnergy)
print("GPU cantilever probe energy: ", gpu_stringCantileverProbeEnergy)
print(f"Cantilever probe energy relative error: {100*np.abs(cpp_stringCantileverProbeEnergy-gpu_stringCantileverProbeEnergy)/cpp_stringCantileverProbeEnergy}")


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

