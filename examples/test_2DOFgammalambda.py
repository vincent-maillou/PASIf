import sys
sys.path.append('/home/louvet/Documents/02_code/Springtronics/')

import Springtronics as spr
import Springtronics.HelperFunctions as hf

# Standard library imports
import matplotlib.pyplot as plt
import numpy as np
import time

USE_SOUND_FILE=True
TRAJ=False
GRAD= True
CHAIN=False

system = spr.MechanicalSystem()
if USE_SOUND_FILE:
    m = .05
    k = (2*500*np.pi)**2*m
    b = 10*2*np.sqrt(k*m)
    m_cant = 10
    k_cant = (2*50*np.pi)**2*m_cant
    b_cant = .75*2*np.sqrt(k_cant*m_cant)
    gamma  = 1e6
    duffing  = 0 #lambda
else:    
    m = 2
    b = 5
    k = 10
    m_cant = 200
    b_cant = 50
    k_cant = 500
    gamma  = 5
    duffing  = 0 #lambda


force = 1
filelength = 50000*3
numSteps=filelength
sr = 16000
x  = np.linspace(0, filelength/sr, filelength)
n_sites = 2
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

# # # # # spring
system.interactionPotentials[f'{dofName}_L'] =  spr.IntegerPotential(duffing)
system.interactionPotentials[f'{dofName}_L'].degreesOfFreedom[dofName] = 4
system.interactionPotentials[f'{dofName}_L'].strength.parameterized = True


opticalDOF = 'oscillator'
mechanicalDOF = 'cantilever'

system.interactionPotentials[f'OptoCoup_{opticalDOF}_{mechanicalDOF}'] =  spr.IntegerPotential(-gamma)
system.interactionPotentials[f'OptoCoup_{opticalDOF}_{mechanicalDOF}'].degreesOfFreedom[opticalDOF] = 2
system.interactionPotentials[f'OptoCoup_{opticalDOF}_{mechanicalDOF}'].degreesOfFreedom[mechanicalDOF] = 1
system.interactionPotentials[f'OptoCoup_{opticalDOF}_{mechanicalDOF}'].strength.parameterized = True


if CHAIN:
    system = spr.MechanicalSystem()
    n_sites=7
    k_scale_local_relative=100
    k_scale = 10
    gamma  = .01
    duffing  = 1 #lambda
    delay = .5
    delay_per_site = delay/n_sites
    mass = delay_per_site**2
    final_damping = k_scale * delay_per_site # equals sqrt(k_scale*m)
    damping = np.sqrt(mass*k_scale_local_relative*k_scale)


    for i in range(n_sites): # generate chain
        site_name = f'dof_{i}'
        if i==0: site_name='oscillator'
        elif i==(n_sites-1): site_name='cantilever'
        system.degreesOfFreedom[site_name] = spr.ParametricVariable(mass)
        
        # Make local stiffness for expressivity (and positive definiteness (not semi))
        system.interactionPotentials[f'{site_name}_K_l'] =  spr.IntegerPotential(0.5 * k_scale_local_relative*k_scale)
        system.interactionPotentials[f'{site_name}_K_l'].degreesOfFreedom[site_name] = 2
        system.interactionPotentials[f'{site_name}_K_l'].strength.parameterized = True
        system.interactionPotentials[f'{site_name}_B'] = spr.LocalDamping(site_name, damping)

        if i != (n_sites-1) and i!=int(n_sites/2): # connect to next site
            name_dof_2 = f'dof_{i+1}' if i!=(n_sites-2) else 'cantilever'
            hf.makeLinearCoupling(system, site_name, name_dof_2, 
                            k_scale)
            system.interactionPotentials[f'LinCoup_{site_name}_{name_dof_2}'].strength.parameterized = True
        elif i != (n_sites-1):
            name_dof_2 = f'dof_{i+1}' if i!=(n_sites-2) else 'cantilever'

            system.interactionPotentials[f'OptoCoup_{site_name}_{name_dof_2}'] =  spr.IntegerPotential(-gamma)
            system.interactionPotentials[f'OptoCoup_{site_name}_{name_dof_2}'].degreesOfFreedom[site_name] = 2
            system.interactionPotentials[f'OptoCoup_{site_name}_{name_dof_2}'].degreesOfFreedom[name_dof_2] = 1
            system.interactionPotentials[f'OptoCoup_{site_name}_{name_dof_2}'].strength.parameterized = True

            system.interactionPotentials[f'{site_name}_L'] =  spr.IntegerPotential(duffing)
            system.interactionPotentials[f'{site_name}_L'].degreesOfFreedom[site_name] = 4
            system.interactionPotentials[f'{site_name}_L'].strength.parameterized = True
        else: # terminal damping
            system.interactionPotentials[f'{site_name}_B'] = spr.LocalDamping(site_name, final_damping)
            system.interactionPotentials[f'{site_name}_B'].strength.parameterized = True
    #optomechanical coupling (gamma)

if not USE_SOUND_FILE:
    system.excitationSources['step'] = spr.DirectCInjectionSource(f'.01+.1/(1+t)')
    system.interactionPotentials[f'excitation'] = spr.Excitation('oscillator', 'step', 1000)
    trainingSet = [0]
else:
    trainingSet = ['/home/louvet/Documents/01_data/00_one_to_four/training/soundfile_2500']#, '/home/louvet/Documents/01_data/00_one_to_four/training/soundfile_2812', '/home/louvet/Documents/01_data/00_one_to_four/training/soundfile_2020']

    # trainingSet = ['/home/louvet/Documents/01_data/00_one_to_four/training/soundfile_1010']
    # trainingSet = ['/home/louvet/Documents/01_data/00_one_to_four/training/soundfile_1012']

    sr = 16000*16
    excitationFileLength = 78001
    numSteps= int(153600)

    system.excitationSources['soundData2'] = spr.BinaryFileSource(fileList=trainingSet,
                                                    fileLength=excitationFileLength,
                                                    fileDataType='double',
                                                    log2Upsampling=2)
    system.interactionPotentials[f'excitation2'] = spr.Excitation(opticalDOF, 'soundData2', 1e3)

system.probes[f'probe__squared'] = spr.SimpleA2Probe(mechanicalDOF)
system.interactionPotentials['probe__squared_grad'] = system.probes['probe__squared'].makeAdjointSource()
# system.probes[f'probe__multi'] = spr.SimpleA2MultiProbe(f'cantilever', f'oscillator')

# for i in range(1, 5):
#     system.probes[f'probe__multi_{i}'] = spr.SimpleA2MultiProbe(f'dof_{i}', f'dof_{i+1}')


cppEnvironment = spr.CPPEnvironment(numSteps = int(numSteps),
                         timeStep = 1.0/sr,
                         numSweepSteps = len(trainingSet),
                         numThreads=1)
if TRAJ:
    start  = time.time()
    cppTrajectory = cppEnvironment.getTrajectories(system)
    print(f'CPP trajectory time is {time.time()-start}')

start  = time.time()
cppAmplitude = cppEnvironment.getAmplitudes(system, deleteTemp=False)
print(f'CPP amplitude time is {time.time()-start}')

if GRAD:
    start  = time.time()
    cppGrad = np.array(cppEnvironment.getGradients(system, deleteTemp=False))
    print(f'CPP grad time is {time.time()-start}')

# From now CUDA Env testing
vecSystem = [system]
displayCompute = False
displaySystem  = False
displaySolver  = False

cudaEnvironment = spr.CUDAEnvironment(vecSystem, 
                                      numSteps  = numSteps, 
                                      timeStep  = 1/sr,
                                      dCompute_ = displayCompute,
                                      dSystem_  = displaySystem,
                                      dSolver_  = displaySolver)


if not USE_SOUND_FILE:
    excitation = []
    for i in range(filelength):
        excitation.append(.01+.1/(1+i/sr))
    excitationSet = [excitation]
    cudaEnvironment.setExcitations(excitationSet, timeStep = 1.0/sr)
 

print("---------------------------------------------------")

start = time.time()
amplitudes = cudaEnvironment.getAmplitudes(vecSystem)
stop = time.time()
print(f'Total getAmplitude() time: {stop-start} s')

if TRAJ:
    start = time.time()
    gpuTrajectory = cudaEnvironment.getTrajectories(vecSystem, saveSteps_=1)
    stop = time.time()
    print(f'Total getTrajectory() time: {stop-start} s')

if GRAD:
    start = time.time()
    gpuGrad = cudaEnvironment.getGradients([system])
    stop = time.time()
    print(f'Total getGradients() time: {stop-start} s')

cpp_probe = cppAmplitude[0]
gpu_probe = np.array(amplitudes)


if GRAD:
    i = 0
    for potential, p in system.interactionPotentials.items():
        if p.strength.parameterized:
            print(f'Potential {potential}, CPU: {cppGrad[0][i]}, GPU: {gpuGrad[0][i]}, relative error= {(cppGrad[0][i]-gpuGrad[0][i])/cppGrad[0][i]*100}')
            i+= 1


for i in range(gpu_probe.shape[1]):
    relative_error = 100*np.abs((cpp_probe[:,i]-gpu_probe[:,i])/cpp_probe[:,i])
    # relative_error_spy = 100*np.abs((cpp_probe[:,i]-scipy_probe[i])/cpp_probe[:,i])
    print(f"Probe {i}, gpu={gpu_probe[:,i]}, cpu={cpp_probe[:,i]}: {relative_error}")

if TRAJ:
    for i in range(n_sites):
        fig, axs = plt.subplots(2, constrained_layout=True)
        fig.suptitle(f'DOF {i}', fontsize=16)
        axs[0].set_title('CPP Trajectory')
        axs[0].plot(gpuTrajectory[0], cppTrajectory[:,i])
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('amplitude')

        axs[1].set_title('CUDA Trajectory')
        axs[1].plot(gpuTrajectory[0], gpuTrajectory[i+1])
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('amplitude')


    for i in range(gpu_probe.shape[1]):
        fig, axs = plt.subplots(2, constrained_layout=True)
        fig.suptitle(f'Probe {i}', fontsize=16)
        axs[0].set_title('CPP Trajectory')
        axs[0].plot(gpuTrajectory[0], cppTrajectory[:,n_sites*2+i])
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('amplitude')

        axs[1].set_title('CUDA Trajectory')
        axs[1].plot(gpuTrajectory[0], gpuTrajectory[i+1+2*n_sites+gpu_probe.shape[1]])
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('amplitude')

    plt.show()


