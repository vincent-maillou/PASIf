import sys
sys.path.append('/home/louvet/Documents/02_code/Springtronics/')

import Springtronics as spr
import Springtronics.HelperFunctions as hf

# Standard library imports
import matplotlib.pyplot as plt
import numpy as np
import time
import cppyy

USE_SOUND_FILE=True
TRAJ=True
GRAD= False
CHAIN=True

system = spr.MechanicalSystem()
if USE_SOUND_FILE:
    m = .05
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

gamma  = 1e6
duffing  = 1e10 #lambda

force = 1
filelength = 5000
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

# # # # spring
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
    delay = .06
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
    system.excitationSources['step'] = spr.DirectCInjectionSource(f'{force}*(1-t/(NUMSTEPS*TIMESTEP))')
    system.interactionPotentials[f'excitation'] = spr.Excitation(opticalDOF, 'step', 1e7)
    trainingSet = [0]

else:
    trainingSet = ['/home/louvet/Documents/01_data/00_one_to_four/training/soundfile_2500']#, '/home/louvet/Documents/01_data/00_one_to_four/training/soundfile_2812', '/home/louvet/Documents/01_data/00_one_to_four/training/soundfile_2020']

    # trainingSet = ['/home/louvet/Documents/01_data/00_one_to_four/training/soundfile_1010']
    # trainingSet = ['/home/louvet/Documents/01_data/00_one_to_four/training/soundfile_1012']

    sr = 16000*16*2
    excitationFileLength = 78001
    numSteps= int(153600*2)

    system.excitationSources['soundData2'] = spr.BinaryFileSource(fileList=trainingSet,
                                                    fileLength=excitationFileLength,
                                                    fileDataType='double',
                                                    log2Upsampling=3)
    system.interactionPotentials[f'excitation2'] = spr.Excitation(opticalDOF, 'soundData2', 1e3)

system.probes[f'probe__squared'] = spr.SimpleA2Probe(f'cantilever')
system.interactionPotentials['probe__squared_grad'] = system.probes['probe__squared'].makeAdjointSource()
system.probes[f'probe__multi'] = spr.SimpleA2MultiProbe(f'cantilever', f'oscillator')

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
        excitation.append(force*(1-i/numSteps))
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

# # SCIPY:
# from scipy.integrate import solve_ivp
# from scipy.interpolate import interp1d

# K, Gamma, Lambda, F, X0 = cudaEnvironment.pasif.getMatrices()
# data = np.fromfile(trainingSet[0])
# times = np.linspace(0, len(data)/(16000*8), len(data))
# interpolator = interp1d(times, data)


# derivative_string = 'void derivatives (double* source, double* destination, double excitationForce[], int sweepIndex) {\n'
# system.enumerateVariables()
# derivative_string += system.generateDerivatives()
# derivative_string += '}'
# cppyy.cppdef(derivative_string)

# n_probes = gpu_probe.shape[1]
# n_dofs = n_sites*2+n_probes
# destination = np.zeros(n_dofs)
# source = np.random.rand(n_dofs)
# excitation_strength = np.array([0]).astype('double')


# map_cpp = np.arange(2*n_sites+2)
# map_scipy = np.concatenate([
#                 np.arange(n_sites),
#                 np.arange(n_sites) + n_sites+2,
#                 np.arange(2)+n_sites*2+2
# ])



# class A:
#     values = []
#     times = []
#     def __init__(self):
#         pass

#     def interp(self, new_array):
#         res = []
#         self.values = np.array(self.values)
#         for i in range(self.values.shape[1]):
#             res.append(np.interp(new_array, self.times, self.values[:, i]))

#         return np.array(res)

# error_derivative = A()

# def func(t, x, saver):
#     # global error_derivative
#     # res = K@np.copy(x)

#     # for i, val in enumerate(Gamma.val):
#     #     sl, row, col = Gamma.indices[i*3:(i+1)*3]
#     #     res[sl] += val*x[row]*x[col]

#     # for i, val in enumerate(Lambda.val):
#     #     hsl, sl, col, row = Lambda.indices[i*4:(i+1)*4]
#     #     res[hsl] += val*x[sl]*x[col]*x[row]

#     # for i, val in enumerate(F):
#     #     if not np.isclose(val, 0):
#     #         res[i] += val*interpolator(t)

#     res_2 = np.zeros(len(x))
#     excitation_strength[0] = interpolator(t)
#     source[map_cpp] = x[map_scipy]
#     cppyy.gbl.derivatives(source, destination, excitation_strength, 0)
#     res_2[map_scipy] = destination[map_cpp]

#     res_2[n_sites:n_sites+n_probes] = x[n_sites*2+n_probes:-1]
#     res_2[n_sites*2+n_probes] = x[6]**2
#     res_2[n_sites*+2+n_probes+1] = x[6]*x[0]

#     # saver.values.append((res-res_2)**2)
#     # saver.times.append(t)
#     return res_2

# start = time.time()
# sp_res = solve_ivp(func, (0, numSteps/sr), X0, t_eval=np.linspace(0, numSteps/sr, numSteps), method='RK45',
#                     first_step=1/sr, max_step=1/sr, args=[error_derivative])

# # error = error_derivative.interp(gpuTrajectory[0])
# # print(error.shape)

# print(f'Scipy time: {time.time()-start}')
# scipy_probe = sp_res.y[-2:,-1]


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

        # axs[2].set_title('Scipy Trajectory')
        # axs[2].plot(gpuTrajectory[0], sp_res.y[i])
        # axs[2].set_xlabel('time (s)')
        # axs[2].set_ylabel('amplitude')

        # fig, ax = plt.subplots(1)
        # ax.plot(gpuTrajectory[0], error[i])
        # ax.set_title(f'RMS error DOF {i}')
        # ax.set_xlabel('time (s)')
        # ax.set_ylabel('error')

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

        # axs[2].set_title('Scipy Trajectory')
        # axs[2].plot(gpuTrajectory[0], sp_res.y[i+2*n_sites+gpu_probe.shape[1]])
        # axs[2].set_xlabel('time (s)')
        # axs[2].set_ylabel('amplitude')

        # fig, ax = plt.subplots(1)
        # ax.plot(gpuTrajectory[0], error[i+2*n_sites+gpu_probe.shape[1]])
        # ax.set_title(f'RMS error probe {i}')
        # ax.set_xlabel('time (s)')
        # ax.set_ylabel('error')

    plt.show()

    raise a


    fig, axs = plt.subplots(3, constrained_layout=True)
    fig.suptitle('String trajectory', fontsize=16)
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
    fig.suptitle('Cantilever trajectory', fontsize=16)
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


    del cudaEnvironment
