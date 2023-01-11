# PASIf

## 1. User interface
### .1 Importing the module

PASIf is using __pybind11__ to interface with python. The python interface is defined in the file `PASIf.cpp` and the module is callable from python using the following commands:

    import sys
    sys.path.append('path/to/PASIf.so/build')
    import PASIf as pasif

You can then create a GpuDriver object using the following command:

    driver = pasif.GpuDriver(excitationSet, sampleRate)

### .2 GpuDriver class interface

        __GpuDriver(std::vector< std::vector<double> > excitationSet, 
                    uint sampleRate_)

> Constructor of the GpuDriver module. 
> - Takes a vector of excitations forces that can be parsed from the python side using python list or numpy array and the sample rate of the excitation files.
> - The constructor will initialize the GPU environment, allocate the memory for the excitation set and upload it on the device.

        int __loadExcitationsSet(std::vector< std::vector<double> > excitationSet_,
                                 uint sampleRate_)

> This function is used to load a new excitation set on the GPU that will replace the previous one. You can then replace the excitation during your simulation process.
> - Takes a vector of excitations forces that can be parsed from the python side using python list or numpy array and the sample rate of the excitation files.
> - The function will allocate the memory for the excitation set and upload it on the device.

        int __setSystems(std::vector< matrix > & M_,
                         std::vector< matrix > & B_,
                         std::vector< matrix > & K_,
                         std::vector< tensor > & Gamma_,
                         std::vector< matrix > & Lambda_,
                         std::vector< std::vector<reel> > & ForcePattern_,
                         std::vector< std::vector<reel> > & InitialConditions_)

> This function is used to set the system matrices and the initial conditions of the system.

        std::array<std::vector<reel>, 2> __getAmplitudes()

> By calling this function you will get the amplitudes of the system at the end of the simulation. This function will first load the parsed matrices and initial conditions on the GPU and then run the simulation using RK4 scheme.
> - The function will return a vector of vectors of doubles that contains the amplitudes of the system at the end of the simulation for each excitation file.

                   

