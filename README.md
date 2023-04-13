# PASIf

## 1. Compilation

To comnpile the code using the Makefile you'll need to fulfill the following requirments:

- __build-essential__ package with a gcc version <= 11.x
- __python-dev__ package, make sure to match the dev package version with your python version. (3.10 have been used during the development)
- PASIf use __pybind11__ (https://github.com/pybind/pybind11) to link the C++/CUDA code as a python module. Make sure after cloning the project that the submodule have been downloaded as well:

        git submodule init
        git submodule update
  
  Make sure in the Makefile to make the __PYBIND11__ macro to match you'r installed version.

- To compile the CUDA code you'll need the Nvidia compiler __nvcc__. This come in the __NVIDIA HPC SDK__: https://developer.nvidia.com/hpc-sdk  
  This is installed by default in /opt/nvidia/, make sure to add the path to the compiler in you're Linux Path:

        export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/*version*/compilers/bin:$PATH

  Then make sure to match the __NVCCINCLUDE__ macro in the Makefile with your installed version of the HPC SDK.

### Other comments on the Makefile

Depending on the hardware targeted during the compilation make sure that the _-arch_ flag match the architecture of your hardware. This to get the best performances.  

The compilation process will generate a __.so__ python module in the ./build folder.

## 2. Benchmarking

The tools to benchmark Nvidia GPU kernels are __Nsight Compute__ and __Nsight systems__. They come with the __NVIDIA HPC SDK__ but you can also directly donwload the more recent version here: https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2023-2



### Command line tools

        # Run the code with NSYS event capture
        > nsys profile python3 FullInterfaceTesting.py

        # You can open the .nsys-rep output using the graphical interface (nsys-ui) or export it in a .sqlite format and open the result directly in the terminal
        > nsys export --output=rep_name.sqlite -t sqlite report.nsys-rep

        # Open the benchmark in the terminal
        > nsys stats *.sqlite

