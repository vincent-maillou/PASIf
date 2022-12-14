NVCC := nvcc
NVCCFLAGS := -std=c++11 -O3 -g -arch=sm_86 -Xcompiler -Wall
NVCCLIB := -lcublas -lcusparse
NVCCINCLUDE := -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.2/compilers/bin -L/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/math_libs/lib64

PYBIND11 := -I/home/vincent-maillou/anaconda3/include/python3.9 -I/home/vincent-maillou/anaconda3/include/python3.9 -Iextern/pybind11/include

PBUILD := build/
PSRC := src/
PREPORTS := reports/

all: $(PBUILD)PASIf.so $(PBUILD)utest.out

$(PBUILD)PASIf.so: $(PBUILD)PASIf.o $(PBUILD)__GpuDriver.o $(PBUILD)kernels.o $(PBUILD)helpers.o 
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC $(NVCCINCLUDE) $(NVCCLIB) -shared -o $@ $^

$(PBUILD)utest.out: $(PSRC)utest.cu $(PBUILD)kernels.o $(PBUILD)helpers.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC $(NVCCINCLUDE) $(NVCCLIB) -o $@ $^

$(PBUILD)PASIf.o: $(PSRC)PASIf.cpp $(PBUILD)helpers.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c $(python3-config --includes) $(PYBIND11) -o $@ $<

$(PBUILD)__GpuDriver.o: $(PSRC)__GpuDriver.cu $(PSRC)__GpuDriver.cuh $(PBUILD)helpers.o $(PBUILD)kernels.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC $(NVCCINCLUDE) $(NVCCLIB) -c -o $@ $<

$(PBUILD)kernels.o: $(PSRC)kernels.cu $(PSRC)kernels.cuh $(PBUILD)helpers.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC $(NVCCINCLUDE) $(NVCCLIB) -c -o $@ $<

$(PBUILD)helpers.o: $(PSRC)helpers.cu $(PSRC)helpers.cuh
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC $(NVCCINCLUDE) $(NVCCLIB) -c -o $@ $<

profile: all
	nsys profile python3 testanddebug.py
	

clean:
	rm -f $(PBUILD)*.o $(PBUILD)*.so

cleanreports:
	- rm -f $(PREPORTS)*.sqlite
	- rm -f $(PREPORTS)*.nsys-rep