NVCC := nvcc
NVCCFLAGS := -std=c++11 -O3 -g -arch=sm_86 -Xcompiler -Wall #-lineinfo
NVCCLIB := -lcublas -lcusparse
NVCCINCLUDE := -I/usr/local/cuda-12.3/bin -L/usr/local/cuda-12.3/lib64/

PYBIND11 := -I/usr/include/python3.10 -Iextern/pybind11/include

PBUILD := build/
PSRC := src/
PREPORTS := reports/

all: $(PBUILD)PASIfgpu.so 

$(PBUILD)PASIfgpu.so: $(PBUILD)PASIfgpu.o $(PBUILD)__GpuDriver.o $(PBUILD)kernels.o $(PBUILD)helpers.o 
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC $(NVCCINCLUDE) $(NVCCLIB) -shared -o $@ $^

$(PBUILD)utest.out: $(PSRC)utest.cu $(PBUILD)kernels.o $(PBUILD)helpers.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC $(NVCCINCLUDE) $(NVCCLIB) -o $@ $^

$(PBUILD)PASIfgpu.o: $(PSRC)PASIfgpu.cpp $(PBUILD)helpers.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c $(python3-config --includes) $(PYBIND11) -o $@ $<

$(PBUILD)__GpuDriver.o: $(PSRC)__GpuDriver.cu $(PSRC)__GpuDriver.cuh $(PBUILD)helpers.o $(PBUILD)kernels.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC $(NVCCINCLUDE) $(NVCCLIB) -c -o $@ $<

$(PBUILD)kernels.o: $(PSRC)kernels.cu $(PSRC)kernels.cuh $(PBUILD)helpers.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC $(NVCCINCLUDE) $(NVCCLIB) -c -o $@ $<

$(PBUILD)helpers.o: $(PSRC)helpers.cu $(PSRC)helpers.cuh
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC $(NVCCINCLUDE) $(NVCCLIB) -c -o $@ $<

profile: all
	nsys profile python3 FullInterfaceTesting.py
	

clean:
	rm -f $(PBUILD)*.o $(PBUILD)*.so

cleanreports:
	- rm -f $(PREPORTS)*.sqlite
	- rm -f $(PREPORTS)*.nsys-rep
