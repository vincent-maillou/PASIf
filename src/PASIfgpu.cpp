/**
 * @file PASIf.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-11-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
namespace py = pybind11;

#include "helpers.cuh"
#include "__GpuDriver.cuh"



PYBIND11_MODULE(PASIfgpu, m) {
  m.doc() = "Parallel Accelerated Solver Interface";

  py::class_<__GpuDriver>(m, "__GpuDriver")
    .def(py::init<std::vector<std::vector<double>>, uint, uint, bool, bool, bool>(),
      py::arg("excitationSet_"),
      py::arg("sampleRate_"),
      py::arg("numsteps_"),
      py::arg("dCompute_") = false,
      py::arg("dSystem_")  = false,
      py::arg("dSolver_")  = false)
      
    .def("_loadExcitationsSet", &__GpuDriver::_loadExcitationsSet,
      py::arg("excitationSet_"),
      py::arg("sampleRate_"))

    // Setters for the system definition
    .def("_setB", &__GpuDriver::_setB,
      py::arg("B_"))
    .def("_setK", &__GpuDriver::_setK,
      py::arg("K_"))
    .def("_setGamma", &__GpuDriver::_setGamma,
      py::arg("Gamma_"))
    .def("_setLambda", &__GpuDriver::_setLambda,
      py::arg("Lambda_"))
    .def("_setForcePattern", &__GpuDriver::_setForcePattern,
      py::arg("ForcePattern_"))
    .def("_setInitialConditions", &__GpuDriver::_setInitialConditions,
      py::arg("InitialConditions_"))
    .def("_setInterpolationMatrix", &__GpuDriver::_setInterpolationMatrix,
      py::arg("interpolationMatrix_"),
      py::arg("interpolationWindowSize_"))
    .def("_setModulationBuffer", &__GpuDriver::_setModulationBuffer,
      py::arg("modulationBuffer_"))

    .def("_allocateOnDevice", &__GpuDriver::_allocateOnDevice)
    .def("_displaySimuInfos", &__GpuDriver::_displaySimuInfos)

    .def("_getAmplitudes", &__GpuDriver::_getAmplitudes)
    .def("_getTrajectory", &__GpuDriver::_getTrajectory,
      py::arg("saveSteps_") = 1)
    .def("_getGradient", &__GpuDriver::_getGradient,
      py::arg("globalAdjointSize_"),
      py::arg("save"));
}