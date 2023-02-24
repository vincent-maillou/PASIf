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
      

    //            Forward system interface
    .def("_setFwdB", &__GpuDriver::_setFwdB,
      py::arg("values_"),
      py::arg("row_"),
      py::arg("col_"),
      py::arg("n_"))
    .def("_setFwdK", &__GpuDriver::_setFwdK,
      py::arg("values_"),
      py::arg("row_"),
      py::arg("col_"),
      py::arg("n_"))
    .def("_setFwdGamma", &__GpuDriver::_setFwdGamma,
      py::arg("dimensions_"),
      py::arg("values_"),
      py::arg("indices_"))
    .def("_setFwdLambda", &__GpuDriver::_setFwdLambda,
      py::arg("dimensions_"),
      py::arg("values_"),
      py::arg("indices_"))
    .def("_setFwdForcePattern", &__GpuDriver::_setFwdForcePattern,
      py::arg("forcePattern_"))
    .def("_setFwdInitialConditions", &__GpuDriver::_setFwdInitialConditions,
      py::arg("initialConditions_"))

    .def("_allocateSystemOnDevice", &__GpuDriver::_allocateSystemOnDevice)


    //            Backward system interface 


    //            Compute options interface 
    .def("_loadExcitationsSet", &__GpuDriver::_loadExcitationsSet,
      py::arg("excitationSet_"),
      py::arg("sampleRate_"))
    .def("_setInterpolationMatrix", &__GpuDriver::_setInterpolationMatrix,
      py::arg("interpolationMatrix_"),
      py::arg("interpolationWindowSize_"))
    .def("_setModulationBuffer", &__GpuDriver::_setModulationBuffer,
      py::arg("modulationBuffer_"))
      

    //            Solvers interface
    .def("_getAmplitudes", &__GpuDriver::_getAmplitudes)
    .def("_getTrajectory", &__GpuDriver::_getTrajectory,
      py::arg("saveSteps_") = 1)
    .def("_getGradient", &__GpuDriver::_getGradient,
      py::arg("adjointSize_"),
      py::arg("save"));
}