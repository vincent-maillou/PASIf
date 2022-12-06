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


PYBIND11_MODULE(PASIf, m) {
  m.doc() = "Parallel Accelerated Solver Interface";

  py::class_<__GpuDriver>(m, "__GpuDriver")
    .def(py::init<std::vector<std::vector<reel>>, uint>(),
      py::arg("excitationSet_"),
      py::arg("sampleRate_"))
    .def("__setSystems", &__GpuDriver::__setSystems, 
      py::arg("M_"),
      py::arg("B_"),
      py::arg("K_"),
      py::arg("Gamma_"),
      py::arg("Lambda_"),
      py::arg("ForcePattern_"),
      py::arg("InitialConditions_"))
    .def("__getAmplitudes", &__GpuDriver::__getAmplitudes);

}