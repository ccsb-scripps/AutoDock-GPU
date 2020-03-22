#ifndef SPACE_SETTINGS_HPP
#define SPACE_SETTINGS_HPP
#include <Kokkos_Core.hpp>

// Declare the memory and execution spaces.
#if USE_GPU == 1
using MemSpace = Kokkos::CudaSpace;
using ExSpace = Kokkos::Cuda;
shouldnt be here yet
#else
using MemSpace = Kokkos::HostSpace;
#if USE_OMP == 1
using ExSpace = Kokkos::OpenMP;
shouldnt be here yet
#else
using ExSpace = Kokkos::Serial;
#endif
#endif

using DeviceType = Kokkos::Device<ExSpace,MemSpace>;

#endif
