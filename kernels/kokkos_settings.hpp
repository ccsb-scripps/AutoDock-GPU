#ifndef SPACE_SETTINGS_HPP
#define SPACE_SETTINGS_HPP
#include <Kokkos_Core.hpp>

// Declare the memory and execution spaces.
#if USE_GPU == 1
using MemSpace = Kokkos::CudaSpace;
using ExSpace = Kokkos::Cuda;
#else
using MemSpace = Kokkos::HostSpace;
#if USE_OMP == 1
using ExSpace = Kokkos::OpenMP;
#else
using ExSpace = Kokkos::Serial;
#endif
#endif
using DeviceType = Kokkos::Device<ExSpace,MemSpace>;


// Designate a CPU-specific Memory and Execution space (currently just used for the copy/restore)
using CPUSpace = Kokkos::HostSpace;
#if USE_OMP == 1
using CPUExec = Kokkos::OpenMP;
#else
using CPUExec = Kokkos::Serial;
#endif
using HostType = Kokkos::Device<CPUExec,CPUSpace>;

// Set up member_type for device here so it can be passed as function argument
typedef Kokkos::TeamPolicy<ExSpace>::member_type member_type;

// Set up scratch space (short-term memory for each team)
typedef ExSpace::scratch_memory_space ScratchSpace;

// Set up unmanaged kokkos views to wrap around C-style arrays for deep copies
typedef Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> FloatView1D;
typedef Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> IntView1D;
typedef Kokkos::View<unsigned int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> UnsignedIntView1D;


#endif
