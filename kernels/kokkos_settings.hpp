#ifndef SPACE_SETTINGS_HPP
#define SPACE_SETTINGS_HPP
#include <Kokkos_Core.hpp>

// Declare the memory and execution spaces.
#if USE_GPU == 1
using MemSpace = Kokkos::CudaSpace;
using ExSpace = Kokkos::Cuda;
#else
#if USE_OMP == 1
using MemSpace = Kokkos::HostSpace;
using ExSpace = Kokkos::OpenMP;
#else
using MemSpace = Kokkos::HostSpace;
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

// TODO: The two typedefs below use ExSpace, which negates the point of templating everything since now
// the kernels will only run in ExSpace. But I don't see a clean way to solve that at the moment - ALS
// Set up member_type for device here so it can be passed as function argument
typedef Kokkos::TeamPolicy<ExSpace>::member_type member_type;

// Set up scratch space (short-term memory for each team)
typedef ExSpace::scratch_memory_space ScratchSpace;

// Set up unmanaged kokkos views to wrap around C-style arrays for deep copies
typedef Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> FloatView1D;
typedef Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> IntView1D;
typedef Kokkos::View<unsigned int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> UnsignedIntView1D;


#endif
