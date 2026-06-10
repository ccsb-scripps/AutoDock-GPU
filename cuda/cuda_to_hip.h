/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.
For the ROCm/HIP port, Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/

// Single CUDA-to-HIP compatibility shim for the HIP backend (DEVICE=HIP).
// This is the only file that knows HIP symbol names: on ROCm it aliases the
// CUDA runtime spellings the project uses to their HIP equivalents; everywhere
// else it is a plain CUDA-runtime include, so the CUDA build is unchanged.
// Authoritative cuda->hip name source: pytorch
// torch/utils/hipify/cuda_to_hip_mappings.py.

#ifndef CUDA_TO_HIP_H
#define CUDA_TO_HIP_H

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// libc host decls must win over HIP's __device__ memcpy/memset overloads once
// <hip/hip_runtime.h> is in scope inside a .cu compiled as HIP (the host helper
// code in this TU calls memcpy); include them first.
#include <cstring>
#include <cstdlib>
#include <hip/hip_runtime.h>

// Runtime API
#define cudaError_t                  hipError_t
#define cudaSuccess                  hipSuccess
#define cudaGetErrorString           hipGetErrorString
#define cudaGetLastError             hipGetLastError
#define cudaDeviceSynchronize        hipDeviceSynchronize
#define cudaDeviceReset              hipDeviceReset
#define cudaSetDevice                hipSetDevice
#define cudaGetDevice                hipGetDevice
#define cudaGetDeviceCount           hipGetDeviceCount
#define cudaGetDeviceProperties      hipGetDeviceProperties
#define cudaDeviceProp               hipDeviceProp_t
#define cudaDeviceSetLimit           hipDeviceSetLimit
#define cudaLimitPrintfFifoSize      hipLimitPrintfFifoSize
#define cudaMemGetInfo               hipMemGetInfo

// Memory
#define cudaMalloc                   hipMalloc
#define cudaMallocManaged            hipMallocManaged
#define cudaMemAttachGlobal          hipMemAttachGlobal
#define cudaFree                     hipFree
#define cudaMemcpy                   hipMemcpy
#define cudaMemcpyToSymbol           hipMemcpyToSymbol
#define cudaMemcpyFromSymbol         hipMemcpyFromSymbol
#define cudaMemcpyHostToDevice       hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost       hipMemcpyDeviceToHost

#else // CUDA

#include <cuda_runtime.h>

#endif

#endif // CUDA_TO_HIP_H
