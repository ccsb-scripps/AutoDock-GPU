/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

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


#include <cstdint>
#include <cassert>
#include "defines.h"
#include "calcenergy.h"
#include "GpuData.h"

__device__ inline uint64_t llitoulli(int64_t l)
{
	uint64_t u;
	asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
	return u;
}

__device__ inline int64_t ullitolli(uint64_t u)
{
	int64_t l;
	asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
	return l;
}


#define WARPMINIMUMEXCHANGE(tgx, v0, k0, mask) \
	{ \
		float v1    = v0; \
		int k1      = k0; \
		int otgx    = tgx ^ mask; \
		float v2    = __shfl_sync(0xffffffff, v0, otgx); \
		int k2      = __shfl_sync(0xffffffff, k0, otgx); \
		int flag    = ((v1 < v2) ^ (tgx > otgx)) && (v1 != v2); \
		k0          = flag ? k1 : k2; \
		v0          = flag ? v1 : v2; \
	}

#define WARPMINIMUM2(tgx, v0, k0) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 1) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 2) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 4) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 8) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 16)

#define REDUCEINTEGERSUM(value, pAccumulator) \
	if (threadIdx.x == 0) \
	{ \
		*pAccumulator = 0; \
	} \
	__threadfence(); \
	__syncthreads(); \
	if (__any_sync(0xffffffff, value != 0)) \
	{ \
		uint32_t tgx            = threadIdx.x & cData.warpmask; \
		value                  += __shfl_sync(0xffffffff, value, tgx ^ 1); \
		value                  += __shfl_sync(0xffffffff, value, tgx ^ 2); \
		value                  += __shfl_sync(0xffffffff, value, tgx ^ 4); \
		value                  += __shfl_sync(0xffffffff, value, tgx ^ 8); \
		value                  += __shfl_sync(0xffffffff, value, tgx ^ 16); \
		if (tgx == 0) \
		{ \
			atomicAdd(pAccumulator, value); \
		} \
	} \
	__threadfence(); \
	__syncthreads(); \
	value = *pAccumulator; \
	__syncthreads();

#define ATOMICADDI32(pAccumulator, value) atomicAdd(pAccumulator, (value))
#define ATOMICSUBI32(pAccumulator, value) atomicAdd(pAccumulator, -(value))
#define ATOMICADDF32(pAccumulator, value) atomicAdd(pAccumulator, (value))
#define ATOMICSUBF32(pAccumulator, value) atomicAdd(pAccumulator, -(value))

#define REDUCEFLOATSUM(value, pAccumulator) \
	if (threadIdx.x == 0) \
	{ \
		*pAccumulator = 0; \
	} \
	__threadfence(); \
	__syncthreads(); \
	if (__any_sync(0xffffffff, value != 0.0f)) \
	{ \
		uint32_t tgx            = threadIdx.x & cData.warpmask; \
		value                  += __shfl_sync(0xffffffff, value, tgx ^ 1); \
		value                  += __shfl_sync(0xffffffff, value, tgx ^ 2); \
		value                  += __shfl_sync(0xffffffff, value, tgx ^ 4); \
		value                  += __shfl_sync(0xffffffff, value, tgx ^ 8); \
		value                  += __shfl_sync(0xffffffff, value, tgx ^ 16); \
		if (tgx == 0) \
		{ \
			atomicAdd(pAccumulator, value); \
		} \
	} \
	__threadfence(); \
	__syncthreads(); \
	value = (float)(*pAccumulator); \
	__syncthreads();



static __constant__ GpuData cData;
static GpuData cpuData;

void SetKernelsGpuData(GpuData* pData)
{
	cudaError_t status;
	status = cudaMemcpyToSymbol(cData, pData, sizeof(GpuData));
	RTERROR(status, "SetKernelsGpuData copy to cData failed");
	memcpy(&cpuData, pData, sizeof(GpuData));
}

void GetKernelsGpuData(GpuData* pData)
{
	cudaError_t status;
	status = cudaMemcpyFromSymbol(pData, cData, sizeof(GpuData));
	RTERROR(status, "GetKernelsGpuData copy From cData failed");
}


// Kernel files
#include "calcenergy.cu"
#include "calcMergeEneGra.cu"
#include "auxiliary_genetic.cu"
#include "kernel1.cu"
#include "kernel2.cu"
#include "kernel3.cu"
#include "kernel4.cu"
#include "kernel_ad.cu"
#include "kernel_adam.cu"
