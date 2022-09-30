/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.
Copyright (C) 2022 Intel Corporation

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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <cassert>
#include "defines.h"
#include "calcenergy.h"
#include "GpuData.h"
#include "dpcpp_migration.h"

inline uint64_t llitoulli(int64_t l)
{
	uint64_t u;
        /*
        DPCT1053:0: Migration of device assembly code is not supported.
        */
        // asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
        u = l;
        return u;
}

inline int64_t ullitolli(uint64_t u)
{
	int64_t l;
        /*
        DPCT1053:1: Migration of device assembly code is not supported.
        */
        // asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
        l = u;
        return l;
}

/*
DPCT1023:57: The DPC++ sub-group does not support mask options for shuffle.
*/
#define WARPMINIMUMEXCHANGE(tgx, v0, k0, mask)                                 \
        {                                                                      \
                float v1 = v0;                                                               \
                int k1 = k0;                                                                 \
                int otgx = tgx ^ mask;                                                       \
                float v2 = item_ct1.get_sub_group().shuffle(v0, otgx);                   \
                int k2 = item_ct1.get_sub_group().shuffle(k0, otgx);                     \
                int flag = ((v1 < v2) ^ (tgx > otgx)) && (v1 != v2);                         \
                k0 = flag ? k1 : k2;                                                         \
                v0 = flag ? v1 : v2;                                                         \
        }

#define WARPMINIMUM2(tgx, v0, k0) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 1) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 2) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 4) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 8) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 16)

/*
DPCT1023:40: The DPC++ sub-group does not support mask options for
sycl::ext::oneapi::any_of.
*/
/*
DPCT1023:41: The DPC++ sub-group does not support mask options for shuffle.
*/
/*
DPCT1007:39: Migration of this CUDA API is not supported by the Intel(R) DPC++
Compatibility Tool.
*/
#define REDUCEINTEGERSUM(value, pAccumulator)                                          \
        int val = sycl::reduce_over_group(item_ct1.get_group(), value, std::plus<>()); \
        *pAccumulator = val;                                                           \
        item_ct1.barrier(SYCL_MEMORY_SPACE);

#define ATOMICADDI32(pAccumulator, value)                               \
        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::local_space>(*pAccumulator) += ((int) (value))

#define ATOMICSUBI32(pAccumulator, value)                               \
        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::local_space>(*pAccumulator) -= ((int) (value))

/*
DPCT1058:94: "atomicAdd" is not migrated because it is not called in the code.
*/
#define ATOMICADDF32(pAccumulator, value) atomicAdd(pAccumulator, (value))
/*
DPCT1058:93: "atomicAdd" is not migrated because it is not called in the code.
*/
#define ATOMICSUBF32(pAccumulator, value) atomicAdd(pAccumulator, -(value))

/*
DPCT1023:11: The DPC++ sub-group does not support mask options for
sycl::ext::oneapi::any_of.
*/
/*
DPCT1023:12: The DPC++ sub-group does not support mask options for shuffle.
*/
/*
DPCT1064:23: Migrated __any_sync call is used in a macro definition and is not
valid for all macro uses. Adjust the code.
*/
/*
DPCT1064:24: Migrated __shfl_sync call is used in a macro definition and is not
valid for all macro uses. Adjust the code.
*/
/*
DPCT1007:9: Migration of this CUDA API is not supported by the Intel(R) DPC++
Compatibility Tool.
*/

#define REDUCEFLOATSUM(value, pAccumulator) \
        value = sycl::reduce_over_group(item_ct1.get_group(), value, std::plus<>());\
        *pAccumulator = (float) value;\
        item_ct1.barrier(SYCL_MEMORY_SPACE);


static dpct::constant_memory<GpuData, 0> cData;
static GpuData cpuData;

void SetKernelsGpuData(GpuData *pData) try {
        int status;
        /*
        DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(cData.get_ptr(), pData, sizeof(GpuData))
                      .wait(),
                  0);
        /*
        DPCT1001:2: The statement could not be removed.
        */
        RTERROR(status, "SetKernelsGpuData copy to cData failed");
        memcpy(&cpuData, pData, sizeof(GpuData));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void GetKernelsGpuData(GpuData *pData) try {
        int status;
        /*
        DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(pData, cData.get_ptr(), sizeof(GpuData))
                      .wait(),
                  0);
        /*
        DPCT1001:6: The statement could not be removed.
        */
        RTERROR(status, "GetKernelsGpuData copy From cData failed");
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Kernel files
#include "calcenergy.dp.cpp"
#include "calcMergeEneGra.dp.cpp"
#include "auxiliary_genetic.dp.cpp"
#include "kernel1.dp.cpp"
#include "kernel2.dp.cpp"
#include "kernel3.dp.cpp"
#include "kernel4.dp.cpp"
#include "kernel_ad.dp.cpp"
#include "kernel_adam.dp.cpp"
