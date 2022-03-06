#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian
Genetic Algorithm Copyright (C) 2017 TU Darmstadt, Embedded Systems and
Applications Group, Germany. All rights reserved. For some of the code,
Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research
Institute.
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

void

gpu_sum_evals_kernel(sycl::nd_item<3> item_ct1, GpuData cData,
                     int *sSum_evals)
// The GPU global function sums the evaluation counter states
// which are stored in evals_of_new_entities array foreach entity,
// calculates the sums for each run and stores it in evals_of_runs array.
// The number of blocks which should be started equals to num_of_runs,
// since each block performs the summation for one run.
{

        int partsum_evals = 0;
        int *pEvals_of_new_entities =
            cData.pMem_evals_of_new_entities +
            item_ct1.get_group(2) * cData.dockpars.pop_size;
        for (int entity_counter = item_ct1.get_local_id(2);
             entity_counter < cData.dockpars.pop_size;
             entity_counter += item_ct1.get_local_range().get(2))
        {
		partsum_evals += pEvals_of_new_entities[entity_counter];
	}
	
	// Perform warp-wise reduction
        REDUCEINTEGERSUM(partsum_evals, sSum_evals);
        if (item_ct1.get_local_id(2) == 0)
        {
            cData.pMem_gpu_evals_of_runs[item_ct1.get_group(2)] += *sSum_evals;
        }
}

void gpu_sum_evals(uint32_t blocks, uint32_t threadsPerBlock)
{
        /*
        DPCT1049:42: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                extern dpct::constant_memory<GpuData, 0> cData;

                cData.init();

                auto cData_ptr_ct1 = cData.get_ptr();

                sycl::accessor<int, 0, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    sSum_evals_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                          sycl::range<3>(1, 1, threadsPerBlock),
                                      sycl::range<3>(1, 1, threadsPerBlock)),
                    [=](sycl::nd_item<3> item_ct1) {
                            gpu_sum_evals_kernel(
                                item_ct1, *cData_ptr_ct1,
                                sSum_evals_acc_ct1.get_pointer());
                    });
        });
        /*
        DPCT1001:43: The statement could not be removed.
        */
        LAUNCHERROR("gpu_sum_evals_kernel");
}
