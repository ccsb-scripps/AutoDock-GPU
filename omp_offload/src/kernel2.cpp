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
#include "kernels.hpp"

void gpu_sum_evals(uint32_t nruns, 
                   uint32_t work_pteam, 
                   GpuData& cData)
{
    #pragma omp target teams distribute\
     num_teams(nruns) thread_limit(NUM_OF_THREADS_PER_BLOCK)
    for (int idx = 0; idx < nruns; idx++ ) {
    
        int sum_evals = 0;
    //    #pragma omp allocate(sum_evals) allocator(omp_pteam_mem_alloc)

    	int* pEvals_of_new_entities = cData.pMem_evals_of_new_entities + idx * cData.dockpars.pop_size;
        #pragma omp parallel for reduction(+:sum_evals)
        for (int entity_counter = 0; entity_counter < cData.dockpars.pop_size; entity_counter++){
		sum_evals += pEvals_of_new_entities[entity_counter];
        }// End for a team
        cData.pMem_gpu_evals_of_runs[idx] += sum_evals;
        //printf("k2_%d(%d, %d)\n", idx, cData.pMem_gpu_evals_of_runs[idx], sum_evals );
   }// End for teams
}

