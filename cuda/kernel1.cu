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


__global__ void
__launch_bounds__(NUM_OF_THREADS_PER_BLOCK, 1024 / NUM_OF_THREADS_PER_BLOCK)
gpu_calc_initpop_kernel(	
                float* pMem_conformations_current,
                float* pMem_energies_current
)
{
    // Some OpenCL compilers don't allow declaring 
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel, 
	// and then passed to non-kernel functions.
	__shared__ float3 calc_coords[MAX_NUM_OF_ATOMS];
    __shared__ float sFloatAccumulator;
	float  energy = 0.0f;
	int    run_id = blockIdx.x / cData.dockpars.pop_size;
    float* pGenotype = pMem_conformations_current + blockIdx.x * GENOTYPE_LENGTH_IN_GLOBMEM;

	// =============================================================
	gpu_calc_energy(
            pGenotype,
			energy,
			run_id,
			calc_coords,
            &sFloatAccumulator
			);
	// =============================================================  

    // Write out final energy
	if (threadIdx.x == 0) 
    {
		pMem_energies_current[blockIdx.x] = energy;
		cData.pMem_evals_of_new_entities[blockIdx.x] = 1;
	}
}

void gpu_calc_initpop(uint32_t blocks, uint32_t threadsPerBlock, float* pConformations_current, float* pEnergies_current)
{
    gpu_calc_initpop_kernel<<<blocks, threadsPerBlock>>>(pConformations_current, pEnergies_current);
    LAUNCHERROR("gpu_calc_initpop_kernel");
#if 0
    cudaError_t status;
    status = cudaDeviceSynchronize();
    RTERROR(status, "gpu_calc_initpop_kernel");
    status = cudaDeviceReset();
    RTERROR(status, "failed to shut down");
    exit(0);
#endif
}

