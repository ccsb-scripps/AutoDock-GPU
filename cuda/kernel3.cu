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
gpu_perform_LS_kernel(		
            float* pMem_conformations_next,
            float* pMem_energies_next

)
//The GPU global function performs local search on the pre-defined entities of conformations_next.
//The number of blocks which should be started equals to num_of_lsentities*num_of_runs.
//This way the first num_of_lsentities entity of each population will be subjected to local search
//(and each block carries out the algorithm for one entity).
//Since the first entity is always the best one in the current population,
//it is always tested according to the ls probability, and if it not to be
//subjected to local search, the entity with ID num_of_lsentities is selected instead of the first one (with ID 0).
{
	// Some OpenCL compilers don't allow declaring 
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel, 
	// and then passed to non-kernel functions.
	//__shared__ float3 calc_coords[MAX_NUM_OF_ATOMS];
	//__shared__ float* genotype_candidate[ACTUAL_GENOTYPE_LENGTH];
	//__shared__ float* genotype_deviate  [ACTUAL_GENOTYPE_LENGTH];
	//__shared__ float* genotype_bias     [ACTUAL_GENOTYPE_LENGTH];
	//__shared__ float* offspring_genotype[ACTUAL_GENOTYPE_LENGTH];
    __shared__ float rho;
	__shared__ int   cons_succ;
	__shared__ int   cons_fail;
	__shared__ int   iteration_cnt;
	__shared__ int   evaluation_cnt;


	__shared__ float offspring_energy;
    __shared__ float sFloatAccumulator;
	__shared__ int entity_id;
    extern __shared__ float sFloatBuff[];    
	float candidate_energy;
    int run_id;
    
	// Ligand-atom position and partial energies
	float3* calc_coords = (float3*)sFloatBuff; 

    // Genotype pointers
	float* genotype_candidate = (float*)(calc_coords + cData.dockpars.num_of_atoms);
    float* genotype_deviate = (float*)(genotype_candidate + cData.dockpars.num_of_genes);
    float* genotype_bias = (float*)(genotype_deviate + cData.dockpars.num_of_genes);    
    float* offspring_genotype = (float*)(genotype_bias + cData.dockpars.num_of_genes); 

	// Determining run ID and entity ID
	// Initializing offspring genotype
    run_id = blockIdx.x / cData.dockpars.num_of_lsentities;
	if (threadIdx.x == 0)
	{
        entity_id = blockIdx.x % cData.dockpars.num_of_lsentities;

		// Since entity 0 is the best one due to elitism,
		// it should be subjected to random selection
		if (entity_id == 0) {
			// If entity 0 is not selected according to LS-rate,
			// choosing an other entity
			if (100.0f*gpu_randf(cData.pMem_prng_states) > cData.dockpars.lsearch_rate) {
				entity_id = cData.dockpars.num_of_lsentities;					
			}
		}

		offspring_energy = pMem_energies_next[run_id*cData.dockpars.pop_size+entity_id];
		rho = 1.0f;
		cons_succ = 0;
		cons_fail = 0;
		iteration_cnt = 0;
		evaluation_cnt = 0;        
	}
    __threadfence();
    __syncthreads();

    size_t offset = (run_id * cData.dockpars.pop_size + entity_id) * GENOTYPE_LENGTH_IN_GLOBMEM;
	for (uint32_t gene_counter = threadIdx.x;
	     gene_counter < cData.dockpars.num_of_genes;
	     gene_counter+= blockDim.x) {
        offspring_genotype[gene_counter] = pMem_conformations_next[offset + gene_counter];
		genotype_bias[gene_counter] = 0.0f;
	}
    __threadfence();
	__syncthreads();
    

	while ((iteration_cnt < cData.dockpars.max_num_of_iters) && (rho > cData.dockpars.rho_lower_bound))
	{
		// New random deviate
		for (uint32_t gene_counter = threadIdx.x;
		     gene_counter < cData.dockpars.num_of_genes;
		     gene_counter+= blockDim.x)
		{
			genotype_deviate[gene_counter] = rho*(2*gpu_randf(cData.pMem_prng_states)-1);

			// Translation genes
			if (gene_counter < 3) {
				genotype_deviate[gene_counter] *= cData.dockpars.base_dmov_mul_sqrt3;
			}
			// Orientation and torsion genes
			else {
				genotype_deviate[gene_counter] *= cData.dockpars.base_dang_mul_sqrt3;
			}
		}

		// Generating new genotype candidate
		for (uint32_t gene_counter = threadIdx.x;
		     gene_counter < cData.dockpars.num_of_genes;
		     gene_counter+= blockDim.x) {
			   genotype_candidate[gene_counter] = offspring_genotype[gene_counter] + 
							      genotype_deviate[gene_counter]   + 
							      genotype_bias[gene_counter];
		}

		// Evaluating candidate
        __threadfence();
        __syncthreads();

		// ==================================================================
		gpu_calc_energy(
                genotype_candidate,
                candidate_energy,
                run_id,
                calc_coords,
                &sFloatAccumulator
				);
		// =================================================================

		if (threadIdx.x == 0) {
			evaluation_cnt++;
		}
        __threadfence();
        __syncthreads();

		if (candidate_energy < offspring_energy)	// If candidate is better, success
		{
			for (uint32_t gene_counter = threadIdx.x;
			     gene_counter < cData.dockpars.num_of_genes;
			     gene_counter+= blockDim.x)
			{
				// Updating offspring_genotype
				offspring_genotype[gene_counter] = genotype_candidate[gene_counter];

				// Updating genotype_bias
				genotype_bias[gene_counter] = 0.6f*genotype_bias[gene_counter] + 0.4f*genotype_deviate[gene_counter];
			}

			// Work-item 0 will overwrite the shared variables
			// used in the previous if condition
			__threadfence();
            __syncthreads();

			if (threadIdx.x == 0)
			{
				offspring_energy = candidate_energy;
				cons_succ++;
				cons_fail = 0;
			}
		}
		else	// If candidate is worser, check the opposite direction
		{
			// Generating the other genotype candidate
			for (uint32_t gene_counter = threadIdx.x;
			     gene_counter < cData.dockpars.num_of_genes;
			     gene_counter+= blockDim.x) {
				   genotype_candidate[gene_counter] = offspring_genotype[gene_counter] - 
								      genotype_deviate[gene_counter] - 
								      genotype_bias[gene_counter];
			}

			// Evaluating candidate
			__threadfence();
            __syncthreads();

			// =================================================================
			gpu_calc_energy(
                genotype_candidate,
                candidate_energy,
                run_id,
                calc_coords,
                &sFloatAccumulator
            );
			// =================================================================

			if (threadIdx.x == 0) {
				evaluation_cnt++;

				#if defined (DEBUG_ENERGY_KERNEL)
				printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL3-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
				#endif
			}
            __threadfence();
            __syncthreads();

			if (candidate_energy < offspring_energy) // If candidate is better, success
			{
				for (uint32_t gene_counter = threadIdx.x;
				     gene_counter < cData.dockpars.num_of_genes;
			       	     gene_counter+= blockDim.x)
				{
					// Updating offspring_genotype
					offspring_genotype[gene_counter] = genotype_candidate[gene_counter];

					// Updating genotype_bias
					genotype_bias[gene_counter] = 0.6f*genotype_bias[gene_counter] - 0.4f*genotype_deviate[gene_counter];
				}

				// Work-item 0 will overwrite the shared variables
				// used in the previous if condition
                __threadfence();
                __syncthreads();

				if (threadIdx.x == 0)
				{
					offspring_energy = candidate_energy;
					cons_succ++;
					cons_fail = 0;
				}
			}
			else	// Failure in both directions
			{
				for (uint32_t gene_counter = threadIdx.x;
				     gene_counter < cData.dockpars.num_of_genes;
				     gene_counter+= blockDim.x)
					   // Updating genotype_bias
					   genotype_bias[gene_counter] = 0.5f*genotype_bias[gene_counter];

				if (threadIdx.x == 0)
				{
					cons_succ = 0;
					cons_fail++;
				}
			}
		}

		// Changing rho if needed
		if (threadIdx.x == 0)
		{
			iteration_cnt++;

			if (cons_succ >= cData.dockpars.cons_limit)
			{
				rho *= LS_EXP_FACTOR;
				cons_succ = 0;
			}
			else
				if (cons_fail >= cData.dockpars.cons_limit)
				{
					rho *= LS_CONT_FACTOR;
					cons_fail = 0;
				}
		}
        __threadfence();
        __syncthreads();
	}

	// Updating eval counter and energy
	if (threadIdx.x == 0) {
		cData.pMem_evals_of_new_entities[run_id*cData.dockpars.pop_size+entity_id] += evaluation_cnt;
		pMem_energies_next[run_id*cData.dockpars.pop_size+entity_id] = offspring_energy;
	}

	// Mapping torsion angles and writing out results
    offset = (run_id*cData.dockpars.pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM;
	for (uint32_t gene_counter = threadIdx.x;
	     gene_counter < cData.dockpars.num_of_genes;
	     gene_counter+= blockDim.x) {
        if (gene_counter >= 3) {
		    map_angle(offspring_genotype[gene_counter]);
		}
        pMem_conformations_next[offset + gene_counter] = offspring_genotype[gene_counter];
	}
}


void gpu_perform_LS(
    uint32_t blocks,
    uint32_t threads,
    float* pMem_conformations_next,
	float* pMem_energies_next
)
{
    //size_t sz_shared = (9 * cpuData.dockpars.num_of_atoms + 5 * cpuData.dockpars.num_of_genes) * sizeof(float);
    size_t sz_shared = (3 * cpuData.dockpars.num_of_atoms + 4 * cpuData.dockpars.num_of_genes) * sizeof(float);    
    
    
    gpu_perform_LS_kernel<<<blocks, threads, sz_shared>>>(pMem_conformations_next, pMem_energies_next);
    LAUNCHERROR("gpu_perform_LS_kernel");     
#if 0
    cudaError_t status;
    status = cudaDeviceSynchronize();
    RTERROR(status, "gpu_perform_LS_kernel");
    status = cudaDeviceReset();
    RTERROR(status, "failed to shut down");
    exit(0);
#endif
}
