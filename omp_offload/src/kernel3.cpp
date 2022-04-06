/*

miniAD is a miniapp of the GPU version of AutoDock 4.2 running a Lamarckian
Genetic Algorithm Copyright (C) 2017 TU Darmstadt, Embedded Systems and
Applications Group, Germany. All rights reserved. For some of the code,
Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research
Institute.

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
#include "auxiliary_genetic.cpp"
#include "calcenergy.cpp"
#include "kernels.hpp"
//#include "omp.h"
// if defined, new (experimental) SW genotype moves that are dependent
// on nr of atoms and nr of torsions of ligand are used
#define SWAT3  // Third set of Solis-Wets hyperparameters by Andreas Tillack

void gpu_perform_LS(uint32_t pops_by_runs,
	uint32_t work_pteam,
	float* pMem_conformations_next,
	float* pMem_energies_next,
	GpuData& cData,
	GpuDockparameters dockpars)

	// The GPU global function performs local search on the pre-defined entities of
	// conformations_next. The number of blocks which should be started equals to
	// num_of_lsentities*num_of_runs. This way the first num_of_lsentities entity of
	// each population will be subjected to local search (and each block carries out
	//the algorithm for one entity). Since the first entity is always the best one
	// in the current population, it is always tested according to the ls
	// probability, and if it not to be subjected to local search, the entity with ID
	// num_of_lsentities is selected instead of the first one (with ID 0).
{
	// FIXME : thread_limit(NUMOF_THREADS_PER_BLOCK)  generates wrong results
	//#pragma omp target teams distribute thread_limit(NUM_OF_THREADS_PER_BLOCK)
	#pragma omp target teams thread_limit(NUM_OF_THREADS_PER_BLOCK)\
	     num_teams(pops_by_runs)
	{
/*		float genotype_candidate[ACTUAL_GENOTYPE_LENGTH];
                float genotype_deviate[ACTUAL_GENOTYPE_LENGTH];
                float genotype_bias[ACTUAL_GENOTYPE_LENGTH];
		*/
		float sFloatBuff[3 * MAX_NUM_OF_ATOMS + 4 * ACTUAL_GENOTYPE_LENGTH];
                float rho;
                uint32_t cons_succ;
                uint32_t cons_fail;
                uint32_t iteration_cnt;
                int evaluation_cnt;
                //float3struct calc_coords[MAX_NUM_OF_ATOMS];
                //float offspring_genotype[ACTUAL_GENOTYPE_LENGTH];
                float offspring_energy;
                int entity_id;
                float energy_accumulate = 0.0f;

	#pragma omp parallel
	{
		const int threadIdx = omp_get_thread_num();
  		const int blockDim = omp_get_num_threads();
  		const int blockIdx = omp_get_team_num();
  		const int gridDim = omp_get_num_teams();
	//for (uint32_t idx = 0; idx < pops_by_runs; idx++) {  // for teams
	for (uint32_t idx = blockIdx; idx < pops_by_runs; idx+=gridDim) {  // for teams
/*
		
		  #pragma omp allocate(genotype_candidate) allocator(omp_pteam_mem_alloc) 
		  #pragma omp allocate(genotype_deviate) allocator(omp_pteam_mem_alloc) 
		  #pragma omp allocate(genotype_bias) allocator(omp_pteam_mem_alloc) 
		  #pragma omp allocate(rho) allocator(omp_pteam_mem_alloc) 
		  #pragma omp allocate(cons_succ) allocator(omp_pteam_mem_alloc) 
		  #pragma omp allocate(cons_fail) allocator(omp_pteam_mem_alloc) 
		  #pragma omp allocate(iteration_cnt) allocator(omp_pteam_mem_alloc) 
	          #pragma omp allocate(evaluation_cnt) allocator(omp_pteam_mem_alloc) 
		  #pragma omp allocate(calc_coords) allocator(omp_pteam_mem_alloc) 
		  #pragma omp allocate(offspring_genotype) allocator(omp_pteam_mem_alloc) 
		  #pragma omp allocate(offspring_energy) allocator(omp_pteam_mem_alloc) 
		  #pragma omp allocate(entity_id) allocator(omp_pteam_mem_alloc)
		  #pragma omp allocate(energy_accumulate) allocator(omp_pteam_mem_alloc)
	//	  #pragma omp allocate(candidate_energy_2) allocator(omp_pteam_mem_alloc)
*/		 
		float candidate_energy = 0.0f;
		const int run_id = blockIdx / dockpars.num_of_lsentities;

		float3struct* calc_coords = (float3struct*)sFloatBuff;
		float* genotype_candidate = (float*)(calc_coords + MAX_NUM_OF_ATOMS);
		float* genotype_deviate = (float*)(genotype_candidate + ACTUAL_GENOTYPE_LENGTH);
		float* genotype_bias = (float*)(genotype_deviate + ACTUAL_GENOTYPE_LENGTH);
		float* offspring_genotype = (float*)(genotype_bias + ACTUAL_GENOTYPE_LENGTH);

		// Determining run ID and entity ID
		// Initializing offspring genotype
//		run_id = idx / dockpars.num_of_lsentities;
		if (threadIdx == 0)
		{
			//int j = 0;
			entity_id = blockIdx % dockpars.num_of_lsentities;

			// Since entity 0 is the best one due to elitism,
			// it should be subjected to random selection
			if (entity_id == 0) {
				// If entity 0 is not selected according to LS-rate,
				// choosing an other entity
				if (100.0f * gpu_randf(cData.pMem_prng_states, blockIdx, threadIdx) >
					dockpars.lsearch_rate) {
					entity_id = dockpars.num_of_lsentities;
				}
			}

			offspring_energy = pMem_energies_next[run_id * dockpars.pop_size + entity_id];
			rho = 1.0f;
			cons_succ = 0;
			cons_fail = 0;
			iteration_cnt = 0;
			evaluation_cnt = 0;
		}
#pragma omp barrier

		const size_t offset = (run_id * dockpars.pop_size + entity_id) * GENOTYPE_LENGTH_IN_GLOBMEM;

		const int num_of_genes =  dockpars.num_of_genes;
		//	      float candidate_energy;
		//#pragma omp parallel for default(none) \
			shared(offspring_genotype,  genotype_bias, pMem_conformations_next) \
			firstprivate(num_of_genes, offset) //num_threads(num_of_genes)
		for (int gene_counter = threadIdx; gene_counter < num_of_genes;
			gene_counter += blockDim) {
			offspring_genotype[gene_counter] =
				pMem_conformations_next[offset + gene_counter];
			genotype_bias[gene_counter] = 0.0f;
		}
#pragma omp barrier

		while ((iteration_cnt < dockpars.max_num_of_iters) &&
			(rho > dockpars.rho_lower_bound)) {

			#ifdef SWAT3
			const float lig_scale = 1.0f / sqrt((float)dockpars.num_of_atoms);
			const float gene_scale = 1.0f / sqrt((float)dockpars.num_of_genes);
			#endif
			//#pragma omp parallel for reduction(+ : energy_idx) default(none) 
		//	#pragma omp parallel for default(none) \
					shared(cData, genotype_deviate,offspring_genotype,  \
					dockpars, genotype_candidate, calc_coords,genotype_bias ) \
					firstprivate(idx, gene_scale, num_of_genes, run_id ,\
					lig_scale, rho) //num_threads(num_of_genes)
			//for (uint32_t j = 0; j < work_pteam; j++) {
				// New random deviate
				for (int gene_counter = threadIdx; gene_counter < num_of_genes;
					gene_counter += blockDim) {
				#ifdef SWAT3
					genotype_deviate[gene_counter] =
						rho * (2 * gpu_randf(cData.pMem_prng_states, blockIdx, gene_counter) - 1) *
						(gpu_randf(cData.pMem_prng_states, blockIdx, gene_counter) < gene_scale);

					// Translation genes
					if (gene_counter < 3) {
						genotype_deviate[gene_counter] *= dockpars.base_dmov_mul_sqrt3;
					}
					// Orientation and torsion genes
					else {
						if (gene_counter < 6) {
							genotype_deviate[gene_counter] *=
								dockpars.base_dang_mul_sqrt3 * lig_scale;
						}
						else {
							genotype_deviate[gene_counter] *=
								dockpars.base_dang_mul_sqrt3 * gene_scale;
						}
					}
				#else
					genotype_deviate[gene_counter] =
						rho * (2 * gpu_randf(cData.pMem_prng_states, idx, gene_counter) - 1) *
						(gpu_randf(cData.pMem_prng_states, idx, gene_counter) < 0.3f);

					// Translation genes
					if (gene_counter < 3) {
						genotype_deviate[gene_counter] *= dockpars.base_dmov_mul_sqrt3;
					}
					// Orientation and torsion genes
					else {
						genotype_deviate[gene_counter] *= dockpars.base_dang_mul_sqrt3;
					}
					#endif
				}

				// Generating new genotype candidate
				for (int gene_counter = threadIdx; gene_counter < num_of_genes;
					gene_counter += blockDim) {
					genotype_candidate[gene_counter] = offspring_genotype[gene_counter] +
						genotype_deviate[gene_counter] +
						genotype_bias[gene_counter];
				}
			//}
				// Evaluating candidate
			#pragma omp barrier

			        //======================= Calculating Energy ===============//
			        candidate_energy = 0.0f;
        		//	#pragma omp parallel for //num_threads(dockpars.num_of_atoms)
        			for (int atom_id = threadIdx;
                  			atom_id < dockpars.num_of_atoms;
                  			atom_id+= blockDim) {
            				get_atompos( atom_id, calc_coords, cData );
        			}
				
       				 // General rotation moving vector
        			float4struct genrot_movingvec;
        			genrot_movingvec.x = genotype_candidate[0];
        			genrot_movingvec.y = genotype_candidate[1];
        			genrot_movingvec.z = genotype_candidate[2];
        			genrot_movingvec.w = 0.0f;
        			// Convert orientation genes from sex. to radians
        			const float phi         = genotype_candidate[3] * DEG_TO_RAD;
        			const float theta       = genotype_candidate[4] * DEG_TO_RAD;
        			const float genrotangle = genotype_candidate[5] * DEG_TO_RAD;

        			float4struct genrot_unitvec;
        			const float sin_angle = sin(theta);
        			const float s2 = sin(genrotangle * 0.5f);
        			genrot_unitvec.x = s2*sin_angle*cos(phi);
        			genrot_unitvec.y = s2*sin_angle*sin(phi);
        			genrot_unitvec.z = s2*cos(theta);
        			genrot_unitvec.w = cos(genrotangle*0.5f);

			#pragma omp barrier
        			//__threadfence();
        			//__syncthreads();
			
            				for (int rotation_counter  = threadIdx;
                 				rotation_counter  < dockpars.rotbondlist_length;
                 				rotation_counter +=blockDim){
            					rotate_atoms(rotation_counter, calc_coords, cData, dockpars, run_id, genotype_candidate, genrot_unitvec, genrot_movingvec);
					#pragma omp barrier
            				}

			        //float inter_energy = 0.0f;
                    	//	#pragma omp parallel for reduction(+:candidate_energy) num_threads(dockpars.num_of_atoms)
                    		for (int atom_id = threadIdx;
                              		atom_id < dockpars.num_of_atoms;
                              		atom_id+= blockDim){
                        		candidate_energy += calc_interenergy( atom_id, cData, dockpars, calc_coords );
                   		 } // End atom_id for-loop (INTERMOLECULAR ENERGY)

                       	 	//printf("inter energy: %f \n", inter_energy);
                    		//float intra_energy = 0.0f;
                    		//#pragma omp parallel for reduction(+:candidate_energy) num_threads(dockpars.num_of_intraE_contributors)
                    		for (int contributor_counter = threadIdx;
                         		contributor_counter < dockpars.num_of_intraE_contributors;
                         		contributor_counter += blockDim){
                         		candidate_energy += calc_intraenergy( contributor_counter, cData, dockpars, calc_coords );
                	    	}
				if (threadIdx == 0) energy_accumulate = 0;
				#pragma omp barrier
        	                //printf("intra energy: %f \n", intra_energy);
				#pragma omp atomic update
	                    	energy_accumulate +=candidate_energy;
				#pragma omp barrier
				candidate_energy = energy_accumulate;
				//printf("energy: %f, c_energy: %f \n", energy, candidate_energy);
                    // =======================================
		
		if (threadIdx == 0)
				evaluation_cnt++;
			#pragma omp barrier
			//--- thread barrier

			if (candidate_energy < offspring_energy) {  // If candidate is better, success
			//	#pragma omp parallel for default(none) \
					shared(offspring_genotype, genotype_candidate, genotype_bias, genotype_deviate) \
					firstprivate(num_of_genes) //num_threads(num_of_genes)
				for (int gene_counter = threadIdx; gene_counter < num_of_genes;
					gene_counter += blockDim) {
					// Updating offspring_genotype
					offspring_genotype[gene_counter] = genotype_candidate[gene_counter];

					// Updating genotype_bias
					genotype_bias[gene_counter] = 0.6f * genotype_bias[gene_counter] +
						0.4f * genotype_deviate[gene_counter];

				}
					// Work-item 0 will overwrite the shared variables
					// used in the previous if condition
					//--- thread barrier
			#pragma omp barrier
			if (threadIdx == 0)
				{
					offspring_energy = candidate_energy;
					cons_succ++;
					cons_fail = 0;
				}
			}
			else {   // If candidate is worser, check the opposite direction
				// Generating the other genotype candidate
			//	#pragma omp parallel for default(none) \
					shared(offspring_genotype, genotype_candidate, genotype_bias, genotype_deviate) \
					firstprivate(num_of_genes) //num_threads(num_of_genes)
				for (int gene_counter = threadIdx; gene_counter < num_of_genes;
					gene_counter += blockDim) {
					genotype_candidate[gene_counter] =
						offspring_genotype[gene_counter] -
						genotype_deviate[gene_counter] - genotype_bias[gene_counter];
				}

				// Evaluating candidate
				//--- thread barrier
			//#pragma omp barrier
			#pragma omp barrier
			        //======================= Calculating Energy ===============//
				candidate_energy = 0.0f;
			//	#pragma omp parallel for //num_threads(dockpars.num_of_atoms)
                                for (int atom_id = threadIdx;
                                        atom_id < dockpars.num_of_atoms;
                                        atom_id+= blockDim) {
                                        get_atompos( atom_id, calc_coords, cData );
                                }
                                
                                 // General rotation moving vector
                                float4struct genrot_movingvec;
                                genrot_movingvec.x = genotype_candidate[0];
                                genrot_movingvec.y = genotype_candidate[1];
                                genrot_movingvec.z = genotype_candidate[2];
                                genrot_movingvec.w = 0.0f;
                                // Convert orientation genes from sex. to radians
                                const float phi         = genotype_candidate[3] * DEG_TO_RAD;
                                const float theta       = genotype_candidate[4] * DEG_TO_RAD;
                                const float genrotangle = genotype_candidate[5] * DEG_TO_RAD;

                                float4struct genrot_unitvec;
                                const float sin_angle = sin(theta);
                                const float s2 = sin(genrotangle * 0.5f);
                                genrot_unitvec.x = s2*sin_angle*cos(phi);
                                genrot_unitvec.y = s2*sin_angle*sin(phi);
                                genrot_unitvec.z = s2*cos(theta);
                                genrot_unitvec.w = cos(genrotangle*0.5f);

                                //__threadfence();
                                //__syncthreads();
			#pragma omp barrier
				
                                        for (int rotation_counter  = threadIdx;
                                                rotation_counter  < dockpars.rotbondlist_length;
                                                rotation_counter += blockDim){
            					rotate_atoms(rotation_counter, calc_coords, cData, dockpars, run_id, genotype_candidate, genrot_unitvec, genrot_movingvec);
					#pragma omp barrier
                                        }

                               // float inter_energy = 0.0f;
                        //        #pragma omp parallel for reduction(+:candidate_energy) num_threads(dockpars.num_of_atoms)
                                for (int atom_id = threadIdx;
                                        atom_id < dockpars.num_of_atoms;
                                        atom_id+= blockDim){
                                        candidate_energy += calc_interenergy( atom_id, cData, dockpars, calc_coords );
                                 } // End atom_id for-loop (INTERMOLECULAR ENERGY)

                                //printf("inter energy: %f \n", inter_energy);
                                //float intra_energy = 0.0f;
                                //#pragma omp parallel for reduction(+:candidate_energy) num_threads(dockpars.num_of_intraE_contributors)
                                for (int contributor_counter = threadIdx;
                                        contributor_counter < dockpars.num_of_intraE_contributors;
                                        contributor_counter += blockDim){
                                        candidate_energy += calc_intraenergy( contributor_counter, cData, dockpars,  calc_coords );
                                }
                                //printf("intra energy: %f \n", intra_energy);
				if (threadIdx == 0) energy_accumulate = 0;
                                #pragma omp barrier
                                //printf("intra energy: %f \n", intra_energy);
                                #pragma omp atomic update
                                energy_accumulate +=candidate_energy;
                                #pragma omp barrier
                                candidate_energy = energy_accumulate;
				// =================================================================

		//	#pragma omp barrier
		if (threadIdx == 0)
		{
					evaluation_cnt++;

					#if defined(DEBUG_ENERGY_KERNEL)
					printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n",
						"-ENERGY-KERNEL3-", "GRIDS", "INTRA", partial_interE[0],
						partial_intraE[0]);
					#endif
		}

				//--- thread barrier
			#pragma omp barrier

				if (candidate_energy <
					offspring_energy)  // If candidate is better, success
				{
			//		#pragma omp parallel for default(none) \
					    firstprivate(num_of_genes) \
						shared(offspring_genotype, genotype_candidate, genotype_bias, genotype_deviate) //num_threads(num_of_genes)
					for (int gene_counter = threadIdx;
						gene_counter < num_of_genes;
						gene_counter += blockDim) {
						// Updating offspring_genotype
						offspring_genotype[gene_counter] =
							genotype_candidate[gene_counter];

						// Updating genotype_bias
						genotype_bias[gene_counter] =
							0.6f * genotype_bias[gene_counter] -
							0.4f * genotype_deviate[gene_counter];
					}

						// Work-item 0 will overwrite the shared variables
						// used in the previous if condition
						//--- thread barrier
			#pragma omp barrier

				if (threadIdx == 0)
					{
						offspring_energy = candidate_energy;
						cons_succ++;
						cons_fail = 0;
					}
				}
				else { // Failure in both directions
			//		#pragma omp parallel for default(none) firstprivate(num_of_genes) \
					    shared( genotype_bias) //num_threads(num_of_genes)
					for (int gene_counter = threadIdx;
						gene_counter < num_of_genes;
						gene_counter += blockDim){
						// Updating genotype_bias
						genotype_bias[gene_counter] = 0.5f * genotype_bias[gene_counter];
					}

				if (threadIdx == 0)
					{
						cons_succ = 0;
						cons_fail++;
					}
				}
			}

			// Changing rho if needed
			if (threadIdx == 0)
			{
				iteration_cnt++;

				if (cons_succ >= dockpars.cons_limit) {
					rho *= LS_EXP_FACTOR;
					cons_succ = 0;
				}
				else if (cons_fail >= dockpars.cons_limit) {
					rho *= LS_CONT_FACTOR;
					cons_fail = 0;
				}
			}
			//--- thread barrier
			#pragma omp barrier
		}

		// Updating eval counter and energy
		if (threadIdx == 0)
		{
			cData.pMem_evals_of_new_entities[run_id * dockpars.pop_size +
				entity_id] += evaluation_cnt;
			pMem_energies_next[run_id * dockpars.pop_size + entity_id] =
				offspring_energy;
		}

		// Mapping torsion angles and writing out results
//		offset =
//			(run_id * dockpars.pop_size + entity_id) * GENOTYPE_LENGTH_IN_GLOBMEM;
	//	#pragma omp parallel for default(none) firstprivate(offset, num_of_genes, pMem_conformations_next) \
		        shared( offspring_genotype) //num_threads(num_of_genes)
		for (int gene_counter = threadIdx; gene_counter < num_of_genes;
			gene_counter += blockDim) {
			if (gene_counter >= 3) {
				map_angle(offspring_genotype[gene_counter]);
			}
			pMem_conformations_next[offset + gene_counter] =
				offspring_genotype[gene_counter];
		}
	}  // End for a set of teams
	} // end parallel section
	} // end team region
}
