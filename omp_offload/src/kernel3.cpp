/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian
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

// if defined, new (experimental) SW genotype moves that are dependent
// on nr of atoms and nr of torsions of ligand are used
#define SWAT3  // Third set of Solis-Wets hyperparameters by Andreas Tillack

void gpu_perform_LS(uint32_t pops_by_runs,
	uint32_t work_pteam,
	float* pMem_conformations_next,
	float* pMem_energies_next,
	GpuData& cData)

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
	//#pragma omp target teams distribute 
	#pragma omp target teams distribute thread_limit(NUM_OF_THREADS_PER_BLOCK)\
	     num_teams(pops_by_runs) 
	for (uint32_t idx = 0; idx < pops_by_runs; idx++) {  // for teams

		float genotype_candidate[ACTUAL_GENOTYPE_LENGTH];
		float genotype_deviate[ACTUAL_GENOTYPE_LENGTH];
		float genotype_bias[ACTUAL_GENOTYPE_LENGTH];
		float rho;
		uint32_t cons_succ;
		uint32_t cons_fail;
		uint32_t iteration_cnt;
		int evaluation_cnt;
		float3struct calc_coords[MAX_NUM_OF_ATOMS];
		float offspring_genotype[ACTUAL_GENOTYPE_LENGTH];
		float offspring_energy;
		int entity_id;
		float candidate_energy;
		/*
				 #pragma omp allocate(genotype_candidate)
		   allocator(omp_pteam_mem_alloc) #pragma omp allocate(genotype_deviate)
		   allocator(omp_pteam_mem_alloc) #pragma omp allocate(genotype_bias)
		   allocator(omp_pteam_mem_alloc) #pragma omp allocate(rho)
		   allocator(omp_pteam_mem_alloc) #pragma omp allocate(cons_succ)
		   allocator(omp_pteam_mem_alloc) #pragma omp allocate(cons_fail)
		   allocator(omp_pteam_mem_alloc) #pragma omp allocate(iteration_cnt)
		   allocator(omp_pteam_mem_alloc) #pragma omp allocate(evaluation_cnt)
		   allocator(omp_pteam_mem_alloc) #pragma omp allocate(calc_coords)
		   allocator(omp_pteam_mem_alloc) #pragma omp allocate(offspring_genotype)
		   allocator(omp_pteam_mem_alloc) #pragma omp allocate(offspring_energy)
		   allocator(omp_pteam_mem_alloc) #pragma omp allocate(entity_id)
		   allocator(omp_pteam_mem_alloc)
		  */

		const int run_id = idx / cData.dockpars.num_of_lsentities;

		// Determining run ID and entity ID
		// Initializing offspring genotype
//		run_id = idx / cData.dockpars.num_of_lsentities;
		{
			int j = 0;
			entity_id = idx % cData.dockpars.num_of_lsentities;

			// Since entity 0 is the best one due to elitism,
			// it should be subjected to random selection
			if (entity_id == 0) {
				// If entity 0 is not selected according to LS-rate,
				// choosing an other entity
				if (100.0f * gpu_randf(cData.pMem_prng_states, idx, j) >
					cData.dockpars.lsearch_rate) {
					entity_id = cData.dockpars.num_of_lsentities;
				}
			}

			offspring_energy =
				pMem_energies_next[run_id * cData.dockpars.pop_size + entity_id];
			rho = 1.0f;
			cons_succ = 0;
			cons_fail = 0;
			iteration_cnt = 0;
			evaluation_cnt = 0;
		}

		const size_t offset =
			(run_id * cData.dockpars.pop_size + entity_id) * GENOTYPE_LENGTH_IN_GLOBMEM;

		const int num_of_genes =  cData.dockpars.num_of_genes;
		//	      float candidate_energy;
		//--- thread barrier
		#pragma omp parallel for default(none) \
			shared(offspring_genotype,  genotype_bias, pMem_conformations_next) \
			firstprivate(num_of_genes, offset)
		for (int gene_counter = 0; gene_counter < num_of_genes;
			gene_counter += 1) {
			offspring_genotype[gene_counter] =
				pMem_conformations_next[offset + gene_counter];
			genotype_bias[gene_counter] = 0.0f;
		}

		//--- thread barrier
		while ((iteration_cnt < cData.dockpars.max_num_of_iters) &&
			(rho > cData.dockpars.rho_lower_bound)) {

			#ifdef SWAT3
			const float lig_scale = 1.0f / sqrt((float)cData.dockpars.num_of_atoms);
			const float gene_scale = 1.0f / sqrt((float)cData.dockpars.num_of_genes);
			#endif
			//#pragma omp parallel for reduction(+ : energy_idx) default(none) 
			#pragma omp parallel for default(none) \
					shared(cData, genotype_deviate,offspring_genotype,  \
					cData.dockpars, genotype_candidate, calc_coords,genotype_bias ) \
					firstprivate(idx, gene_scale, work_pteam, num_of_genes, run_id ,\
					lig_scale, rho)
			for (uint32_t j = 0; j < work_pteam; j++) {
				// New random deviate
				for (int gene_counter = j; gene_counter < num_of_genes;
					gene_counter += work_pteam) {
				#ifdef SWAT3
					genotype_deviate[gene_counter] =
						rho * (2 * gpu_randf(cData.pMem_prng_states, idx, j) - 1) *
						(gpu_randf(cData.pMem_prng_states, idx, j) < gene_scale);
//printf("%d \n", gpu_randf(cData.pMem_prng_states, idx, j));
					// Translation genes
					if (gene_counter < 3) {
						genotype_deviate[gene_counter] *= cData.dockpars.base_dmov_mul_sqrt3;
					}
					// Orientation and torsion genes
					else {
						if (gene_counter < 6) {
							genotype_deviate[gene_counter] *=
								cData.dockpars.base_dang_mul_sqrt3 * lig_scale;
						}
						else {
							genotype_deviate[gene_counter] *=
								cData.dockpars.base_dang_mul_sqrt3 * gene_scale;
						}
					}
				#else
					genotype_deviate[gene_counter] =
						rho * (2 * gpu_randf(cData.pMem_prng_states, idx, j) - 1) *
						(gpu_randf(cData.pMem_prng_states, idx, j) < 0.3f);

					// Translation genes
					if (gene_counter < 3) {
						genotype_deviate[gene_counter] *= cData.dockpars.base_dmov_mul_sqrt3;
					}
					// Orientation and torsion genes
					else {
						genotype_deviate[gene_counter] *= cData.dockpars.base_dang_mul_sqrt3;
					}
					#endif
				}

				// Generating new genotype candidate
				for (int gene_counter = j; gene_counter < num_of_genes;
					gene_counter += work_pteam) {
					genotype_candidate[gene_counter] = offspring_genotype[gene_counter] +
						genotype_deviate[gene_counter] +
						genotype_bias[gene_counter];
				}
			}
				// Evaluating candidate
				//--- thread barrier

			        //======================= Calculating Energy ===============//
			        candidate_energy = 0.0f;
        			#pragma omp parallel for
        			for (int atom_id = 0;
                  			atom_id < cData.dockpars.num_of_atoms;
                  			atom_id+= 1) {
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
			
			        int num_of_rotcyc = cData.dockpars.rotbondlist_length/work_pteam;
        			for(int rot=0; rot < num_of_rotcyc; rot++){
            				int start = rot*work_pteam;
            				int end = start +work_pteam;
            				if ( end > cData.dockpars.rotbondlist_length ) end = cData.dockpars.rotbondlist_length;
            				#pragma omp parallel for  
            				for (int rotation_counter  = start;
                 				rotation_counter  < end;
                 				rotation_counter++){
            					rotate_atoms(rotation_counter, calc_coords, cData, run_id, genotype_candidate, genrot_unitvec, genrot_movingvec);
            				}
        			} // End rotation_counter for-loop	

			        //float inter_energy = 0.0f;
                    		#pragma omp parallel for reduction(+:candidate_energy)
                    		for (int atom_id = 0;
                              		atom_id < cData.dockpars.num_of_atoms;
                              		atom_id+= 1){
                        		candidate_energy += calc_interenergy( atom_id, cData, calc_coords );
                   		 } // End atom_id for-loop (INTERMOLECULAR ENERGY)

                       	 	//printf("inter energy: %f \n", inter_energy);
                    		//float intra_energy = 0.0f;
                    		#pragma omp parallel for reduction(+:candidate_energy)
                    		for (int contributor_counter = 0;
                         		contributor_counter < cData.dockpars.num_of_intraE_contributors;
                         		contributor_counter += 1){
                         		candidate_energy += calc_intraenergy( contributor_counter, cData, calc_coords );
                	    	}
        	                //printf("intra energy: %f \n", intra_energy);
	                    	//candidate_energy = (inter_energy +intra_energy);
                    // =======================================
		
				evaluation_cnt++;
			//--- thread barrier

			if (candidate_energy < offspring_energy) {  // If candidate is better, success
				#pragma omp parallel for default(none) \
					shared(offspring_genotype, genotype_candidate, genotype_bias, genotype_deviate) \
					firstprivate(num_of_genes)
				for (int gene_counter = 0; gene_counter < num_of_genes;
					gene_counter += 1) {
					// Updating offspring_genotype
					offspring_genotype[gene_counter] = genotype_candidate[gene_counter];

					// Updating genotype_bias
					genotype_bias[gene_counter] = 0.6f * genotype_bias[gene_counter] +
						0.4f * genotype_deviate[gene_counter];

					// Work-item 0 will overwrite the shared variables
					// used in the previous if condition
					//--- thread barrier

				}
				//if (j == 0) {
				{
					offspring_energy = candidate_energy;
					cons_succ++;
					cons_fail = 0;
				}
			}
			else {   // If candidate is worser, check the opposite direction
				// Generating the other genotype candidate
				#pragma omp parallel for default(none) \
					shared(offspring_genotype, genotype_candidate, genotype_bias, genotype_deviate) \
					firstprivate(num_of_genes)
				for (int gene_counter = 0; gene_counter < num_of_genes;
					gene_counter += 1) {
					genotype_candidate[gene_counter] =
						offspring_genotype[gene_counter] -
						genotype_deviate[gene_counter] - genotype_bias[gene_counter];
				}

				// Evaluating candidate
				//--- thread barrier
			        //======================= Calculating Energy ===============//
				candidate_energy = 0.0f;
				#pragma omp parallel for
                                for (int atom_id = 0;
                                        atom_id < cData.dockpars.num_of_atoms;
                                        atom_id+= 1) {
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
				
		                int num_of_rotcyc = cData.dockpars.rotbondlist_length/work_pteam;
                                for(int rot=0; rot < num_of_rotcyc; rot++){
                                        int start = rot*work_pteam;
                                        int end = start +work_pteam;
                                        if ( end > cData.dockpars.rotbondlist_length ) end = cData.dockpars.rotbondlist_length;
                                        #pragma omp parallel for
                                        for (int rotation_counter  = start;
                                                rotation_counter  < end;
                                                rotation_counter++){
            					rotate_atoms(rotation_counter, calc_coords, cData, run_id, genotype_candidate, genrot_unitvec, genrot_movingvec);
                                        }
                                } // End rotation_counter for-loop

                                //float inter_energy = 0.0f;
                                #pragma omp parallel for reduction(+:candidate_energy)
                                for (int atom_id = 0;
                                        atom_id < cData.dockpars.num_of_atoms;
                                        atom_id+= 1){
                                        candidate_energy += calc_interenergy( atom_id, cData, calc_coords );
                                 } // End atom_id for-loop (INTERMOLECULAR ENERGY)

                                //printf("inter energy: %f \n", inter_energy);
                                //float intra_energy = 0.0f;
                                #pragma omp parallel for reduction(+:candidate_energy)
                                for (int contributor_counter = 0;
                                        contributor_counter < cData.dockpars.num_of_intraE_contributors;
                                        contributor_counter += 1){
                                        candidate_energy += calc_intraenergy( contributor_counter, cData, calc_coords );
                                }
                                //printf("intra energy: %f \n", intra_energy);
                            //    candidate_energy = (inter_energy +intra_energy);
				// =================================================================

					evaluation_cnt++;

					#if defined(DEBUG_ENERGY_KERNEL)
					printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n",
						"-ENERGY-KERNEL3-", "GRIDS", "INTRA", partial_interE[0],
						partial_intraE[0]);
					#endif

				//--- thread barrier

				if (candidate_energy <
					offspring_energy)  // If candidate is better, success
				{
					#pragma omp parallel for default(none) \
					    firstprivate(num_of_genes) \
						shared(offspring_genotype, genotype_candidate, genotype_bias, genotype_deviate)
					for (int gene_counter = 0;
						gene_counter < num_of_genes;
						gene_counter += 1) {
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

					//if (j == 0) 
					{
						offspring_energy = candidate_energy;
						cons_succ++;
						cons_fail = 0;
					}
				}
				else { // Failure in both directions
					#pragma omp parallel for default(none) firstprivate(num_of_genes) \
					    shared( genotype_bias)
					for (int gene_counter = 0;
						gene_counter < num_of_genes;
						gene_counter += 1){
						// Updating genotype_bias
						genotype_bias[gene_counter] = 0.5f * genotype_bias[gene_counter];
					}

					if (0 == 0) {
						cons_succ = 0;
						cons_fail++;
					}
				}
			}

			// Changing rho if needed
			//if (j == 0) 
			{
				iteration_cnt++;

				if (cons_succ >= cData.dockpars.cons_limit) {
					rho *= LS_EXP_FACTOR;
					cons_succ = 0;
				}
				else if (cons_fail >= cData.dockpars.cons_limit) {
					rho *= LS_CONT_FACTOR;
					cons_fail = 0;
				}
			}
			//--- thread barrier
		}

		// Updating eval counter and energy
		//if (j == 0) 
		{
			cData.pMem_evals_of_new_entities[run_id * cData.dockpars.pop_size +
				entity_id] += evaluation_cnt;
			pMem_energies_next[run_id * cData.dockpars.pop_size + entity_id] =
				offspring_energy;
		}

		// Mapping torsion angles and writing out results
//		offset =
//			(run_id * cData.dockpars.pop_size + entity_id) * GENOTYPE_LENGTH_IN_GLOBMEM;
		#pragma omp parallel for default(none) firstprivate(offset, num_of_genes, pMem_conformations_next) \
		        shared( offspring_genotype)
		for (int gene_counter = 0; gene_counter < num_of_genes;
			gene_counter += 1) {
			if (gene_counter >= 3) {
				map_angle(offspring_genotype[gene_counter]);
			}
			pMem_conformations_next[offset + gene_counter] =
				offspring_genotype[gene_counter];
		}
	}  // End for a set of teams
}
