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
#include "calcenergy.cpp"
#include "auxiliary_genetic.cpp"

//#define DEBUG_ENERGY_KERNEL4

void gpu_gen_and_eval_newpops(
    uint32_t pops_by_runs,
    uint32_t work_pteam,
    float* pMem_conformations_current,
    float* pMem_energies_current,
    float* pMem_conformations_next,
    float* pMem_energies_next,
    GpuData& cData,
    GpuDockparameters dockpars
)
{
   #pragma omp target teams thread_limit(NUM_OF_THREADS_PER_BLOCK)\
             num_teams(pops_by_runs)
   {
	
	float offspring_genotype[ACTUAL_GENOTYPE_LENGTH];
        int parent_candidates[4];
        float candidate_energies[4];
        int parents[2];
        int covr_point[2];
        float randnums[10];
        float bestEnergy[NUM_OF_THREADS_PER_BLOCK];
        int bestID[NUM_OF_THREADS_PER_BLOCK];
        float3struct calc_coords[MAX_NUM_OF_ATOMS];
        float energy_accumulate = 0.0f;

        #pragma omp parallel
         {
             const int threadIdx = omp_get_thread_num();
             const int blockDim = omp_get_num_threads();
             const int blockIdx = omp_get_team_num();
             const int gridDim = omp_get_num_teams();
             for (uint32_t idx = blockIdx; idx < pops_by_runs; idx+=gridDim) 
	     {
        	float energy;
        	int run_id;
        	int temp_covr_point;
        	// In this case this compute-unit is responsible for elitist selection
        	if ((idx % dockpars.pop_size) == 0) {
                // Find and copy best member of population to position 0
                if (threadIdx <dockpars.pop_size)
                {
                   bestID[threadIdx] = idx + threadIdx;
                   bestEnergy[threadIdx] = pMem_energies_current[idx + threadIdx];
                }
            
                // Scan through population (we already picked up a work_pteam's worth above so skip)
                for (int i = idx + blockDim + threadIdx; i < idx + dockpars.pop_size; i += blockDim)
                {
                   float e = pMem_energies_current[i];
                   if (e < bestEnergy[threadIdx])
                   {
                      bestID[threadIdx] = i;
                      bestEnergy[threadIdx] = e;
                   }
               	}
           
        	if (threadIdx == 0)
            	{
            	    for(uint32_t entity_counter = 1; entity_counter < blockDim; entity_counter++)
                       if ((bestEnergy[entity_counter] < bestEnergy[0]) && (entity_counter < dockpars.pop_size)){
                          bestEnergy[0] = bestEnergy[entity_counter];
                          bestID[0] = bestID[entity_counter];
                        }
            	    pMem_energies_next[idx] = bestEnergy[0];
                    cData.pMem_evals_of_new_entities[idx] = 0;
                 }
//  */      
//--- thread barrier
		#pragma omp barrier
            
                // Copy best genome to next generation
                int dOffset = idx * GENOTYPE_LENGTH_IN_GLOBMEM;
               //int sOffset = bestID * GENOTYPE_LENGTH_IN_GLOBMEM;
                int sOffset = bestID[0] * GENOTYPE_LENGTH_IN_GLOBMEM;
                for (int i = threadIdx; i < dockpars.num_of_genes; i += blockDim)
                {
                    pMem_conformations_next[dOffset + i] = pMem_conformations_current[sOffset + i];
                }
        	}
        	else
        	{
            	    // Generating the following random numbers:
                    // [0..3] for parent candidates,
                    // [4..5] for binary tournaments, [6] for deciding crossover,
                    // [7..8] for crossover points, [9] for local search
                     for (uint32_t gene_counter = threadIdx;
                          gene_counter < 10;
                          gene_counter += blockDim) {
                          randnums[gene_counter] = gpu_randf(cData.pMem_prng_states, idx, gene_counter);
                     }
#if 0
        if ((threadIdx == 0) && (idx == 1))
        {
            printf("%06d ", idx);
            for (int i = 0; i < 10; i++)
                printf("%12.6f ", randnums[i]);
            printf("\n");
        }
#endif
            	// Determining run ID
            	run_id = idx / dockpars.pop_size;
//--- thread barrier
		#pragma omp barrier
            	if (threadIdx < 4){	//it is not ensured that the four candidates will be different...
                    parent_candidates[threadIdx]  = (int) (dockpars.pop_size*randnums[threadIdx]); //using randnums[0..3]
                    candidate_energies[threadIdx] = pMem_energies_current[run_id*dockpars.pop_size+parent_candidates[threadIdx]];
            	}
//--- thread barrier
		#pragma omp barrier
	    	if (threadIdx < 2){
                    // Notice: dockpars_tournament_rate was scaled down to [0,1] in host
                    // to reduce number of operations in device
                    if (candidate_energies[2*threadIdx] < candidate_energies[2*threadIdx+1]){
                        if (/*100.0f**/randnums[4+threadIdx] < dockpars.tournament_rate) {		//using randnum[4..5]
                            parents[threadIdx] = parent_candidates[2*threadIdx];
                        }
                        else {
                            parents[threadIdx] = parent_candidates[2*threadIdx+1];
                        }
                    }
                    else
                    {
                        if (/*100.0f**/randnums[4+threadIdx] < dockpars.tournament_rate) {
                            parents[threadIdx] = parent_candidates[2*threadIdx+1];
                        }
                        else {
                            parents[threadIdx] = parent_candidates[2*threadIdx];
                        }
                    }
           	 }
//--- thread barrier
		#pragma omp barrier
            	// Performing crossover
            	// Notice: dockpars_crossover_rate was scaled down to [0,1] in host
            	// to reduce number of operations in device
            	if (/*100.0f**/randnums[6] < dockpars.crossover_rate)	// Using randnums[6]
            	{
		    if(threadIdx <2){
                        // Using randnum[7..8]
                        covr_point[threadIdx] = (int) ((dockpars.num_of_genes-1)*randnums[7+threadIdx]);
                	}
    //--- thread barrier
		#pragma omp barrier

                    // covr_point[0] should store the lower crossover-point
                    if (threadIdx == 0)
                    {
                        if (covr_point[1] < covr_point[0]) {
                            temp_covr_point = covr_point[1];
                            covr_point[1]   = covr_point[0];
                            covr_point[0]   = temp_covr_point;
                        }
                    }

    //--- thread barrier
		#pragma omp barrier
                     for (uint32_t gene_counter = threadIdx;
                     gene_counter < dockpars.num_of_genes;
                     gene_counter+= blockDim)
                     {
                        // Two-point crossover
                         if (covr_point[0] != covr_point[1])
                         {
                              if ((gene_counter <= covr_point[0]) || (gene_counter > covr_point[1]))
                            	offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*dockpars.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
                              else
                            	offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*dockpars.pop_size+parents[1])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
                          }
                    // Single-point crossover
                          else
                          {
                        	if (gene_counter <= covr_point[0])
                            	   offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*dockpars.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
                        	else
                                  offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*dockpars.pop_size+parents[1])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
                    	   }
                      }
               }
               else	//no crossover
               {
                     for (uint32_t gene_counter = threadIdx;
                     gene_counter < dockpars.num_of_genes;
                     gene_counter+= blockDim)
                     {
                    	offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*dockpars.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM + gene_counter];
                     }
                } // End of crossover

//--- thread barrier
		#pragma omp barrier
            	// Performing mutation
            	for (uint32_t gene_counter = threadIdx;
                     gene_counter < dockpars.num_of_genes;
                     gene_counter+= blockDim)
                {
                     // Notice: dockpars_mutation_rate was scaled down to [0,1] in host
                     // to reduce number of operations in device
                     if (/*100.0f**/gpu_randf(cData.pMem_prng_states, idx, gene_counter) < dockpars.mutation_rate)
                     {
                    	// Translation genes
                    	if (gene_counter < 3) {
                            offspring_genotype[gene_counter] += dockpars.abs_max_dmov*(2*gpu_randf(cData.pMem_prng_states, idx, gene_counter)-1);
                        }
                    // Orientation and torsion genes
                        else {
                            offspring_genotype[gene_counter] += dockpars.abs_max_dang*(2*gpu_randf(cData.pMem_prng_states, idx, gene_counter)-1);
                            map_angle(offspring_genotype[gene_counter]);
                     }

                    }
                } // End of mutation

//--- thread barrier
		#pragma omp barrier        
            
                //======================= Calculating Energy ===============//
		   energy = 0.0f;
                   for (uint atom_id = threadIdx;
                           atom_id < dockpars.num_of_atoms;
                           atom_id+=blockDim) {
                           get_atompos( atom_id, calc_coords, cData );
                   }
                   //      */
                    // General rotation moving vector
                   float4struct genrot_movingvec;
                   genrot_movingvec.x = offspring_genotype[0];
                   genrot_movingvec.y = offspring_genotype[1];
                   genrot_movingvec.z = offspring_genotype[2];
                   genrot_movingvec.w = 0.0f;
                   // Convert orientation genes from sex. to radians
                   const float phi         = offspring_genotype[3] * DEG_TO_RAD;
                   const float theta       = offspring_genotype[4] * DEG_TO_RAD;
                   const float genrotangle = offspring_genotype[5] * DEG_TO_RAD;

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
                 				rotation_counter +=blockDim){
				rotate_atoms(rotation_counter, calc_coords, cData, dockpars, run_id, offspring_genotype, genrot_unitvec, genrot_movingvec);
			#pragma omp barrier
       	 	     } // End rotation_counter for-loop
                   for (int atom_id = threadIdx;
                              		atom_id < dockpars.num_of_atoms;
                              		atom_id+= blockDim){
                        		energy += calc_interenergy( atom_id, cData, dockpars, calc_coords );
                   		 } // End atom_id for-loop (INTERMOLECULAR ENERGY)

                    for (int contributor_counter = threadIdx;
                         		contributor_counter < dockpars.num_of_intraE_contributors;
                         		contributor_counter += blockDim){
                         		energy += calc_intraenergy( contributor_counter, cData, dockpars, calc_coords );
                	    	}
				if (threadIdx == 0) energy_accumulate = 0;
				#pragma omp barrier
        	                //printf("intra energy: %f \n", intra_energy);
				#pragma omp atomic update
	                    	energy_accumulate +=energy;
				#pragma omp barrier
				energy = energy_accumulate;
		   //energy = (inter_energy +intra_energy);
                   // =================================================================
           if (threadIdx == 0) {
                pMem_energies_next[idx] = energy;
                cData.pMem_evals_of_new_entities[idx] = 1;
                #if defined (DEBUG_ENERGY_KERNEL4)
                printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL4-", "GRIDS", "INTRA", interE, intraE);
                #endif
            }

            // Copying new offspring to next generation
            for (uint32_t gene_counter = threadIdx;
                 gene_counter < dockpars.num_of_genes;
                 gene_counter+= blockDim)
            {
                pMem_conformations_next[idx * GENOTYPE_LENGTH_IN_GLOBMEM + gene_counter] = offspring_genotype[gene_counter];
            }
	}

        } // End for a set of teams
	}// End for parellel section
    }  // End team region
}


