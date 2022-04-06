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


void gpu_calc_initpop(	uint32_t pops_by_runs, 
			uint32_t work_pteam, 
			float* pMem_conformations_current, 
			float* pMem_energies_current, 
			GpuData& cData,
			GpuDockparameters dockpars )
{

#pragma omp target teams thread_limit(NUM_OF_THREADS_PER_BLOCK)\
	     num_teams(pops_by_runs)
	{    
        float3struct calc_coords[MAX_NUM_OF_ATOMS];
	float energy_accumulate = 0.0f;

		#pragma omp parallel
		{
		const int threadIdx = omp_get_thread_num();
  		const int blockDim = omp_get_num_threads();
  		const int blockIdx = omp_get_team_num();
  		const int gridDim = omp_get_num_teams();
		for (uint32_t idx = blockIdx; idx < pops_by_runs; idx+=gridDim) { 
        
        		int run_id = idx / dockpars.pop_size;
        		float* pGenotype = pMem_conformations_current + idx * GENOTYPE_LENGTH_IN_GLOBMEM;
        //======================= Calculating Energy ===============//  
        		float energy = 0.0f;

        			for (int atom_id = threadIdx;
                  			atom_id < dockpars.num_of_atoms;
                  			atom_id+= blockDim) {
            				get_atompos( atom_id, calc_coords, cData );
        			}
				
       				 // General rotation moving vector
        			float4struct genrot_movingvec;
        			genrot_movingvec.x = pGenotype[0];
        			genrot_movingvec.y = pGenotype[1];
        			genrot_movingvec.z = pGenotype[2];
        			genrot_movingvec.w = 0.0f;
        			// Convert orientation genes from sex. to radians
        			const float phi         = pGenotype[3] * DEG_TO_RAD;
        			const float theta       = pGenotype[4] * DEG_TO_RAD;
        			const float genrotangle = pGenotype[5] * DEG_TO_RAD;

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
            					rotate_atoms(rotation_counter, calc_coords, cData, dockpars, run_id, pGenotype, genrot_unitvec, genrot_movingvec);
					#pragma omp barrier
            				}

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
        // ======================================= 
         //  printf("energy: %f \n", energy);
        // Write out final energy
	if (threadIdx== 0) 
    	{
        pMem_energies_current[idx] = energy;
        cData.pMem_evals_of_new_entities[idx] = 1;
	}

     	}  // End for a set of teams
        } // end parallel section
        } // end team region


}

