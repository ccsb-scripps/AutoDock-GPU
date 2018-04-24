// Gradient-based steepest descent minimizer
// Alternative to Solis-Wetts

__kernel void __attribute__ ((reqd_work_group_size(NUM_OF_THREADS_PER_BLOCK,1,1)))
gradient_minimizer(	
			char   dockpars_num_of_atoms,
			char   dockpars_num_of_atypes,
			int    dockpars_num_of_intraE_contributors,
			char   dockpars_gridsize_x,
			char   dockpars_gridsize_y,
			char   dockpars_gridsize_z,
							    		// g1 = gridsize_x
			uint   dockpars_gridsize_x_times_y, 		// g2 = gridsize_x * gridsize_y
			uint   dockpars_gridsize_x_times_y_times_z,	// g3 = gridsize_x * gridsize_y * gridsize_z
			float  dockpars_grid_spacing,
	 __global const float* restrict dockpars_fgrids, // This is too large to be allocated in __constant 
			int    dockpars_rotbondlist_length,
			float  dockpars_coeff_elec,
			float  dockpars_coeff_desolv,
	  __global      float* restrict dockpars_conformations_next,
	  __global      float* restrict dockpars_energies_next,
  	  __global 	int*   restrict dockpars_evals_of_new_entities,
	  __global      uint*  restrict dockpars_prng_states,
			int    dockpars_pop_size,
			int    dockpars_num_of_genes,
			float  dockpars_lsearch_rate,
			uint   dockpars_num_of_lsentities,
			float  dockpars_qasp,
	     __constant float* atom_charges_const,
    	     __constant char*  atom_types_const,
	     __constant char*  intraE_contributors_const,
    	     __constant float* VWpars_AC_const,
    	     __constant float* VWpars_BD_const,
             __constant float* dspars_S_const,
             __constant float* dspars_V_const,
             __constant int*   rotlist_const,
    	     __constant float* ref_coords_x_const,
    	     __constant float* ref_coords_y_const,
             __constant float* ref_coords_z_const,
    	     __constant float* rotbonds_moving_vectors_const,
             __constant float* rotbonds_unit_vectors_const,
             __constant float* ref_orientation_quats_const,
	     __constant int*   rotbonds_const,
	     __constant int*   rotbonds_atoms_const,
	     __constant int*   num_rotating_atoms_per_rotbond_const,
    			// Specific gradient-minimizer args
		    	uint      	  gradMin_maxiters,
			uint      	  gradMin_maxfails
/*
			,
	    		float             gradMin_alpha,
    	     __constant float* gradMin_conformation_min_perturbation     // minimal values for gene perturbation, originally as the scalar "dxmin"
*/
)
//The GPU global function performs gradient-based minimization on (some) entities of conformations_next.
//The number of OpenCL compute units (CU) which should be started equals to num_of_minEntities*num_of_runs.
//This way the first num_of_lsentities entity of each population will be subjected to local search
//(and each CU carries out the algorithm for one entity).
//Since the first entity is always the best one in the current population,
//it is always tested according to the ls probability, and if it not to be
//subjected to local search, the entity with ID num_of_lsentities is selected instead of the first one (with ID 0).
{
	// -----------------------------------------------------------------------------
	// Determining entity, and its run, energy, and genotype
	__local int   entity_id;
	__local int   run_id;
  	__local float energy;
	__local float genotype[ACTUAL_GENOTYPE_LENGTH];

	// Variables for gradient minimizer
  	__local uint iteration_cnt;  	// minimizer iteration counter
	__local uint failure_cnt;  	// minimizer failure counter
	__local bool exit;		

	// Number of energy-evaluations counter
	__local int  evaluation_cnt;

	if (get_local_id(0) == 0)
	{
		// Choosing a random entity out of the entire population
/*
		run_id = get_group_id(0);
		//entity_id = (uint)(dockpars_pop_size * gpu_randf(dockpars_prng_states));
		entity_id = 0;
*/

///*
		run_id = get_group_id(0) / dockpars_num_of_lsentities;
		entity_id = get_group_id(0) % dockpars_num_of_lsentities;

		// Since entity 0 is the best one due to elitism,
		// it should be subjected to random selection
		if (entity_id == 0) {
			// If entity 0 is not selected according to LS-rate,
			// choosing an other entity
			if (100.0f*gpu_randf(dockpars_prng_states) > dockpars_lsearch_rate) {
				entity_id = dockpars_num_of_lsentities;					
			}
		}
//*/
		
		energy = dockpars_energies_next[run_id*dockpars_pop_size+entity_id];

		#if defined (DEBUG_MINIMIZER)
		printf("\nrun_id:  %5u entity_id: %5u  initial_energy: %.5f\n", run_id, entity_id, energy);
		#endif

		// Initializing gradient-minimizer counters and flags
    		iteration_cnt  = 0;
		failure_cnt    = 0;
		/*exit           = false;*/
		evaluation_cnt = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

  	async_work_group_copy(genotype,
  			      dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
                              dockpars_num_of_genes, 0);

  	// -----------------------------------------------------------------------------
  	// Some OpenCL compilers don't allow declaring 
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel, 
	// and then passed to non-kernel functions.
	
	// Partial results for checking validity of genotype
	__local bool is_candidate_genotype_valid [ACTUAL_GENOTYPE_LENGTH];
           
	// Partial results of the gradient step
	__local float gradient          [ACTUAL_GENOTYPE_LENGTH];
	__local float candidate_energy;
	__local float candidate_genotype[ACTUAL_GENOTYPE_LENGTH];

	// -------------------------------------------------------------------
	// Calculate gradients (forces) for intermolecular energy
	// Derived from autodockdev/maps.py
	// -------------------------------------------------------------------
	// Gradient of the intermolecular energy per each ligand atom
	__local float gradient_inter_x[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_y[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_z[MAX_NUM_OF_ATOMS];

	// Gradient of the intramolecular energy per each ligand atom
	__local float gradient_intra_x[MAX_NUM_OF_ATOMS];
	__local float gradient_intra_y[MAX_NUM_OF_ATOMS];
	__local float gradient_intra_z[MAX_NUM_OF_ATOMS];
	__local float gradient_per_intracontributor[MAX_INTRAE_CONTRIBUTORS];

	// Accummulated gradient per each ligand atom
	__local float gradient_x[MAX_NUM_OF_ATOMS];
	__local float gradient_y[MAX_NUM_OF_ATOMS];
	__local float gradient_z[MAX_NUM_OF_ATOMS];	
	// -------------------------------------------------------------------

	// Ligand-atom position and partial energies
	__local float calc_coords_x[MAX_NUM_OF_ATOMS];
	__local float calc_coords_y[MAX_NUM_OF_ATOMS];
	__local float calc_coords_z[MAX_NUM_OF_ATOMS];
	__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];

	#if defined (DEBUG_ENERGY_KERNEL5)
	__local float partial_interE[NUM_OF_THREADS_PER_BLOCK];
	__local float partial_intraE[NUM_OF_THREADS_PER_BLOCK];
	#endif
	// -----------------------------------------------------------------------------

	// Step size of the steepest descent
	float alpha;

	// Initilizing each (work-item)-specific alpha
	for(uint i = get_local_id(0); 
		 i < dockpars_num_of_genes; 
		 i+= NUM_OF_THREADS_PER_BLOCK) {
			if (i<3)      { alpha = TRANGENE_ALPHA;	}
			else if (i<6) { alpha = ROTAGENE_ALPHA; } 
			else 	      { alpha = TORSGENE_ALPHA;	}

			#if defined (DEBUG_MINIMIZER)
			//printf("(%-3u) %-15.15f\n", i, alpha);
			#endif
	}

	// Perform gradient-descent iterations

	#if 0
	// 7cpa
	float grid_center_x = 49.836;
	float grid_center_y = 17.609;
	float grid_center_z = 36.272;
	float ligand_center_x = 49.2216976744186;
	float ligand_center_y = 17.793953488372097;
	float ligand_center_z = 36.503837209302326;
	float shoemake_gene_u1 = 0.02;
	float shoemake_gene_u2 = 0.23;
	float shoemake_gene_u3 = 0.95;
	#endif

	#if 0
	// 3tmn
	float grid_center_x = 52.340;
	float grid_center_y = 15.029;
	float grid_center_z = -2.932;
	float ligand_center_x = 52.22740741;
	float ligand_center_y = 15.51751852;
	float ligand_center_z = -2.40896296;
	#endif

	do {
		#if 0
		// Specific input genotypes for a ligand with no rotatable bonds (1ac8).
		// Translation genes must be expressed in grids in OCLADock (genotype [0|1|2]).
		// However, for testing purposes, 
		// we start using translation values in real space (Angstrom): {31.79575, 93.743875, 47.699875}
		// Rotation genes are expresed in the Shoemake space: genotype [3|4|5]
		// xyz_gene_gridspace = gridcenter_gridspace + (input_gene_realspace - gridcenter_realspace)/gridsize

		// 1ac8				
		genotype[0] = 30 + (31.79575  - 31.924) / 0.375;
		genotype[1] = 30 + (93.743875 - 93.444) / 0.375;
		genotype[2] = 30 + (47.699875 - 47.924) / 0.375;
		genotype[3] = 0.1f;
		genotype[4] = 0.5f;
		genotype[5] = 0.9f;
		#endif

		#if 0
		// 3tmn
		genotype[0] = 30 + (ligand_center_x - grid_center_x) / 0.375;
		genotype[1] = 30 + (ligand_center_y - grid_center_y) / 0.375;
		genotype[2] = 30 + (ligand_center_z - grid_center_z) / 0.375;
		genotype[3] = shoemake_gene_u1;
		genotype[4] = shoemake_gene_u2;
		genotype[5] = shoemake_gene_u3;
		genotype[6] = 0.0f;
		genotype[7] = 0.0f;
		genotype[8] = 0.0f;
		genotype[9] = 0.0f;
		genotype[10] = 0.0f;
		genotype[11] = 0.0f;
		genotype[12] = 0.0f;
		genotype[13] = 0.0f;
		genotype[14] = 0.0f;
		genotype[15] = 0.0f;
		genotype[16] = 0.0f;
		genotype[17] = 0.0f;
		genotype[18] = 0.0f;
		genotype[19] = 0.0f;
		genotype[20] = 0.0f;
		#endif

		// Calculating gradient
		// =============================================================
		gpu_calc_gradient(
				dockpars_rotbondlist_length,
				dockpars_num_of_atoms,
				dockpars_gridsize_x,
				dockpars_gridsize_y,
				dockpars_gridsize_z,
								    	// g1 = gridsize_x
				dockpars_gridsize_x_times_y, 		// g2 = gridsize_x * gridsize_y
				dockpars_gridsize_x_times_y_times_z,	// g3 = gridsize_x * gridsize_y * gridsize_z
				dockpars_fgrids,
				dockpars_num_of_atypes,
				dockpars_num_of_intraE_contributors,
				dockpars_grid_spacing,
				dockpars_coeff_elec,
				dockpars_qasp,
				dockpars_coeff_desolv,
				// Some OpenCL compilers don't allow declaring 
				// local variables within non-kernel functions.
				// These local variables must be declared in a kernel, 
				// and then passed to non-kernel functions.
				genotype,
				&energy,
				&run_id,

				calc_coords_x,
				calc_coords_y,
				calc_coords_z,

			        atom_charges_const,
				atom_types_const,
				intraE_contributors_const,
				VWpars_AC_const,
				VWpars_BD_const,
				dspars_S_const,
				dspars_V_const,
				rotlist_const,
				ref_coords_x_const,
				ref_coords_y_const,
				ref_coords_z_const,
				rotbonds_moving_vectors_const,
				rotbonds_unit_vectors_const,
				ref_orientation_quats_const,
				rotbonds_const,
				rotbonds_atoms_const,
				num_rotating_atoms_per_rotbond_const
			 	// Gradient-related arguments
			 	// Calculate gradients (forces) for intermolecular energy
			 	// Derived from autodockdev/maps.py
				,
				dockpars_num_of_genes,
				gradient_inter_x,
				gradient_inter_y,
				gradient_inter_z,
				gradient_intra_x,
				gradient_intra_y,
				gradient_intra_z,
				gradient_x,
				gradient_y,
				gradient_z,
				gradient_per_intracontributor,
				gradient
				);
		// =============================================================

		barrier(CLK_LOCAL_MEM_FENCE);

		
		for(uint i = get_local_id(0); i < dockpars_num_of_genes; i+= NUM_OF_THREADS_PER_BLOCK) {

	     		// Taking step
			candidate_genotype[i] = genotype[i] - alpha * gradient[i];	

			#if defined (DEBUG_MINIMIZER)
			//printf("(%-3u) %-0.15f %-10.10f %-10.10f %-10.10f\n", i, alpha, genotype[i], gradient[i], candidate_genotype[i]);
			#endif

			// Checking if every gene of candidate_genotype
			// is neither a nan, not inf
   			is_candidate_genotype_valid[i] = (isnan(candidate_genotype[i]) == 0) && (isinf(candidate_genotype[i]) == 0);

			// Verifying that Shoemake genes 
			// do not get out of valid region
			if ((i > 2) && (i < 6)) {
				if (is_candidate_genotype_valid[i] == true) {
					//printf("BEFORE is_candidate_genotype_valid[%u]?: %s\n", i, (is_candidate_genotype_valid[i] == true)?"yes":"no");
					if ((candidate_genotype[i] < 0.0f) || (candidate_genotype[i] > 1.0f)){
						is_candidate_genotype_valid[i] = false;
						//printf("AFTER  is_candidate_genotype_valid[%u]?: %s\n", i, (is_candidate_genotype_valid[i] == true)?"yes":"no");
					}
				}
			}
	   	}
		
		barrier(CLK_LOCAL_MEM_FENCE);

		__local bool is_valid;

		// Reducing all "valid" values of genes
		// into their corresponding genotype
		if (get_local_id(0) == 0) { 
			is_valid = true;
			
		  	for(uint i = 0; i < dockpars_num_of_genes;i++) {
		    		is_valid = is_valid && is_candidate_genotype_valid[i];
			
				#if defined (DEBUG_MINIMIZER)
				//printf("is_candidate_genotype_valid[%u]?: %s\n", i, (is_candidate_genotype_valid[i] == true)?"yes":"no");
				#endif
		  	}
			/*
			if (is_valid == false) {
				exit = true;
			}
			*/
			#if defined (DEBUG_MINIMIZER)
			//printf("is_valid?: %s, exit?: %s\n", (is_valid == true)?"yes":"no", (exit == true)?"yes":"no");
			if (is_valid == false) {
				printf("Candidate genome is invalid!\n");
			}
			#endif
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		/*if (exit == false) {*/
		if (is_valid == true) {
				// =============================================================
				gpu_calc_energy(dockpars_rotbondlist_length,
						dockpars_num_of_atoms,
						dockpars_gridsize_x,
						dockpars_gridsize_y,
						dockpars_gridsize_z,
										    	// g1 = gridsize_x
						dockpars_gridsize_x_times_y, 		// g2 = gridsize_x * gridsize_y
						dockpars_gridsize_x_times_y_times_z,	// g3 = gridsize_x * gridsize_y * gridsize_z
						dockpars_fgrids,
						dockpars_num_of_atypes,
						dockpars_num_of_intraE_contributors,
						dockpars_grid_spacing,
						dockpars_coeff_elec,
						dockpars_qasp,
						dockpars_coeff_desolv,
						candidate_genotype,
						&candidate_energy,
						&run_id,
						// Some OpenCL compilers don't allow declaring 
						// local variables within non-kernel functions.
						// These local variables must be declared in a kernel, 
						// and then passed to non-kernel functions.
						calc_coords_x,
						calc_coords_y,
						calc_coords_z,
						partial_energies,
						#if defined (DEBUG_ENERGY_KERNEL1)
						partial_interE,
						partial_intraE,
						#endif

						atom_charges_const,
						atom_types_const,
						intraE_contributors_const,
						VWpars_AC_const,
						VWpars_BD_const,
						dspars_S_const,
						dspars_V_const,
						rotlist_const,
						ref_coords_x_const,
						ref_coords_y_const,
						ref_coords_z_const,
						rotbonds_moving_vectors_const,
						rotbonds_unit_vectors_const,
						ref_orientation_quats_const
						);
				// =============================================================

				// Updating number of energy-evaluations counter
				if (get_local_id(0) == 0) {
					evaluation_cnt = evaluation_cnt + 1;
				}
				barrier(CLK_LOCAL_MEM_FENCE);

				// Checking if E(candidate_genotype) < E(genotype)
				if (candidate_energy < energy){
					// Updating energy
					if (get_local_id(0) == 0) {				
						energy = candidate_energy;
					}

					for(uint i = get_local_id(0); 
					 	 i < dockpars_num_of_genes; 
					 	 i+= NUM_OF_THREADS_PER_BLOCK) {

						// Updating genotype
						genotype [i] = candidate_genotype[i];
			
						// Up-scaling alpha by one order magnitud
						alpha = alpha*/*10*/(5/(failure_cnt == 0?1:(failure_cnt)));

						#if defined (DEBUG_MINIMIZER)
						//printf("(%-3u) %-15.15f %-10.10f %-10.10f %-10.10f\n", i, alpha, genotype[i], gradient[i], candidate_genotype[i]);
						#endif
					}
				}
				// If E (candidate) is worse 
				else {
					// Update failure counter
					if (get_local_id(0) == 0) {
						failure_cnt = failure_cnt + 1;

						#if defined (DEBUG_MINIMIZER)
						printf("Candidate energy has not improved!\n");
						#endif
					}
					barrier(CLK_LOCAL_MEM_FENCE);

					// Down-scaling alpha by one order magnitud.
					// Genotype is not updated, meaning that search will be
					// started over from the same point from with different alpha
					for(uint i = get_local_id(0); 
						 i < dockpars_num_of_genes; 
						 i+= NUM_OF_THREADS_PER_BLOCK) {
						alpha = native_divide(alpha, 5*failure_cnt/*10*/);

						#if defined (DEBUG_MINIMIZER)
						//printf("(%-3u) %-15.15f\n", i, alpha);
						#endif
					}	
				} // End of energy comparison
			/*} // End of if(valid genotypes)*/
		} 
		else {
			// Update failure counter
			if (get_local_id(0) == 0) {
				failure_cnt = failure_cnt + 1;
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			// Down-scaling alpha by one order magnitud.
			// Genotype is not updated, meaning that search will be
			// started over from the same point from with different alpha
			for(uint i = get_local_id(0); 
				 i < dockpars_num_of_genes; 
				 i+= NUM_OF_THREADS_PER_BLOCK) {
				alpha = native_divide(alpha, 5*failure_cnt/*10*/);

				#if defined (DEBUG_MINIMIZER)
				//printf("(%-3u) %-15.15f\n", i, alpha);
				#endif
			}
		} // End of if(exit==false)

		barrier(CLK_LOCAL_MEM_FENCE);

		// Updating number of stepest-descent iterations
		if (get_local_id(0) == 0) {
	    		iteration_cnt = iteration_cnt + 1;

			#if defined (DEBUG_MINIMIZER)
			printf("# mini-iters: %-3u, # ener-evals: %-3u, # failures: %-3u, energy: %f\n", iteration_cnt, evaluation_cnt, failure_cnt, energy);
			#endif

			#if defined (DEBUG_ENERGY_KERNEL5)
			printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL5-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
			#endif
		}

  	} while ((iteration_cnt < gradMin_maxiters) && (failure_cnt < gradMin_maxfails) /*&& (exit == false)*/);

	// -----------------------------------------------------------------------------

  	// Updating eval counter and energy
	if (get_local_id(0) == 0) {
		dockpars_evals_of_new_entities[run_id*dockpars_pop_size+entity_id] += evaluation_cnt;
		dockpars_energies_next[run_id*dockpars_pop_size+entity_id] = energy;

		#if defined (DEBUG_MINIMIZER)
		printf("-------> End of grad-min cycle, num of evals: %u, final energy: %.5f\n", evaluation_cnt, energy);
		#endif
	}

	// Mapping torsion angles
	for (uint gene_counter = get_local_id(0);
	     	  gene_counter < dockpars_num_of_genes;
	          gene_counter+= NUM_OF_THREADS_PER_BLOCK) {
		   if (gene_counter >= 6) {
			    map_angle(&(genotype[gene_counter]));
		   }
	}

	// Updating old offspring in population
	barrier(CLK_LOCAL_MEM_FENCE);

	async_work_group_copy(dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
			      genotype,
			      dockpars_num_of_genes, 0);
}
