// Gradient-based FIRE minimizer
// Alternative to Solis-Wetts / Steepest-Descent

// Fire parameters (TODO: to be moved to header file?)
#define SUCCESS_MIN		5

#define DT_MAX      		10.0f
#define DT_MIN			1e-6
#define DT_MAX_DIV_THREE	(DT_MAX / 3.0f)
#define DT_INC			1.2f
#define DT_DEC			0.8f

#define ALPHA_START 		0.2f
#define ALPHA_DEC		0.99f


__kernel void __attribute__ ((reqd_work_group_size(NUM_OF_THREADS_PER_BLOCK,1,1)))
gradient_minFire(	
			    char   dockpars_num_of_atoms,
			    char   dockpars_num_of_atypes,
			    int    dockpars_num_of_intraE_contributors,
			    char   dockpars_gridsize_x,
			    char   dockpars_gridsize_y,
			    char   dockpars_gridsize_z,
							    		// g1 = gridsize_x
			    uint   dockpars_gridsize_x_times_y, 	// g2 = gridsize_x * gridsize_y
			    uint   dockpars_gridsize_x_times_y_times_z,	// g3 = gridsize_x * gridsize_y * gridsize_z
			    float  dockpars_grid_spacing,
	 __global const     float* restrict dockpars_fgrids, 		// This is too large to be allocated in __constant 
			    int    dockpars_rotbondlist_length,
			    float  dockpars_coeff_elec,
			    float  dockpars_coeff_desolv,
	  __global          float* restrict dockpars_conformations_next,
	  __global          float* restrict dockpars_energies_next,
  	  __global 	    int*   restrict dockpars_evals_of_new_entities,
	  __global          uint*  restrict dockpars_prng_states,
			    int    dockpars_pop_size,
			    int    dockpars_num_of_genes,
			    float  dockpars_lsearch_rate,
			    uint   dockpars_num_of_lsentities,
			    uint   dockpars_max_num_of_iters,
			    float  dockpars_qasp,
			    float  dockpars_smooth,

	  __constant        kernelconstant_interintra* 	 kerconst_interintra,
	  __global const    kernelconstant_intracontrib* kerconst_intracontrib,
	  __constant        kernelconstant_intra*	 kerconst_intra,
	  __constant        kernelconstant_rotlist*   	 kerconst_rotlist,
	  __constant        kernelconstant_conform*	 kerconst_conform
			,
	  __constant int*   	  rotbonds_const,
	  __global   const int*   rotbonds_atoms_const,
	  __constant int*   	  num_rotating_atoms_per_rotbond_const
			,
	  __global   const float* angle_const,
	  __constant       float* dependence_on_theta_const,
	  __constant       float* dependence_on_rotangle_const
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

	// Iteration counter fot the minimizer
  	__local uint iteration_cnt;  	

	if (get_local_id(0) == 0)
	{
		// Choosing a random entity out of the entire population
/*
		run_id = get_group_id(0);
		//entity_id = (uint)(dockpars_pop_size * gpu_randf(dockpars_prng_states));
		entity_id = 0;
*/
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
		
		energy = dockpars_energies_next[run_id*dockpars_pop_size+entity_id];

		#if defined (DEBUG_MINIMIZER)
		printf("\nrun_id:  %5u entity_id: %5u  initial_energy: %.7f\n", run_id, entity_id, energy);
		#endif

		// Initializing gradient-minimizer counters and flags
    		iteration_cnt  = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

  	event_t ev = async_work_group_copy(genotype,
  			      		   dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
                              		   dockpars_num_of_genes, 0);

	// Asynchronous copy should be finished by here
	wait_group_events(1, &ev);

  	// -----------------------------------------------------------------------------
  	// Some OpenCL compilers don't allow declaring 
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel, 
	// and then passed to non-kernel functions.
	           
	// Partial results of the gradient step
	__local float gradient[ACTUAL_GENOTYPE_LENGTH];

	// -------------------------------------------------------------------
	// Calculate gradients (forces) for intermolecular energy
	// Derived from autodockdev/maps.py
	// -------------------------------------------------------------------
	// Gradient of the intermolecular energy per each ligand atom
	// Also used to store the accummulated gradient per each ligand atom
	__local float gradient_inter_x[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_y[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_z[MAX_NUM_OF_ATOMS];

	// Gradient of the intramolecular energy per each ligand atom
	__local float gradient_intra_x[MAX_NUM_OF_ATOMS];
	__local float gradient_intra_y[MAX_NUM_OF_ATOMS];
	__local float gradient_intra_z[MAX_NUM_OF_ATOMS];

	// -------------------------------------------------------------------
	// Ligand-atom position and partial energies
	__local float calc_coords_x[MAX_NUM_OF_ATOMS];
	__local float calc_coords_y[MAX_NUM_OF_ATOMS];
	__local float calc_coords_z[MAX_NUM_OF_ATOMS];
	__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];

	#if defined (DEBUG_ENERGY_KERNEL)
	__local float partial_interE[NUM_OF_THREADS_PER_BLOCK];
	__local float partial_intraE[NUM_OF_THREADS_PER_BLOCK];
	#endif

	// -----------------------------------------------------------------------------
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

	// Defining lower and upper bounds for genotypes
	__local float lower_bounds_genotype[ACTUAL_GENOTYPE_LENGTH];
	__local float upper_bounds_genotype[ACTUAL_GENOTYPE_LENGTH];

	for (uint gene_counter = get_local_id(0);
	          gene_counter < dockpars_num_of_genes;
	          gene_counter+= NUM_OF_THREADS_PER_BLOCK) {

		// Translation genes ranges are within the gridbox
		if (gene_counter <= 2) {
			lower_bounds_genotype [gene_counter] = 0.0f;
			upper_bounds_genotype [gene_counter] = (gene_counter == 0) ? dockpars_gridsize_x: 
							       (gene_counter == 1) ? dockpars_gridsize_y: 
										     dockpars_gridsize_z;
		// Orientation and torsion genes range between [0, 360]
		// See auxiliary_genetic.cl/map_angle()
		} else {
			lower_bounds_genotype [gene_counter] = 0.0f;
			upper_bounds_genotype [gene_counter] = 360.0f;
		}

		#if defined (DEBUG_MINIMIZER)
		//printf("(%-3u) %-10.7f %-10.7f %-10.7f\n", gene_counter, genotype[gene_counter], lower_bounds_genotype[gene_counter], upper_bounds_genotype[gene_counter]);
		#endif
	}

	// Calculating gradient
	barrier(CLK_LOCAL_MEM_FENCE);

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
			dockpars_smooth,

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

			kerconst_interintra,
			kerconst_intracontrib,
			kerconst_intra,
			kerconst_rotlist,
			kerconst_conform
			,
			rotbonds_const,
			rotbonds_atoms_const,
			num_rotating_atoms_per_rotbond_const
			,
	     		angle_const,
	     		dependence_on_theta_const,
	     		dependence_on_rotangle_const
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
			gradient
			);
	// =============================================================

	// FIRE counters
	__local float velocity [ACTUAL_GENOTYPE_LENGTH];// velocity
	__local float alpha;				// alpha
	__local uint count_success;			// count_success
	__local float dt;				// "dt"

	// Calculating the gradient/velocity norm
	__local float gradient_tmp [ACTUAL_GENOTYPE_LENGTH];
	__local float gradient_norm;
	__local float inv_gradient_norm;
	__local float velocity_tmp [ACTUAL_GENOTYPE_LENGTH];
	__local float velocity_norm;

	__local float velnorm_div_gradnorm;

	// Defining FIRE power
	__local float power_tmp [ACTUAL_GENOTYPE_LENGTH];
	__local float power;

	// Calculating gradient norm
	for (uint gene_counter = get_local_id(0);
	          gene_counter < dockpars_num_of_genes;
	          gene_counter+= NUM_OF_THREADS_PER_BLOCK) {

		 gradient_tmp [gene_counter] = gradient [gene_counter] * gradient [gene_counter];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_local_id(0) == 0) {

		// By the way, initializing
		alpha         = ALPHA_START;
		count_success = 0;
		dt            = DT_MAX_DIV_THREE;

		// Continuing calculation of gradient norm
		gradient_norm = 0.0f;
		
		// Summing up squares
		for (uint i = 0; i < dockpars_num_of_genes; i++) {
			gradient_norm += gradient_tmp [i];
		}
		
		gradient_norm     = native_sqrt(gradient_norm);
		inv_gradient_norm = native_recip(gradient_norm);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Starting velocity
	for (uint gene_counter = get_local_id(0);
	          gene_counter < dockpars_num_of_genes;
	          gene_counter+= NUM_OF_THREADS_PER_BLOCK) {
		 velocity [gene_counter] = - gradient [gene_counter] * inv_gradient_norm * ALPHA_START;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

/*
	if (get_local_id(0) == 0 ){
		printf("dt:%f, DT_MIN:%f, power: %f\n", dt, DT_MIN, power);
	}
*/

	// The termination criteria is based on 
	// a maximum number of iterations, and
	// the minimum step size allowed for single-floating point numbers 
	// (IEEE-754 single float has a precision of about 7 decimal digits)
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

		// ***********************************************************************************	

		// Calculating power
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint gene_counter = get_local_id(0);
	        	  gene_counter < dockpars_num_of_genes;
	        	  gene_counter+= NUM_OF_THREADS_PER_BLOCK) {
			
			// Calculating power
			power_tmp [gene_counter] = gradient [gene_counter] * velocity [gene_counter];

			// Calculating velocity norm
			velocity_tmp [gene_counter] = velocity [gene_counter] * velocity [gene_counter];

			// Calculating gradient norm
			gradient_tmp [gene_counter] = gradient [gene_counter] * gradient [gene_counter];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if (get_local_id(0) == 0) {
			power         = 0.0f;
			velocity_norm = 0.0f;
			gradient_norm = 0.0f;
		
			// Summing dot products
			for (uint i = 0; i < dockpars_num_of_genes; i++) {
				power         += power_tmp    [i];
				velocity_norm += velocity_tmp [i];
				gradient_norm += gradient_tmp [i];
			}

			power             = -power;
			velocity_norm     = native_sqrt(velocity_norm);
			gradient_norm     = native_sqrt(gradient_norm);
			inv_gradient_norm = native_recip(gradient_norm);
			
			// Note: alpha is included as a factor here
			velnorm_div_gradnorm = alpha * velocity_norm * inv_gradient_norm;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// Calculating velocity
		for (uint gene_counter = get_local_id(0);
	        	  gene_counter < dockpars_num_of_genes;
	        	  gene_counter+= NUM_OF_THREADS_PER_BLOCK) {

			velocity [gene_counter] = (1 - alpha) * velocity [gene_counter] - velnorm_div_gradnorm * gradient [gene_counter];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// Going uphill (against the gradient)
		if (power < 0.0f) {
		
			// Using same equation as for starting velocity
			for (uint gene_counter = get_local_id(0);
		        	  gene_counter < dockpars_num_of_genes;
		        	  gene_counter+= NUM_OF_THREADS_PER_BLOCK) {			
			
				velocity [gene_counter] = - gradient [gene_counter] * inv_gradient_norm * ALPHA_START;	
			}

		 	if (get_local_id(0) == 0) {
				count_success = 0;
				alpha         = ALPHA_START;
				dt 	      = dt * DT_DEC; 
				//printf("UPHILL dt:%f, DT_MIN:%f, power: %f, count: %u \n", dt, DT_MIN, power, count_success);
			}
		}
		// Going downhill
		else {
			if (get_local_id(0) == 0) {
				count_success ++;

				// Reaching minimum number of consecutive successful steps (power >= 0)
				if (count_success > SUCCESS_MIN) {
					dt    = fmin (dt * DT_INC, DT_MAX);	// increase dt
					alpha = alpha * ALPHA_DEC; 		// decrease alpha
				}
				//printf("DOWNHILL dt:%f, DT_MIN:%f, power: %f, count: %u \n", dt, DT_MIN, power, count_success);
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// ***********************************************************************************	
		
		for (uint gene_counter = get_local_id(0);
	        	  gene_counter < dockpars_num_of_genes;
	        	  gene_counter+= NUM_OF_THREADS_PER_BLOCK) {			
			
			// Creating new genotypes
			genotype [gene_counter] = genotype [gene_counter] + dt * velocity [gene_counter];	

			// Putting genes back within bounds
			genotype[gene_counter] = fmin(genotype[gene_counter], upper_bounds_genotype[gene_counter]);
			genotype[gene_counter] = fmax(genotype[gene_counter], lower_bounds_genotype[gene_counter]);
		}

		// Calculating gradient
		barrier(CLK_LOCAL_MEM_FENCE);

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
				dockpars_smooth,

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

				kerconst_interintra,
				kerconst_intracontrib,
				kerconst_intra,
				kerconst_rotlist,
				kerconst_conform
				,
				rotbonds_const,
				rotbonds_atoms_const,
				num_rotating_atoms_per_rotbond_const
				,
	     			angle_const,
	     			dependence_on_theta_const,
	     			dependence_on_rotangle_const
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
				gradient
				);
		// =============================================================

/*		
		if ((get_group_id(0) == 0) && (get_local_id(0) == 0)) {
			for(uint i = 0; i < dockpars_num_of_genes; i++) {
				printf("gradient[%u]=%f \n", i, gradient[i]);
			}
		}
*/
			
		// Evaluating genotype
		barrier(CLK_LOCAL_MEM_FENCE);

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
				dockpars_smooth,

				genotype,
				&energy,
				&run_id,
				// Some OpenCL compilers don't allow declaring 
				// local variables within non-kernel functions.
				// These local variables must be declared in a kernel, 
				// and then passed to non-kernel functions.
				calc_coords_x,
				calc_coords_y,
				calc_coords_z,
				partial_energies,
				#if defined (DEBUG_ENERGY_KERNEL)
				partial_interE,
				partial_intraE,
				#endif
#if 0
				true,
#endif
				kerconst_interintra,
				kerconst_intracontrib,
				kerconst_intra,
				kerconst_rotlist,
				kerconst_conform
				);
		// =============================================================

		#if defined (DEBUG_ENERGY_KERNEL)
		if ((get_group_id(0) == 0) && (get_local_id(0) == 0)) {
			for(uint i = 0; i < dockpars_num_of_genes; i++) {
				printf("genotype[%u]=%f \n", i, genotype[i]);
			}
			printf("partial_interE=%f \n", partial_interE[0]);
			printf("partial_intraE=%f \n", partial_intraE[0]);
		}
		#endif

		// Updating number of fire iterations (energy evaluations)
		if (get_local_id(0) == 0) {
	    		iteration_cnt = iteration_cnt + 1;

			#if defined (DEBUG_MINIMIZER)
			printf("# minimizer-iters: %-3u, E: %10.7f\n", iteration_cnt, energy);
			#endif

			#if defined (DEBUG_ENERGY_KERNEL)
			printf("%-18s [%-5s]---{%-5s}   [%-10.7f]---{%-10.7f}\n", "-ENERGY-KERNEL5-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
			#endif
		}
		barrier(CLK_LOCAL_MEM_FENCE);

  	//} //while (dt > DT_MIN);
	} while ((iteration_cnt < dockpars_max_num_of_iters) && (dt > DT_MIN));

	// -----------------------------------------------------------------------------

  	// Updating eval counter and energy
	if (get_local_id(0) == 0) {
		dockpars_evals_of_new_entities[run_id*dockpars_pop_size+entity_id] += iteration_cnt;
		dockpars_energies_next[run_id*dockpars_pop_size+entity_id] = energy;

		#if defined (DEBUG_MINIMIZER)
		printf("-------> End of grad-min cycle, num of evals: %u, final energy: %.7f\n", iteration_cnt, energy);
		#endif
	}

	// Mapping torsion angles
	for (uint gene_counter = get_local_id(0);
	     	  gene_counter < dockpars_num_of_genes;
	          gene_counter+= NUM_OF_THREADS_PER_BLOCK) {
		   if (gene_counter >= 3) {
			    map_angle(&(genotype[gene_counter]));
		   }
	}

	// Updating old offspring in population
	barrier(CLK_LOCAL_MEM_FENCE);

	event_t ev2 = async_work_group_copy(dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
			                    genotype,
			                    dockpars_num_of_genes, 0);

	// Asynchronous copy should be finished by here
	wait_group_events(1, &ev2);
}
