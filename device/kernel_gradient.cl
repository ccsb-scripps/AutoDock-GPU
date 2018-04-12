// Implementation of the gradient-based minimizer
// This will ideally replace the LS

// Original source in https://stackoverflow.com/a/27910756




// FIXME: original call of stepGPU
// stepGPU<<<iDivUp(M, BLOCK_SIZE), BLOCK_SIZE>>>
// foo<<<N,1>>> means N blocks and 1thread in each block

__kernel void __attribute__ ((reqd_work_group_size(NUM_OF_THREADS_PER_BLOCK,1,1)))
gradient_minimizer(	
			char   dockpars_num_of_atoms,
			char   dockpars_num_of_atypes,
			int    dockpars_num_of_intraE_contributors,
			char   dockpars_gridsize_x,
			char   dockpars_gridsize_y,
			char   dockpars_gridsize_z,
			float  dockpars_grid_spacing,
	 __global const float* restrict dockpars_fgrids, // This is too large to be allocated in __constant 
			int    dockpars_rotbondlist_length,
			float  dockpars_coeff_elec,
			float  dockpars_coeff_desolv,
	  __global      float* restrict dockpars_conformations_next,
	  __global      float* restrict dockpars_energies_next,
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
    			// Specific gradient-minimizer args
    //  __global float* restrict dockpars_conformations_next,   // initial population
                                                                // whose (some) entities (genotypes) are to be minimized
    			float             gradMin_tol,
		    	uint      	  gradMin_maxiter,
	    		float             gradMin_alpha,
	    		float             gradMin_h,
    	     __constant float* gradMin_conformation_min_perturbation     // minimal values for gene perturbation, originally as the scalar "dxmin"
    	     //unsigned int      gradMin_M,                              // dimensionality of the input data (=num_of_genes)
             //  __global float* restrict dockpars_conformations_next,   // optimized genotype are to be store back here, originally as "d_xopt"
             //  __global float* restrict dockpars_energies_next,        // minimized energy, originally as "fopt"

    // Following kernel args were used to send counters back to host
    // Commented here as they are not needed
    //__global unsigned int* restrict gradMin_nIter,
    //__global float*        restrict gradMin_gNorm,
    //__global float*        restrict gradMin_perturbation        // originally as "dx"
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
	// Determining entity, and its run and energy
	__local int   entity_id;
	__local int   run_id;
  	__local float local_energy;

	if (get_local_id(0) == 0)
	{
		entity_id = get_group_id(0) % dockpars_num_of_lsentities;
		run_id = get_group_id(0) / dockpars_num_of_lsentities;
		
		// Since entity-ID=0 is the best one due to elitism, 
		// it should be subjected to random selection
		if (entity_id == 0) {
/*
			if (100.0f*gpu_randf(dockpars_prng_states) > dockpars_lsearch_rate) {
				entity_id = dockpars_num_of_lsentities;	 // If entity-ID=0 is not selected according to LS-rate,
									 // then choose another entity

			}
*/
			entity_id = 1;
		}

		local_energy = dockpars_energies_next[run_id*dockpars_pop_size+entity_id];

/*
		printf("run_id: %u, entity_id: %u, local_energy: %f\n", run_id, entity_id, local_energy);
*/

/*
		printf("BEFORE GRADIENT - local_energy: %f\n", local_energy);
*/

	}

	barrier(CLK_LOCAL_MEM_FENCE);

  	// -----------------------------------------------------------------------------
	// Initializing variables for gradient minimizer
  	__local float   local_gNorm;                                  // gradient norm (shared in the CU), originally as "gnorm"
  	__local uint 	local_nIter;                                  // iteration counter, originally as "niter"
  	__local float   local_perturbation [ACTUAL_GENOTYPE_LENGTH];  // perturbation, originally as "dx"
  	__local float   local_genotype     [ACTUAL_GENOTYPE_LENGTH];  // optimization vector, originally as "d_x"

	if (get_local_id(0) == 0) {
		local_gNorm = FLT_MAX;
    		local_nIter = 0;
  	}

  	for(uint i = get_local_id(0); 
		 i < dockpars_num_of_genes; 
		 i+= NUM_OF_THREADS_PER_BLOCK) {
    		local_perturbation[i] = FLT_MAX;
  	}

	barrier(CLK_LOCAL_MEM_FENCE);

  	async_work_group_copy(local_genotype,
  			      dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
                              dockpars_num_of_genes, 0);

  	// -----------------------------------------------------------------------------
  	// Some OpenCL compilers don't allow declaring 
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel, 
	// and then passed to non-kernel functions.
	
	// Partial results in "is_gradDescent_enabled()"
	__local bool is_perturb_gt_gene_min [ACTUAL_GENOTYPE_LENGTH];
  	__local bool is_gradDescentEn;                

	// Partial results of the gradient step
	__local float local_gradient      [ACTUAL_GENOTYPE_LENGTH]; // gradient, originally as "d_g"
	__local float local_genotype_new  [ACTUAL_GENOTYPE_LENGTH]; // new actual solution, originally as "d_xnew"
	__local float local_genotype_diff [ACTUAL_GENOTYPE_LENGTH]; // difference between actual and old solution, originally as "d_xdiff"

	// Partial results in "dot" product
	__local float dotProduct          [ACTUAL_GENOTYPE_LENGTH];                                 

	// -------------------------------------------------------------------
	// Calculate gradients (forces) for intermolecular energy
	// Derived from autodockdev/maps.py
	// -------------------------------------------------------------------
	// Variables to store gradient of 
	// the intermolecular energy per each ligand atom
	__local float gradient_inter_x[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_y[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_z[MAX_NUM_OF_ATOMS];

	// Variables to store gradient of 
	// the intramolecular energy per each ligand atom
	__local float gradient_intra_x[MAX_NUM_OF_ATOMS];
	__local float gradient_intra_y[MAX_NUM_OF_ATOMS];
	__local float gradient_intra_z[MAX_NUM_OF_ATOMS];
	__local float gradient_per_intracontributor[MAX_INTRAE_CONTRIBUTORS];
	

	// -------------------------------------------------------------------

	// Variables to store partial energies
	__local float calc_coords_x[MAX_NUM_OF_ATOMS];
	__local float calc_coords_y[MAX_NUM_OF_ATOMS];
	__local float calc_coords_z[MAX_NUM_OF_ATOMS];
	__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];

	// -----------------------------------------------------------------------------
	// Perform gradient-descent iterations
	while (is_gradDescent_enabled(
				      is_perturb_gt_gene_min,
				      &local_gNorm,
    				      gradMin_tol,
    				      &local_nIter,
    				      gradMin_maxiter,
    				      local_perturbation,
    				      gradMin_conformation_min_perturbation,
    				      &is_gradDescentEn,
				      dockpars_num_of_genes) == true) {
		
		// Calculating gradient
		// =============================================================
		gpu_calc_gradient(
				dockpars_rotbondlist_length,
				dockpars_num_of_atoms,
				dockpars_gridsize_x,
				dockpars_gridsize_y,
				dockpars_gridsize_z,
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
				local_genotype,
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
				ref_orientation_quats_const
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
				gradient_per_intracontributor,
				local_gradient
				);
		// =============================================================

		for(uint i = get_local_id(0); 
			 i < dockpars_num_of_genes; 
			 i+= NUM_OF_THREADS_PER_BLOCK) {

	     		// Taking step
	     		local_genotype_new[i]  = local_genotype[i] - gradMin_alpha * local_gradient[i];

	     		// Updating terminatiodockpars_num_of_genesn metrics
	     		local_genotype_diff[i] = local_genotype_new[i] - local_genotype[i];

	     		// Updating current solution
	     		local_genotype[i] = local_genotype_new[i];
	   	}

		// Updating number of stepest-descent iterations
		if (get_local_id(0) == 0) {
	    		local_nIter = local_nIter + 1;
		}

	    	// Storing the norm of all gradients
		local_gNorm = inner_product(local_gradient, local_gradient, dockpars_num_of_genes, dotProduct);

		if (get_local_id(0) == 0) {
			local_gNorm = native_sqrt(local_gNorm);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Storing all gene-based perturbations
		for(uint i = get_local_id(0); 
			 i < dockpars_num_of_genes; 
			 i+= NUM_OF_THREADS_PER_BLOCK) {
	     		local_perturbation[i] = local_genotype_diff [i];
   		}

/*
		if (get_local_id(0) == 0) {
			printf("Entity: %u, Run: %u, minimized E: %f\n", entity_id, run_id, local_energy);
		}		
*/

/*
		if (get_local_id(0) == 0) {
			printf("Number of gradient iterations: %u\n", local_nIter);
		}
*/
  	}
	// -----------------------------------------------------------------------------

  	// Calculating energy
	// =============================================================
	gpu_calc_energy(dockpars_rotbondlist_length,
			dockpars_num_of_atoms,
			dockpars_gridsize_x,
			dockpars_gridsize_y,
			dockpars_gridsize_z,
			dockpars_fgrids,
			dockpars_num_of_atypes,
			dockpars_num_of_intraE_contributors,
			dockpars_grid_spacing,
			dockpars_coeff_elec,
			dockpars_qasp,
			dockpars_coeff_desolv,
			local_genotype,
			&local_energy,
			&run_id,
			// Some OpenCL compilers don't allow declaring 
			// local variables within non-kernel functions.
			// These local variables must be declared in a kernel, 
			// and then passed to non-kernel functions.
			calc_coords_x,
			calc_coords_y,
			calc_coords_z,
			partial_energies,

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

/*
	if (get_local_id(0) == 0) {
		printf("AFTER- GRADIENT - local_energy: %f\n\n", local_energy);
	}
*/

  	// Copying final genotype and energy into global memory
	if (get_local_id(0) == 0) {
		  dockpars_energies_next[run_id*dockpars_pop_size+entity_id] = local_energy;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	async_work_group_copy(dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
			      local_genotype,
			      dockpars_num_of_genes,0);


	// FIXME: maybe not used outside?
  	// Copying final "local" results into "global" memory
  	local_nIter = local_nIter - 1;
}
