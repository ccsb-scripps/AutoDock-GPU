// Implementation of the gradient-based minimizer
// This will ideally replace the LS

// Original source in https://stackoverflow.com/a/27910756




// FIXME: original call of stepGPU
// stepGPU<<<iDivUp(M, BLOCK_SIZE), BLOCK_SIZE>>>
// foo<<<N,1>>> means N blocks and 1thread in each block

__kernel void __attribute__ ((reqd_work_group_size(NUM_OF_THREADS_PER_BLOCK,1,1)))
gradient_minimizer(	char   dockpars_num_of_atoms,
			char   dockpars_num_of_atypes,
			int    dockpars_num_of_intraE_contributors,
			char   dockpars_gridsize_x,
			char   dockpars_gridsize_y,
			char   dockpars_gridsize_z,
			float  dockpars_grid_spacing,

			#if defined (RESTRICT_ARGS)
	  __global const float* restrict dockpars_fgrids, // cannot be allocated in __constant (too large)
			#else
	  __global const float* dockpars_fgrids,          // cannot be allocated in __constant (too large)
			#endif
			
			int    dockpars_rotbondlist_length,
			float  dockpars_coeff_elec,
			float  dockpars_coeff_desolv,

			#if defined (RESTRICT_ARGS)
	  __global float* restrict dockpars_conformations_next,
	  __global float* restrict dockpars_energies_next,
	  //__global int*   restrict dockpars_evals_of_new_entities,
	  __global unsigned int* restrict dockpars_prng_states,
			#else
	  __global float* dockpars_conformations_next,
	  __global float* dockpars_energies_next,
	  //__global int*   dockpars_evals_of_new_entities,
	  __global unsigned int* dockpars_prng_states,
			#endif

			int    dockpars_pop_size,
			int    dockpars_num_of_genes,
			float  dockpars_lsearch_rate,
			unsigned int dockpars_num_of_lsentities,
			//float  dockpars_rho_lower_bound,
			//float  dockpars_base_dmov_mul_sqrt3,
			//float  dockpars_base_dang_mul_sqrt3,
			//unsigned int dockpars_cons_limit,
			//unsigned int dockpars_max_num_of_iters,
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
		    	unsigned int      gradMin_maxiter,
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
	// Determine entity, and its run and energy
	__local int   entity_id;
	__local int   run_id;
  	__local float local_energy;

	if (get_local_id(0) == 0)
	{
		entity_id = get_group_id(0) % dockpars_num_of_lsentities;
		run_id = get_group_id(0) / dockpars_num_of_lsentities;
		
		// Since entity-ID=0 is the best one due to elitism, it should be subjected to random selection
		if (entity_id == 0)
			if (100.0f*gpu_randf(dockpars_prng_states) > dockpars_lsearch_rate)
				entity_id = dockpars_num_of_lsentities;	 // If entity-ID=0 is not selected according to LS-rate,
									 // then choose another entity

		local_energy = dockpars_energies_next[run_id*dockpars_pop_size+entity_id];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

  	// -----------------------------------------------------------------------------
	// Initialize variables for gradient minimizer
  	__local float        local_gNorm;                                  // gradient norm (shared in the CU), originally as "gnorm"
  	__local unsigned int local_nIter;                                  // iteration counter, originally as "niter"
  	__local float        local_perturbation [ACTUAL_GENOTYPE_LENGTH];  // perturbation, originally as "dx"
  	__local float        local_genotype[ACTUAL_GENOTYPE_LENGTH];       // optimization vector, originally as "d_x"

	if (get_local_id(0) == 0) {
		local_gNorm = FLT_MAX;
    		local_nIter = 0;
  	}

  	for(unsigned int i=get_local_id(0); 
			 i<dockpars_num_of_genes; 
			 i+=NUM_OF_THREADS_PER_BLOCK) {
    		local_perturbation[i] = FLT_MAX;
  	}

	barrier(CLK_LOCAL_MEM_FENCE);

  	async_work_group_copy(local_genotype,
  			      dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
                              dockpars_num_of_genes,0);

  	// -----------------------------------------------------------------------------
  	// Allocate space
	__local float local_gradient      [ACTUAL_GENOTYPE_LENGTH]; // gradient, originally as "d_g"
	__local float local_genotype_new  [ACTUAL_GENOTYPE_LENGTH]; // new actual solution, originally as "d_xnew"
	__local float local_genotype_diff [ACTUAL_GENOTYPE_LENGTH]; // difference between actual and old solution, originally as "d_xdiff"

	// Some OpenCL compilers don't allow declaring 
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel, 
	// and then passed to non-kernel functions.
  	__local bool  is_gradDescentEn;                             // used in is_gradDescent_enabled()
	__local float innerProduct;                                 // used in inner_product()

	// -------------------------------------------------------------------
	// L30nardoSV
	// Calculate gradients (forces) for intermolecular energy
	// Derived from autodockdev/maps.py
	// -------------------------------------------------------------------
	// Variables to store gradient of 
	// the intermolecular energy per each ligand atom

	// Some OpenCL compilers don't allow declaring 
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel, 
	// and then passed to non-kernel functions.
	__local float gradient_inter_x[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_y[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_z[MAX_NUM_OF_ATOMS];

	// Enable gradient calculation for this kernel
	__local bool  is_enabled_gradient_calc;
	if (get_local_id(0) == 0) {
		is_enabled_gradient_calc = true;
	}

	// Final gradient resulting out of gradient calculation
	__local float gradient_genotype[GENOTYPE_LENGTH_IN_GLOBMEM];
	// -------------------------------------------------------------------

	// Variables to store partial energies
	__local float calc_coords_x[MAX_NUM_OF_ATOMS];
	__local float calc_coords_y[MAX_NUM_OF_ATOMS];
	__local float calc_coords_z[MAX_NUM_OF_ATOMS];
	__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];

	// -----------------------------------------------------------------------------
	// Perform gradient-descent iterations
	while (is_gradDescent_enabled(&local_gNorm,
    				      gradMin_tol,
    				      &local_nIter,
    				      gradMin_maxiter,
    				      local_perturbation,
    				      gradMin_conformation_min_perturbation,
    				      &is_gradDescentEn,
				      dockpars_num_of_genes) == true) {

    		stepGPU(// Args for minimization
			local_genotype,
		        local_genotype_new,
      			local_genotype_diff,
      			local_gradient,
      			gradMin_alpha,
      			gradMin_h,
      			dockpars_num_of_genes, //gradMin_M

			// Args for energy and gradient calculation
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

		 	// -------------------------------------------------------------------
		 	// L30nardoSV
		 	// Gradient-related arguments
		 	// Calculate gradients (forces) for intermolecular energy
		 	// Derived from autodockdev/maps.py
		 	// -------------------------------------------------------------------
			,
			&is_enabled_gradient_calc,
			gradient_inter_x,
			gradient_inter_y,
			gradient_inter_z,

			gradient_genotype
      			);
			// -------------------------------------------------------------------

    		// Store the norm of all gradients
    		local_gNorm = native_sqrt(inner_product(local_gradient, 
						        local_gradient,
                                                        dockpars_num_of_genes,
                                                        &innerProduct));

    		// Here it is stored a reduced norm of all perturbations
	    	local_perturbation[0] = native_sqrt(inner_product(local_genotype_diff,
                                                                  local_genotype_diff,
                                                                  dockpars_num_of_genes,
                                                                  &innerProduct));
    		local_nIter = local_nIter + 1;
  	}
	// -----------------------------------------------------------------------------

  	// Functional calculation
	// stepGPU () calls gpu_calc_energy ()
	// the former calculates energy + gradients
	
  	// Copy final genotype and energy into global memory
	if (get_local_id(0) == 0) {
		  dockpars_energies_next[run_id*dockpars_pop_size+entity_id] = local_energy;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	async_work_group_copy(dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
			      local_genotype,
			      dockpars_num_of_genes,0);


  	// copy final "local" results into "global" memory
  	local_nIter = local_nIter - 1;
}
