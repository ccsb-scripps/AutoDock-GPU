// Implementation of the gradient-based minimizer
// This will ideally replace the LS

// Original source in https://stackoverflow.com/a/27910756

#define DEBUG_MINIMIZER

#define TRANGENE_ALPHA 1E-8
#define ROTAGENE_ALPHA 1E-15
#define TORSGENE_ALPHA 1E-3


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
	     __constant int*   rotbonds_const,
	     __constant int*   rotbonds_atoms_const,
	     __constant int*   num_rotating_atoms_per_rotbond_const,
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

#if 0
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
#endif


	if (get_local_id(0) == 0)
	{
		// Choosing a random entity out of the entire population
		entity_id = (uint)(dockpars_pop_size * gpu_randf(dockpars_prng_states));
		run_id = get_group_id(0);
		
		local_energy = dockpars_energies_next[run_id*dockpars_pop_size+entity_id];

		#if defined (DEBUG_MINIMIZER)
		//printf("run_id:  %5u entity_id: %5u  local_energy: %.5f\n", run_id, entity_id, local_energy);
		//printf("%-40s %f\n", "BEFORE GRADIENT - local_energy: ", local_energy);
		#endif

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

	// Accummulated gradient
	__local float gradient_x[MAX_NUM_OF_ATOMS];
	__local float gradient_y[MAX_NUM_OF_ATOMS];
	__local float gradient_z[MAX_NUM_OF_ATOMS];	
	// -------------------------------------------------------------------

	// Variables to store partial energies
	__local float calc_coords_x[MAX_NUM_OF_ATOMS];
	__local float calc_coords_y[MAX_NUM_OF_ATOMS];
	__local float calc_coords_z[MAX_NUM_OF_ATOMS];
	__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];

	#if defined (DEBUG_ENERGY)
	__local float partial_interE [NUM_OF_THREADS_PER_BLOCK];
	__local float partial_intraE [NUM_OF_THREADS_PER_BLOCK];
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

	do {
		#if 0
		// Specific input genotypes for a ligand with no rotatable bonds (1ac8).
		// Translation genes must be expressed in grids in OCLADock (local_genotype [0|1|2]).
		// However, for testing purposes, 
		// we start using translation values in real space (Angstrom): {31.79575, 93.743875, 47.699875}
		// Rotation genes are expresed in the Shoemake space: local_genotype [3|4|5]
		// xyz_gene_gridspace = gridcenter_gridspace + (input_gene_realspace - gridcenter_realspace)/gridsize

		// 1ac8				
		local_genotype[0] = 30 + (31.79575  - 31.924) / 0.375;
		local_genotype[1] = 30 + (93.743875 - 93.444) / 0.375;
		local_genotype[2] = 30 + (47.699875 - 47.924) / 0.375;
		local_genotype[3] = 0.1f;
		local_genotype[4] = 0.5f;
		local_genotype[5] = 0.9f;
		#endif

		#if 0
		// 3tmn
		local_genotype[0] = 30 + (ligand_center_x - grid_center_x) / 0.375;
		local_genotype[1] = 30 + (ligand_center_y - grid_center_y) / 0.375;
		local_genotype[2] = 30 + (ligand_center_z - grid_center_z) / 0.375;
		local_genotype[3] = shoemake_gene_u1;
		local_genotype[4] = shoemake_gene_u2;
		local_genotype[5] = shoemake_gene_u3;
		local_genotype[6] = 0.0f;
		local_genotype[7] = 0.0f;
		local_genotype[8] = 0.0f;
		local_genotype[9] = 0.0f;
		local_genotype[10] = 0.0f;
		local_genotype[11] = 0.0f;
		local_genotype[12] = 0.0f;
		local_genotype[13] = 0.0f;
		local_genotype[14] = 0.0f;
		local_genotype[15] = 0.0f;
		local_genotype[16] = 0.0f;
		local_genotype[17] = 0.0f;
		local_genotype[18] = 0.0f;
		local_genotype[19] = 0.0f;
		local_genotype[20] = 0.0f;
		#endif

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
				local_gradient
				);
		// =============================================================

		barrier(CLK_LOCAL_MEM_FENCE);

		float alpha;

		for(uint i = get_local_id(0); 
			 i < dockpars_num_of_genes; 
			 i+= NUM_OF_THREADS_PER_BLOCK) {

	     		// Taking step
			//local_genotype_new[i]  = local_genotype[i] - gradMin_alpha * local_gradient[i];	

			if (i<3)      { alpha = TRANGENE_ALPHA;	}
			else if (i<6) { alpha = ROTAGENE_ALPHA; } 
			else 	      { alpha = TORSGENE_ALPHA;	}

			local_genotype_new[i]  = local_genotype[i] - alpha * local_gradient[i];	

			#if defined (DEBUG_MINIMIZER)
			printf("(%u) %-15.15f %-10.10f %-10.10f %-10.10f\n", i, alpha, local_genotype[i], local_gradient[i], local_genotype_new[i]);
			#endif

	     		// Updating termination metrics
	     		local_genotype_diff[i] = local_genotype_new[i] - local_genotype[i];

	     		// Updating current solution
	     		local_genotype[i] = local_genotype_new[i];

			// Storing all gene-based perturbations
			local_perturbation[i] = local_genotype_diff [i];
	   	}

		// Updating number of stepest-descent iterations
		if (get_local_id(0) == 0) {
	    		local_nIter = local_nIter + 1;

			#if defined (DEBUG_MINIMIZER)
			printf("Number of grad-minimizer iterations: %u\n", local_nIter);
			#endif
		}




	    	// Storing the norm of all gradients
		gradient_norm(local_gradient, dockpars_num_of_genes, dotProduct, &local_gNorm);

		/*
		// Storing all gene-based perturbations
		for(uint i = get_local_id(0); 
			 i < dockpars_num_of_genes; 
			 i+= NUM_OF_THREADS_PER_BLOCK) {
	     		local_perturbation[i] = local_genotype_diff [i];
   		}
		*/

/*
		if (get_local_id(0) == 0) {
			printf("Entity: %u, Run: %u, minimized E: %f\n", entity_id, run_id, local_energy);
		}		
*/
		is_gradDescent_enabled(
				      	is_perturb_gt_gene_min,
				      	&local_gNorm,
    				      	gradMin_tol,
    				      	&local_nIter,
    				      	gradMin_maxiter,
    				      	local_perturbation,
					local_genotype,
    				      	gradMin_conformation_min_perturbation,
				      	dockpars_num_of_genes,
    				      	&is_gradDescentEn
				      );
  	} while (is_gradDescentEn == true);

	// -----------------------------------------------------------------------------

	#if 0
	// 1ac8
	local_genotype[0] = 30 + (31.79575  - 31.924) / 0.375;
	local_genotype[1] = 30 + (93.743875 - 93.444) / 0.375;
	local_genotype[2] = 30 + (47.699875 - 47.924) / 0.375;
	local_genotype[3] = 0.1f;
	local_genotype[4] = 0.5f;
	local_genotype[5] = 0.9f;
	#endif

	#if 0
	// 7cpa
	local_genotype[0] = 30 + (ligand_center_x - grid_center_x) / 0.375;
	local_genotype[1] = 30 + (ligand_center_y - grid_center_y) / 0.375;
	local_genotype[2] = 30 + (ligand_center_z - grid_center_z) / 0.375;
	local_genotype[3] = shoemake_gene_u1;
	local_genotype[4] = shoemake_gene_u2;
	local_genotype[5] = shoemake_gene_u3;
	local_genotype[6] = 0.0f;
	local_genotype[7] = 0.0f;
	local_genotype[8] = 0.0f;
	local_genotype[9] = 0.0f;
	local_genotype[10] = 0.0f;
	local_genotype[11] = 0.0f;
	local_genotype[12] = 0.0f;
	local_genotype[13] = 0.0f;
	local_genotype[14] = 0.0f;
	local_genotype[15] = 0.0f;
	local_genotype[16] = 0.0f;
	local_genotype[17] = 0.0f;
	local_genotype[18] = 0.0f;
	local_genotype[19] = 0.0f;
	local_genotype[20] = 0.0f;
	#endif

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
			#if defined (DEBUG_ENERGY)
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

	//barrier(CLK_LOCAL_MEM_FENCE);

	#if defined (DEBUG_ENERGY)
	if (get_local_id(0) == 0) {
		printf("%-20s %-10.8f\n", "GRIDE: ", partial_interE[0]);
		printf("%-20s %-10.8f\n", "INTRAE: ", partial_intraE[0]);
		printf("\n");
	}
	#endif

	#if defined (DEBUG_MINIMIZER)
	if (get_local_id(0) == 0) {
		printf("%-40s %f\n", "AFTER- GRADIENT - local_energy: ", local_energy);
		//printf("\n");
	}
	#endif

  	// Copying final genotype and energy into global memory
	if (get_local_id(0) == 0) {
		  dockpars_energies_next[run_id*dockpars_pop_size+entity_id] = local_energy;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	async_work_group_copy(dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
			      local_genotype,
			      dockpars_num_of_genes, 0);


	// FIXME: maybe not used outside?
  	// Copying final "local" results into "global" memory
/*
  	local_nIter = local_nIter - 1;
*/
}
