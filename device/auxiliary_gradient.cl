// Implementation of auxiliary functions for the gradient-based minimizer

bool is_gradDescent_enabled(__local    float*        a_gNorm,
                                       float         gradMin_tol,
                            __local    unsigned int* a_nIter,
                                       unsigned int  gradMin_maxiter,
                            __local    float*        a_perturbation,
                            __constant float*        gradMin_conformation_min_perturbation,
                            __local    bool*         is_gradDescentEn,
				       uint          gradMin_numElements)
{
	bool is_gNorm_gt_gMin       = (a_gNorm[0] >= gradMin_tol);
	bool is_nIter_lt_maxIter    = (a_nIter[0] <= gradMin_maxiter);

	bool is_perturb_gt_gene_min [ACTUAL_GENOTYPE_LENGTH];
	bool is_perturb_gt_genotype = true;

	// For every gene, let's determine 
	// if perturbation is greater than min conformation
  	for(uint i=get_local_id(0); 
		 i<gradMin_numElements; 
		 i+=NUM_OF_THREADS_PER_BLOCK) {
   		is_perturb_gt_gene_min[i] = (a_perturbation[i] >= gradMin_conformation_min_perturbation[i]);
  	}

	barrier(CLK_LOCAL_MEM_FENCE);

  	// Reduce all is_perturb_gt_gene_min's 
	// into their corresponding genotype
  	for(uint i=get_local_id(0); 
		 i<gradMin_numElements;
		 i+=NUM_OF_THREADS_PER_BLOCK) {
    		is_perturb_gt_genotype = is_perturb_gt_genotype && is_perturb_gt_gene_min[i];
  	}

  	barrier(CLK_LOCAL_MEM_FENCE);

  	// Reduce all three previous 
	// partial evaluations (gNorm, nIter, perturb) into a final one
  	if (get_local_id(0) == 0) {
      		is_gradDescentEn[0] = is_gNorm_gt_gMin && is_nIter_lt_maxIter && is_perturb_gt_genotype;
  	}

  	barrier(CLK_LOCAL_MEM_FENCE);

  	return is_gradDescentEn[0];
}

void stepGPU (// Args for minimization
	      __local float*       local_genotype,         // originally as "d_x"
              __local float*       local_genotype_new,     // originally as "d_xnew"
              __local float*       local_genotype_diff,    // originally as "d_xdiff"
              __local float*       local_gradient,         // originally as "d_g"
                      float        gradMin_alpha,          // originally as "alpha"
                      float        gradMin_h,              // originally as "h"
                      unsigned int gradMin_inputSize,      // originally as "M". initially labelled as "gradMin_M"

	       // Args for energy and gradient calculation
		      int    	   dockpars_rotbondlist_length,
		      char	   dockpars_num_of_atoms,
		      char         dockpars_gridsize_x,
		      char         dockpars_gridsize_y,
		      char         dockpars_gridsize_z,
		#if defined (RESTRICT_ARGS)
    __global const float* restrict dockpars_fgrids, // cannot be allocated in __constant (too large)
		#else
    __global const float*          dockpars_fgrids, // cannot be allocated in __constant (too large)
		#endif
	              char   	   dockpars_num_of_atypes,
		      int    	   dockpars_num_of_intraE_contributors,
		      float  	   dockpars_grid_spacing,
	              float  	   dockpars_coeff_elec,
	              float        dockpars_qasp,
		      float  	   dockpars_coeff_desolv,

              __local float* 	   genotype,
	      __local float* 	   energy,
              __local int*   	   run_id,

              // Some OpenCL compilers don't allow declaring 
	      // local variables within non-kernel functions.
	      // These local variables must be declared in a kernel, 
	      // and then passed to non-kernel functions.
		    __local float* calc_coords_x,
		    __local float* calc_coords_y,
		    __local float* calc_coords_z,
		    __local float* partial_energies,

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
                 __constant float* ref_orientation_quats_const

		 // -------------------------------------------------------------------
		 // L30nardoSV
		 // Gradient-related arguments
		 // Calculate gradients (forces) for intermolecular energy
		 // Derived from autodockdev/maps.py
		 // -------------------------------------------------------------------
		
		 // "is_enabled_gradient_calc": enables gradient calculation.
		 // In Genetic-Generation: no need for gradients
		 // In Gradient-Minimizer: must calculate gradients
		 ,
		 __local bool*  is_enabled_gradient_calc,
	    	 __local float* gradient_inter_x,
	         __local float* gradient_inter_y,
	         __local float* gradient_inter_z,

		 __local float* gradient_genotype
)
{
	// Calculate gradient

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
			genotype,
			energy,
			run_id,
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
			is_enabled_gradient_calc,
			gradient_inter_x,
			gradient_inter_y,
			gradient_inter_z,

			gradient_genotype
			);
			// -------------------------------------------------------------------
	// =============================================================


	// TODO: Transform gradients_inter_{x|y|z} 
	// into local_gradients[i] (with four quaternion genes)
	// Derived from autodockdev/motions.py/forces_to_delta_genes()

	







	// TODO: Transform local_gradients[i] (with four quaternion genes)
	// into local_gradients[i] (with three Shoemake genes)
	// Derived from autodockdev/motions.py/_get_cube3_gradient()



	for(unsigned int i=get_local_id(0); 
			 i<gradMin_inputSize; 
			 i+=NUM_OF_THREADS_PER_BLOCK) {
     		// Take step
     		// FIXME: add conditional evaluation of max grad
     		local_genotype_new[i]  = local_genotype[i] - gradMin_alpha * local_gradient[i];

     		// Update termination metrics
     		local_genotype_diff[i] = local_genotype_new[i] - local_genotype[i];

     		// Update current solution
     		local_genotype[i] = local_genotype_new[i];
   	}
}


float inner_product(__local float*       vector1,
                    __local float*       vector2,
                            unsigned int inputSize,
                    __local float*       init) {

	if(get_local_id(0) == 0) {
		init[0] = 0.0f;
  	}

  	barrier(CLK_LOCAL_MEM_FENCE);

	for(unsigned int i=get_local_id(0); 
			 i<inputSize; 
			 i+=NUM_OF_THREADS_PER_BLOCK) {
		init[0] += vector1[i] * vector2[i];
  	}

	return init[0];
}






// Implementation of gradient calculator
// Originally written in Python by Diogo Martins
// Initially coded within gpu_calc_energy()

