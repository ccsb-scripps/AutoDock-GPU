//#define DEBUG_GRADDESC_ENABLED


// Implementation of auxiliary functions 
// for the gradient-based minimizer
void is_gradDescent_enabled(
			    __local    bool* 	is_genotype_valid,
                            __local    uint*    local_nIter,
                                       uint     gradMin_maxiter,
			    __local    float*   local_candidate_genotype,
				       uint     gradMin_numElements,
                            __local    bool*    is_gradDescentEn
)
{
	bool is_gNorm_gt_gMin;
	bool is_nIter_lt_maxIter;
	bool is_valid;
	
	if (get_local_id(0) == 0) {
		is_nIter_lt_maxIter = (local_nIter[0] < gradMin_maxiter);
		is_valid = true;
	}	

/*
	// Verifying that Shoemake genes do not get out of valid range.
	// If they do so, then set them to 0.0f
	if (get_local_id(0) < 3){
		if ((local_candidate_genotype[get_local_id(0)] < 0.0f) && (local_candidate_genotype[get_local_id(0)] > 1.0f)) {
			local_candidate_genotype[get_local_id(0)] = 0.0f;
		}
	}
*/	

	// Using every of its genes, let's determine 
	// if candidate genotype is valid 
  	for(uint i = get_local_id(0); 
		 i < gradMin_numElements; 
		 i+= NUM_OF_THREADS_PER_BLOCK) {
   		is_genotype_valid[i] = !isnan(local_candidate_genotype[i]) && !isinf(local_candidate_genotype[i]);
  	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_local_id(0) == 0) {
		
		// Reduce all "valid" values of genes
		// into their corresponding genotype
	  	for(uint i = 0; 
			 i < gradMin_numElements;
			 i++) {
	    		is_valid = is_valid && is_genotype_valid[i];
			
			#if defined (DEBUG_GRADDESC_ENABLED)
			//printf("is_genotype_valid[%u]?: %s\n", i, (is_genotype_valid[i] == true)?"yes":"no");
			#endif

	  	}
	
		// Reduce all previous partial evaluations (nIter, valid) into a final one
		*is_gradDescentEn = is_nIter_lt_maxIter && is_valid;

		#if defined (DEBUG_GRADDESC_ENABLED)
		//printf("is_gNorm_gt_gMin?: %s\n", (is_gNorm_gt_gMin == true)?"yes":"no");
		//printf("is_nIter_lt_maxIter?: %s\n", (is_nIter_lt_maxIter == true)?"yes":"no");
		//printf("is_valid?: %s\n", (is_valid == true)?"yes":"no");
		printf("Continue grad-mini?: %s\n", (*is_gradDescentEn == true)?"yes":"no");
		#endif
	}
}
