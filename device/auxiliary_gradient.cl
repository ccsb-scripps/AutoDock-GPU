#define DEBUG_GRADDESC_ENABLED


// Implementation of auxiliary functions 
// for the gradient-based minimizer
void is_gradDescent_enabled(
			    __local    bool* 	is_perturb_gt_gene_min,
			    __local    float*   local_gNorm,
                                       float    gradMin_tol,
                            __local    uint*    local_nIter,
                                       uint     gradMin_maxiter,
                            __local    float*   local_perturbation,
			    __local    float*   local_genotype,
                            __constant float*   gradMin_conformation_min_perturbation,
				       uint     gradMin_numElements,
                            __local    bool*    is_gradDescentEn
)
{
	bool is_gNorm_gt_gMin;
	bool is_nIter_lt_maxIter;
	bool is_perturb_gt_genotype;
	
	if (get_local_id(0) == 0) {
		//is_gNorm_gt_gMin    = (local_gNorm[0] >= gradMin_tol);
		is_nIter_lt_maxIter = (local_nIter[0] < gradMin_maxiter);
		//is_perturb_gt_genotype = true;


	}	

	// Verifying that Shoemake genes do not get out of valid range.
	// If they do so, then set them to 0.0f
	if (get_local_id(0) < 3){
		if ((local_genotype[get_local_id(0)] < 0.0f) && (local_genotype[get_local_id(0)] > 1.0f)) {
			local_genotype[get_local_id(0)] = 0.0f;
		}
	}
	

	#if 0
	// For every gene, let's determine 
	// if perturbation is greater than min conformation
  	for(uint i = get_local_id(0); 
		 i < gradMin_numElements; 
		 i+= NUM_OF_THREADS_PER_BLOCK) {
   		is_perturb_gt_gene_min[i] = (local_perturbation[i] >= gradMin_conformation_min_perturbation[i]);
  	}
	#endif

	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_local_id(0) == 0) {
		
	#if 0
		// Reduce all is_perturb_gt_gene_min's 
		// into their corresponding genotype
	  	for(uint i = 0; 
			 i < gradMin_numElements;
			 i++) {
	    		is_perturb_gt_genotype = is_perturb_gt_genotype && is_perturb_gt_gene_min[i];
			
			#if defined (DEBUG_GRADDESC_ENABLED)
			printf("is_perturb_gt_gene_min[%u]?: %s\n", i, (is_perturb_gt_gene_min[i] == true)?"yes":"no");
			#endif

	  	}
	#endif

		// Reduce all three previous 
		// partial evaluations (gNorm, nIter, perturb) into a final one
		//*is_gradDescentEn = is_gNorm_gt_gMin && is_nIter_lt_maxIter && is_perturb_gt_genotype;

		*is_gradDescentEn = is_nIter_lt_maxIter;

	#if 1
		#if defined (DEBUG_GRADDESC_ENABLED)
		if (get_local_id(0) == 0) {
			//printf("is_gNorm_gt_gMin?: %s\n", (is_gNorm_gt_gMin == true)?"yes":"no");
			//printf("is_nIter_lt_maxIter?: %s\n", (is_nIter_lt_maxIter == true)?"yes":"no");
			//printf("is_perturb_gt_genotype?: %s\n", (is_perturb_gt_genotype == true)?"yes":"no");
			printf("Continue gradient iteration?: %s\n", (*is_gradDescentEn == true)?"yes":"no");
		}
		#endif
	#endif

	}
}

void gradient_norm(
		   __local float*	vector1,
                           uint 	inputSize,
                   __local float*       init,
		   __local float*       inner_product
) {

	float temp = 0.0f;

	// Element-wise multiplication
	for(uint i = get_local_id(0); 
		 i < inputSize; 
		 i+= NUM_OF_THREADS_PER_BLOCK) {
		init[i] = vector1[i] * vector1[i];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// Accumulating dot product,
	// and then getting the norm
	if(get_local_id(0) == 0) {
		for(uint i = 0; 
			 i < inputSize; 
			 i ++) {
			temp += init[i];
		}

		*inner_product = native_sqrt(temp);
	}
}

