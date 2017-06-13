__kernel void __attribute__ ((reqd_work_group_size(NUM_OF_THREADS_PER_BLOCK,1,1)))
perform_LS( 		 char   dockpars_num_of_atoms,
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
	  __global int*   restrict dockpars_evals_of_new_entities,
	  __global unsigned int* restrict dockpars_prng_states,
		#else
	 	    __global float* dockpars_conformations_next,
        __global float* dockpars_energies_next,
	      __global int*   dockpars_evals_of_new_entities,
		    __global unsigned int* dockpars_prng_states,
		#endif

		       	    int    dockpars_pop_size,
			          int    dockpars_num_of_genes,
			          float  dockpars_lsearch_rate,
			          unsigned int dockpars_num_of_lsentities,
			          float  dockpars_rho_lower_bound,
			          float  dockpars_base_dmov_mul_sqrt3,
			          float  dockpars_base_dang_mul_sqrt3,
			          unsigned int dockpars_cons_limit,
			          unsigned int dockpars_max_num_of_iters,
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
          __constant float* ref_orientation_quats_const
)
//The GPU global function performs local search on the pre-defined entities of conformations_next.
//The number of blocks which should be started equals to num_of_lsentities*num_of_runs.
//This way the first num_of_lsentities entity of each population will be subjected to local search
//(and each block carries out the algorithm for one entity).
//Since the first entity is always the best one in the current population,
//it is always tested according to the ls probability, and if it not to be
//subjected to local search, the entity with ID num_of_lsentities is selected instead of the first one (with ID 0).
{
	__local float genotype_candidate[ACTUAL_GENOTYPE_LENGTH];
	__local float genotype_deviate  [ACTUAL_GENOTYPE_LENGTH];
	__local float genotype_bias     [ACTUAL_GENOTYPE_LENGTH];
  __local float rho;
	__local int   cons_succ;
	__local int   cons_fail;
	__local int   iteration_cnt;
	__local float candidate_energy;
	__local int   evaluation_cnt;
	int gene_counter;

	__local float offspring_genotype[ACTUAL_GENOTYPE_LENGTH];
	__local int run_id;
	__local int entity_id;
	__local float offspring_energy;

	//determining run ID and entity ID, initializing
	if (get_local_id(0) == 0)
	{
		run_id = get_group_id(0) / dockpars_num_of_lsentities;
		entity_id = get_group_id(0) % dockpars_num_of_lsentities;

		//since entity 0 is the best one due to elitism, should be subjected to random selection
		if (entity_id == 0)
			if (100.0f*gpu_randf(dockpars_prng_states) > dockpars_lsearch_rate)
				entity_id = dockpars_num_of_lsentities;	//if entity 0 is not selected according to LS rate,
									                              //choosing an other entity

		offspring_energy = dockpars_energies_next[run_id*dockpars_pop_size+entity_id];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

#if defined (ASYNC_COPY)
  async_work_group_copy(offspring_genotype,
			                  dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
                        dockpars_num_of_genes,0);
#else
	for (gene_counter=get_local_id(0);
	     gene_counter<dockpars_num_of_genes;
	     gene_counter+=NUM_OF_THREADS_PER_BLOCK)
		   offspring_genotype[gene_counter] = dockpars_conformations_next[(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
#endif

	for (gene_counter=get_local_id(0);
	     gene_counter<dockpars_num_of_genes;
	     gene_counter += NUM_OF_THREADS_PER_BLOCK)
		   genotype_bias[gene_counter] = 0.0f;

	if (get_local_id(0) == 0) {
		rho = 1.0f;
		cons_succ = 0;
		cons_fail = 0;
		iteration_cnt = 0;
		evaluation_cnt = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	while ((iteration_cnt < dockpars_max_num_of_iters) && (rho > dockpars_rho_lower_bound))
	{
		//new random deviate
		for (gene_counter=get_local_id(0);
		     gene_counter<dockpars_num_of_genes;
		     gene_counter += NUM_OF_THREADS_PER_BLOCK)
		{
			genotype_deviate[gene_counter] = rho*(2*gpu_randf(dockpars_prng_states)-1);

			if (gene_counter < 3)
				genotype_deviate[gene_counter] *= dockpars_base_dmov_mul_sqrt3;
			else
				genotype_deviate[gene_counter] *= dockpars_base_dang_mul_sqrt3;
		}

		//generating new genotype candidate
		for (gene_counter=get_local_id(0);
		     gene_counter<dockpars_num_of_genes;
		     gene_counter += NUM_OF_THREADS_PER_BLOCK)
			   genotype_candidate[gene_counter] = offspring_genotype[gene_counter] + genotype_deviate[gene_counter] + genotype_bias[gene_counter];

		//evaluating candidate
		barrier(CLK_LOCAL_MEM_FENCE);

		// ==================================================================
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
				            genotype_candidate,
				            &candidate_energy,
				            &run_id,

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
				            ref_orientation_quats_const);
		// =================================================================

		if (get_local_id(0) == 0)
			evaluation_cnt++;

		barrier(CLK_LOCAL_MEM_FENCE);

		if (candidate_energy < offspring_energy)	//if candidate is better, success
		{
			for (gene_counter=get_local_id(0);
			     gene_counter<dockpars_num_of_genes;
			     gene_counter += NUM_OF_THREADS_PER_BLOCK)
			{
				//updating offspring_genotype
				offspring_genotype[gene_counter] = genotype_candidate[gene_counter];

				//updating genotype_bias
				genotype_bias[gene_counter] = 0.6f*genotype_bias[gene_counter] + 0.4f*genotype_deviate[gene_counter];
			}

			//thread 0 will overwrite the shared variables
			//used in the previous if condition,
			//all threads have to be after if
			barrier(CLK_LOCAL_MEM_FENCE);

			if (get_local_id(0) == 0)
			{
				offspring_energy = candidate_energy;
				cons_succ++;
				cons_fail = 0;
			}
		}
		else	//if candidate is worser, check the opposite direction
		{
			//generating the other genotype candidate
			for (gene_counter=get_local_id(0);
			     gene_counter<dockpars_num_of_genes;
			     gene_counter += NUM_OF_THREADS_PER_BLOCK)
				   genotype_candidate[gene_counter] = offspring_genotype[gene_counter] - genotype_deviate[gene_counter] - genotype_bias[gene_counter];

			//evaluating candidate
			barrier(CLK_LOCAL_MEM_FENCE);

			// =================================================================
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
					            genotype_candidate,
					            &candidate_energy,
					            &run_id,

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
	               		  ref_orientation_quats_const);
			// =================================================================

			if (get_local_id(0) == 0)
				evaluation_cnt++;

			barrier(CLK_LOCAL_MEM_FENCE);

			if (candidate_energy < offspring_energy)//if candidate is better, success
			{
				for (gene_counter=get_local_id(0);
				     gene_counter<dockpars_num_of_genes;
			       gene_counter += NUM_OF_THREADS_PER_BLOCK)
				{
					//updating offspring_genotype
					offspring_genotype[gene_counter] = genotype_candidate[gene_counter];

					//updating genotype_bias
					genotype_bias[gene_counter] = 0.6f*genotype_bias[gene_counter] - 0.4f*genotype_deviate[gene_counter];
				}

				//thread 0 will overwrite the shared variables
				//used in the previous if condition,
				//all threads have to be after if
				barrier(CLK_LOCAL_MEM_FENCE);

				if (get_local_id(0) == 0)
				{
					offspring_energy = candidate_energy;
					cons_succ++;
					cons_fail = 0;
				}
			}
			else	//failure in both directions
			{
				for (gene_counter=get_local_id(0);
				     gene_counter<dockpars_num_of_genes;
				     gene_counter += NUM_OF_THREADS_PER_BLOCK)
					   //updating genotype_bias
					   genotype_bias[gene_counter] = 0.5f*genotype_bias[gene_counter];

				if (get_local_id(0) == 0)
				{
					cons_succ = 0;
					cons_fail++;
				}
			}
		}

		//changing rho if needed
		if (get_local_id(0) == 0)
		{
			iteration_cnt++;

			if (cons_succ >= dockpars_cons_limit)
			{
				rho *= LS_EXP_FACTOR;
				cons_succ = 0;
			}
			else
				if (cons_fail >= dockpars_cons_limit)
				{
					rho *= LS_CONT_FACTOR;
					cons_fail = 0;
				}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//updating eval counter and energy
	if (get_local_id(0) == 0)
	{
		dockpars_evals_of_new_entities[run_id*dockpars_pop_size+entity_id] += evaluation_cnt;
		dockpars_energies_next[run_id*dockpars_pop_size+entity_id] = offspring_energy;
	}

	//mapping angles
	for (gene_counter=get_local_id(0);
	     gene_counter<dockpars_num_of_genes;
	     gene_counter+=NUM_OF_THREADS_PER_BLOCK)
		   if (gene_counter >=  3)
			    map_angle(&(offspring_genotype[gene_counter]));

	//updating old offspring in population
	barrier(CLK_LOCAL_MEM_FENCE);

#if defined (ASYNC_COPY)
  async_work_group_copy(dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
                        offspring_genotype,
                        dockpars_num_of_genes,0);
#else
	for (gene_counter=get_local_id(0);
	     gene_counter<dockpars_num_of_genes;
       gene_counter+=NUM_OF_THREADS_PER_BLOCK)
		   dockpars_conformations_next[(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter] = offspring_genotype[gene_counter];
#endif
}
