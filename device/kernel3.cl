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


// if defined, new (experimental) SW genotype moves that are dependent
// on nr of atoms and nr of torsions of ligand are used
#define SWAT3 // Third set of Solis-Wets hyperparameters by Andreas Tillack

__kernel void __attribute__ ((reqd_work_group_size(NUM_OF_THREADS_PER_BLOCK,1,1)))
perform_LS(
                 int    dockpars_num_of_atoms,
                 int    dockpars_true_ligand_atoms,
                 int    dockpars_num_of_atypes,
                 int    dockpars_num_of_map_atypes,
                 int    dockpars_num_of_intraE_contributors,
                 int    dockpars_gridsize_x,
                 int    dockpars_gridsize_y,
                 int    dockpars_gridsize_z,
                                                             // g1 = gridsize_x
                 uint   dockpars_gridsize_x_times_y,         // g2 = gridsize_x * gridsize_y
                 uint   dockpars_gridsize_x_times_y_times_z, // g3 = gridsize_x * gridsize_y * gridsize_z
                 float  dockpars_grid_spacing,
  __global const float* restrict dockpars_fgrids, // This is too large to be allocated in __constant
                 int    dockpars_rotbondlist_length,
                 float  dockpars_coeff_elec,
                 float  dockpars_elec_min_distance,
                 float  dockpars_coeff_desolv,
  __global       float* restrict dockpars_conformations_next,
  __global       float* restrict dockpars_energies_next,
  __global       int*   restrict dockpars_evals_of_new_entities,
  __global       uint*  restrict dockpars_prng_states,
                 int    dockpars_pop_size,
                 int    dockpars_num_of_genes,
                 float  dockpars_lsearch_rate,
                 uint   dockpars_num_of_lsentities,
                 float  dockpars_rho_lower_bound,
                 float  dockpars_base_dmov_mul_sqrt3,
                 float  dockpars_base_dang_mul_sqrt3,
                 uint   dockpars_cons_limit,
                 uint   dockpars_max_num_of_iters,
                 float  dockpars_qasp,
                 float  dockpars_smooth,

__constant       kernelconstant_interintra*   kerconst_interintra,
  __global const kernelconstant_intracontrib* kerconst_intracontrib,
__constant       kernelconstant_intra*        kerconst_intra,
__constant       kernelconstant_rotlist*      kerconst_rotlist,
__constant       kernelconstant_conform*      kerconst_conform
          )
// The GPU global function performs local search on the pre-defined entities of conformations_next.
// The number of blocks which should be started equals to num_of_lsentities*num_of_runs.
// This way the first num_of_lsentities entity of each population will be subjected to local search
// (and each block carries out the algorithm for one entity).
// Since the first entity is always the best one in the current population,
// it is always tested according to the ls probability, and if it not to be
// subjected to local search, the entity with ID num_of_lsentities is selected instead of the first one (with ID 0).
{
	// Some OpenCL compilers don't allow declaring
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel,
	// and then passed to non-kernel functions.
	__local float genotype_candidate[ACTUAL_GENOTYPE_LENGTH];
	__local float genotype_deviate  [ACTUAL_GENOTYPE_LENGTH];
	__local float genotype_bias     [ACTUAL_GENOTYPE_LENGTH];
        __local float rho;
	__local uint  cons_succ;
	__local uint  cons_fail;
	__local uint  iteration_cnt;
	__local float candidate_energy;
	__local int   evaluation_cnt;
	int gene_counter;

	__local float offspring_genotype[ACTUAL_GENOTYPE_LENGTH];
	__local int run_id;
	__local int entity_id;
	__local float offspring_energy;

	__local float4 calc_coords[MAX_NUM_OF_ATOMS];
	__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];

	#if defined (DEBUG_ENERGY_KERNEL)
	__local float partial_interE [NUM_OF_THREADS_PER_BLOCK];
	__local float partial_intraE [NUM_OF_THREADS_PER_BLOCK];
	#endif

	int tidx = get_local_id(0);
	// Determining run ID and entity ID
	// Initializing offspring genotype
	if (tidx == 0)
	{
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

		offspring_energy = dockpars_energies_next[run_id*dockpars_pop_size+entity_id];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	event_t ev = async_work_group_copy(offspring_genotype,
	                                   dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
	                                   dockpars_num_of_genes, 0);

	for (gene_counter = tidx;
	     gene_counter < dockpars_num_of_genes;
	     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
	{
		genotype_bias[gene_counter] = 0.0f;
	}

	if (tidx == 0) {
		rho = 1.0f;
		cons_succ = 0;
		cons_fail = 0;
		iteration_cnt = 0;
		evaluation_cnt = 0;
	}


	// Asynchronous copy should be finished by here
	wait_group_events(1, &ev);
	barrier(CLK_LOCAL_MEM_FENCE);

#ifdef SWAT3
	float lig_scale = 1.0f/sqrt((float)dockpars_num_of_atoms);
	float gene_scale = 1.0f/sqrt((float)dockpars_num_of_genes);
#endif
	while ((iteration_cnt < dockpars_max_num_of_iters) && (rho > dockpars_rho_lower_bound))
	{
		// New random deviate
		for (gene_counter = tidx;
		     gene_counter < dockpars_num_of_genes;
		     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
		{
#ifdef SWAT3
			genotype_deviate[gene_counter] = rho * (2.0f*gpu_randf(dockpars_prng_states)-1.0f) * (gpu_randf(dockpars_prng_states) < gene_scale);

			// Translation genes
			if (gene_counter < 3) {
				genotype_deviate[gene_counter] *= dockpars_base_dmov_mul_sqrt3;
			}
			// Orientation and torsion genes
			else {
				if (gene_counter < 6) {
					genotype_deviate[gene_counter] *= dockpars_base_dang_mul_sqrt3 * lig_scale;
				} else {
					genotype_deviate[gene_counter] *= dockpars_base_dang_mul_sqrt3 * gene_scale;
				}
			}
#else
			genotype_deviate[gene_counter] = rho*(2.0f*gpu_randf(dockpars_prng_states)-1.0f) * (gpu_randf(dockpars_prng_states) < 0.3f);

			// Translation genes
			if (gene_counter < 3) {
				genotype_deviate[gene_counter] *= dockpars_base_dmov_mul_sqrt3;
			}
			// Orientation and torsion genes
			else {
				genotype_deviate[gene_counter] *= dockpars_base_dang_mul_sqrt3;
			}
#endif
		}

		// Generating new genotype candidate
		for (gene_counter = tidx;
		     gene_counter < dockpars_num_of_genes;
		     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
		{
			genotype_candidate[gene_counter] = offspring_genotype[gene_counter] +
			                                   genotype_deviate[gene_counter]   +
			                                   genotype_bias[gene_counter];
		}

		// Evaluating candidate
		barrier(CLK_LOCAL_MEM_FENCE);

		// ==================================================================
		gpu_calc_energy( dockpars_rotbondlist_length,
		                 dockpars_num_of_atoms,
		                 dockpars_true_ligand_atoms,
		                 dockpars_gridsize_x,
		                 dockpars_gridsize_y,
		                 dockpars_gridsize_z,
		                                                     // g1 = gridsize_x
		                 dockpars_gridsize_x_times_y,         // g2 = gridsize_x * gridsize_y
		                 dockpars_gridsize_x_times_y_times_z, // g3 = gridsize_x * gridsize_y * gridsize_z
		                 dockpars_fgrids,
		                 dockpars_num_of_atypes,
		                 dockpars_num_of_map_atypes,
		                 dockpars_num_of_intraE_contributors,
		                 dockpars_grid_spacing,
		                 dockpars_coeff_elec,
		                 dockpars_elec_min_distance,
		                 dockpars_qasp,
		                 dockpars_coeff_desolv,
		                 dockpars_smooth,
		                 genotype_candidate,
		                 &candidate_energy,
		                 &run_id,
		                 // Some OpenCL compilers don't allow declaring
		                 // local variables within non-kernel functions.
		                 // These local variables must be declared in a kernel,
		                 // and then passed to non-kernel functions.
		                 calc_coords,
		                 partial_energies,
		                 #if defined (DEBUG_ENERGY_KERNEL)
		                 partial_interE,
		                 partial_intraE,
		                 #endif
#if 0
		                 false,
#endif
		                 kerconst_interintra,
		                 kerconst_intracontrib,
		                 kerconst_intra,
		                 kerconst_rotlist,
		                 kerconst_conform
		              );
		// =================================================================

		if (tidx == 0) {
			evaluation_cnt++;
			#if defined (DEBUG_ENERGY_KERNEL)
			printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL3-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
			#endif
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (candidate_energy < offspring_energy) // If candidate is better, success
		{
			for (gene_counter = tidx;
			     gene_counter < dockpars_num_of_genes;
			     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
			{
				// Updating offspring_genotype
				offspring_genotype[gene_counter] = genotype_candidate[gene_counter];

				// Updating genotype_bias
				genotype_bias[gene_counter] = 0.6f*genotype_bias[gene_counter] + 0.4f*genotype_deviate[gene_counter];
			}

			// Work-item 0 will overwrite the shared variables
			// used in the previous if condition
			barrier(CLK_LOCAL_MEM_FENCE);

			if (tidx == 0)
			{
				offspring_energy = candidate_energy;
				cons_succ++;
				cons_fail = 0;
			}
		}
		else // If candidate is worse, check the opposite direction
		{
			// Generating the other genotype candidate
			for (gene_counter = tidx;
			     gene_counter < dockpars_num_of_genes;
			     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
			{
				genotype_candidate[gene_counter] = offspring_genotype[gene_counter] -
				                                   genotype_deviate[gene_counter] -
				                                   genotype_bias[gene_counter];
			}

			// Evaluating candidate
			barrier(CLK_LOCAL_MEM_FENCE);

			// =================================================================
			gpu_calc_energy( dockpars_rotbondlist_length,
			                 dockpars_num_of_atoms,
			                 dockpars_true_ligand_atoms,
			                 dockpars_gridsize_x,
			                 dockpars_gridsize_y,
			                 dockpars_gridsize_z,
			                                                      // g1 = gridsize_x
			                 dockpars_gridsize_x_times_y,         // g2 = gridsize_x * gridsize_y
			                 dockpars_gridsize_x_times_y_times_z, // g3 = gridsize_x * gridsize_y * gridsize_z
			                 dockpars_fgrids,
			                 dockpars_num_of_atypes,
			                 dockpars_num_of_map_atypes,
			                 dockpars_num_of_intraE_contributors,
			                 dockpars_grid_spacing,
			                 dockpars_coeff_elec,
			                 dockpars_elec_min_distance,
			                 dockpars_qasp,
			                 dockpars_coeff_desolv,
			                 dockpars_smooth,
			                 genotype_candidate,
			                 &candidate_energy,
			                 &run_id,
			                 // Some OpenCL compilers don't allow declaring
			                 // local variables within non-kernel functions.
			                 // These local variables must be declared in a kernel,
			                 // and then passed to non-kernel functions.
			                 calc_coords,
			                 partial_energies,
			                 #if defined (DEBUG_ENERGY_KERNEL)
			                 partial_interE,
			                 partial_intraE,
			                 #endif
#if 0
			                 false,
#endif
			                 kerconst_interintra,
			                 kerconst_intracontrib,
			                 kerconst_intra,
			                 kerconst_rotlist,
			                 kerconst_conform
			               );
			// =================================================================

			if (tidx == 0) {
				evaluation_cnt++;

				#if defined (DEBUG_ENERGY_KERNEL)
				printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL3-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
				#endif
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			if (candidate_energy < offspring_energy) // If candidate is better, success
			{
				for (gene_counter = tidx;
				     gene_counter < dockpars_num_of_genes;
				     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
				{
					// Updating offspring_genotype
					offspring_genotype[gene_counter] = genotype_candidate[gene_counter];

					// Updating genotype_bias
					genotype_bias[gene_counter] = 0.6f*genotype_bias[gene_counter] - 0.4f*genotype_deviate[gene_counter];
				}

				// Work-item 0 will overwrite the shared variables
				// used in the previous if condition
				barrier(CLK_LOCAL_MEM_FENCE);

				if (tidx == 0)
				{
					offspring_energy = candidate_energy;
					cons_succ++;
					cons_fail = 0;
				}
			}
			else // Failure in both directions
			{
				for (gene_counter = tidx;
				     gene_counter < dockpars_num_of_genes;
				     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
					// Updating genotype_bias
					genotype_bias[gene_counter] = 0.5f*genotype_bias[gene_counter];

				if (tidx == 0)
				{
					cons_succ = 0;
					cons_fail++;
				}
			}
		}

		// Changing rho if needed
		if (tidx == 0)
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

	// Updating eval counter and energy
	if (tidx == 0) {
		dockpars_evals_of_new_entities[run_id*dockpars_pop_size+entity_id] += evaluation_cnt;
		dockpars_energies_next[run_id*dockpars_pop_size+entity_id] = offspring_energy;
	}

	// Mapping torsion angles
	for (gene_counter = tidx+3;
	     gene_counter < dockpars_num_of_genes;
	     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
	{
		map_angle(&(offspring_genotype[gene_counter]));
	}

	// Updating old offspring in population
	barrier(CLK_LOCAL_MEM_FENCE);
	event_t ev2 = async_work_group_copy(dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
	                                    offspring_genotype,
	                                    dockpars_num_of_genes,0);

	// Asynchronous copy should be finished by here
	wait_group_events(1, &ev2);
}
