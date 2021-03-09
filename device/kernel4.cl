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


// if defined, new (experimental) GA genotype mutation, similar to experimental SW move,
// dependent on nr of atoms on torsions of ligand is used, not ready yet ...
// #define GA_MUTATION_TEST

//#define DEBUG_ENERGY_KERNEL4

__kernel void __attribute__ ((reqd_work_group_size(NUM_OF_THREADS_PER_BLOCK,1,1)))
gpu_gen_and_eval_newpops(
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
                __global const float* restrict  dockpars_conformations_current,
                __global       float* restrict  dockpars_energies_current,
                __global       float* restrict  dockpars_conformations_next,
                __global       float* restrict  dockpars_energies_next,
                __global       int*   restrict  dockpars_evals_of_new_entities,
                __global       uint*  restrict dockpars_prng_states,
                               int    dockpars_pop_size,
                               int    dockpars_num_of_genes,
                               float  dockpars_tournament_rate,
                               float  dockpars_crossover_rate,
                               float  dockpars_mutation_rate,
                               float  dockpars_abs_max_dmov,
                               float  dockpars_abs_max_dang,
                               float  dockpars_qasp,
                               float  dockpars_smooth,

              __constant       kernelconstant_interintra*   kerconst_interintra,
                __global const kernelconstant_intracontrib* kerconst_intracontrib,
              __constant       kernelconstant_intra*        kerconst_intra,
              __constant       kernelconstant_rotlist*      kerconst_rotlist,
              __constant       kernelconstant_conform*      kerconst_conform
                        )
// The GPU global function
{
	// Some OpenCL compilers don't allow declaring 
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel, 
	// and then passed to non-kernel functions.
	__local float offspring_genotype[ACTUAL_GENOTYPE_LENGTH];
	__local int parent_candidates[4];
	__local float candidate_energies[4];
	__local int parents[2];
	__local int run_id;
	__local int covr_point[2];
	__local float randnums[10];
	int temp_covr_point;
	int gene_counter;
	__local float energy; // could be shared since only thread 0 will use it

	__local float best_energies[NUM_OF_THREADS_PER_BLOCK];
	__local int best_IDs[NUM_OF_THREADS_PER_BLOCK];
        __local int best_ID[1]; //__local int best_ID;

	__local float4 calc_coords[MAX_NUM_OF_ATOMS];
	__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];
	#if defined (DEBUG_ENERGY_KERNEL)
	__local float partial_interE [NUM_OF_THREADS_PER_BLOCK];
	__local float partial_intraE [NUM_OF_THREADS_PER_BLOCK];
	#endif

	int tidx = get_local_id(0);
	// In this case this compute-unit is responsible for elitist selection
	if ((get_group_id(0) % dockpars_pop_size) == 0) {
		gpu_perform_elitist_selection(dockpars_pop_size,
		                              dockpars_energies_current,
		                              dockpars_energies_next,
		                              dockpars_evals_of_new_entities,
		                              dockpars_num_of_genes,
		                              dockpars_conformations_next,
		                              dockpars_conformations_current,
		                              best_energies,
		                              best_IDs,
		                              best_ID);
	}
	else
	{
		// Generating the following random numbers:
		// [0..3] for parent candidates,
		// [4..5] for binary tournaments, [6] for deciding crossover,
		// [7..8] for crossover points, [9] for local search
		for (gene_counter = tidx;
		     gene_counter < 10;
		     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
		{
			randnums[gene_counter] = gpu_randf(dockpars_prng_states);
		}

		// Determining run ID
		if (tidx == 0) {
			run_id = get_group_id(0) / dockpars_pop_size;
		}

		// Performing binary tournament selection
		barrier(CLK_LOCAL_MEM_FENCE);

		for (gene_counter = tidx;
		     gene_counter < 4;
		     gene_counter+= NUM_OF_THREADS_PER_BLOCK)\
		{ //it is not ensured that the four candidates will be different...
			parent_candidates[gene_counter]  = (int) (dockpars_pop_size*randnums[gene_counter]); //using randnums[0..3]
			candidate_energies[gene_counter] = dockpars_energies_current[run_id*dockpars_pop_size+parent_candidates[gene_counter]];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for (gene_counter = tidx;
		     gene_counter < 2;
		     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
		{
			// Notice: dockpars_tournament_rate was scaled down to [0,1] in host
			// to reduce number of operations in device
			if (candidate_energies[2*gene_counter] < candidate_energies[2*gene_counter+1])
				if (/*100.0f**/randnums[4+gene_counter] < dockpars_tournament_rate) { //using randnum[4..5]
					parents[gene_counter] = parent_candidates[2*gene_counter];
				}
				else {
					parents[gene_counter] = parent_candidates[2*gene_counter+1];
				}
			else
				if (/*100.0f**/randnums[4+gene_counter] < dockpars_tournament_rate) {
					parents[gene_counter] = parent_candidates[2*gene_counter+1];
				}
				else {
					parents[gene_counter] = parent_candidates[2*gene_counter];
				}
		}

		// Performing crossover
		barrier(CLK_LOCAL_MEM_FENCE);

		// Notice: dockpars_crossover_rate was scaled down to [0,1] in host
		// to reduce number of operations in device
		if (/*100.0f**/randnums[6] < dockpars_crossover_rate) // Using randnums[6]
		{
			for (gene_counter = tidx;
			     gene_counter < 2;
			     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
			{
				// Using randnum[7..8]
				covr_point[gene_counter] = (int) ((dockpars_num_of_genes-1)*randnums[7+gene_counter]);
			}

			barrier(CLK_LOCAL_MEM_FENCE);
			
			// covr_point[0] should store the lower crossover-point
			if (tidx == 0) {
				if (covr_point[1] < covr_point[0]) {
					temp_covr_point = covr_point[1];
					covr_point[1]   = covr_point[0];
					covr_point[0]   = temp_covr_point;
				}
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			for (gene_counter = tidx;
			     gene_counter < dockpars_num_of_genes;
			     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
			{
				// Two-point crossover
				if (covr_point[0] != covr_point[1]) 
				{
					if ((gene_counter <= covr_point[0]) || (gene_counter > covr_point[1]))
						offspring_genotype[gene_counter] = dockpars_conformations_current[(run_id*dockpars_pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
					else
						offspring_genotype[gene_counter] = dockpars_conformations_current[(run_id*dockpars_pop_size+parents[1])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
				}
				// Single-point crossover
				else
				{
					if (gene_counter <= covr_point[0])
						offspring_genotype[gene_counter] = dockpars_conformations_current[(run_id*dockpars_pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
					else
						offspring_genotype[gene_counter] = dockpars_conformations_current[(run_id*dockpars_pop_size+parents[1])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
				}
			}

		}
		else // no crossover
		{
			event_t ev = async_work_group_copy(offspring_genotype,
			                                   dockpars_conformations_current+(run_id*dockpars_pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM,
			                                   dockpars_num_of_genes, 0);
			// Asynchronous copy should be finished by here
			wait_group_events(1, &ev);
		} // End of crossover

		barrier(CLK_LOCAL_MEM_FENCE);

		// Performing mutation
#ifdef GA_MUTATION_TEST
		float rot_scale = native_sqrt(native_divide((float)dockpars_num_of_genes,(float)dockpars_num_of_atoms));
#endif
		for (gene_counter = tidx;
		     gene_counter < dockpars_num_of_genes;
		     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
		{
			// Notice: dockpars_mutation_rate was scaled down to [0,1] in host
			// to reduce number of operations in device
			if (/*100.0f**/gpu_randf(dockpars_prng_states) < dockpars_mutation_rate)
			{
#ifdef GA_MUTATION_TEST
				float pmone = (2.0f*gpu_randf(dockpars_prng_states)-1.0f);

				// Translation genes
				if (gene_counter < 3) {
					offspring_genotype[gene_counter] += pmone * dockpars_abs_max_dmov;
				}
				// Orientation and torsion genes
				else {
					if (gene_counter < 6) {
						offspring_genotype[gene_counter] += pmone * dockpars_abs_max_dang * rot_scale;
					} else {
						offspring_genotype[gene_counter] += pmone * dockpars_abs_max_dang;
					}
					map_angle(&(offspring_genotype[gene_counter]));
				}
#else
				// Translation genes
				if (gene_counter < 3) {
					offspring_genotype[gene_counter] += dockpars_abs_max_dmov*(2*gpu_randf(dockpars_prng_states)-1);
				}
				// Orientation and torsion genes
				else {
					offspring_genotype[gene_counter] += dockpars_abs_max_dang*(2*gpu_randf(dockpars_prng_states)-1);
					map_angle(&(offspring_genotype[gene_counter]));
				}
#endif
			}
		} // End of mutation

		// Calculating energy of new offspring
		barrier(CLK_LOCAL_MEM_FENCE);

		// =============================================================
		gpu_calc_energy(dockpars_rotbondlist_length,
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
		                offspring_genotype,
		                &energy,
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
		// =============================================================

		if (tidx == 0) {
			dockpars_evals_of_new_entities[get_group_id(0)] = 1;
			dockpars_energies_next[get_group_id(0)] = energy;

			#if defined (DEBUG_ENERGY_KERNEL4)
			printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL4-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
			#endif
		}

		// Copying new offspring to next generation
		event_t ev2 = async_work_group_copy(dockpars_conformations_next + GENOTYPE_LENGTH_IN_GLOBMEM*get_group_id(0),
		                                    offspring_genotype,
		                                    dockpars_num_of_genes, 0);

		// Asynchronous copy should be finished by here
		wait_group_events(1, &ev2);
  }
}
