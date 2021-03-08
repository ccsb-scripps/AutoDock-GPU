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


// Gradient-based steepest descent minimizer
// Alternative to Solis-Wets

//#define DEBUG_ENERGY_KERNEL5
//#define PRINT_ENERGIES
//#define PRINT_GENES_AND_GRADS
//#define PRINT_ATOMIC_COORDS

// Enable DEBUG_MINIMIZER for a seeing a detailed SD evolution
// If only PRINT_MINIMIZER_ENERGY_EVOLUTION is enabled,
// then a only a simplified SD evolution will be shown
//#define DEBUG_MINIMIZER
//#define PRINT_MINIMIZER_ENERGY_EVOLUTION

// Enable this for debugging SD from a defined initial genotype
//#define DEBUG_INITIAL_2BRT

__kernel void __attribute__ ((reqd_work_group_size(NUM_OF_THREADS_PER_BLOCK,1,1)))
gradient_minSD(
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
                     uint   dockpars_max_num_of_iters,
                     float  dockpars_qasp,
                     float  dockpars_smooth,

    __constant       kernelconstant_interintra*   kerconst_interintra,
      __global const kernelconstant_intracontrib* kerconst_intracontrib,
    __constant       kernelconstant_intra*        kerconst_intra,
    __constant       kernelconstant_rotlist*      kerconst_rotlist,
    __constant       kernelconstant_conform*      kerconst_conform,

    __constant       int*   rotbonds_const,
      __global const int*   rotbonds_atoms_const,
    __constant       int*   num_rotating_atoms_per_rotbond_const,

      __global const float* angle_const,
    __constant       float* dependence_on_theta_const,
    __constant       float* dependence_on_rotangle_const
              )
// The GPU global function performs gradient-based minimization on (some) entities of conformations_next.
// The number of OpenCL compute units (CU) which should be started equals to num_of_minEntities*num_of_runs.
// This way the first num_of_lsentities entity of each population will be subjected to local search
// (and each CU carries out the algorithm for one entity).
// Since the first entity is always the best one in the current population,
// it is always tested according to the ls probability, and if it not to be
// subjected to local search, the entity with ID num_of_lsentities is selected instead of the first one (with ID 0).
{
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------

	// Determining entity, and its run, energy, and genotype
	__local int   entity_id;
	__local int   run_id;
	__local float energy;
	__local float genotype[ACTUAL_GENOTYPE_LENGTH];

	// Iteration counter fot the minimizer
	__local uint iteration_cnt;

	// Stepsize for the minimizer
	__local float stepsize;

	int tidx = get_local_id(0);
	if (tidx == 0)
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

		// Initializing gradient-minimizer counters and flags
		iteration_cnt  = 0;
		stepsize       = STEP_START;

		#if defined (DEBUG_MINIMIZER) || defined (PRINT_MINIMIZER_ENERGY_EVOLUTION)
		printf("\n");
		printf("-------> Start of SD minimization cycle\n");
		printf("%20s %6u\n", "run_id: ", run_id);
		printf("%20s %6u\n", "entity_id: ", entity_id);
		printf("\n");
		printf("%20s \n", "LGA genotype: ");
		printf("%20s %.6f\n", "initial energy: ", energy);
		printf("%20s %.6f\n", "initial stepsize: ", stepsize);
		#endif
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	event_t ev = async_work_group_copy(genotype,
	                                   dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
	                                   dockpars_num_of_genes, 0);
	// Asynchronous copy should be finished by here
	wait_group_events(1, &ev);

	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	
	// Partial results of the gradient step
	__local float gradient          [ACTUAL_GENOTYPE_LENGTH];
	__local float candidate_energy;
	__local float candidate_genotype[ACTUAL_GENOTYPE_LENGTH];

	// -------------------------------------------------------------------
	// Calculate gradients (forces) for intermolecular energy
	// Derived from autodockdev/maps.py
	// -------------------------------------------------------------------
	// Gradient of the intermolecular energy per each ligand atom
	// Also used to store the accummulated gradient per each ligand atom
	__local int   i_gradient_x[MAX_NUM_OF_ATOMS];
	__local int   i_gradient_y[MAX_NUM_OF_ATOMS];
	__local int   i_gradient_z[MAX_NUM_OF_ATOMS];

	__local float f_gradient_x[MAX_NUM_OF_ATOMS];
	__local float f_gradient_y[MAX_NUM_OF_ATOMS];
	__local float f_gradient_z[MAX_NUM_OF_ATOMS];

	// Ligand-atom position and partial energies
	__local float4 calc_coords[MAX_NUM_OF_ATOMS];
	__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];

	#if defined (DEBUG_ENERGY_KERNEL)
	__local float partial_interE[NUM_OF_THREADS_PER_BLOCK];
	__local float partial_intraE[NUM_OF_THREADS_PER_BLOCK];
	#endif

	// Enable this for debugging SD from a defined initial genotype
	#if defined (DEBUG_INITIAL_2BRT)
	if (tidx == 0) {
		// 2brt
		genotype[0] = 24.093334f;
		genotype[1] = 24.658667f;
		genotype[2] = 24.210667f;
		genotype[3] = 50.0f;
		genotype[4] = 50.0f;
		genotype[5] = 50.0f;
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
	}
	// Evaluating candidate
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
	
	                genotype, /*WARNING: calculating the energy of the hardcoded genotype*/
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
	                true,
#endif
	                kerconst_interintra,
	                kerconst_intracontrib,
	                kerconst_intra,
	                kerconst_rotlist,
	                kerconst_conform
	               );
	// =============================================================

	// WARNING: hardcoded has priority over LGA genotype.
	// That means, if DEBUG_INITIAL_2BRT is defined, then
	// LGA genotype is not used (only for debugging purposes)
	if (tidx == 0)
	{
		printf("\n");
		printf("%20s \n", "hardcoded genotype: ");
		printf("%20s %.6f\n", "initial energy: ", energy);
		printf("%20s %.6f\n\n", "initial stepsize: ", stepsize);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	#endif

	// Perform gradient-descent iterations

	#if 0
	// 7cpa
	float grid_center_x = 49.836f;
	float grid_center_y = 17.609f;
	float grid_center_z = 36.272f;
	float ligand_center_x = 49.2216976744186f;
	float ligand_center_y = 17.793953488372097f;
	float ligand_center_z = 36.503837209302326f;
	float shoemake_gene_u1 = 0.02f;
	float shoemake_gene_u2 = 0.23f;
	float shoemake_gene_u3 = 0.95f;
	#endif

	#if 0
	// 3tmn
	float grid_center_x = 52.340f;
	float grid_center_y = 15.029f;
	float grid_center_z = -2.932f;
	float ligand_center_x = 52.22740741f;
	float ligand_center_y = 15.51751852f;
	float ligand_center_z = -2.40896296f;
	#endif

	// Calculating maximum possible stepsize (alpha)
	__local float max_trans_grad, max_rota_grad, max_tors_grad;
	__local float max_trans_stepsize, max_rota_stepsize, max_tors_stepsize;
	__local float max_stepsize;

	// Storing torsion gradients here
	__local float torsions_gradient[ACTUAL_GENOTYPE_LENGTH];

	// The termination criteria is based on
	// a maximum number of iterations, and
	// the minimum step size allowed for single-floating point numbers
	// (IEEE-754 single float has a precision of about 6 decimal digits)
	do {
		#if 0
		// Specific input genotypes for a ligand with no rotatable bonds (1ac8).
		// Translation genes must be expressed in grids in AutoDock-GPU (genotype [0|1|2]).
		// However, for testing purposes, 
		// we start using translation values in real space (Angstrom): {31.79575, 93.743875, 47.699875}
		// Rotation genes are expresed in the Shoemake space: genotype [3|4|5]
		// xyz_gene_gridspace = gridcenter_gridspace + (input_gene_realspace - gridcenter_realspace)/gridsize

		// 1ac8
		genotype[0] = 30f + (31.79575f  - 31.924f) / dockpars_grid_spacing;
		genotype[1] = 30f + (93.743875f - 93.444f) / dockpars_grid_spacing;
		genotype[2] = 30f + (47.699875f - 47.924f) / dockpars_grid_spacing;
		genotype[3] = 0.1f;
		genotype[4] = 0.5f;
		genotype[5] = 0.9f;
		#endif

		#if 0
		// 3tmn
		genotype[0] = 30f + (ligand_center_x - grid_center_x) / dockpars_grid_spacing;
		genotype[1] = 30f + (ligand_center_y - grid_center_y) / dockpars_grid_spacing;
		genotype[2] = 30f + (ligand_center_z - grid_center_z) / dockpars_grid_spacing;
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

		#if 0
		// 2j5s
		genotype[0] = 28.464f;
		genotype[1] = 25.792762f;
		genotype[2] = 23.740571f;
		genotype[3] = 50.0f;
		genotype[4] = 50.0f;
		genotype[5] = 50.0f;
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

		#if 0
		// 2brt
		genotype[0] = 24.093334f;
		genotype[1] = 24.658667f;
		genotype[2] = 24.210667f;
		genotype[3] = 50.0f;
		genotype[4] = 50.0f;
		genotype[5] = 50.0f;
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

		// Printing number of stepest-descent iterations
		#if defined (DEBUG_MINIMIZER)
		if (tidx == 0) {
			printf("%s\n", "----------------------------------------------------------");
		}
		#endif
		
		#if defined (DEBUG_MINIMIZER) || defined (PRINT_MINIMIZER_ENERGY_EVOLUTION)
		if (tidx == 0) {
			printf("%-15s %-3u ", "# SD iteration: ", iteration_cnt);
		}
		#endif

		// Calculating gradient
		barrier(CLK_LOCAL_MEM_FENCE);

		// =============================================================
		gpu_calc_gradient(dockpars_rotbondlist_length,
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
		                  // Some OpenCL compilers don't allow declaring
		                  // local variables within non-kernel functions.
		                  // These local variables must be declared in a kernel,
		                  // and then passed to non-kernel functions.
		                  genotype,
		                  &energy,
		                  &run_id,
		                  calc_coords,
		                  kerconst_interintra,
		                  kerconst_intracontrib,
		                  kerconst_intra,
		                  kerconst_rotlist,
		                  kerconst_conform,
		                  rotbonds_const,
		                  rotbonds_atoms_const,
		                  num_rotating_atoms_per_rotbond_const,
		                  angle_const,
		                  dependence_on_theta_const,
		                  dependence_on_rotangle_const,
		                  // Gradient-related arguments
		                  dockpars_num_of_genes,
		                  (__local float*)i_gradient_x, (__local float*)i_gradient_y, (__local float*)i_gradient_z,
		                  f_gradient_x, f_gradient_y, f_gradient_z,
		                  gradient
		                 );
		// =============================================================

		// This could be enabled back for double checking
		#if 0
		#if defined (DEBUG_ENERGY_KERNEL5)
		if (/*(get_group_id(0) == 0) &&*/ (tidx == 0)) {
		
			#if defined (PRINT_GENES_AND_GRADS)
			for(int i = 0; i < dockpars_num_of_genes; i++) {
				if (i == 0) {
					printf("\n%s\n", "----------------------------------------------------------");
					printf("%13s %13s %5s %15s %15s\n", "gene_id", "gene.value", "|", "gene.grad", "(autodockdevpy units)");
				}
				printf("%13u %13.6f %5s %15.6f %15.6f\n", i, genotype[i], "|", gradient[i], (i<3)? (gradient[i]/dockpars_grid_spacing):(gradient[i]*180.0f/PI_FLOAT));
			}
			#endif

			#if defined (PRINT_ATOMIC_COORDS)
			for(int i = 0; i < dockpars_num_of_atoms; i++) {
				if (i == 0) {
					printf("\n%s\n", "----------------------------------------------------------");
					printf("%s\n", "Coordinates calculated by calcgradient.cl");
					printf("%12s %12s %12s %12s\n", "atom_id", "coords.x", "coords.y", "coords.z");
				}
				printf("%12u %12.6f %12.6f %12.6f\n", i, calc_coords_x[i], calc_coords_y[i], calc_coords_z[i]);
			}
			printf("\n");
			#endif
		}
		#endif
		#endif

		if (tidx == 0) {
			// Finding maximum of the absolute value for the three translation gradients
			max_trans_grad = fmax(fabs(gradient[0]), fabs(gradient[1]));
			max_trans_grad = fmax(max_trans_grad, fabs(gradient[2]));

			// MAX_DEV_TRANSLATION needs to be expressed in grid size first
			max_trans_stepsize = native_divide(native_divide(MAX_DEV_TRANSLATION, dockpars_grid_spacing), max_trans_grad);

			// Finding maximum of the absolute value for the three rotation gradients
			max_rota_grad = fmax(fabs(gradient[3]), fabs(gradient[4]));	
			max_rota_grad = fmax(max_rota_grad, fabs(gradient[5]));	

			// Note that MAX_DEV_ROTATION is already expressed approprietly
			max_rota_stepsize = native_divide(MAX_DEV_ROTATION, max_rota_grad);
		}

		// Copying torsions genes
		for( int i = tidx;
		         i < dockpars_num_of_genes-6;
		         i+= NUM_OF_THREADS_PER_BLOCK)
		{
			torsions_gradient[i] = fabs(gradient[i+6]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// Calculating maximum absolute torsional gene
		// https://stackoverflow.com/questions/36465581/opencl-find-max-in-array
		for (int i=(dockpars_num_of_genes-6)/2; i>=1; i/=2){
			if (tidx < i) {
			// This could be enabled back for details
			#if 0
			#if defined (DEBUG_MINIMIZER)
			printf("---====--- %u %u %10.10f %-0.10f\n", i, tidx, torsions_gradient[tidx], torsions_gradient[tidx + i]);
			#endif
			#endif

				if (torsions_gradient[tidx] < torsions_gradient[tidx + i]) {
					torsions_gradient[tidx] = torsions_gradient[tidx + i];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if (tidx == 0) {
			max_tors_grad = torsions_gradient[tidx];
			max_tors_stepsize = native_divide(MAX_DEV_TORSION, max_tors_grad);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if (tidx == 0) {
			// Calculating the maximum stepsize using previous three
			max_stepsize = fmin(max_trans_stepsize, max_rota_stepsize);
			max_stepsize = fmin(max_stepsize, max_tors_stepsize);

			// Capping the stepsize
			stepsize = fmin(stepsize, max_stepsize);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for(int i = tidx; i < dockpars_num_of_genes; i+= NUM_OF_THREADS_PER_BLOCK) {
			// Taking step
			candidate_genotype[i] = genotype[i] - stepsize * gradient[i];

			#if defined (DEBUG_MINIMIZER)
			if (i == 0) {
				printf("\n%s\n", "After calculating gradients:");
				printf("%13s %13s %5s %15s %5s %20s\n", "gene_id", "gene", "|", "grad", "|", "cand.gene");
			}
			printf("%13u %13.6f %5s %15.6f %5s %18.6f\n", i, genotype[i], "|", gradient[i], "|", candidate_genotype[i]);

			if (i == 0) {
				// This could be enabled back for double checking
				#if 0
				for(int i = 0; i < dockpars_num_of_genes; i++) {
					if (i == 0) {
						printf("\n%s\n", "----------------------------------------------------------");
						printf("\n%s\n", "After calculating gradients:");
						printf("%13s %13s %5s %15s %20s\n", "gene_id", "gene", "|", "grad", " grad (devpy units)");
					}
					printf("%13u %13.6f %5s %15.6f %18.6f\n", i, genotype[i], "|", gradient[i], (i<3)? (gradient[i]/dockpars_grid_spacing):(gradient[i]*180.0f/PI_FLOAT));
				}
				#endif

				printf("\n");
				printf("%20s %10.6f\n", "max_trans_grad: ", max_trans_grad);
				printf("%20s %10.6f\n", "max_rota_grad: ", max_rota_grad);
				printf("%20s %10.6f\n", "max_tors_grad: ", max_tors_grad);

				printf("\n");
				printf("%20s %10.6f\n", "max_trans_stepsize: ", max_trans_stepsize);
				printf("%20s %10.6f\n", "max_rota_stepsize: " , max_rota_stepsize);
				printf("%20s %10.6f\n", "max_tors_stepsize: " , max_tors_stepsize);

				printf("\n");
				printf("%20s %10.6f\n", "max_stepsize: ", max_stepsize);
				printf("%20s %10.6f\n", "stepsize: ", stepsize);
			}
			#endif
		}
		
		// Evaluating candidate
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
		
		                candidate_genotype, /*WARNING: calculating the energy of the hardcoded genotype*/
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
		                true,
#endif
		                kerconst_interintra,
		                kerconst_intracontrib,
		                kerconst_intra,
		                kerconst_rotlist,
		                kerconst_conform
		               );
		// =============================================================

		#if defined (DEBUG_ENERGY_KERNEL5)
		if (/*(get_group_id(0) == 0) &&*/ (tidx == 0)) {
			#if defined (PRINT_ENERGIES)
			printf("\n");
			printf("%-10s %-10.6f \n", "intra: ",  partial_intraE[0]);
			printf("%-10s %-10.6f \n", "grids: ",  partial_interE[0]);
			printf("%-10s %-10.6f \n", "Energy: ", (partial_intraE[0] + partial_interE[0]));
			#endif

			#if defined (PRINT_GENES_AND_GRADS)
			for(int i = 0; i < dockpars_num_of_genes; i++) {
				if (i == 0) {
					printf("\n%s\n", "----------------------------------------------------------");
					printf("%13s %13s %5s %15s %15s\n", "gene_id", "cand-gene.value"/* "gene.value"*/, "|", "gene.grad", "(autodockdevpy units)");
				}
				printf("%13u %13.6f %5s %15.6f %15.6f\n", i, candidate_genotype[i] /*genotype[i]*/, "|", gradient[i], (i<3)? (gradient[i]/dockpars_grid_spacing):(gradient[i]*180.0f/PI_FLOAT));
			}
			#endif

			#if defined (PRINT_ATOMIC_COORDS)
			for(int i = 0; i < dockpars_num_of_atoms; i++) {
				if (i == 0) {
					printf("\n%s\n", "----------------------------------------------------------");
					printf("%s\n", "Coordinates calculated by calcenergy.cl");
					printf("%12s %12s %12s %12s\n", "atom_id", "coords.x", "coords.y", "coords.z");
				}
				printf("%12u %12.6f %12.6f %12.6f\n", i, calc_coords_x[i], calc_coords_y[i], calc_coords_z[i]);
			}
			printf("\n");
			#endif
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		#endif

		#if defined (DEBUG_MINIMIZER)
		if (tidx == 0) {
			printf("\n");
			printf("%s\n", "After calculating energies:");
			printf("%13s %5s %15s\n", "energy", "|", "cand.energy");
			printf("%13.6f %5s %15.6f\n", energy, "|", candidate_energy);
			printf("\n");
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		#endif

		// Checking if E(candidate_genotype) < E(genotype)
		if (candidate_energy < energy){
			for( int i = tidx;
			         i < dockpars_num_of_genes;
			         i+= NUM_OF_THREADS_PER_BLOCK)
			{
				#if defined (DEBUG_MINIMIZER)
				//printf("(%-3u) %-15.7f %-10.7f %-10.7f %-10.7f\n", i, stepsize, genotype[i], gradient[i], candidate_genotype[i]);

				if (i == 0) {
					printf("%s\n", "Energy IMPROVED! ... then update genotype:");
					printf("%13s %13s %5s %15s\n", "gene_id", "old.gene", "|", "new.gene");
				}
				printf("%13u %13.6f %5s %15.6f\n", i, genotype[i], "|", candidate_genotype[i]);

				#endif
				if (i == 0) {
					#if defined (DEBUG_MINIMIZER)
					printf("\n%s\n", "Energy IMPROVED! ... then increase stepsize and update energy:");
					#endif

					// Increase stepsize
					stepsize *= STEP_INCREASE;

					// Updating energy
					energy = candidate_energy;
				}

				// Updating genotype
				genotype[i] = candidate_genotype[i];
			}
		}
		else {
			#if defined (DEBUG_MINIMIZER)
			if (tidx == 0) {
				printf("%s\n", "NO energy improvement! ... then decrease stepsize:");
			}
			#endif

			if (tidx == 0) {
				stepsize *= STEP_DECREASE;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// Updating number of stepest-descent iterations (energy evaluations)
		if (tidx == 0) {
			iteration_cnt = iteration_cnt + 1;

			#if defined (DEBUG_MINIMIZER) || defined (PRINT_MINIMIZER_ENERGY_EVOLUTION)
			printf("%20s %10.10f %20s %10.6f\n", "new.stepsize: ", stepsize, "new.energy: ", energy);
			#endif

			#if defined (DEBUG_ENERGY_KERNEL5)
			printf("%-18s [%-5s]---{%-5s}   [%-10.7f]---{%-10.7f}\n", "-ENERGY-KERNEL5-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
			#endif
		}
		barrier(CLK_LOCAL_MEM_FENCE);

	} while ((iteration_cnt < dockpars_max_num_of_iters) && (stepsize > 1E-8f));
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------

  	// Updating eval counter and energy
	if (tidx == 0) {
		dockpars_evals_of_new_entities[run_id*dockpars_pop_size+entity_id] += iteration_cnt;
		dockpars_energies_next[run_id*dockpars_pop_size+entity_id] = energy;

		#if defined (DEBUG_MINIMIZER) || defined (PRINT_MINIMIZER_ENERGY_EVOLUTION)
		printf("\n");
		printf("Termination criteria: ( stepsize <= %10.10f ) OR ( #sd-iters >= %-3u )\n", 1E-8, dockpars_max_num_of_iters);
		printf("-------> End of SD minimization cycle, num of energy evals: %u, final energy: %.6f\n", iteration_cnt, energy);
		#endif
	}

	// Mapping torsion angles
	for ( int gene_counter = tidx;
	          gene_counter < dockpars_num_of_genes;
	          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
	{
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
