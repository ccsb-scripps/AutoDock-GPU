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


#define ADADELTA_AUTOSTOP // Stopping criterion from Solis-Wets

// Gradient-based adadelta minimizer
// https://arxiv.org/pdf/1212.5701.pdf
// Alternative to Solis-Wets / Steepest-Descent / FIRE

// "rho": controls degree of memory of previous gradients
//        ranges between [0, 1[
//        "rho" = 0.9 most popular value
// "epsilon":  to better condition the square root

// Adadelta parameters (TODO: to be moved to header file?)
//#define RHO             0.9f
//#define EPSILON         1e-6f
#define RHO             0.8f
#define EPSILON         1e-2f

// Enabling "DEBUG_ENERGY_ADADELTA" requires
// manually enabling "DEBUG_ENERGY_KERNEL" in calcenergy.cl
//#define DEBUG_ENERGY_ADADELTA
//#define PRINT_ADADELTA_ENERGIES
//#define PRINT_ADADELTA_GENES_AND_GRADS
//#define PRINT_ADADELTA_ATOMIC_COORDS
//#define DEBUG_SQDELTA_ADADELTA

// Enable DEBUG_ADADELTA_MINIMIZER for a seeing a detailed ADADELTA evolution
// If only PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION is enabled,
// then a only a simplified ADADELTA evolution will be shown
//#define DEBUG_ADADELTA_MINIMIZER
//#define PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION

// Enable this for debugging ADADELTA from a defined initial genotype
//#define DEBUG_ADADELTA_INITIAL_2BRT

__kernel void __attribute__ ((reqd_work_group_size(NUM_OF_THREADS_PER_BLOCK,1,1)))
gradient_minAD(
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
	int tidx = get_local_id(0);

	// Determining entity, and its run, energy, and genotype
	__local int   entity_id;
	__local int   run_id;
	__local float energy;
	__local float genotype[ACTUAL_GENOTYPE_LENGTH];

	// Iteration counter fot the minimizer
	__local uint iteration_cnt;

	if (tidx == 0)
	{
		run_id = get_group_id(0) / dockpars_num_of_lsentities;
		entity_id = get_group_id(0) - run_id * dockpars_num_of_lsentities; // modulus in different form

		// Since entity 0 is the best one due to elitism,
		// it should be subjected to random selection
		if (entity_id == 0) {
			// If entity 0 is not selected according to LS-rate,
			// choosing another entity
			if (100.0f*gpu_randf(dockpars_prng_states) > dockpars_lsearch_rate) {
				entity_id = dockpars_num_of_lsentities; // AT - Should this be (uint)(dockpars_pop_size * gpu_randf(dockpars_prng_states))?
			}
		}
		
		energy = dockpars_energies_next[run_id*dockpars_pop_size+entity_id];

		// Initializing gradient-minimizer counters and flags
		iteration_cnt  = 0;

		#if defined (DEBUG_ADADELTA_MINIMIZER) || defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
		printf("\n");
		printf("-------> Start of ADADELTA minimization cycle\n");
		printf("%20s %6u\n", "run_id: ", run_id);
		printf("%20s %6u\n", "entity_id: ", entity_id);
		printf("\n");
		printf("%20s \n", "LGA genotype: ");
		printf("%20s %.6f\n", "initial energy: ", energy);
		#endif
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	event_t ev = async_work_group_copy(genotype,
	                                   dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
	                                   dockpars_num_of_genes, 0);

	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------

	// Partial results of the gradient step
	__local float gradient[ACTUAL_GENOTYPE_LENGTH];

	// Energy may go up, so we keep track of the best energy ever calculated.
	// Then, we return the genotype corresponding 
	// to the best observed energy, i.e. "best_genotype"
	__local float best_energy;
	__local float best_genotype[ACTUAL_GENOTYPE_LENGTH];

	// -------------------------------------------------------------------
	// Calculate gradients (forces) for intermolecular energy
	// Derived from autodockdev/maps.py
	// -------------------------------------------------------------------
	// Gradient of the intermolecular energy per each ligand atom
	// Also used to store the accummulated gradient per each ligand atom
#ifdef FLOAT_GRADIENTS
	__local float   gradient_x[MAX_NUM_OF_ATOMS];
	__local float   gradient_y[MAX_NUM_OF_ATOMS];
	__local float   gradient_z[MAX_NUM_OF_ATOMS];
#else
	__local int   gradient_x[MAX_NUM_OF_ATOMS];
	__local int   gradient_y[MAX_NUM_OF_ATOMS];
	__local int   gradient_z[MAX_NUM_OF_ATOMS];
#endif
	__local float accumulator_x[NUM_OF_THREADS_PER_BLOCK];
	__local float accumulator_y[NUM_OF_THREADS_PER_BLOCK];
	__local float accumulator_z[NUM_OF_THREADS_PER_BLOCK];

	// Ligand-atom position and partial energies
	__local float4 calc_coords[MAX_NUM_OF_ATOMS];
	__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];

	#if defined (DEBUG_ENERGY_KERNEL)
	__local float partial_interE[NUM_OF_THREADS_PER_BLOCK];
	__local float partial_intraE[NUM_OF_THREADS_PER_BLOCK];
	#endif

	// Vector for storing squared gradients E[g^2]
	__local float square_gradient[ACTUAL_GENOTYPE_LENGTH];

	// Update vector, i.e., "delta".
	// It is added to the genotype to create the next genotype.
	// E.g. in steepest descent "delta" is -1.0 * stepsize * gradient
	__local float delta[ACTUAL_GENOTYPE_LENGTH];

	// Squared updates E[dx^2]
	__local float square_delta[ACTUAL_GENOTYPE_LENGTH];

	// Asynchronous copy should be finished by here
	wait_group_events(1, &ev);

	// Enable this for debugging ADADELTA from a defined initial genotype
#if defined (DEBUG_ADADELTA_INITIAL_2BRT)
	if (tidx == 0) {
		// 2brt
		genotype[0]  = 24.093334f;
		genotype[1]  = 24.658667f;
		genotype[2]  = 24.210667f;
		genotype[3]  = 50.0f;
		genotype[4]  = 50.0f;
		genotype[5]  = 50.0f;
		genotype[6]  = 0.0f;
		genotype[7]  = 0.0f;
		genotype[8]  = 0.0f;
		genotype[9]  = 0.0f;
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
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#endif // DEBUG_ADADELTA_INITIAL_2BRT

	// Initializing vectors
	for( int i = tidx;
	         i < dockpars_num_of_genes;
	         i+= NUM_OF_THREADS_PER_BLOCK)
	{
		gradient[i]        = 0.0f;
		square_gradient[i] = 0.0f;
		delta[i]           = 0.0f;
		square_delta[i]    = 0.0f;
		best_genotype [i] = genotype [i];
	}

	// Initializing best energy
	if (tidx == 0) {
		best_energy = INFINITY;
	}

#ifdef ADADELTA_AUTOSTOP
	__local float rho;
	__local int   cons_succ;
	__local int   cons_fail;
	if (tidx == 0) {
		rho = 1.0f;
		cons_succ = 0;
		cons_fail = 0;
	}
#endif
	// Perform adadelta iterations

	// The termination criteria is based on
	// a maximum number of iterations, and
	// the minimum step size allowed for single-floating point numbers
	// (IEEE-754 single float has a precision of about 6 decimal digits)
	do {
		// Printing number of ADADELTA iterations
		#if defined (DEBUG_ADADELTA_MINIMIZER) || defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
		if (tidx == 0) {
			#if defined (DEBUG_ADADELTA_MINIMIZER)
			printf("%s\n", "----------------------------------------------------------");
			#endif
			printf("%-15s %-3u ", "# ADADELTA iteration: ", iteration_cnt);
		}
		#endif

		// =============================================================
		// =============================================================
		// =============================================================
		// Calculating energy & gradient
		barrier(CLK_LOCAL_MEM_FENCE);

		gpu_calc_energrad(
		                  dockpars_rotbondlist_length,
		                  dockpars_num_of_atoms,
		                  dockpars_true_ligand_atoms,
		                  dockpars_gridsize_x,
		                  dockpars_gridsize_y,
		                  dockpars_gridsize_z,
		                  // g1 = gridsize_x
		                  dockpars_gridsize_x_times_y, 		// g2 = gridsize_x * gridsize_y
		                  dockpars_gridsize_x_times_y_times_z,	// g3 = gridsize_x * gridsize_y * gridsize_z
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
		                  partial_energies,
		                  #if defined (DEBUG_ENERGY_KERNEL)
		                  partial_interE,
		                  partial_intraE,
		                  #endif
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
		                  gradient_x, gradient_y, gradient_z,
		                  accumulator_x, accumulator_y, accumulator_z,
		                  gradient
		                 );
		// =============================================================
		// =============================================================
		// =============================================================
		#if defined (DEBUG_ENERGY_ADADELTA)
		if (tidx == 0) {
			#if defined (PRINT_ADADELTA_ENERGIES)
			printf("\n");
			printf("%-10s %-10.6f \n", "intra: ",  partial_intraE[0]);
			printf("%-10s %-10.6f \n", "grids: ",  partial_interE[0]);
			printf("%-10s %-10.6f \n", "Energy: ", (partial_intraE[0] + partial_interE[0]));
			#endif

			#if defined (PRINT_ADADELTA_GENES_AND_GRADS)
			for(int i = 0; i < dockpars_num_of_genes; i++) {
				if (i == 0) {
					printf("\n%s\n", "----------------------------------------------------------");
					printf("%13s %13s %5s %15s %15s\n", "gene_id", "gene.value", "|", "gene.grad", "(autodockdevpy units)");
				}
				printf("%13u %13.6f %5s %15.6f %15.6f\n", i, genotype[i], "|", gradient[i], (i<3)? (gradient[i]/dockpars_grid_spacing):(gradient[i]*180.0f/PI_FLOAT));
			}
			#endif

			#if defined (PRINT_ADADELTA_ATOMIC_COORDS)
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
		#endif // DEBUG_ENERGY_ADADELTA

		for( int i = tidx;
		         i < dockpars_num_of_genes;
		         i+= NUM_OF_THREADS_PER_BLOCK)
		{
			if (energy < best_energy) // we need to be careful not to change best_energy until we had a chance to update the whole array
				best_genotype[i] = genotype[i];

			// Accummulating gradient^2 (eq.8 in the paper)
			// square_gradient corresponds to E[g^2]
			square_gradient[i] = RHO * square_gradient[i] + (1.0f - RHO) * gradient[i] * gradient[i];

			// Computing update (eq.9 in the paper)
			delta[i] = -1.0f * gradient[i] * native_sqrt(native_divide((float)(square_delta[i] + EPSILON), (float)(square_gradient[i] + EPSILON)));

			// Accummulating update^2
			// square_delta corresponds to E[dx^2]
			square_delta[i] = RHO * square_delta[i] + (1.0f - RHO) * delta[i] * delta [i];

			// Applying update
			genotype[i] += delta[i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		#if defined (DEBUG_SQDELTA_ADADELTA)
		if (/*(get_group_id(0) == 0) &&*/ (tidx == 0)) {
			for(int i = 0; i < dockpars_num_of_genes; i++) {
				if (i == 0) {
					printf("\n%s\n", "----------------------------------------------------------");
					printf("%13s %20s %15s %15s %15s\n", "gene", "sq_grad", "delta", "sq_delta", "new.genotype");
				}
				printf("%13u %20.6f %15.6f %15.6f %15.6f\n", i, square_gradient[i], delta[i], square_delta[i], genotype[i]);
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		#endif

		// Updating number of ADADELTA iterations (energy evaluations)
		if (tidx == 0) {
			if (energy < best_energy)
			{
				best_energy = energy;
#ifdef ADADELTA_AUTOSTOP
				cons_succ++;
				cons_fail = 0;
#endif
			}
#ifdef ADADELTA_AUTOSTOP
			else
			{
				cons_succ = 0;
				cons_fail++;
			}
#endif

			iteration_cnt = iteration_cnt + 1;

			#if defined (DEBUG_ADADELTA_MINIMIZER) || defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
			printf("%20s %10.6f\n", "new.energy: ", energy);
			#endif

			#if defined (DEBUG_ENERGY_ADADELTA)
			printf("%-18s [%-5s]---{%-5s}   [%-10.7f]---{%-10.7f}\n", "-ENERGY-KERNEL7-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
			#endif
#ifdef ADADELTA_AUTOSTOP
			if (cons_succ >= 4)
			{
				rho *= LS_EXP_FACTOR;
				cons_succ = 0;
			}
			else
				if (cons_fail >= 4)
				{
					rho *= LS_CONT_FACTOR;
					cons_fail = 0;
				}
#endif
		}
		barrier(CLK_LOCAL_MEM_FENCE); // making sure that iteration_cnt is up-to-date
#ifdef ADADELTA_AUTOSTOP
	} while ((iteration_cnt < dockpars_max_num_of_iters)  && (rho > 0.01f));
#else
	} while (iteration_cnt < dockpars_max_num_of_iters);
#endif
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------

	// Mapping torsion angles
	for ( int gene_counter = tidx+3;
	          gene_counter < dockpars_num_of_genes;
	          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
	{
		map_angle(&(best_genotype[gene_counter]));
	}

	// Updating old offspring in population
	barrier(CLK_LOCAL_MEM_FENCE);
	event_t ev2 = async_work_group_copy(dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
	                                    best_genotype,
	                                    dockpars_num_of_genes, 0);
	// Updating eval counter and energy
	if (tidx == 0) {
		dockpars_evals_of_new_entities[run_id*dockpars_pop_size+entity_id] += iteration_cnt;
		dockpars_energies_next[run_id*dockpars_pop_size+entity_id] = best_energy;
		#if defined (DEBUG_ADADELTA_MINIMIZER) || defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
		printf("\n");
		printf("Termination criteria: ( #adadelta-iters >= %-3u )\n", dockpars_max_num_of_iters);
		printf("-------> End of ADADELTA minimization cycle, num of energy evals: %u, final energy: %.6f\n", iteration_cnt, best_energy);
		#endif
	}

	// Asynchronous copy should be finished by here
	wait_group_events(1, &ev2);
}
