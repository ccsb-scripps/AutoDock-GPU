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


// Gradient-based fire minimizer
// FIRE: (F)ast (I)nertial (R)elaxation (E)ngine
// https://www.math.uni-bielefeld.de/~gaehler/papers/fire.pdf
// https://doi.org/10.1103/PhysRevLett.97.170201
// Alternative to Solis-Wets / Steepest-Descent / AdaDelta

// These parameters differ from the original implementation:
//            - DT_INC is larger          [ larger increments of "dt" ]
//            - DT_DEC is closer to 1.0   [ smaller decrements of "dt" ]
//            - ALPHA_START is larger     [ less inertia ]

// As a result, this implementation is "less local" than the original.
// In other words, it is easier to exit the current local minima and
// jump to a nearby local minima.

// Fire parameters (TODO: to be moved to header file?)
#define SUCCESS_MIN      5     // N_min   = 5
#define DT_INC           1.2f  // f_inc   = 1.1
#define DT_DEC           0.8f  // f_dec   = 0.5
#define ALPHA_START      0.2f  // a_start = 0.1
#define ALPHA_DEC        0.99f // f_a     = 0.99

// Tunable parameters
// This one tuned by trial and error
#define DT_MAX           10.0f
#define DT_MAX_DIV_THREE (DT_MAX / 3.0f)

// New parameter
// Not in original implementation
// if "dt" becomes smaller than this value, stop optimization
#define DT_MIN           1e-6f

// Enabling "DEBUG_ENERGY_FIRE" requires
// manually enabling "DEBUG_ENERGY_KERNEL" in calcenergy.cl
//#define DEBUG_ENERGY_FIRE
//#define PRINT_FIRE_ENERGIES
//#define PRINT_FIRE_GENES_AND_GRADS
//#define PRINT_FIRE_ATOMIC_COORDS
//#define PRINT_FIRE_PARAMETERS

// Enable DEBUG_FIRE_MINIMIZER for a seeing a detailed FIRE evolution
// If only PRINT_FIRE_MINIMIZER_ENERGY_EVOLUTION is enabled,
// then a only a simplified FIRE evolution will be shown
//#define DEBUG_FIRE_MINIMIZER
//#define PRINT_FIRE_MINIMIZER_ENERGY_EVOLUTION

// Enable this for debugging FIRE from a defined initial genotype
//#define DEBUG_FIRE_INITIAL_2BRT

__kernel void __attribute__ ((reqd_work_group_size(NUM_OF_THREADS_PER_BLOCK,1,1)))
gradient_minFire(
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

	uint tidx = get_local_id(0);
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

		#if defined (DEBUG_FIRE_MINIMIZER) || defined (PRINT_FIRE_MINIMIZER_ENERGY_EVOLUTION)
		printf("\n");
		printf("-------> Start of FIRE minimization cycle\n");
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
	// Asynchronous copy should be finished by here
	wait_group_events(1, &ev);

	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------

	// Partial results of the gradient step
	__local float gradient[ACTUAL_GENOTYPE_LENGTH];
	__local float candidate_gradient[ACTUAL_GENOTYPE_LENGTH];

	// Energy may go up, so we keep track of the best energy ever calculated.
	// Then, we return the genotype corresponding 
	// to the best observed energy, i.e. "best_genotype"
	__local float best_energy;
	__local float best_genotype[ACTUAL_GENOTYPE_LENGTH];

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

	// Enable this for debugging FIRE from a defined initial genotype
	#if defined (DEBUG_FIRE_INITIAL_2BRT)
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

	// FIRE counters
	__local float velocity [ACTUAL_GENOTYPE_LENGTH]; // velocity
	__local float alpha;                             // alpha
	__local uint  count_success;                     // count_success
	__local float dt;                                // "dt"

	// Calculating the gradient/velocity norm
	__local float gradient_tmp [ACTUAL_GENOTYPE_LENGTH];
	__local float inv_gradient_norm;
	__local float velocity_tmp [ACTUAL_GENOTYPE_LENGTH];
	__local float velocity_norm;
	__local float velnorm_div_gradnorm;

	// Defining FIRE power
	__local float power_tmp [ACTUAL_GENOTYPE_LENGTH];
	__local float power;

	// Calculating gradient-norm components
	for ( int gene_counter = tidx;
	          gene_counter < dockpars_num_of_genes;
	          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
	{
		gradient_tmp [gene_counter] = gradient [gene_counter] * gradient [gene_counter];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (tidx == 0) {
		// nitializing
		alpha         = ALPHA_START;
		count_success = 0;
		dt            = DT_MAX_DIV_THREE;

		// Initializing "gradient norm" to 0.0f,
		// but stored it in inv_gradient_norm
		inv_gradient_norm = 0.0f;
		
		// Summing up squares to continue calculation of "gradient-norm"
		for (int i = 0; i < dockpars_num_of_genes; i++) {
			inv_gradient_norm += gradient_tmp [i];
		}
		
		// Note: ALPHA_START is included as a factor here
		inv_gradient_norm = native_sqrt( inv_gradient_norm);
		inv_gradient_norm = ALPHA_START * native_recip(inv_gradient_norm);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Starting velocity
	// This equation was found by trial and error
	for ( int gene_counter = tidx;
	          gene_counter < dockpars_num_of_genes;
	          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
	{
		velocity [gene_counter] = - gradient [gene_counter] * inv_gradient_norm;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Keeping track of best genotype which 
	// may or may not be the last genotype
	if (tidx == 0) {
		best_energy = energy;
		for (int i = 0; i < dockpars_num_of_genes; i++) {
			best_genotype [i] = genotype [i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform fire iterations

	// The termination criteria is based on 
	// a maximum number of iterations, and
	// value of "dt" (fire specific)
	do {
		// Printing number of FIRE iterations
		#if defined (DEBUG_FIRE_MINIMIZER)
		if (tidx == 0) {
			printf("%s\n", "----------------------------------------------------------");
		}
		#endif
		
		#if defined (DEBUG_FIRE_MINIMIZER) || defined (PRINT_FIRE_MINIMIZER_ENERGY_EVOLUTION)
		if (tidx == 0) {
			printf("%-15s %-3u ", "# FIRE iteration: ", iteration_cnt);
		}
		#endif

		// Creating new (candidate) genotypes
		for ( int gene_counter = tidx;
		          gene_counter < dockpars_num_of_genes;
		          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
		{
			candidate_genotype [gene_counter] = genotype [gene_counter] + dt * velocity [gene_counter];
		}

// Replacing separate gradient and energy 
// calculations with a single & unified
// gpu_calc_energrad() function
// IMPORTANT: be careful with input/output (RE) assignment
// of genotypes, energy, and gradients
#if 0
		// =============================================================
		// Calculating (candidate) gradient
		// from "candidate_genotype"
		barrier(CLK_LOCAL_MEM_FENCE);

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
		                  candidate_genotype,
		                  &candidate_energy,
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
		                  i_gradient_x, i_gradient_y, i_gradient_z,
		                  f_gradient_x, f_gradient_y, f_gradient_z,
		                  candidate_gradient
		                 );
		// =============================================================

		// Evaluating (candidate) genotype
		// i.e. get (candidate) energy
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
#endif

		// =============================================================
		// =============================================================
		// =============================================================
		// Calculating energy & gradient
		barrier(CLK_LOCAL_MEM_FENCE);

		gpu_calc_energrad(dockpars_rotbondlist_length,
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
		                  candidate_genotype,
		                  &candidate_energy,
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
		                  i_gradient_x, i_gradient_y, i_gradient_z,
		                  f_gradient_x, f_gradient_y, f_gradient_z,
		                  candidate_gradient
		                 );
		// =============================================================
		// =============================================================
		// =============================================================

		// Calculating power
		// power is force * velocity.
		// force = -gradient
		barrier(CLK_LOCAL_MEM_FENCE);

		for ( int gene_counter = tidx;
		          gene_counter < dockpars_num_of_genes;
		          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
		{
			// Calculating power
			power_tmp [gene_counter] = -candidate_gradient [gene_counter] * velocity [gene_counter];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if (tidx == 0) {
			power = 0.0f;
			// Summing dot products
			for (int i = 0; i < dockpars_num_of_genes; i++) {
				power += power_tmp [i];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		#if defined (DEBUG_ENERGY_FIRE)
		if (/*(get_group_id(0) == 0) &&*/ (tidx == 0)) {
			#if defined (PRINT_FIRE_ENERGIES)
			printf("\n");
			printf("%-10s %-10.6f \n", "intra: ",  partial_intraE[0]);
			printf("%-10s %-10.6f \n", "grids: ",  partial_interE[0]);
			printf("%-10s %-10.6f \n", "Energy: ", (partial_intraE[0] + partial_interE[0]));
			#endif

			printf("\n");
			printf("%-15s %-10.6f \n", "energy: "     ,  energy);
			printf("%-15s %-10.6f \n", "best_energy: ",  best_energy);

			printf("\n%-15s %-10.6f \n","dt: "           ,  dt);
			printf("%-15s %-10.6f \n",  "power: "        ,  power);
			printf("%-15s %-10.6f \n",  "alpha: "        ,  alpha);
			printf("%-15s %-10u \n\n",   "count_success: " ,  count_success);

			#if defined (PRINT_FIRE_GENES_AND_GRADS)
			for(int i = 0; i < dockpars_num_of_genes; i++) {
				if (i == 0) {
					//printf("\n%s\n", "----------------------------------------------------------");
					printf("%13s %13s %5s %15s %21s %15s\n", "gene_id", "genotype", "|", "gradient", "(autodockdevpy units)", "velocity");
				}
				printf("%13u %13.6f %5s %15.6f %21.6f %15.6f\n", i, genotype[i], "|", gradient[i], (i<3)? (gradient[i]/dockpars_grid_spacing):(gradient[i]*180.0f/PI_FLOAT), velocity[i]);
			}

			for(int i = 0; i < dockpars_num_of_genes; i++) {
				if (i == 0) {
					//printf("\n%s\n", "----------------------------------------------------------");
					printf("\n");
					printf("%13s %13s %5s %15s\n", "gene_id", "cand_genotype", "|", "cand_gradient");
				}
				printf("%13u %13.6f %5s %15.6f\n", i, candidate_genotype[i], "|", candidate_gradient[i]);
			}
			#endif

			#if defined (PRINT_FIRE_ATOMIC_COORDS)
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

		// Going uphill (against the gradient)
		if (power < 0.0f) {
			// Using same equation as for starting velocity
			for ( int gene_counter = tidx;
			          gene_counter < dockpars_num_of_genes;
			          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
			{
				// Calculating gradient-norm
				gradient_tmp [gene_counter] = gradient [gene_counter] * gradient [gene_counter];
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			if (tidx == 0) {
				inv_gradient_norm = 0.0f;

				// Summing dot products
				for (int i = 0; i < dockpars_num_of_genes; i++) {
					inv_gradient_norm += gradient_tmp [i];
				}

				// Note: ALPHA is included as a factor here
				inv_gradient_norm = native_sqrt(inv_gradient_norm);
				inv_gradient_norm = ALPHA_START * native_recip(inv_gradient_norm);
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			for ( int gene_counter = tidx;
			          gene_counter < dockpars_num_of_genes;
			          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
			{
				velocity [gene_counter] = - gradient [gene_counter] * inv_gradient_norm;
			}
			barrier(CLK_LOCAL_MEM_FENCE);

		 	if (tidx == 0) {
				count_success = 0;
				alpha         = ALPHA_START;
				dt            = dt * DT_DEC;

				#if defined PRINT_FIRE_PARAMETERS
				printf("\nPower is negative :( = %f\n", power);
				printf("\n %15s %10.7f\n %15s %10.7f\n", "alpha: ", alpha,  "dt: ", dt);
				#endif
			}
		}
		// Going downhill
		else {
			if (tidx == 0) {
				count_success ++;

				#if defined PRINT_FIRE_PARAMETERS
				printf("\nPower is positive :) = %f\n", power);
				printf("\n %15s %10.7f\n %15s %10.7f\n", "old alpha: ", alpha, "old dt: ", dt);
				#endif

				// Reaching minimum number of consecutive successful steps (power >= 0)
				if (count_success > SUCCESS_MIN) {
					dt    = fmin (dt * DT_INC, DT_MAX);	// increase dt
					alpha = alpha * ALPHA_DEC; 		// decrease alpha

					#if defined PRINT_FIRE_PARAMETERS
					printf("\n count_success > %u\n", SUCCESS_MIN);
					printf("\n %10s %7.7f\n %10s %7.7f\n", "new alpha: ", alpha, "new dt: ", dt);
					#endif
				}
				else {
					#if defined PRINT_FIRE_PARAMETERS
					printf("\n count_success <= %u, do NOT update alpha or dt\n", SUCCESS_MIN);
					#endif
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// --------------------------------------
		// Always update: energy, genotype, gradient
		for ( int gene_counter = tidx;
		          gene_counter < dockpars_num_of_genes;
		          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
		{
			if (gene_counter == 0) {
				energy = candidate_energy;
			}
			genotype [gene_counter] = candidate_genotype [gene_counter];
			gradient [gene_counter] = candidate_gradient [gene_counter];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// --------------------------------------
		// Calculating gradient-norm
		for ( int gene_counter = tidx;
		          gene_counter < dockpars_num_of_genes;
		          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
		{
			gradient_tmp [gene_counter] = gradient [gene_counter] * gradient [gene_counter];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if (tidx == 0) {
			inv_gradient_norm = 0.0f;

			// Summing dot products
			for (int i = 0; i < dockpars_num_of_genes; i++) {
				inv_gradient_norm += gradient_tmp [i];
			}

			inv_gradient_norm = native_sqrt(inv_gradient_norm);
			inv_gradient_norm = native_recip(inv_gradient_norm);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// --------------------------------------
		// Calculating velocity-norm
		for ( int gene_counter = tidx;
		          gene_counter < dockpars_num_of_genes;
		          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
		{
			velocity_tmp [gene_counter] = velocity [gene_counter] * velocity [gene_counter];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if (tidx == 0) {
			velocity_norm  = 0.0f;

			// Summing dot products
			for (int i = 0; i < dockpars_num_of_genes; i++) {
				velocity_norm += velocity_tmp [i];
			}

			// Note: alpha is included as a factor here
			velocity_norm = native_sqrt(velocity_norm);
			velnorm_div_gradnorm = alpha * velocity_norm * inv_gradient_norm;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// --------------------------------------
		// Calculating velocity
		for ( int gene_counter = tidx;
		          gene_counter < dockpars_num_of_genes;
		          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
		{
			// NOTE: "velnorm_div_gradnorm" includes already the "alpha" factor
			velocity [gene_counter] = (1 - alpha) * velocity [gene_counter] - gradient [gene_counter] * velnorm_div_gradnorm;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// --------------------------------------
		// Updating number of fire iterations (energy evaluations)
		if (tidx == 0) {
	    		iteration_cnt = iteration_cnt + 1;

			#if defined (DEBUG_FIRE_MINIMIZER) || defined (PRINT_FIRE_MINIMIZER_ENERGY_EVOLUTION)
			printf("%20s %10.6f\n", "new.energy: ", energy);
			#endif

			#if defined (DEBUG_ENERGY_KERNEL6)
			printf("%-18s [%-5s]---{%-5s}   [%-10.7f]---{%-10.7f}\n", "-ENERGY-KERNEL6-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
			#endif

			if (energy <  best_energy) {
				best_energy = energy;

				for(int i = 0; i < dockpars_num_of_genes; i++) { 
					best_genotype[i] = genotype[i];
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

	} while ((iteration_cnt < dockpars_max_num_of_iters) && (dt > DT_MIN));
  	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------

  	// Updating eval counter and energy
	if (tidx == 0) {
		dockpars_evals_of_new_entities[run_id*dockpars_pop_size+entity_id] += iteration_cnt;
		dockpars_energies_next[run_id*dockpars_pop_size+entity_id] = best_energy;

		#if defined (DEBUG_FIRE_MINIMIZER) || defined (PRINT_FIRE_MINIMIZER_ENERGY_EVOLUTION)
		printf("\n");
		printf("Termination criteria: ( dt >= %10.10f ) OR ( #fire-iters >= %-3u )\n", DT_MIN, dockpars_max_num_of_iters);
		printf("-------> End of FIRE minimization cycle, num of energy evals: %u, final energy: %.6f\n", iteration_cnt, best_energy);
		#endif
	}

	// Mapping torsion angles
	for ( int gene_counter = tidx;
	          gene_counter < dockpars_num_of_genes;
	          gene_counter+= NUM_OF_THREADS_PER_BLOCK)
	{
		if (gene_counter >= 3) {
			map_angle(&(best_genotype[gene_counter]));
		}
	}

	// Updating old offspring in population
	barrier(CLK_LOCAL_MEM_FENCE);

	event_t ev2 = async_work_group_copy(dockpars_conformations_next+(run_id*dockpars_pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM,
	                                    best_genotype,
	                                    dockpars_num_of_genes, 0);

	// Asynchronous copy should be finished by here
	wait_group_events(1, &ev2);
}
