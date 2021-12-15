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

#include <cstdint>
#include "kernels.hpp"
#include "auxiliary_genetic.cpp"
#include "calcMergeEneGra.cpp"
#include "omp.h"

#define ADADELTA_AUTOSTOP

// Gradient-based adadelta minimizer
// https://arxiv.org/pdf/1212.5701.pdf
// Alternative to Solis-Wets / Steepest-Descent / FIRE

// "rho": controls degree of memory of previous gradients
//        ranges between [0, 1[
//        "rho" = 0.9 most popular value
// "epsilon":  to better condition the square root

// Adadelta parameters (TODO: to be moved to header file?)
//#define RHO             0.9f
//#define EPSILON         1e-6
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

void gpu_gradient_minAD(        uint32_t pops_by_runs,
                                uint32_t work_pteam,
                                float* pMem_conformations_next,
                                float* pMem_energies_next,
                                GpuData& cData,
                                GpuDockparameters dockpars
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

	// Energy may go up, so we keep track of the best energy ever calculated.
	// Then, we return the genotype corresponding 
	// to the best observed energy, i.e. "best_genotype"
	#pragma omp target teams thread_limit(NUM_OF_THREADS_PER_BLOCK)\
	     num_teams(pops_by_runs)
	{
	int entity_id;
	float best_energy;
	float sFloatAccumulator;

	volatile float sFloatBuff[6 * MAX_NUM_OF_ATOMS + 5 * ACTUAL_GENOTYPE_LENGTH];

	// Ligand-atom position and partial energies
	float3struct* calc_coords = (float3struct*)sFloatBuff;

	// Gradient of the intermolecular energy per each ligand atom
	// Also used to store the accummulated gradient per each ligand atom
#ifdef FLOAT_GRADIENTS
	float3struct* cartesian_gradient = (float3struct*)(calc_coords + MAX_NUM_OF_ATOMS);
#else
	int3struct* cartesian_gradient = (int3struct*)(calc_coords + MAX_NUM_OF_ATOMS);
#endif
	// Genotype pointers
	float* genotype = (float*)(cartesian_gradient + MAX_NUM_OF_ATOMS); // so far used 3*2*MAX_NUM_OF_ATOMS
	float* best_genotype = genotype + ACTUAL_GENOTYPE_LENGTH;

	// Partial results of the gradient step
	float* gradient = best_genotype + ACTUAL_GENOTYPE_LENGTH;

	// Squared updates E[dx^2]
	float* square_delta = gradient + ACTUAL_GENOTYPE_LENGTH;

	// Vector for storing squared gradients E[g^2]
	float* square_gradient = square_delta + ACTUAL_GENOTYPE_LENGTH; // so far used 5*ACTUAL_GENOTYPE_LENGTH
#ifdef ADADELTA_AUTOSTOP
        float rho;
        int cons_succ;
        int cons_fail;
#endif

	#pragma omp parallel
	{
		const int threadIdx = omp_get_thread_num();
  		const int blockDim = omp_get_num_threads();
  		const int blockIdx = omp_get_team_num();
  		const int gridDim = omp_get_num_teams();
	for (uint32_t idx = blockIdx; idx < pops_by_runs; idx+=gridDim) {  // for teams
	 // Determining entity, and its run, energy, and genotype
        float energy;
        int   run_id = blockIdx / dockpars.num_of_lsentities;

	// Iteration counter for the minimizer
	uint32_t iteration_cnt = 0;

	if (threadIdx == 0)
	{
		// Since entity 0 is the best one due to elitism,
		// it should be subjected to random selection
		entity_id = blockIdx % dockpars.num_of_lsentities;
		if (entity_id == 0) {
			// If entity 0 is not selected according to LS-rate,
			// choosing another entity
			if (100.0f*gpu_randf(cData.pMem_prng_states, blockIdx, threadIdx) > dockpars.lsearch_rate) {
				entity_id = dockpars.num_of_lsentities; // AT - Should this be (uint)(dockpars_pop_size * gpu_randf(dockpars_prng_states))?
			}
		}

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
#pragma omp barrier
	//__syncthreads();
	energy = pMem_energies_next[run_id * dockpars.pop_size + entity_id];

	int offset = (run_id * dockpars.pop_size + entity_id) * GENOTYPE_LENGTH_IN_GLOBMEM;
	for (int i = threadIdx ; i < dockpars.num_of_genes; i += blockDim)
	{
		genotype[i] = pMem_conformations_next[offset + i];
	}

	// -------------------------------------------------------------------
	// Calculate gradients (forces) for intermolecular energy
	// Derived from autodockdev/maps.py
	// -------------------------------------------------------------------

	#if defined (DEBUG_ENERGY_KERNEL)
	float interE;
	float intraE;
	#endif

	// Update vector, i.e., "delta".
	// It is added to the genotype to create the next genotype.
	// E.g. in steepest descent "delta" is -1.0 * stepsize * gradient

	// Asynchronous copy should be finished by here
#pragma omp barrier
//	__syncthreads();

	// Initializing vectors
	for(uint32_t i = threadIdx;
	             i < dockpars.num_of_genes;
	             i+= blockDim)
	{
		gradient[i]        = 0.0f;
		square_gradient[i] = 0.0f;
		square_delta[i]    = 0.0f;
		best_genotype[i] = genotype[i];
	}

	// Initializing best energy
	if (threadIdx == 0) {
		best_energy = INFINITY;
	}

#ifdef ADADELTA_AUTOSTOP
	if (threadIdx == 0) {
		rho = 1.0f;
		cons_succ = 0;
		cons_fail = 0;
	}
#endif

	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// Perform adadelta iterations

	// The termination criteria is based on 
	// a maximum number of iterations, and
	// the minimum step size allowed for single-floating point numbers 
	// (IEEE-754 single float has a precision of about 6 decimal digits)
	do {
		// Printing number of ADADELTA iterations
		#if defined (DEBUG_ADADELTA_MINIMIZER) || defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
		if (threadIdx == 0) {
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
#pragma omp barrier
//		__syncthreads();

		gpu_calc_energrad(
		                  // Some OpenCL compilers don't allow declaring
		                  // local variables within non-kernel functions.
		                  // These local variables must be declared in a kernel,
		                  // and then passed to non-kernel functions.
				  threadIdx,
				  blockDim,
				  cData,
				  dockpars,
		                  genotype,
		                  energy,
		                  run_id,
		                  calc_coords,
		                  #if defined (DEBUG_ENERGY_KERNEL)
		                  interE,
		                  intraE,
		                  #endif
		                  // Gradient-related arguments
		                  // Calculate gradients (forces) for intermolecular energy
		                  // Derived from autodockdev/maps.py
		                  cartesian_gradient,
		                  gradient,
		                  &sFloatAccumulator
		                 );

		// =============================================================
		// =============================================================
		// =============================================================
		#if defined (DEBUG_ENERGY_ADADELTA)
		if (threadIdx == 0) {
			#if defined (PRINT_ADADELTA_ENERGIES)
			printf("\n");
			printf("%-10s %-10.6f \n", "intra: ",  intraE);
			printf("%-10s %-10.6f \n", "grids: ",  interE);
			printf("%-10s %-10.6f \n", "Energy: ", intraE + interE);
			#endif

			#if defined (PRINT_ADADELTA_GENES_AND_GRADS)
			for(uint i = 0; i < dockpars.num_of_genes; i++) {
				if (i == 0) {
					printf("\n%s\n", "----------------------------------------------------------");
					printf("%13s %13s %5s %15s %15s\n", "gene_id", "gene.value", "|", "gene.grad", "(autodockdevpy units)");
				}
				printf("%13u %13.6f %5s %15.6f %15.6f\n", i, genotype[i], "|", gradient[i], (i<3)? (gradient[i]/0.375f):(gradient[i]*180.0f/PI_FLOAT));
			}
			#endif

			#if defined (PRINT_ADADELTA_ATOMIC_COORDS)
			for(uint i = 0; i < dockpars.num_of_atoms; i++) {
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
#pragma omp barrier
//		__syncthreads();
		#endif // DEBUG_ENERGY_ADADELTA

		for(int i = threadIdx;
		        i < dockpars.num_of_genes;
		        i+= blockDim)
		{
			if (energy < best_energy) // we need to be careful not to change best_energy until we had a chance to update the whole array
				best_genotype[i] = genotype[i];

			// Accumulating gradient^2 (eq.8 in the paper)
			// square_gradient corresponds to E[g^2]
			square_gradient[i] = RHO * square_gradient[i] + (1.0f - RHO) * gradient[i] * gradient[i];

			// Computing update (eq.9 in the paper)
			float delta = -1.0f * gradient[i] * sqrt((float)(square_delta[i] + EPSILON) / (float)(square_gradient[i] + EPSILON));

			// Accumulating update^2
			// square_delta corresponds to E[dx^2]
			square_delta[i] = RHO * square_delta[i] + (1.0f - RHO) * delta * delta;

			// Applying update
			genotype[i] += delta;
		}
#pragma omp barrier
//		__syncthreads();

		#if defined (DEBUG_SQDELTA_ADADELTA)
		if (/*(get_group_id(0) == 0) &&*/ (threadIdx == 0)) {
			for(int i = 0; i < dockpars.num_of_genes; i++) {
				if (i == 0) {
					printf("\n%s\n", "----------------------------------------------------------");
					printf("%13s %20s %15s %15s %15s\n", "gene", "sq_grad", "delta", "sq_delta", "new.genotype");
				}
				printf("%13u %20.6f %15.6f %15.6f %15.6f\n", i, square_gradient[i], delta[i], square_delta[i], genotype[i]);
			}
		}
#pragma omp barrier
//		__syncthreads();
		#endif

		// Updating number of ADADELTA iterations (energy evaluations)
		iteration_cnt = iteration_cnt + 1;
		if (threadIdx == 0) {
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
			{
				if (cons_fail >= 4)
				{
					rho *= LS_CONT_FACTOR;
					cons_fail = 0;
				}
			}
#endif
		}
#pragma omp barrier
//			__syncthreads(); // making sure that iteration_cnt is up-to-date
#ifdef ADADELTA_AUTOSTOP
	} while ((iteration_cnt < dockpars.max_num_of_iters)  && (rho > 0.01f));
#else
	} while (iteration_cnt < dockpars.max_num_of_iters);
#endif
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// Mapping torsion angles
	for (uint32_t gene_counter = threadIdx+3;
	              gene_counter < dockpars.num_of_genes;
	              gene_counter += blockDim)
	{
		map_angle(best_genotype[gene_counter]);
	}

	// Updating old offspring in population
#pragma omp barrier
//	__syncthreads();

	offset = (run_id * dockpars.pop_size + entity_id) * GENOTYPE_LENGTH_IN_GLOBMEM;
	for (uint gene_counter = threadIdx;
	          gene_counter < dockpars.num_of_genes;
	          gene_counter+= blockDim)
	{
		pMem_conformations_next[gene_counter + offset] = best_genotype[gene_counter];
	}

	// Updating eval counter and energy
	if (threadIdx == 0) {
		cData.pMem_evals_of_new_entities[run_id * dockpars.pop_size + entity_id] += iteration_cnt;
		pMem_energies_next[run_id * dockpars.pop_size + entity_id] = best_energy;

		#if defined (DEBUG_ADADELTA_MINIMIZER) || defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
		printf("\n");
		printf("Termination criteria: ( #adadelta-iters >= %-3u )\n", dockpars_max_num_of_iters);
		printf("-------> End of ADADELTA minimization cycle, num of energy evals: %u, final energy: %.6f\n", iteration_cnt, best_energy);
		#endif
	}
	}  // End for a set of teams
	} // end parallel section
	} // end team region
}


