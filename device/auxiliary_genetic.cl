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


// -------------------------------------------------------
//
// -------------------------------------------------------
uint gpu_rand(__global uint* restrict prng_states)
// The GPU device function generates a random int
// with a linear congruential generator.
// Each thread (supposing num_of_runs*pop_size blocks and NUM_OF_THREADS_PER_BLOCK threads per block)
// has its own state which is stored in the global memory area pointed by
// prng_states (thread with ID tx in block with ID bx stores its state in prng_states[bx*NUM_OF_THREADS_PER_BLOCK+$
// The random number generator uses the gcc linear congruential generator constants.
{
	uint state;

	// Current state of the threads own PRNG
	// state = prng_states[get_group_id(0)*NUM_OF_THREADS_PER_BLOCK + get_local_id(0)];
	state = prng_states[get_global_id(0)];

	// Calculating next state
	state = (RAND_A*state+RAND_C);

	// Saving next state to memory
	// prng_states[get_group_id(0)*NUM_OF_THREADS_PER_BLOCK + get_local_id(0)] = state;
	prng_states[get_global_id(0)] = state;

  return state;
}

// -------------------------------------------------------
//
// -------------------------------------------------------
float gpu_randf(__global uint* restrict prng_states)
// The GPU device function generates a
// random float greater than (or equal to) 0 and less than 1.
// It uses gpu_rand() function.
{
	float state;

	// State will be between 0 and 1
	state =  native_divide(gpu_rand(prng_states),MAX_UINT)*0.999999f;

	return state;
}

// -------------------------------------------------------
//
// -------------------------------------------------------
void map_angle(__local float* angle)
// The GPU device function maps
// the input parameter to the interval 0...360
// (supposing that it is an angle).
{
	while (*angle >= 360.0f) {
		*angle -= 360.0f;
	}

	while (*angle < 0.0f) {
		*angle += 360.0f;
	}
}

// -------------------------------------------------------
//
// -------------------------------------------------------
void gpu_perform_elitist_selection(
                                            int    dockpars_pop_size,
                                   __global float* restrict dockpars_energies_current,
                                   __global float* restrict dockpars_energies_next,
                                   __global int*   restrict dockpars_evals_of_new_entities,
                                            int    dockpars_num_of_genes,
                                   __global float* restrict dockpars_conformations_next,
                             __global const float* restrict dockpars_conformations_current,
                                   __local  float* best_energies,
                                   __local  int*   best_IDs,
                                   __local  int*   best_ID
                                  )
// The GPU device function performs elitist selection,
// that is, it looks for the best entity in conformations_current and
// energies_current of the run that corresponds to the block ID,
// and copies it to the place of the first entity in
// conformations_next and energies_next.
{
	int entity_counter;
	int gene_counter;
	float best_energy;
	int tidx = get_local_id(0);

	if (tidx < dockpars_pop_size) {
		best_energies[tidx] = dockpars_energies_current[get_group_id(0)+tidx];
		best_IDs[tidx] = tidx;
	}

	for (entity_counter = NUM_OF_THREADS_PER_BLOCK+tidx;
	     entity_counter < dockpars_pop_size;
	     entity_counter+= NUM_OF_THREADS_PER_BLOCK)
	{
		if (dockpars_energies_current[get_group_id(0)+entity_counter] < best_energies[tidx]) {
			best_energies[tidx] = dockpars_energies_current[get_group_id(0)+entity_counter];
			best_IDs[tidx] = entity_counter;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// This could be implemented with a tree-like structure
	// which may be slightly faster
	if (tidx == 0)
	{
		best_energy = best_energies[0];
		best_ID[0] = best_IDs[0];

		for (entity_counter = 1;
		     entity_counter < NUM_OF_THREADS_PER_BLOCK;
		     entity_counter++)
		{
			if ((best_energies[entity_counter] < best_energy) && (entity_counter < dockpars_pop_size)) {
				best_energy = best_energies[entity_counter];
				best_ID[0] = best_IDs[entity_counter];
			}
		}

		// Setting energy value of new entity
		dockpars_energies_next[get_group_id(0)] = best_energy;

		// Zero (0) evals were performed for entity selected with elitism (since it was copied only)
		dockpars_evals_of_new_entities[get_group_id(0)] = 0;
	}
	// "best_id" stores the id of the best entity in the population,
	// Copying genotype and energy value to the first entity of new population
	barrier(CLK_LOCAL_MEM_FENCE);

	for (gene_counter = tidx;
	     gene_counter < dockpars_num_of_genes;
	     gene_counter+= NUM_OF_THREADS_PER_BLOCK)
	{
		dockpars_conformations_next[GENOTYPE_LENGTH_IN_GLOBMEM*get_group_id(0)+gene_counter] = dockpars_conformations_current[GENOTYPE_LENGTH_IN_GLOBMEM*get_group_id(0) + GENOTYPE_LENGTH_IN_GLOBMEM*best_ID[0]+gene_counter];
	}
}
