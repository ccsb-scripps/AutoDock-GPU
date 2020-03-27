/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

AutoDock is a Trade Mark of the Scripps Research Institute.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

*/

#include "random.hpp"

//The GPU device function performs elitist selection,
//that is, it looks for the best entity in current.conformations and
//current.energies of the run that corresponds to the block ID,
//and copies it to the place of the first entity in
//next.conformations and next.energies
template<class Device>
KOKKOS_INLINE_FUNCTION void perform_elitist_selection(const member_type& team_member, const Generation<Device>& current, const Generation<Device>& next, const DockingParams<Device>& docking_params)
{
        // Get team and league ranks
        int tidx = team_member.team_rank();
        int lidx = team_member.league_rank();
	int team_size = team_member.team_size();

	// Scratch float and int of length team_size
	TeamFloat best_energies(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));
	TeamInt best_IDs(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));

        if (tidx < docking_params.pop_size) {
                best_energies[tidx] = current.energies(lidx+tidx);
                best_IDs[tidx] = tidx;
        }

        for (int entity_counter = team_size+tidx;
             entity_counter < docking_params.pop_size;
             entity_counter+= team_size) {

             if (current.energies(lidx+entity_counter) < best_energies[tidx]) {
                best_energies[tidx] = current.energies(lidx+entity_counter);
                best_IDs[tidx] = entity_counter;
             }
        }

        team_member.team_barrier();

        // This could be implemented with a tree-like structure
        // which may be slightly faster
        if (tidx == 0) {
                for (int entity_counter = 1;
                     entity_counter < team_size;
                     entity_counter++) {

                     if ((best_energies[entity_counter] < best_energies(0)) && (entity_counter < docking_params.pop_size)) {
                              best_energies(0) = best_energies[entity_counter];
                              best_IDs(0) = best_IDs[entity_counter];
                     }
                }

                // Setting energy value of new entity
                next.energies(lidx) = best_energies(0);

                // Zero (0) evals were performed for entity selected with elitism (since it was copied only)
                docking_params.evals_of_new_entities(lidx) = 0;
        }

        // "best_id" stores the id of the best entity in the population,
        // Copying genotype and energy value to the first entity of new population
        team_member.team_barrier();

	copy_genotype(team_member, next, lidx, current, (lidx + best_IDs(0)));
}


template<class Device>
KOKKOS_INLINE_FUNCTION void crossover(const member_type& team_member, const Generation<Device>& current, const DockingParams<Device>& docking_params, const GeneticParams& genetic_params, const int run_id, TenFloat randnums, TwoInt parents,
					Genotype offspring_genotype)
{
        // Get team and league ranks
        int tidx = team_member.team_rank();
        int lidx = team_member.league_rank();
        int team_size = team_member.team_size();

	TwoInt covr_point(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));

	// Notice: dockpars_crossover_rate was scaled down to [0,1] in host
	// to reduce number of operations in device
	if (/*100.0f**/randnums[6] < genetic_params.crossover_rate)   // Using randnums[6]
	{
		// Using randnum[7..8]
		covr_point[0] = (int) ((docking_params.num_of_genes-1)*randnums[7]);
		covr_point[1] = (int) ((docking_params.num_of_genes-1)*randnums[8]);

		team_member.team_barrier();

		// covr_point[0] should store the lower crossover-point
		if (tidx == 0) {
			if (covr_point[1] < covr_point[0]) {
				int temp_covr_point = covr_point[1];
				covr_point[1]   = covr_point[0];
				covr_point[0]   = temp_covr_point;
			}
		}

		team_member.team_barrier();

		for (int gene_counter = tidx;
		     gene_counter < docking_params.num_of_genes;
		     gene_counter+= team_size)
		{
			// Two-point crossover
			if (covr_point[0] != covr_point[1])
			{
				if ((gene_counter <= covr_point[0]) || (gene_counter > covr_point[1]))
					offspring_genotype[gene_counter] = current.conformations((run_id*docking_params.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter);
				else
					offspring_genotype[gene_counter] = current.conformations((run_id*docking_params.pop_size+parents[1])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter);
			}
			// Single-point crossover
			else
			{
				if (gene_counter <= covr_point[0])
					offspring_genotype[gene_counter] = current.conformations((run_id*docking_params.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter);
				else
					offspring_genotype[gene_counter] = current.conformations((run_id*docking_params.pop_size+parents[1])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter);
			}
		}

	}
	else    //no crossover
	{
		copy_genotype(team_member, offspring_genotype, current, (run_id*docking_params.pop_size+parents[0]));

                team_member.team_barrier();
	}
}

template<class Device>
KOKKOS_INLINE_FUNCTION void mutation(const member_type& team_member, const DockingParams<Device>& docking_params, const GeneticParams& genetic_params,
				     Genotype offspring_genotype)
{
        // Get team and league ranks
        int tidx = team_member.team_rank();
        int team_size = team_member.team_size();

	for (int gene_counter = tidx;
	     gene_counter < docking_params.num_of_genes;
	     gene_counter+= team_size)
	{
		// Notice: dockpars_mutation_rate was scaled down to [0,1] in host
		// to reduce number of operations in device
		if (/*100.0f**/rand_float(team_member, docking_params) < genetic_params.mutation_rate)
		{
			// Translation genes
			if (gene_counter < 3) {
				offspring_genotype[gene_counter] += genetic_params.abs_max_dmov*(2*rand_float(team_member, docking_params)-1);
			}
			// Orientation and torsion genes
			else {
				offspring_genotype[gene_counter] += genetic_params.abs_max_dang*(2*rand_float(team_member, docking_params)-1);

				// Quick modulo
				while (offspring_genotype[gene_counter] >= 360.0f) { offspring_genotype[gene_counter] -= 360.0f; }
				while (offspring_genotype[gene_counter] < 0.0f   ) { offspring_genotype[gene_counter] += 360.0f; }
			}

		}
	}
}
