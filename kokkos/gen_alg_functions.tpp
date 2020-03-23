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

//The GPU device function performs elitist selection,
//that is, it looks for the best entity in conformations_current and
//energies_current of the run that corresponds to the block ID,
//and copies it to the place of the first entity in
//conformations_next and energies_next.
template<class Device>
KOKKOS_INLINE_FUNCTION void perform_elitist_selection(const member_type& team_member, const DockingParams<Device>& docking_params)
{
        // Get team and league ranks
        int tidx = team_member.team_rank();
        int lidx = team_member.league_rank();
	int team_size = team_member.team_size();

	float best_energies[team_size];
        int best_IDs[team_size];
        int best_ID;

        int entity_counter;
        int gene_counter;
        float best_energy;

        if (tidx < docking_params.pop_size) {
                best_energies[tidx] = docking_params.energies_current(lidx+tidx);
                best_IDs[tidx] = tidx;
        }

        for (entity_counter = team_size+tidx;
             entity_counter < docking_params.pop_size;
             entity_counter+= team_size) {

             if (docking_params.energies_current(lidx+entity_counter) < best_energies[tidx]) {
                best_energies[tidx] = docking_params.energies_current(lidx+entity_counter);
                best_IDs[tidx] = entity_counter;
             }
        }

        team_member.team_barrier();

        // This could be implemented with a tree-like structure
        // which may be slightly faster
        if (tidx == 0) {
                best_energy = best_energies[0];
                best_ID = best_IDs[0];

                for (entity_counter = 1;
                     entity_counter < team_size;
                     entity_counter++) {

                     if ((best_energies[entity_counter] < best_energy) && (entity_counter < docking_params.pop_size)) {
                              best_energy = best_energies[entity_counter];
                              best_ID = best_IDs[entity_counter];
                     }
                }

                // Setting energy value of new entity
                docking_params.energies_next(lidx) = best_energy;

                // Zero (0) evals were performed for entity selected with elitism (since it was copied only)
                docking_params.evals_of_new_entities(lidx) = 0;
        }

        // "best_id" stores the id of the best entity in the population,
        // Copying genotype and energy value to the first entity of new population
        team_member.team_barrier();

        for (gene_counter = tidx;
             gene_counter < docking_params.num_of_genes;
             gene_counter+= team_size) {
             docking_params.conformations_next(GENOTYPE_LENGTH_IN_GLOBMEM*lidx+gene_counter)
		     = docking_params.conformations_current(GENOTYPE_LENGTH_IN_GLOBMEM*lidx + GENOTYPE_LENGTH_IN_GLOBMEM*best_ID+gene_counter);
        }

}
