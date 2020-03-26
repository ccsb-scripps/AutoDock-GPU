#include "calcenergy.hpp"
#include "gen_alg_functions.hpp"
#include "random.hpp"

// TODO - templatize ExSpace - ALS
template<class Device>
void gen_alg_eval_new(Generation<Device>& current, Generation<Device>& next, Dockpars* mypars,DockingParams<Device>& docking_params,GeneticParams& genetic_params,Constants<Device>& consts)
{
	// Outer loop over mypars->pop_size * mypars->num_of_runs
        int league_size = mypars->pop_size * mypars->num_of_runs;
        Kokkos::parallel_for (Kokkos::TeamPolicy<ExSpace> (league_size, NUM_OF_THREADS_PER_BLOCK ),
                        KOKKOS_LAMBDA (member_type team_member)
        {
                // Get team and league ranks
                int tidx = team_member.team_rank();
                int lidx = team_member.league_rank();

		// This compute-unit is responsible for elitist selection
		if ((lidx % docking_params.pop_size) == 0) {
			perform_elitist_selection(team_member, current, next, docking_params);
		}else{
			// Some local arrays
			float offspring_genotype[ACTUAL_GENOTYPE_LENGTH];
			float randnums[10];
			int parent_candidates[4];
			float candidate_energies[4];
			int parents[2];
			
			// Generating the following random numbers:
			// [0..3] for parent candidates,
			// [4..5] for binary tournaments, [6] for deciding crossover,
			// [7..8] for crossover points, [9] for local search
			for (int gene_counter = tidx;
				gene_counter < 10;
				gene_counter+= team_member.team_size()) {
				randnums[gene_counter] = rand_float(team_member, docking_params);
			}

			// Determine which run this team is doing
			int run_id;
			if (tidx == 0) {
				run_id = lidx/docking_params.pop_size;
			}

			team_member.team_barrier();

			// Binary tournament selection
			// it is not ensured that the four candidates will be different...
			for (int parent_counter = tidx;
                              parent_counter < 4;
                              parent_counter+= team_member.team_size()){
				parent_candidates[parent_counter]  = (int) (docking_params.pop_size*randnums[parent_counter]); //using randnums[0..3]
				candidate_energies[parent_counter] = current.energies(run_id*docking_params.pop_size+parent_candidates[parent_counter]);
			}

			team_member.team_barrier();

			// Choose parents
			for (int parent_counter = tidx;
                              parent_counter < 2;
                              parent_counter+= team_member.team_size()) {
				// Notice: dockpars_tournament_rate was scaled down to [0,1] in host
				// to reduce number of operations in device
				if (candidate_energies[2*parent_counter] < candidate_energies[2*parent_counter+1])
					if (/*100.0f**/randnums[4+parent_counter] < genetic_params.tournament_rate) {                //using randnum[4..5]
						parents[parent_counter] = parent_candidates[2*parent_counter];
					} else {
						parents[parent_counter] = parent_candidates[2*parent_counter+1];
					}
				else
					if (/*100.0f**/randnums[4+parent_counter] < genetic_params.tournament_rate) {
						parents[parent_counter] = parent_candidates[2*parent_counter+1];
					} else {
						parents[parent_counter] = parent_candidates[2*parent_counter];
					}
			}

			team_member.team_barrier();

			crossover(team_member, current, docking_params, genetic_params, run_id, randnums, parents, offspring_genotype);

			team_member.team_barrier();

			mutation(team_member, docking_params, genetic_params, offspring_genotype);

			team_member.team_barrier();

			// Get the current energy for each run
			float energy = calc_energy(team_member, docking_params, consts, offspring_genotype);

			// Copy to global views
			if( tidx == 0 ) {
				next.energies(lidx) = energy;
				docking_params.evals_of_new_entities(lidx) = 1;
			}

			// FIX ME Copying new offspring to next generation, maybe parallelizable - ALS
			for (int i_geno = 0; i_geno<docking_params.num_of_genes; i_geno++) {
				next.conformations(i_geno + GENOTYPE_LENGTH_IN_GLOBMEM*lidx) = offspring_genotype[i_geno];
			}

			team_member.team_barrier();
		}
        });
}
