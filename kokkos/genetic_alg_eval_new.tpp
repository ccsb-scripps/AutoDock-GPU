#include "calcenergy.hpp"

// TODO - templatize ExSpace - ALS
template<class Device>
void kokkos_gen_alg_eval_new(Dockpars* mypars,DockingParams<Device>& docking_params,GeneticParams& genetic_params,Conform<Device>& conform, RotList<Device>& rotlist, IntraContrib<Device>& intracontrib, InterIntra<Device>& interintra, Intra<Device>& intra)
{
	// Outer loop over mypars->pop_size * mypars->num_of_runs
        int league_size = mypars->pop_size * mypars->num_of_runs;
        Kokkos::parallel_for (Kokkos::TeamPolicy<ExSpace> (league_size, Kokkos::AUTO() ),
                        KOKKOS_LAMBDA (member_type team_member)
        {
                // Get team and league ranks
                int tidx = team_member.team_rank();
                int lidx = team_member.league_rank();

		// This compute-unit is responsible for elitist selection
	        if ((lidx % docking_params.pop_size) == 0) {
//			perform_elitist_selection(docking_params);
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
//			get_randnums();

			// Determine which run this team is doing
        		int run_id;
        		if (tidx == 0) {
        		        run_id = lidx/docking_params.pop_size;
        		}

			team_member.team_barrier();

			// Binary tournament selection
			if (tidx < 4) {        //it is not ensured that the four candidates will be different...
                        	parent_candidates[tidx]  = (int) (docking_params.pop_size*randnums[tidx]); //using randnums[0..3]
                        	candidate_energies[tidx] = docking_params.energies_current(run_id*docking_params.pop_size+parent_candidates[tidx]);
                	}

			team_member.team_barrier();

			// Choose parents
			if (tidx < 2) {
				// Notice: dockpars_tournament_rate was scaled down to [0,1] in host
				// to reduce number of operations in device
				if (candidate_energies[2*tidx] < candidate_energies[2*tidx+1])
					if (/*100.0f**/randnums[4+tidx] < genetic_params.tournament_rate) {                //using randnum[4..5]
						parents[tidx] = parent_candidates[2*tidx];
					} else {
						parents[tidx] = parent_candidates[2*tidx+1];
					}
				else
					if (/*100.0f**/randnums[4+tidx] < genetic_params.tournament_rate) {
						parents[tidx] = parent_candidates[2*tidx+1];
					} else {
						parents[tidx] = parent_candidates[2*tidx];
					}
			}

			team_member.team_barrier();

//			crossover();

			team_member.team_barrier();

//			mutation();

			team_member.team_barrier();

			// Get the current energy for each run
			float energy = kokkos_calc_energy(team_member, docking_params,conform, rotlist, intracontrib, interintra, intra, offspring_genotype);

			// Copy to global views
			if( tidx == 0 ) {
				docking_params.energies_next(lidx) = energy;
				docking_params.evals_of_new_entities(lidx) = 1;
			}

			// FIX ME Copying new offspring to next generation, maybe parallelizable - ALS
			for (int i_geno = 0; i_geno<docking_params.num_of_genes; i_geno++) {
				docking_params.conformations_next(i_geno + GENOTYPE_LENGTH_IN_GLOBMEM*lidx)
						= offspring_genotype[i_geno];
			}

			team_member.team_barrier();
		}
        });
}
