#include "calcenergy.hpp"

// TODO - templatize ExSpace - ALS
template<class Device>
void kokkos_calc_init_pop(Generation<Device>& current, Dockpars* mypars,DockingParams<Device>& docking_params,Conform<Device>& conform, RotList<Device>& rotlist, IntraContrib<Device>& intracontrib, InterIntra<Device>& interintra, Intra<Device>& intra)
{
	// Outer loop over mypars->pop_size * mypars->num_of_runs
        int league_size = mypars->pop_size * mypars->num_of_runs;
        Kokkos::parallel_for (Kokkos::TeamPolicy<ExSpace> (league_size, Kokkos::AUTO() ),
                        KOKKOS_LAMBDA (member_type team_member)
        {
                // Get team and league ranks
                int tidx = team_member.team_rank();
                int lidx = team_member.league_rank();

		// FIX ME Copy this genotype to local memory, maybe unnecessary, maybe parallelizable - ALS
        	float genotype[ACTUAL_GENOTYPE_LENGTH];
        	for (int i_geno = 0; i_geno<ACTUAL_GENOTYPE_LENGTH; i_geno++) {
        	        genotype[i_geno] = current.conformations(i_geno + GENOTYPE_LENGTH_IN_GLOBMEM*lidx);
        	}

		// Get the current energy for each run
		float energy = kokkos_calc_energy(team_member, docking_params,conform, rotlist, intracontrib, interintra, intra, genotype);

		// Copy to global views
                if( tidx == 0 ) {
                        current.energies(lidx) = energy;
                        docking_params.evals_of_new_entities(lidx) = 1;
                }
        });
}
