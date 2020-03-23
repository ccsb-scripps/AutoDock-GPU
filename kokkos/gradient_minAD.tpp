#include "calcenergy.hpp"

// TODO - templatize ExSpace - ALS
template<class Device>
void kokkos_gradient_minAD(Dockpars* mypars,DockingParams<Device>& docking_params,Conform<Device>& conform, RotList<Device>& rotlist, IntraContrib<Device>& intracontrib, InterIntra<Device>& interintra, Intra<Device>& intra, Grads<Device>& grads, AxisCorrection<Device>& axis_correction)
{
	// Outer loop over mypars->pop_size * mypars->num_of_runs
        int league_size = mypars->pop_size * mypars->num_of_runs;
        Kokkos::parallel_for (Kokkos::TeamPolicy<ExSpace> (league_size, Kokkos::AUTO() ),
                        KOKKOS_LAMBDA (member_type team_member)
        {
                // Get team and league ranks
                int tidx = team_member.team_rank();
                int lidx = team_member.league_rank();

		// Get the current energy for each run
		float energy = kokkos_calc_energy(team_member, docking_params,conform, rotlist, intracontrib, interintra, intra);

		// Copy to global views
                if( tidx == 0 ) {
                        docking_params.energies_current(lidx) = energy;
                        docking_params.evals_of_new_entities(lidx) = 1;
                }
        });
}
