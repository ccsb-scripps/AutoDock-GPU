#include "calcenergy.hpp"
#include "genotype_funcs.hpp"

// TODO - templatize ExSpace - ALS
template<class Device>
void calc_init_pop(Generation<Device>& current, Dockpars* mypars,DockingParams<Device>& docking_params,Constants<Device>& consts)
{
	// Outer loop over mypars->pop_size * mypars->num_of_runs
        int league_size = mypars->pop_size * mypars->num_of_runs;

	// Get the size of the shared memory allocation
	size_t shmem_size = Coordinates::shmem_size() + Genotype::shmem_size();
        Kokkos::parallel_for (Kokkos::TeamPolicy<ExSpace> (league_size, NUM_OF_THREADS_PER_BLOCK ).
                              set_scratch_size(KOKKOS_TEAM_SCRATCH_OPT,Kokkos::PerTeam(shmem_size)),
                              KOKKOS_LAMBDA (member_type team_member)
        {
                // Get team and league ranks
                int tidx = team_member.team_rank();
                int lidx = team_member.league_rank();

		Genotype genotype(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));
		copy_genotype(team_member, docking_params.num_of_genes, genotype, current, lidx);

		team_member.team_barrier();

		// Get the current energy for each run
		float energy = calc_energy(team_member, docking_params, consts, genotype);

		// Copy to global views
                if( tidx == 0 ) {
                        current.energies(lidx) = energy;
                        docking_params.evals_of_new_entities(lidx) = 1;
                }
        });
}
