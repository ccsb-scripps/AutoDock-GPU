// TODO - templatize ExSpace - ALS
template<class Device>
void sum_evals(Dockpars* mypars,DockingParams<Device>& docking_params,Kokkos::View<int*,DeviceType> evals_of_runs)
{
        // Outer loop over mypars->num_of_runs
        int league_size = mypars->num_of_runs;
        Kokkos::parallel_for (Kokkos::TeamPolicy<ExSpace> (league_size, NUM_OF_THREADS_PER_BLOCK ),
                        KOKKOS_LAMBDA (member_type team_member)
        {
                // Get team and league ranks
                int tidx = team_member.team_rank();
                int lidx = team_member.league_rank();

                // Reduce new_entities
                int sum_evals;
                Kokkos::parallel_reduce (Kokkos::TeamThreadRange (team_member, docking_params.pop_size),
                [=] (int& idx, int& l_evals) {
                        l_evals += docking_params.evals_of_new_entities(lidx*docking_params.pop_size + idx);
                }, sum_evals);

                team_member.team_barrier();

                // Add to global view
                if( tidx == 0 ) {
                        evals_of_runs(lidx) += sum_evals;
                }
        });
}
