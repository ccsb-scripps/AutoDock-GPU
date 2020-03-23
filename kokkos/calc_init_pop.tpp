// TODO - templatize ExSpace - ALS
template<class Device>
void kokkos_calc_init_pop(Dockpars* mypars,DockingParams<Device>& docking_params,Conform<Device>& conform, RotList<Device>& rotlist, IntraContrib<Device>& intracontrib, InterIntra<Device>& interintra, Intra<Device>& intra)
{
	// Outer loop over mypars->pop_size * mypars->num_of_runs
        int league_size = mypars->pop_size * mypars->num_of_runs;
        typedef Kokkos::TeamPolicy<ExSpace>::member_type member_type;
        Kokkos::parallel_for (Kokkos::TeamPolicy<ExSpace> (league_size, Kokkos::AUTO() ),
                        KOKKOS_LAMBDA (member_type team_member)
        {
                // Get team and league ranks
                int tidx = team_member.team_rank();
                int lidx = team_member.league_rank();

                // Determine which run this team is doing
                int run_id;
                if (tidx == 0) {
                        run_id = lidx/docking_params.pop_size;
                }

                // Copy this genotype to local memory, maybe unnecessary - ALS
                float genotype[ACTUAL_GENOTYPE_LENGTH];
                for (int i_geno = 0; i_geno<ACTUAL_GENOTYPE_LENGTH; i_geno++) {
                        genotype[i_geno] = docking_params.conformations_current
                                                (i_geno + GENOTYPE_LENGTH_IN_GLOBMEM*lidx);
                }

                team_member.team_barrier();

                // This section was gpu_calc_energy. Break it out again later for reuse in other kernels - ALS

                // GETTING ATOMIC POSITIONS
                // ALS - This view needs to be in scratch space FIX ME
                Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords("calc_coords");
                Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, (int)(docking_params.num_of_atoms)),
                [=] (int& idx) {
                        kokkos_get_atom_pos(idx, conform, calc_coords);
                });

                // CALCULATING ATOMIC POSITIONS AFTER ROTATIONS
                // General rotation moving vector
                float4struct genrot_movingvec;
                genrot_movingvec.x = genotype[0];
                genrot_movingvec.y = genotype[1];
                genrot_movingvec.z = genotype[2];
                genrot_movingvec.w = 0.0;

		// Convert orientation genes from sex. to radians
                float phi         = genotype[3] * DEG_TO_RAD;
                float theta       = genotype[4] * DEG_TO_RAD;
                float genrotangle = genotype[5] * DEG_TO_RAD;
                float4struct genrot_unitvec;
                float sin_angle = sin(theta);
                float s2 = sin(genrotangle*0.5f);
                genrot_unitvec.x = s2*sin_angle*cos(phi);
                genrot_unitvec.y = s2*sin_angle*sin(phi);
                genrot_unitvec.z = s2*cos(theta);
                genrot_unitvec.w = cos(genrotangle*0.5f);

                team_member.team_barrier();

                // FIX ME This will break once multi-threading - ALS
                // Loop over the rot bond list and carry out all the rotations
                Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, docking_params.rotbondlist_length),
                [=] (int& idx) {
                        kokkos_rotate_atoms(idx, conform, rotlist, run_id, genotype, genrot_movingvec, genrot_unitvec, calc_coords);
                });

                team_member.team_barrier();

                // CALCULATING INTERMOLECULAR ENERGY
                float energy_inter;
                // loop over atoms
                Kokkos::parallel_reduce (Kokkos::TeamThreadRange (team_member, (int)(docking_params.num_of_atoms)),
                [=] (int& idx, float& l_energy_inter) {
                        l_energy_inter += kokkos_calc_intermolecular_energy(idx, docking_params, interintra, calc_coords);
                }, energy_inter);

                // CALCULATING INTRAMOLECULAR ENERGY
                float energy_intra;
                // loop over intraE contributors
                Kokkos::parallel_reduce (Kokkos::TeamThreadRange (team_member, docking_params.num_of_intraE_contributors),
                [=] (int& idx, float& l_energy_intra) {
                        l_energy_intra += kokkos_calc_intramolecular_energy(idx, docking_params, intracontrib, interintra, intra, calc_coords);
                }, energy_intra);

                team_member.team_barrier();

                // End of gpu_calc_energy

		// Copy to global views
                if( tidx == 0 ) {
                        docking_params.energies_current(lidx) = energy_inter + energy_intra;
                        docking_params.evals_of_new_entities(lidx) = 1;
                }
        });
}
