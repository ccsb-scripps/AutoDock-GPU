#include "calcenergrad.hpp"
#include "ada_functions.hpp"
#include "random.hpp"

// TODO - templatize ExSpace - ALS
template<class Device>
void gradient_minAD(Generation<Device>& next, Dockpars* mypars,DockingParams<Device>& docking_params,Constants<Device>& consts)
{
	// Outer loop
        int league_size = docking_params.num_of_lsentities * mypars->num_of_runs;

	// Get the size of the shared memory allocation
        size_t shmem_size = Coordinates::shmem_size() + 2*Genotype::shmem_size() + 3*GenotypeAux::shmem_size();
	Kokkos::parallel_for (Kokkos::TeamPolicy<ExSpace> (league_size, NUM_OF_THREADS_PER_BLOCK ).
                              set_scratch_size(KOKKOS_TEAM_SCRATCH_OPT,Kokkos::PerTeam(shmem_size)),
                        KOKKOS_LAMBDA (member_type team_member)
        {
                // Get team and league ranks
                int tidx = team_member.team_rank();
                int lidx = team_member.league_rank();

		int gpop_idx;
		int run_id;
		int entity_id;

		float energy;

		if (tidx == 0)
		{
			run_id = lidx / docking_params.num_of_lsentities;
			entity_id = lidx - run_id * docking_params.num_of_lsentities; // modulus in different form

			// Since entity 0 is the best one due to elitism,
			// it should be subjected to random selection
			if (entity_id == 0) {
				// If entity 0 is not selected according to LS-rate,
				// choosing another entity
				if (100.0f*rand_float(team_member, docking_params) > docking_params.lsearch_rate) {
					entity_id = docking_params.num_of_lsentities; // AT - Should this be (uint)(dockpars_pop_size * gpu_randf(dockpars_prng_states))?
				}
			}

			gpop_idx = run_id*docking_params.pop_size+entity_id; // global population index

			// Get current energy
			energy = next.energies(gpop_idx);

#if defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
			printf("\n-------> Start of ADADELTA minimization cycle\n");
			printf("%20s %6u\n", "run_id: ", run_id);
			printf("%20s %6u\n", "entity_id: ", entity_id);
			printf("\n%20s \n", "LGA genotype: ");
			printf("%20s %.6f\n", "initial energy: ", energy);
#endif
		}

		team_member.team_barrier();

		// FIX ME Copy this genotype to local memory, maybe unnecessary, maybe parallelizable - ALS
                Genotype genotype(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));
                for (int i_geno = 0; i_geno<docking_params.num_of_genes; i_geno++) {
                        genotype[i_geno] = next.conformations(i_geno + GENOTYPE_LENGTH_IN_GLOBMEM*gpop_idx);
                }

		team_member.team_barrier();

		// Initializing best genotype and energy
		Genotype best_genotype(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));
		for(uint i = tidx; i < docking_params.num_of_genes; i+= team_member.team_size()) {
			best_genotype [i] = genotype [i];
		}
		float best_energy;
		if (tidx == 0) {
			best_energy = INFINITY; // Why isnt this set to energy? - ALS
		}

		// Initializing variable arrays for gradient descent
		GenotypeAux square_gradient(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));
		GenotypeAux square_delta(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));
		for(uint i = tidx; i < ACTUAL_GENOTYPE_LENGTH; i+= team_member.team_size()) {
                        square_gradient[i]=0;
			square_delta[i]=0;
                }


		// Initialize iteration controls
		unsigned int iteration_cnt;
		if (tidx == 0) {
			iteration_cnt  = 0;
		}
#ifdef AD_RHO_CRITERION
		float rho;
		int   cons_succ;
		int   cons_fail;
		if (tidx == 0) {
			rho = 1.0f; 
			cons_succ = 0;
			cons_fail = 0;
		}
#endif

		team_member.team_barrier();

		// Declare/allocate coordinates for internal use by calc_energrad only. Must be outside of loop since there is
		// no way to de/reallocate things in Kokkos team scratch
		Coordinates calc_coords(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));

		// Perform adadelta iterations on gradient
		GenotypeAux gradient(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));
        // The termination criteria is based on
        // a maximum number of iterations, and
        // the minimum step size allowed for single-floating point numbers
        // (IEEE-754 single float has a precision of about 6 decimal digits)
        do {
		// Calculating energy & gradient
                calc_energrad(team_member, docking_params, genotype, consts, calc_coords,
                                     energy, gradient);

		team_member.team_barrier();

		// we need to be careful not to change best_energy until we had a chance to update the whole array
		if (energy < best_energy){
			for(uint i = tidx;
				 i < docking_params.num_of_genes;
				 i+= team_member.team_size()) {
				 best_genotype[i] = genotype[i];
			}
		}

		team_member.team_barrier();

		// Update genotype based on gradient
                genotype_gradient_descent(team_member, docking_params, gradient, square_gradient, square_delta, genotype);

                team_member.team_barrier();

		// Iteration controls
		// Updating for rho criterion
#ifdef AD_RHO_CRITERION
		if (tidx == 0) {
			if (energy < best_energy) {
                                cons_succ++;
                                cons_fail = 0;
                        } else {
                                cons_succ = 0;
                                cons_fail++;
                        }

                        if (cons_succ >= 4) {
                                rho *= LS_EXP_FACTOR;
                                cons_succ = 0;
                        } else if (cons_fail >= 4) {
				rho *= LS_CONT_FACTOR;
				cons_fail = 0;
                        }
		}
#endif
		// Updating number of ADADELTA iterations (energy evaluations)
		if (tidx == 0) {
#if defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
			printf("%-15s %-3u ", "# ADADELTA iteration: ", iteration_cnt);
                        printf("%20s %10.6f\n", "new.energy: ", energy);
#endif
			iteration_cnt = iteration_cnt + 1;
			if (energy < best_energy) {
				best_energy = energy;
			}
                }

                team_member.team_barrier(); // making sure that iteration_cnt is up-to-date

#ifdef AD_RHO_CRITERION
        } while ((iteration_cnt < docking_params.max_num_of_iters)  && (rho > 0.01));
#else
        } while (iteration_cnt < docking_params.max_num_of_iters);
#endif
	// Descent complete
	// -----------------------------------------------------------------------------

		// Modulo torsion angles
		for (uint gene_counter = tidx+3;
			  gene_counter < docking_params.num_of_genes;
			  gene_counter+= team_member.team_size()) {
                        while (best_genotype[gene_counter] >= 360.0f) { best_genotype[gene_counter] -= 360.0f; }
                        while (best_genotype[gene_counter] < 0.0f   ) { best_genotype[gene_counter] += 360.0f; }
		}

		team_member.team_barrier();

                // Copy to global views
                if( tidx == 0 ) {
                        next.energies(gpop_idx) = best_energy;
                        docking_params.evals_of_new_entities(gpop_idx) += iteration_cnt;
                }

                // FIX ME Copying new offspring to next generation, maybe parallelizable - ALS
                for (int i_geno = 0; i_geno<docking_params.num_of_genes; i_geno++) {
                        next.conformations(i_geno + GENOTYPE_LENGTH_IN_GLOBMEM*gpop_idx) = best_genotype[i_geno];
                }

                team_member.team_barrier();
        });
}
