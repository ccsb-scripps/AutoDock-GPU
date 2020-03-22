#ifndef DOCKINGPARAMS_HPP
#define DOCKINGPARAMS_HPP
// Docking parameters for Kokkos implementation
template<class Device>
struct DockingParams
{
        char            num_of_atoms;
        char            num_of_atypes;
        int             num_of_intraE_contributors;
        char            gridsize_x;
        char            gridsize_y;
        char            gridsize_z;
	unsigned int    g2;
	unsigned int    g3;
	float           grid_spacing;
	Kokkos::View<float*,Device> fgrids;
        int             rotbondlist_length;
        float           coeff_elec;
        float           coeff_desolv;
        Kokkos::View<float*,Device> conformations_current;
        Kokkos::View<float*,Device> energies_current;

// COMMENTED PARAMETERS MAY BE ADDED LATER OR ELSEWHERE - ALS	
//        float*          conformations_next;
//        float*          energies_next;
        Kokkos::View<int*,Device> evals_of_new_entities;
//        unsigned int*   prng_states;
        int             pop_size;
/*        int             num_of_genes;
        float           tournament_rate;
        float           crossover_rate;
        float           mutation_rate;
        float           abs_max_dmov;
        float           abs_max_dang;
        float           lsearch_rate;
*/
        float           smooth;
/*        unsigned int    num_of_lsentities;
        float           rho_lower_bound;
        float           base_dmov_mul_sqrt3;
        float           base_dang_mul_sqrt3;
        unsigned int    cons_limit;
        unsigned int    max_num_of_iters;
*/
        float           qasp;

	// Constructor
	DockingParams(const Liganddata& myligand_reference, const Gridinfo* mygrid, const Dockpars* mypars, const float* cpu_floatgrids, const float* cpu_init_populations)
		: fgrids("fgrids", 4 * (mygrid->num_of_atypes+2) * (mygrid->size_xyz[0]) * (mygrid->size_xyz[1]) * (mygrid->size_xyz[2])),
		  conformations_current("conformations_current", mypars->pop_size * mypars->num_of_runs * GENOTYPE_LENGTH_IN_GLOBMEM),
		  energies_current("energies_current", mypars->pop_size * mypars->num_of_runs),
		  evals_of_new_entities("evals_of_new_entities", mypars->pop_size * mypars->num_of_runs)
	{
		num_of_atoms  = ((char)  myligand_reference.num_of_atoms);
		num_of_atypes = ((char)  myligand_reference.num_of_atypes);
		num_of_intraE_contributors = ((int) myligand_reference.num_of_intraE_contributors);
		gridsize_x    = ((char)  mygrid->size_xyz[0]);
		gridsize_y    = ((char)  mygrid->size_xyz[1]);
		gridsize_z    = ((char)  mygrid->size_xyz[2]);
		g2 = gridsize_x * gridsize_y;
                g3 = gridsize_x * gridsize_y * gridsize_z;

		grid_spacing  = ((float) mygrid->spacing);
		rotbondlist_length = ((int) NUM_OF_THREADS_PER_BLOCK*(myligand_reference.num_of_rotcyc));
		coeff_elec    = ((float) mypars->coeffs.scaled_AD4_coeff_elec);
		coeff_desolv  = ((float) mypars->coeffs.AD4_coeff_desolv);
		pop_size      = mypars->pop_size;
		qasp            = mypars->qasp;
		smooth          = mypars->smooth;
	}
};

#endif
