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
        Kokkos::View<float*,Device> conformations_next;
        Kokkos::View<float*,Device> energies_next;
        Kokkos::View<int*,Device> evals_of_new_entities;
        Kokkos::View<unsigned int*,Device> prng_states;
        int             pop_size;
	float           smooth;
	float           qasp;

// COMMENTED PARAMETERS MAY BE ADDED LATER OR ELSEWHERE - ALS 
/*        int             num_of_genes;
        float           lsearch_rate;
        unsigned int    num_of_lsentities;
        float           rho_lower_bound;
        float           base_dmov_mul_sqrt3;
        float           base_dang_mul_sqrt3;
        unsigned int    cons_limit;
        unsigned int    max_num_of_iters;
*/

	// Constructor
	DockingParams(const Liganddata& myligand_reference, const Gridinfo* mygrid, const Dockpars* mypars, float* cpu_floatgrids, float* cpu_init_populations)
		: fgrids("fgrids", 4 * (mygrid->num_of_atypes+2) * (mygrid->size_xyz[0]) * (mygrid->size_xyz[1]) * (mygrid->size_xyz[2])),
		  conformations_current("conformations_current", mypars->pop_size * mypars->num_of_runs * GENOTYPE_LENGTH_IN_GLOBMEM),
		  energies_current("energies_current", mypars->pop_size * mypars->num_of_runs),
		  evals_of_new_entities("evals_of_new_entities", mypars->pop_size * mypars->num_of_runs)
	{
		// Copy in scalars
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

		// Note kokkos views are initialized to zero by default

                // Copy arrays
		// First wrap the C style arrays with an unmanaged kokkos view, then deep copy to the device
		typedef Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> FloatView1D;
		FloatView1D fgrids_view(cpu_floatgrids, fgrids.extent(0));
		Kokkos::deep_copy(fgrids, fgrids_view);

                FloatView1D init_pop_view(cpu_init_populations, conformations_current.extent(0));
                Kokkos::deep_copy(conformations_current, init_pop_view);
	}
};

#endif
