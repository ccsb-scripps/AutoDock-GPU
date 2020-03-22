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
};

#endif
