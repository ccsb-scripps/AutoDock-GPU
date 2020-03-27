#ifndef GENOTYPE_FUNCS_HPP
#define GENOTYPE_FUNCS_HPP

#include "common_typedefs.hpp"

// Loops should be over docking_params.num_of_genes not ACTUAL_GENOTYPE_LENGTH - ALS

// Perhaps these could be replaced with kokkos deep_copies, however it may require
// something sophisticated that isnt worth it unless the speedup is large

// global to local copy
template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, Genotype genotype, const Generation<Device>& generation, int which_pop)
{
	int offset = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop;
	Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, ACTUAL_GENOTYPE_LENGTH),
        		[=] (int& idx) {
        	genotype[idx] = generation.conformations(offset + idx);
        });
}

// local to global copy
template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, const Generation<Device>& generation, int which_pop, Genotype genotype)
{
        int offset = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop;
        Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, ACTUAL_GENOTYPE_LENGTH),
                        [=] (int& idx) {
                generation.conformations(offset + idx) = genotype[idx];
        });
}

// local to local copy - note, not a template because Device isnt present. May need to change - ALS
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, Genotype genotype_copy, Genotype genotype)
{
        Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, ACTUAL_GENOTYPE_LENGTH),
                        [=] (int& idx) {
                genotype_copy(idx) = genotype[idx];
        });
}

// global to global copy
template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, const Generation<Device>& generation_copy, int which_pop_copy, const Generation<Device>& generation, int which_pop)
{
        int offset = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop;
	int offset_copy = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop_copy;
        Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, ACTUAL_GENOTYPE_LENGTH),
                        [=] (int& idx) {
                generation_copy.conformations(offset_copy + idx) = generation.conformations(offset + idx);
        });
}

#endif
