#ifndef GENOTYPE_FUNCS_HPP
#define GENOTYPE_FUNCS_HPP

#include "common_typedefs.hpp"

template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, Genotype genotype, const Generation<Device>& generation, int which_pop)
{
	int offset = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop;
	Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, ACTUAL_GENOTYPE_LENGTH),
        		[=] (int& idx) {
        	genotype[idx] = generation.conformations(offset + idx);
        });
}

template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, const Generation<Device>& generation, int which_pop, Genotype genotype)
{
        int offset = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop;
        Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, ACTUAL_GENOTYPE_LENGTH),
                        [=] (int& idx) {
                generation.conformations(offset + idx) = genotype[idx];
        });
}

template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, Genotype genotype_copy, Genotype genotype)
{
        Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, ACTUAL_GENOTYPE_LENGTH),
                        [=] (int& idx) {
                genotype_copy(idx) = genotype[idx];
        });
}

#endif
