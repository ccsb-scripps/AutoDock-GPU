#ifndef GENERATION_HPP
#define GENERATION_HPP
// Conformations and energy data of each generation
template<class Device>
struct Generation
{
        Kokkos::View<float*,Device> conformations;
        Kokkos::View<float*,Device> energies;

	// Constructor with no initial values (just zero)
	Generation(const int gen_size)
		: conformations("conformations", gen_size * GENOTYPE_LENGTH_IN_GLOBMEM),
		  energies("energies",           gen_size) {}
};

#endif
