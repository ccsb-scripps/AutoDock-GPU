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

	// Constructor with initial conformations
        Generation(const int gen_size, float* cpu_init_populations)
                : conformations("conformations", gen_size * GENOTYPE_LENGTH_IN_GLOBMEM),
                  energies("energies",           gen_size)
        {
                // Note kokkos views are initialized to zero by default

                // Copy in initial value
                // First wrap the C style arrays with an unmanaged kokkos view, then deep copy to the device
                FloatView1D init_pop_view(cpu_init_populations, conformations.extent(0));
                Kokkos::deep_copy(conformations, init_pop_view);
        }
};

#endif
