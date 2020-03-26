#ifndef GENETICPARAMS_HPP
#define GENETICPARAMS_HPP

// Parameters for genetic algorithm for Kokkos implementation
struct GeneticParams
{
        float           tournament_rate;
        float           crossover_rate;
        float           mutation_rate;
        float           abs_max_dmov;
        float           abs_max_dang;

	// Constructor
	GeneticParams(const Dockpars* mypars)
	{
		// Notice: tournament_rate, crossover_rate, mutation_rate
		// were scaled down to [0,1] in host to reduce number of operations in device
		tournament_rate = mypars->tournament_rate/100.0f;
		crossover_rate  = mypars->crossover_rate/100.0f;
		mutation_rate   = mypars->mutation_rate/100.f;
		abs_max_dang    = mypars->abs_max_dang;
		abs_max_dmov    = mypars->abs_max_dmov;
	}
};

#endif
