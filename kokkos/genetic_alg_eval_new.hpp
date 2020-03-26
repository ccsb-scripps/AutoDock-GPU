#ifndef GENETIC_ALG_EVAL_NEW_HPP
#define GENETIC_ALG_EVAL_NEW_HPP

template<class Device>
void kokkos_gen_alg_eval_new(Generation<Device>& current, Generation<Device>& next, Dockpars* mypars,DockingParams<Device>& docking_params,GeneticParams& genetic_params,Constants<Device>& consts);

#include "genetic_alg_eval_new.tpp"

#endif
