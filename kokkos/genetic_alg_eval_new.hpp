#ifndef GENETIC_ALG_EVAL_NEW_HPP
#define GENETIC_ALG_EVAL_NEW_HPP

template<class Device>
void kokkos_gen_alg_eval_new(Dockpars* mypars,DockingParams<Device>& docking_params,Conform<Device>& conform, RotList<Device>& rotlist, IntraContrib<Device>& intracontrib, InterIntra<Device>& interintra, Intra<Device>& intra);

#include "genetic_alg_eval_new.tpp"

#endif
