#ifndef CALC_INIT_POP_HPP
#define CALC_INIT_POP_HPP

template<class Device>
void kokkos_calc_init_pop(Generation<Device>& current, Dockpars* mypars,DockingParams<Device>& docking_params,Conform<Device>& conform, RotList<Device>& rotlist, IntraContrib<Device>& intracontrib, InterIntra<Device>& interintra, Intra<Device>& intra);

#include "calc_init_pop.tpp"

#endif
