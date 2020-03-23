#ifndef GRADIENT_MINAD_HPP
#define GRADIENT_MINAD_HPP

template<class Device>
void kokkos_gradient_minAD(Dockpars* mypars,DockingParams<Device>& docking_params,Conform<Device>& conform, RotList<Device>& rotlist, IntraContrib<Device>& intracontrib, InterIntra<Device>& interintra, Intra<Device>& intra);

#include "gradient_minAD.tpp"

#endif
