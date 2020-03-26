#ifndef GRADIENT_MINAD_HPP
#define GRADIENT_MINAD_HPP

template<class Device>
void kokkos_gradient_minAD(Generation<Device>& next, Dockpars* mypars,DockingParams<Device>& docking_params,Constants<Device>& consts);

#include "gradient_minAD.tpp"

#endif
