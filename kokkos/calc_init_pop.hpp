#ifndef CALC_INIT_POP_HPP
#define CALC_INIT_POP_HPP

template<class Device>
void calc_init_pop(Generation<Device>& current, Dockpars* mypars,DockingParams<Device>& docking_params,Constants<Device>& consts);

#include "calc_init_pop.tpp"

#endif
