#ifndef SUM_EVALS_HPP
#define SUM_EVALS_HPP

template<class Device>
void kokkos_sum_evals(Dockpars* mypars,DockingParams<Device>& docking_params,Kokkos::View<int*,DeviceType> evals_of_runs);

#include "sum_evals.tpp"

#endif
