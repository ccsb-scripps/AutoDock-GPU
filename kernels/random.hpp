#ifndef RANDOM_HPP
#define RANDOM_HPP

template<class Device>
KOKKOS_INLINE_FUNCTION unsigned int my_rand(const member_type& team_member, const DockingParams<Device>& docking_params);

template<class Device>
KOKKOS_INLINE_FUNCTION float rand_float(const member_type& team_member, const DockingParams<Device>& docking_params);

#include "random.tpp"

#endif
