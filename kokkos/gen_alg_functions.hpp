#ifndef GEN_ALG_FUNCTIONS_HPP
#define GEN_ALG_FUNCTIONS_HPP

template<class Device>
KOKKOS_INLINE_FUNCTION void perform_elitist_selection(const member_type& team_member, const DockingParams<Device>& docking_params);

#include "gen_alg_functions.tpp"

#endif
