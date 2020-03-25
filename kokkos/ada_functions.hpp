#ifndef ADA_FUNCTIONS_HPP
#define ADA_FUNCTIONS_HPP

template<class Device>
KOKKOS_INLINE_FUNCTION void genotype_gradient_descent(const member_type& team_member, const DockingParams<Device>& docking_params, float* gradient, float* square_gradient, float* square_delta, float* genotype);

#include "ada_functions.tpp"

#endif
