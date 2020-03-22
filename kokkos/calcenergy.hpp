#ifndef CALCENERGY_HPP
#define CALCENERGY_HPP

template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_get_atom_pos(const int atom_id, const Conform<Device>& conform, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_rotate_atoms(const int rotation_counter, const Conform<Device>& conform, const RotList<Device>& rotlist, const int run_id, const float* genotype, const float4struct& genrot_movingvec, const float4struct& genrot_unitvec, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);

#include "calcenergy.tpp"

#endif
