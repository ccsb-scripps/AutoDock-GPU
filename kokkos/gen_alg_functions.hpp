#ifndef GEN_ALG_FUNCTIONS_HPP
#define GEN_ALG_FUNCTIONS_HPP

template<class Device>
KOKKOS_INLINE_FUNCTION void perform_elitist_selection(const member_type& team_member, const DockingParams<Device>& docking_params);

template<class Device>
KOKKOS_INLINE_FUNCTION void crossover(const member_type& team_member, const DockingParams<Device>& docking_params, const GeneticParams& genetic_params, const int run_id, const float* randnums, const int* parents,
                                        float* offspring_genotype);

template<class Device>
KOKKOS_INLINE_FUNCTION void mutation(const member_type& team_member, const DockingParams<Device>& docking_params, const GeneticParams& genetic_params,
                                     float* offspring_genotype);

#include "gen_alg_functions.tpp"

#endif
