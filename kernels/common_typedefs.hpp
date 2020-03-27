#ifndef COMMON_TYPEDEFS_HPP
#define COMMON_TYPEDEFS_HPP

#include "float4struct.hpp"

// View type on scratch memory that is used in energy and gradient calculations
// Coordinates of all atoms
typedef Kokkos::View<float4struct[MAX_NUM_OF_ATOMS],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Coordinates;

// Genotype
typedef Kokkos::View<float[ACTUAL_GENOTYPE_LENGTH],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Genotype;

// Identical to Genotype, but for auxiliary arrays (e.g. gradient) that arent technically genotypes themselves. To avoid confusion, shouldnt be labeled as a genotype
typedef Kokkos::View<float[ACTUAL_GENOTYPE_LENGTH],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> GenotypeAux;

#endif
