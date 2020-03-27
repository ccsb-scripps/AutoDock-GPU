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

// Array of length team_size for use in perform_elitist_selection
typedef Kokkos::View<float[NUM_OF_THREADS_PER_BLOCK],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TeamFloat;
typedef Kokkos::View<int[NUM_OF_THREADS_PER_BLOCK],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TeamInt;

// Arrays of different fixed sizes (maybe unnecessary but fixed probably performs better so use it if length is known at compile time)
typedef Kokkos::View<int[2],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TwoInt;
typedef Kokkos::View<float[10],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TenFloat;
typedef Kokkos::View<int[4],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> FourInt;
typedef Kokkos::View<float[4],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> FourFloat;

#endif
