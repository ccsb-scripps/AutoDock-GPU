#ifndef KERNELCONSTS_HPP
#define KERNELCONSTS_HPP

//#include "defines.h"

// Constants used on the device for Kokkos implementation
template<class Device>
struct InterIntra
{
       Kokkos::View<float[MAX_NUM_OF_ATOMS],Device> atom_charges_const;
       Kokkos::View<char[MAX_NUM_OF_ATOMS],Device>  atom_types_const;

       InterIntra() : atom_charges_const("atom_charges_const"),
		      atom_types_const("atom_types_const") {};
};

template<class Device>
struct IntraContrib
{
       Kokkos::View<char[3*MAX_INTRAE_CONTRIBUTORS],Device>  intraE_contributors_const;

       IntraContrib() : intraE_contributors_const("intraE_contributors_const") {};
};

template<class Device>
struct Intra
{
       Kokkos::View<float[2*ATYPE_NUM],Device> reqm_const; // 1st ATYPE_NUM entries = vdW, 2nd ATYPE_NUM entries = hbond
       Kokkos::View<unsigned int[ATYPE_NUM],Device> atom1_types_reqm_const;
       Kokkos::View<unsigned int[ATYPE_NUM],Device> atom2_types_reqm_const;
       Kokkos::View<float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES],Device> VWpars_AC_const;
       Kokkos::View<float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES],Device> VWpars_BD_const;
       Kokkos::View<float[MAX_NUM_OF_ATYPES],Device> dspars_S_const;
       Kokkos::View<float[MAX_NUM_OF_ATYPES],Device> dspars_V_const;

       Intra() : reqm_const("reqm_const"),
		 atom1_types_reqm_const("atom1_types_reqm_const"),
		 atom2_types_reqm_const("atom2_types_reqm_const"),
		 VWpars_AC_const("VWpars_AC_const"),
		 VWpars_BD_const("VWpars_BD_const"),
		 dspars_S_const("dspars_S_const"),
		 dspars_V_const("dspars_V_const") {};
};

template<class Device>
struct RotList
{
       Kokkos::View<int[MAX_NUM_OF_ROTATIONS],Device> rotlist_const;

       RotList() : rotlist_const("rotlist_const") {};
};

template<class Device>
struct Conform
{
       Kokkos::View<float[3*MAX_NUM_OF_ATOMS],Device> ref_coords_const;
       Kokkos::View<float[3*MAX_NUM_OF_ROTBONDS],Device> rotbonds_moving_vectors_const;
       Kokkos::View<float[3*MAX_NUM_OF_ROTBONDS],Device> rotbonds_unit_vectors_const;
       Kokkos::View<float[4*MAX_NUM_OF_RUNS],Device> ref_orientation_quats_const;

       Conform() : ref_coords_const("ref_coords_const"),
		   rotbonds_moving_vectors_const("rotbonds_moving_vectors_const"),
		   rotbonds_unit_vectors_const("rotbonds_unit_vectors_const"),
		   ref_orientation_quats_const("ref_orientation_quats_const") {};
};

#endif
