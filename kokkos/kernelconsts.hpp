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

        // Copy from a host version
        void deep_copy(InterIntra<HostType> interintra_h)
	{
		Kokkos::deep_copy(atom_charges_const,interintra_h.atom_charges_const);
		Kokkos::deep_copy(atom_types_const,interintra_h.atom_types_const);
	};
};

template<class Device>
struct IntraContrib
{
        Kokkos::View<char[3*MAX_INTRAE_CONTRIBUTORS],Device>  intraE_contributors_const;

	IntraContrib() : intraE_contributors_const("intraE_contributors_const") {};

	// Copy from a host version
        void deep_copy(IntraContrib<HostType> intracontrib_h)
        {
                Kokkos::deep_copy(intraE_contributors_const,intracontrib_h.intraE_contributors_const);
        };
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

	// Copy from a host version
        void deep_copy(Intra<HostType> intra_h)
        {       
                Kokkos::deep_copy(reqm_const, intra_h.reqm_const);
		Kokkos::deep_copy(atom1_types_reqm_const, intra_h.atom1_types_reqm_const);
                Kokkos::deep_copy(atom2_types_reqm_const, intra_h.atom2_types_reqm_const);
                Kokkos::deep_copy(VWpars_AC_const, intra_h.VWpars_AC_const);
                Kokkos::deep_copy(VWpars_BD_const, intra_h.VWpars_BD_const);
                Kokkos::deep_copy(dspars_S_const, intra_h.dspars_S_const);
                Kokkos::deep_copy(dspars_V_const, intra_h.dspars_V_const);
        };
};

template<class Device>
struct RotList
{
       Kokkos::View<int[MAX_NUM_OF_ROTATIONS],Device> rotlist_const;

       RotList() : rotlist_const("rotlist_const") {};

	// Copy from a host version
        void deep_copy(RotList<HostType> rot_list_h)
        {       
                Kokkos::deep_copy(rotlist_const,rot_list_h.rotlist_const);
        };
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

	// Copy from a host version
        void deep_copy(Conform<HostType> conform_h)
        {
                Kokkos::deep_copy(ref_coords_const, conform_h.ref_coords_const);
		Kokkos::deep_copy(rotbonds_moving_vectors_const, conform_h.rotbonds_moving_vectors_const);
		Kokkos::deep_copy(rotbonds_unit_vectors_const, conform_h.rotbonds_unit_vectors_const);
		Kokkos::deep_copy(ref_orientation_quats_const, conform_h.ref_orientation_quats_const);
        };
};

template<class Device>
struct Grads
{
        // Added for calculating torsion-related gradients.
        // Passing list of rotbond-atoms ids to the GPU.
        // Contains the same information as processligand.h/Liganddata->rotbonds 
        Kokkos::View<int[2*MAX_NUM_OF_ROTBONDS],Device> rotbonds;

        // Contains the same information as processligand.h/Liganddata->atom_rotbonds
        // "atom_rotbonds": array that contains the rotatable bonds - atoms assignment.
        // If the element atom_rotbonds[atom index][rotatable bond index] is equal to 1,
        // it means,that the atom must be rotated if the bond rotates. A 0 means the opposite.
        Kokkos::View<int[MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS],Device> rotbonds_atoms;
        Kokkos::View<int[MAX_NUM_OF_ROTBONDS],Device> num_rotating_atoms_per_rotbond;

        Grads() : rotbonds("rotbonds"),
       		  rotbonds_atoms("rotbonds_atoms"),
		  num_rotating_atoms_per_rotbond("num_rotating_atoms_per_rotbond") {};

        // Copy from a host version
        void deep_copy(Grads<HostType> grads_h)
        {
                Kokkos::deep_copy(rotbonds,grads_h.rotbonds);
                Kokkos::deep_copy(rotbonds_atoms,grads_h.rotbonds_atoms);
                Kokkos::deep_copy(num_rotating_atoms_per_rotbond,grads_h.num_rotating_atoms_per_rotbond);
        };
};

#endif
