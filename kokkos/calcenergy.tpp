/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

AutoDock is a Trade Mark of the Scripps Research Institute.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

*/

KOKKOS_INLINE_FUNCTION float4struct kokkos_quaternion_cross(const float4struct a, const float4struct b)
{
	float4struct result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;
	result.w = 0.0f;
	return result;
}

KOKKOS_INLINE_FUNCTION float4struct kokkos_quaternion_multiply(const float4struct a, const float4struct b)
{
	float4struct result;
	result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
	result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
	result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
	result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
	return result;
}

KOKKOS_INLINE_FUNCTION float4struct kokkos_quaternion_rotate(const float4struct v, const float4struct rot)
{
	float4struct qcross = kokkos_quaternion_cross(rot,v);
	float4struct qcross2 = kokkos_quaternion_cross(rot,qcross);
	float4struct result;
	result.x = v.x + 2.0f*qcross.x*rot.w + 2.0f*qcross2.x;
	result.y = v.y + 2.0f*qcross.y*rot.w + 2.0f*qcross2.y;
	result.z = v.z + 2.0f*qcross.z*rot.w + 2.0f*qcross2.z;
	result.w = v.w + 2.0f*qcross.w*rot.w + 2.0f*qcross2.w;
	return result;
}

// trilinear interpolation
template<class Device>
KOKKOS_INLINE_FUNCTION float kokkos_trilinear_interp(Kokkos::View<float*,Device> fgrids, const int i, const float* weights)
{
	return (fgrids(i+idx_000)*weights[idx_000] +
		fgrids(i+idx_010)*weights[idx_010] +
		fgrids(i+idx_001)*weights[idx_001] +
		fgrids(i+idx_011)*weights[idx_011] +
		fgrids(i+idx_100)*weights[idx_100] +
		fgrids(i+idx_110)*weights[idx_110] +
		fgrids(i+idx_101)*weights[idx_101] +
		fgrids(i+idx_111)*weights[idx_111]);
}

// GETTING ATOMIC POSITIONS
template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_get_atom_pos(const int atom_id, const Conform<Device>& conform, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords)
{
        // Initializing gradients (forces)
        // Derived from autodockdev/maps.py
        // Initialize coordinates
        calc_coords(atom_id).x = conform.ref_coords_const[3*atom_id];
        calc_coords(atom_id).y = conform.ref_coords_const[3*atom_id+1];
        calc_coords(atom_id).z = conform.ref_coords_const[3*atom_id+2];
	calc_coords(atom_id).w = 0.0f;
}

// CALCULATING ATOMIC POSITIONS AFTER ROTATIONS
template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_rotate_atoms(const int rotation_counter, const Conform<Device>& conform, const RotList<Device>& rotlist, const int run_id, const float* genotype, const float4struct& genrot_movingvec, const float4struct& genrot_unitvec, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords)
{
        int rotation_list_element = rotlist.rotlist_const(rotation_counter);
                
        if ((rotation_list_element & RLIST_DUMMY_MASK) == 0)    // If not dummy rotation
        {       
                uint atom_id = rotation_list_element & RLIST_ATOMID_MASK;
                        
                // Capturing atom coordinates
                float4struct atom_to_rotate = calc_coords(atom_id);
                        
                // initialize with general rotation values
                float4struct rotation_unitvec = genrot_unitvec;
                float4struct rotation_movingvec = genrot_movingvec;
                        
                if ((rotation_list_element & RLIST_GENROT_MASK) == 0) // If rotating around rotatable bond
                {       
                        uint rotbond_id = (rotation_list_element & RLIST_RBONDID_MASK) >> RLIST_RBONDID_SHIFT;
                        
                        float rotation_angle = genotype[6+rotbond_id]*DEG_TO_RAD*0.5f;
                        float s = sin(rotation_angle);
                        rotation_unitvec.x = s*conform.rotbonds_unit_vectors_const(3*rotbond_id);
                        rotation_unitvec.y = s*conform.rotbonds_unit_vectors_const(3*rotbond_id+1);
                        rotation_unitvec.z = s*conform.rotbonds_unit_vectors_const(3*rotbond_id+2);
                        rotation_unitvec.w = cos(rotation_angle);
                        rotation_movingvec.x = conform.rotbonds_moving_vectors_const(3*rotbond_id);
                        rotation_movingvec.y = conform.rotbonds_moving_vectors_const(3*rotbond_id+1);
                        rotation_movingvec.z = conform.rotbonds_moving_vectors_const(3*rotbond_id+2);
			rotation_movingvec.w = 0;
                        // Performing additionally the first movement which
                        // is needed only if rotating around rotatable bond
                        atom_to_rotate = atom_to_rotate - rotation_movingvec;
                }
                
                float4struct quatrot_left = rotation_unitvec;
                // Performing rotation
                if ((rotation_list_element & RLIST_GENROT_MASK) != 0)   // If general rotation, 
                                                                        // two rotations should be performed
                                                                        // (multiplying the quaternions)
                {       
                        // Calculating quatrot_left*ref_orientation_quats_const,
                        // which means that reference orientation rotation is the first
                        uint rid4 = 4*run_id;
			float4struct ref_orientation;
			ref_orientation.x = conform.ref_orientation_quats_const(rid4+0);
			ref_orientation.y = conform.ref_orientation_quats_const(rid4+1);
			ref_orientation.z = conform.ref_orientation_quats_const(rid4+2);
			ref_orientation.w = conform.ref_orientation_quats_const(rid4+3);
                        quatrot_left = kokkos_quaternion_multiply(quatrot_left, ref_orientation);
                }
                
                // Performing final movement and storing values
                calc_coords(atom_id) = kokkos_quaternion_rotate(atom_to_rotate,quatrot_left) + rotation_movingvec;
                
        }
}


// CALCULATING INTERMOLECULAR ENERGY
template<class Device>
KOKKOS_INLINE_FUNCTION float kokkos_calc_intermolecular_energy(const int atom_id, const DockingParams<Device>& dock_params, const InterIntra<Device>& interintra, const Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords)
{
	float partial_energy = 0.0f;

	uint atom_typeid = interintra.atom_types_const(atom_id);
	float x = calc_coords(atom_id).x;
	float y = calc_coords(atom_id).y;
	float z = calc_coords(atom_id).z;
	float q = interintra.atom_charges_const(atom_id);
	if ((x < 0) || (y < 0) || (z < 0) || (x >= dock_params.gridsize_x-1)
					  || (y >= dock_params.gridsize_y-1)
					  || (z >= dock_params.gridsize_z-1)){
		partial_energy += 16777216.0f; //100000.0f;
		return partial_energy; // get on with loop as our work here is done (we crashed into the walls)
	}
	// Getting coordinates
	uint x_low  = (uint)floor(x);
	uint y_low  = (uint)floor(y);
	uint z_low  = (uint)floor(z);

	float dx = x - x_low;
	float omdx = 1.0 - dx;
	float dy = y - y_low;
	float omdy = 1.0 - dy;
	float dz = z - z_low;
	float omdz = 1.0 - dz;

	// Calculating interpolation weights
	float weights[8];
	weights [idx_000] = omdx*omdy*omdz;
	weights [idx_100] = dx*omdy*omdz;
	weights [idx_010] = omdx*dy*omdz;
	weights [idx_110] = dx*dy*omdz;
	weights [idx_001] = omdx*omdy*dz;
	weights [idx_101] = dx*omdy*dz;
	weights [idx_011] = omdx*dy*dz;
	weights [idx_111] = dx*dy*dz;

	// Grid index at 000
	int grid_ind_000 = (x_low  + y_low*dock_params.gridsize_x  + z_low*dock_params.g2)<<2;
	unsigned long mul_tmp = atom_typeid*dock_params.g3<<2;
	// Calculating affinity energy
	partial_energy += kokkos_trilinear_interp(dock_params.fgrids, grid_ind_000+mul_tmp, weights);

	// Capturing electrostatic values
	atom_typeid = dock_params.num_of_atypes;

	mul_tmp = atom_typeid*dock_params.g3<<2;
	// Calculating electrostatic energy
	partial_energy += q * kokkos_trilinear_interp(dock_params.fgrids, grid_ind_000+mul_tmp, weights);

	// Capturing desolvation values
	atom_typeid = dock_params.num_of_atypes+1;

	mul_tmp = atom_typeid*dock_params.g3<<2;
	// Calculating desolvation energy
	partial_energy += fabs(q) * kokkos_trilinear_interp(dock_params.fgrids, grid_ind_000+mul_tmp, weights);

	return partial_energy;
}

// CALCULATING INTRAMOLECULAR ENERGY
template<class Device>
KOKKOS_INLINE_FUNCTION float kokkos_calc_intramolecular_energy(const int contributor_counter, const DockingParams<Device>& dock_params, const IntraContrib<Device>& intracontrib, const InterIntra<Device>& interintra, const Intra<Device>& intra, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords)
{
        float partial_energy = 0.0f;
        float delta_distance = 0.5f*dock_params.smooth;

	// Getting atom IDs
	uint atom1_id = intracontrib.intraE_contributors_const(3*contributor_counter);
	uint atom2_id = intracontrib.intraE_contributors_const(3*contributor_counter+1);
	uint hbond = (uint)(intracontrib.intraE_contributors_const(3*contributor_counter+2) == 1);    // evaluates to 1 in case of H-bond, 0 otherwise

	// Calculating vector components of vector going
	// from first atom's to second atom's coordinates
	float subx = calc_coords(atom1_id).x - calc_coords(atom2_id).x;
	float suby = calc_coords(atom1_id).y - calc_coords(atom2_id).y;
	float subz = calc_coords(atom1_id).z - calc_coords(atom2_id).z;

	// Calculating atomic_distance
	float atomic_distance = sqrt(subx*subx + suby*suby + subz*subz)*dock_params.grid_spacing;

	// Getting type IDs
	uint atom1_typeid = interintra.atom_types_const(atom1_id);
	uint atom2_typeid = interintra.atom_types_const(atom2_id);

	uint atom1_type_vdw_hb = intra.atom1_types_reqm_const(atom1_typeid);
	uint atom2_type_vdw_hb = intra.atom2_types_reqm_const(atom2_typeid);


	// Calculating energy contributions
	// Cuttoff1: internuclear-distance at 8A only for vdw and hbond.
	if (atomic_distance < 8.0f)
	{       
		// Getting optimum pair distance (opt_distance) from reqm and reqm_hbond
		// reqm: equilibrium internuclear separation 
		//       (sum of the vdW radii of two like atoms (A)) in the case of vdW
		// reqm_hbond: equilibrium internuclear separation
		//       (sum of the vdW radii of two like atoms (A)) in the case of hbond 
		float opt_distance = intra.reqm_const(atom1_type_vdw_hb+ATYPE_NUM*hbond) + intra.reqm_const(atom2_type_vdw_hb+ATYPE_NUM*hbond);
		
		// Getting smoothed distance
		// smoothed_distance = function(atomic_distance, opt_distance)
		float smoothed_distance = opt_distance;
		
		if (atomic_distance <= (opt_distance - delta_distance)) {
			smoothed_distance = atomic_distance + delta_distance;
		}
		if (atomic_distance >= (opt_distance + delta_distance)) {
			smoothed_distance = atomic_distance - delta_distance;
		}
		// Calculating van der Waals / hydrogen bond term
		uint idx = atom1_typeid * dock_params.num_of_atypes + atom2_typeid;
		partial_energy += (intra.VWpars_AC_const(idx) / pow(smoothed_distance,12)) -
				  (intra.VWpars_BD_const(idx) / pow(smoothed_distance,6+4*hbond));

	} // if cuttoff1 - internuclear-distance at 8A

	// Calculating energy contributions
	// Cuttoff2: internuclear-distance at 20.48A only for el and sol.
	if (atomic_distance < 20.48f)
	{
		float q1 = interintra.atom_charges_const(atom1_id);
		float q2 = interintra.atom_charges_const(atom2_id);
		float dist2 = atomic_distance*atomic_distance;
		// Calculating desolvation term
		float desolv_energy =  ((intra.dspars_S_const(atom1_typeid) +
					 dock_params.qasp*fabs(q1)) * intra.dspars_V_const(atom2_typeid) +
					(intra.dspars_S_const(atom2_typeid) +
					 dock_params.qasp*fabs(q2)) * intra.dspars_V_const(atom1_typeid)) *
					( dock_params.coeff_desolv*(12.96f-0.1063f*dist2*(1.0f-0.001947f*dist2)) /
							(12.96f+dist2*(0.4137f+dist2*(0.00357f+0.000112f*dist2))) ); // *native_exp(-0.03858025f*atomic_distance*atomic_distance);

		// Calculating electrostatic term
		float dist_shift=atomic_distance+1.261f;
		dist2=dist_shift*dist_shift;
		float diel = (1.105f / dist2) + 0.0104f;
		float es_energy = (dock_params.coeff_elec * q1 * q2) / atomic_distance;
		partial_energy += diel * es_energy + desolv_energy;
	} // if cuttoff2 - internuclear-distance at 20.48A

	// ------------------------------------------------
	// Required only for flexrings
	// Checking if this is a CG-G0 atomic pair.
	// If so, then adding energy term (E = G * distance).
	// Initial specification required NON-SMOOTHED distance.
	// This interaction is evaluated at any distance,
	// so no cuttoffs considered here!
	if (((atom1_type_vdw_hb == ATYPE_CG_IDX) && (atom2_type_vdw_hb == ATYPE_G0_IDX)) ||
	    ((atom1_type_vdw_hb == ATYPE_G0_IDX) && (atom2_type_vdw_hb == ATYPE_CG_IDX))) {
		partial_energy += G * atomic_distance;
	}
	// ------------------------------------------------


        return partial_energy;
}