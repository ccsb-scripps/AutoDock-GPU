/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/


//#define DEBUG_ENERGY_KERNEL

// No needed to be included as all kernel sources are stringified
#if 0
#include "calcenergy_basic.h"
#endif

typedef struct
{
	float atom_charges_const  [MAX_NUM_OF_ATOMS];
	int   atom_types_const    [MAX_NUM_OF_ATOMS];
	int   atom_types_map_const[MAX_NUM_OF_ATOMS];
	char  ignore_inter_const  [MAX_NUM_OF_ATOMS];
} kernelconstant_interintra;

typedef struct
{
	int intraE_contributors_const[2*MAX_INTRAE_CONTRIBUTORS];
} kernelconstant_intracontrib;

typedef struct
{
	unsigned int       atom_types_reqm_const[MAX_NUM_OF_ATYPES];
	unsigned short int VWpars_exp_const     [MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
	float              reqm_AB_const        [MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
	float              VWpars_AC_const      [MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
	float              VWpars_BD_const      [MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
	float              dspars_S_const       [MAX_NUM_OF_ATYPES];
	float              dspars_V_const       [MAX_NUM_OF_ATYPES];
} kernelconstant_intra;

typedef struct
{
	int rotlist_const     [MAX_NUM_OF_ROTATIONS];
} kernelconstant_rotlist;

typedef struct
{
	float ref_coords_const             [3*MAX_NUM_OF_ATOMS];
	float rotbonds_moving_vectors_const[3*MAX_NUM_OF_ROTBONDS];
	float rotbonds_unit_vectors_const  [3*MAX_NUM_OF_ROTBONDS];
} kernelconstant_conform;

// Magic positive integer exponent power ... -AT
// - it's in here mostly for testing/optimizing Cuda performance:
// - OpenCL's native_pwr(float, uint) is about 15%-20% faster, ...
// - but Cuda's floating point power isn't optimized for positive integers so it's much slower
inline float positive_power(float a, uint exp)
{
	// example a = 1.4, exp = 7
	float result=(exp & 1)?a:1.0f; // result = 1.4
	while(exp>>=1){ // [1] (exp>>=1) = true; exp -> 3 [2] true, exp -> 1 [3] false
		a *= a; // [1] a = 1.4^2 [2] a = 1.4^4
		result=(exp & 1)?result*a:result; // [1] result = 1.4^3 [2] result = 1.4^7
	}
	
	return result;
}

#define invpi2 1.0f/(PI_TIMES_2)

inline float fmod_pi2(float x)
{
	return x-(int)(invpi2*x)*PI_TIMES_2;
}

#define fast_acos_a  9.78056e-05f
#define fast_acos_b -0.00104588f
#define fast_acos_c  0.00418716f
#define fast_acos_d -0.00314347f
#define fast_acos_e  2.74084f
#define fast_acos_f  0.370388f
#define fast_acos_o -(fast_acos_a+fast_acos_b+fast_acos_c+fast_acos_d)

inline float fast_acos(float cosine)
{
	float x=fabs(cosine);
	float x2=x*x;
	float x3=x2*x;
	float x4=x3*x;
	float ac=(((fast_acos_o*x4+fast_acos_a)*x3+fast_acos_b)*x2+fast_acos_c)*x+fast_acos_d+
		 fast_acos_e*native_sqrt(2.0f-native_sqrt(2.0f+2.0f*x))-fast_acos_f*native_sqrt(2.0f-2.0f*x);
	return copysign(ac,cosine) + (cosine<0.0f)*PI_FLOAT;
}

inline float4 quaternion_multiply(float4 a, float4 b)
{
	float4 result = { a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, // x
			  a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, // y
			  a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, // z
			  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z }; // w
	return result;
}

inline float4 quaternion_rotate(float4 v, float4 rot)
{
	float4 result;
	
	float4 z=cross(rot,v) * 2.0f;
	result = v + z*rot.w + cross(rot,z);
	
	return result;
}

// All related pragmas are in defines.h (accesible by host and device code)

void gpu_calc_energy(
                           int     dockpars_rotbondlist_length,
                           int     dockpars_num_of_atoms,
                           int     dockpars_true_ligand_atoms,
                           int     dockpars_gridsize_x,
                           int     dockpars_gridsize_y,
                           int     dockpars_gridsize_z,
                                                                        // g1 = gridsize_x
                           uint    dockpars_gridsize_x_times_y,         // g2 = gridsize_x * gridsize_y
                           uint    dockpars_gridsize_x_times_y_times_z, // g3 = gridsize_x * gridsize_y * gridsize_z
            __global const float*  restrict dockpars_fgrids, // This is too large to be allocated in __constant
                           int     dockpars_num_of_atypes,
                           int     dockpars_num_of_map_atypes,
                           int     dockpars_num_of_intraE_contributors,
                           float   dockpars_grid_spacing,
                           float   dockpars_coeff_elec,
                           float   dockpars_elec_min_distance,
                           float   dockpars_qasp,
                           float   dockpars_coeff_desolv,
                           float   dockpars_smooth,

                           // Some OpenCL compilers don't allow declaring
                           // local variables within non-kernel functions.
                           // These local variables must be declared in a kernel,
                           // and then passed to non-kernel functions.
                   __local float*  genotype,
                   __local float*  energy,
                   __local int*    run_id,

                   __local float4* calc_coords,
                   __local float*  partial_energies,

            #if defined (DEBUG_ENERGY_KERNEL)
                   __local float*  partial_interE,
                   __local float*  partial_intraE,
            #endif
#if 0
                           bool    debug,
#endif
          __constant       kernelconstant_interintra*   kerconst_interintra,
            __global const kernelconstant_intracontrib* kerconst_intracontrib,
          __constant       kernelconstant_intra*        kerconst_intra,
          __constant       kernelconstant_rotlist*      kerconst_rotlist,
          __constant       kernelconstant_conform*      kerconst_conform
                    )
// The GPU device function calculates the energy of the entity described by genotype, dockpars and the liganddata
// arrays in constant memory and returns it in the energy parameter. The parameter run_id has to be equal to the ID
// of the run whose population includes the current entity (which can be determined with blockIdx.x), since this
// determines which reference orientation should be used.
{
	int tidx = get_local_id(0);
	partial_energies[tidx] = 0.0f;

	#if defined (DEBUG_ENERGY_KERNEL)
	partial_interE[tidx] = 0.0f;
	partial_intraE[tidx] = 0.0f;
	#endif

	// Initializing gradients (forces) 
	// Derived from autodockdev/maps.py
	for ( int atom_id = tidx;
	          atom_id < dockpars_num_of_atoms;
	          atom_id+= NUM_OF_THREADS_PER_BLOCK)
	{
		// Initialize coordinates
		calc_coords[atom_id] = (float4)(kerconst_conform->ref_coords_const[3*atom_id],
		                                kerconst_conform->ref_coords_const[3*atom_id+1],
		                                kerconst_conform->ref_coords_const[3*atom_id+2],0);
	}

	// General rotation moving vector
	float4 genrot_movingvec;
	genrot_movingvec.x = genotype[0];
	genrot_movingvec.y = genotype[1];
	genrot_movingvec.z = genotype[2];
	genrot_movingvec.w = 0.0f;
	// Convert orientation genes from sex. to radians
	float phi         = genotype[3] * DEG_TO_RAD;
	float theta       = genotype[4] * DEG_TO_RAD;
	float genrotangle = genotype[5] * DEG_TO_RAD;

	float4 genrot_unitvec;
	float sin_angle = native_sin(theta);
	float s2 = native_sin(genrotangle*0.5f);
	genrot_unitvec.x = s2*sin_angle*native_cos(phi);
	genrot_unitvec.y = s2*sin_angle*native_sin(phi);
	genrot_unitvec.z = s2*native_cos(theta);
	genrot_unitvec.w = native_cos(genrotangle*0.5f);

	uint g1 = dockpars_gridsize_x;
	uint g2 = dockpars_gridsize_x_times_y;
	uint g3 = dockpars_gridsize_x_times_y_times_z;

	barrier(CLK_LOCAL_MEM_FENCE);

	// ================================================
	// CALCULATING ATOMIC POSITIONS AFTER ROTATIONS
	// ================================================
	for ( int rotation_counter = tidx;
	          rotation_counter < dockpars_rotbondlist_length;
	          rotation_counter+=NUM_OF_THREADS_PER_BLOCK)
	{
		int rotation_list_element = kerconst_rotlist->rotlist_const[rotation_counter];
		if ((rotation_list_element & RLIST_DUMMY_MASK) == 0) // If not dummy rotation
		{
			int atom_id = rotation_list_element & RLIST_ATOMID_MASK;
			// Capturing atom coordinates
			float4 atom_to_rotate = calc_coords[atom_id];
			// initialize with general rotation values
			float4 rotation_unitvec;
			float4 rotation_movingvec;
			if (atom_id < dockpars_true_ligand_atoms){
				rotation_unitvec = genrot_unitvec;
				rotation_movingvec = genrot_movingvec;
			} else{
				rotation_unitvec.x = 0.0f; rotation_unitvec.y = 0.0f; rotation_unitvec.z = 0.0f;
				rotation_unitvec.w = 1.0f;
				rotation_movingvec.x = 0.0f; rotation_movingvec.y = 0.0f; rotation_movingvec.z = 0.0f;
				rotation_movingvec.w = 0.0f;
			}

			if ((rotation_list_element & RLIST_GENROT_MASK) == 0) // If rotating around rotatable bond
			{
				uint rotbond_id = (rotation_list_element & RLIST_RBONDID_MASK) >> RLIST_RBONDID_SHIFT;
				float rotation_angle = genotype[6+rotbond_id]*DEG_TO_RAD*0.5f;
				float s = native_sin(rotation_angle);
				rotation_unitvec = (float4)(s*kerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id],
				                            s*kerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id+1],
				                            s*kerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id+2],
				                            native_cos(rotation_angle));
				rotation_movingvec = (float4)(kerconst_conform->rotbonds_moving_vectors_const[3*rotbond_id],
				                              kerconst_conform->rotbonds_moving_vectors_const[3*rotbond_id+1],
				                              kerconst_conform->rotbonds_moving_vectors_const[3*rotbond_id+2],0.0f);
				// Performing additionally the first movement which
				// is needed only if rotating around rotatable bond
				atom_to_rotate -= rotation_movingvec;
			}
			// Performing rotation and final movement
			calc_coords[atom_id] = quaternion_rotate(atom_to_rotate,rotation_unitvec) + rotation_movingvec;
		} // End if-statement not dummy rotation
		barrier(CLK_LOCAL_MEM_FENCE);
	} // End rotation_counter for-loop

	// ================================================
	// CALCULATING INTERMOLECULAR ENERGY
	// ================================================
	float weights[8];
	float cube[8];
	for ( int atom_id = tidx;
	          atom_id < dockpars_num_of_atoms;
	          atom_id+= NUM_OF_THREADS_PER_BLOCK)
	{
		if (kerconst_interintra->ignore_inter_const[atom_id]>0) // first two atoms of a flex res are to be ignored here
			continue;
		uint atom_typeid = kerconst_interintra->atom_types_map_const[atom_id];
		float x = calc_coords[atom_id].x;
		float y = calc_coords[atom_id].y;
		float z = calc_coords[atom_id].z;
		float q = kerconst_interintra->atom_charges_const[atom_id];
		if ((x < 0) || (y < 0) || (z < 0) || (x >= dockpars_gridsize_x-1)
		                                  || (y >= dockpars_gridsize_y-1)
		                                  || (z >= dockpars_gridsize_z-1)){
			partial_energies[tidx] += 16777216.0f; //100000.0f;
			#if defined (DEBUG_ENERGY_KERNEL)
			partial_interE[tidx] += 16777216.0f;
			#endif
			continue; // get on with loop as our work here is done (we crashed into the walls)
		}
		// Getting coordinates
		float x_low  = floor(x);
		float y_low  = floor(y);
		float z_low  = floor(z);

		// Grid value at 000
		__global const float* grid_value_000 = dockpars_fgrids + ((ulong)(x_low  + y_low*g1  + z_low*g2)<<2);

		float dx = x - x_low;
		float omdx = 1.0f - dx;
		float dy = y - y_low;
		float omdy = 1.0f - dy;
		float dz = z - z_low;
		float omdz = 1.0f - dz;

		// Calculating interpolation weights
		weights [idx_000] = omdx*omdy*omdz;
		weights [idx_010] = omdx*dy*omdz;
		weights [idx_001] = omdx*omdy*dz;
		weights [idx_011] = omdx*dy*dz;
		weights [idx_100] = dx*omdy*omdz;
		weights [idx_110] = dx*dy*omdz;
		weights [idx_101] = dx*omdy*dz;
		weights [idx_111] = dx*dy*dz;

		ulong mul_tmp = atom_typeid*g3<<2;
		cube[0] = *(grid_value_000+mul_tmp+0);
		cube[1] = *(grid_value_000+mul_tmp+1);
		cube[2] = *(grid_value_000+mul_tmp+2);
		cube[3] = *(grid_value_000+mul_tmp+3);
		cube[4] = *(grid_value_000+mul_tmp+4);
		cube[5] = *(grid_value_000+mul_tmp+5);
		cube[6] = *(grid_value_000+mul_tmp+6);
		cube[7] = *(grid_value_000+mul_tmp+7);
		// Calculating affinity energy
		partial_energies[tidx] += cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7];
		#if defined (DEBUG_ENERGY_KERNEL)
		partial_interE[tidx] += cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7];
		#endif

		// Capturing electrostatic values
		atom_typeid = dockpars_num_of_map_atypes;

		mul_tmp = atom_typeid*g3<<2; // different atom type id to get charge IA
		cube[0] = *(grid_value_000+mul_tmp+0);
		cube[1] = *(grid_value_000+mul_tmp+1);
		cube[2] = *(grid_value_000+mul_tmp+2);
		cube[3] = *(grid_value_000+mul_tmp+3);
		cube[4] = *(grid_value_000+mul_tmp+4);
		cube[5] = *(grid_value_000+mul_tmp+5);
		cube[6] = *(grid_value_000+mul_tmp+6);
		cube[7] = *(grid_value_000+mul_tmp+7);

		// Calculating affinity energy
		partial_energies[tidx] += q * (cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7]);
		#if defined (DEBUG_ENERGY_KERNEL)
		partial_interE[tidx] += q *(cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7]);
		#endif

		// Need only magnitude of charge from here on down
		q = fabs(q);
		// Capturing desolvation values (atom_typeid+1 compared to above => mul_tmp + g3*4)
		mul_tmp += g3<<2;
		cube[0] = *(grid_value_000+mul_tmp+0);
		cube[1] = *(grid_value_000+mul_tmp+1);
		cube[2] = *(grid_value_000+mul_tmp+2);
		cube[3] = *(grid_value_000+mul_tmp+3);
		cube[4] = *(grid_value_000+mul_tmp+4);
		cube[5] = *(grid_value_000+mul_tmp+5);
		cube[6] = *(grid_value_000+mul_tmp+6);
		cube[7] = *(grid_value_000+mul_tmp+7);

		// Calculating affinity energy
		partial_energies[tidx] += q * (cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7]);
		#if defined (DEBUG_ENERGY_KERNEL)
		partial_interE[tidx] += q *(cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7]);
		#endif
	} // End atom_id for-loop (INTERMOLECULAR ENERGY)

#if defined (DEBUG_ENERGY_KERNEL)
	barrier(CLK_LOCAL_MEM_FENCE);

	// reduction to calculate energy
	for ( int off=NUM_OF_THREADS_PER_BLOCK>>1; off>0; off >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (tidx < off)
			partial_intraE[tidx] += partial_intraE[tidx+off];
	}
#endif

	// In paper: intermolecular and internal energy calculation
	// are independent from each other, -> NO BARRIER NEEDED
	// but require different operations,
	// thus, they can be executed only sequentially on the GPU.
	float delta_distance = 0.5f*dockpars_smooth;
	float smoothed_distance;

	// ================================================
	// CALCULATING INTRAMOLECULAR ENERGY
	// ================================================
	for ( int contributor_counter = tidx;
	          contributor_counter < dockpars_num_of_intraE_contributors;
	          contributor_counter +=NUM_OF_THREADS_PER_BLOCK)
#if 0
if (tidx == 0) {
	for ( int contributor_counter = 0;
	          contributor_counter < dockpars_num_of_intraE_contributors;
	          contributor_counter ++)
#endif
	{
#if 0
		// Only for testing smoothing
		float smoothed_intraE = 0.0f;
		float raw_intraE_vdw_hb = 0.0f;
		float raw_intraE_el     = 0.0f;
		float raw_intraE_sol    = 0.0f;
		float raw_intraE        = 0.0f;
#endif
		// Getting atom IDs
		uint atom1_id = kerconst_intracontrib->intraE_contributors_const[2*contributor_counter];
		uint atom2_id = kerconst_intracontrib->intraE_contributors_const[2*contributor_counter+1];

		// Calculating vector components of vector going
		// from first atom's to second atom's coordinates
		float subx = calc_coords[atom1_id].x - calc_coords[atom2_id].x;
		float suby = calc_coords[atom1_id].y - calc_coords[atom2_id].y;
		float subz = calc_coords[atom1_id].z - calc_coords[atom2_id].z;

		// Calculating atomic_distance
		float atomic_distance = native_sqrt(subx*subx + suby*suby + subz*subz)*dockpars_grid_spacing;

		// Getting type IDs
		uint atom1_typeid = kerconst_interintra->atom_types_const[atom1_id];
		uint atom2_typeid = kerconst_interintra->atom_types_const[atom2_id];

		uint atom1_type_vdw_hb = kerconst_intra->atom_types_reqm_const [atom1_typeid];
		uint atom2_type_vdw_hb = kerconst_intra->atom_types_reqm_const [atom2_typeid];

		// ------------------------------------------------
		// Required only for flexrings
		// Checking if this is a CG-G0 atomic pair.
		// If so, then adding energy term (E = G * distance).
		// Initial specification required NON-SMOOTHED distance.
		// This interaction is evaluated at any distance,
		// so no cuttoffs considered here!
		// vbond is G when calculating flexrings, 0.0 otherwise
		float vbond = G * (float)(((atom1_type_vdw_hb == ATYPE_CG_IDX) && (atom2_type_vdw_hb == ATYPE_G0_IDX)) ||
		                          ((atom1_type_vdw_hb == ATYPE_G0_IDX) && (atom2_type_vdw_hb == ATYPE_CG_IDX)));
		partial_energies[tidx] += vbond * atomic_distance;

		// Calculating energy contributions
		// Cuttoff1: internuclear-distance at 8A only for vdw and hbond
		if (atomic_distance < 8.0f)
		{
			uint idx = atom1_typeid * dockpars_num_of_atypes + atom2_typeid;
			ushort exps = kerconst_intra->VWpars_exp_const[idx];
			char m=(exps & 0xFF00)>>8;
			char n=(exps & 0xFF);
			// Getting optimum pair distance (opt_distance)
			float opt_distance = kerconst_intra->reqm_AB_const[idx];

			// Getting smoothed distance
			// smoothed_distance = function(atomic_distance, opt_distance)
			float opt_dist_delta = opt_distance - atomic_distance;
			if(fabs(opt_dist_delta)>=delta_distance){
				smoothed_distance = atomic_distance + copysign(delta_distance,opt_dist_delta);
			} else smoothed_distance = opt_distance;
			// Calculating van der Waals / hydrogen bond term
			partial_energies[tidx] += (kerconst_intra->VWpars_AC_const[idx]-native_powr(smoothed_distance,m-n)*kerconst_intra->VWpars_BD_const[idx])
			                          *native_powr(smoothed_distance,-m);
			#if defined (DEBUG_ENERGY_KERNEL)
			partial_intraE[tidx] += (kerconst_intra->VWpars_AC_const[idx]-native_powr(smoothed_distance,m-n)*kerconst_intra->VWpars_BD_const[idx])
			                        *native_powr(smoothed_distance,-m));
			#endif
		} // if cuttoff1 - internuclear-distance at 8A

		// Calculating energy contributions
		// Cuttoff2: internuclear-distance at 20.48A only for el and sol.
		if (atomic_distance < 20.48f)
		{
			if(atomic_distance<dockpars_elec_min_distance) atomic_distance=dockpars_elec_min_distance;
			float q1 = kerconst_interintra->atom_charges_const[atom1_id];
			float q2 = kerconst_interintra->atom_charges_const[atom2_id];
			float dist2 = atomic_distance*atomic_distance;
			// Calculating desolvation term
			// 1/25.92 = 0.038580246913580245
			float desolv_energy =  ((kerconst_intra->dspars_S_const[atom1_typeid] +
			                         dockpars_qasp*fabs(q1)) * kerconst_intra->dspars_V_const[atom2_typeid] +
			                        (kerconst_intra->dspars_S_const[atom2_typeid] +
			                         dockpars_qasp*fabs(q2)) * kerconst_intra->dspars_V_const[atom1_typeid]) *
			                       native_divide (
			                                      dockpars_coeff_desolv*(12.96f-0.1063f*dist2*(1.0f-0.001947f*dist2)),
			                                      (12.96f+dist2*(0.4137f+dist2*(0.00357f+0.000112f*dist2)))
			                                     );
			// Calculating electrostatic term
			float dist_shift=atomic_distance+1.26366f;
			dist2=dist_shift*dist_shift;
			float diel = 1.10859f*native_recip(dist2)+0.010358f;
			float es_energy = dockpars_coeff_elec * q1 * q2 * native_recip(atomic_distance);
			partial_energies[tidx] += diel * es_energy + desolv_energy;
			#if 0
			smoothed_intraE += native_divide (
			                                  dockpars_coeff_elec * q1 * q2,
			                                  atomic_distance * (DIEL_A + native_divide(DIEL_B,(1.0f + DIEL_K*native_exp(-DIEL_B_TIMES_H*atomic_distance))))
			                                 ) +
			                   ((kerconst_intra->dspars_S_const[atom1_typeid] +
			                     dockpars_qasp*fabs(q1))*kerconst_intra->dspars_V_const[atom2_typeid] +
			                    (kerconst_intra->dspars_S_const[atom2_typeid] +
			                     dockpars_qasp*fabs(q2))*kerconst_intra->dspars_V_const[atom1_typeid]) *
			                   dockpars_coeff_desolv*native_exp(-0.03858025f*native_powr(atomic_distance, 2));
			raw_intraE_el = native_divide (
			                               dockpars_coeff_elec * q1 * q2,
			                               atomic_distance * (DIEL_A + native_divide(DIEL_B,(1.0f + DIEL_K*native_exp(-DIEL_B_TIMES_H*atomic_distance))))
			                              );
			raw_intraE_sol = ((kerconst_intra->dspars_S_const[atom1_typeid] +
			                   dockpars_qasp*fabs(q1))*kerconst_intra->dspars_V_const[atom2_typeid] +
			                  (kerconst_intra->dspars_S_const[atom2_typeid] +
			                   dockpars_qasp*fabs(q2))*kerconst_intra->dspars_V_const[atom1_typeid]) *
			                 dockpars_coeff_desolv*native_exp(-0.03858025f*native_powr(atomic_distance, 2));
			#endif

			#if defined (DEBUG_ENERGY_KERNEL)
			partial_intraE[tidx] += native_divide (
			                                       dockpars_coeff_elec * q1 * q2,
			                                       atomic_distance * (DIEL_A + native_divide(DIEL_B,(1.0f + DIEL_K*native_exp(-DIEL_B_TIMES_H*atomic_distance))))
			                                      ) +
			                        ((kerconst_intra->dspars_S_const[atom1_typeid] +
			                          dockpars_qasp*fabs(q1)) * kerconst_intra->dspars_V_const[atom2_typeid] +
			                         (kerconst_intra->dspars_S_const[atom2_typeid] +
			                          dockpars_qasp*fabs(q2))*kerconst_intra->dspars_V_const[atom1_typeid]) *
			                        dockpars_coeff_desolv*native_exp(-0.03858025f*native_powr(atomic_distance, 2));
			#endif
		} // if cuttoff2 - internuclear-distance at 20.48A
	} // End contributor_counter for-loop (INTRAMOLECULAR ENERGY)

	barrier(CLK_LOCAL_MEM_FENCE);

	// reduction to calculate energy
	for ( int off=NUM_OF_THREADS_PER_BLOCK>>1; off>0; off >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (tidx < off)
		{
			partial_energies[tidx] += partial_energies[tidx+off];
#if defined (DEBUG_ENERGY_KERNEL)
			partial_intraE[tidx] += partial_intraE[tidx+off];
#endif
		}
	}
	if (tidx == 0)
		*energy = partial_energies[0];
}

