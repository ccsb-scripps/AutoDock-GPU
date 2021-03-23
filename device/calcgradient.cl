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


/*
#include "calcenergy_basic.h"
*/
// All related pragmas are in defines.h (accesible by host and device code)


// The GPU device function calculates the energy's gradient (forces or derivatives) 
// of the entity described by genotype, dockpars and the ligand-data
// arrays in constant memory and returns it in the "gradient_genotype" parameter. 
// The parameter "run_id" has to be equal to the ID of the run 
// whose population includes the current entity (which can be determined with get_group_id(0)), 
// since this determines which reference orientation should be used.

//#define PRINT_GRAD_TRANSLATION_GENES
//#define PRINT_GRAD_ROTATION_GENES
//#define PRINT_GRAD_TORSION_GENES

// The following is a scaling of gradients.
// Initially all genotypes and gradients
// were expressed in grid-units (translations)
// and sexagesimal degrees (rotation and torsion angles).
// Expressing them using angstroms / radians
// might help gradient-based minimizers.
// This conversion is applied to final gradients.
#define CONVERT_INTO_ANGSTROM_RADIAN

// Scaling factor to multiply the gradients of 
// the genes expressed in degrees (all genes except the first three) 
// (GRID-SPACING * GRID-SPACING) / (DEG_TO_RAD * DEG_TO_RAD) = 461.644
#define SCFACTOR_ANGSTROM_RADIAN 1.0f/(DEG_TO_RAD * DEG_TO_RAD)

void map_priv_angle(float* angle)
// The GPU device function maps
// the input parameter to the interval 0...360
// (supposing that it is an angle).
{
	while (*angle >= 360.0f) {
		*angle -= 360.0f;
	}
	while (*angle < 0.0f) {
		*angle += 360.0f;
	}
}

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

// Atomic operations used in gradients of intra contributors.
// Only atomic_cmpxchg() works on floats. 
// So for atomic add on floats, this link was used:
// https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
void atomicAdd_g_f(volatile __local float *addr, float val)
{
	union{
		unsigned int u32;
		float f32;
	} next, expected, current;

	current.f32 = *addr;

	do{
		expected.f32 = current.f32;
		next.f32 = expected.f32 + val;
		current.u32 = atomic_cmpxchg( (volatile __local unsigned int *)addr, expected.u32, next.u32);
	} while( current.u32 != expected.u32 );
}

void atomicSub_g_f(volatile __local float *addr, float val)
{
	union{
		unsigned int u32;
		float f32;
	} next, expected, current;

	current.f32 = *addr;

	do{
		expected.f32 = current.f32;
		next.f32 = expected.f32 - val;
		current.u32 = atomic_cmpxchg( (volatile __local unsigned int *)addr, expected.u32, next.u32);
	} while( current.u32 != expected.u32 );
}

void gpu_calc_gradient(
                             int    dockpars_rotbondlist_length,
                             int    dockpars_num_of_atoms,
                             int    dockpars_true_ligand_atoms,
                             int    dockpars_gridsize_x,
                             int    dockpars_gridsize_y,
                             int    dockpars_gridsize_z,
                                                                         // g1 = gridsize_x
                             uint   dockpars_gridsize_x_times_y,         // g2 = gridsize_x * gridsize_y
                             uint   dockpars_gridsize_x_times_y_times_z, // g3 = gridsize_x * gridsize_y * gridsize_z
              __global const float* restrict dockpars_fgrids, // This is too large to be allocated in __constant 
                             int    dockpars_num_of_atypes,
                             int    dockpars_num_of_map_atypes,
                             int    dockpars_num_of_intraE_contributors,
                             float  dockpars_grid_spacing,
                             float  dockpars_coeff_elec,
                             float  dockpars_elec_min_distance,
                             float  dockpars_qasp,
                             float  dockpars_coeff_desolv,
                             float  dockpars_smooth,

                             // Some OpenCL compilers don't allow declaring
                             // local variables within non-kernel functions.
                             // These local variables must be declared in a kernel,
                             // and then passed to non-kernel functions.
                     __local float* genotype,
                     __local float* energy,
                     __local int*   run_id,

                    __local float4* calc_coords,

                 __constant kernelconstant_interintra*   kerconst_interintra,
             __global const kernelconstant_intracontrib* kerconst_intracontrib,
                 __constant kernelconstant_intra*        kerconst_intra,
                 __constant kernelconstant_rotlist*      kerconst_rotlist,
                 __constant kernelconstant_conform*      kerconst_conform,

                 __constant int*    rotbonds_const,
             __global const int*    rotbonds_atoms_const,
                 __constant int*    num_rotating_atoms_per_rotbond_const,

             __global const float*  angle_const,
                 __constant float*  dependence_on_theta_const,
                 __constant float*  dependence_on_rotangle_const,

                            int     dockpars_num_of_genes,
                    __local float*  gradient_inter_x,
                    __local float*  gradient_inter_y,
                    __local float*  gradient_inter_z,
                    __local float*  gradient_intra_x,
                    __local float*  gradient_intra_y,
                    __local float*  gradient_intra_z,
                    __local float*  gradient_genotype
                      )
{
	int tidx = get_local_id(0);
	// Initializing gradients (forces)
	// Derived from autodockdev/maps.py
	for (int atom_id = tidx;
	         atom_id < dockpars_num_of_atoms;
	         atom_id+= NUM_OF_THREADS_PER_BLOCK)
	{
		// Intermolecular gradients
		gradient_inter_x[atom_id] = 0.0f;
		gradient_inter_y[atom_id] = 0.0f;
		gradient_inter_z[atom_id] = 0.0f;
		// Intramolecular gradients
		gradient_intra_x[atom_id] = 0.0f;
		gradient_intra_y[atom_id] = 0.0f;
		gradient_intra_z[atom_id] = 0.0f;
	}

	// Initializing gradient genotypes
	for ( int gene_cnt = tidx;
	          gene_cnt < dockpars_num_of_genes;
	          gene_cnt+= NUM_OF_THREADS_PER_BLOCK)
	{
		gradient_genotype[gene_cnt] = 0.0f;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Convert orientation genes from sex. to radians
	float phi         = genotype[3] * DEG_TO_RAD;
	float theta       = genotype[4] * DEG_TO_RAD;
	float genrotangle = genotype[5] * DEG_TO_RAD;

	float genrot_unitvec [3];
	float sin_angle = native_sin(theta);
	genrot_unitvec [0] = sin_angle*native_cos(phi);
	genrot_unitvec [1] = sin_angle*native_sin(phi);
	genrot_unitvec [2] = native_cos(theta);

	uint g1 = dockpars_gridsize_x;
	uint g2 = dockpars_gridsize_x_times_y;
  	uint g3 = dockpars_gridsize_x_times_y_times_z;

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
			float atom_to_rotate[3];
			if ((rotation_list_element & RLIST_FIRSTROT_MASK) != 0) // If first rotation of this atom
			{
				atom_to_rotate[0] = kerconst_conform->ref_coords_const[3*atom_id];
				atom_to_rotate[1] = kerconst_conform->ref_coords_const[3*atom_id+1];
				atom_to_rotate[2] = kerconst_conform->ref_coords_const[3*atom_id+2];
			}
			else
			{
				atom_to_rotate[0] = calc_coords[atom_id].x;
				atom_to_rotate[1] = calc_coords[atom_id].y;
				atom_to_rotate[2] = calc_coords[atom_id].z;
			}
			// Capturing rotation vectors and angle
			float rotation_unitvec[3];
			float rotation_movingvec[3];
			float rotation_angle;

			float quatrot_left_x, quatrot_left_y, quatrot_left_z, quatrot_left_q;
			float quatrot_temp_x, quatrot_temp_y, quatrot_temp_z, quatrot_temp_q;

			if ((rotation_list_element & RLIST_GENROT_MASK) != 0) // If general rotation
			{
				if (atom_id < dockpars_true_ligand_atoms){
					rotation_unitvec[0] = genrot_unitvec[0];
					rotation_unitvec[1] = genrot_unitvec[1];
					rotation_unitvec[2] = genrot_unitvec[2];

					rotation_movingvec[0] = genotype[0];
					rotation_movingvec[1] = genotype[1];
					rotation_movingvec[2] = genotype[2];

					rotation_angle = genrotangle;
				} else{
					rotation_unitvec[0] = 1.0f;
					rotation_unitvec[1] = 0.0f;
					rotation_unitvec[2] = 0.0f;

					rotation_movingvec[0] = 0.0f;
					rotation_movingvec[1] = 0.0f;
					rotation_movingvec[2] = 0.0f;

					rotation_angle = 0.0f;
				}
			}
			else // If rotating around rotatable bond
			{
				uint rotbond_id = (rotation_list_element & RLIST_RBONDID_MASK) >> RLIST_RBONDID_SHIFT;

				rotation_unitvec[0] = kerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id];
				rotation_unitvec[1] = kerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id+1];
				rotation_unitvec[2] = kerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id+2];

				rotation_movingvec[0] = kerconst_conform->rotbonds_moving_vectors_const[3*rotbond_id];
				rotation_movingvec[1] = kerconst_conform->rotbonds_moving_vectors_const[3*rotbond_id+1];
				rotation_movingvec[2] = kerconst_conform->rotbonds_moving_vectors_const[3*rotbond_id+2];

				rotation_angle = genotype[6+rotbond_id]*DEG_TO_RAD;

				// Performing additionally the first movement which 
				// is needed only if rotating around rotatable bond
				atom_to_rotate[0] -= rotation_movingvec[0];
				atom_to_rotate[1] -= rotation_movingvec[1];
				atom_to_rotate[2] -= rotation_movingvec[2];
			}

			// Transforming orientation and torsion angles into quaternions
			rotation_angle  = rotation_angle * 0.5f;
			float sin_angle = native_sin(rotation_angle);
			quatrot_left_q  = native_cos(rotation_angle);
			quatrot_left_x  = sin_angle*rotation_unitvec[0];
			quatrot_left_y  = sin_angle*rotation_unitvec[1];
			quatrot_left_z  = sin_angle*rotation_unitvec[2];

			quatrot_temp_q = 0 -
			                 quatrot_left_x*atom_to_rotate [0] -
			                 quatrot_left_y*atom_to_rotate [1] -
			                 quatrot_left_z*atom_to_rotate [2];
			quatrot_temp_x = quatrot_left_q*atom_to_rotate [0] +
			                 quatrot_left_y*atom_to_rotate [2] -
			                 quatrot_left_z*atom_to_rotate [1];
			quatrot_temp_y = quatrot_left_q*atom_to_rotate [1] -
			                 quatrot_left_x*atom_to_rotate [2] +
			                 quatrot_left_z*atom_to_rotate [0];
			quatrot_temp_z = quatrot_left_q*atom_to_rotate [2] +
			                 quatrot_left_x*atom_to_rotate [1] -
			                 quatrot_left_y*atom_to_rotate [0];

			atom_to_rotate [0] = 0 -
			                     quatrot_temp_q*quatrot_left_x +
			                     quatrot_temp_x*quatrot_left_q -
			                     quatrot_temp_y*quatrot_left_z +
			                     quatrot_temp_z*quatrot_left_y;
			atom_to_rotate [1] = 0 -
			                     quatrot_temp_q*quatrot_left_y +
			                     quatrot_temp_x*quatrot_left_z +
			                     quatrot_temp_y*quatrot_left_q -
			                     quatrot_temp_z*quatrot_left_x;
			atom_to_rotate [2] = 0 -
			                     quatrot_temp_q*quatrot_left_z -
			                     quatrot_temp_x*quatrot_left_y +
			                     quatrot_temp_y*quatrot_left_x +
			                     quatrot_temp_z*quatrot_left_q;
			// Performing final movement and storing values
			calc_coords[atom_id].x = atom_to_rotate [0] + rotation_movingvec[0];
			calc_coords[atom_id].y = atom_to_rotate [1] + rotation_movingvec[1];
			calc_coords[atom_id].z = atom_to_rotate [2] + rotation_movingvec[2];

		} // End if-statement not dummy rotation

		barrier(CLK_LOCAL_MEM_FENCE);

	} // End rotation_counter for-loop

	// ================================================
	// CALCULATING INTERMOLECULAR GRADIENTS
	// ================================================
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
		                                  || (z >= dockpars_gridsize_z-1)) {
			// Setting gradients (forces) penalties.
			// These are valid as long as they are high
			gradient_inter_x[atom_id] += 16777216.0f;
			gradient_inter_y[atom_id] += 16777216.0f;
			gradient_inter_z[atom_id] += 16777216.0f;
		}
		else
		{
			// Getting coordinates
			uint x_low  = (uint)floor(x);
			uint y_low  = (uint)floor(y);
			uint z_low  = (uint)floor(z);
			uint x_high = (uint)ceil(x);
			uint y_high = (uint)ceil(y);
			uint z_high = (uint)ceil(z);
			float dx = x - x_low;
			float dy = y - y_low;
			float dz = z - z_low;

			//printf("%-15s %-5u %-10.8f %-10.8f %-10.8f\n", "dx,dy,dz", atom_id, dx, dy, dz);

			// Calculating interpolation weights
			float weights[8];
			weights [idx_000] = (1.0f-dx)*(1.0f-dy)*(1.0f-dz);
			weights [idx_100] = dx*(1.0f-dy)*(1.0f-dz);
			weights [idx_010] = (1.0f-dx)*dy*(1.0f-dz);
			weights [idx_110] = dx*dy*(1.0f-dz);
			weights [idx_001] = (1.0f-dx)*(1.0f-dy)*dz;
			weights [idx_101] = dx*(1.0f-dy)*dz;
			weights [idx_011] = (1.0f-dx)*dy*dz;
			weights [idx_111] = dx*dy*dz;

			// Capturing affinity values
			uint ylow_times_g1  = y_low*g1;
			uint yhigh_times_g1 = y_high*g1;
		  	uint zlow_times_g2  = z_low*g2;
			uint zhigh_times_g2 = z_high*g2;

			// Grid offset
			ulong offset_cube_000 = (x_low  + ylow_times_g1  + zlow_times_g2)<<2;
			ulong offset_cube_100 = (x_high + ylow_times_g1  + zlow_times_g2)<<2;
			ulong offset_cube_010 = (x_low  + yhigh_times_g1 + zlow_times_g2)<<2;
			ulong offset_cube_110 = (x_high + yhigh_times_g1 + zlow_times_g2)<<2;
			ulong offset_cube_001 = (x_low  + ylow_times_g1  + zhigh_times_g2)<<2;
			ulong offset_cube_101 = (x_high + ylow_times_g1  + zhigh_times_g2)<<2;
			ulong offset_cube_011 = (x_low  + yhigh_times_g1 + zhigh_times_g2)<<2;
			ulong offset_cube_111 = (x_high + yhigh_times_g1 + zhigh_times_g2)<<2;

			ulong mul_tmp = atom_typeid*g3<<2;

			float cube[8];
			cube [idx_000] = *(dockpars_fgrids + offset_cube_000 + mul_tmp);
			cube [idx_100] = *(dockpars_fgrids + offset_cube_100 + mul_tmp);
			cube [idx_010] = *(dockpars_fgrids + offset_cube_010 + mul_tmp);
			cube [idx_110] = *(dockpars_fgrids + offset_cube_110 + mul_tmp);
			cube [idx_001] = *(dockpars_fgrids + offset_cube_001 + mul_tmp);
			cube [idx_101] = *(dockpars_fgrids + offset_cube_101 + mul_tmp);
			cube [idx_011] = *(dockpars_fgrids + offset_cube_011 + mul_tmp);
			cube [idx_111] = *(dockpars_fgrids + offset_cube_111 + mul_tmp);

			// -------------------------------------------------------------------
			// Deltas dx, dy, dz are already normalized 
			// (by host/src/getparameters.cpp) in AutoDock-GPU.
			// The correspondance between vertices in xyz axes is:
			// 0, 1, 2, 3, 4, 5, 6, 7  and  000, 100, 010, 001, 101, 110, 011, 111
			// -------------------------------------------------------------------
			/*
			    deltas: (x-x0)/(x1-x0), (y-y0...
			    vertices: (000, 100, 010, 001, 101, 110, 011, 111)        

			          Z
			          '
			          3 - - - - 6
			         /.        /|
			        4 - - - - 7 |
			        | '       | |
			        | 0 - - - + 2 -- Y
			        '/        |/
			        1 - - - - 5
			       /
			      X
			*/

			// Intermediate values for vectors in x-direction
			float x10, x52, x43, x76;
			float vx_z0, vx_z1;

			// Intermediate values for vectors in y-direction
			float y20, y51, y63, y74;
			float vy_z0, vy_z1;

			// Intermediate values for vectors in z-direction
			float z30, z41, z62, z75;
			float vz_y0, vz_y1;

			// -------------------------------------------------------------------
			// Calculating gradients (forces) corresponding to 
			// "atype" intermolecular energy
			// Derived from autodockdev/maps.py
			// -------------------------------------------------------------------

			// Vector in x-direction
			x10 = cube [idx_100] - cube [idx_000]; // z = 0
			x52 = cube [idx_110] - cube [idx_010]; // z = 0
			x43 = cube [idx_101] - cube [idx_001]; // z = 1
			x76 = cube [idx_111] - cube [idx_011]; // z = 1
			vx_z0 = (1.0f - dy) * x10 + dy * x52;     // z = 0
			vx_z1 = (1.0f - dy) * x43 + dy * x76;     // z = 1
			gradient_inter_x[atom_id] += (1.0f - dz) * vx_z0 + dz * vx_z1;

			// Vector in y-direction
			y20 = cube[idx_010] - cube [idx_000];	// z = 0
			y51 = cube[idx_110] - cube [idx_100];	// z = 0
			y63 = cube[idx_011] - cube [idx_001];	// z = 1
			y74 = cube[idx_111] - cube [idx_101];	// z = 1
			vy_z0 = (1.0f - dx) * y20 + dx * y51;	// z = 0
			vy_z1 = (1.0f - dx) * y63 + dx * y74;	// z = 1
			gradient_inter_y[atom_id] += (1.0f - dz) * vy_z0 + dz * vy_z1;

			// Vectors in z-direction
			z30 = cube [idx_001] - cube [idx_000];	// y = 0
			z41 = cube [idx_101] - cube [idx_100];	// y = 0
			z62 = cube [idx_011] - cube [idx_010];	// y = 1 
			z75 = cube [idx_111] - cube [idx_110];	// y = 1
			vz_y0 = (1.0f - dx) * z30 + dx * z41;	// y = 0
			vz_y1 = (1.0f - dx) * z62 + dx * z75;	// y = 1
			gradient_inter_z[atom_id] += (1.0f - dy) * vz_y0 + dy * vz_y1;

			//printf("%-15s %-3u %-10.8f %-10.8f %-10.8f %-10.8f %-10.8f %-10.8f\n", "atom aff", atom_id, vx_z0, vx_z1, vy_z0, vy_z1, vz_y0, vz_y1);

			// -------------------------------------------------------------------
			// Calculating gradients (forces) corresponding to 
			// "elec" intermolecular energy
			// Derived from autodockdev/maps.py
			// -------------------------------------------------------------------

			// Capturing electrostatic values
			atom_typeid = dockpars_num_of_map_atypes;

			mul_tmp = atom_typeid*g3<<2;
			cube [idx_000] = *(dockpars_fgrids + offset_cube_000 + mul_tmp);
			cube [idx_100] = *(dockpars_fgrids + offset_cube_100 + mul_tmp);
      			cube [idx_010] = *(dockpars_fgrids + offset_cube_010 + mul_tmp);
      			cube [idx_110] = *(dockpars_fgrids + offset_cube_110 + mul_tmp);
		       	cube [idx_001] = *(dockpars_fgrids + offset_cube_001 + mul_tmp);
		        cube [idx_101] = *(dockpars_fgrids + offset_cube_101 + mul_tmp);
		        cube [idx_011] = *(dockpars_fgrids + offset_cube_011 + mul_tmp);
		        cube [idx_111] = *(dockpars_fgrids + offset_cube_111 + mul_tmp);

			// Vector in x-direction
			x10 = cube [idx_100] - cube [idx_000]; // z = 0
			x52 = cube [idx_110] - cube [idx_010]; // z = 0
			x43 = cube [idx_101] - cube [idx_001]; // z = 1
			x76 = cube [idx_111] - cube [idx_011]; // z = 1
			vx_z0 = (1.0f - dy) * x10 + dy * x52;     // z = 0
			vx_z1 = (1.0f - dy) * x43 + dy * x76;     // z = 1
			gradient_inter_x[atom_id] += q * ((1.0f - dz) * vx_z0 + dz * vx_z1);

			// Vector in y-direction
			y20 = cube[idx_010] - cube [idx_000];	// z = 0
			y51 = cube[idx_110] - cube [idx_100];	// z = 0
			y63 = cube[idx_011] - cube [idx_001];	// z = 1
			y74 = cube[idx_111] - cube [idx_101];	// z = 1
			vy_z0 = (1.0f - dx) * y20 + dx * y51;	// z = 0
			vy_z1 = (1.0f - dx) * y63 + dx * y74;	// z = 1
			gradient_inter_y[atom_id] += q *((1.0f - dz) * vy_z0 + dz * vy_z1);

			// Vectors in z-direction
			z30 = cube [idx_001] - cube [idx_000];	// y = 0
			z41 = cube [idx_101] - cube [idx_100];	// y = 0
			z62 = cube [idx_011] - cube [idx_010];	// y = 1 
			z75 = cube [idx_111] - cube [idx_110];	// y = 1
			vz_y0 = (1.0f - dx) * z30 + dx * z41;	// y = 0
			vz_y1 = (1.0f - dx) * z62 + dx * z75;	// y = 1
			gradient_inter_z[atom_id] += q *((1.0f - dy) * vz_y0 + dy * vz_y1);

			//printf("%-15s %-3u %-10.8f %-10.8f %-10.8f %-10.8f %-10.8f %-10.8f\n", "elec", atom_id, vx_z0, vx_z1, vy_z0, vy_z1, vz_y0, vz_y1);

			// -------------------------------------------------------------------
			// Calculating gradients (forces) corresponding to 
			// "dsol" intermolecular energy
			// Derived from autodockdev/maps.py
			// -------------------------------------------------------------------

			// Capturing desolvation values
			atom_typeid = dockpars_num_of_map_atypes+1;

			mul_tmp = atom_typeid*g3<<2;
			cube [idx_000] = *(dockpars_fgrids + offset_cube_000 + mul_tmp);
			cube [idx_100] = *(dockpars_fgrids + offset_cube_100 + mul_tmp);
			cube [idx_010] = *(dockpars_fgrids + offset_cube_010 + mul_tmp);
			cube [idx_110] = *(dockpars_fgrids + offset_cube_110 + mul_tmp);
			cube [idx_001] = *(dockpars_fgrids + offset_cube_001 + mul_tmp);
			cube [idx_101] = *(dockpars_fgrids + offset_cube_101 + mul_tmp);
			cube [idx_011] = *(dockpars_fgrids + offset_cube_011 + mul_tmp);
			cube [idx_111] = *(dockpars_fgrids + offset_cube_111 + mul_tmp);

			// Vector in x-direction
			x10 = cube [idx_100] - cube [idx_000]; // z = 0
			x52 = cube [idx_110] - cube [idx_010]; // z = 0
			x43 = cube [idx_101] - cube [idx_001]; // z = 1
			x76 = cube [idx_111] - cube [idx_011]; // z = 1
			vx_z0 = (1.0f - dy) * x10 + dy * x52;     // z = 0
			vx_z1 = (1.0f - dy) * x43 + dy * x76;     // z = 1
			gradient_inter_x[atom_id] += fabs(q) * ((1.0f - dz) * vx_z0 + dz * vx_z1);

			// Vector in y-direction
			y20 = cube[idx_010] - cube [idx_000];	// z = 0
			y51 = cube[idx_110] - cube [idx_100];	// z = 0
			y63 = cube[idx_011] - cube [idx_001];	// z = 1
			y74 = cube[idx_111] - cube [idx_101];	// z = 1
			vy_z0 = (1.0f - dx) * y20 + dx * y51;	// z = 0
			vy_z1 = (1.0f - dx) * y63 + dx * y74;	// z = 1
			gradient_inter_y[atom_id] += fabs(q) *((1.0f - dz) * vy_z0 + dz * vy_z1);

			// Vectors in z-direction
			z30 = cube [idx_001] - cube [idx_000];	// y = 0
			z41 = cube [idx_101] - cube [idx_100];	// y = 0
			z62 = cube [idx_011] - cube [idx_010];	// y = 1 
			z75 = cube [idx_111] - cube [idx_110];	// y = 1
			vz_y0 = (1.0f - dx) * z30 + dx * z41;	// y = 0
			vz_y1 = (1.0f - dx) * z62 + dx * z75;	// y = 1
			gradient_inter_z[atom_id] += fabs(q) *((1.0f - dy) * vz_y0 + dy * vz_y1);

			//printf("%-15s %-3u %-10.8f %-10.8f %-10.8f %-10.8f %-10.8f %-10.8f\n", "desol", atom_id, vx_z0, vx_z1, vy_z0, vy_z1, vz_y0, vz_y1);
			// -------------------------------------------------------------------
		}

	} // End atom_id for-loop (INTERMOLECULAR ENERGY)

	// Inter- and intra-molecular energy calculation
	// are independent from each other, so NO barrier is needed here.
	// As these two require different operations,
	// they can be executed only sequentially on the GPU.

	// ================================================
	// CALCULATING INTRAMOLECULAR GRADIENTS
	// ================================================
	for ( int contributor_counter = tidx;
	          contributor_counter < dockpars_num_of_intraE_contributors;
	          contributor_counter+= NUM_OF_THREADS_PER_BLOCK)
	{
		// Storing in a private variable
		// the gradient contribution of each contributing atomic pair
		float priv_gradient_per_intracontributor= 0.0f;

		// Getting atom IDs
		uint atom1_id = kerconst_intracontrib->intraE_contributors_const[2*contributor_counter];
		uint atom2_id = kerconst_intracontrib->intraE_contributors_const[2*contributor_counter+1];

		/*
		printf ("%-5u %-5u %-5u\n", contributor_counter, atom1_id, atom2_id);
		*/

		// Calculating vector components of vector going
		// from first atom's to second atom's coordinates
		float subx = calc_coords[atom1_id].x - calc_coords[atom2_id].x;
		float suby = calc_coords[atom1_id].y - calc_coords[atom2_id].y;
		float subz = calc_coords[atom1_id].z - calc_coords[atom2_id].z;

		// Calculating atomic distance
		float dist = native_sqrt(subx*subx + suby*suby + subz*subz);
		float atomic_distance = dist*dockpars_grid_spacing;

		// Getting type IDs
		uint atom1_typeid = kerconst_interintra->atom_types_const[atom1_id];
		uint atom2_typeid = kerconst_interintra->atom_types_const[atom2_id];

		uint atom1_type_vdw_hb = kerconst_intra->atom_types_reqm_const [atom1_typeid];
	     	uint atom2_type_vdw_hb = kerconst_intra->atom_types_reqm_const [atom2_typeid];
		//printf ("%-5u %-5u %-5u\n", contributor_counter, atom1_id, atom2_id);

		ushort exps = kerconst_intra->VWpars_exp_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid];
		char m=(exps & 0xFF00)>>8;
		char n=(exps & 0xFF);
		// Getting optimum pair distance (opt_distance) from reqm and reqm_hbond
		float opt_distance = kerconst_intra->reqm_AB_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid];

		// Getting smoothed distance
		// smoothed_distance = function(atomic_distance, opt_distance)
		float smoothed_distance;
		float delta_distance = 0.5f*dockpars_smooth;

		if (atomic_distance <= (opt_distance - delta_distance)) {
			smoothed_distance = atomic_distance + delta_distance;
		}
		else if (atomic_distance < (opt_distance + delta_distance)) {
			smoothed_distance = opt_distance;
		}
		else { // else if (atomic_distance >= (opt_distance + delta_distance))
			smoothed_distance = atomic_distance - delta_distance;
		}

		// Calculating gradient contributions
		// Cuttoff1: internuclear-distance at 8A only for vdw and hbond.
		if (atomic_distance < 8.0f)
		{
			// Calculating van der Waals / hydrogen bond term
			priv_gradient_per_intracontributor += native_divide (-(float)m*kerconst_intra->VWpars_AC_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],
			                                                      native_powr(smoothed_distance/*atomic_distance*/, m+1)
			                                                    );

			priv_gradient_per_intracontributor += native_divide ((float)n*kerconst_intra->VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],
			                                                     native_powr(smoothed_distance/*atomic_distance*/, n+1)
			                                                    );
		} // if cuttoff1 - internuclear-distance at 8A

		// Calculating energy contributions
		// Cuttoff2: internuclear-distance at 20.48A only for el and sol.
		if (atomic_distance < 20.48f)
		{
			if(atomic_distance<dockpars_elec_min_distance) atomic_distance=dockpars_elec_min_distance;
			// Calculating electrostatic term
			// http://www.wolframalpha.com/input/?i=1%2F(x*(A%2B(B%2F(1%2BK*exp(-h*B*x)))))
			float upper = DIEL_A*native_powr(native_exp(DIEL_B_TIMES_H*atomic_distance) + DIEL_K, 2) + (DIEL_B)*native_exp(DIEL_B_TIMES_H*atomic_distance)*(DIEL_B_TIMES_H_TIMES_K*atomic_distance + native_exp(DIEL_B_TIMES_H*atomic_distance) + DIEL_K);
			float lower = native_powr(atomic_distance, 2) * native_powr(DIEL_A * (native_exp(DIEL_B_TIMES_H*atomic_distance) + DIEL_K) + DIEL_B * native_exp(DIEL_B_TIMES_H*atomic_distance), 2);

			priv_gradient_per_intracontributor +=  -dockpars_coeff_elec * kerconst_interintra->atom_charges_const[atom1_id] * kerconst_interintra->atom_charges_const[atom2_id] * native_divide (upper, lower);

			// Calculating desolvation term
			priv_gradient_per_intracontributor += ((kerconst_intra->dspars_S_const[atom1_typeid] + dockpars_qasp*fabs(kerconst_interintra->atom_charges_const[atom1_id])) * kerconst_intra->dspars_V_const[atom2_typeid] +
			                                       (kerconst_intra->dspars_S_const[atom2_typeid] + dockpars_qasp*fabs(kerconst_interintra->atom_charges_const[atom2_id])) * kerconst_intra->dspars_V_const[atom1_typeid]
			                                      ) * dockpars_coeff_desolv * /*-0.07716049382716049*/ -0.077160f * atomic_distance * native_exp(/*-0.038580246913580245*/ -0.038580f *native_powr(atomic_distance, 2));
		} // if cuttoff2 - internuclear-distance at 20.48A

		// ------------------------------------------------
		// Required only for flexrings
		// Checking if this is a CG-G0 atomic pair.
		// If so, then adding energy term (E = G * distance).
		// Initial specification required NON-SMOOTHED distance.
		// This interaction is evaluated at any distance,
		// so no cuttoffs considered here!
		if (((atom1_type_vdw_hb == ATYPE_CG_IDX) && (atom2_type_vdw_hb == ATYPE_G0_IDX)) ||
		    ((atom1_type_vdw_hb == ATYPE_G0_IDX) && (atom2_type_vdw_hb == ATYPE_CG_IDX)))
		{
			priv_gradient_per_intracontributor += G;
		}
		// ------------------------------------------------

		// Decomposing "priv_gradient_per_intracontributor" 
		// into the contribution of each atom of the pair.
		// Distances in Angstroms of vector that goes from 
		// "atom1_id"-to-"atom2_id", therefore - subx, - suby, and - subz are used
		float subx_div_dist = native_divide(-subx, dist);
		float suby_div_dist = native_divide(-suby, dist);
		float subz_div_dist = native_divide(-subz, dist);

		float priv_intra_gradient_x = priv_gradient_per_intracontributor * subx_div_dist;
		float priv_intra_gradient_y = priv_gradient_per_intracontributor * suby_div_dist;
		float priv_intra_gradient_z = priv_gradient_per_intracontributor * subz_div_dist;
		
		// Calculating gradients in xyz components.
		// Gradients for both atoms in a single contributor pair
		// have the same magnitude, but opposite directions
		atomicSub_g_f(&gradient_intra_x[atom1_id], priv_intra_gradient_x);
		atomicSub_g_f(&gradient_intra_y[atom1_id], priv_intra_gradient_y);
		atomicSub_g_f(&gradient_intra_z[atom1_id], priv_intra_gradient_z);

		atomicAdd_g_f(&gradient_intra_x[atom2_id], priv_intra_gradient_x);
		atomicAdd_g_f(&gradient_intra_y[atom2_id], priv_intra_gradient_y);
		atomicAdd_g_f(&gradient_intra_z[atom2_id], priv_intra_gradient_z);
	} // End contributor_counter for-loop (INTRAMOLECULAR ENERGY)
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Accumulating inter- and intramolecular gradients
	for ( int atom_cnt = tidx;
	          atom_cnt < dockpars_num_of_atoms;
	          atom_cnt+= NUM_OF_THREADS_PER_BLOCK)
	{
		// Grid gradients were calculated in the grid space,
		// so they have to be put back in Angstrom.

		// Intramolecular gradients were already in Angstrom,
		// so no scaling for them is required.
		gradient_inter_x[atom_cnt] = native_divide(gradient_inter_x[atom_cnt], dockpars_grid_spacing);
		gradient_inter_y[atom_cnt] = native_divide(gradient_inter_y[atom_cnt], dockpars_grid_spacing);
		gradient_inter_z[atom_cnt] = native_divide(gradient_inter_z[atom_cnt], dockpars_grid_spacing);

		#if defined (PRINT_GRAD_ROTATION_GENES)
		if (atom_cnt == 0) {
			printf("\n%s\n", "----------------------------------------------------------");
			printf("%s\n", "Gradients: inter and intra");
			printf("%10s %13s %13s %13s %5s %13s %13s %13s\n", "atom_id", "grad_intER.x", "grad_intER.y", "grad_intER.z", "|", "grad_intRA.x", "grad_intRA.y", "grad_intRA.z");
		}
		printf("%10u %13.6f %13.6f %13.6f %5s %13.6f %13.6f %13.6f\n", atom_cnt, gradient_inter_x[atom_cnt], gradient_inter_y[atom_cnt], gradient_inter_z[atom_cnt], "|", gradient_intra_x[atom_cnt], gradient_intra_y[atom_cnt], gradient_intra_z[atom_cnt]);
		#endif

		// Re-using "gradient_inter_*" for total gradient (inter+intra)
		gradient_inter_x[atom_cnt] += gradient_intra_x[atom_cnt];
		gradient_inter_y[atom_cnt] += gradient_intra_y[atom_cnt];
		gradient_inter_z[atom_cnt] += gradient_intra_z[atom_cnt];

		//printf("%-15s %-5u %-10.8f %-10.8f %-10.8f\n", "grad_grid", atom_cnt, gradient_inter_x[atom_cnt], gradient_inter_y[atom_cnt], gradient_inter_z[atom_cnt]);
		//printf("%-15s %-5u %-10.8f %-10.8f %-10.8f\n", "grad_intra", atom_cnt, gradient_intra_x[atom_cnt], gradient_intra_y[atom_cnt], gradient_intra_z[atom_cnt]);
		//printf("%-15s %-5u %-10.8f %-10.8f %-10.8f\n", "calc_coords", atom_cnt, calc_coords_x[atom_cnt], calc_coords_y[atom_cnt], calc_coords_z[atom_cnt]);

		#if defined (PRINT_GRAD_ROTATION_GENES)
		if (atom_cnt == 0) {
			printf("\n%s\n", "----------------------------------------------------------");
			printf("%s\n", "Gradients: total = inter + intra");
			printf("%10s %13s %13s %13s\n", "atom_id", "grad.x", "grad.y", "grad.z");
		}
		printf("%10u %13.6f %13.6f %13.6f \n", atom_cnt, gradient_inter_x[atom_cnt], gradient_inter_y[atom_cnt], gradient_inter_z[atom_cnt]);
		#endif
	}

	barrier(CLK_LOCAL_MEM_FENCE);



	// ------------------------------------------
	// Obtaining translation-related gradients
	// ------------------------------------------
	if (tidx == 0) {
		for ( int lig_atom_id = 0;
		          lig_atom_id<dockpars_true_ligand_atoms;
		          lig_atom_id++)
		{
			// Re-using "gradient_inter_*" for total gradient (inter+intra)
			gradient_genotype[0] += gradient_inter_x[lig_atom_id]; // gradient for gene 0: gene x
			gradient_genotype[1] += gradient_inter_y[lig_atom_id]; // gradient for gene 1: gene y
			gradient_genotype[2] += gradient_inter_z[lig_atom_id]; // gradient for gene 2: gene z
		}

		// Scaling gradient for translational genes as 
		// their corresponding gradients were calculated in the space 
		// where these genes are in Angstrom,
		// but AutoDock-GPU translational genes are within in grids
		gradient_genotype[0] *= dockpars_grid_spacing;
		gradient_genotype[1] *= dockpars_grid_spacing;
		gradient_genotype[2] *= dockpars_grid_spacing;

		#if defined (PRINT_GRAD_TRANSLATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("gradient_x:%f\n", gradient_genotype [0]);
		printf("gradient_y:%f\n", gradient_genotype [1]);
		printf("gradient_z:%f\n", gradient_genotype [2]);
		#endif
	}

	// ------------------------------------------
	// Obtaining rotation-related gradients
	// ------------------------------------------ 
				
	// Transform gradients_inter_{x|y|z} 
	// into local_gradients[i] (with four quaternion genes)
	// Derived from autodockdev/motions.py/forces_to_delta_genes()

	// Transform local_gradients[i] (with four quaternion genes)
	// into local_gradients[i] (with three Shoemake genes)
	// Derived from autodockdev/motions.py/_get_cube3_gradient()
	// ------------------------------------------
	if ((tidx == 1) | (NUM_OF_THREADS_PER_BLOCK<1)) {
		float3 torque_rot;
		torque_rot.x = 0.0f;
		torque_rot.y = 0.0f;
		torque_rot.z = 0.0f;

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-20s %-10.6f %-10.6f %-10.6f\n", "initial torque: ", torque_rot.x, torque_rot.y, torque_rot.z);
		#endif

		// Declaring a variable to hold the center of rotation 
		// In getparameters.cpp, it indicates 
		// translation genes are in grid spacing (instead of Angstroms)
		float3 about;
		about.x = genotype[0];
		about.y = genotype[1];
		about.z = genotype[2];
	
		// Temporal variable to calculate translation differences.
		// They are converted back to Angstroms here
		float3 r;
			
		for ( int lig_atom_id = 0;
		          lig_atom_id<dockpars_true_ligand_atoms;
		          lig_atom_id++)
		{
			r.x = (calc_coords[lig_atom_id].x - about.x) * dockpars_grid_spacing;
			r.y = (calc_coords[lig_atom_id].y - about.y) * dockpars_grid_spacing;
			r.z = (calc_coords[lig_atom_id].z - about.z) * dockpars_grid_spacing;

			// Re-using "gradient_inter_*" for total gradient (inter+intra)
			float3 force;
			force.x = gradient_inter_x[lig_atom_id];
			force.y = gradient_inter_y[lig_atom_id]; 
			force.z = gradient_inter_z[lig_atom_id];

			torque_rot += cross(r, force);

			#if defined (PRINT_GRAD_ROTATION_GENES)
#if 0
			printf("%-20s %-10u\n", "contrib. of atom-id: ", lig_atom_id);
			printf("%-20s %-10.5f %-10.5f %-10.5f\n", "r             : ", r.x, r.y, r.z);
			printf("%-20s %-10.5f %-10.5f %-10.5f\n", "force         : ", force.x, force.y, force.z);
			printf("%-20s %-10.5f %-10.5f %-10.5f\n", "partial torque: ", torque_rot.x, torque_rot.y, torque_rot.z);
			printf("\n");
#endif
			// This printing is similar to autodockdevpy
			if (lig_atom_id == 0) {
				printf("\n%s\n", "----------------------------------------------------------");
				printf("%s\n", "Torque: atom-based accumulation of torque");
				printf("%10s %10s %10s %10s %5s %12s %12s %12s %5s %11s %11s %11s\n", "atom_id", "r.x", "r.y", "r.z", "|", "force.x", "force.y", "force.z", "|", "torque.x", "torque.y", "torque.z");
			}
			printf("%10u %10.6f %10.6f %10.6f %5s %12.6f %12.6f %12.6f %5s %12.6f %12.6f %12.6f\n", lig_atom_id, r.x, r.y, r.z, "|", force.x, force.y, force.z, "|", torque_rot.x, torque_rot.y, torque_rot.z);
			//printf("%-10u %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f\n", lig_atom_id, r.x, r.y, r.z, force.x, force.y, force.z);
			#endif
		}

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-20s %-10.6f %-10.6f %-10.6f\n", "final torque: ", torque_rot.x, torque_rot.y, torque_rot.z);
		#endif

		// Derived from rotation.py/axisangle_to_q()
		// genes[3:7] = rotation.axisangle_to_q(torque, rad)
		float torque_length = fast_length(torque_rot);
		
		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-20s %-10.6f\n", "torque length: ", torque_length);
		#endif

		// Finding the quaternion that performs
		// the infinitesimal rotation around torque axis
		float4 quat_torque;
		#if 0
		quat_torque.w = native_cos(HALF_INFINITESIMAL_RADIAN);
		quat_torque.x = fast_normalize(torque_rot).x * native_sin(HALF_INFINITESIMAL_RADIAN);
		quat_torque.y = fast_normalize(torque_rot).y * native_sin(HALF_INFINITESIMAL_RADIAN);
		quat_torque.z = fast_normalize(torque_rot).z * native_sin(HALF_INFINITESIMAL_RADIAN);
		#endif

		quat_torque.w = COS_HALF_INFINITESIMAL_RADIAN;
		quat_torque.x = fast_normalize(torque_rot).x * SIN_HALF_INFINITESIMAL_RADIAN;
		quat_torque.y = fast_normalize(torque_rot).y * SIN_HALF_INFINITESIMAL_RADIAN;
		quat_torque.z = fast_normalize(torque_rot).z * SIN_HALF_INFINITESIMAL_RADIAN;

		#if defined (PRINT_GRAD_ROTATION_GENES)
		#if 0		
		printf("fast_normalize(torque_rot).x:%-.6f\n", fast_normalize(torque_rot).x);
		printf("fast_normalize(torque_rot).y:%-.6f\n", fast_normalize(torque_rot).y);
		printf("fast_normalize(torque_rot).z:%-.6f\n", fast_normalize(torque_rot).z);
		#endif

		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-20s %-10.6f\n", "INFINITESIMAL_RADIAN: ", INFINITESIMAL_RADIAN);

		printf("%-20s %-10.6f %-10.6f %-10.6f %-10.6f\n", "quat_torque (w,x,y,z): ", quat_torque.w, quat_torque.x, quat_torque.y, quat_torque.z);
		#endif

		// Converting quaternion gradients into orientation gradients 
		// Derived from autodockdev/motion.py/_get_cube3_gradient

		// This is where we are in the orientation axis-angle space
		// Equivalent to "current_oclacube" in autodockdev/motions.py
		// TODO: Check very initial input orientation genes
		float current_phi, current_theta, current_rotangle;
		current_phi      = genotype[3]; // phi      (in sexagesimal (DEG) unbounded)
		current_theta    = genotype[4]; // theta    (in sexagesimal (DEG) unbounded)
		current_rotangle = genotype[5]; // rotangle (in sexagesimal (DEG) unbounded)

		map_priv_angle(&current_phi);      // phi      (in DEG bounded within [0, 360])
		map_priv_angle(&current_theta);    // theta    (in DEG bounded within [0, 360])
		map_priv_angle(&current_rotangle); // rotangle (in DEG bounded within [0, 360])

		current_phi      = current_phi      * DEG_TO_RAD; // phi      (in RAD)
		current_theta    = current_theta    * DEG_TO_RAD; // theta    (in RAD)
 		current_rotangle = current_rotangle * DEG_TO_RAD; // rotangle (in RAD)

		bool is_theta_gt_pi = (current_theta > PI_FLOAT) ? true: false;

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s %-10.6f %-10.6f %-10.6f\n", "current_axisangle (1,2,3): ", current_phi, current_theta, current_rotangle);
		#endif		

		// This is where we are in quaternion space
		// current_q = oclacube_to_quaternion(angles)
		float4 current_q;

		// Axis of rotation
		float rotaxis_x = native_sin(current_theta) * native_cos(current_phi);
		float rotaxis_y = native_sin(current_theta) * native_sin(current_phi);
		float rotaxis_z = native_cos(current_theta);
		
		float ang;
		ang = current_rotangle * 0.5f;
		current_q.w = native_cos(ang);
		current_q.x = rotaxis_x * native_sin(ang);
		current_q.y = rotaxis_y * native_sin(ang);
		current_q.z = rotaxis_z * native_sin(ang);

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s %-10.6f %-10.6f %-10.6f %-10.6f\n", "current_q (w,x,y,z): ", current_q.w, current_q.x, current_q.y, current_q.z);
		#endif

		// This is where we want to be in quaternion space
		float4 target_q;

		// target_q = rotation.q_mult(q, current_q)
		// Derived from autodockdev/rotation.py/q_mult()
		// In our terms means q_mult(quat_{w|x|y|z}, current_q{w|x|y|z})
		target_q.w = quat_torque.w*current_q.w - quat_torque.x*current_q.x - quat_torque.y*current_q.y - quat_torque.z*current_q.z;// w
		target_q.x = quat_torque.w*current_q.x + quat_torque.x*current_q.w + quat_torque.y*current_q.z - quat_torque.z*current_q.y;// x
		target_q.y = quat_torque.w*current_q.y + quat_torque.y*current_q.w + quat_torque.z*current_q.x - quat_torque.x*current_q.z;// y
		target_q.z = quat_torque.w*current_q.z + quat_torque.z*current_q.w + quat_torque.x*current_q.y - quat_torque.y*current_q.x;// z
		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s %-10.6f %-10.6f %-10.6f %-10.6f\n", "target_q (w,x,y,z): ", target_q.w, target_q.x, target_q.y, target_q.z);
		#endif

		// This is where we want to be in the orientation axis-angle space
		float target_phi, target_theta, target_rotangle;

		// target_oclacube = quaternion_to_oclacube(target_q, theta_larger_than_pi)
		// Derived from autodockdev/motions.py/quaternion_to_oclacube()
		// In our terms means quaternion_to_oclacube(target_q{w|x|y|z}, theta_larger_than_pi)

		ang = acos(target_q.w);
		target_rotangle = 2.0f * ang;

		float inv_sin_ang = native_recip(native_sin(ang));
		rotaxis_x = target_q.x * inv_sin_ang;
		rotaxis_y = target_q.y * inv_sin_ang;
		rotaxis_z = target_q.z * inv_sin_ang;

		target_theta = acos(rotaxis_z);

		if (is_theta_gt_pi == false) {
			target_phi   = fmod((atan2( rotaxis_y,  rotaxis_x) + PI_TIMES_2), PI_TIMES_2);
		} else {
			target_phi   = fmod((atan2(-rotaxis_y, -rotaxis_x) + PI_TIMES_2), PI_TIMES_2);
			target_theta = PI_TIMES_2 - target_theta;
		}

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s %-10.6f %-10.6f %-10.6f\n", "target_axisangle (1,2,3): ", target_phi, target_theta, target_rotangle);
		#endif
		
		// The infinitesimal rotation will produce an infinitesimal displacement
		// in shoemake space. This is to guarantee that the direction of
		// the displacement in shoemake space is not distorted.
		// The correct amount of displacement in shoemake space is obtained
		// by multiplying the infinitesimal displacement by shoemake_scaling:
		float orientation_scaling = torque_length * INV_INFINITESIMAL_RADIAN;

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s %-10.6f\n", "orientation_scaling: ", orientation_scaling);
		#endif

		// Derivates in cube3
		float grad_phi, grad_theta, grad_rotangle;
		/*
		grad_phi      = orientation_scaling * (target_phi      - current_phi);
		grad_theta    = orientation_scaling * (target_theta    - current_theta);
		grad_rotangle = orientation_scaling * (target_rotangle - current_rotangle);
		*/
		grad_phi      = orientation_scaling * (fmod(target_phi 	 - current_phi 	    + PI_FLOAT, PI_TIMES_2) - PI_FLOAT);
		grad_theta    = orientation_scaling * (fmod(target_theta    - current_theta    + PI_FLOAT, PI_TIMES_2) - PI_FLOAT);
		grad_rotangle = orientation_scaling * (fmod(target_rotangle - current_rotangle + PI_FLOAT, PI_TIMES_2) - PI_FLOAT);

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s \n", "grad_axisangle (1,2,3) - before empirical scaling: ");
		printf("%-13s %-13s %-13s \n", "grad_phi", "grad_theta", "grad_rotangle");
		printf("%-13.6f %-13.6f %-13.6f\n", grad_phi, grad_theta, grad_rotangle);
		#endif

		// Correcting theta gradients interpolating 
		// values from correction look-up-tables
		// (X0,Y0) and (X1,Y1) are known points
		// How to find the Y value in the straight line between Y0 and Y1,
		// corresponding to a certain X?
		/*
			| dependence_on_theta_const
			| dependence_on_rotangle_const
			|
			|
			|                        Y1
			|
			|             Y=?
			|    Y0
			|_________________________________ angle_const
			     X0         X        X1
		*/

		// Finding the index-position of "grad_delta" in the "angle_const" array
		//uint index_theta    = floor(native_divide(current_theta    - angle_const[0], angle_delta));
		//uint index_rotangle = floor(native_divide(current_rotangle - angle_const[0], angle_delta));
		uint index_theta    = floor((current_theta    - angle_const[0]) * inv_angle_delta);
		uint index_rotangle = floor((current_rotangle - angle_const[0]) * inv_angle_delta);

		// Interpolating theta values
		// X0 -> index - 1
		// X1 -> index + 1
		// Expresed as weighted average:
		// Y = [Y0 * ((X1 - X) / (X1-X0))] +  [Y1 * ((X - X0) / (X1-X0))]
		// Simplified for GPU (less terms):
		// Y = [Y0 * (X1 - X) + Y1 * (X - X0)] / (X1 - X0)
		// Taking advantage of constant:
		// Y = [Y0 * (X1 - X) + Y1 * (X - X0)] * inv_angle_delta

		float X0, Y0;
		float X1, Y1;
		float dependence_on_theta; //Y = dependence_on_theta

		// Using interpolation on out-of-bounds elements results in hang
		if ((index_theta <= 0) || (index_theta >= 999))
		{
			dependence_on_theta = dependence_on_theta_const[stick_to_bounds(index_theta,0,999)];
		} else
		{
			X0 = angle_const[index_theta];
			X1 = angle_const[index_theta+1];
			Y0 = dependence_on_theta_const[index_theta];
			Y1 = dependence_on_theta_const[index_theta+1];
			dependence_on_theta = (Y0 * (X1-current_theta) + Y1 * (current_theta-X0)) * inv_angle_delta;
		}

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s %-10.6f\n", "dependence_on_theta: ", dependence_on_theta);
		#endif

		// Interpolating rotangle values
		float dependence_on_rotangle; // Y = dependence_on_rotangle
		// Using interpolation on previous and/or next elements results in hang
		// Using interpolation on out-of-bounds elements results in hang
		if ((index_rotangle <= 0) || (index_rotangle >= 999))
		{
			dependence_on_rotangle = dependence_on_rotangle_const[stick_to_bounds(index_rotangle,0,999)];
		} else
		{
			X0 = angle_const[index_rotangle];
			X1 = angle_const[index_rotangle+1];
			Y0 = dependence_on_rotangle_const[index_rotangle];
			Y1 = dependence_on_rotangle_const[index_rotangle+1];
			dependence_on_rotangle = (Y0 * (X1-current_rotangle) + Y1 * (current_rotangle-X0)) * inv_angle_delta;
		}

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s %-10.6f\n", "dependence_on_rotangle: ", dependence_on_rotangle);
		#endif

		// Setting gradient rotation-related genotypes in cube
		// Multiplicating by DEG_TO_RAD is to make it uniform to DEG (see torsion gradients)
		gradient_genotype[3] = native_divide(grad_phi, (dependence_on_theta * dependence_on_rotangle)) * DEG_TO_RAD;
		gradient_genotype[4] = native_divide(grad_theta, dependence_on_rotangle)                       * DEG_TO_RAD;
		gradient_genotype[5] = grad_rotangle                                                           * DEG_TO_RAD;
		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s \n", "grad_axisangle (1,2,3) - after empirical scaling: ");
		printf("%-13s %-13s %-13s \n", "grad_phi", "grad_theta", "grad_rotangle");
		printf("%-13.6f %-13.6f %-13.6f\n", gradient_genotype[3], gradient_genotype[4], gradient_genotype[5]);
		#endif
	}

	// ------------------------------------------
	// Obtaining torsion-related gradients
	// ------------------------------------------

	for ( int rotbond_id = tidx;
	          rotbond_id < dockpars_num_of_genes-6;
	          rotbond_id +=NUM_OF_THREADS_PER_BLOCK)
	{
		#if defined (PRINT_GRAD_TORSION_GENES)
		if (rotbond_id == 0) {
			printf("\n%s\n", "NOTE: torsion gradients are calculated by many work-items");
		}
		#endif
		// Querying ids of atoms belonging to the rotatable bond in question
		int atom1_id = rotbonds_const[2*rotbond_id];
		int atom2_id = rotbonds_const[2*rotbond_id+1];

		float3 atomRef_coords;
		atomRef_coords.x = calc_coords[atom1_id].x;
		atomRef_coords.y = calc_coords[atom1_id].y;
		atomRef_coords.z = calc_coords[atom1_id].z;

		#if defined (PRINT_GRAD_TORSION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-5s %3u \n\t %-5s %3i \n\t %-5s %3i\n", "gene: ", (rotbond_id+6), "atom1: ", atom1_id, "atom2: ", atom2_id);
		#endif

		float3 rotation_unitvec;
		/*
		rotation_unitvec.x = kerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id];
		rotation_unitvec.y = kerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id+1];
		rotation_unitvec.z = kerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id+2];
		*/
		rotation_unitvec.x = calc_coords[atom2_id].x - calc_coords[atom1_id].x;
		rotation_unitvec.y = calc_coords[atom2_id].y - calc_coords[atom1_id].y;
		rotation_unitvec.z = calc_coords[atom2_id].z - calc_coords[atom1_id].z;
		rotation_unitvec = fast_normalize(rotation_unitvec);

		#if defined (PRINT_GRAD_TORSION_GENES)
		printf("\n");
		printf("%-15s \n\t %-10.6f %-10.6f %-10.6f\n", "unitvec: ", rotation_unitvec.x, rotation_unitvec.y, rotation_unitvec.z);
		#endif

		// Torque of torsions
		float3 torque_tor;
		torque_tor.x = 0.0f;
		torque_tor.y = 0.0f;
		torque_tor.z = 0.0f;
		// Iterating over each ligand atom that rotates 
		// if the bond in question rotates
		for ( int rotable_atom_cnt = 0;
		          rotable_atom_cnt<num_rotating_atoms_per_rotbond_const[rotbond_id];
		          rotable_atom_cnt++)
		{
			uint lig_atom_id = rotbonds_atoms_const[MAX_NUM_OF_ATOMS*rotbond_id + rotable_atom_cnt];
			// Calculating torque on point "A"
			// (could be any other point "B" along the rotation axis)
			float3 atom_coords;
			atom_coords.x = calc_coords[lig_atom_id].x;
			atom_coords.y = calc_coords[lig_atom_id].y;
			atom_coords.z = calc_coords[lig_atom_id].z;
			// Temporal variable to calculate translation differences.
			// They are converted back to Angstroms here
			float3 r;
			r.x = (atom_coords.x - atomRef_coords.x) * dockpars_grid_spacing;
			r.y = (atom_coords.y - atomRef_coords.y) * dockpars_grid_spacing;
			r.z = (atom_coords.z - atomRef_coords.z) * dockpars_grid_spacing;

			// Re-using "gradient_inter_*" for total gradient (inter+intra)
			float3 atom_force;
			atom_force.x = gradient_inter_x[lig_atom_id];
			atom_force.y = gradient_inter_y[lig_atom_id];
			atom_force.z = gradient_inter_z[lig_atom_id];
			torque_tor += cross(r, atom_force);
			#if defined (PRINT_GRAD_TORSION_GENES)
			if (rotable_atom_cnt == 0) {
				printf("\n %-30s %3i\n", "contributor for gene : ", (rotbond_id+6));
			}
			//printf("%-15s %-10u\n", "rotable_atom_cnt: ", rotable_atom_cnt);
			//printf("%-15s %-10u\n", "atom_id: ", lig_atom_id);
			printf("\t %-15s %-10.6f %-10.6f %-10.6f \t %-15s %-10.6f %-10.6f %-10.6f\n", "atom_coords: ", atom_coords.x, atom_coords.y, atom_coords.z, "atom_force: ", atom_force.x, atom_force.y, atom_force.z);
			//printf("%-15s %-10.6f %-10.6f %-10.6f\n", "r: ", r.x, r.y, r.z);
			//printf("%-15s %-10.6f %-10.6f %-10.6f\n", "atom_force: ", atom_force.x, atom_force.y, atom_force.z);
			//printf("%-15s %-10.6f %-10.6f %-10.6f\n", "torque_tor: ", torque_tor.x, torque_tor.y, torque_tor.z);
			#endif

		}
		#if defined (PRINT_GRAD_TORSION_GENES)
		printf("\n");
		#endif

		// Projecting torque on rotation axis
		float torque_on_axis = dot(rotation_unitvec, torque_tor);

		// Assignment of gene-based gradient
		gradient_genotype[rotbond_id+6] = torque_on_axis * DEG_TO_RAD /*(M_PI / 180.0f)*/;

		#if defined (PRINT_GRAD_TORSION_GENES)
		printf("gradient_torsion [%u] :%f\n", rotbond_id+6, gradient_genotype [rotbond_id+6]);
		#endif
		
	} // End of iterations over rotatable bonds
	//----------------------------------

	barrier(CLK_LOCAL_MEM_FENCE);

	#if defined (CONVERT_INTO_ANGSTROM_RADIAN)
	for ( int gene_cnt = tidx;
	          gene_cnt < dockpars_num_of_genes;
	          gene_cnt+= NUM_OF_THREADS_PER_BLOCK)
	{
		if (gene_cnt > 2) {
			gradient_genotype[gene_cnt] *= dockpars_grid_spacing * dockpars_grid_spacing * SCFACTOR_ANGSTROM_RADIAN;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	#endif
}
