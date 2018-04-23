/*

OCLADock, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.

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


//#define DEBUG_GRAD_TRANSLATION_GENES
//#define DEBUG_GRAD_ROTATION_GENES
//#define DEBUG_GRAD_TORSION_GENES
//#define DEBUG_ENERGY_KERNEL5

void gpu_calc_gradient(	    
				int    dockpars_rotbondlist_length,
				char   dockpars_num_of_atoms,
			    	char   dockpars_gridsize_x,
			    	char   dockpars_gridsize_y,
			    	char   dockpars_gridsize_z,
		 __global const float* restrict dockpars_fgrids, // This is too large to be allocated in __constant 
		            	char   dockpars_num_of_atypes,
		            	int    dockpars_num_of_intraE_contributors,
			    	float  dockpars_grid_spacing,
			    	float  dockpars_coeff_elec,
			    	float  dockpars_qasp,
			    	float  dockpars_coeff_desolv,

				// Some OpenCL compilers don't allow declaring 
				// local variables within non-kernel functions.
				// These local variables must be declared in a kernel, 
				// and then passed to non-kernel functions.
		    	__local float* genotype,
			__local float* energy,
		    	__local int*   run_id,

		    	__local float* calc_coords_x,
		    	__local float* calc_coords_y,
		    	__local float* calc_coords_z,

	             __constant float* atom_charges_const,
                     __constant char*  atom_types_const,
                     __constant char*  intraE_contributors_const,
                     __constant float* VWpars_AC_const,
                     __constant float* VWpars_BD_const,
                     __constant float* dspars_S_const,
                     __constant float* dspars_V_const,
                     __constant int*   rotlist_const,
                     __constant float* ref_coords_x_const,
                     __constant float* ref_coords_y_const,
                     __constant float* ref_coords_z_const,
                     __constant float* rotbonds_moving_vectors_const,
                     __constant float* rotbonds_unit_vectors_const,
                     __constant float* ref_orientation_quats_const,
		     __constant int*   rotbonds_const,
		     __constant int*   rotbonds_atoms_const,
		     __constant int*   num_rotating_atoms_per_rotbond_const

		    // Gradient-related arguments
		    // Calculate gradients (forces) for intermolecular energy
		    // Derived from autodockdev/maps.py
		    // "is_enabled_gradient_calc": enables gradient calculation.
		    // In Genetic-Generation: no need for gradients
		    // In Gradient-Minimizer: must calculate gradients
			,
			    int    dockpars_num_of_genes,
	    	    __local float* gradient_inter_x,
	            __local float* gradient_inter_y,
	            __local float* gradient_inter_z,
		    __local float* gradient_intra_x,
		    __local float* gradient_intra_y,
		    __local float* gradient_intra_z,
		    __local float* gradient_x,
		    __local float* gradient_y,
		    __local float* gradient_z,
	            __local float* gradient_per_intracontributor,
		    __local float* gradient_genotype			
)
{
	// Initializing gradients (forces) 
	// Derived from autodockdev/maps.py
	for (uint atom_id = get_local_id(0);
		  atom_id < dockpars_num_of_atoms;
		  atom_id+= NUM_OF_THREADS_PER_BLOCK) {
		// Intermolecular gradients
		gradient_inter_x[atom_id] = 0.0f;
		gradient_inter_y[atom_id] = 0.0f;
		gradient_inter_z[atom_id] = 0.0f;
		// Intramolecular gradients
		gradient_intra_x[atom_id] = 0.0f;
		gradient_intra_y[atom_id] = 0.0f;
		gradient_intra_z[atom_id] = 0.0f;
	}

	// Initializing gradients per intramolecular contributor pairs 
	for (uint intracontrib_atompair_id = get_local_id(0);
		  intracontrib_atompair_id < dockpars_num_of_intraE_contributors;
		  intracontrib_atompair_id+= NUM_OF_THREADS_PER_BLOCK) {
		gradient_per_intracontributor[intracontrib_atompair_id] = 0.0f;
	}

	// Initializing gradient genotypes
	for (uint gene_cnt = get_local_id(0);
		  gene_cnt < dockpars_num_of_genes;
		  gene_cnt+= NUM_OF_THREADS_PER_BLOCK) {
		gradient_genotype[gene_cnt] = 0.0f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	uchar g1 = dockpars_gridsize_x;
	uint  g2 = dockpars_gridsize_x * dockpars_gridsize_y;
  	uint  g3 = dockpars_gridsize_x * dockpars_gridsize_y * dockpars_gridsize_z;


	// ================================================
	// CALCULATING ATOMIC POSITIONS AFTER ROTATIONS
	// ================================================
	for (uint rotation_counter = get_local_id(0);
	          rotation_counter < dockpars_rotbondlist_length;
	          rotation_counter+=NUM_OF_THREADS_PER_BLOCK)
	{
		int rotation_list_element = rotlist_const[rotation_counter];

		if ((rotation_list_element & RLIST_DUMMY_MASK) == 0)	// If not dummy rotation
		{
			uint atom_id = rotation_list_element & RLIST_ATOMID_MASK;

			// Capturing atom coordinates
			float atom_to_rotate[3];

			if ((rotation_list_element & RLIST_FIRSTROT_MASK) != 0)	// If first rotation of this atom
			{
				atom_to_rotate[0] = ref_coords_x_const[atom_id];
				atom_to_rotate[1] = ref_coords_y_const[atom_id];
				atom_to_rotate[2] = ref_coords_z_const[atom_id];
			}
			else
			{
				atom_to_rotate[0] = calc_coords_x[atom_id];
				atom_to_rotate[1] = calc_coords_y[atom_id];
				atom_to_rotate[2] = calc_coords_z[atom_id];
			}

			// Capturing rotation vectors and angle
			float rotation_movingvec[3];

			float quatrot_left_x, quatrot_left_y, quatrot_left_z, quatrot_left_q;
			float quatrot_temp_x, quatrot_temp_y, quatrot_temp_z, quatrot_temp_q;

			if ((rotation_list_element & RLIST_GENROT_MASK) != 0)	// If general rotation
			{
				// Rotational genes in the Shoemake space are expressed in radians
				float u1 = genotype[3];
				float u2 = genotype[4];
				float u3 = genotype[5];

				// u1, u2, u3 should be within their valid range of [0,1]
				quatrot_left_q = native_sqrt(1 - u1) * native_sin(PI_TIMES_2*u2); 
				quatrot_left_x = native_sqrt(1 - u1) * native_cos(PI_TIMES_2*u2);
				quatrot_left_y = native_sqrt(u1)     * native_sin(PI_TIMES_2*u3);
				quatrot_left_z = native_sqrt(u1)     * native_cos(PI_TIMES_2*u3);

				rotation_movingvec[0] = genotype[0];
				rotation_movingvec[1] = genotype[1];
				rotation_movingvec[2] = genotype[2];
			}
			else	// If rotating around rotatable bond
			{
				uint rotbond_id = (rotation_list_element & RLIST_RBONDID_MASK) >> RLIST_RBONDID_SHIFT;

				float rotation_unitvec[3];
				rotation_unitvec[0] = rotbonds_unit_vectors_const[3*rotbond_id];
				rotation_unitvec[1] = rotbonds_unit_vectors_const[3*rotbond_id+1];
				rotation_unitvec[2] = rotbonds_unit_vectors_const[3*rotbond_id+2];
				float rotation_angle = genotype[6+rotbond_id]*DEG_TO_RAD;

				rotation_movingvec[0] = rotbonds_moving_vectors_const[3*rotbond_id];
				rotation_movingvec[1] = rotbonds_moving_vectors_const[3*rotbond_id+1];
				rotation_movingvec[2] = rotbonds_moving_vectors_const[3*rotbond_id+2];

				// Performing additionally the first movement which 
				// is needed only if rotating around rotatable bond
				atom_to_rotate[0] -= rotation_movingvec[0];
				atom_to_rotate[1] -= rotation_movingvec[1];
				atom_to_rotate[2] -= rotation_movingvec[2];

				// Transforming torsion angles into quaternions
				rotation_angle  = native_divide(rotation_angle, 2.0f);
				float sin_angle = native_sin(rotation_angle);
				quatrot_left_q  = native_cos(rotation_angle);
				quatrot_left_x  = sin_angle*rotation_unitvec[0];
				quatrot_left_y  = sin_angle*rotation_unitvec[1];
				quatrot_left_z  = sin_angle*rotation_unitvec[2];
			}

			// Performing rotation
			if ((rotation_list_element & RLIST_GENROT_MASK) != 0)	// If general rotation,
										// two rotations should be performed
										// (multiplying the quaternions)
			{
				// Calculating quatrot_left*ref_orientation_quats_const,
				// which means that reference orientation rotation is the first
				quatrot_temp_q = quatrot_left_q;
				quatrot_temp_x = quatrot_left_x;
				quatrot_temp_y = quatrot_left_y;
				quatrot_temp_z = quatrot_left_z;

				quatrot_left_q = quatrot_temp_q*ref_orientation_quats_const[4*(*run_id)]-
						 quatrot_temp_x*ref_orientation_quats_const[4*(*run_id)+1]-
						 quatrot_temp_y*ref_orientation_quats_const[4*(*run_id)+2]-
						 quatrot_temp_z*ref_orientation_quats_const[4*(*run_id)+3];
				quatrot_left_x = quatrot_temp_q*ref_orientation_quats_const[4*(*run_id)+1]+
						 ref_orientation_quats_const[4*(*run_id)]*quatrot_temp_x+
						 quatrot_temp_y*ref_orientation_quats_const[4*(*run_id)+3]-
						 ref_orientation_quats_const[4*(*run_id)+2]*quatrot_temp_z;
				quatrot_left_y = quatrot_temp_q*ref_orientation_quats_const[4*(*run_id)+2]+
						 ref_orientation_quats_const[4*(*run_id)]*quatrot_temp_y+
						 ref_orientation_quats_const[4*(*run_id)+1]*quatrot_temp_z-
						 quatrot_temp_x*ref_orientation_quats_const[4*(*run_id)+3];
				quatrot_left_z = quatrot_temp_q*ref_orientation_quats_const[4*(*run_id)+3]+
						 ref_orientation_quats_const[4*(*run_id)]*quatrot_temp_z+
						 quatrot_temp_x*ref_orientation_quats_const[4*(*run_id)+2]-
						 ref_orientation_quats_const[4*(*run_id)+1]*quatrot_temp_y;
			}

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
			calc_coords_x[atom_id] = atom_to_rotate [0] + rotation_movingvec[0];
			calc_coords_y[atom_id] = atom_to_rotate [1] + rotation_movingvec[1];
			calc_coords_z[atom_id] = atom_to_rotate [2] + rotation_movingvec[2];

		} // End if-statement not dummy rotation

		barrier(CLK_LOCAL_MEM_FENCE);

	} // End rotation_counter for-loop

	// ================================================
	// CALCULATING INTERMOLECULAR GRADIENTS
	// ================================================
	for (uint atom_id = get_local_id(0);
	          atom_id < dockpars_num_of_atoms;
	          atom_id+= NUM_OF_THREADS_PER_BLOCK)
	{
		uint atom_typeid = atom_types_const[atom_id];
		float x = calc_coords_x[atom_id];
		float y = calc_coords_y[atom_id];
		float z = calc_coords_z[atom_id];
		float q = atom_charges_const[atom_id];

		if ((x < 0) || (y < 0) || (z < 0) || (x >= dockpars_gridsize_x-1)
				                  || (y >= dockpars_gridsize_y-1)
						  || (z >= dockpars_gridsize_z-1)){
			
			// Setting gradients (forces) penalties.
			// These are valid as long as they are high
			gradient_inter_x[atom_id] += 16777216.0f;
			gradient_inter_y[atom_id] += 16777216.0f;
			gradient_inter_z[atom_id] += 16777216.0f;
		}
		else
		{
			// Getting coordinates
			int x_low  = (int)floor(x); 
			int y_low  = (int)floor(y); 
			int z_low  = (int)floor(z);
			int x_high = (int)ceil(x); 
			int y_high = (int)ceil(y); 
			int z_high = (int)ceil(z);
			float dx = x - x_low; 
			float dy = y - y_low; 
			float dz = z - z_low;

			//printf("%-15s %-5u %-10.8f %-10.8f %-10.8f\n", "dx,dy,dz", atom_id, dx, dy, dz);

			// Calculating interpolation weights
			float weights[2][2][2];
			weights [0][0][0] = (1-dx)*(1-dy)*(1-dz);
			weights [1][0][0] = dx*(1-dy)*(1-dz);
			weights [0][1][0] = (1-dx)*dy*(1-dz);
			weights [1][1][0] = dx*dy*(1-dz);
			weights [0][0][1] = (1-dx)*(1-dy)*dz;
			weights [1][0][1] = dx*(1-dy)*dz;
			weights [0][1][1] = (1-dx)*dy*dz;
			weights [1][1][1] = dx*dy*dz;

			// Capturing affinity values
			uint ylow_times_g1  = y_low*g1;
			uint yhigh_times_g1 = y_high*g1;
		  	uint zlow_times_g2  = z_low*g2;
			uint zhigh_times_g2 = z_high*g2;

			// Grid offset
			uint offset_cube_000 = x_low  + ylow_times_g1  + zlow_times_g2;
			uint offset_cube_100 = x_high + ylow_times_g1  + zlow_times_g2;
			uint offset_cube_010 = x_low  + yhigh_times_g1 + zlow_times_g2;
			uint offset_cube_110 = x_high + yhigh_times_g1 + zlow_times_g2;
			uint offset_cube_001 = x_low  + ylow_times_g1  + zhigh_times_g2;
			uint offset_cube_101 = x_high + ylow_times_g1  + zhigh_times_g2;
			uint offset_cube_011 = x_low  + yhigh_times_g1 + zhigh_times_g2;
			uint offset_cube_111 = x_high + yhigh_times_g1 + zhigh_times_g2;

			uint mul_tmp = atom_typeid*g3;

			float cube[2][2][2];
			cube [0][0][0] = *(dockpars_fgrids + offset_cube_000 + mul_tmp);
			cube [1][0][0] = *(dockpars_fgrids + offset_cube_100 + mul_tmp);
			cube [0][1][0] = *(dockpars_fgrids + offset_cube_010 + mul_tmp);
		        cube [1][1][0] = *(dockpars_fgrids + offset_cube_110 + mul_tmp);
		        cube [0][0][1] = *(dockpars_fgrids + offset_cube_001 + mul_tmp);
			cube [1][0][1] = *(dockpars_fgrids + offset_cube_101 + mul_tmp);
                        cube [0][1][1] = *(dockpars_fgrids + offset_cube_011 + mul_tmp);
                        cube [1][1][1] = *(dockpars_fgrids + offset_cube_111 + mul_tmp);

			// -------------------------------------------------------------------
			// Deltas dx, dy, dz are already normalized 
			// (by host/src/getparameters.cpp) in OCLaDock.
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
			x10 = cube [1][0][0] - cube [0][0][0]; // z = 0
			x52 = cube [1][1][0] - cube [0][1][0]; // z = 0
			x43 = cube [1][0][1] - cube [0][0][1]; // z = 1
			x76 = cube [1][1][1] - cube [0][1][1]; // z = 1
			vx_z0 = (1 - dy) * x10 + dy * x52;     // z = 0
			vx_z1 = (1 - dy) * x43 + dy * x76;     // z = 1
			gradient_inter_x[atom_id] += (1 - dz) * vx_z0 + dz * vx_z1;

			// Vector in y-direction
			y20 = cube[0][1][0] - cube [0][0][0];	// z = 0
			y51 = cube[1][1][0] - cube [1][0][0];	// z = 0
			y63 = cube[0][1][1] - cube [0][0][1];	// z = 1
			y74 = cube[1][1][1] - cube [1][0][1];	// z = 1
			vy_z0 = (1 - dx) * y20 + dx * y51;	// z = 0
			vy_z1 = (1 - dx) * y63 + dx * y74;	// z = 1
			gradient_inter_y[atom_id] += (1 - dz) * vy_z0 + dz * vy_z1;

			// Vectors in z-direction
			z30 = cube [0][0][1] - cube [0][0][0];	// y = 0
			z41 = cube [1][0][1] - cube [1][0][0];	// y = 0
			z62 = cube [0][1][1] - cube [0][1][0];	// y = 1 
			z75 = cube [1][1][1] - cube [1][1][0];	// y = 1
			vz_y0 = (1 - dx) * z30 + dx * z41;	// y = 0
			vz_y1 = (1 - dx) * z62 + dx * z75;	// y = 1
			gradient_inter_z[atom_id] += (1 - dy) * vz_y0 + dy * vz_y1;

			//printf("%-15s %-3u %-10.8f %-10.8f %-10.8f %-10.8f %-10.8f %-10.8f\n", "atom aff", atom_id, vx_z0, vx_z1, vy_z0, vy_z1, vz_y0, vz_y1);

			// -------------------------------------------------------------------
			// Calculating gradients (forces) corresponding to 
			// "elec" intermolecular energy
			// Derived from autodockdev/maps.py
			// -------------------------------------------------------------------

			// Capturing electrostatic values
			atom_typeid = dockpars_num_of_atypes;

			mul_tmp = atom_typeid*g3;
			cube [0][0][0] = *(dockpars_fgrids + offset_cube_000 + mul_tmp);
			cube [1][0][0] = *(dockpars_fgrids + offset_cube_100 + mul_tmp);
      			cube [0][1][0] = *(dockpars_fgrids + offset_cube_010 + mul_tmp);
      			cube [1][1][0] = *(dockpars_fgrids + offset_cube_110 + mul_tmp);
		       	cube [0][0][1] = *(dockpars_fgrids + offset_cube_001 + mul_tmp);
		        cube [1][0][1] = *(dockpars_fgrids + offset_cube_101 + mul_tmp);
		        cube [0][1][1] = *(dockpars_fgrids + offset_cube_011 + mul_tmp);
		        cube [1][1][1] = *(dockpars_fgrids + offset_cube_111 + mul_tmp);

			// Vector in x-direction
			x10 = cube [1][0][0] - cube [0][0][0]; // z = 0
			x52 = cube [1][1][0] - cube [0][1][0]; // z = 0
			x43 = cube [1][0][1] - cube [0][0][1]; // z = 1
			x76 = cube [1][1][1] - cube [0][1][1]; // z = 1
			vx_z0 = (1 - dy) * x10 + dy * x52;     // z = 0
			vx_z1 = (1 - dy) * x43 + dy * x76;     // z = 1
			gradient_inter_x[atom_id] += q * ((1 - dz) * vx_z0 + dz * vx_z1);

			// Vector in y-direction
			y20 = cube[0][1][0] - cube [0][0][0];	// z = 0
			y51 = cube[1][1][0] - cube [1][0][0];	// z = 0
			y63 = cube[0][1][1] - cube [0][0][1];	// z = 1
			y74 = cube[1][1][1] - cube [1][0][1];	// z = 1
			vy_z0 = (1 - dx) * y20 + dx * y51;	// z = 0
			vy_z1 = (1 - dx) * y63 + dx * y74;	// z = 1
			gradient_inter_y[atom_id] += q *((1 - dz) * vy_z0 + dz * vy_z1);

			// Vectors in z-direction
			z30 = cube [0][0][1] - cube [0][0][0];	// y = 0
			z41 = cube [1][0][1] - cube [1][0][0];	// y = 0
			z62 = cube [0][1][1] - cube [0][1][0];	// y = 1 
			z75 = cube [1][1][1] - cube [1][1][0];	// y = 1
			vz_y0 = (1 - dx) * z30 + dx * z41;	// y = 0
			vz_y1 = (1 - dx) * z62 + dx * z75;	// y = 1
			gradient_inter_z[atom_id] += q *((1 - dy) * vz_y0 + dy * vz_y1);

			//printf("%-15s %-3u %-10.8f %-10.8f %-10.8f %-10.8f %-10.8f %-10.8f\n", "elec", atom_id, vx_z0, vx_z1, vy_z0, vy_z1, vz_y0, vz_y1);

			// -------------------------------------------------------------------
			// Calculating gradients (forces) corresponding to 
			// "dsol" intermolecular energy
			// Derived from autodockdev/maps.py
			// -------------------------------------------------------------------

			// Capturing desolvation values
			atom_typeid = dockpars_num_of_atypes+1;

			mul_tmp = atom_typeid*g3;
			cube [0][0][0] = *(dockpars_fgrids + offset_cube_000 + mul_tmp);
			cube [1][0][0] = *(dockpars_fgrids + offset_cube_100 + mul_tmp);
      			cube [0][1][0] = *(dockpars_fgrids + offset_cube_010 + mul_tmp);
      			cube [1][1][0] = *(dockpars_fgrids + offset_cube_110 + mul_tmp);
      			cube [0][0][1] = *(dockpars_fgrids + offset_cube_001 + mul_tmp);
      			cube [1][0][1] = *(dockpars_fgrids + offset_cube_101 + mul_tmp);
      			cube [0][1][1] = *(dockpars_fgrids + offset_cube_011 + mul_tmp);
      			cube [1][1][1] = *(dockpars_fgrids + offset_cube_111 + mul_tmp);

			// Vector in x-direction
			x10 = cube [1][0][0] - cube [0][0][0]; // z = 0
			x52 = cube [1][1][0] - cube [0][1][0]; // z = 0
			x43 = cube [1][0][1] - cube [0][0][1]; // z = 1
			x76 = cube [1][1][1] - cube [0][1][1]; // z = 1
			vx_z0 = (1 - dy) * x10 + dy * x52;     // z = 0
			vx_z1 = (1 - dy) * x43 + dy * x76;     // z = 1
			gradient_inter_x[atom_id] += fabs(q) * ((1 - dz) * vx_z0 + dz * vx_z1);

			// Vector in y-direction
			y20 = cube[0][1][0] - cube [0][0][0];	// z = 0
			y51 = cube[1][1][0] - cube [1][0][0];	// z = 0
			y63 = cube[0][1][1] - cube [0][0][1];	// z = 1
			y74 = cube[1][1][1] - cube [1][0][1];	// z = 1
			vy_z0 = (1 - dx) * y20 + dx * y51;	// z = 0
			vy_z1 = (1 - dx) * y63 + dx * y74;	// z = 1
			gradient_inter_y[atom_id] += fabs(q) *((1 - dz) * vy_z0 + dz * vy_z1);

			// Vectors in z-direction
			z30 = cube [0][0][1] - cube [0][0][0];	// y = 0
			z41 = cube [1][0][1] - cube [1][0][0];	// y = 0
			z62 = cube [0][1][1] - cube [0][1][0];	// y = 1 
			z75 = cube [1][1][1] - cube [1][1][0];	// y = 1
			vz_y0 = (1 - dx) * z30 + dx * z41;	// y = 0
			vz_y1 = (1 - dx) * z62 + dx * z75;	// y = 1
			gradient_inter_z[atom_id] += fabs(q) *((1 - dy) * vz_y0 + dy * vz_y1);

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
	for (uint contributor_counter = get_local_id(0);
	          contributor_counter < dockpars_num_of_intraE_contributors;
	          contributor_counter+= NUM_OF_THREADS_PER_BLOCK)
	{
		// Getting atom IDs
		uint atom1_id = intraE_contributors_const[3*contributor_counter];
		uint atom2_id = intraE_contributors_const[3*contributor_counter+1];
	
		/*
		printf ("%-5u %-5u %-5u\n", contributor_counter, atom1_id, atom2_id);
		*/
		
		// Calculating vector components of vector going
		// from first atom's to second atom's coordinates
		float subx = calc_coords_x[atom1_id] - calc_coords_x[atom2_id];
		float suby = calc_coords_y[atom1_id] - calc_coords_y[atom2_id];
		float subz = calc_coords_z[atom1_id] - calc_coords_z[atom2_id];

		// Calculating atomic distance
		float atomic_distance = native_sqrt(subx*subx + suby*suby + subz*subz)*dockpars_grid_spacing;

		if (atomic_distance < 1.0f)
			atomic_distance = 1.0f;

		// Calculating gradient contributions
		if ((atomic_distance < 8.0f) && (atomic_distance < 20.48f))
		{
			// Getting type IDs
			uint atom1_typeid = atom_types_const[atom1_id];
			uint atom2_typeid = atom_types_const[atom2_id];
			//printf ("%-5u %-5u %-5u\n", contributor_counter, atom1_id, atom2_id);

			// Calculating van der Waals / hydrogen bond term
			gradient_per_intracontributor[contributor_counter] += native_divide (-12*VWpars_AC_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],
									                     native_powr(atomic_distance, 13)
									       		    );

			if (intraE_contributors_const[3*contributor_counter+2] == 1) {	//H-bond
				gradient_per_intracontributor[contributor_counter] += native_divide (10*VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],
										                     native_powr(atomic_distance, 11)
												    );
			}
			else {	//van der Waals
				gradient_per_intracontributor[contributor_counter] += native_divide (6*VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],
										                     native_powr(atomic_distance, 7)
										                    );
			}

			// Calculating electrostatic term
			// http://www.wolframalpha.com/input/?i=1%2F(x*(A%2B(B%2F(1%2BK*exp(-h*B*x)))))
			float upper = DIEL_A*native_powr(native_exp(DIEL_B_TIMES_H*atomic_distance) + DIEL_K, 2) + (DIEL_B)*native_exp(DIEL_B_TIMES_H*atomic_distance)*(DIEL_B_TIMES_H_TIMES_K*atomic_distance + native_exp(DIEL_B_TIMES_H*atomic_distance) + DIEL_K);
		
			float lower = native_powr(atomic_distance, 2) * native_powr(DIEL_A * (native_exp(DIEL_B_TIMES_H*atomic_distance) + DIEL_K) + DIEL_B * native_exp(DIEL_B_TIMES_H*atomic_distance), 2);

        		gradient_per_intracontributor[contributor_counter] +=  -dockpars_coeff_elec * atom_charges_const[atom1_id] * atom_charges_const[atom2_id] * native_divide (upper, lower);

			// Calculating desolvation term
			gradient_per_intracontributor[contributor_counter] += (
									       (dspars_S_const[atom1_typeid] + dockpars_qasp*fabs(atom_charges_const[atom1_id])) * dspars_V_const[atom2_typeid] +
							                       (dspars_S_const[atom2_typeid] + dockpars_qasp*fabs(atom_charges_const[atom2_id])) * dspars_V_const[atom1_typeid]
				        				      ) *
					                       			dockpars_coeff_desolv * -0.07716049382716049 * atomic_distance * native_exp(-0.038580246913580245*native_powr(atomic_distance, 2));

		}

	} // End contributor_counter for-loop (INTRAMOLECULAR ENERGY)

	barrier(CLK_LOCAL_MEM_FENCE);

	// Accumulating gradients from "gradient_per_intracontributor" for each each
	if (get_local_id(0) == 0) {
		for (uint contributor_counter = 0;
			  contributor_counter < dockpars_num_of_intraE_contributors;
			  contributor_counter ++) {

			// Getting atom IDs
			uint atom1_id = intraE_contributors_const[3*contributor_counter];
			uint atom2_id = intraE_contributors_const[3*contributor_counter+1];

			// Calculating xyz distances in Angstroms of vector
			// that goes from "atom1_id"-to-"atom2_id"
			float subx = (calc_coords_x[atom2_id] - calc_coords_x[atom1_id]);
			float suby = (calc_coords_y[atom2_id] - calc_coords_y[atom1_id]);
			float subz = (calc_coords_z[atom2_id] - calc_coords_z[atom1_id]);
			float dist = native_sqrt(subx*subx + suby*suby + subz*subz);

			float subx_div_dist = native_divide(subx, dist);
			float suby_div_dist = native_divide(suby, dist);
			float subz_div_dist = native_divide(subz, dist);

			// Calculating gradients in xyz components.
			// Gradients for both atoms in a single contributor pair
			// have the same magnitude, but opposite directions
			gradient_intra_x[atom1_id] -= gradient_per_intracontributor[contributor_counter] * subx_div_dist;
			gradient_intra_y[atom1_id] -= gradient_per_intracontributor[contributor_counter] * suby_div_dist;
			gradient_intra_z[atom1_id] -= gradient_per_intracontributor[contributor_counter] * subz_div_dist;

			gradient_intra_x[atom2_id] += gradient_per_intracontributor[contributor_counter] * subx_div_dist;
			gradient_intra_y[atom2_id] += gradient_per_intracontributor[contributor_counter] * suby_div_dist;
			gradient_intra_z[atom2_id] += gradient_per_intracontributor[contributor_counter] * subz_div_dist;

			//printf("%-20s %-10u %-5u %-5u %-10.8f\n", "grad_intracontrib", contributor_counter, atom1_id, atom2_id, gradient_per_intracontributor[contributor_counter]);
		}
	}
	

	barrier(CLK_LOCAL_MEM_FENCE);

	// Accumulating inter- and intramolecular gradients
	for (uint atom_cnt = get_local_id(0);
		  atom_cnt < dockpars_num_of_atoms;
		  atom_cnt+= NUM_OF_THREADS_PER_BLOCK) {

		// Grid gradients were calculated in the grid space,
		// so they have to be put back in Angstrom.

		// Intramolecular gradients were already in Angstrom,
		// so no scaling for them is required.
		gradient_inter_x[atom_cnt] = native_divide(gradient_inter_x[atom_cnt], dockpars_grid_spacing);
		gradient_inter_y[atom_cnt] = native_divide(gradient_inter_y[atom_cnt], dockpars_grid_spacing);
		gradient_inter_z[atom_cnt] = native_divide(gradient_inter_z[atom_cnt], dockpars_grid_spacing);

		gradient_x[atom_cnt] = gradient_inter_x[atom_cnt] + gradient_intra_x[atom_cnt];
		gradient_y[atom_cnt] = gradient_inter_y[atom_cnt] + gradient_intra_y[atom_cnt];
		gradient_z[atom_cnt] = gradient_inter_z[atom_cnt] + gradient_intra_z[atom_cnt];
	
		//printf("%-15s %-5u %-10.8f %-10.8f %-10.8f\n", "grad_grid", atom_cnt, gradient_inter_x[atom_cnt], gradient_inter_y[atom_cnt], gradient_inter_z[atom_cnt]);

		//printf("%-15s %-5u %-10.8f %-10.8f %-10.8f\n", "grad_intra", atom_cnt, gradient_intra_x[atom_cnt], gradient_intra_y[atom_cnt], gradient_intra_z[atom_cnt]);

		//printf("%-15s %-5u %-10.8f %-10.8f %-10.8f\n", "calc_coords", atom_cnt, calc_coords_x[atom_cnt], calc_coords_y[atom_cnt], calc_coords_z[atom_cnt]);

	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// ------------------------------------------
	// Obtaining translation-related gradients
	// ------------------------------------------
	if (get_local_id(0) == 0) {
		for (uint lig_atom_id = 0;
			  lig_atom_id<dockpars_num_of_atoms;
			  lig_atom_id++) {
			gradient_genotype[0] += gradient_x[lig_atom_id]; // gradient for gene 0: gene x
			gradient_genotype[1] += gradient_y[lig_atom_id]; // gradient for gene 1: gene y
			gradient_genotype[2] += gradient_z[lig_atom_id]; // gradient for gene 2: gene z
		}

		// Scaling gradient for translational genes as 
		// their corresponding gradients were calculated in the space 
		// where these genes are in Angstrom,
		// but OCLaDock translational genes are within in grids
		gradient_genotype[0] *= dockpars_grid_spacing;
		gradient_genotype[1] *= dockpars_grid_spacing;
		gradient_genotype[2] *= dockpars_grid_spacing;

		#if defined (DEBUG_GRAD_TRANSLATION_GENES)
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
	if (get_local_id(0) == 1) {

		float3 torque_rot;
		torque_rot.x = 0.0f;
		torque_rot.y = 0.0f;
		torque_rot.z = 0.0f;

		#if defined (DEBUG_GRAD_ROTATION_GENES)
		printf("%-20s %-10.5f %-10.5f %-10.5f\n", "initial torque: ", torque_rot.x, torque_rot.y, torque_rot.z);
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
			
		for (uint lig_atom_id = 0;
			  lig_atom_id<dockpars_num_of_atoms;
			  lig_atom_id++) {
			r.x = (calc_coords_x[lig_atom_id] - about.x) * dockpars_grid_spacing; 
			r.y = (calc_coords_y[lig_atom_id] - about.y) * dockpars_grid_spacing;  
			r.z = (calc_coords_z[lig_atom_id] - about.z) * dockpars_grid_spacing; 

			float3 force;
			force.x	= gradient_x[lig_atom_id];
			force.y	= gradient_y[lig_atom_id]; 
			force.z	= gradient_z[lig_atom_id];

			torque_rot += cross(r, force);

			#if defined (DEBUG_GRAD_ROTATION_GENES)
			printf("%-20s %-10u\n", "contrib. of atom-id: ", lig_atom_id);
			printf("%-20s %-10.5f %-10.5f %-10.5f\n", "r             : ", r.x, r.y, r.z);
			printf("%-20s %-10.5f %-10.5f %-10.5f\n", "force         : ", force.x, force.y, force.z);
			printf("%-20s %-10.5f %-10.5f %-10.5f\n", "partial torque: ", torque_rot.x, torque_rot.y, torque_rot.z);
			printf("\n");
			#endif
		}

		#if defined (DEBUG_GRAD_ROTATION_GENES)
		printf("%-20s %-10.5f %-10.5f %-10.5f\n", "final torque: ", torque_rot.x, torque_rot.y, torque_rot.z);
		#endif

		// Derived from rotation.py/axisangle_to_q()
		// genes[3:7] = rotation.axisangle_to_q(torque, rad)
		float torque_length = fast_length(torque_rot);
		
		#if defined (DEBUG_GRAD_ROTATION_GENES)
		printf("%-20s %-10.5f\n", "torque length: ", torque_length);
		#endif

		// Infinitesimal rotation in radians
		const float infinitesimal_radian = 1E-5;

		// Finding the quaternion that performs
		// the infinitesimal rotation around torque axis
		float4 quat_torque;
		quat_torque.w = native_cos(infinitesimal_radian*0.5f);
		quat_torque.x = fast_normalize(torque_rot).x * native_sin(infinitesimal_radian*0.5f);
		quat_torque.y = fast_normalize(torque_rot).y * native_sin(infinitesimal_radian*0.5f);
		quat_torque.z = fast_normalize(torque_rot).z * native_sin(infinitesimal_radian*0.5f);

		#if defined (DEBUG_GRAD_ROTATION_GENES)
		printf("%-20s %-10.5f %-10.5f %-10.5f %-10.5f\n", "quat_torque (w,x,y,z): ", quat_torque.w, quat_torque.x, quat_torque.y, quat_torque.z);
		#endif

		// Converting quaternion gradients into Shoemake gradients 
		// Derived from autodockdev/motion.py/_get_cube3_gradient

		// This is where we are in Shoemake space
		float current_u1, current_u2, current_u3;
		current_u1 = genotype[3]; // check very initial input Shoemake genes
		current_u2 = genotype[4];
		current_u3 = genotype[5];
		
		#if defined (DEBUG_GRAD_ROTATION_GENES)
		printf("%-30s %-10.5f %-10.5f %-10.5f\n", "current_u (1,2,3): ", genotype[3], genotype[4], genotype[5]);
		#endif		

		// This is where we are in quaternion space
		// current_q = cube3_to_quaternion(current_u)
		float4 current_q;
		current_q.w = native_sqrt(1-current_u1) * native_sin(PI_TIMES_2*current_u2);
		current_q.x = native_sqrt(1-current_u1) * native_cos(PI_TIMES_2*current_u2);
		current_q.y = native_sqrt(current_u1)   * native_sin(PI_TIMES_2*current_u3);
		current_q.z = native_sqrt(current_u1)   * native_cos(PI_TIMES_2*current_u3);

		#if defined (DEBUG_GRAD_ROTATION_GENES)
		printf("%-30s %-10.8f %-10.8f %-10.8f %-10.8f\n", "current_q (w,x,y,z): ", current_q.w, current_q.x, current_q.y, current_q.z);
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
		#if defined (DEBUG_GRAD_ROTATION_GENES)
		printf("%-30s %-10.8f %-10.8f %-10.8f %-10.8f\n", "target_q (w,x,y,z): ", target_q.w, target_q.x, target_q.y, target_q.z);
		#endif

		// This is where we want to be in Shoemake space
		float target_u1, target_u2, target_u3;

		// target_u = quaternion_to_cube3(target_q)
		// Derived from autodockdev/motions.py/quaternion_to_cube3()
		// In our terms means quaternion_to_cube3(target_q{w|x|y|z})
		target_u1 = target_q.y*target_q.y + target_q.z*target_q.z;
		target_u2 = atan2(target_q.w, target_q.x);
		target_u3 = atan2(target_q.y, target_q.z);
		
		if (target_u2 < 0.0f)       { target_u2 += PI_TIMES_2; }
		if (target_u2 > PI_TIMES_2) { target_u2 -= PI_TIMES_2; }
		if (target_u3 < 0.0f) 	    { target_u3 += PI_TIMES_2; }
		if (target_u3 > PI_TIMES_2) { target_u3 -= PI_TIMES_2; }

		#if defined (DEBUG_GRAD_ROTATION_GENES)
		printf("%-30s %-10.8f %-10.8f %-10.8f\n", "target_u (1,2,3) - after mapping: ", target_u1, target_u2, target_u3);
		#endif
		
   		// The infinitesimal rotation will produce an infinitesimal displacement
    		// in shoemake space. This is to guarantee that the direction of
    		// the displacement in shoemake space is not distorted.
    		// The correct amount of displacement in shoemake space is obtained
		// by multiplying the infinitesimal displacement by shoemake_scaling:
		float shoemake_scaling = torque_length / infinitesimal_radian;

		// Derivates in cube3
		// "current_u2" and "current_u3" are mapped into 
		// the same range [0, 2PI] of "target_u2" and "target_u3"
		float grad_u1, grad_u2, grad_u3;
		grad_u1 = shoemake_scaling * (target_u1 - current_u1);
		grad_u2 = shoemake_scaling * (target_u2 - current_u2 * PI_TIMES_2);
		grad_u3 = shoemake_scaling * (target_u3 - current_u3 * PI_TIMES_2);

		#if defined (DEBUG_GRAD_ROTATION_GENES)
		printf("%-30s %-10.8f %-10.8f %-10.8f\n", "grad_u (1,2,3) - before emp. scaling: ", grad_u1, grad_u2, grad_u3);
		#endif
			
		// Empirical scaling
		float temp_u1 = genotype[3];
			
		if ((0.0f < temp_u1) && (temp_u1 < 1.0f)){
			grad_u1 *= ((1.0f/temp_u1) + (1.0f/(1.0f-temp_u1)));
		}
		grad_u2 *= 4.0f * (1.0f-temp_u1);
		grad_u3 *= 4.0f * temp_u1;

		#if defined (DEBUG_GRAD_ROTATION_GENES)
		printf("%-30s %-10.8f %-10.8f %-10.8f\n", "grad_u (1,2,3) - after emp. scaling: ", grad_u1, grad_u2, grad_u3);
		#endif
		
		// Setting gradient rotation-related genotypes in cube3.
		// Scaling gradient for u2 and u3 genes as 
		// their corresponding gradients were calculated in the space where u2/3 are within [0, 2PI]
		// but OCLaDock u2/3 genes are within [0, 1]
		gradient_genotype[3] = grad_u1;
		gradient_genotype[4] = grad_u2 * PI_TIMES_2; 
		gradient_genotype[5] = grad_u3 * PI_TIMES_2;
	}

	// ------------------------------------------
	// Obtaining torsion-related gradients
	// ------------------------------------------
	if (get_local_id(0) == 2) {

		for (uint rotbond_id = 0;
			  rotbond_id < dockpars_num_of_genes-6;
			  rotbond_id ++) {

			// Querying ids of atoms belonging to the rotatable bond in question
			int atom1_id = rotbonds_const[2*rotbond_id];
			int atom2_id = rotbonds_const[2*rotbond_id+1];

			float3 atomRef_coords;
			atomRef_coords.x = calc_coords_x[atom1_id];
			atomRef_coords.y = calc_coords_y[atom1_id];
			atomRef_coords.z = calc_coords_z[atom1_id];

			#if defined (DEBUG_GRAD_TORSION_GENES)
			printf("%-15s %-10u\n", "rotbond_id: ", rotbond_id);
			printf("%-15s %-10i\n", "atom1_id: ", atom1_id);
			printf("%-15s %-10.8f %-10.8f %-10.8f\n", "atom1_coords: ", calc_coords_x[atom1_id], calc_coords_y[atom1_id], calc_coords_z[atom1_id]);
			printf("%-15s %-10i\n", "atom2_id: ", atom2_id);
			printf("%-15s %-10.8f %-10.8f %-10.8f\n", "atom2_coords: ", calc_coords_x[atom2_id], calc_coords_y[atom2_id], calc_coords_z[atom2_id]);
			printf("\n");
			#endif		

			float3 rotation_unitvec;
			/*
			rotation_unitvec.x = rotbonds_unit_vectors_const[3*rotbond_id];
			rotation_unitvec.y = rotbonds_unit_vectors_const[3*rotbond_id+1];
			rotation_unitvec.z = rotbonds_unit_vectors_const[3*rotbond_id+2];
			*/
			rotation_unitvec.x = calc_coords_x[atom2_id] - calc_coords_x[atom1_id];
			rotation_unitvec.y = calc_coords_y[atom2_id] - calc_coords_y[atom1_id];
			rotation_unitvec.z = calc_coords_z[atom2_id] - calc_coords_z[atom1_id];
			rotation_unitvec = fast_normalize(rotation_unitvec);

			// Torque of torsions
			float3 torque_tor;
			torque_tor.x = 0.0f;
			torque_tor.y = 0.0f;
			torque_tor.z = 0.0f;

			// Iterating over each ligand atom that rotates 
			// if the bond in question rotates
			for (uint rotable_atom_cnt = 0;
				  rotable_atom_cnt<num_rotating_atoms_per_rotbond_const[rotbond_id];
				  rotable_atom_cnt++) {

				uint lig_atom_id = rotbonds_atoms_const[MAX_NUM_OF_ATOMS*rotbond_id + rotable_atom_cnt];

				// Calculating torque on point "A" 
				// (could be any other point "B" along the rotation axis)
				float3 atom_coords;
				atom_coords.x = calc_coords_x[lig_atom_id];
				atom_coords.y = calc_coords_y[lig_atom_id];
				atom_coords.z = calc_coords_z[lig_atom_id];

				// Temporal variable to calculate translation differences.
				// They are converted back to Angstroms here
				float3 r;
				r.x = (atom_coords.x - atomRef_coords.x) * dockpars_grid_spacing;
				r.y = (atom_coords.y - atomRef_coords.y) * dockpars_grid_spacing;
				r.z = (atom_coords.z - atomRef_coords.z) * dockpars_grid_spacing;

				float3 atom_force;
				atom_force.x = gradient_x[lig_atom_id]; 
				atom_force.y = gradient_y[lig_atom_id];
				atom_force.z = gradient_z[lig_atom_id];

				torque_tor += cross(r, atom_force);

				#if defined (DEBUG_GRAD_TORSION_GENES)
				printf("\n");
				printf("%-15s %-10u\n", "rotable_atom_cnt: ", rotable_atom_cnt);
				printf("%-15s %-10u\n", "atom_id: ", lig_atom_id);
				printf("%-15s %-10.8f %-10.8f %-10.8f\n", "atom_coords: ", atom_coords.x, atom_coords.y, atom_coords.z);
				printf("%-15s %-10.8f %-10.8f %-10.8f\n", "r: ", r.x, r.y, r.z);
				printf("%-15s %-10.8f %-10.8f %-10.8f\n", "unitvec: ", rotation_unitvec.x, rotation_unitvec.y, rotation_unitvec.z);
				printf("%-15s %-10.8f %-10.8f %-10.8f\n", "atom_force: ", atom_force.x, atom_force.y, atom_force.z);
				printf("%-15s %-10.8f %-10.8f %-10.8f\n", "torque_tor: ", torque_tor.x, torque_tor.y, torque_tor.z);
				#endif

			}
			#if defined (DEBUG_GRAD_TORSION_GENES)
			printf("\n");
			#endif

			// Projecting torque on rotation axis
			float torque_on_axis = dot(rotation_unitvec, torque_tor);

			// Assignment of gene-based gradient
			gradient_genotype[rotbond_id+6] = torque_on_axis * (M_PI / 180.0f);

			#if defined (DEBUG_GRAD_TORSION_GENES)
			printf("gradient_torsion [%u] :%f\n", rotbond_id+6, gradient_genotype [rotbond_id+6]);
			#endif
			
		} // End of iterations over rotatable bonds
	}

}
