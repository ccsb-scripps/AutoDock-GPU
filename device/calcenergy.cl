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


#include "calcenergy_basic.h"

// All related pragmas are in defines.h (accesible by host and device code)

void gpu_calc_energy(	    int    dockpars_rotbondlist_length,
			    char   dockpars_num_of_atoms,
			    char   dockpars_gridsize_x,
			    char   dockpars_gridsize_y,
			    char   dockpars_gridsize_z,
		#if defined (RESTRICT_ARGS)
			__global const float* restrict dockpars_fgrids, // cannot be allocated in __constant (too large)
		#else
			__global const float* dockpars_fgrids, 		// cannot be allocated in __constant (too large)
		#endif
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
		    __local float* partial_energies,

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

		    // Gradient-related arguments
		    // Calculate gradients (forces) for intermolecular energy
		    // Derived from autodockdev/maps.py
		
		    // "is_enabled_gradient_calc": enables gradient calculation.
		    // In Genetic-Generation: no need for gradients
		    // In Gradient-Minimizer: must calculate gradients
		    __local bool*  is_enabled_gradient_calc,
	    	    __local float* gradient_inter_x,
	            __local float* gradient_inter_y,
	            __local float* gradient_inter_z,
		    __local float* gradient_genotype	
			
)

//The GPU device function calculates the energy of the entity described by genotype, dockpars and the liganddata
//arrays in constant memory and returns it in the energy parameter. The parameter run_id has to be equal to the ID
//of the run whose population includes the current entity (which can be determined with blockIdx.x), since this
//determines which reference orientation should be used.
{
	partial_energies[get_local_id(0)] = 0.0f;

	// -------------------------------------------------------------------
	// Calculate gradients (forces) for intermolecular energy
	// Derived from autodockdev/maps.py
	// -------------------------------------------------------------------
	if (*is_enabled_gradient_calc) {
		for (uint atom_id = get_local_id(0);
		          atom_id < dockpars_num_of_atoms;
		          atom_id+= NUM_OF_THREADS_PER_BLOCK) {
			gradient_inter_x[atom_id] = 0.0f;
			gradient_inter_y[atom_id] = 0.0f;
			gradient_inter_z[atom_id] = 0.0f;
		}
	}

#if 0
	// Rotational genes in the Shoemake space expressed in radians
	float u1, u2, u3; 
	
	u1 = genotype[3];
	u2 = genotype[4]/**DEG_TO_RAD*/;
	u3 = genotype[5]/**DEG_TO_RAD*/;
#endif

#if defined (IMPROVE_GRID)
	// INTERMOLECULAR for-loop (intermediate results)
	// It stores a product of two chars
	//uint mul_tmp;

	uchar g1 = dockpars_gridsize_x;
	uint  g2 = dockpars_gridsize_x * dockpars_gridsize_y;
  	uint  g3 = dockpars_gridsize_x * dockpars_gridsize_y * dockpars_gridsize_z;
#else

#endif

	// ================================================
	// CALCULATE ATOMIC POSITIONS AFTER ROTATIONS
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
				// -------------------------------------------------------------------
				// Replacing rotation genes: from spherical space to Shoemake space
				// gene [0:2]: translation -> kept as original x, y, z
				// gene [3:5]: rotation    -> transformed into Shoemake (u1: adimensional, u2&u3: sexagesimal)
				// gene [6:N]: torsions	   -> kept as original angles	(all in sexagesimal)

				// Shoemake ranges:
				// u1: [0, 1]
				// u2: [0: 2PI] or [0: 360]

				// Random generator in the host is changed:
				// LCG (original, myrand()) -> CPP std (rand())
				// -------------------------------------------------------------------

				// Transforming Shoemake (u1, u2, u3) genes into quaternions
				// FIXME: add precision choices with preprocessor directives: 
				// NATIVE_PRECISION, HALF_PRECISION, Full precision

				// Rotational genes in the Shoemake space expressed in radians
				float u1, u2, u3; 
	
				u1 = genotype[3];
				u2 = genotype[4]/**DEG_TO_RAD*/;
				u3 = genotype[5]/**DEG_TO_RAD*/;

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
				// FIXME: add precision choices with preprocessor directives: 
				// NATIVE_PRECISION, HALF_PRECISION, Full precision
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

	// -------------------------------------------------------------------
	// Calculate gradients (forces) for intermolecular energy
	// Derived from autodockdev/maps.py
	// -------------------------------------------------------------------
	// Variables to store gradient of 
	// the intermolecular energy per each ligand atom

	// Some OpenCL compilers don't allow declaring 
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel, 
	// and then passed to non-kernel functions.
	/*
	__local float gradient_inter_x[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_y[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_z[MAX_NUM_OF_ATOMS];
	*/

	// Deltas dx, dy, dz are already normalized 
	// (by host/src/getparameters.cpp) in OCLaDock.
	// The correspondance between vertices in xyz axes is:
	// 0, 1, 2, 3, 4, 5, 6, 7  and  000, 100, 010, 001, 101, 110, 011, 111
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

	// ================================================
	// CALCULATE INTERMOLECULAR ENERGY
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
			partial_energies[get_local_id(0)] += 16777216.0f; //100000.0f;
			
			// -------------------------------------------------------------------
			// Calculate gradients (forces) for intermolecular energy
			// Derived from autodockdev/maps.py
			// -------------------------------------------------------------------

			if (*is_enabled_gradient_calc) {
				// Penalty values are valid as long as they are high
				gradient_inter_x[atom_id] += 16777216.0f;
				gradient_inter_y[atom_id] += 16777216.0f;
				gradient_inter_z[atom_id] += 16777216.0f;
			}
			// -------------------------------------------------------------------
		}
		else
		{
			// Get coordinates
			int x_low  = (int)floor(x); 
			int y_low  = (int)floor(y); 
			int z_low  = (int)floor(z);
			int x_high = (int)ceil(x); 
			int y_high = (int)ceil(y); 
			int z_high = (int)ceil(z);
			float dx = x - x_low; 
			float dy = y - y_low; 
			float dz = z - z_low;

			// Calculate interpolation weights
			float weights[2][2][2];
			weights [0][0][0] = (1-dx)*(1-dy)*(1-dz);
			weights [1][0][0] = dx*(1-dy)*(1-dz);
			weights [0][1][0] = (1-dx)*dy*(1-dz);
			weights [1][1][0] = dx*dy*(1-dz);
			weights [0][0][1] = (1-dx)*(1-dy)*dz;
			weights [1][0][1] = dx*(1-dy)*dz;
			weights [0][1][1] = (1-dx)*dy*dz;
			weights [1][1][1] = dx*dy*dz;

			// Capture affinity values
#if defined (IMPROVE_GRID)
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
			// Calculate gradients (forces) corresponding to 
			// "atype" intermolecular energy
			// Derived from autodockdev/maps.py
			// -------------------------------------------------------------------

			if (*is_enabled_gradient_calc) {
				// vector in x-direction
				/*
			 	x10 = grid[int(vertices[1])] - grid[int(vertices[0])] # z = 0
				x52 = grid[int(vertices[5])] - grid[int(vertices[2])] # z = 0
				x43 = grid[int(vertices[4])] - grid[int(vertices[3])] # z = 1
				x76 = grid[int(vertices[7])] - grid[int(vertices[6])] # z = 1
				vx_z0 = (1-yd) * x10 + yd * x52     #  z = 0
				vx_z1 = (1-yd) * x43 + yd * x76     #  z = 1
				gradient[0] = (1-zd) * vx_z0 + zd * vx_z1 
				*/

				x10 = cube [1][0][0] - cube [0][0][0]; // z = 0
				x52 = cube [1][1][0] - cube [0][1][0]; // z = 0
				x43 = cube [1][0][1] - cube [0][0][1]; // z = 1
				x76 = cube [1][1][1] - cube [0][1][1]; // z = 1
				vx_z0 = (1 - dy) * x10 + dy * x52;     // z = 0
				vx_z1 = (1 - dy) * x43 + dy * x76;     // z = 1
				gradient_inter_x[atom_id] += (1 - dz) * vx_z0 + dz * vx_z1;

				// vector in y-direction
				/*
				y20 = grid[int(vertices[2])] - grid[int(vertices[0])] # z = 0
				y51 = grid[int(vertices[5])] - grid[int(vertices[1])] # z = 0
				y63 = grid[int(vertices[6])] - grid[int(vertices[3])] # z = 1
			for (uint rotation_counter = get_local_id(0);
	          rotation_counter < dockpars_rotbondlist_length;
	          rotation_counter+=NUM_OF_THREADS_PER_BLOCK)
	{	y74 = grid[int(vertices[7])] - grid[int(vertices[4])] # z = 1
				vy_z0 = (1-xd) * y20 + xd * y51     #  z = 0
				vy_z1 = (1-xd) * y63 + xd * y74     #  z = 1
				gradient[1] = (1-zd) * vy_z0 + zd * vy_z1
				*/

				y20 = cube[0][1][0] - cube [0][0][0];	// z = 0
				y51 = cube[1][1][0] - cube [1][0][0];	// z = 0
				y63 = cube[0][1][1] - cube [0][0][1];	// z = 1
				y74 = cube[1][1][1] - cube [1][0][1];	// z = 1
				vy_z0 = (1 - dx) * y20 + dx * y51;	// z = 0
				vy_z1 = (1 - dx) * y63 + dx * y74;	// z = 1
				gradient_inter_y[atom_id] += (1 - dz) * vy_z0 + dz * vy_z1;

				// vectors in z-direction
				/*	
				z30 = grid[int(vertices[3])] - grid[int(vertices[0])] # y = 0
				z41 = grid[int(vertices[4])] - grid[int(vertices[1])] # y = 0
				z62 = grid[int(vertices[6])] - grid[int(vertices[2])] # y = 1
				z75 = grid[int(vertices[7])] - grid[int(vertices[5])] # y = 1
				vz_y0 = (1-xd) * z30 + xd * z41     # y = 0
				vz_y1 = (1-xd) * z62 + xd * z75     # y = 1
				gradient[2] = (1-yd) * vz_y0 + yd * vz_y1
				*/

				z30 = cube [0][0][1] - cube [0][0][0];	// y = 0
				z41 = cube [1][0][1] - cube [1][0][0];	// y = 0
				z62 = cube [0][1][1] - cube [0][1][0];	// y = 1 
				z75 = cube [1][1][1] - cube [1][1][0];	// y = 1
				vz_y0 = (1 - dx) * z30 + dx * z41;	// y = 0
				vz_y1 = (1 - dx) * z62 + dx * z75;	// y = 1
				gradient_inter_z[atom_id] += (1 - dy) * vz_y0 + dy * vz_y1;
			}
			// -------------------------------------------------------------------	
#else
			// -------------------------------------------------------------------
			// FIXME: this block within the "#else" preprocessor directive 
			// provides NO gradient corresponding to "atype" intermolecular energy
			// -------------------------------------------------------------------	

			cube [0][0][0] = GETGRIDVALUE(dockpars_fgrids, 
						      dockpars_gridsize_x,
						      dockparsdockpars_num_of_atoms;_gridsize_y, 
						      dockpars_gridsize_z,
						      atom_typeid, z_low, y_low, x_low);
			cube [1][0][0] = GETGRIDVALUE(dockpars_fgrids, 
						      dockpars_gridsize_x,
						      dockpars_gridsize_y, 
						      dockpars_gridsize_z,
						      atom_typeid, z_low, y_low, x_high);
			cube [0][1][0] = GETGRIDVALUE(dockpars_fgrids, 
						      dockpars_gridsize_x,
						      dockpars_gridsize_y, 
						      dockpars_gridsize_z,
						      atom_typeid, z_low, y_high, x_low);
			cube [1][1][0] = GETGRIDVALUE(dockpars_fgrids, 
						      dockpars_gridsize_x,
						      dockpars_gridsize_y, 
						      dockpars_gridsize_z,
						      atom_typeid, z_low, y_high, x_high);
			cube [0][0][1] = GETGRIDVALUE(dockpars_fgrids, 
						      dockpars_gridsize_x,
						      dockpars_gridsize_y, 
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_low, x_low);
			cube [1][0][1] = GETGRIDVALUE(dockpars_fgrids, 
						      dockpars_gridsize_x,
						      dockpars_gridsize_y, 
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_low, x_high);
			cube [0][1][1] = GETGRIDVALUE(dockpars_fgrids, 
						      dockpars_gridsize_x,
						      dockpars_gridsize_y, 
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_high, x_low);
			cube [1][1][1] = GETGRIDVALUE(dockpars_fgrids, 
						      dockpars_gridsize_x,
						      dockpars_gridsize_y, 
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_high, x_high);
#endif

			//calculating affinity energy
			partial_energies[get_local_id(0)] += TRILININTERPOL(cube, weights);

			//capturing electrostatic values
			atom_typeid = dockpars_num_of_atypes;

#if defined (IMPROVE_GRID)
			mul_tmp = atom_typeid*g3;
			cube [0][0][0] = *(dockpars_fgrids + offset_cube_000 + mul_tmp);
			cube [1][0][0] = *(dockpars_fgrids + offset_cube_100 + mul_tmp);
      			cube [0][1][0] = *(dockpars_fgrids + offset_cube_010 + mul_tmp);
      			cube [1][1][0] = *(dockpars_fgrids + offset_cube_110 + mul_tmp);
		       	cube [0][0][1] = *(dockpars_fgrids + offset_cube_001 + mul_tmp);
		        cube [1][0][1] = *(dockpars_fgrids + offset_cube_101 + mul_tmp);
		        cube [0][1][1] = *(dockpars_fgrids + offset_cube_011 + mul_tmp);
		        cube [1][1][1] = *(dockpars_fgrids + offset_cube_111 + mul_tmp);

			// -------------------------------------------------------------------
			// Calculate gradients (forces) corresponding to 
			// "elec" intermolecular energy
			// Derived from autodockdev/maps.py
			// -------------------------------------------------------------------

			if (*is_enabled_gradient_calc) {
				// vector in x-direction
				x10 = cube [1][0][0] - cube [0][0][0]; // z = 0
				x52 = cube [1][1][0] - cube [0][1][0]; // z = 0
				x43 = cube [1][0][1] - cube [0][0][1]; // z = 1
				x76 = cube [1][1][1] - cube [0][1][1]; // z = 1
				vx_z0 = (1 - dy) * x10 + dy * x52;     // z = 0
				vx_z1 = (1 - dy) * x43 + dy * x76;     // z = 1
				gradient_inter_x[atom_id] += (1 - dz) * vx_z0 + dz * vx_z1;

				// vector in y-direction
				y20 = cube[0][1][0] - cube [0][0][0];	// z = 0
				y51 = cube[1][1][0] - cube [1][0][0];	// z = 0
				y63 = cube[0][1][1] - cube [0][0][1];	// z = 1
				y74 = cube[1][1][1] - cube [1][0][1];	// z = 1
				vy_z0 = (1 - dx) * y20 + dx * y51;	// z = 0
				vy_z1 = (1 - dx) * y63 + dx * y74;	// z = 1
				gradient_inter_y[atom_id] += (1 - dz) * vy_z0 + dz * vy_z1;

				// vectors in z-direction
				z30 = cube [0][0][1] - cube [0][0][0];	// y = 0
				z41 = cube [1][0][1] - cube [1][0][0];	// y = 0
				z62 = cube [0][1][1] - cube [0][1][0];	// y = 1 
				z75 = cube [1][1][1] - cube [1][1][0];	// y = 1
				vz_y0 = (1 - dx) * z30 + dx * z41;	// y = 0
				vz_y1 = (1 - dx) * z62 + dx * z75;	// y = 1
				gradient_inter_z[atom_id] += (1 - dy) * vz_y0 + dy * vz_y1;
			}
			// -------------------------------------------------------------------
#else
			// -------------------------------------------------------------------
			// FIXME: this block within the "#else" preprocessor directive 
			// provides NO gradient corresponding to "elec" intermolecular energybas
			// -------------------------------------------------------------------

			cube [0][0][0] = GETGRIDVALUE(dockpars_fgrids, 
						      dockpars_gridsize_x,
						      dockpars_gridsize_y, 
						      dockpars_gridsize_z,
					 	      atom_typeid, z_low, y_low, x_low);
			cube [1][0][0] = GETGRIDVALUE(dockpars_fgrids, 
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_low, y_low, x_high);
			cube [0][1][0] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
                                                      dockpars_gridsize_z,
						      atom_typeid, z_low, y_high, x_low);
			cube [1][1][0] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,bas
						      atom_typeid, z_low, y_high, x_high);
			cube [0][0][1] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_low, x_low);
			cube [1][0][1] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_low, x_high);
			cube [0][1][1] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_high, x_low);
			cube [1][1][1] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_high, x_high);
#endif

			//calculating electrosatic energy
			partial_energies[get_local_id(0)] += q * TRILININTERPOL(cube, weights);

			//capturing desolvation values
			atom_typeid = dockpars_num_of_atypes+1;

#if defined (IMPROVE_GRID)
			mul_tmp = atom_typeid*g3;
			cube [0][0][0] = *(dockpars_fgrids + offset_cube_000 + mul_tmp);
			cube [1][0][0] = *(dockpars_fgrids + offset_cube_100 + mul_tmp);
      			cube [0][1][0] = *(dockpars_fgrids + offset_cube_010 + mul_tmp);
      			cube [1][1][0] = *(dockpars_fgrids + offset_cube_110 + mul_tmp);
      			cube [0][0][1] = *(dockpars_fgrids + offset_cube_001 + mul_tmp);
      			cube [1][0][1] = *(dockpars_fgrids + offset_cube_101 + mul_tmp);
      			cube [0][1][1] = *(dockpars_fgrids + offset_cube_011 + mul_tmp);
      			cube [1][1][1] = *(dockpars_fgrids + offset_cube_111 + mul_tmp);

			// -------------------------------------------------------------------
			// Calculate gradients (forces) corresponding to 
			// "dsol" intermolecular energy
			// Derived from autodockdev/maps.py
			// -------------------------------------------------------------------

			if (*is_enabled_gradient_calc) {
				// vector in x-direction
				x10 = cube [1][0][0] - cube [0][0][0]; // z = 0
				x52 = cube [1][1][0] - cube [0][1][0]; // z = 0
				x43 = cube [1][0][1] - cube [0][0][1]; // z = 1
				x76 = cube [1][1][1] - cube [0][1][1]; // z = 1
				vx_z0 = (1 - dy) * x10 + dy * x52;     // z = 0
				vx_z1 = (1 - dy) * x43 + dy * x76;     // z = 1
				gradient_inter_x[atom_id] += (1 - dz) * vx_z0 + dz * vx_z1;

				// vector in y-direction
				y20 = cube[0][1][0] - cube [0][0][0];	// z = 0
				y51 = cube[1][1][0] - cube [1][0][0];	// z = 0
				y63 = cube[0][1][1] - cube [0][0][1];	// z = 1
				y74 = cube[1][1][1] - cube [1][0][1];	// z = 1
				vy_z0 = (1 - dx) * y20 + dx * y51;	// z = 0
				vy_z1 = (1 - dx) * y63 + dx * y74;	// z = 1
				gradient_inter_y[atom_id] += (1 - dz) * vy_z0 + dz * vy_z1;

				// vectors in z-direction
				z30 = cube [0][0][1] - cube [0][0][0];	// y = 0
				z41 = cube [1][0][1] - cube [1][0][0];	// y = 0
				z62 = cube [0][1][1] - cube [0][1][0];	// y = 1 
				z75 = cube [1][1][1] - cube [1][1][0];	// y = 1
				vz_y0 = (1 - dx) * z30 + dx * z41;	// y = 0
				vz_y1 = (1 - dx) * z62 + dx * z75;	// y = 1
				gradient_inter_z[atom_id] += (1 - dy) * vz_y0 + dy * vz_y1;
			}
			// -------------------------------------------------------------------
#else
			// -------------------------------------------------------------------
			// FIXME: this block within the "#else" preprocessor directive 
			// provides NO gradient corresponding to "dsol" intermolecular energy
			// -------------------------------------------------------------------

			cube [0][0][0] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_low, y_low, x_low);
			cube [1][0][0] = GETGRIDVALUE(dockpars_fgridsu2,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_low, y_low, x_high);
			cube [0][1][0] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,rotbonds_unit_vectors_const
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_low, y_high, x_low);
			cube [1][1][0] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_low, y_high, x_high);
			cube [0][0][1] = GETGRIDVALUE(dockpars_fgrids, 
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_low, x_low);
			cube [1][0][1] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_low, x_high);
			cube [0][1][1] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_high, x_low);
			cube [1][1][1] = GETGRIDVALUE(dockpars_fgrids,
						      dockpars_gridsize_x,
						      dockpars_gridsize_y,
						      dockpars_gridsize_z,
						      atom_typeid, z_high, y_high, x_high);
#endif

			//calculating desolvation energy
			partial_energies[get_local_id(0)] += fabs(q) * TRILININTERPOL(cube, weights);
		}

	} // End atom_id for-loop (INTERMOLECULAR ENERGY)

	// In paper: intermolecular and internal energy calculation
	// are independent from each other, -> NO BARRIER NEEDED
  	// but require different operations,
	// thus, they can be executed only sequentially on the GPU.

	// ================================================
	// CALCULATE INTRAMOLECULAR ENERGY
	// ================================================
	for (uint contributor_counter = get_local_id(0);
	          contributor_counter < dockpars_num_of_intraE_contributors;
	          contributor_counter +=NUM_OF_THREADS_PER_BLOCK)
	{
		//getting atom IDs
		uint atom1_id = intraE_contributors_const[3*contributor_counter];
		uint atom2_id = intraE_contributors_const[3*contributor_counter+1];

		//calculating address of first atom's coordinates
		float subx = calc_coords_x[atom1_id];
		float suby = calc_coords_y[atom1_id];
		float subz = calc_coords_z[atom1_id];

		//calculating address of second atom's coordinates
		subx -= calc_coords_x[atom2_id];
		suby -= calc_coords_y[atom2_id];
		subz -= calc_coords_z[atom2_id];

		//calculating distance (atomic_distance)
#if defined (NATIVE_PRECISION)
		float atomic_distance = native_sqrt(subx*subx + suby*suby + subz*subz)*dockpars_grid_spacing;
#elif defined (HALF_PRECISION)
		float atomic_distance = half_sqrt(subx*subx + suby*suby + subz*subz)*dockpars_grid_spacing;
#else	// Full precision
		float atomic_distance = sqrt(subx*subx + suby*suby + subz*subz)*dockpars_grid_spacing;
#endif

		if (atomic_distance < 1.0f)
			atomic_distance = 1.0f;

		//calculating energy contributions
		if ((atomic_distance < 8.0f) && (atomic_distance < 20.48f))
		{
			// Getting type IDs
			uint atom1_typeid = atom_types_const[atom1_id];
			uint atom2_typeid = atom_types_const[atom2_id];

			//calculating van der Waals / hydrogen bond term
#if defined (NATIVE_PRECISION)
			partial_energies[get_local_id(0)] += native_divide(VWpars_AC_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(atomic_distance,12));
#elif defined (HALF_PRECISION)
			pfor (uint rotation_counter = get_local_id(0);
	          rotation_counter < dockpars_rotbondlist_length;
	          rotation_counter+=NUM_OF_THREADS_PER_BLOCK)
	{artial_energies[get_local_id(0)] += half_divide(VWpars_AC_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],half_powr(atomic_distance,12));
#else	// Full precision
			partial_energies[get_local_id(0)] += VWpars_AC_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid]/powr(atomic_distance,12);
#endif

			if (intraE_contributors_const[3*contributor_counter+2] == 1)	//H-bond
#if defined (NATIVE_PRECISION)
				partial_energies[get_local_id(0)] -= native_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(atomic_distance,10));
#elif defined (HALF_PRECISION)
				partial_energies[get_local_id(0)] -= half_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],half_powr(atomic_distance,10));
#else	// Full precision
				partial_energies[get_local_id(0)] -= VWpars_BD_const[atom1_typeid*dockpars_num_of_atypes+atom2_typeid]/powr(atomic_distance,10);
#endif

			else	//van der Waals
#if defined (NATIVE_PRECISION)
				partial_energies[get_local_id(0)] -= native_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(atomic_distance,6));
#elif defined (HALF_PRECISION)
				partial_energies[get_local_id(0)] -= half_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],half_powr(atomic_distance,6));
#else	// Full precision
				partial_energies[get_local_id(0)] -= VWpars_BD_const[atom1_typeid*dockpars_num_of_atypes+atom2_typeid]/powr(atomic_distance,6);
#endif

			//calculating electrostatic term
#if defined (NATIVE_PRECISION)
        partial_energies[get_local_id(0)] += native_divide (
                                                             dockpars_coeff_elec * atom_charges_const[atom1_id] * atom_charges_const[atom2_id],
                                                             atomic_distance * (-8.5525f + native_divide(86.9525f,(1.0f + 7.7839f*native_exp(-0.3154f*atomic_distance))))
                                                             );
#elif defined (HALF_PRECISION)
        partial_energies[get_local_id(0)] += half_divide (
                                                             dockpars_coeff_elec * atom_charges_const[atom1_id] * atom_charges_const[atom2_id],
                                                             atomic_distance * (-8.5525f + half_divide(86.9525f,(1.0f + 7.7839f*half_exp(-0.3154f*atomic_distance))))
                                                             );
#else	// Full precision
				partial_energies[get_local_id(0)] += dockpars_coeff_elec*atom_charges_const[atom1_id]*atom_charges_const[atom2_id]/
			                                       (atomic_distance*(-8.5525f + 86.9525f/(1.0f + 7.7839f*exp(-0.3154f*atomic_distance))));
#endif

			//calculating desolvation term
#if defined (NATIVE_PRECISION)
			partial_energies[get_local_id(0)] += ((dspars_S_const[atom1_typeid] +
							       dockpars_qasp*fabs(atom_charges_const[atom1_id]))*dspars_V_const[atom2_typeid] +
					                       (dspars_S_const[atom2_typeid] +
							       dockpars_qasp*fabs(atom_charges_const[atom2_id]))*dspars_V_const[atom1_typeid]) *
					                       dockpars_coeff_desolv*native_exp(-atomic_distance*native_divide(atomic_distance,25.92f));
#elif defined (HALF_PRECISION)
			partial_energies[get_local_id(0)] += ((dspars_S_const[atom1_typeid] +
							       dockpars_qasp*fabs(atom_charges_const[atom1_id]))*dspars_V_const[atom2_typeid] +
					                       (dspars_S_const[atom2_typeid] +
							       dockpars_qasp*fabs(atom_charges_const[atom2_id]))*dspars_V_const[atom1_typeid]) *
					                       dockpars_coeff_desolv*half_exp(-atomic_distance*half_divide(atomic_distance,25.92f));
#else	// Full precision
			partial_energies[get_local_id(0)] += ((dspars_S_const[atom1_typeid] +
							       dockpars_qasp*fabs(atom_charges_const[atom1_id]))*dspars_V_const[atom2_typeid] +
					                       (dspars_S_const[atom2_typeid] +
							       dockpars_qasp*fabs(atom_charges_const[atom2_id]))*dspars_V_const[atom1_typeid]) *
					                       dockpars_coeff_desolv*exp(-atomic_distance*atomic_distance/25.92f);
#endif

		}
	} // End contributor_counter for-loop (INTRAMOLECULAR ENERGY)









	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_local_id(0) == 0)
	{
		*energy = partial_energies[0];

		for (uint contributor_counter=1;
		          contributor_counter<NUM_OF_THREADS_PER_BLOCK;
		          contributor_counter++)
		{
			*energy += partial_energies[contributor_counter];
		}
	}

	// -------------------------------------------------------------------
	// Calculate gradients (forces) corresponding to (interE + intraE)
	// Derived from autodockdev/motions.py/forces_to_delta()
	// -------------------------------------------------------------------
	
	// Could be barrier removed if another work-item is used? 
	// (e.g. get_locla_id(0) == 1)
	barrier(CLK_LOCAL_MEM_FENCE);

	// FIXME: done so far only for interE
	if (get_local_id(0) == 0) {
		if (*is_enabled_gradient_calc) {
			gradient_genotype [0] = 0.0f;
			gradient_genotype [1] = 0.0f;
			gradient_genotype [2] = 0.0f;
		
			// ------------------------------------------
			// translation-related gradients
			// ------------------------------------------
			for (uint lig_atom_id = 0;
				  lig_atom_id<dockpars_num_of_atoms;
				  lig_atom_id++) {
				gradient_genotype [0] += gradient_inter_x[lig_atom_id]; // gradient for gene 0: gene x
				gradient_genotype [1] += gradient_inter_y[lig_atom_id]; // gradient for gene 1: gene y
				gradient_genotype [2] += gradient_inter_z[lig_atom_id]; // gradient for gene 2: gene z
			}

			// ------------------------------------------
			// rotation-related gradients 
			// ------------------------------------------
			float3 torque_rot = (float3)(0.0f, 0.0f, 0.0f);

			// center of rotation 
			// In getparameters.cpp, it indicates 
			// translation genes are in grid spacing (instead of Angstroms)
			float about[3];
			about[0] = genotype[0]; 
			about[1] = genotype[1];
			about[2] = genotype[2];
		
			// Temporal variable to calculate translation differences.
			// They are converted back to Angstroms here
			float3 r;
			
			for (uint lig_atom_id = 0;
				  lig_atom_id<dockpars_num_of_atoms;
				  lig_atom_id++) {
				r.x = (calc_coords_x[lig_atom_id] - about[0]) * dockpars_grid_spacing; 
				r.y = (calc_coords_y[lig_atom_id] - about[1]) * dockpars_grid_spacing;  
				r.z = (calc_coords_z[lig_atom_id] - about[2]) * dockpars_grid_spacing; 
				torque_rot += cross(r, torque_rot);
			}

			const float rad = 1E-8;
			const float rad_div_2 = native_divide(rad, 2);

			
			float quat_w, quat_x, quat_y, quat_z;

			// Derived from rotation.py/axisangle_to_q()
			// genes[3:7] = rotation.axisangle_to_q(torque, rad)
			torque_rot = fast_normalize(torque_rot);
			quat_x = torque_rot.x;
			quat_y = torque_rot.y;
			quat_z = torque_rot.z;

			// rotation-related gradients are expressed here in quaternions
			quat_w = native_cos(rad_div_2);
			quat_x = quat_x * native_sin(rad_div_2);
			quat_y = quat_y * native_sin(rad_div_2);
			quat_z = quat_z * native_sin(rad_div_2);

			// convert quaternion gradients into Shoemake gradients 
			// Derived from autodockdev/motion.py/_get_cube3_gradient

			// where we are in cube3
			float current_u1, current_u2, current_u3;
			current_u1 = genotype[3]; // check very initial input Shoemake genes
			current_u2 = genotype[4];
			current_u3 = genotype[5];

			// where we are in quaternion space
			// current_q = cube3_to_quaternion(current_u)
			float current_qw, current_qx, current_qy, current_qz;
			current_qw = native_sqrt(1-current_u1) * native_sin(PI_TIMES_2*current_u2);
			current_qx = native_sqrt(1-current_u1) * native_cos(PI_TIMES_2*current_u2);
			current_qy = native_sqrt(current_u1)   * native_sin(PI_TIMES_2*current_u3);
			current_qz = native_sqrt(current_u1)   * native_cos(PI_TIMES_2*current_u3);

			// where we want to be in quaternion space
			float target_qw, target_qx, target_qy, target_qz;

			// target_q = rotation.q_mult(q, current_q)
			// Derived from autodockdev/rotation.py/q_mult()
			// In our terms means q_mult(quat_{w|x|y|z}, current_q{w|x|y|z})
			target_qw = quat_w*current_qw - quat_x*current_qx - quat_y*current_qy - quat_z*current_qz;// w
			target_qx = quat_w*current_qx + quat_x*current_qw + quat_y*current_qz - quat_z*current_qy;// x
			target_qy = quat_w*current_qy + quat_y*current_qw + quat_z*current_qx - quat_x*current_qz;// y
			target_qz = quat_w*current_qz + quat_z*current_qw + quat_x*current_qy - quat_y*current_qx;// z

			// where we want ot be in cube3
			float target_u1, target_u2, target_u3;

			// target_u = quaternion_to_cube3(target_q)
			// Derived from autodockdev/motions.py/quaternion_to_cube3()
			// In our terms means quaternion_to_cube3(target_q{w|x|y|z})
			target_u1 = target_qy*target_qy + target_qz*target_qz;
			target_u2 = atan2pi(target_qw, target_qx)/PI_TIMES_2;
			target_u3 = atan2pi(target_qy, target_qz)/PI_TIMES_2;

			// derivates in cube3
			float grad_u1, grad_u2, grad_u3;
			grad_u1 = target_u1 - current_u1;
			grad_u2 = target_u2 - current_u2;
			grad_u3 = target_u3 - current_u3;
			
			// empirical scaling
			float temp_u1 = genotype[3];
			
			if ((temp_u1 > 1.0f) || (temp_u1 < 0.0f)){
				grad_u1 *= ((1/temp_u1) + (1/(1-temp_u1)));
			}
			grad_u2 *= 4 * (1-temp_u1);
			grad_u3 *= 4 * temp_u1;
			
			// set gradient rotation-ralated genotypes in cube3
			gradient_genotype[3] = grad_u1;
			gradient_genotype[4] = grad_u2;
			gradient_genotype[5] = grad_u3;
			
			// ------------------------------------------
			// torsion-related gradients 
			// ------------------------------------------
			
			// Iterate over rotations,
			// but identify those associated with rotatable bonds
			// Then iterate over rotatable bonds
			for (uint rotation_counter = 0;
			          rotation_counter < dockpars_rotbondlist_length;
			          rotation_counter ++)
			{
				int rotation_list_element = rotlist_const[rotation_counter];

				if ((rotation_list_element & RLIST_DUMMY_MASK) == 0)	// If not dummy rotation
				{
					uint atom_id = rotation_list_element & RLIST_ATOMID_MASK;

					if ((rotation_list_element & RLIST_GENROT_MASK) != 0)	// If general rotation
					{

					}
					else	// If rotating around rotatable bond
					{
						uint rotbond_id = (rotation_list_element & RLIST_RBONDID_MASK) >> RLIST_RBONDID_SHIFT;

						float3 rotation_unitvec;
						rotation_unitvec.x = rotbonds_unit_vectors_const[3*rotbond_id];
						rotation_unitvec.y = rotbonds_unit_vectors_const[3*rotbond_id+1];
						rotation_unitvec.z = rotbonds_unit_vectors_const[3*rotbond_id+2];

						// Torque of torsions
						float3 torque_tor = (float3)(0.0f, 0.0f, 0.0f);

						// Iterate over each ligand atom
						for (unsigned int lig_atom_id = 0;
								  lig_atom_id<dockpars_num_of_atoms;
								  lig_atom_id++) {
							// Calculate torque on point "A" 
							// (could be any other point "B" along the rotation axis)
							float3 atom_coords = {calc_coords_x[lig_atom_id], 
								              calc_coords_y[lig_atom_id], 
								              calc_coords_z[lig_atom_id]};

							float3 atom_force  = {gradient_inter_x[lig_atom_id],
								              gradient_inter_y[lig_atom_id],
							                      gradient_inter_z[lig_atom_id]};

							float3 rotation_movingvec;
							rotation_movingvec.x = rotbonds_moving_vectors_const[3*rotbond_id];
							rotation_movingvec.y = rotbonds_moving_vectors_const[3*rotbond_id+1];
							rotation_movingvec.z = rotbonds_moving_vectors_const[3*rotbond_id+2];

							torque_tor += cross((atom_coords-rotation_movingvec), atom_force);
						}

						// Project torque on rotation axis
						float torque_on_axis = dot(rotation_unitvec, torque_tor);

						// Assignment of gene-based gradient
						gradient_genotype[rotbond_id] = torque_on_axis;

					} // End of general / rotatable-bond rotation

				} // End of not dummy rotation

			} // End of iterations over rotations

		} // End if (*is_enabled_gradient_calc)

	} // End gradient-calculation performed by single work-item

}

#include "kernel1.cl"
#include "kernel2.cl"
#include "auxiliary_genetic.cl"
#include "kernel4.cl"
#include "kernel3.cl"
