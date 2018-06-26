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
			__global const float* dockpars_fgrids, // cannot be allocated in __constant (too large)
		#endif
		            char   dockpars_num_of_atypes,
		            int    dockpars_num_of_intraE_contributors,
			    float  dockpars_grid_spacing,
			    float  dockpars_coeff_elec,
			    float  dockpars_qasp,
			    float  dockpars_coeff_desolv,

		    __local float* genotype,
		    __local float* energy,
		    __local int*   run_id,

                    // Some OpenCL compilers don't allow local var outside kernels
		    // so this local vars are passed from a kernel
		    __local float* calc_coords_x,
		    __local float* calc_coords_y,
		    __local float* calc_coords_z,
		    __local float* partial_energies,

	       __constant float* atom_charges_const,
               __constant char*  atom_types_const,
               __constant char*  intraE_contributors_const,
	                  float  dockpars_smooth,
	       __constant float* reqm,
	       __constant float* reqm_hbond,
      	       __constant uint*  atom1_types_reqm,
	       __constant uint*  atom2_types_reqm,
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
               __constant float* ref_orientation_quats_const
)

//The GPU device function calculates the energy of the entity described by genotype, dockpars and the liganddata
//arrays in constant memory and returns it in the energy parameter. The parameter run_id has to be equal to the ID
//of the run whose population includes the current entity (which can be determined with blockIdx.x), since this
//determines which reference orientation should be used.
{
	int contributor_counter;
	/*char*/uint atom1_id, atom2_id, atom1_typeid, atom2_typeid;

	// Name changed to distance_leo to avoid
	// errors as "distance" is the name of OpenCL function
	//float subx, suby, subz, distance;
	float subx, suby, subz, distance_leo;

	float x, y, z, dx, dy, dz, q;
	float cube[2][2][2];
	float weights[2][2][2];
	int x_low, x_high, y_low, y_high, z_low, z_high;

	float phi, theta, genrotangle, rotation_angle, sin_angle;
	float genrot_unitvec[3], rotation_unitvec[3], rotation_movingvec[3];
	int rotation_counter, rotation_list_element;
	float atom_to_rotate[3];
	int atom_id, rotbond_id;
	float quatrot_left_x, quatrot_left_y, quatrot_left_z, quatrot_left_q;
	float quatrot_temp_x, quatrot_temp_y, quatrot_temp_z, quatrot_temp_q;

        // Some OpenCL compilers don't allow local var outside kernels
	// so this local vars are passed from a kernel
	//__local float calc_coords_x[MAX_NUM_OF_ATOMS];
	//__local float calc_coords_y[MAX_NUM_OF_ATOMS];
	//__local float calc_coords_z[MAX_NUM_OF_ATOMS];
	//__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];

	partial_energies[get_local_id(0)] = 0.0f;

	//CALCULATE CONFORMATION

	//calculate vectors for general rotation
	phi         = genotype[3]*DEG_TO_RAD;
	theta       = genotype[4]*DEG_TO_RAD;
	genrotangle = genotype[5]*DEG_TO_RAD;

#if defined (IMPROVE_GRID)

	#if defined (NATIVE_PRECISION)
	sin_angle = native_sin(theta);
	genrot_unitvec [0] = sin_angle*native_cos(phi);
	genrot_unitvec [1] = sin_angle*native_sin(phi);
	genrot_unitvec [2] = native_cos(theta);
	#elif defined (HALF_PRECISION)
	sin_angle = half_sin(theta);
	genrot_unitvec [0] = sin_angle*half_cos(phi);
	genrot_unitvec [1] = sin_angle*half_sin(phi);
	genrot_unitvec [2] = half_cos(theta);
	#else	// Full precision
	sin_angle = sin(theta);
	genrot_unitvec [0] = sin_angle*cos(phi);
	genrot_unitvec [1] = sin_angle*sin(phi);
	genrot_unitvec [2] = cos(theta);
	#endif

	// INTERMOLECULAR for-loop (intermediate results)
	// It stores a product of two chars
	unsigned int mul_tmp;

	unsigned char g1 = dockpars_gridsize_x;
	unsigned int  g2 = dockpars_gridsize_x * dockpars_gridsize_y;
  	unsigned int  g3 = dockpars_gridsize_x * dockpars_gridsize_y * dockpars_gridsize_z;

	unsigned int ylow_times_g1, yhigh_times_g1;
	unsigned int zlow_times_g2, zhigh_times_g2;

	unsigned int cube_000;
	unsigned int cube_100;
	unsigned int cube_010;
	unsigned int cube_110;
	unsigned int cube_001;
  	unsigned int cube_101;
  	unsigned int cube_011;
  	unsigned int cube_111;

#else
	sin_angle = sin(theta);
	genrot_unitvec [0] = sin_angle*cos(phi);
	genrot_unitvec [1] = sin_angle*sin(phi);
	genrot_unitvec [2] = cos(theta);
#endif

	// ================================================
	// Iterating over elements of rotation list
	// ================================================
	for (rotation_counter = get_local_id(0);
	     rotation_counter < dockpars_rotbondlist_length;
	     rotation_counter+=NUM_OF_THREADS_PER_BLOCK)
	{
		rotation_list_element = rotlist_const[rotation_counter];

		if ((rotation_list_element & RLIST_DUMMY_MASK) == 0)	//if not dummy rotation
		{
			atom_id = rotation_list_element & RLIST_ATOMID_MASK;

			//capturing atom coordinates
			if ((rotation_list_element & RLIST_FIRSTROT_MASK) != 0)	//if firts rotation of this atom
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

			//capturing rotation vectors and angle
			if ((rotation_list_element & RLIST_GENROT_MASK) != 0)	//if general rotation
			{
				rotation_unitvec[0] = genrot_unitvec[0];
				rotation_unitvec[1] = genrot_unitvec[1];
				rotation_unitvec[2] = genrot_unitvec[2];

				rotation_angle = genrotangle;

				rotation_movingvec[0] = genotype[0];
				rotation_movingvec[1] = genotype[1];
				rotation_movingvec[2] = genotype[2];
			}
			else	//if rotating around rotatable bond
			{
				rotbond_id = (rotation_list_element & RLIST_RBONDID_MASK) >> RLIST_RBONDID_SHIFT;

				rotation_unitvec[0] = rotbonds_unit_vectors_const[3*rotbond_id];
				rotation_unitvec[1] = rotbonds_unit_vectors_const[3*rotbond_id+1];
				rotation_unitvec[2] = rotbonds_unit_vectors_const[3*rotbond_id+2];
				rotation_angle = genotype[6+rotbond_id]*DEG_TO_RAD;

				rotation_movingvec[0] = rotbonds_moving_vectors_const[3*rotbond_id];
				rotation_movingvec[1] = rotbonds_moving_vectors_const[3*rotbond_id+1];
				rotation_movingvec[2] = rotbonds_moving_vectors_const[3*rotbond_id+2];

				//in addition, performing the first movement which is needed only if rotating around rotatable bond
				atom_to_rotate[0] -= rotation_movingvec[0];
				atom_to_rotate[1] -= rotation_movingvec[1];
				atom_to_rotate[2] -= rotation_movingvec[2];
			}

			//performing rotation

			rotation_angle = rotation_angle * 0.5f;

#if defined (NATIVE_PRECISION)
			/*rotation_angle = native_divide(rotation_angle,2);*/
			quatrot_left_q = native_cos(rotation_angle);
			sin_angle = native_sin(rotation_angle);
#elif defined (HALF_PRECISION)
			/*rotation_angle = half_divide(rotation_angle,2);*/
			quatrot_left_q = half_cos(rotation_angle);
			sin_angle = half_sin(rotation_angle);
#else	// Full precision
			/*rotation_angle = rotation_angle/2;*/
			quatrot_left_q = cos(rotation_angle);
			sin_angle = sin(rotation_angle);
#endif
			quatrot_left_x = sin_angle*rotation_unitvec[0];
			quatrot_left_y = sin_angle*rotation_unitvec[1];
			quatrot_left_z = sin_angle*rotation_unitvec[2];

			if ((rotation_list_element & RLIST_GENROT_MASK) != 0)	// if general rotation,
																														// two rotations should be performed
																														// (multiplying the quaternions)
			{
				//calculating quatrot_left*ref_orientation_quats_const,
				//which means that reference orientation rotation is the first
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

			//performing final movement and storing values
			calc_coords_x[atom_id] = atom_to_rotate [0] + rotation_movingvec[0];
			calc_coords_y[atom_id] = atom_to_rotate [1] + rotation_movingvec[1];
			calc_coords_z[atom_id] = atom_to_rotate [2] + rotation_movingvec[2];

		} // End if-statement not dummy rotation

		barrier(CLK_LOCAL_MEM_FENCE);

	} // End rotation_counter for-loop

	// ================================================
	// CALCULATE INTERMOLECULAR ENERGY
	// ================================================
	for (atom1_id = get_local_id(0);
	     atom1_id < dockpars_num_of_atoms;
	     atom1_id+= NUM_OF_THREADS_PER_BLOCK)
	{
		atom1_typeid = atom_types_const[atom1_id];
		x = calc_coords_x[atom1_id];
		y = calc_coords_y[atom1_id];
		z = calc_coords_z[atom1_id];
		q = atom_charges_const[atom1_id];

		if ((x < 0) || (y < 0) || (z < 0) || (x >= dockpars_gridsize_x-1)
				                  || (y >= dockpars_gridsize_y-1)
						  || (z >= dockpars_gridsize_z-1)){
			partial_energies[get_local_id(0)] += 16777216.0f; //100000.0f;
		}
		else
		{
			//get coordinates
			x_low = (int)floor(x); y_low = (int)floor(y); z_low = (int)floor(z);
			x_high = (int)ceil(x); y_high = (int)ceil(y); z_high = (int)ceil(z);
			dx = x - x_low; dy = y - y_low; dz = z - z_low;

			//calculate interpolation weights
			weights [0][0][0] = (1-dx)*(1-dy)*(1-dz);
			weights [1][0][0] = dx*(1-dy)*(1-dz);
			weights [0][1][0] = (1-dx)*dy*(1-dz);
			weights [1][1][0] = dx*dy*(1-dz);
			weights [0][0][1] = (1-dx)*(1-dy)*dz;
			weights [1][0][1] = dx*(1-dy)*dz;
			weights [0][1][1] = (1-dx)*dy*dz;
			weights [1][1][1] = dx*dy*dz;

			//capturing affinity values
#if defined (IMPROVE_GRID)
			ylow_times_g1  = y_low*g1;
			yhigh_times_g1 = y_high*g1;
		  	zlow_times_g2  = z_low*g2;
			zhigh_times_g2 = z_high*g2;

			cube_000 = x_low  + ylow_times_g1  + zlow_times_g2;
			cube_100 = x_high + ylow_times_g1  + zlow_times_g2;
			cube_010 = x_low  + yhigh_times_g1 + zlow_times_g2;
			cube_110 = x_high + yhigh_times_g1 + zlow_times_g2;
			cube_001 = x_low  + ylow_times_g1  + zhigh_times_g2;
			cube_101 = x_high + ylow_times_g1  + zhigh_times_g2;
			cube_011 = x_low  + yhigh_times_g1 + zhigh_times_g2;
			cube_111 = x_high + yhigh_times_g1 + zhigh_times_g2;
			mul_tmp = atom1_typeid*g3;

			cube [0][0][0] = *(dockpars_fgrids + cube_000 + mul_tmp);
			cube [1][0][0] = *(dockpars_fgrids + cube_100 + mul_tmp);
      			cube [0][1][0] = *(dockpars_fgrids + cube_010 + mul_tmp);
      			cube [1][1][0] = *(dockpars_fgrids + cube_110 + mul_tmp);
      			cube [0][0][1] = *(dockpars_fgrids + cube_001 + mul_tmp);
      			cube [1][0][1] = *(dockpars_fgrids + cube_101 + mul_tmp);
      			cube [0][1][1] = *(dockpars_fgrids + cube_011 + mul_tmp);
      			cube [1][1][1] = *(dockpars_fgrids + cube_111 + mul_tmp);

#else
			cube [0][0][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_low, x_low);
			cube [1][0][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_low, x_high);
			cube [0][1][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      								  dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_high, x_low);
			cube [1][1][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_high, x_high);
			cube [0][0][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_high, y_low, x_low);
			cube [1][0][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      								  atom1_typeid, z_high, y_low, x_high);
			cube [0][1][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_high, y_high, x_low);
			cube [1][1][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_high, y_high, x_high);
#endif

			//calculating affinity energy
			partial_energies[get_local_id(0)] += TRILININTERPOL(cube, weights);

			//capturing electrostatic values
			atom1_typeid = dockpars_num_of_atypes;

#if defined (IMPROVE_GRID)
			mul_tmp = atom1_typeid*g3;
			cube [0][0][0] = *(dockpars_fgrids + cube_000 + mul_tmp);
			cube [1][0][0] = *(dockpars_fgrids + cube_100 + mul_tmp);
      			cube [0][1][0] = *(dockpars_fgrids + cube_010 + mul_tmp);
      			cube [1][1][0] = *(dockpars_fgrids + cube_110 + mul_tmp);
      			cube [0][0][1] = *(dockpars_fgrids + cube_001 + mul_tmp);
      			cube [1][0][1] = *(dockpars_fgrids + cube_101 + mul_tmp);
      			cube [0][1][1] = *(dockpars_fgrids + cube_011 + mul_tmp);
      			cube [1][1][1] = *(dockpars_fgrids + cube_111 + mul_tmp);

#else
			cube [0][0][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      		 						  dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_low, x_low);
			cube [1][0][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_low, x_high);
			cube [0][1][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_high, x_low);
			cube [1][1][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_high, x_high);
			cube [0][0][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_high, y_low, x_low);
			cube [1][0][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_high, y_low, x_high);
			cube [0][1][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_high, y_high, x_low);
			cube [1][1][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_high, y_high, x_high);
#endif

			//calculating electrosatic energy
			partial_energies[get_local_id(0)] += q * TRILININTERPOL(cube, weights);

			//capturing desolvation values
			atom1_typeid = dockpars_num_of_atypes+1;

#if defined (IMPROVE_GRID)
			mul_tmp = atom1_typeid*g3;
			cube [0][0][0] = *(dockpars_fgrids + cube_000 + mul_tmp);
			cube [1][0][0] = *(dockpars_fgrids + cube_100 + mul_tmp);
      cube [0][1][0] = *(dockpars_fgrids + cube_010 + mul_tmp);
      cube [1][1][0] = *(dockpars_fgrids + cube_110 + mul_tmp);
      cube [0][0][1] = *(dockpars_fgrids + cube_001 + mul_tmp);
      cube [1][0][1] = *(dockpars_fgrids + cube_101 + mul_tmp);
      cube [0][1][1] = *(dockpars_fgrids + cube_011 + mul_tmp);
      cube [1][1][1] = *(dockpars_fgrids + cube_111 + mul_tmp);

#else
			cube [0][0][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_low, x_low);
			cube [1][0][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_low, x_high);
			cube [0][1][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_high, x_low);
			cube [1][1][0] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_low, y_high, x_high);
			cube [0][0][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_high, y_low, x_low);
			cube [1][0][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_high, y_low, x_high);
			cube [0][1][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_high, y_high, x_low);
			cube [1][1][1] = GETGRIDVALUE(dockpars_fgrids, dockpars_gridsize_x,
						      									dockpars_gridsize_y, dockpars_gridsize_z,
						      									atom1_typeid, z_high, y_high, x_high);
#endif

			//calculating desolvation energy
			partial_energies[get_local_id(0)] += fabs(q) * TRILININTERPOL(cube, weights);
		}

	} // End atom1_id for-loop

	// In paper: intermolecular and internal energy calculation
	// are independent from each other, -> NO BARRIER NEEDED
  // but require different operations,
	// thus, they can be executed only sequentially on the GPU.

	// ================================================
	// CALCULATE INTRAMOLECULAR ENERGY
	// ================================================
	for (contributor_counter = get_local_id(0);
	     contributor_counter < dockpars_num_of_intraE_contributors;
	     contributor_counter +=NUM_OF_THREADS_PER_BLOCK)
	{
		//getting atom IDs
		atom1_id = intraE_contributors_const[3*contributor_counter];
		atom2_id = intraE_contributors_const[3*contributor_counter+1];

		//calculating address of first atom's coordinates
		subx = calc_coords_x[atom1_id];
		suby = calc_coords_y[atom1_id];
		subz = calc_coords_z[atom1_id];

		//calculating address of second atom's coordinates
		subx -= calc_coords_x[atom2_id];
		suby -= calc_coords_y[atom2_id];
		subz -= calc_coords_z[atom2_id];

		//calculating distance (distance_leo)
#if defined (NATIVE_PRECISION)
		distance_leo = native_sqrt(subx*subx + suby*suby + subz*subz)*dockpars_grid_spacing;
#elif defined (HALF_PRECISION)
		distance_leo = half_sqrt(subx*subx + suby*suby + subz*subz)*dockpars_grid_spacing;
#else	// Full precision
		distance_leo = sqrt(subx*subx + suby*suby + subz*subz)*dockpars_grid_spacing;
#endif

		//getting type IDs
		atom1_typeid = atom_types_const[atom1_id];
		atom2_typeid = atom_types_const[atom2_id];

		uint atom1_type_vdw_hb = atom1_types_reqm [atom1_typeid];
     	        uint atom2_type_vdw_hb = atom2_types_reqm [atom2_typeid];

		//getting optimum pair distance (opt_distance) from reqm and reqm_hbond
		//reqm: equilibrium internuclear separation 
		//      (sum of the vdW radii of two like atoms (A)) in the case of vdW
		//reqm_hbond: equilibrium internuclear separation
		//	(sum of the vdW radii of two like atoms (A)) in the case of hbond 
		float opt_distance;

		if (intraE_contributors_const[3*contributor_counter+2] == 1)	//H-bond
		{
			opt_distance = reqm_hbond [atom1_type_vdw_hb] + reqm_hbond [atom2_type_vdw_hb];
		}
		else	//van der Waals
		{
			opt_distance = 0.5f*(reqm [atom1_type_vdw_hb] + reqm [atom2_type_vdw_hb]);
		}

		//getting smoothed distance
		//smoothed_distance = function(distance_leo, opt_distance)
		float smoothed_distance;
		float delta_distance = 0.5f*dockpars_smooth;

		if (distance_leo <= (opt_distance - delta_distance)) {
			smoothed_distance = distance_leo + delta_distance;
		}
		else if (distance_leo < (opt_distance + delta_distance)) {
			smoothed_distance = opt_distance;
		}
		else { // else if (distance_leo >= (opt_distance + delta_distance))
			smoothed_distance = distance_leo - delta_distance;
		}

		//calculating energy contributions
		//cuttoff: internuclear-distance at 8A
		//cutoff only for vdw and hbond
		//el and sol contributions are calculated at all distances
		if (distance_leo < 8.0f)
		{
			//calculating van der Waals / hydrogen bond term
#if defined (NATIVE_PRECISION)
			partial_energies[get_local_id(0)] += native_divide(VWpars_AC_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(smoothed_distance,12));
#elif defined (HALF_PRECISION)
			partial_energies[get_local_id(0)] += half_divide(VWpars_AC_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],half_powr(smoothed_distance,12));
#else	// Full precision
			partial_energies[get_local_id(0)] += VWpars_AC_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid]/powr(smoothed_distance,12);
#endif

			if (intraE_contributors_const[3*contributor_counter+2] == 1)	//H-bond
#if defined (NATIVE_PRECISION)
				partial_energies[get_local_id(0)] -= native_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(smoothed_distance,10));
#elif defined (HALF_PRECISION)
				partial_energies[get_local_id(0)] -= half_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],half_powr(smoothed_distance,10));
#else	// Full precision
				partial_energies[get_local_id(0)] -= VWpars_BD_const[atom1_typeid*dockpars_num_of_atypes+atom2_typeid]/powr(smoothed_distance,10);
#endif

			else	//van der Waals
#if defined (NATIVE_PRECISION)
				partial_energies[get_local_id(0)] -= native_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(smoothed_distance,6));
#elif defined (HALF_PRECISION)
				partial_energies[get_local_id(0)] -= half_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],half_powr(smoothed_distance,6));
#else	// Full precision
				partial_energies[get_local_id(0)] -= VWpars_BD_const[atom1_typeid*dockpars_num_of_atypes+atom2_typeid]/powr(smoothed_distance,6);
#endif

		} // if cuttoff - internuclear-distance at 8A	

		if (distance_leo < 20.48f)
		{
			//calculating electrostatic term
	#if defined (NATIVE_PRECISION)
			partial_energies[get_local_id(0)] += native_divide (
		                                                     dockpars_coeff_elec * atom_charges_const[atom1_id] * atom_charges_const[atom2_id],
		                                                     distance_leo * (-8.5525f + native_divide(86.9525f,(1.0f + 7.7839f*native_exp(-0.3154f*distance_leo))))
		                                                     );
	#elif defined (HALF_PRECISION)
			partial_energies[get_local_id(0)] += half_divide (
		                                                     dockpars_coeff_elec * atom_charges_const[atom1_id] * atom_charges_const[atom2_id],
		                                                     distance_leo * (-8.5525f + half_divide(86.9525f,(1.0f + 7.7839f*half_exp(-0.3154f*distance_leo))))
		                                                     );
	#else	// Full precision
			partial_energies[get_local_id(0)] += dockpars_coeff_elec*atom_charges_const[atom1_id]*atom_charges_const[atom2_id]/
					                             (distance_leo*(-8.5525f + 86.9525f/(1.0f + 7.7839f*exp(-0.3154f*distance_leo))));
	#endif

			//calculating desolvation term
	#if defined (NATIVE_PRECISION)
			partial_energies[get_local_id(0)] += ((dspars_S_const[atom1_typeid] +
								       											 dockpars_qasp*fabs(atom_charges_const[atom1_id]))*dspars_V_const[atom2_typeid] +
							              					 (dspars_S_const[atom2_typeid] +
								       								 			 dockpars_qasp*fabs(atom_charges_const[atom2_id]))*dspars_V_const[atom1_typeid]) *
							               					 dockpars_coeff_desolv*native_exp(-distance_leo*native_divide(distance_leo,25.92f));
	#elif defined (HALF_PRECISION)
			partial_energies[get_local_id(0)] += ((dspars_S_const[atom1_typeid] +
								       											 dockpars_qasp*fabs(atom_charges_const[atom1_id]))*dspars_V_const[atom2_typeid] +
							              					 (dspars_S_const[atom2_typeid] +
								       								 			 dockpars_qasp*fabs(atom_charges_const[atom2_id]))*dspars_V_const[atom1_typeid]) *
							               					 dockpars_coeff_desolv*half_exp(-distance_leo*half_divide(distance_leo,25.92f));
	#else	// Full precision
			partial_energies[get_local_id(0)] += ((dspars_S_const[atom1_typeid] +
					   				       									     	 dockpars_qasp*fabs(atom_charges_const[atom1_id]))*dspars_V_const[atom2_typeid] +
							              				   	 (dspars_S_const[atom2_typeid] +
								       								 			 dockpars_qasp*fabs(atom_charges_const[atom2_id]))*dspars_V_const[atom1_typeid]) *
							               					 dockpars_coeff_desolv*exp(-distance_leo*distance_leo/25.92f);
	#endif

		} // if cuttoff - internuclear-distance at 20.48A
	
	} // End contributor_counter for-loop

	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_local_id(0) == 0)
	{
		*energy = partial_energies[0];

		for (contributor_counter=1;
		     contributor_counter<NUM_OF_THREADS_PER_BLOCK;
		     contributor_counter++)
		{
			*energy += partial_energies[contributor_counter];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
}

#include "kernel1.cl"
#include "kernel2.cl"
#include "auxiliary_genetic.cl"
#include "kernel4.cl"
#include "kernel3.cl"
