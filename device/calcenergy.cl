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

//#define DEBUG_ENERGY_KERNEL1
//#define DEBUG_ENERGY_KERNEL4
//#define DEBUG_ENERGY_KERNEL3

#include "calcenergy_basic.h"

// All related pragmas are in defines.h (accesible by host and device code)

void gpu_calc_energy(	    
				int    dockpars_rotbondlist_length,
				char   dockpars_num_of_atoms,
			    	char   dockpars_gridsize_x,
			    	char   dockpars_gridsize_y,
			    	char   dockpars_gridsize_z,
								    		// g1 = gridsize_x
				uint   dockpars_gridsize_x_times_y, 		// g2 = gridsize_x * gridsize_y
				uint   dockpars_gridsize_x_times_y_times_z,	// g3 = gridsize_x * gridsize_y * gridsize_z
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
		    	__local float* partial_energies,

			#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
			__local float* partial_interE,
			__local float* partial_intraE,
			#endif

	             __constant float* atom_charges_const,
                     __constant char*  atom_types_const,
                     __constant char*  intraE_contributors_const,
	                  	float  dockpars_smooth,
	       	     __constant float* reqm,
	       	     __constant float* reqm_hbond,
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
	partial_energies[get_local_id(0)] = 0.0f;

	#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
	partial_interE[get_local_id(0)] = 0.0f;
	partial_intraE[get_local_id(0)] = 0.0f;
	#endif

	uchar g1 = dockpars_gridsize_x;
	uint  g2 = dockpars_gridsize_x_times_y;
  	uint  g3 = dockpars_gridsize_x_times_y_times_z;

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
				// Rotational genes in the Shoemake space expressed in radians
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
				rotation_angle  = rotation_angle * 0.5f;
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
	// CALCULATING INTERMOLECULAR ENERGY
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
	
			#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
			partial_interE[get_local_id(0)] += 16777216.0f;
			#endif
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

			// Calculating affinity energy
			partial_energies[get_local_id(0)] += TRILININTERPOL(cube, weights);

			#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
			partial_interE[get_local_id(0)] += TRILININTERPOL(cube, weights);
			#endif

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

			// Calculating electrostatic energy
			partial_energies[get_local_id(0)] += q * TRILININTERPOL(cube, weights);

			#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
			partial_interE[get_local_id(0)] += q * TRILININTERPOL(cube, weights);
			#endif

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

			// Calculating desolvation energy
			partial_energies[get_local_id(0)] += fabs(q) * TRILININTERPOL(cube, weights);

			#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
			partial_interE[get_local_id(0)] += fabs(q) * TRILININTERPOL(cube, weights);
			#endif
		}

	} // End atom_id for-loop (INTERMOLECULAR ENERGY)


	#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_local_id(0) == 0)
	{
		float energy_interE = partial_interE[0];

		for (uint contributor_counter=1;
		          contributor_counter<NUM_OF_THREADS_PER_BLOCK;
		          contributor_counter++)
		{
			energy_interE += partial_interE[contributor_counter];
		}
		partial_interE[0] = energy_interE;
		//printf("%-20s %-10.8f\n", "energy_interE: ", energy_interE);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	#endif


	// In paper: intermolecular and internal energy calculation
	// are independent from each other, -> NO BARRIER NEEDED
  	// but require different operations,
	// thus, they can be executed only sequentially on the GPU.

	// ================================================
	// CALCULATING INTRAMOLECULAR ENERGY
	// ================================================
	for (uint contributor_counter = get_local_id(0);
	          contributor_counter < dockpars_num_of_intraE_contributors;
	          contributor_counter +=NUM_OF_THREADS_PER_BLOCK)
	{
		// Getting atom IDs
		uint atom1_id = intraE_contributors_const[3*contributor_counter];
		uint atom2_id = intraE_contributors_const[3*contributor_counter+1];

		// Calculating vector components of vector going
		// from first atom's to second atom's coordinates
		float subx = calc_coords_x[atom1_id] - calc_coords_x[atom2_id];
		float suby = calc_coords_y[atom1_id] - calc_coords_y[atom2_id];
		float subz = calc_coords_z[atom1_id] - calc_coords_z[atom2_id];

		// Calculating atomic_distance
		float atomic_distance = native_sqrt(subx*subx + suby*suby + subz*subz)*dockpars_grid_spacing;

		// Calculating energy contributions
		if (atomic_distance < 8.0f)
		{
			// Getting type IDs
			uint atom1_typeid = atom_types_const[atom1_id];
			uint atom2_typeid = atom_types_const[atom2_id];

			// Getting optimum pair distance (opt_distance) from reqm and reqm_hbond
			// reqm: equilibrium internuclear separation 
			//       (sum of the vdW radii of two like atoms (A)) in the case of vdW
			// reqm_hbond: equilibrium internuclear separation
			//  	 (sum of the vdW radii of two like atoms (A)) in the case of hbond 
			float opt_distance;

			if (intraE_contributors_const[3*contributor_counter+2] == 1)	//H-bond
			{
				opt_distance = reqm_hbond [atom1_typeid] + reqm_hbond [atom2_typeid];
			}
			else	//van der Waals
			{
				opt_distance = 0.5f*(reqm [atom1_typeid] + reqm [atom2_typeid]);
			}

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

/*
			if (get_local_id (0) == 0) {

				if (intraE_contributors_const[3*contributor_counter+2] == 1)	//H-bond
				{
					printf("%-5s %u %u %f %f %f %f %f %f\n", "hbond", atom1_typeid, atom2_typeid, reqm_hbond [atom1_typeid], reqm_hbond [atom2_typeid], opt_distance, delta_distance, atomic_distance, smoothed_distance);

				}
				else	//van der Waals
				{
					printf("%-5s %u %u %f %f %f %f %f %f\n", "vdw", atom1_typeid, atom2_typeid, reqm [atom1_typeid], reqm [atom2_typeid], opt_distance, delta_distance, atomic_distance, smoothed_distance);	
				}
			}
*/

			// Calculating van der Waals / hydrogen bond term
			partial_energies[get_local_id(0)] += native_divide(VWpars_AC_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(smoothed_distance/*atomic_distance*/,12));

			#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
			partial_intraE[get_local_id(0)] += native_divide(VWpars_AC_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(smoothed_distance/*atomic_distance*/,12));
			#endif

			if (intraE_contributors_const[3*contributor_counter+2] == 1) {	//H-bond
				partial_energies[get_local_id(0)] -= native_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(smoothed_distance/*atomic_distance*/,10));
				#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
				partial_intraE[get_local_id(0)] -= native_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(smoothed_distance/*atomic_distance*/,10));
				#endif
			}
			else {	//van der Waals
				partial_energies[get_local_id(0)] -= native_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(smoothed_distance/*atomic_distance*/,6));

				#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
				partial_intraE[get_local_id(0)] -= native_divide(VWpars_BD_const[atom1_typeid * dockpars_num_of_atypes+atom2_typeid],native_powr(smoothed_distance/*atomic_distance*/,6));
				#endif
			}

			// Calculating electrostatic term
        		partial_energies[get_local_id(0)] += native_divide (
                                                             dockpars_coeff_elec * atom_charges_const[atom1_id] * atom_charges_const[atom2_id],
                                                             atomic_distance * (DIEL_A + native_divide(DIEL_B,(1.0f + DIEL_K*native_exp(-DIEL_B_TIMES_H*atomic_distance))))
                                                             );

			#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
			partial_intraE[get_local_id(0)] += native_divide (
                                                             dockpars_coeff_elec * atom_charges_const[atom1_id] * atom_charges_const[atom2_id],
                                                             atomic_distance * (DIEL_A + native_divide(DIEL_B,(1.0f + DIEL_K*native_exp(-DIEL_B_TIMES_H*atomic_distance))))
                                                             );
			#endif

			// Calculating desolvation term
			// 1/25.92 = 0.038580246913580245
			partial_energies[get_local_id(0)] += ((dspars_S_const[atom1_typeid] +
							       dockpars_qasp*fabs(atom_charges_const[atom1_id]))*dspars_V_const[atom2_typeid] +
					                       (dspars_S_const[atom2_typeid] +
							       dockpars_qasp*fabs(atom_charges_const[atom2_id]))*dspars_V_const[atom1_typeid]) *
					                       dockpars_coeff_desolv*native_exp(-0.03858025f*native_powr(atomic_distance, 2));

			#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
			partial_intraE[get_local_id(0)] += ((dspars_S_const[atom1_typeid] +
							       dockpars_qasp*fabs(atom_charges_const[atom1_id]))*dspars_V_const[atom2_typeid] +
					                       (dspars_S_const[atom2_typeid] +
							       dockpars_qasp*fabs(atom_charges_const[atom2_id]))*dspars_V_const[atom1_typeid]) *
					                       dockpars_coeff_desolv*native_exp(-0.03858025f*native_powr(atomic_distance, 2));
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

	barrier(CLK_LOCAL_MEM_FENCE);

	#if defined (DEBUG_ENERGY_KERNEL1) || defined (DEBUG_ENERGY_KERNEL4) || defined (DEBUG_ENERGY_KERNEL3)
	if (get_local_id(0) == 0)
	{
		float energy_intraE = partial_intraE[0];

		for (uint contributor_counter=1;
		          contributor_counter<NUM_OF_THREADS_PER_BLOCK;
		          contributor_counter++)
		{
			energy_intraE += partial_intraE[contributor_counter];
		}
		partial_intraE[0] = energy_intraE;
		//printf("%-20s %-10.8f\n", "energy_intraE: ", energy_intraE);

	}
	barrier(CLK_LOCAL_MEM_FENCE);
	#endif

}

#include "kernel1.cl"
#include "kernel2.cl"
#include "auxiliary_genetic.cl"
#include "kernel4.cl"
#include "kernel3.cl"
#include "calcgradient.cl"
#include "kernel_gradient.cl"
