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


#define TERMBITS         10
#define MAXTERM          ((float)(1 << (31 - TERMBITS - 8))) // 2^(31 - 10 - 8) = 2^13 = 8192
#define TERMSCALE        ((float)(1 << TERMBITS)) // 2^10 = 1024
#define ONEOVERTERMSCALE (1.0f / TERMSCALE) // 1 / 1024 = 0.000977

// Enables full floating point gradient calculation.
// Use is not advised as:
// - the determinism gradients (aka integer gradients) are much faster *and*
// - speed up the local search convergence
// Please only use for debugging
// #define FLOAT_GRADIENTS

// Enable restoring map gradient
// Currently, this is not a good idea
// #define RESTORING_MAP_GRADIENT

// IMPORTANT: the code of gradient calculation was the initial template.
// Then, statements corresponding to enery calculations were added gradually.
// The latter can be distinguised this way: they are place within lines without indentation.

void gpu_calc_energrad(
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
                     __local float* partial_energies,
#if defined (DEBUG_ENERGY_KERNEL)
                     __local float* partial_interE,
                     __local float* partial_intraE,
#endif
            __constant       kernelconstant_interintra*   kerconst_interintra,
              __global const kernelconstant_intracontrib* kerconst_intracontrib,
            __constant       kernelconstant_intra*        kerconst_intra,
            __constant       kernelconstant_rotlist*      kerconst_rotlist,
            __constant       kernelconstant_conform*      kerconst_conform,

            __constant       int*   rotbonds_const,
              __global const int*   rotbonds_atoms_const,
            __constant int*         num_rotating_atoms_per_rotbond_const,

              __global const float* angle_const,
            __constant       float* dependence_on_theta_const,
            __constant       float* dependence_on_rotangle_const,
                             int    dockpars_num_of_genes,
#ifdef FLOAT_GRADIENTS
                     __local float* gradient_x,
                     __local float* gradient_y,
                     __local float* gradient_z,
#else
                     __local int*   gradient_x,
                     __local int*   gradient_y,
                     __local int*   gradient_z,
#endif
                     __local float* accumulator_x,
                     __local float* accumulator_y,
                     __local float* accumulator_z,
                     __local float* gradient_genotype
                      )
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
	          atom_id < MAX_NUM_OF_ATOMS; // makes sure that gradient sum reductions give correct results if dockpars_num_atoms < NUM_OF_THREADS_PER_BLOCK
	          atom_id+= NUM_OF_THREADS_PER_BLOCK)
	{
		// Initialize coordinates
		calc_coords[atom_id] = (float4)(kerconst_conform->ref_coords_const[3*atom_id],
		                                kerconst_conform->ref_coords_const[3*atom_id+1],
		                                kerconst_conform->ref_coords_const[3*atom_id+2],0);
		// Integer gradients
		gradient_x[atom_id] = 0;
		gradient_y[atom_id] = 0;
		gradient_z[atom_id] = 0;
	}

	// Initializing gradient genotypes
	for ( int gene_cnt = tidx;
	          gene_cnt < dockpars_num_of_genes;
	          gene_cnt+= NUM_OF_THREADS_PER_BLOCK)
	{
		gradient_genotype[gene_cnt] = 0.0f;
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
	float is_theta_gt_pi = 1.0f-2.0f*(float)(sin_angle < 0.0f);

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
	// CALCULATING INTERMOLECULAR GRADIENTS
	// ================================================
	float inv_grid_spacing = native_recip(dockpars_grid_spacing);
	float weights[8];
	float cube[8];
	for ( int atom_id = tidx;
	          atom_id < dockpars_num_of_atoms;
	          atom_id+= NUM_OF_THREADS_PER_BLOCK)
	{
		if (kerconst_interintra->ignore_inter_const[atom_id]>0) // first two atoms of a flex res are to be ignored here
			continue;
		float x = calc_coords[atom_id].x;
		float y = calc_coords[atom_id].y;
		float z = calc_coords[atom_id].z;
		float q = kerconst_interintra->atom_charges_const[atom_id];
		uint atom_typeid = kerconst_interintra->atom_types_map_const[atom_id];

		if ((x < 0) || (y < 0) || (z < 0) || (x >= dockpars_gridsize_x-1)
		                                  || (y >= dockpars_gridsize_y-1)
		                                  || (z >= dockpars_gridsize_z-1)){
			x -= 0.5f * dockpars_gridsize_x;
			y -= 0.5f * dockpars_gridsize_y;
			z -= 0.5f * dockpars_gridsize_z;
#ifdef RESTORING_MAP_GRADIENT
			partial_energies[tidx] += 21.0f * (x*x+y*y+z*z); //100000.0f;
#else
			partial_energies[tidx] += 16777216.0f; //100000.0f;
#endif
			#if defined (DEBUG_ENERGY_KERNEL)
			partial_interE[tidx] += 21.0f * (x*x+y*y+z*z);
			#endif
#ifdef RESTORING_MAP_GRADIENT
			// Setting gradients (forces) penalties.
			// The idea here is to push the offending
			// molecule towards the center
#ifdef FLOAT_GRADIENTS
			gradient_x[atom_id] += TERMSCALE * 42.0f * x * inv_grid_spacing;
			gradient_y[atom_id] += TERMSCALE * 42.0f * y * inv_grid_spacing;
			gradient_z[atom_id] += TERMSCALE * 42.0f * z * inv_grid_spacing;
#else
			gradient_x[atom_id] += convert_int_rte( TERMSCALE * 42.0f * x * inv_grid_spacing );
			gradient_y[atom_id] += convert_int_rte( TERMSCALE * 42.0f * y * inv_grid_spacing );
			gradient_z[atom_id] += convert_int_rte( TERMSCALE * 42.0f * z * inv_grid_spacing );
#endif // FLOAT_GRADIENTS
#else
			gradient_x[atom_id] += 16777216.0f;
			gradient_y[atom_id] += 16777216.0f;
			gradient_z[atom_id] += 16777216.0f;
#endif
			continue;
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

		// -------------------------------------------------------------------
		// Calculating gradients (forces) corresponding to 
		// "atype" intermolecular energy
		// Derived from autodockdev/maps.py
		// -------------------------------------------------------------------

		// Vector in x-direction
/*		x10 = cube [idx_100] - cube [idx_000]; // z = 0
		x52 = cube [idx_110] - cube [idx_010]; // z = 0
		x43 = cube [idx_101] - cube [idx_001]; // z = 1
		x76 = cube [idx_111] - cube [idx_011]; // z = 1
		vx_z0 = omdy * x10 + dy * x52;     // z = 0
		vx_z1 = omdy * x43 + dy * x76;     // z = 1
		gradient_inter_x[atom_id] += omdz * vx_z0 + dz * vx_z1;

		// AT - reduced to two variables:
		vx_z0 = omdy * (cube [idx_100] - cube [idx_000]) + dy * (cube [idx_110] - cube [idx_010]);     // z = 0
		vx_z1 = omdy * (cube [idx_101] - cube [idx_001]) + dy * (cube [idx_111] - cube [idx_011]);     // z = 1 */

		// AT - all in one go with no intermediate variables (following calcs are similar)
		// Vector in x-direction
		float gx = (omdz * (omdy * (cube [idx_100] - cube [idx_000]) + dy * (cube [idx_110] - cube [idx_010])) +
		              dz * (omdy * (cube [idx_101] - cube [idx_001]) + dy * (cube [idx_111] - cube [idx_011]))) * inv_grid_spacing;
		// Vector in y-direction
		float gy = (omdz * (omdx * (cube [idx_010] - cube [idx_000]) + dx * (cube [idx_110] - cube [idx_100])) +
		              dz * (omdx * (cube [idx_011] - cube [idx_001]) + dx * (cube [idx_111] - cube [idx_101]))) * inv_grid_spacing;
		// Vectors in z-direction
		float gz = (omdy * (omdx * (cube [idx_001] - cube [idx_000]) + dx * (cube [idx_101] - cube [idx_100])) +
		              dy * (omdx * (cube [idx_011] - cube [idx_010]) + dx * (cube [idx_111] - cube [idx_110]))) * inv_grid_spacing;
		// -------------------------------------------------------------------
		// Calculating gradients (forces) corresponding to 
		// "elec" intermolecular energy
		// Derived from autodockdev/maps.py
		// -------------------------------------------------------------------

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

		float qg = q*inv_grid_spacing;
		// Vector in x-direction
		gx += qg * ( omdz * (omdy * (cube [idx_100] - cube [idx_000]) + dy * (cube [idx_110] - cube [idx_010])) +
		               dz * (omdy * (cube [idx_101] - cube [idx_001]) + dy * (cube [idx_111] - cube [idx_011])));
		// Vector in y-direction
		gy += qg * ( omdz * (omdx * (cube [idx_010] - cube [idx_000]) + dx * (cube [idx_110] - cube [idx_100])) +
		               dz * (omdx * (cube [idx_011] - cube [idx_001]) + dx * (cube [idx_111] - cube [idx_101])));
		// Vectors in z-direction
		gz += qg * ( omdy * (omdx * (cube [idx_001] - cube [idx_000]) + dx * (cube [idx_101] - cube [idx_100])) +
		               dy * (omdx * (cube [idx_011] - cube [idx_010]) + dx * (cube [idx_111] - cube [idx_110])));
		// -------------------------------------------------------------------
		// Calculating gradients (forces) corresponding to 
		// "dsol" intermolecular energy
		// Derived from autodockdev/maps.py
		// -------------------------------------------------------------------
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

		qg = fabs(qg);
		// Vector in x-direction
		gx += qg * ( omdz * (omdy * (cube [idx_100] - cube [idx_000]) + dy * (cube [idx_110] - cube [idx_010])) +
		               dz * (omdy * (cube [idx_101] - cube [idx_001]) + dy * (cube [idx_111] - cube [idx_011])));
		// Vector in y-direction
		gy += qg * ( omdz * (omdx * (cube [idx_010] - cube [idx_000]) + dx * (cube [idx_110] - cube [idx_100])) +
		               dz * (omdx * (cube [idx_011] - cube [idx_001]) + dx * (cube [idx_111] - cube [idx_101])));
		// Vectors in z-direction
		gz += qg * ( omdy * (omdx * (cube [idx_001] - cube [idx_000]) + dx * (cube [idx_101] - cube [idx_100])) +
		               dy * (omdx * (cube [idx_011] - cube [idx_010]) + dx * (cube [idx_111] - cube [idx_110])));
		// -------------------------------------------------------------------
#ifdef FLOAT_GRADIENTS
		gradient_x[atom_id] += gx;
		gradient_y[atom_id] += gy;
		gradient_z[atom_id] += gz;
#else
		gradient_x[atom_id] += convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * gx)));
		gradient_y[atom_id] += convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * gy)));
		gradient_z[atom_id] += convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * gz)));
#endif
	} // End atom_id for-loop (INTERMOLECULAR ENERGY)

	// Inter- and intra-molecular energy calculation
	// are independent from each other, so NO barrier is needed here.
	// As these two require different operations,
	// they can be executed only sequentially on the GPU.
	float delta_distance = 0.5f*dockpars_smooth;
	float smoothed_distance;

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

		// Calculating vector components of vector going
		// from first atom's to second atom's coordinates
		float subx = calc_coords[atom1_id].x - calc_coords[atom2_id].x;
		float suby = calc_coords[atom1_id].y - calc_coords[atom2_id].y;
		float subz = calc_coords[atom1_id].z - calc_coords[atom2_id].z;

		// Calculating atomic_distance
		float dist = native_sqrt(subx*subx + suby*suby + subz*subz);
		float atomic_distance = dist*dockpars_grid_spacing;

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
		priv_gradient_per_intracontributor += vbond;
		// ------------------------------------------------

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
			float rmn=native_powr(smoothed_distance,m-n);
			float rm=native_powr(smoothed_distance,-m);
			partial_energies[tidx] += (kerconst_intra->VWpars_AC_const[idx]-rmn*kerconst_intra->VWpars_BD_const[idx])*rm;
			priv_gradient_per_intracontributor += (n*kerconst_intra->VWpars_BD_const[idx]*rmn-m*kerconst_intra->VWpars_AC_const[idx])*rm*native_recip(smoothed_distance);
			#if defined (DEBUG_ENERGY_KERNEL)
			partial_intraE[tidx] += (kerconst_intra->VWpars_AC_const[idx]-rmn*kerconst_intra->VWpars_BD_const[idx])*rm;
			#endif
		} // if cuttoff1 - internuclear-distance at 8A

		// Calculating energy contributions
		// Cuttoff2: internuclear-distance at 20.48A only for el and sol.
		if (atomic_distance < 20.48f)
		{
			if(atomic_distance<dockpars_elec_min_distance) atomic_distance=dockpars_elec_min_distance;
			float q1 = kerconst_interintra->atom_charges_const[atom1_id];
			float q2 = kerconst_interintra->atom_charges_const[atom2_id];
//			float exp_el = native_exp(DIEL_B_TIMES_H*atomic_distance);
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
//			                       dockpars_coeff_desolv*native_exp(-0.03858025f*atomic_distance*atomic_distance);
			// Calculating electrostatic term
/*			partial_energies[tidx] += native_divide (
			                                         dockpars_coeff_elec * q1 * q2,
			                                         atomic_distance * (DIEL_A + native_divide(DIEL_B,(1.0f + native_divide(DIEL_K,exp_el))))
			                                        ) +
			                          desolv_energy;*/
#ifndef DIEL_FIT_ABC
			float dist_shift=atomic_distance+1.26366f;
			dist2=dist_shift*dist_shift;
			float diel = 1.10859f*native_recip(dist2)+0.010358f;
#else
			float dist_shift=atomic_distance+1.588f;
			dist2=dist_shift*dist_shift;
			float disth_shift=atomic_distance+0.794f;
			float disth4=disth_shift*disth_shift;
			disth4*=disth4;
			float diel = 1.404f*native_recip(dist2)+0.072f*native_recip(disth4)+0.00831f;
#endif
			float es_energy = dockpars_coeff_elec * q1 * q2 * native_recip(atomic_distance);
			partial_energies[tidx] += diel * es_energy + desolv_energy;

			#if defined (DEBUG_ENERGY_KERNEL)
/*			partial_intraE[tidx] += native_divide (
			                                       dockpars_coeff_elec * q1 * q2,
			                                       atomic_distance * (DIEL_A + native_divide(DIEL_B,(1.0f + native_divide(DIEL_K,exp_el))))
			                                      ) +
			                        desolv_energy;*/
			partial_intraE[tidx] += diel * es_energy + desolv_energy;
			#endif

			// http://www.wolframalpha.com/input/?i=1%2F(x*(A%2B(B%2F(1%2BK*exp(-h*B*x)))))
/*			float exp_el_DIEL_K = exp_el + DIEL_K;
			float upper = DIEL_A * exp_el_DIEL_K*exp_el_DIEL_K +
			              DIEL_B * exp_el * (DIEL_B_TIMES_H_TIMES_K*atomic_distance + exp_el_DIEL_K);
			float lower = atomic_distance * (DIEL_A * exp_el_DIEL_K + DIEL_B * exp_el);
			lower *= lower;*/

//			priv_gradient_per_intracontributor +=  -dockpars_coeff_elec * q1 * q2 * native_divide (upper, lower) -
//			                                       0.0771605f * atomic_distance * desolv_energy;
			priv_gradient_per_intracontributor +=  -es_energy*native_recip(atomic_distance) * diel
#ifndef DIEL_FIT_ABC
			                                       -es_energy * 2.21718f*native_recip(dist2*dist_shift)
#else
			                                       -es_energy * (2.808f * native_recip(dist2*dist_shift)+0.288f*native_recip(disth4*disth_shift))
#endif
			                                       -0.0771605f * atomic_distance * desolv_energy; // 1/3.6^2 = 1/12.96 = 0.0771605
		} // if cuttoff2 - internuclear-distance at 20.48A

		// Decomposing "priv_gradient_per_intracontributor"
		// into the contribution of each atom of the pair.
		// Distances in Angstroms of vector that goes from
		// "atom1_id"-to-"atom2_id", therefore - subx, - suby, and - subz are used
		float grad_div_dist = -priv_gradient_per_intracontributor*native_recip(dist);
#ifdef FLOAT_GRADIENTS
		float priv_intra_gradient_x = subx * grad_div_dist;
		float priv_intra_gradient_y = suby * grad_div_dist;
		float priv_intra_gradient_z = subz * grad_div_dist;
#else
		int priv_intra_gradient_x = convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * subx * grad_div_dist)));
		int priv_intra_gradient_y = convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * suby * grad_div_dist)));
		int priv_intra_gradient_z = convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * subz * grad_div_dist)));
#endif
		// Calculating gradients in xyz components.
		// Gradients for both atoms in a single contributor pair
		// have the same magnitude, but opposite directions
#ifdef FLOAT_GRADIENTS // the floating point atomic adds are time consuming
		atomicSub_g_f(&gradient_x[atom1_id], priv_intra_gradient_x);
		atomicSub_g_f(&gradient_y[atom1_id], priv_intra_gradient_y);
		atomicSub_g_f(&gradient_z[atom1_id], priv_intra_gradient_z);

		atomicAdd_g_f(&gradient_x[atom2_id], priv_intra_gradient_x);
		atomicAdd_g_f(&gradient_y[atom2_id], priv_intra_gradient_y);
		atomicAdd_g_f(&gradient_z[atom2_id], priv_intra_gradient_z);
#else
		atomic_sub(&gradient_x[atom1_id], priv_intra_gradient_x);
		atomic_sub(&gradient_y[atom1_id], priv_intra_gradient_y);
		atomic_sub(&gradient_z[atom1_id], priv_intra_gradient_z);

		atomic_add(&gradient_x[atom2_id], priv_intra_gradient_x);
		atomic_add(&gradient_y[atom2_id], priv_intra_gradient_y);
		atomic_add(&gradient_z[atom2_id], priv_intra_gradient_z);
#endif
	} // End contributor_counter for-loop (INTRAMOLECULAR ENERGY)
	barrier(CLK_LOCAL_MEM_FENCE);

	// -------------------------------------------------------
	// Obtaining energy and translation-related gradients
	// -------------------------------------------------------
	accumulator_x[tidx] = 0.0f;
	accumulator_y[tidx] = 0.0f;
	accumulator_z[tidx] = 0.0f;
	for ( int atom_cnt = tidx;
	          atom_cnt < dockpars_num_of_atoms;
	          atom_cnt+= NUM_OF_THREADS_PER_BLOCK)
	{
#ifdef FLOAT_GRADIENTS
		accumulator_x[tidx] += gradient_x[atom_cnt];
		accumulator_y[tidx] += gradient_y[atom_cnt];
		accumulator_z[tidx] += gradient_z[atom_cnt];
#else
		accumulator_x[tidx] += ONEOVERTERMSCALE * gradient_x[atom_cnt];
		accumulator_y[tidx] += ONEOVERTERMSCALE * gradient_y[atom_cnt];
		accumulator_z[tidx] += ONEOVERTERMSCALE * gradient_z[atom_cnt];
#endif
	}
	// reduction over partial energies and prepared "gradient_intra_*" values
	for ( int off=NUM_OF_THREADS_PER_BLOCK>>1; off>0; off >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (tidx < off)
		{
			partial_energies[tidx] += partial_energies[tidx+off];
#if defined (DEBUG_ENERGY_KERNEL)
			partial_intraE[tidx] += partial_intraE[tidx+off];
#endif
			accumulator_x[tidx] += accumulator_x[tidx+off];
			accumulator_y[tidx] += accumulator_y[tidx+off];
			accumulator_z[tidx] += accumulator_z[tidx+off];
		}
	}
	__local int* i_gradient_genotype = (__local int*)gradient_genotype;
	if (tidx == 0) {
		*energy = partial_energies[0];
		// Scaling gradient for translational genes as
		// their corresponding gradients were calculated in the space
		// where these genes are in Angstrom,
		// but AutoDock-GPU translational genes are within in grids
		i_gradient_genotype[0] = convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * accumulator_x[0] * dockpars_grid_spacing)));
		i_gradient_genotype[1] = convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * accumulator_y[0] * dockpars_grid_spacing)));
		i_gradient_genotype[2] = convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * accumulator_z[0] * dockpars_grid_spacing)));
		#if defined (PRINT_GRAD_TRANSLATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("gradient_x:%f\n", i_gradient_genotype [0]);
		printf("gradient_y:%f\n", i_gradient_genotype [1]);
		printf("gradient_z:%f\n", i_gradient_genotype [2]);
		#endif
	}
	barrier(CLK_LOCAL_MEM_FENCE);

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

	accumulator_x[tidx] = 0.0f;
	accumulator_y[tidx] = 0.0f;
	accumulator_z[tidx] = 0.0f;
	for ( int atom_cnt = tidx;
	          atom_cnt < dockpars_true_ligand_atoms;
	          atom_cnt+= NUM_OF_THREADS_PER_BLOCK)
	{
		float4 r = (calc_coords[atom_cnt] - genrot_movingvec) * dockpars_grid_spacing;
		// Re-using "gradient_inter_*" for total gradient (inter+intra)
		float4 force;
#ifdef FLOAT_GRADIENTS
		force.x = gradient_x[atom_cnt];
		force.y = gradient_y[atom_cnt];
		force.z = gradient_z[atom_cnt];
#else
		force.x = ONEOVERTERMSCALE * gradient_x[atom_cnt];
		force.y = ONEOVERTERMSCALE * gradient_y[atom_cnt];
		force.z = ONEOVERTERMSCALE * gradient_z[atom_cnt];
#endif
		force.w = 0.0f;
		float4 torque_rot = cross(r, force);
		accumulator_x[tidx] += torque_rot.x;
		accumulator_y[tidx] += torque_rot.y;
		accumulator_z[tidx] += torque_rot.z;
	}
	// do a reduction over the total gradient containing prepared "gradient_intra_*" values
	for ( int off=NUM_OF_THREADS_PER_BLOCK>>1; off>0; off >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (tidx < off)
		{
			accumulator_x[tidx] += accumulator_x[tidx+off];
			accumulator_y[tidx] += accumulator_y[tidx+off];
			accumulator_z[tidx] += accumulator_z[tidx+off];
		}
	}
	if (tidx == 0) {
		float4 torque_rot;
		torque_rot.x = accumulator_x[0];
		torque_rot.y = accumulator_y[0];
		torque_rot.z = accumulator_z[0];

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-20s %-10.6f %-10.6f %-10.6f\n", "final torque: ", torque_rot.x, torque_rot.y, torque_rot.z);
		#endif

		// Derived from rotation.py/axisangle_to_q()
		// genes[3:7] = rotation.axisangle_to_q(torque, rad)
		float torque_length = fast_length(torque_rot);
		torque_length += (torque_length<1e-20f)*1e-20f;
		
		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-20s %-10.6f\n", "torque length: ", torque_length);
		#endif

		// Finding the quaternion that performs
		// the infinitesimal rotation around torque axis
		float4 quat_torque = torque_rot * SIN_HALF_INFINITESIMAL_RADIAN * native_recip(torque_length);
		quat_torque.w = COS_HALF_INFINITESIMAL_RADIAN;

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-20s %-10.6f\n", "INFINITESIMAL_RADIAN: ", INFINITESIMAL_RADIAN);
		printf("%-20s %-10.6f %-10.6f %-10.6f %-10.6f\n", "quat_torque (w,x,y,z): ", quat_torque.w, quat_torque.x, quat_torque.y, quat_torque.z);
		#endif

		// Converting quaternion gradients into orientation gradients 
		// Derived from autodockdev/motion.py/_get_cube3_gradient
		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s %-10.6f %-10.6f %-10.6f %-10.6f\n", "current_q (w,x,y,z): ", genrot_unitvec.w, genrot_unitvec.x, genrot_unitvec.y, genrot_unitvec.z);
		#endif

		// This is where we want to be in quaternion space
		// target_q = rotation.q_mult(q, current_q)
		float4 target_q = quaternion_multiply(quat_torque, genrot_unitvec);

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s %-10.6f %-10.6f %-10.6f %-10.6f\n", "target_q (w,x,y,z): ", target_q.w, target_q.x, target_q.y, target_q.z);
		#endif

		// This is where we are in the orientation axis-angle space
		// Equivalent to "current_oclacube" in autodockdev/motions.py
		float current_phi      = fmod_pi2(PI_TIMES_2 + phi);
		float current_theta    = fmod_pi2(PI_TIMES_2 + theta);
		float current_rotangle = fmod_pi2(PI_TIMES_2 + genrotangle);

		// This is where we want to be in the orientation axis-angle space
		float target_phi, target_theta, target_rotangle;

		// target_oclacube = quaternion_to_oclacube(target_q, theta_larger_than_pi)
		// Derived from autodockdev/motions.py/quaternion_to_oclacube()
		// In our terms means quaternion_to_oclacube(target_q{w|x|y|z}, theta_larger_than_pi)
		target_rotangle = 2.0f * fast_acos(target_q.w); // = 2.0f * ang;
		float inv_sin_ang = native_rsqrt(1.0f-target_q.w*target_q.w); // = 1.0/native_sin(ang);

		target_theta = PI_TIMES_2 + is_theta_gt_pi * fast_acos( target_q.z * inv_sin_ang );
		target_phi   = fmod_pi2((atan2( is_theta_gt_pi*target_q.y, is_theta_gt_pi*target_q.x) + PI_TIMES_2));

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
		grad_phi      = orientation_scaling * (fmod_pi2(target_phi      - current_phi      + PI_FLOAT) - PI_FLOAT);
		grad_theta    = orientation_scaling * (fmod_pi2(target_theta    - current_theta    + PI_FLOAT) - PI_FLOAT);
		grad_rotangle = orientation_scaling * (fmod_pi2(target_rotangle - current_rotangle + PI_FLOAT) - PI_FLOAT);

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
		i_gradient_genotype[3] = convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * native_divide(grad_phi, (dependence_on_theta * dependence_on_rotangle)) * DEG_TO_RAD)));
		i_gradient_genotype[4] = convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * native_divide(grad_theta, dependence_on_rotangle) * DEG_TO_RAD)));
		i_gradient_genotype[5] = convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * grad_rotangle * DEG_TO_RAD)));
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
	int num_torsion_genes = dockpars_num_of_genes-6;
	for ( int idx = tidx;
	          idx < num_torsion_genes * dockpars_num_of_atoms;
	          idx += NUM_OF_THREADS_PER_BLOCK)
	{
		int rotable_atom_cnt = idx / num_torsion_genes;
		int rotbond_id = idx - rotable_atom_cnt * num_torsion_genes; // this is a bit cheaper than % (modulo)

		if (rotable_atom_cnt >= num_rotating_atoms_per_rotbond_const[rotbond_id])
			continue; // Nothing to do

		// Querying ids of atoms belonging to the rotatable bond in question
		int atom1_id = rotbonds_const[2*rotbond_id];
		int atom2_id = rotbonds_const[2*rotbond_id+1];

		float4 atomRef_coords;
		atomRef_coords = calc_coords[atom1_id];
		float4 rotation_unitvec = fast_normalize(calc_coords[atom2_id] - atomRef_coords);

		// Torque of torsions
		uint lig_atom_id = rotbonds_atoms_const[MAX_NUM_OF_ATOMS*rotbond_id + rotable_atom_cnt];
		float4 torque_tor, r, atom_force;

		// Calculating torque on point "A"
		// They are converted back to Angstroms here
		r = (calc_coords[lig_atom_id] - atomRef_coords);

		// Re-using "gradient_inter_*" for total gradient (inter+intra)
#ifdef FLOAT_GRADIENTS
		atom_force.x = gradient_x[lig_atom_id];
		atom_force.y = gradient_y[lig_atom_id];
		atom_force.z = gradient_z[lig_atom_id];
#else
		atom_force.x = ONEOVERTERMSCALE * gradient_x[lig_atom_id];
		atom_force.y = ONEOVERTERMSCALE * gradient_y[lig_atom_id];
		atom_force.z = ONEOVERTERMSCALE * gradient_z[lig_atom_id];
#endif
		atom_force.w = 0.0f;

		torque_tor = cross(r, atom_force);
		float torque_on_axis = dot(rotation_unitvec, torque_tor) * dockpars_grid_spacing; // it is cheaper to do a scalar multiplication than a vector one

		// Assignment of gene-based gradient
		// - this works because a * (a_1 + a_2 + ... + a_n) = a*a_1 + a*a_2 + ... + a*a_n
		atomic_add(&i_gradient_genotype[rotbond_id+6], convert_int_rte(fmin(MAXTERM, fmax(-MAXTERM, TERMSCALE * torque_on_axis * DEG_TO_RAD)))); /*(M_PI / 180.0f)*/;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for ( int gene_cnt = tidx;
	          gene_cnt < dockpars_num_of_genes;
	          gene_cnt+= NUM_OF_THREADS_PER_BLOCK)
	{
		gradient_genotype[gene_cnt] = ONEOVERTERMSCALE * (float)i_gradient_genotype[gene_cnt];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	#if defined (CONVERT_INTO_ANGSTROM_RADIAN)
	for ( int gene_cnt = tidx+3; // Only for gene_cnt > 2 means start gene_cnt at 3
	          gene_cnt < dockpars_num_of_genes;
	          gene_cnt+= NUM_OF_THREADS_PER_BLOCK)
	{
		gradient_genotype[gene_cnt] *= dockpars_grid_spacing * dockpars_grid_spacing * SCFACTOR_ANGSTROM_RADIAN;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	#endif
}
