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


// IMPORTANT: The following block contains definitions
// already made either in energy or gradient calculation files.
// For that reason, these are commented here.

// IMPORTANT: the code of gradient calculation was the initial template.
// Then, statements corresponding to enery calculations were added gradually.
// The latter can be distinguised this way: they are place within lines without indentation.

#define CONVERT_INTO_ANGSTROM_RADIAN  // DO NOT UNDEFINE, NO REALLY! DO!!! NOT!!! UNDEFINE!!! SML 200608 
#define SCFACTOR_ANGSTROM_RADIAN (1.0f/(DEG_TO_RAD * DEG_TO_RAD))

// Enables full floating point gradient calculation.
// Use is not advised as:
// - the determinism gradients (aka integer gradients) are much faster *and*
// - speed up the local search convergence
// Please only use for debugging
// #define FLOAT_GRADIENTS

// Enable restoring map gradient
// Currently, this is not a good idea
// #define RESTORING_MAP_GRADIENT

__device__ void gpu_calc_energrad(
                                  float*  genotype,
                                  float&  global_energy,
                                  int&    run_id,
                                  float3* calc_coords,
#if defined (DEBUG_ENERGY_KERNEL)
                                  float&  interE,
                                  float&  pintraE,
#endif
#ifdef FLOAT_GRADIENTS
                                  float3* gradient,
#else
                                  int3*   gradient,
#endif
                                  float*  fgradient_genotype,
                                  float*  pFloatAccumulator
                                 )
{
	float energy = 0.0f;
#if defined (DEBUG_ENERGY_KERNEL)
	interE = 0.0f;
	intraE = 0.0f;
#endif

	// Initializing gradients (forces)
	// Derived from autodockdev/maps.py
	for (uint32_t atom_id = threadIdx.x;
	              atom_id < cData.dockpars.num_of_atoms; // makes sure that gradient sum reductions give correct results if dockpars_num_atoms < NUM_OF_THREADS_PER_BLOCK
	              atom_id+= blockDim.x)
	{
		// Initialize coordinates
		calc_coords[atom_id].x = cData.pKerconst_conform->ref_coords_const[3*atom_id];
		calc_coords[atom_id].y = cData.pKerconst_conform->ref_coords_const[3*atom_id+1];
		calc_coords[atom_id].z = cData.pKerconst_conform->ref_coords_const[3*atom_id+2];

		// Intermolecular gradients
		gradient[atom_id].x = 0;
		gradient[atom_id].y = 0;
		gradient[atom_id].z = 0;
	}

	// Initializing gradient genotypes
	for (uint32_t gene_cnt = threadIdx.x;
	              gene_cnt < cData.dockpars.num_of_genes;
	              gene_cnt+= blockDim.x)
	{
		fgradient_genotype[gene_cnt] = 0;
	}

	// General rotation moving vector
	float4 genrot_movingvec;

	// Convert orientation genes from sex. to radians
	float phi         = genotype[3] * DEG_TO_RAD;
	float theta       = genotype[4] * DEG_TO_RAD;
	float genrotangle = genotype[5] * DEG_TO_RAD;

	float4 genrot_unitvec;
	float is_theta_gt_pi, sin_half_rotangle, sin_theta;
	if(cData.dockpars.true_ligand_atoms){
		genrot_movingvec.x = genotype[0];
		genrot_movingvec.y = genotype[1];
		genrot_movingvec.z = genotype[2];
		genrot_movingvec.w = 0.0f;
		sin_theta = sin(theta);
		float cos_theta = cos(theta);
		sin_half_rotangle = sin(genrotangle*0.5f);
		genrot_unitvec.x = sin_half_rotangle*sin_theta*cos(phi);
		genrot_unitvec.y = sin_half_rotangle*sin_theta*sin(phi);
		genrot_unitvec.z = sin_half_rotangle*cos_theta;
		genrot_unitvec.w = cos(genrotangle*0.5f);
		is_theta_gt_pi = 1.0f-2.0f*(float)(sin_theta < 0.0f);
	}

	uint32_t  g1 = cData.dockpars.gridsize_x;
	uint32_t  g2 = cData.dockpars.gridsize_x_times_y;
	uint32_t  g3 = cData.dockpars.gridsize_x_times_y_times_z;

	__syncthreads();

	// ================================================
	// CALCULATING ATOMIC POSITIONS AFTER ROTATIONS
	// ================================================
	for (uint32_t rotation_counter = threadIdx.x;
	              rotation_counter < cData.dockpars.rotbondlist_length;
	              rotation_counter+=blockDim.x)
	{
		int rotation_list_element = cData.pKerconst_rotlist->rotlist_const[rotation_counter];

		if ((rotation_list_element & RLIST_DUMMY_MASK) == 0) // If not dummy rotation
		{
			uint32_t atom_id = rotation_list_element & RLIST_ATOMID_MASK;

			// Capturing atom coordinates
			float4 atom_to_rotate;
			atom_to_rotate.x = calc_coords[atom_id].x;
			atom_to_rotate.y = calc_coords[atom_id].y;
			atom_to_rotate.z = calc_coords[atom_id].z;
			atom_to_rotate.w = 0.0f;

			// initialize with general rotation values
			float4 rotation_unitvec;
			float4 rotation_movingvec;
			if (atom_id < cData.dockpars.true_ligand_atoms){
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
				uint32_t rotbond_id = (rotation_list_element & RLIST_RBONDID_MASK) >> RLIST_RBONDID_SHIFT;

				float rotation_angle = genotype[6+rotbond_id]*DEG_TO_RAD*0.5f;
				float s = sin(rotation_angle);
				rotation_unitvec.x = s*cData.pKerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id];
				rotation_unitvec.y = s*cData.pKerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id+1];
				rotation_unitvec.z = s*cData.pKerconst_conform->rotbonds_unit_vectors_const[3*rotbond_id+2];
				rotation_unitvec.w = cos(rotation_angle);
				rotation_movingvec.x = cData.pKerconst_conform->rotbonds_moving_vectors_const[3*rotbond_id];
				rotation_movingvec.y = cData.pKerconst_conform->rotbonds_moving_vectors_const[3*rotbond_id+1];
				rotation_movingvec.z = cData.pKerconst_conform->rotbonds_moving_vectors_const[3*rotbond_id+2];

				// Performing additionally the first movement which
				// is needed only if rotating around rotatable bond
				atom_to_rotate.x -= rotation_movingvec.x;
				atom_to_rotate.y -= rotation_movingvec.y;
				atom_to_rotate.z -= rotation_movingvec.z;
			}

			// Performing rotation and final movement
			float4 qt = quaternion_rotate(atom_to_rotate,rotation_unitvec);
			calc_coords[atom_id].x = qt.x + rotation_movingvec.x;
			calc_coords[atom_id].y = qt.y + rotation_movingvec.y;
			calc_coords[atom_id].z = qt.z + rotation_movingvec.z;

		} // End if-statement not dummy rotation
			__syncthreads();
	} // End rotation_counter for-loop

	// ================================================
	// CALCULATING INTERMOLECULAR GRADIENTS
	// ================================================
	float weights[8];
	float cube[8];
	float inv_grid_spacing=1.0f/cData.dockpars.grid_spacing;
	for (uint32_t atom_id = threadIdx.x;
	              atom_id < cData.dockpars.num_of_atoms;
	              atom_id+= blockDim.x)
	{
		if (cData.pKerconst_interintra->ignore_inter_const[atom_id]>0) // first two atoms of a flex res are to be ignored here
			continue;
		float x = calc_coords[atom_id].x;
		float y = calc_coords[atom_id].y;
		float z = calc_coords[atom_id].z;
		float q = cData.pKerconst_interintra->atom_charges_const[atom_id];
		uint32_t atom_typeid = cData.pKerconst_interintra->atom_types_map_const[atom_id];

		if ((x < 0) || (y < 0) || (z < 0) || (x >= cData.dockpars.gridsize_x-1)
		                                  || (y >= cData.dockpars.gridsize_y-1)
		                                  || (z >= cData.dockpars.gridsize_z-1)){
#ifdef RESTORING_MAP_GRADIENT
			x -= 0.5f * cData.dockpars.gridsize_x;
			y -= 0.5f * cData.dockpars.gridsize_y;
			z -= 0.5f * cData.dockpars.gridsize_z;
			energy += 21.0f * (x*x+y*y+z*z); //100000.0f;
			#if defined (DEBUG_ENERGY_KERNEL)
			interE += 21.0f * (x*x+y*y+z*z);
			#endif
			// Setting gradients (forces) penalties.
			// The idea here is to push the offending
			// molecule towards the center rather
#ifdef FLOAT_GRADIENTS
			gradient[atom_id].x += 42.0f * x * inv_grid_spacing;
			gradient[atom_id].y += 42.0f * y * inv_grid_spacing;
			gradient[atom_id].z += 42.0f * z * inv_grid_spacing;
#else
			gradient[atom_id].x += lrintf(TERMSCALE * 42.0f * x * inv_grid_spacing);
			gradient[atom_id].y += lrintf(TERMSCALE * 42.0f * y * inv_grid_spacing);
			gradient[atom_id].z += lrintf(TERMSCALE * 42.0f * z * inv_grid_spacing);
#endif // FLOAT_GRADIENTS
#else
			energy += 16777216.0f; //100000.0f;
			#if defined (DEBUG_ENERGY_KERNEL)
			interE += 16777216.0f; //100000.0f;
			#endif
			gradient[atom_id].x += 16777216.0f;
			gradient[atom_id].y += 16777216.0f;
			gradient[atom_id].z += 16777216.0f;
#endif
			continue;
		}
		// Getting coordinates
		float x_low  = floor(x);
		float y_low  = floor(y);
		float z_low  = floor(z);

		// Grid value at 000
		float* grid_value_000 = cData.pMem_fgrids + ((ulong)(x_low  + y_low*g1  + z_low*g2)<<2);

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
		energy += cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7];
		#if defined (DEBUG_ENERGY_KERNEL)
		interE += cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7];
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
		atom_typeid = cData.dockpars.num_of_map_atypes;

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
		energy += q * (cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7]);
		#if defined (DEBUG_ENERGY_KERNEL)
		interE += q *(cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7]);
		#endif

		float q1 = q * inv_grid_spacing;
		// Vector in x-direction
		gx += q1 * ( omdz * (omdy * (cube [idx_100] - cube [idx_000]) + dy * (cube [idx_110] - cube [idx_010])) +
		               dz * (omdy * (cube [idx_101] - cube [idx_001]) + dy * (cube [idx_111] - cube [idx_011])));
		// Vector in y-direction
		gy += q1 * ( omdz * (omdx * (cube [idx_010] - cube [idx_000]) + dx * (cube [idx_110] - cube [idx_100])) +
		               dz * (omdx * (cube [idx_011] - cube [idx_001]) + dx * (cube [idx_111] - cube [idx_101])));
		// Vectors in z-direction
		gz += q1 * ( omdy * (omdx * (cube [idx_001] - cube [idx_000]) + dx * (cube [idx_101] - cube [idx_100])) +
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
		energy += q * (cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7]);
		#if defined (DEBUG_ENERGY_KERNEL)
		interE += q *(cube[0]*weights[0] + cube[1]*weights[1] + cube[2]*weights[2] + cube[3]*weights[3] + cube[4]*weights[4] + cube[5]*weights[5] + cube[6]*weights[6] + cube[7]*weights[7]);
		#endif

		q1 = fabs(q1);
		// Vector in x-direction
		gx += q1 * ( omdz * (omdy * (cube [idx_100] - cube [idx_000]) + dy * (cube [idx_110] - cube [idx_010])) +
		               dz * (omdy * (cube [idx_101] - cube [idx_001]) + dy * (cube [idx_111] - cube [idx_011])));
		// Vector in y-direction
		gy += q1 * ( omdz * (omdx * (cube [idx_010] - cube [idx_000]) + dx * (cube [idx_110] - cube [idx_100])) +
		               dz * (omdx * (cube [idx_011] - cube [idx_001]) + dx * (cube [idx_111] - cube [idx_101])));
		// Vectors in z-direction
		gz += q1 * ( omdy * (omdx * (cube [idx_001] - cube [idx_000]) + dx * (cube [idx_101] - cube [idx_100])) +
		               dy * (omdx * (cube [idx_011] - cube [idx_010]) + dx * (cube [idx_111] - cube [idx_110])));
		// -------------------------------------------------------------------
#ifdef FLOAT_GRADIENTS
		gradient[atom_id].x += gx;
		gradient[atom_id].y += gy;
		gradient[atom_id].z += gz;
#else
		gradient[atom_id].x += lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * gx)));
		gradient[atom_id].y += lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * gy)));
		gradient[atom_id].z += lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * gz)));
#endif
	} // End atom_id for-loop (INTERMOLECULAR ENERGY)
	__syncthreads();

	// Inter- and intra-molecular energy calculation
	// are independent from each other, so NO barrier is needed here.
	// As these two require different operations,
	// they can be executed only sequentially on the GPU.
	float delta_distance = 0.5f * cData.dockpars.smooth;
	float smoothed_distance;

	// ================================================
	// CALCULATING INTRAMOLECULAR GRADIENTS
	// ================================================
#ifdef REPRO
	// Simplest way to ensure random order of atomic addition doesn't make answers irreproducible: use only 1 thread
	if (threadIdx.x==0) for (uint32_t contributor_counter = 0; contributor_counter < cData.dockpars.num_of_intraE_contributors; contributor_counter+= 1) {
#else 
	for (uint32_t contributor_counter = threadIdx.x;
	              contributor_counter < cData.dockpars.num_of_intraE_contributors;
	              contributor_counter+= blockDim.x) {
#endif
		// Storing in a private variable 
		// the gradient contribution of each contributing atomic pair
		float priv_gradient_per_intracontributor= 0.0f;

		// Getting atom IDs
		uint32_t atom1_id = cData.pKerconst_intracontrib->intraE_contributors_const[2*contributor_counter];
		uint32_t atom2_id = cData.pKerconst_intracontrib->intraE_contributors_const[2*contributor_counter+1];

		// Calculating vector components of vector going
		// from first atom's to second atom's coordinates
		float subx = calc_coords[atom1_id].x - calc_coords[atom2_id].x;
		float suby = calc_coords[atom1_id].y - calc_coords[atom2_id].y;
		float subz = calc_coords[atom1_id].z - calc_coords[atom2_id].z;

		// Calculating atomic_distance
		float dist = sqrt(subx*subx + suby*suby + subz*subz);
		float atomic_distance = dist * cData.dockpars.grid_spacing;

		// Getting type IDs
		uint32_t atom1_typeid = cData.pKerconst_interintra->atom_types_const[atom1_id];
		uint32_t atom2_typeid = cData.pKerconst_interintra->atom_types_const[atom2_id];

		uint32_t atom1_type_vdw_hb = cData.pKerconst_intra->atom_types_reqm_const [atom1_typeid];
		uint32_t atom2_type_vdw_hb = cData.pKerconst_intra->atom_types_reqm_const [atom2_typeid];

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
		energy += vbond * atomic_distance;
		priv_gradient_per_intracontributor += vbond;
		// ------------------------------------------------

		// Calculating energy contributions
		// Cuttoff1: internuclear-distance at 8A only for vdw and hbond.
		if (atomic_distance < 8.0f)
		{
			uint32_t idx = atom1_typeid * cData.dockpars.num_of_atypes + atom2_typeid;
			ushort exps = cData.pKerconst_intra->VWpars_exp_const[idx];
			char m=(exps & 0xFF00)>>8;
			char n=(exps & 0xFF);
			// Getting optimum pair distance (opt_distance)
			float opt_distance = cData.pKerconst_intra->reqm_AB_const[idx];

			// Getting smoothed distance
			// smoothed_distance = function(atomic_distance, opt_distance)
			float opt_dist_delta = opt_distance - atomic_distance;
			if(fabs(opt_dist_delta)>=delta_distance){
				smoothed_distance = atomic_distance + copysign(delta_distance,opt_dist_delta);
			} else smoothed_distance = opt_distance;
			// Calculating van der Waals / hydrogen bond term
			float rmn = __powf(smoothed_distance,m-n);
			float rm = __powf(smoothed_distance,-m);
			energy += (cData.pKerconst_intra->VWpars_AC_const[idx]
			           -rmn*cData.pKerconst_intra->VWpars_BD_const[idx])*rm;
			priv_gradient_per_intracontributor += (n*cData.pKerconst_intra->VWpars_BD_const[idx]*rmn
			                                      -m*cData.pKerconst_intra->VWpars_AC_const[idx])*rm
			                                      /smoothed_distance;
			#if defined (DEBUG_ENERGY_KERNEL)
			intraE += (cData.pKerconst_intra->VWpars_AC_const[idx]
			           -rmn*cData.pKerconst_intra->VWpars_BD_const[idx])*rm
			#endif
		} // if cuttoff1 - internuclear-distance at 8A

		// Calculating energy contributions
		// Cuttoff2: internuclear-distance at 20.48A only for el and sol.
		if (atomic_distance < 20.48f)
		{
			if(atomic_distance<cData.dockpars.elec_min_distance)
				atomic_distance=cData.dockpars.elec_min_distance;
			float q1 = cData.pKerconst_interintra->atom_charges_const[atom1_id];
			float q2 = cData.pKerconst_interintra->atom_charges_const[atom2_id];
//			float exp_el = native_exp(DIEL_B_TIMES_H*atomic_distance);
			float dist2 = atomic_distance*atomic_distance;
			// Calculating desolvation term
			// 1/25.92 = 0.038580246913580245
			float desolv_energy =  ((cData.pKerconst_intra->dspars_S_const[atom1_typeid] +
						 cData.dockpars.qasp*fabs(q1)) * cData.pKerconst_intra->dspars_V_const[atom2_typeid] +
						(cData.pKerconst_intra->dspars_S_const[atom2_typeid] +
						 cData.dockpars.qasp*fabs(q2)) * cData.pKerconst_intra->dspars_V_const[atom1_typeid]) *
						 (
							cData.dockpars.coeff_desolv*(12.96f-0.1063f*dist2*(1.0f-0.001947f*dist2)) /
							(12.96f+dist2*(0.4137f+dist2*(0.00357f+0.000112f*dist2)))
						 );

			// Calculating electrostatic term
#ifndef DIEL_FIT_ABC
			float dist_shift=atomic_distance+1.26366f;
			dist2=dist_shift*dist_shift;
			float diel = 1.10859f / dist2 + 0.010358f;
#else
			float dist_shift=atomic_distance+1.588f;
			dist2=dist_shift*dist_shift;
			float disth_shift=atomic_distance+0.794f;
			float disth4=disth_shift*disth_shift;
			disth4*=disth4;
			float diel = 1.404f / dist2 + 0.072f / disth4 + 0.00831f;
#endif
			float es_energy = cData.dockpars.coeff_elec * q1 * q2 / atomic_distance;
			energy += diel * es_energy + desolv_energy;

			#if defined (DEBUG_ENERGY_KERNEL)
			intraE += diel * es_energy + desolv_energy;
			#endif

			// http://www.wolframalpha.com/input/?i=1%2F(x*(A%2B(B%2F(1%2BK*exp(-h*B*x)))))
/*			float exp_el_DIEL_K = exp_el + DIEL_K;
			float upper = DIEL_A * exp_el_DIEL_K*exp_el_DIEL_K +
			              DIEL_B * exp_el * (DIEL_B_TIMES_H_TIMES_K*atomic_distance + exp_el_DIEL_K);
			float lower = atomic_distance * (DIEL_A * exp_el_DIEL_K + DIEL_B * exp_el);
			lower *= lower;*/

			priv_gradient_per_intracontributor +=  -(es_energy / atomic_distance) * diel
#ifndef DIEL_FIT_ABC
			                                       -es_energy * 2.21718f / (dist2*dist_shift)
#else
			                                       -es_energy * ((2.808f / (dist2*dist_shift)) + (0.288f / (disth4*disth_shift)))
#endif
			                                       -0.0771605f * atomic_distance * desolv_energy; // 1/3.6^2 = 1/12.96 = 0.0771605
		} // if cuttoff2 - internuclear-distance at 20.48A

		// Decomposing "priv_gradient_per_intracontributor"
		// into the contribution of each atom of the pair.
		// Distances in Angstroms of vector that goes from
		// "atom1_id"-to-"atom2_id", therefore - subx, - suby, and - subz are used
		float grad_div_dist = -priv_gradient_per_intracontributor / dist;
#ifdef FLOAT_GRADIENTS
		float priv_intra_gradient_x = subx * grad_div_dist;
		float priv_intra_gradient_y = suby * grad_div_dist;
		float priv_intra_gradient_z = subz * grad_div_dist;
#else
		int priv_intra_gradient_x = lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * subx * grad_div_dist)));
		int priv_intra_gradient_y = lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * suby * grad_div_dist)));
		int priv_intra_gradient_z = lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * subz * grad_div_dist)));
#endif
		
		// Calculating gradients in xyz components.
		// Gradients for both atoms in a single contributor pair
		// have the same magnitude, but opposite directions
#ifdef FLOAT_GRADIENTS
		ATOMICSUBF32(&gradient[atom1_id].x, priv_intra_gradient_x);
		ATOMICSUBF32(&gradient[atom1_id].y, priv_intra_gradient_y);
		ATOMICSUBF32(&gradient[atom1_id].z, priv_intra_gradient_z);

		ATOMICADDF32(&gradient[atom2_id].x, priv_intra_gradient_x);
		ATOMICADDF32(&gradient[atom2_id].y, priv_intra_gradient_y);
		ATOMICADDF32(&gradient[atom2_id].z, priv_intra_gradient_z);
#else
		ATOMICSUBI32(&gradient[atom1_id].x, priv_intra_gradient_x);
		ATOMICSUBI32(&gradient[atom1_id].y, priv_intra_gradient_y);
		ATOMICSUBI32(&gradient[atom1_id].z, priv_intra_gradient_z);

		ATOMICADDI32(&gradient[atom2_id].x, priv_intra_gradient_x);
		ATOMICADDI32(&gradient[atom2_id].y, priv_intra_gradient_y);
		ATOMICADDI32(&gradient[atom2_id].z, priv_intra_gradient_z);
#endif
	} // End contributor_counter for-loop (INTRAMOLECULAR ENERGY)
	__syncthreads();

	// Transform gradients_inter_{x|y|z} 
	// into local_gradients[i] (with four quaternion genes)
	// Derived from autodockdev/motions.py/forces_to_delta_genes()

	// Transform local_gradients[i] (with four quaternion genes)
	// into local_gradients[i] (with three Shoemake genes)
	// Derived from autodockdev/motions.py/_get_cube3_gradient()
	// ------------------------------------------

	// start by populating "gradient_intra_*" with torque values
	float4 torque_rot;
	torque_rot.x = 0.0f;
	torque_rot.y = 0.0f;
	torque_rot.z = 0.0f;
	float gx = 0.0f;
	float gy = 0.0f;
	float gz = 0.0f;
	// overall rotation is only for the moving ligand
	for (uint32_t atom_cnt = threadIdx.x;
	              atom_cnt < cData.dockpars.true_ligand_atoms;
	              atom_cnt+= blockDim.x) {
		float3 r;
		r.x = (calc_coords[atom_cnt].x - genrot_movingvec.x) * cData.dockpars.grid_spacing;
		r.y = (calc_coords[atom_cnt].y - genrot_movingvec.y) * cData.dockpars.grid_spacing;
		r.z = (calc_coords[atom_cnt].z - genrot_movingvec.z) * cData.dockpars.grid_spacing;

		// Re-using "gradient_inter_*" for total gradient (inter+intra)
		float3 force;
#ifdef FLOAT_GRADIENTS
		force.x = gradient[atom_cnt].x;
		force.y = gradient[atom_cnt].y;
		force.z = gradient[atom_cnt].z;
#else
		force.x = ONEOVERTERMSCALE * (float)gradient[atom_cnt].x;
		force.y = ONEOVERTERMSCALE * (float)gradient[atom_cnt].y;
		force.z = ONEOVERTERMSCALE * (float)gradient[atom_cnt].z;
#endif
		gx += force.x;
		gy += force.y;
		gz += force.z;
		float4 tr = cross(r, force);
		torque_rot.x += tr.x;
		torque_rot.y += tr.y;
		torque_rot.z += tr.z;
	}

#ifdef USE_NVTENSOR
	/* Begin: Reduction using tensor units */

	// Implementation based on M.Sc. thesis by Gabin Schieffer at KTH:
	// "Accelerating a Molecular Docking Application by Leveraging Modern Heterogeneous Computing Systemx"
	// https://www.diva-portal.org/smash/get/diva2:1786161/FULLTEXT01.pdf

	// 1. Convert data-to-be-reduced from float to half
	// and place it in a shared memory array
	#ifdef USE_TCEC
	__shared__ __align__(256) float data_to_be_reduced[4*NUM_OF_THREADS_PER_BLOCK];
	data_to_be_reduced[4*threadIdx.x] = torque_rot.x;
	data_to_be_reduced[4*threadIdx.x + 1] = torque_rot.y;
	data_to_be_reduced[4*threadIdx.x + 2] = torque_rot.z;
	data_to_be_reduced[4*threadIdx.x + 3] = energy;
	#else
	__shared__ __align__(256) half data_to_be_reduced[4*NUM_OF_THREADS_PER_BLOCK];
	data_to_be_reduced[4*threadIdx.x] = __float2half(torque_rot.x);
	data_to_be_reduced[4*threadIdx.x + 1] = __float2half(torque_rot.y);
	data_to_be_reduced[4*threadIdx.x + 2] = __float2half(torque_rot.z);
	data_to_be_reduced[4*threadIdx.x + 3] = __float2half(energy);
	#endif

	// 2. Perform reduction via tensor units
	reduce_via_tensor_units(data_to_be_reduced);

	// 3. Retrieve results from shared memory
	#ifdef USE_TCEC
	torque_rot.x = data_to_be_reduced[0];
	torque_rot.y = data_to_be_reduced[1];
	torque_rot.z = data_to_be_reduced[2];
	energy = data_to_be_reduced[3];
	#else
	torque_rot.x = __half2float(data_to_be_reduced[0]);
	torque_rot.y = __half2float(data_to_be_reduced[1]);
	torque_rot.z = __half2float(data_to_be_reduced[2]);
	energy = __half2float(data_to_be_reduced[3]);
	#endif

	/* End: Reduction using tensor units */
#else
	// Reduction over the total gradient containing prepared "gradient_intra_*" values
	REDUCEFLOATSUM(torque_rot.x, pFloatAccumulator);
	REDUCEFLOATSUM(torque_rot.y, pFloatAccumulator);
	REDUCEFLOATSUM(torque_rot.z, pFloatAccumulator);

	// Reduction over partial energies and prepared "gradient_intra_*" values
	REDUCEFLOATSUM(energy, pFloatAccumulator);
#endif

	// TODO
	// -------------------------------------------------------
	// Obtaining energy and translation-related gradients
	// -------------------------------------------------------

#if defined (DEBUG_ENERGY_KERNEL)
	REDUCEFLOATSUM(intraE, pFloatAccumulator);
#endif

#ifdef USE_NVTENSOR
	/* Begin: Reduction using tensor units */

	// Implementation based on M.Sc. thesis by Gabin Schieffer at KTH:
	// "Accelerating a Molecular Docking Application by Leveraging Modern Heterogeneous Computing Systemx"
	// https://www.diva-portal.org/smash/get/diva2:1786161/FULLTEXT01.pdf

	// 1. Convert data-to-be-reduced from float to half
	// and place it in a shared memory array

	#ifdef USE_TCEC
	data_to_be_reduced[4*threadIdx.x] = gx;
	data_to_be_reduced[4*threadIdx.x + 1] = gy;
	data_to_be_reduced[4*threadIdx.x + 2] = gz;
	#else
	data_to_be_reduced[4*threadIdx.x] = __float2half(gx);
	data_to_be_reduced[4*threadIdx.x + 1] = __float2half(gy);
	data_to_be_reduced[4*threadIdx.x + 2] = __float2half(gz);
	#endif

	// 2. Perform reduction via tensor units
	reduce_via_tensor_units(data_to_be_reduced);

	// 3. Retrieve results from shared memory
	#ifdef USE_TCEC
	gx = data_to_be_reduced[0];
	gy = data_to_be_reduced[1];
	gz = data_to_be_reduced[2];
	#else
	gx = __half2float(data_to_be_reduced[0]);
	gy = __half2float(data_to_be_reduced[1]);
	gz = __half2float(data_to_be_reduced[2]);
	#endif

	/* End: Reduction using tensor units */
#else
	REDUCEFLOATSUM(gx, pFloatAccumulator);
	REDUCEFLOATSUM(gy, pFloatAccumulator);
	REDUCEFLOATSUM(gz, pFloatAccumulator);
#endif

	global_energy = energy;
#ifndef FLOAT_GRADIENTS
	int* gradient_genotype = (int*)fgradient_genotype;
#endif
	if ((threadIdx.x == 0) && (cData.dockpars.true_ligand_atoms)) {
		// Scaling gradient for translational genes as
		// their corresponding gradients were calculated in the space
		// where these genes are in Angstrom,
		// but AutoDock-GPU translational genes are within in grids
#ifdef FLOAT_GRADIENTS
		fgradient_genotype[0] = gx * cData.dockpars.grid_spacing;
		fgradient_genotype[1] = gy * cData.dockpars.grid_spacing;
		fgradient_genotype[2] = gz * cData.dockpars.grid_spacing;

		#if defined (PRINT_GRAD_TRANSLATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("gradient_x:%f\n", fgradient_genotype [0]);
		printf("gradient_y:%f\n", fgradient_genotype [1]);
		printf("gradient_z:%f\n", fgradient_genotype [2]);
		#endif
#else
		gradient_genotype[0] = lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * gx * cData.dockpars.grid_spacing)));
		gradient_genotype[1] = lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * gy * cData.dockpars.grid_spacing)));
		gradient_genotype[2] = lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * gz * cData.dockpars.grid_spacing)));

		#if defined (PRINT_GRAD_TRANSLATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("gradient_x:%f\n", gradient_genotype [0]);
		printf("gradient_y:%f\n", gradient_genotype [1]);
		printf("gradient_z:%f\n", gradient_genotype [2]);
		#endif
#endif
	}
	__syncthreads();

	// ------------------------------------------
	// Obtaining rotation-related gradients
	// ------------------------------------------
	if ((threadIdx.x == 0) && (cData.dockpars.true_ligand_atoms)) {
		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-20s %-10.6f %-10.6f %-10.6f\n", "final torque: ", torque_rot.x, torque_rot.y, torque_rot.z);
		#endif

		// Derived from rotation.py/axisangle_to_q()
		// genes[3:7] = rotation.axisangle_to_q(torque, rad)
		float torque_length = norm3df(torque_rot.x, torque_rot.y, torque_rot.z);
		float orientation_scaling = orientation_scaling = (torque_length<INFINITESIMAL_RADIAN) ? 1.0f : torque_length * INV_INFINITESIMAL_RADIAN;

		float torque_scale = (torque_length<INFINITESIMAL_RADIAN) ? 0.5f - torque_length*torque_length/48.0f : SIN_HALF_INFINITESIMAL_RADIAN/torque_length;

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-20s %-10.6f\n", "torque length: ", torque_length);
		#endif

		// Finding the quaternion that performs
		// the infinitesimal rotation around torque axis
		float4 quat_torque;
		quat_torque.x = torque_rot.x * torque_scale;
		quat_torque.y = torque_rot.y * torque_scale;
		quat_torque.z = torque_rot.z * torque_scale;
		quat_torque.w = (torque_length<INFINITESIMAL_RADIAN) ? 1.0f-torque_length*torque_length*0.125f : COS_HALF_INFINITESIMAL_RADIAN;

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
		float sin_ang = sqrt(1.0f-target_q.w*target_q.w); // = native_sin(ang);

		target_theta = PI_TIMES_2 + is_theta_gt_pi * fast_acos(target_q.z / sin_ang );
		target_phi   = fmod_pi2((atan2( is_theta_gt_pi*target_q.y, is_theta_gt_pi*target_q.x) + PI_TIMES_2));

		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s %-10.6f %-10.6f %-10.6f\n", "target_axisangle (1,2,3): ", target_phi, target_theta, target_rotangle);
		#endif
		
		// The infinitesimal rotation will produce an infinitesimal displacement
		// in shoemake space. This is to guarantee that the direction of
		// the displacement in shoemake space is not distorted.
		// The correct amount of displacement in shoemake space is obtained
		// by multiplying the infinitesimal displacement by shoemake_scaling
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

		float rot_angle_corr = 4.0f * sin_half_rotangle * sin_half_rotangle; // 4*sin(rotangle/2)^2
		
		// Setting gradient rotation-related genotypes in cube
		// Multiplicating by DEG_TO_RAD is to make it uniform to DEG (see torsion gradients)
#ifdef FLOAT_GRADIENTS
		fgradient_genotype[3] = grad_phi * sin_theta * sin_theta * rot_angle_corr * DEG_TO_RAD;
		fgradient_genotype[4] = grad_theta * rot_angle_corr * DEG_TO_RAD;
		fgradient_genotype[5] = grad_rotangle * DEG_TO_RAD;
		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s \n", "grad_axisangle (1,2,3) - after empirical scaling: ");
		printf("%-13s %-13s %-13s \n", "grad_phi", "grad_theta", "grad_rotangle");
		printf("%-13.6f %-13.6f %-13.6f\n", fgradient_genotype[3], fgradient_genotype[4], fgradient_genotype[5]);
		#endif
#else
		gradient_genotype[3] = lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * grad_phi * sin_theta * sin_theta * rot_angle_corr * DEG_TO_RAD)));
		gradient_genotype[4] = lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * grad_theta * rot_angle_corr * DEG_TO_RAD)));
		gradient_genotype[5] = lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * grad_rotangle * DEG_TO_RAD)));
		#if defined (PRINT_GRAD_ROTATION_GENES)
		printf("\n%s\n", "----------------------------------------------------------");
		printf("%-30s \n", "grad_axisangle (1,2,3) - after empirical scaling: ");
		printf("%-13s %-13s %-13s \n", "grad_phi", "grad_theta", "grad_rotangle");
		printf("%-13.6f %-13.6f %-13.6f\n", gradient_genotype[3], gradient_genotype[4], gradient_genotype[5]);
		#endif
#endif
	}
	__syncthreads();

	// ------------------------------------------
	// Obtaining torsion-related gradients
	// ------------------------------------------
	uint32_t num_torsion_genes = cData.dockpars.num_of_genes-6;
	for (uint32_t idx = threadIdx.x; idx < num_torsion_genes * cData.dockpars.num_of_atoms; idx += blockDim.x) {
		uint32_t rotable_atom_cnt = idx / num_torsion_genes;
		uint32_t rotbond_id = idx - rotable_atom_cnt * num_torsion_genes; // this is a bit cheaper than % (modulo)

		if (rotable_atom_cnt >= cData.pMem_num_rotating_atoms_per_rotbond_const[rotbond_id])
			continue; // Nothing to do

		// Querying ids of atoms belonging to the rotatable bond in question
		int atom1_id = cData.pMem_rotbonds_const[2*rotbond_id];
		int atom2_id = cData.pMem_rotbonds_const[2*rotbond_id+1];

		float3 atomRef_coords;
		atomRef_coords.x = calc_coords[atom1_id].x;
		atomRef_coords.y = calc_coords[atom1_id].y;
		atomRef_coords.z = calc_coords[atom1_id].z;
		float3 rotation_unitvec;

		rotation_unitvec.x = calc_coords[atom2_id].x - atomRef_coords.x;
		rotation_unitvec.y = calc_coords[atom2_id].y - atomRef_coords.y;
		rotation_unitvec.z = calc_coords[atom2_id].z - atomRef_coords.z;
		float l = rnorm3df(rotation_unitvec.x, rotation_unitvec.y, rotation_unitvec.z);
		rotation_unitvec.x *= l;
		rotation_unitvec.y *= l;
		rotation_unitvec.z *= l;

		// Torque of torsions
		uint lig_atom_id = cData.pMem_rotbonds_atoms_const[MAX_NUM_OF_ATOMS*rotbond_id + rotable_atom_cnt];
		float4 torque_tor;
		float3 r, atom_force;

		// Calculating torque on point "A"
		// They are converted back to Angstroms here
		r.x = (calc_coords[lig_atom_id].x - atomRef_coords.x);
		r.y = (calc_coords[lig_atom_id].y - atomRef_coords.y);
		r.z = (calc_coords[lig_atom_id].z - atomRef_coords.z);

		// Re-using "gradient_inter_*" for total gradient (inter+intra)
#ifdef FLOAT_GRADIENTS
		atom_force.x = gradient[lig_atom_id].x;
		atom_force.y = gradient[lig_atom_id].y;
		atom_force.z = gradient[lig_atom_id].z;
#else
		atom_force.x = ONEOVERTERMSCALE * gradient[lig_atom_id].x;
		atom_force.y = ONEOVERTERMSCALE * gradient[lig_atom_id].y;
		atom_force.z = ONEOVERTERMSCALE * gradient[lig_atom_id].z;
#endif
		torque_tor = cross(r, atom_force);
		float torque_on_axis = (rotation_unitvec.x * torque_tor.x  +
					rotation_unitvec.y * torque_tor.y  +
					rotation_unitvec.z * torque_tor.z) * cData.dockpars.grid_spacing;

		// Assignment of gene-based gradient
		// - this works because a * (a_1 + a_2 + ... + a_n) = a*a_1 + a*a_2 + ... + a*a_n
#ifdef FLOAT_GRADIENTS
		ATOMICADDF32(&fgradient_genotype[rotbond_id+6], torque_on_axis * DEG_TO_RAD); /*(M_PI / 180.0f)*/;
#else
		ATOMICADDI32(&gradient_genotype[rotbond_id+6], lrintf(fminf(MAXTERM, fmaxf(-MAXTERM, TERMSCALE * torque_on_axis * DEG_TO_RAD)))); /*(M_PI / 180.0f)*/;
#endif
	}
	__syncthreads();

#ifndef FLOAT_GRADIENTS
	for (uint32_t gene_cnt = threadIdx.x;
	              gene_cnt < cData.dockpars.num_of_genes;
	              gene_cnt+= blockDim.x) {
		fgradient_genotype[gene_cnt] = ONEOVERTERMSCALE * (float)gradient_genotype[gene_cnt];
	}
	__syncthreads();
#endif
	#if defined (CONVERT_INTO_ANGSTROM_RADIAN)
	for (uint32_t gene_cnt = threadIdx.x+3; // Only for gene_cnt > 2 means start gene_cnt at 3
	              gene_cnt < cData.dockpars.num_of_genes;
	              gene_cnt+= blockDim.x)
	{
		fgradient_genotype[gene_cnt] *= cData.dockpars.grid_spacing * cData.dockpars.grid_spacing * SCFACTOR_ANGSTROM_RADIAN;
	}
	__syncthreads();
	#endif
}
