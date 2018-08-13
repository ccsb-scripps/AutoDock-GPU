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


__kernel void __attribute__ ((reqd_work_group_size(NUM_OF_THREADS_PER_BLOCK,1,1)))
gpu_calc_initpop(	char   dockpars_num_of_atoms,
			char   dockpars_num_of_atypes,
			int    dockpars_num_of_intraE_contributors,
			char   dockpars_gridsize_x,
			char   dockpars_gridsize_y,
			char   dockpars_gridsize_z,
			float  dockpars_grid_spacing,
		#if defined (RESTRICT_ARGS)
		  __global const float* restrict dockpars_fgrids, // cannot be allocated in __constant (too large)
		#else
		  __global const float* dockpars_fgrids,          // cannot be allocated in __constant (too large)
		#endif
			int    dockpars_rotbondlist_length,
			float  dockpars_coeff_elec,
			float  dockpars_coeff_desolv,

		#if defined (RESTRICT_ARGS)
			__global const float* restrict dockpars_conformations_current,
			__global float* restrict dockpars_energies_current,
		        __global int*   restrict dockpars_evals_of_new_entities,
		#else
			__global const float* dockpars_conformations_current,
			__global float* dockpars_energies_current,
		        __global int*   dockpars_evals_of_new_entities,
		#endif

			int    dockpars_pop_size,
			float  dockpars_qasp,

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
){
	__local float  genotype[ACTUAL_GENOTYPE_LENGTH];
	__local float  energy;
	__local int    run_id;

        // Some OpenCL compilers don't allow local var outside kernels
        // so this local vars are passed from a kernel
	__local float calc_coords_x[MAX_NUM_OF_ATOMS];
	__local float calc_coords_y[MAX_NUM_OF_ATOMS];
	__local float calc_coords_z[MAX_NUM_OF_ATOMS];
	__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];

	event_t ev = async_work_group_copy(genotype,
			                   dockpars_conformations_current + GENOTYPE_LENGTH_IN_GLOBMEM*get_group_id(0),
			                   ACTUAL_GENOTYPE_LENGTH, 0);

	wait_group_events(1,&ev);

	//determining run ID
	if (get_local_id(0) == 0) {
		run_id = get_group_id(0) / dockpars_pop_size;
	}

	// Asynchronous copy should be finished by here
	wait_group_events(1,&ev);

	// Evaluating initial genotype
	barrier(CLK_LOCAL_MEM_FENCE);

	// =============================================================
	gpu_calc_energy(dockpars_rotbondlist_length,
			dockpars_num_of_atoms,
			dockpars_gridsize_x,
			dockpars_gridsize_y,
			dockpars_gridsize_z,
			dockpars_fgrids,
			dockpars_num_of_atypes,
			dockpars_num_of_intraE_contributors,
			dockpars_grid_spacing,
			dockpars_coeff_elec,
			dockpars_qasp,
			dockpars_coeff_desolv,
			genotype,
			&energy,
			&run_id,
			// Some OpenCL compilers don't allow local var outside kernels
			// so this local vars are passed from a kernel
			calc_coords_x,
			calc_coords_y,
			calc_coords_z,
			partial_energies,

	                atom_charges_const,
		        atom_types_const,
			intraE_contributors_const,
			dockpars_smooth,
			reqm,
			reqm_hbond,
	     	        atom1_types_reqm,
	     	        atom2_types_reqm,
			VWpars_AC_const,
			VWpars_BD_const,
			dspars_S_const,
			dspars_V_const,
			rotlist_const,
			ref_coords_x_const,
			ref_coords_y_const,
			ref_coords_z_const,
			rotbonds_moving_vectors_const,
			rotbonds_unit_vectors_const,
			ref_orientation_quats_const);
	// =============================================================

	if (get_local_id(0) == 0) {
		dockpars_energies_current[get_group_id(0)] = energy;
		dockpars_evals_of_new_entities[get_group_id(0)] = 1;
	}
}
