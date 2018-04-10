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
gpu_calc_initpop(	
			char   dockpars_num_of_atoms,
			char   dockpars_num_of_atypes,
			int    dockpars_num_of_intraE_contributors,
			char   dockpars_gridsize_x,
			char   dockpars_gridsize_y,
			char   dockpars_gridsize_z,
			float  dockpars_grid_spacing,
	 __global const float* restrict dockpars_fgrids, // This is too large to be allocated in __constant 
			int    dockpars_rotbondlist_length,
			float  dockpars_coeff_elec,
			float  dockpars_coeff_desolv,
	 __global const float* restrict dockpars_conformations_current,
	 __global       float* restrict dockpars_energies_current,
	 __global       int*   restrict dockpars_evals_of_new_entities,
			int    dockpars_pop_size,
			float  dockpars_qasp,
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
             __constant float* ref_orientation_quats_const
){
        // Some OpenCL compilers don't allow declaring 
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel, 
	// and then passed to non-kernel functions.
	__local float  genotype[GENOTYPE_LENGTH_IN_GLOBMEM];
	__local float  energy;
	__local int    run_id;

	__local float calc_coords_x[MAX_NUM_OF_ATOMS];
	__local float calc_coords_y[MAX_NUM_OF_ATOMS];
	__local float calc_coords_z[MAX_NUM_OF_ATOMS];
	__local float partial_energies[NUM_OF_THREADS_PER_BLOCK];

	// Copying genotype from global memory
	event_t ev = async_work_group_copy(genotype,
			                   dockpars_conformations_current + GENOTYPE_LENGTH_IN_GLOBMEM*get_group_id(0),
			                   GENOTYPE_LENGTH_IN_GLOBMEM, 0);

	wait_group_events(1,&ev);

	// Determining run-ID
	if (get_local_id(0) == 0) {
		run_id = get_group_id(0) / dockpars_pop_size;
	}

	// -------------------------------------------------------------------
	// Calculate gradients (forces) for intermolecular energy
	// Derived from autodockdev/maps.py
	// -------------------------------------------------------------------

	// Disabling gradient calculation for this kernel
	__local bool  is_enabled_gradient_calc;
	if (get_local_id(0) == 0) {
		is_enabled_gradient_calc = false;
	}

	// Variables to store gradient of 
	// the intermolecular energy per each ligand atom
	__local float gradient_inter_x[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_y[MAX_NUM_OF_ATOMS];
	__local float gradient_inter_z[MAX_NUM_OF_ATOMS];
	
	// Final gradient resulting out of gradient calculation
	__local float gradient_genotype[GENOTYPE_LENGTH_IN_GLOBMEM];
	// -------------------------------------------------------------------

	// =============================================================
	// WARNING: only energy of work-item=0 will be valid
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
			// Some OpenCL compilers don't allow declaring 
			// local variables within non-kernel functions.
			// These local variables must be declared in a kernel, 
			// and then passed to non-kernel functions.
			calc_coords_x,
			calc_coords_y,
			calc_coords_z,
			partial_energies,

	                atom_charges_const,
		        atom_types_const,
			intraE_contributors_const,
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
			ref_orientation_quats_const
		 	// Gradient-related arguments
		 	// Calculate gradients (forces) for intermolecular energy
		 	// Derived from autodockdev/maps.py
			,
			&is_enabled_gradient_calc,
			gradient_inter_x,
			gradient_inter_y,
			gradient_inter_z,
			gradient_genotype
			);
	// =============================================================

	if (get_local_id(0) == 0) {
		dockpars_energies_current[get_group_id(0)] = energy;
		dockpars_evals_of_new_entities[get_group_id(0)] = 1;
	}
}
