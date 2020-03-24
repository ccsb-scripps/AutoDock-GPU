/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

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

// The GPU device function calculates the energy's gradient (forces or derivatives) 
// of the entity described by genotype, dockpars and the ligand-data
// arrays in constant memory and returns it in the "gradient" parameter. 
// The parameter "run_id" has to be equal to the ID of the run 
// whose population includes the current entity (which can be determined with get_group_id(0)), 
// since this determines which reference orientation should be used.

//#define PRINT_GRAD_TRANSLATION_GENES
//#define PRINT_GRAD_ROTATION_GENES
//#define PRINT_GRAD_TORSION_GENES

//#define ENABLE_PARALLEL_GRAD_TORSION

// The following is a scaling of gradients.
// Initially all genotypes and gradients
// were expressed in grid-units (translations)
// and sexagesimal degrees (rotation and torsion angles).
// Expressing them using angstroms / radians
// might help gradient-based minimizers.
// This conversion is applied to final gradients.
//#define CONVERT_INTO_ANGSTROM_RADIAN

// Scaling factor to multiply the gradients of 
// the genes expressed in degrees (all genes except the first three) 
// (GRID-SPACING * GRID-SPACING) / (DEG_TO_RAD * DEG_TO_RAD) = 461.644
//#define SCFACTOR_ANGSTROM_RADIAN ((0.375 * 0.375)/(DEG_TO_RAD * DEG_TO_RAD))

#include "calcenergy.hpp"

// CALCULATING INTERMOLECULAR ENERGY

template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_calc_energrad(const member_type& team_member, const DockingParams<Device>& docking_params,const float *genotype,const Conform<Device>& conform, const RotList<Device>& rotlist, const IntraContrib<Device>& intracontrib, const InterIntra<Device>& interintra, const Intra<Device>& intra, const Grads<Device>& grads, const AxisCorrection<Device>& axis_correction, float& energy, float* gradient)
{
        // Get team and league ranks
        int tidx = team_member.team_rank();
        int lidx = team_member.league_rank();

        // Determine which run this team is doing
        int run_id;
        if (tidx == 0) {
                run_id = lidx/docking_params.pop_size;
        }

        team_member.team_barrier();

	// Initialize energy - FIX ME - should be array length team_size - ALS
	float partial_energies = 0.0f;

	// Gradient of the intermolecular energy per each ligand atom
        // Also used to store the accummulated gradient per each ligand atom
        float gradient_inter_x[MAX_NUM_OF_ATOMS];
        float gradient_inter_y[MAX_NUM_OF_ATOMS];
        float gradient_inter_z[MAX_NUM_OF_ATOMS];

        // Gradient of the intramolecular energy per each ligand atom
        float gradient_intra_x[MAX_NUM_OF_ATOMS];
        float gradient_intra_y[MAX_NUM_OF_ATOMS];
        float gradient_intra_z[MAX_NUM_OF_ATOMS];

	// Initializing gradients (forces) 
        // Derived from autodockdev/maps.py
        for (int atom_id = tidx; atom_id < MAX_NUM_OF_ATOMS; atom_id+= team_member.team_size()) {
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
        for (int gene_cnt = tidx; gene_cnt < docking_params.num_of_genes; gene_cnt+= team_member.team_size()) {
                gradient[gene_cnt] = 0.0f;
        }

	// GETTING ATOMIC POSITIONS
	// ALS - This view needs to be in scratch space FIX ME
	Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords("calc_coords");
	Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, (int)(docking_params.num_of_atoms)),
	[=] (int& idx) {
		kokkos_get_atom_pos(idx, conform, calc_coords);
	});

	// CALCULATING ATOMIC POSITIONS AFTER ROTATIONS
	// General rotation moving vector
	float4struct genrot_movingvec;
	genrot_movingvec.x = genotype[0];
	genrot_movingvec.y = genotype[1];
	genrot_movingvec.z = genotype[2];
	genrot_movingvec.w = 0.0;

	// Convert orientation genes from sex. to radians
	float phi         = genotype[3] * DEG_TO_RAD;
	float theta       = genotype[4] * DEG_TO_RAD;
	float genrotangle = genotype[5] * DEG_TO_RAD;

	float4struct genrot_unitvec;
	float sin_angle = sin(theta);
	float s2 = sin(genrotangle*0.5f);
	genrot_unitvec.x = s2*sin_angle*cos(phi);
	genrot_unitvec.y = s2*sin_angle*sin(phi);
	genrot_unitvec.z = s2*cos(theta);
	genrot_unitvec.w = cos(genrotangle*0.5f);
	bool is_theta_gt_pi = 1.0-2.0*(float)(sin_angle < 0.0f); // WTF - ALS

	team_member.team_barrier();

	// FIX ME This will break once multi-threading - ALS
	// Loop over the rot bond list and carry out all the rotations
	Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, docking_params.rotbondlist_length),
	[=] (int& idx) {
		kokkos_rotate_atoms(idx, conform, rotlist, run_id, genotype, genrot_movingvec, genrot_unitvec, calc_coords);
	});

	team_member.team_barrier();

	// CALCULATING INTERMOLECULAR GRADIENTS
//	kokkos_calc_intermolecular_gradients();

	// CALCULATING INTRAMOLECULAR GRADIENTS
//	kokkos_calc_intramolecular_gradients();

	team_member.team_barrier();

	// ACCUMULATE INTER- AND INTRAMOLECULAR GRADIENTS
//	kokkos_acculumate_interintra_gradients();

	team_member.team_barrier();

	// Obtaining energy and translation-related gradients
//	kokkos_calc_energy_and_translation_gradients();

	team_member.team_barrier();

	// Obtaining rotation-related gradients
//	kokkos_calc_rotation_gradients();

	team_member.team_barrier();

	// Obtaining torsion-related gradients
//	kokkos_calc_torsion_gradients();

#if defined (CONVERT_INTO_ANGSTROM_RADIAN)
	team_member.team_barrier();

        for (int gene_cnt = tidx+3; // Only for gene_cnt > 2 means start gene_cnt at 3
                  gene_cnt < docking_params.num_of_genes;
                  gene_cnt+= team_member.team_size()) {
                gradient[gene_cnt] *= SCFACTOR_ANGSTROM_RADIAN;
        }
#endif

	team_member.team_barrier();
}
