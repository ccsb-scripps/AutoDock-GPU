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






















#ifndef CALCENERGY_H_
#define CALCENERGY_H_

#include <math.h>
#include <stdio.h>
#include <cstdint>

#include "calcenergy_basic.h"
#include "miscellaneous.h"
#include "processligand.h"
#include "getparameters.h"

// This struct is passed to the GPU global functions (OpenCL kernels) as input.
// Its members are parameters related to the ligand, the grid
// and the genetic algorithm, or they are pointers of GPU (ADM FPGA) memory areas
// used for storing different data such as the current
// and the next population genotypes and energies, the grids,
// the evaluation counters and the random number generator states.
typedef struct
{
	char  	 	num_of_atoms;
	char   		num_of_atypes;
	char		num_of_map_atypes;
	int    		num_of_intraE_contributors;
	char   		gridsize_x;
	char   		gridsize_y;
	char   		gridsize_z;
	float  		grid_spacing;
	float* 		fgrids;
	int    		rotbondlist_length;
	float  		coeff_elec;
	float  		coeff_desolv;
	float* 		conformations_current;
	float* 		energies_current;
	float* 		conformations_next;
	float* 		energies_next;
	int*   		evals_of_new_entities;
	unsigned int* 	prng_states;
	int    		pop_size;
	int    		num_of_genes;
	float  		tournament_rate;
	float  		crossover_rate;
	float  		mutation_rate;
	float  		abs_max_dmov;
	float  		abs_max_dang;
	float  		lsearch_rate;
	float 		smooth;
	unsigned int 	num_of_lsentities;
	float  		rho_lower_bound;
	float  		base_dmov_mul_sqrt3;
	float  		base_dang_mul_sqrt3;
	unsigned int 	cons_limit;
	unsigned int 	max_num_of_iters;
	float  		qasp;
} Dockparameters;

// ----------------------------------------------------------------------
// The original function does CUDA calls initializing const kernel data.
// We create a struct to hold those constants and return them <here>
// (<here> = where prepare_const_fields_for_gpu() was called),
// so we can send them to kernels from <here>, instead of from calcenergy.cpp
// as originally.
// ----------------------------------------------------------------------

// Constant struct
/*
typedef struct
{
       float atom_charges_const[MAX_NUM_OF_ATOMS];
       char  atom_types_const  [MAX_NUM_OF_ATOMS];
       char  intraE_contributors_const[3*MAX_INTRAE_CONTRIBUTORS];
       float reqm_const        [ATYPE_NUM];
       float reqm_hbond_const  [ATYPE_NUM];
       unsigned int  atom1_types_reqm_const [ATYPE_NUM];
       unsigned int  atom2_types_reqm_const [ATYPE_NUM];
       float VWpars_AC_const   [MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
       float VWpars_BD_const   [MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
       float dspars_S_const    [MAX_NUM_OF_ATYPES];
       float dspars_V_const    [MAX_NUM_OF_ATYPES];
       int   rotlist_const     [MAX_NUM_OF_ROTATIONS];
       float ref_coords_x_const[MAX_NUM_OF_ATOMS];
       float ref_coords_y_const[MAX_NUM_OF_ATOMS];
       float ref_coords_z_const[MAX_NUM_OF_ATOMS];
       float rotbonds_moving_vectors_const[3*MAX_NUM_OF_ROTBONDS];
       float rotbonds_unit_vectors_const  [3*MAX_NUM_OF_ROTBONDS];
       float ref_orientation_quats_const  [4*MAX_NUM_OF_RUNS];
	// Added for calculating torsion-related gradients.
	// Passing list of rotbond-atoms ids to the GPU.
	// Contains the same information as processligand.h/Liganddata->rotbonds 
       int   rotbonds [2*MAX_NUM_OF_ROTBONDS];	

	// Contains the same information as processligand.h/Liganddata->atom_rotbonds
	// "atom_rotbonds": array that contains the rotatable bonds - atoms assignment.
	// If the element atom_rotbonds[atom index][rotatable bond index] is equal to 1,
	// it means,that the atom must be rotated if the bond rotates. A 0 means the opposite.
       int  rotbonds_atoms [MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS];

       int  num_rotating_atoms_per_rotbond [MAX_NUM_OF_ROTBONDS];
} kernelconstant;
*/

typedef struct
{
       float     atom_charges_const[MAX_NUM_OF_ATOMS];
       uint32_t  atom_types_const  [MAX_NUM_OF_ATOMS];
       uint32_t  atom_types_map_const  [MAX_NUM_OF_ATOMS];
} kernelconstant_interintra;

typedef struct
{
       uint32_t  intraE_contributors_const[3*MAX_INTRAE_CONTRIBUTORS];
} kernelconstant_intracontrib;

typedef struct
{
       float reqm_const [2*ATYPE_NUM]; // 1st ATYPE_NUM entries = vdW, 2nd ATYPE_NUM entries = hbond
       unsigned int  atom1_types_reqm_const [ATYPE_NUM];
       unsigned int  atom2_types_reqm_const [ATYPE_NUM];
       float VWpars_AC_const   [MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
       float VWpars_BD_const   [MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
       float dspars_S_const    [MAX_NUM_OF_ATYPES];
       float dspars_V_const    [MAX_NUM_OF_ATYPES];
} kernelconstant_intra;

typedef struct
{
       int   rotlist_const     [MAX_NUM_OF_ROTATIONS];
} kernelconstant_rotlist;

typedef struct
{
       float ref_coords_const[3*MAX_NUM_OF_ATOMS];
       float rotbonds_moving_vectors_const[3*MAX_NUM_OF_ROTBONDS];
       float rotbonds_unit_vectors_const  [3*MAX_NUM_OF_ROTBONDS];
       float ref_orientation_quats_const  [4*MAX_NUM_OF_RUNS];
} kernelconstant_conform;

typedef struct
{
	// Added for calculating torsion-related gradients.
	// Passing list of rotbond-atoms ids to the GPU.
	// Contains the same information as processligand.h/Liganddata->rotbonds 
       int   rotbonds [2*MAX_NUM_OF_ROTBONDS];	

	// Contains the same information as processligand.h/Liganddata->atom_rotbonds
	// "atom_rotbonds": array that contains the rotatable bonds - atoms assignment.
	// If the element atom_rotbonds[atom index][rotatable bond index] is equal to 1,
	// it means,that the atom must be rotated if the bond rotates. A 0 means the opposite.
       int  rotbonds_atoms [MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS];
       int  num_rotating_atoms_per_rotbond [MAX_NUM_OF_ROTBONDS];
} kernelconstant_grads;

/*
int prepare_const_fields_for_gpu(Liganddata* 	   myligand_reference,
				 Dockpars*   	   mypars,
				 float*      	   cpu_ref_ori_angles,
				 kernelconstant* KerConst);
*/

int prepare_const_fields_for_gpu(Liganddata* 	   		myligand_reference,
				 Dockpars*   	   		mypars,
				 float*      	   		cpu_ref_ori_angles,
				 kernelconstant_interintra*	KerConst_interintra,
				 kernelconstant_intracontrib*	KerConst_intracontrib,
				 kernelconstant_intra*		KerConst_intra,
				 kernelconstant_rotlist*	KerConst_rotlist,
				 kernelconstant_conform*	KerConst_conform,
				 kernelconstant_grads*          KerConst_grads);

void make_reqrot_ordering(char number_of_req_rotations[MAX_NUM_OF_ATOMS],
			  char atom_id_of_numrots[MAX_NUM_OF_ATOMS],
		          int  num_of_atoms);

int gen_rotlist(Liganddata* myligand,
		int         rotlist[MAX_NUM_OF_ROTATIONS]);

#endif /* CALCENERGY_H_ */
