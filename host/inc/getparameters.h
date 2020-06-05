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










#ifndef GETPARAMETERS_H_
#define GETPARAMETERS_H_

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <vector>

#include "defines.h"
#include "processligand.h"
#include "processgrid.h"
#include "miscellaneous.h"
#include "calcenergy_basic.h"
#include "filelist.hpp"

typedef struct
{
	double AD4_coeff_vdW;
	double AD4_coeff_hb;
	double scaled_AD4_coeff_elec;
	double AD4_coeff_desolv;
	double AD4_coeff_tors;
} AD4_free_energy_coeffs;

typedef struct
//Struct which contains the docking parameters (partly parameters for fpga)
{
	unsigned long num_of_energy_evals;
	unsigned long num_of_generations;
		bool nev_provided;
		bool use_heuristics;
	unsigned long max_num_of_energy_evals;
		float abs_max_dmov;
		float abs_max_dang;
		float mutation_rate;
		float crossover_rate;
		float lsearch_rate;
		float smooth;
	unsigned long num_of_ls;
		char  ls_method[128];
		float tournament_rate;
		float rho_lower_bound;
		float base_dmov_mul_sqrt3;
		float base_dang_mul_sqrt3;
	unsigned long cons_limit;
	unsigned long max_num_of_iters;
	unsigned long pop_size;
		char  initpop_gen_or_loadfile;
		char  gen_pdbs;
		char  fldfile [128];
		char  ligandfile [128];
		char  xrayligandfile [128];
		bool  given_xrayligandfile;
		float ref_ori_angles [3];
	unsigned long devnum;
		bool  autostop;
		float stopstd;
		char  cgmaps;
	unsigned long num_of_runs;
		char  reflig_en_reqired;
		char  unbound_model;
	AD4_free_energy_coeffs coeffs;
		char  handle_symmetry;
		char  gen_finalpop;
		char  gen_best;
		char  resname [128];
		float qasp;
		float rmsd_tolerance;
        float adam_beta1;
        float adam_beta2;
        float adam_epsilon;
} Dockpars;

int get_filelist(const int* argc,
                      char** argv,
		     FileList& filelist);

int get_filenames_and_ADcoeffs(const int*,
			           char**,
				Dockpars*,
				const bool);

void get_commandpars(const int*,
		         char**,
			double*,
		      Dockpars*);

void gen_initpop_and_reflig(Dockpars*       mypars,
			    float*          init_populations,
			    float*          ref_ori_angles,
			    Liganddata*     myligand,
			    const Gridinfo* mygrid);

#endif /* GETPARAMETERS_H_ */


