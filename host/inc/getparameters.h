/*
 * (C) 2013. Evopro Innovation Kft.
 *
 * getparameters.h
 *
 *  Created on: 2008.10.22.
 *      Author: pechan.imre
 */

#ifndef GETPARAMETERS_H_
#define GETPARAMETERS_H_

#include <math.h>
#include <string.h>
#include <stdio.h>

#include "defines.h"
#include "processligand.h"
#include "processgrid.h"
#include "miscellaneous.h"

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
		float abs_max_dmov;
		float abs_max_dang;
		float mutation_rate;
		float crossover_rate;
		float lsearch_rate;
	unsigned long num_of_ls;
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
		float ref_ori_angles [3];
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
} Dockpars;

int get_filenames_and_ADcoeffs(const int*,
			           char**,
				Dockpars*);

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


