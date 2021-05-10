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


#ifndef GETPARAMETERS_H_
#define GETPARAMETERS_H_

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <time.h>

#include "defines.h"
#include "processligand.h"
#include "processgrid.h"
#include "miscellaneous.h"
#include "calcenergy_basic.h"
#include "filelist.hpp"

#define LS_METHOD_STRING_LEN 8


typedef struct
{
	double AD4_coeff_vdW;
	double AD4_coeff_hb;
	double scaled_AD4_coeff_elec;
	double AD4_coeff_desolv;
	double AD4_coeff_tors;
} AD4_free_energy_coeffs;

// this needs to be a constexpr so it's resolved at compile time and not an object in multiple .o files
// (requires GCC > 4.9.0 which had a bug)
constexpr AD4_free_energy_coeffs unbound_models[3] = {
                                                      /* this model assumes the BOUND conformation is the SAME as the UNBOUND, default in AD4.2 */
                                                      {0.1662,0.1209,ELEC_SCALE_FACTOR*0.1406,0.1322,0.2983},
                                                      /* this model assumes the unbound conformation is EXTENDED, default if AD4.0 */
                                                      {0.1560,0.0974,ELEC_SCALE_FACTOR*0.1465,0.1159,0.2744},
                                                      /* this model assumes the unbound conformation is COMPACT */
                                                      {0.1641,0.0531,ELEC_SCALE_FACTOR*0.1272,0.0603,0.2272}
                                                     };

// Struct which contains the docking parameters (partly parameters for fpga)
typedef struct _Dockpars
{                                                                 // default values
	int                       devnum                          = -1;
	int                       devices_requested               = 1; // this is AD-GPU ...
	uint32_t                  seed[3]                         = {(uint32_t)time(NULL),(uint32_t)processid(),0};
	unsigned long             num_of_energy_evals             = 2500000;
	unsigned long             num_of_generations              = 42000;
	bool                      nev_provided                    = false;
	bool                      use_heuristics                  = true; // Flag if we want to use Diogo's heuristics
	unsigned long             heuristics_max                  = 12000000; // Maximum number of evaluations under the heuristics (12M max evaluates to 80% of 3M evals calculated by heuristics -> 2.4M)
	float                     abs_max_dmov;                   // depends on grid spacing
	float                     abs_max_dang                    = 90; // +/- 90°
	float                     mutation_rate                   = 2;  // 2%
	float                     crossover_rate                  = 80; // 80%
	float                     tournament_rate                 = 60; // 60%
	float                     lsearch_rate                    = 100; // 1000%
	float                     smooth                          = 0.5f;
	int                       nr_deriv_atypes                 = 0;    // this is to support: -derivtype C1,C2,C3=C
	deriv_atype*              deriv_atypes                    = NULL; // or even: -derivtype C1,C2,C3=C/S4=S/H5=HD
	int                       nr_mod_atype_pairs              = 0;    // this is to support: -modpair C1:S4,1.60,1.200,13,7
	pair_mod*                 mod_atype_pairs                 = NULL; // or even: -modpair C1:S4,1.60,1.200,13,7/C1:C3,1.20 0.025
	char                      ls_method[LS_METHOD_STRING_LEN] = "ad"; // "sw": Solis-Wets,
	                                                                  // "sd": Steepest-Descent
	                                                                  // "fire": FIRE, https://www.math.uni-bielefeld.de/~gaehler/papers/fire.pdf
	                                                                  // "ad": ADADELTA, https://arxiv.org/abs/1212.5701
	                                                                  // "adam": ADAM (currently only on Cuda)
	int                       initial_sw_generations          = 0;
	float                     rho_lower_bound                 = 0.01; // 0.01;
	float                     base_dmov_mul_sqrt3;            // depends on grid spacing
	float                     base_dang_mul_sqrt3             = 75.0*sqrt(3.0); // 75°
	unsigned long             cons_limit                      = 4;
	unsigned long             max_num_of_iters                = 300;
	unsigned long             pop_size                        = 150;
	char*                     load_xml                        = NULL;
	bool                      xml2dlg                         = false;
	bool                      contact_analysis                = false; // by default no distance-based contact analysis is performed
	std::vector<ReceptorAtom> receptor_atoms;
	unsigned int              nr_receptor_atoms               = 0;
	unsigned int*             receptor_map                    = NULL;
	unsigned int*             receptor_map_list               = NULL;
	float                     R_cutoff                        = 2.1;
	float                     H_cutoff                        = 3.7;
	float                     V_cutoff                        = 4.0;
	unsigned int              xml_files                       = 0;
	bool                      dlg2stdout                      = false;
	int                       gen_pdbs                        = 0;
	char*                     dpffile                         = NULL;
	char*                     fldfile                         = NULL;
	char*                     ligandfile                      = NULL;
	char*                     flexresfile                     = NULL;
	char*                     xrayligandfile                  = NULL;  // by default will be ligand file name
	char*                     resname                         = NULL; // by default will be ligand file basename
	bool                      given_xrayligandfile            = false; // That is, not given (explicitly by the user)
	bool                      autostop                        = true;
	unsigned int              as_frequency                    = 5;
	float                     stopstd                         = 0.15;
	bool                      cgmaps                          = false; // default is false (use a single map for every CGx or Gx atom type)
	unsigned long             num_of_runs                     = 20;
	unsigned int              list_nr                         = 0;
	bool                      reflig_en_required              = false;
	int                       unbound_model                   = 0;                 // bound same as unbound, the coefficients
	AD4_free_energy_coeffs    coeffs                          = unbound_models[0]; // are also set in get_filenames_and_ADcoeffs()
	float                     elec_min_distance               = 0.01;
	bool                      handle_symmetry                 = true;
	bool                      gen_finalpop                    = false;
	bool                      gen_best                        = false;
	float                     qasp                            = 0.01097f;
	float                     rmsd_tolerance                  = 2.0; // 2 Angstroem
	float                     adam_beta1                      = 0.9f;
	float                     adam_beta2                      = 0.999f;
	float                     adam_epsilon                    = 1.0e-8f;
	bool                      output_dlg                      = true; // dlg output file will be generated (by default)
	bool                      output_xml                      = true; // xml output file will be generated (by default)
} Dockpars;

inline bool add_deriv_atype(
                            Dockpars* mypars,
                            char*     name,
                            int       length
                           )
{
	mypars->nr_deriv_atypes++;
	mypars->deriv_atypes=(deriv_atype*)realloc(mypars->deriv_atypes,mypars->nr_deriv_atypes*sizeof(deriv_atype));
	if(mypars->deriv_atypes==NULL){
		printf("Error: Cannot allocate memory for -derivtype.\n");
		exit(1);
	}
	mypars->deriv_atypes[mypars->nr_deriv_atypes-1].nr=mypars->nr_deriv_atypes;
	if(length<4){
		strncpy(mypars->deriv_atypes[mypars->nr_deriv_atypes-1].deriv_name,name,length);
		mypars->deriv_atypes[mypars->nr_deriv_atypes-1].deriv_name[length]='\0';
	} else return false; // name is too long
	// make sure name hasn't already been used
	for(int i=0; i<mypars->nr_deriv_atypes-1; i++){
		if (strcmp(mypars->deriv_atypes[i].deriv_name, mypars->deriv_atypes[mypars->nr_deriv_atypes-1].deriv_name) == 0){
			printf("Error: -derivtype type name \"%s\" has already been used.\n",mypars->deriv_atypes[i].deriv_name);
			exit(2);
		}
	}
	return true;
}

bool argcmp(
            const char* arg,
            const char* cmd,
            const char shortarg = '\0'
           );

int preparse_dpf(
                 const int*      argc,
                       char**    argv,
                       Dockpars* mypars,
                       Gridinfo* mygrid,
                       FileList& filelist
                );

int get_filelist(
                 const int*      argc,
                       char**    argv,
                       Dockpars* mypars,
                       Gridinfo* mygrid,
                       FileList& filelist
                );

int get_filenames_and_ADcoeffs(
                               const int*,
                                     char**,
                                     Dockpars*,
                               const bool
                              );

void print_options(
                   const char* program_name
                  );

int get_commandpars(
                    const int*,
                          char**,
                          double*,
                          Dockpars*,
                    const bool late_call = true
                   );

std::vector<ReceptorAtom> read_receptor_atoms(
                                              const char* receptor_name
                                             );

std::vector<ReceptorAtom> read_receptor(
                                        const char* receptor_name,
                                        Gridinfo* mygrid,
                                        unsigned int* &in_reach_map,
                                        unsigned int* &atom_map_list,
                                        double cutoff = 4.2
                                       );

void read_xml_filenames(
                        char* xml_filename,
                        char* &dpf_filename,
                        char* &grid_filename,
                        char* &ligand_filename,
                        char* &flexres_filename,
                        unsigned int &list_nr,
                        uint32_t seed[3]
                       );

std::vector<float> read_xml_genomes(
                                    char* xml_filename,
                                    float grid_spacing,
                                    int &nrot,
                                    bool store_axisangle=false
                                   );

void gen_initpop_and_reflig(
                                  Dockpars*   mypars,
                                  float*      init_populations,
                                  Liganddata* myligand,
                            const Gridinfo*   mygrid
                           );

// these are the dpf file tokens we currently support -- although some by ignoring them ;-)
enum {DPF_UNKNOWN = -1,    DPF_NULL = 0,          DPF_COMMENT,      DPF_BLANK_LINE,
      DPF_MOVE,            DPF_FLD,               DPF_MAP,          DPF_ABOUT,
      DPF_TRAN0,           DPF_AXISANGLE0,        DPF_QUATERNION0,  DPF_QUAT0,
      DPF_DIHE0,           DPF_NDIHE,             DPF_TORSDOF,      DPF_INTNBP_COEFFS,
      DPF_INTNBP_REQM_EPS, DPF_RUNS,              DPF_GALS,         DPF_OUTLEV,
      DPF_RMSTOL,          DPF_EXTNRG,            DPF_INTELEC,      DPF_SMOOTH,
      DPF_SEED,            DPF_E0MAX,             DPF_SET_GA,       DPF_SET_SW1,
      DPF_SET_PSW1,        DPF_ANALYSIS,          GA_pop_size,      GA_num_generations,
      GA_num_evals,        GA_window_size,        GA_elitism,       GA_mutation_rate,
      GA_crossover_rate,   GA_Cauchy_alpha,       GA_Cauchy_beta,   SW_max_its,
      SW_max_succ,         SW_max_fail,           SW_rho,           SW_lb_rho,
      LS_search_freq,      DPF_PARAMETER_LIBRARY, DPF_LIGAND_TYPES, DPF_POPFILE,
      DPF_FLEXRES,         DPF_ELECMAP,           DPF_DESOLVMAP,    DPF_UNBOUND_MODEL};

int dpf_token(const char* token);

#endif /* GETPARAMETERS_H_ */


