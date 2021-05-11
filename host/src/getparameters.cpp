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


#include <cstdint>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <locale>

#include "getparameters.h"

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

int dpf_token(const char* token)
{
	const struct {
		const char* string;
		const int   token;
		bool        evaluate; // need to evaluate parameters (and either use it or make sure it's a certain value)
	} supported_dpf_tokens [] = {
	      {"ligand",             DPF_MOVE,              true},  /* movable ligand file name */
	      {"move",               DPF_MOVE,              true},  /* movable ligand file name */
	      {"fld",                DPF_FLD,               true},  /* grid data file name */
	      {"map",                DPF_MAP,               true},  /* grid map specifier */
	      {"about",              DPF_ABOUT,             false}, /* rotate about */
	      {"tran0",              DPF_TRAN0,             true},  /* translate (needs to be "random") */
	      {"axisangle0",         DPF_AXISANGLE0,        true},  /* rotation axisangle (needs to be "random") */
	      {"quaternion0",        DPF_QUATERNION0,       true},  /* quaternion (of rotation, needs to be "random") */
	      {"quat0",              DPF_QUAT0,             true},  /* quaternion (of rotation, needs to be "random") */
	      {"dihe0",              DPF_DIHE0,             true},  /* number of dihedrals (needs to be "random") */
	      {"ndihe",              DPF_NDIHE,             false}, /* number of dihedrals (is in pdbqt) */
	      {"torsdof",            DPF_TORSDOF,           false}, /* torsional degrees of freedom (is in pdbqt) */
	      {"intnbp_coeffs",      DPF_INTNBP_COEFFS,     true},  /* internal pair energy coefficients */
	      {"intnbp_r_eps",       DPF_INTNBP_REQM_EPS,   true},  /* internal pair energy coefficients */
	      {"runs",               DPF_RUNS,              true},  /* number of runs */
	      {"ga_run",             DPF_GALS,              true},  /* run a number of runs */
	      {"gals_run",           DPF_GALS,              true},  /* run a number of runs */
	      {"outlev",             DPF_OUTLEV,            false}, /* output level */
	      {"rmstol",             DPF_RMSTOL,            true},  /* RMSD cluster tolerance */
	      {"extnrg",             DPF_EXTNRG,            false}, /* external grid energy */
	      {"intelec",            DPF_INTELEC,           true},  /* calculate ES energy (needs not be "off") */
	      {"smooth",             DPF_SMOOTH,            true},  /* smoothing range */
	      {"seed",               DPF_SEED,              true},  /* random number seed */
	      {"e0max",              DPF_E0MAX,             false}, /* simanneal max inital energy (ignored) */
	      {"set_ga",             DPF_SET_GA,            false}, /* use genetic algorithm (yes, that's us) */
	      {"set_sw1",            DPF_SET_SW1,           false}, /* use Solis-Wets (we are by default)*/
	      {"set_psw1",           DPF_SET_PSW1,          false}, /* use pseudo Solis-Wets (nope, SW) */
	      {"analysis",           DPF_ANALYSIS,          false}, /* analysis data (we're doing it) */
	      {"ga_pop_size",        GA_pop_size,           true},  /* population size */
	      {"ga_num_generations", GA_num_generations,    true},  /* number of generations */
	      {"ga_num_evals",       GA_num_evals,          true},  /* number of evals */
	      {"ga_window_size",     GA_window_size,        false}, /* genetic algorithm window size */
	      {"ga_elitism",         GA_elitism,            false}, /* GA parameters: */
	      {"ga_mutation_rate",   GA_mutation_rate,      true},  /*     The ones set to true */
	      {"ga_crossover_rate",  GA_crossover_rate,     true},  /*     have a corresponding */
	      {"ga_cauchy_alpha",    GA_Cauchy_alpha,       false}, /*     parameter in AD-GPU  */
	      {"ga_cauchy_beta",     GA_Cauchy_beta,        false}, /*     the others ignored   */
	      {"sw_max_its",         SW_max_its,            true},  /* local search iterations */
	      {"sw_max_succ",        SW_max_succ,           true},  /* cons. success limit */
	      {"sw_max_fail",        SW_max_fail,           true},  /* cons. failure limit */
	      {"sw_rho",             SW_rho,                false}, /* rho - is 1.0 here */
	      {"sw_lb_rho",          SW_lb_rho,             true},  /* lower bound of rho */
	      {"ls_search_freq",     LS_search_freq,        false}, /* ignored as likely wrong for algorithm here */
	      {"parameter_file",     DPF_PARAMETER_LIBRARY, false}, /* parameter file (use internal currently) */
	      {"ligand_types",       DPF_LIGAND_TYPES,      true},  /* ligand types used */
	      {"output_pop_file",    DPF_POPFILE,           false}, /* output population to file */
	      {"flexible_residues",  DPF_FLEXRES,           true},  /* flexibe residue file name */
	      {"flexres",            DPF_FLEXRES,           true},  /* flexibe residue file name */
	      {"elecmap",            DPF_ELECMAP,           false}, /* electrostatic grid map (we use fld file basename) */
	      {"desolvmap",          DPF_DESOLVMAP,         false}, /* desolvation grid map (we use fld file basename) */
	      {"dsolvmap",           DPF_DESOLVMAP,         false}, /* desolvation grid map (we use fld file basename) */
	      {"unbound_model",      DPF_UNBOUND_MODEL,     true}   /* unbound model (bound|extended|compact) */
	                            };

	if (token[0]=='\0')
		return DPF_BLANK_LINE;
	if (token[0]=='#')
		return DPF_COMMENT;

	for (int i=0; i<(int)(sizeof(supported_dpf_tokens)/sizeof(*supported_dpf_tokens)); i++){
		if(stricmp(supported_dpf_tokens[i].string,token) == 0){
			if(supported_dpf_tokens[i].evaluate)
				return supported_dpf_tokens[i].token;
			else
				return DPF_NULL;
			break; // found
		}
	}

	return DPF_UNKNOWN;
}

int parse_dpf(
              Dockpars* mypars,
              Gridinfo* mygrid,
              FileList& filelist
             )
{
	if (mypars->dpffile)
	{
		std::ifstream file(mypars->dpffile);
		if(file.fail()){
			printf("\nError: Could not open dpf file %s. Check path and permissions.\n",mypars->dpffile);
			return 1;
		}
		mypars->elec_min_distance = 0.5; // default for AD4
		std::string line;
		char tempstr[256], argstr[256];
		char* args[2];
		int tempint, i, len;
		float tempfloat;
		int line_count = 0;
		int ltype_nr = 0;
		int mtype_nr = 0;
		// use a 4-times longer runway for atom types definable in one dpf
		// - this is to allow more reactive types and to enable the strategy
		//   to define all possible atom types for a set of ligands once for
		//   performance reasons (as they can be read once)
		// - each ligand is still going to be limited to MAX_NUM_OF_ATYPES
		char ltypes[4*MAX_NUM_OF_ATYPES][4];
		char* typestr;
		memset(ltypes,0,16*MAX_NUM_OF_ATYPES*sizeof(char));
		unsigned int idx;
		pair_mod* curr_pair;
		float paramA, paramB;
		int m, n;
		char typeA[4], typeB[4];
		filelist.max_len = 256;
		bool new_device = false; // indicate if current mypars has a new device requested
		unsigned int run_cnt=0;
		while(std::getline(file, line)) {
			line_count++;
			trim(line); // Remove leading and trailing whitespace
			tempstr[0]='\0';
			sscanf(line.c_str(),"%255s",tempstr);
			int token_id = dpf_token(tempstr);
			switch(token_id){
				case DPF_MOVE: // movable ligand file name
						if(!mypars->xml2dlg){
							sscanf(line.c_str(),"%*s %255s",argstr);
							if(mypars->ligandfile) free(mypars->ligandfile);
							mypars->ligandfile = strdup(argstr);
						}
						break;
				case DPF_FLEXRES: // flexibe residue file name
						if(!mypars->xml2dlg){
							sscanf(line.c_str(),"%*s %255s",argstr);
							if(mypars->flexresfile) free(mypars->flexresfile);
							mypars->flexresfile = strdup(argstr);
						}
						break;
				case DPF_FLD: // grid data file name
						if(!mypars->xml2dlg){
							sscanf(line.c_str(),"%*s %255s",argstr);
							// Add the .fld file
							if(mypars->fldfile) free(mypars->fldfile);
							mypars->fldfile = strdup(argstr); // this allows using the dpf to set up all parameters but the ligand
							// Filling mygrid according to the specified fld file
							mygrid->info_read = false;
							if (get_gridinfo(mypars->fldfile, mygrid) != 0)
							{
								printf("\nError: get_gridinfo failed with fld file specified with <%s> parameter at %s:%u.\n",tempstr,mypars->dpffile,line_count);
								return 1;
							}
						}
						break;
				case DPF_LIGAND_TYPES: // ligand types used
						len=-1;
						for(i=strlen(tempstr); i<(int)line.size(); i++){
							if(isspace(line[i])){ // whitespace
								len=-1;
							} else{ // not whitespace aka an atom type
								if(len<0){ // new type starts
									len=i;
									ltype_nr++;
									if(ltype_nr>4*MAX_NUM_OF_ATYPES){
										printf("\nError: Too many atoms types (>%d)defined in the <%s> parameter at %s:%u.\n",4*MAX_NUM_OF_ATYPES,tempstr,mypars->dpffile,line_count);
										return 1;
									}
								}
								if(i-len<3){
									ltypes[ltype_nr-1][i-len] = line[i];
									ltypes[ltype_nr-1][i-len+1] = '\0'; // make extra sure we terminate the string
								} else{
									printf("\nError: Atom types are limited to 3 characters in <%s> parameter at %s:%u.\n",tempstr,mypars->dpffile,line_count);
									return 1;
								}
							}
						}
						mtype_nr=0;
						break;
				case DPF_MAP: // grid map specifier
						sscanf(line.c_str(),"%*s %255s",argstr);
						argstr[strlen(argstr)-4] = '\0'; // get rid of .map extension
						typestr=strchr(argstr+strlen(argstr)-4,'.')+1; // 4 chars for atom type
						if(mtype_nr>=ltype_nr){
							printf("\nError: More map files specified than atom types at %s:%u (ligand types need to be specified before maps).\n",mypars->dpffile,line_count);
							return 1;
						}
						if(strcmp(typestr,ltypes[mtype_nr])){ // derived type
							if(mypars->nr_deriv_atypes==0){ // get the derived atom types started
								mypars->deriv_atypes=(deriv_atype*)malloc(sizeof(deriv_atype));
								if(mypars->deriv_atypes==NULL){
									printf("Error: Cannot allocate memory for derivative type.\n");
									return 1;
								}
							}
							if(!add_deriv_atype(mypars,ltypes[mtype_nr],strlen(ltypes[mtype_nr]))){
								printf("Error: Derivative (ligand type %s) names can only be upto 3 characters long.\n",ltypes[mtype_nr]);
								return 1;
							}
							idx = mypars->nr_deriv_atypes-1;
							strcpy(mypars->deriv_atypes[idx].base_name,typestr);
#ifdef DERIVTYPE_INFO
							printf("%i: %s=%s\n",mypars->deriv_atypes[idx].nr,mypars->deriv_atypes[idx].deriv_name,mypars->deriv_atypes[idx].base_name);
#endif
						}
						mtype_nr++;
						break;
				case DPF_INTNBP_COEFFS: // internal pair energy coefficients
				case DPF_INTNBP_REQM_EPS: // internal pair energy coefficients
						if(sscanf(line.c_str(), "%*s %f %f %d %d %3s %3s", &paramA, &paramB, &m, &n, typeA, typeB)<6){
							printf("Error: Syntax error for <%s>, 6 values are required at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						if(m==n){
							printf("Error: Syntax error for <%s>, exponents need to be different at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						if(token_id==DPF_INTNBP_COEFFS){
							tempfloat = pow(paramB/paramA*n/m,m-n); // reqm
							paramA = paramB*float(m-n)/(pow(tempfloat,n)*m); // epsAB
							paramB = tempfloat; // rAB
						}
						// parameters are sorted out, now add to modpairs
						mypars->nr_mod_atype_pairs++;
						if(mypars->nr_mod_atype_pairs==1)
							mypars->mod_atype_pairs=(pair_mod*)malloc(sizeof(pair_mod));
						else
							mypars->mod_atype_pairs=(pair_mod*)realloc(mypars->mod_atype_pairs, mypars->nr_mod_atype_pairs*sizeof(pair_mod));
						if(mypars->mod_atype_pairs==NULL){
							printf("Error: Cannot allocate memory for <%s> pair energy modification.\n at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						curr_pair=&mypars->mod_atype_pairs[mypars->nr_mod_atype_pairs-1];
						strcpy(curr_pair->A,typeA);
						strcpy(curr_pair->B,typeB);
						curr_pair->nr_parameters=4;
						curr_pair->parameters=(float*)malloc(curr_pair->nr_parameters*sizeof(float));
						if(curr_pair->parameters==NULL){
							printf("Error: Cannot allocate memory for <%s> pair energy modification.\n at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						curr_pair->parameters[0]=paramA;
						curr_pair->parameters[1]=paramB;
						curr_pair->parameters[2]=m;
						curr_pair->parameters[3]=n;
#ifdef MODPAIR_INFO
						printf("%i: %s:%s",mypars->nr_mod_atype_pairs,curr_pair->A,curr_pair->B);
						for(idx=0; idx<curr_pair->nr_parameters; idx++)
							printf(",%f",curr_pair->parameters[idx]);
						printf("\n");
#endif
						break;
				case DPF_TRAN0: // translate                     (needs to be "random")
				case DPF_AXISANGLE0: // rotation axisangle       (needs to be "random")
				case DPF_QUATERNION0: // quaternion (of rotation, needs to be "random")
				case DPF_QUAT0: // quaternion       (of rotation, needs to be "random")
				case DPF_DIHE0: // number of dihedrals           (needs to be "random")
						sscanf(line.c_str(),"%*s %255s",argstr);
						if(stricmp(argstr,"random")){
							printf("\nError: Currently only \"random\" is supported as <%s> parameter at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case DPF_RUNS: // set number of runs
				case DPF_GALS: // actually run a search (only if xml2dlg isn't specified)
						if(!mypars->xml2dlg){
							sscanf(line.c_str(),"%*s %d",&tempint);
							if ((tempint >= 1) && (tempint <= MAX_NUM_OF_RUNS)){
								mypars->num_of_runs = (int) tempint;
							} else{
								printf("Error: Value of <%s> at %s:%u must be an integer between 1 and %d.\n",tempstr,mypars->dpffile,line_count,MAX_NUM_OF_RUNS);
								return 1;
							}
							if(token_id!=DPF_RUNS){
								// Add the fld file to use
								if (!mypars->fldfile){
									printf("\nError: No map file on record yet. Please specify a map file before the first ligand.\n");
									return 1;
								}
								filelist.fld_files.push_back(mypars->fldfile);
								mypars->list_nr++;
								// If more than one unique protein, cant do map preloading yet
								if (filelist.fld_files.size()>1){
									filelist.preload_maps=false;
								}
								// Add the ligand to filelist
								filelist.ligand_files.push_back(mypars->ligandfile);
								// Default resname is filelist basename
								if(mypars->resname) free(mypars->resname);
								len=strlen(mypars->ligandfile)-6; // .pdbqt = 6 chars
								if(len>0){
									mypars->resname = (char*)malloc((len+1)*sizeof(char));
									strncpy(mypars->resname,mypars->ligandfile,len); // Default is ligand file basename
									mypars->resname[len]='\0';
								} else mypars->resname = strdup("docking"); // Fallback to old default
								filelist.resnames.push_back(mypars->resname);
								if(new_device) mypars->devices_requested++;
								// Before pushing parameters and grids back make sure
								// the filename pointers are unique
								if(filelist.mypars.size()>0){ // mypars and mygrids have same size
									if((filelist.mypars.back().flexresfile) &&
									   (filelist.mypars.back().flexresfile==mypars->flexresfile))
										mypars->flexresfile=strdup(mypars->flexresfile);
									if((filelist.mypars.back().xrayligandfile) &&
									   (filelist.mypars.back().xrayligandfile==mypars->xrayligandfile))
										mypars->xrayligandfile=strdup(mypars->xrayligandfile);
									if((filelist.mygrids.back().grid_file_path) &&
									   (filelist.mygrids.back().grid_file_path==mygrid->grid_file_path))
										mygrid->grid_file_path=strdup(mygrid->grid_file_path);
									if((filelist.mygrids.back().receptor_name) &&
									   (filelist.mygrids.back().receptor_name==mygrid->receptor_name))
										mygrid->receptor_name=strdup(mygrid->receptor_name);
									if((filelist.mygrids.back().map_base_name) &&
									   (filelist.mygrids.back().map_base_name==mygrid->map_base_name))
										mygrid->map_base_name=strdup(mygrid->map_base_name);
								}
								// Add the parameter block now that resname is set
								filelist.mypars.push_back(*mypars);
								// Also add the grid
								filelist.mygrids.push_back(*mygrid);
							}
						} else{
							if(token_id!=DPF_RUNS) run_cnt++;
							if((mypars->list_nr>0) && (run_cnt>=mypars->list_nr)) return 0; // finished
						}
						break;
				case DPF_INTELEC: // calculate ES energy (needs not be "off")
						sscanf(line.c_str(),"%*s %255s",argstr);
						if(stricmp(argstr,"off")==0){
							printf("\nError: \"Off\" is not supported as <%s> parameter at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case DPF_SMOOTH: // smoothing range
						sscanf(line.c_str(),"%*s %f",&tempfloat);
						// smooth is measured in Angstrom
						if ((tempfloat >= 0.0f) && (tempfloat <= 0.5f)){
							mypars->smooth = tempfloat;
						} else{
							printf("Error: Value of <%s> at %s:%u must be a float between 0 and 0.5.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case DPF_SEED: // random number seed
						m=0; n=0; i=0;
						if(sscanf(line.c_str(),"%*s %d %d %d",&m, &n, &i)>0){ // one or more numbers
							mypars->seed[0]=m; mypars->seed[1]=n; mypars->seed[2]=i;
						} else // only warn here to not crash on unsupported values (we have a different RNG so if they'd be used they'd be useless to us anyway)
							printf("Warning: Only numerical values currently supported for <%s> at %s:%u.\n",tempstr,mypars->dpffile,line_count);
						break;
				case DPF_RMSTOL: // RMSD clustering tolerance
						sscanf(line.c_str(),"%*s %f",&tempfloat);
						if (tempfloat > 0.0){
							mypars->rmsd_tolerance = tempfloat;
						} else{
							printf("Error: Value of <%s> at %s:%u must be greater than 0.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case GA_pop_size: // population size
						sscanf(line.c_str(),"%*s %d",&tempint);
						if ((tempint >= 2) && (tempint <= MAX_POPSIZE)){
							mypars->pop_size = (unsigned long) (tempint);
						} else{
							printf("Error: Value of <%s> at %s:%u must be an integer between 2 and %d.\n",tempstr,mypars->dpffile,line_count,MAX_POPSIZE);
							return 1;
						}
						break;
				case GA_num_generations: // number of generations
						sscanf(line.c_str(),"%*s %d",&tempint);
						if ((tempint > 0) && (tempint < 16250000)){
							mypars->num_of_generations = (unsigned long) tempint;
						} else{
							printf("Error: Value of <%s> at %s:%u must be between 0 and 16250000.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case GA_num_evals: // number of evals
						sscanf(line.c_str(),"%*s %d",&tempint);
						if ((tempint > 0) && (tempint < 0x7FFFFFFF)){
							mypars->num_of_energy_evals = (unsigned long) tempint;
							mypars->nev_provided = true;
						} else{
							printf("Error: Value of <%s> at %s:%u must be between 0 and 2^31-1.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case GA_mutation_rate: // mutation rate
						sscanf(line.c_str(),"%*s %f",&tempfloat);
						tempfloat*=100.0;
						if ((tempfloat >= 0.0) && (tempfloat < 100.0)){
							mypars->mutation_rate = tempfloat;
						} else{
							printf("Error: Value of <%s> at %s:%u must be a float between 0 and 1.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case GA_crossover_rate: // crossover rate
						sscanf(line.c_str(),"%*s %f",&tempfloat);
						tempfloat*=100.0;
						if ((tempfloat >= 0.0) && (tempfloat <= 100.0)){
							mypars->crossover_rate = tempfloat;
						} else{
							printf("Error: Value of <%s> at %s:%u must be a float between 0 and 1.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case SW_max_its: // local search iterations
						sscanf(line.c_str(),"%*s %d",&tempint);
						if ((tempint > 0) && (tempint < 262144)){
							mypars->max_num_of_iters = (unsigned long) tempint;
						} else{
							printf("Error: Value of <%s> at %s:%u must be an integer between 1 and 262143.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case SW_max_succ: // cons. success limit
				case SW_max_fail: // cons. failure limit
						sscanf(line.c_str(),"%*s %d",&tempint);
						if ((tempint > 0) && (tempint < 256)){
							mypars->cons_limit = (unsigned long) (tempint);
						} else{
							printf("Error: Value of <%s> at %s:%u must be an integer between 1 and 255.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case SW_lb_rho: // lower bound of rho
						sscanf(line.c_str(),"%*s %f",&tempfloat);
						if ((tempfloat >= 0.0) && (tempfloat < 1.0)){
							mypars->rho_lower_bound = tempfloat;
						} else{
							printf("Error: Value of <%s> at %s:%u must be a float between 0 and 1.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case DPF_UNBOUND_MODEL: // unbound model (bound|extended|compact)
						sscanf(line.c_str(),"%*s %255s",argstr);
						if(stricmp(argstr,"bound")==0){
							mypars->unbound_model = 0;
							mypars->coeffs = unbound_models[mypars->unbound_model];
						} else if(stricmp(argstr,"extended")==0){
							mypars->unbound_model = 1;
							mypars->coeffs = unbound_models[mypars->unbound_model];
						} else if(stricmp(argstr,"compact")==0){
							mypars->unbound_model = 2;
							mypars->coeffs = unbound_models[mypars->unbound_model];
						} else{
							printf("Error: Unsupported value for <%s> at %s:%u. Value must be one of (bound|extend|compact).\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case DPF_COMMENT: // we use comments to allow specifying AD-GPU command lines
						sscanf(line.c_str(),"%*s %255s %255s",tempstr,argstr);
						if(tempstr[0]=='-'){ // potential command line argument
							i=2; // one command line argument to be parsed
							args[0]=tempstr;
							args[1]=argstr;
							if(get_commandpars(&i,args,&(mygrid->spacing),mypars,false)<2){
								printf("Warning: Command line option '%s' at %s:%u is not supported inside a dpf file.\n",tempstr,mypars->dpffile,line_count);
							}
							// count GPUs in case we set a different one
							if(argcmp("devnum",tempstr,'D')){
								new_device=false;
								for(i=0; (i<(int)filelist.mypars.size())&&!new_device; i++){
									if(mypars->devnum==filelist.mypars[i].devnum){
										new_device=true;
									}
								}
							}
						}
						break;
				case DPF_UNKNOWN: // error condition
						printf("\nError: Unknown or unsupported dpf token <%s> at %s:%u.\n",tempstr,mypars->dpffile,line_count);
						return 1;
				default: // means there's a keyword detected that's not yet implemented here
						printf("<%s> has not yet been implemented.\n",tempstr);
				case DPF_BLANK_LINE: // nothing to do here
				case DPF_NULL:
						break;
			}
		}
	}
	return 0;
}

int preparse_dpf(
                 const int*      argc,
                       char**    argv,
                       Dockpars* mypars,
                       Gridinfo* mygrid,
                       FileList& filelist
                )
// This function checks if a dpf file is used and, if runs are specified, map and ligand information
// is stored in the filelist; flexres information and which location in the dpf parameters are in each
// run is stored separately to allow logical parsing with the correct parameters initialized per run
{
	bool output_multiple_warning = true;
	std::vector<std::string> xml_files;
	bool read_more_xml_files = false;
	int error;
	for (int i=1; i<(*argc)-1+(read_more_xml_files); i++)
	{
		if (argcmp("help", argv[i], 'h')){
			print_options(argv[0]);
		}
		// wildcards for -xml2dlg are allowed (or multiple file names)
		// - if more than one xml file is specified this way, they will end up in xml_files
		// the test below is to stop reading arguments as filenames when another argument starts with "-"
		if (read_more_xml_files && (argv[i][0]=='-')){
			read_more_xml_files = false;
			if(i>=(*argc)-1) break; // ignore last argument if there is no parameter specified
		} else if (read_more_xml_files) xml_files.push_back(argv[i]); // copy argument into xml_files when read_more_xml_files is true
		
		// Argument: dpf file name.
		if (argcmp("import_dpf", argv[i], 'I')){
			if(mypars->dpffile){
				free(mypars->dpffile);
				if(output_multiple_warning){
					printf("Warning: Multiple --import_dpf (-I) arguments, only the last one will be used.");
					output_multiple_warning = false;
				}
			}
			mypars->dpffile = strdup(argv[i+1]);
		}
		
		// Argument: load initial data from xml file and reconstruct dlg, then finish
		if (argcmp("xml2dlg", argv [i], 'X'))
		{
			mypars->load_xml = strdup(argv[i+1]);
			read_more_xml_files = true;
			mypars->xml2dlg = true;
			mypars->xml_files = 1;
		}
		
		if (argcmp("contact_analysis", argv[i], 'C'))
		{
			float temp;
			error = sscanf(argv[i+1], "%f,%f,%f", &temp, &temp, &temp);
			if(error==1){
				sscanf(argv [i+1], "%d", &error);
				if (error == 0)
					mypars->contact_analysis = false;
				else
					mypars->contact_analysis = true;
			} else{
				if(error!=3){
					printf("\nError: Argument --contact_analysis (-C) expects either one parameter to enable/disable (i.e. --contact_analysis 1)\n"
					         "       or exactly three parameters to specify cutoffs (default: --contact_analysis %.1f,%.1f,%.1f)\n", mypars->R_cutoff, mypars->H_cutoff, mypars->V_cutoff);
					return 1;
				}
				sscanf(argv[i+1], "%f,%f,%f", &(mypars->R_cutoff), &(mypars->H_cutoff), &(mypars->V_cutoff));
				mypars->contact_analysis = true;
			}
		}
		
		// Argument: print dlg output to stdout instead of to a file
		if (argcmp("dlg2stdout", argv [i], '2'))
		{
			sscanf(argv [i+1], "%d", &error);
			if (error == 0)
				mypars->dlg2stdout = false;
			else
				mypars->dlg2stdout = true;
		}
	}
	
	bool specified_dpf = (mypars->dpffile!=NULL);
	if(specified_dpf){
		if((error=parse_dpf(mypars,mygrid,filelist))) return error;
	}
	
	if(xml_files.size()>0){ // use filelist parameter list in case multiple xml files are converted
		mypars->xml_files = xml_files.size();
		if(mypars->xml_files>100){ // output progress bar
			printf("Preparing ");
			if(mypars->contact_analysis)
				printf("analysis\n");
			else
				printf("conversion\n");
			printf("0%%      20%%       40%%       60%%       80%%     100%%\n");
			printf("---------+---------+---------+---------+---------+\n");
		}
		Dockpars orig_pars;
		if(!specified_dpf) orig_pars = *mypars;
		for(unsigned int i=0; i<mypars->xml_files; i++){
			if(mypars->xml_files>100){
				if((50*(i+1)) % mypars->xml_files < 50){
					printf("*"); fflush(stdout);
				}
			}
			// make sure to NOT free the previous ones as otherwise the strings of other mypars will be gone too ...
			char* prev_fld_file=mypars->fldfile;
			if(!specified_dpf){
				*mypars = orig_pars;
				mypars->dpffile=NULL;
			}
			mypars->fldfile=NULL;
			mypars->ligandfile=NULL;
			mypars->flexresfile=NULL;
			// load_xml is the xml file from which the other parameters will be set
			mypars->load_xml = strdup(xml_files[i].c_str());
			read_xml_filenames(mypars->load_xml,
			                   mypars->dpffile,
			                   mypars->fldfile,
			                   mypars->ligandfile,
			                   mypars->flexresfile,
			                   mypars->list_nr,
			                   mypars->seed);
			if(!specified_dpf){ // parse dpf file in XML file unless user specified one
				if((error=parse_dpf(mypars,mygrid,filelist))) return error;
			}
			mypars->pop_size=1;
			// Filling mygrid according to the specified fld file
			mygrid->info_read = false;
			if (get_gridinfo(mypars->fldfile, mygrid) != 0)
			{
				printf("\nError: get_gridinfo failed with fld file (%s) specified in %s.\n",mypars->fldfile,mypars->load_xml);
				return 1;
			}
			if(prev_fld_file){ // unfortunately, some strcmp implementation segfault with NULL as input
				if(strcmp(prev_fld_file,mypars->fldfile) != 0)
					filelist.fld_files.push_back(mypars->fldfile);
			} else filelist.fld_files.push_back(mypars->fldfile);

			// If more than one unique protein, cant do map preloading yet
			if (filelist.fld_files.size()>1)
				filelist.preload_maps=false;
			
			// Add the ligand filename in the xml to the filelist
			filelist.ligand_files.push_back(mypars->ligandfile);
			filelist.mypars.push_back(*mypars);
			filelist.mygrids.push_back(*mygrid);
		}
		if(mypars->xml_files>100) printf("\n\n");
		filelist.nfiles = mypars->xml_files;
	} else{
		filelist.nfiles = filelist.ligand_files.size();
	}
	if(filelist.nfiles>0){
		filelist.used = true;
		if(mypars->contact_analysis && filelist.preload_maps){
			std::string receptor_name=mygrid->grid_file_path;
			if(strlen(mygrid->grid_file_path)>0) receptor_name+="/";
			receptor_name += mygrid->receptor_name;
			receptor_name += ".pdbqt";
			mypars->receptor_atoms = read_receptor(receptor_name.c_str(),mygrid,mypars->receptor_map,mypars->receptor_map_list);
			mypars->nr_receptor_atoms = mypars->receptor_atoms.size();
		}
	}
	return 0;
}

int get_filelist(
                 const int*      argc,
                       char**    argv,
                       Dockpars* mypars,
                       Gridinfo* mygrid,
                       FileList& filelist
                )
// The function checks if a filelist has been provided according to the proper command line arguments.
// If it is, it loads the .fld, .pdbqt, and resname files into vectors
{
	if(mypars->xml2dlg){ // no file list for -xml2dlg (wildcards are allowed in argument)
		filelist.preload_maps&=filelist.used;
		return 0;
	}
	bool output_multiple_warning = true;
	for (int i=1; i<(*argc)-1; i++)
	{
		// Argument: file name that contains list of files.
		if (argcmp("filelist", argv[i], 'B'))
		{
			filelist.used = true;
			if(filelist.filename){
				free(filelist.filename);
				if(output_multiple_warning){
					printf("Warning: Multiple --filelist (-B) arguments, only the last one will be used.");
					output_multiple_warning = false;
				}
			}
			filelist.filename = strdup(argv[i+1]);
		}
	}

	if (filelist.filename){ // true when -filelist specifies a filename
	                        // filelist.used may be true when dpf file is specified as it uses the filelist to store runs
		std::ifstream file(filelist.filename);
		if(file.fail()){
			printf("\nError: Could not open filelist %s. Check path and permissions.\n",filelist.filename);
			return 1;
		}
		std::string line;
		bool prev_line_was_fld=false;
		unsigned int initial_res_count = filelist.resnames.size();
		int len;
		int line_count=0;
		while(std::getline(file, line)) {
			line_count++;
			trim(line); // Remove leading and trailing whitespace
			len = line.size();
			if(len>filelist.max_len) filelist.max_len = len;
			if (len>=4 && line.compare(len-4,4,".fld") == 0){
				if (prev_line_was_fld){ // Overwrite the previous fld file if two in a row
					filelist.fld_files[filelist.fld_files.size()-1] = line;
					printf("\nWarning: using second listed .fld file in line %d\n",line_count);
				} else {
					// Add the .fld file
					filelist.fld_files.push_back(line);
					prev_line_was_fld=true;

					// If more than one unique protein, cant do map preloading yet
					if (filelist.fld_files.size()>1){
						filelist.preload_maps=false;
					}
				}
				// Filling mygrid according to the specified fld file
				mygrid->info_read = false;
				if (get_gridinfo(filelist.fld_files[filelist.fld_files.size()-1].c_str(), mygrid) != 0)
				{
					printf("\nError: get_gridinfo failed with fld file specified in filelist.\n");
					return 1;
				}
			} else if (len>=6 && line.compare(len-6,6,".pdbqt") == 0){
				// Add the .pdbqt
				filelist.ligand_files.push_back(line);
				mypars->list_nr++;
				// Before pushing parameters and grids back make sure
				// the filename pointers are unique
				if(filelist.mypars.size()>0){ // mypars and mygrids have same size
					if((filelist.mypars.back().flexresfile) &&
					   (filelist.mypars.back().flexresfile==mypars->flexresfile))
						mypars->flexresfile=strdup(mypars->flexresfile);
					if((filelist.mypars.back().xrayligandfile) &&
					   (filelist.mypars.back().xrayligandfile==mypars->xrayligandfile))
						mypars->xrayligandfile=strdup(mypars->xrayligandfile);
					if((filelist.mygrids.back().grid_file_path) &&
					   (filelist.mygrids.back().grid_file_path==mygrid->grid_file_path))
						mygrid->grid_file_path=strdup(mygrid->grid_file_path);
					if((filelist.mygrids.back().receptor_name) &&
					   (filelist.mygrids.back().receptor_name==mygrid->receptor_name))
						mygrid->receptor_name=strdup(mygrid->receptor_name);
					if((filelist.mygrids.back().map_base_name) &&
					   (filelist.mygrids.back().map_base_name==mygrid->map_base_name))
						mygrid->map_base_name=strdup(mygrid->map_base_name);
				}
				// Add the parameter block
				filelist.mypars.push_back(*mypars);
				// Add the grid info
				filelist.mygrids.push_back(*mygrid);
				if (filelist.fld_files.size()==0){
					if(mygrid->info_read){ // already read a map file in with dpf import
						printf("\nUsing map file from dpf import.\n");
						filelist.fld_files.push_back(mypars->fldfile);
					} else{
						printf("\nError: No map file on record yet. Please specify a .fld file before the first ligand (%s).\n",line.c_str());
						return 1;
					}
				}
				if (filelist.ligand_files.size()>filelist.fld_files.size()){
					// If this ligand doesnt have a protein preceding it, use the previous protein
					filelist.fld_files.push_back(filelist.fld_files[filelist.fld_files.size()-1]);
				}
				prev_line_was_fld=false;
			} else if (len>0) {
				// Anything else in the file is assumed to be the resname
				filelist.resnames.push_back(line);
			}
		}

		filelist.nfiles = filelist.ligand_files.size();

		if (filelist.ligand_files.size()==0){
			printf("\nError: No ligands, through lines ending with the .pdbqt suffix, have been specified.\n");
			return 1;
		}
		if (filelist.ligand_files.size() != filelist.resnames.size()){
			if(filelist.resnames.size()-initial_res_count>0){ // make sure correct number of resnames were specified when they were specified
				printf("\nError: Inconsistent number of resnames (%lu) compared to ligands (%lu)!\n",filelist.resnames.size(),filelist.ligand_files.size());
			} else{ // otherwise add default resname (ligand basename)
				for(unsigned int i=filelist.resnames.size(); i<filelist.ligand_files.size(); i++)
					filelist.resnames.push_back(filelist.ligand_files[i].substr(0,filelist.ligand_files[i].size()-6));
			}
			return 1;
		}
		for(unsigned int i=initial_res_count; i<filelist.ligand_files.size(); i++){
			if(filelist.mypars[i].fldfile) free(filelist.mypars[i].fldfile);
			filelist.mypars[i].fldfile = strdup(filelist.fld_files[i].c_str());
			if(filelist.mypars[i].ligandfile) free(filelist.mypars[i].ligandfile);
			filelist.mypars[i].ligandfile = strdup(filelist.ligand_files[i].c_str());
			if(filelist.mypars[i].resname) free(filelist.mypars[i].resname);
			filelist.mypars[i].resname = strdup(filelist.resnames[i].c_str());
		}
	}
	filelist.preload_maps&=filelist.used;

	return 0;
}

int get_filenames_and_ADcoeffs(
                               const int*      argc,
                                     char**    argv,
                                     Dockpars* mypars,
                               const bool      multiple_files
                              )
// The function fills the file name and coeffs fields of mypars parameter
// according to the proper command line arguments.
{
	int i;
	int ffile_given, lfile_given;
	long tempint;
	
	ffile_given = (mypars->fldfile!=NULL);
	lfile_given = (mypars->ligandfile!=NULL);
	
	for (i=1; i<(*argc)-1; i++)
	{
		if (!multiple_files){
			// Argument: grid parameter file name.
			if (argcmp("ffile", argv[i], 'M'))
			{
				ffile_given = 1;
				mypars->fldfile = strdup(argv[i+1]);
			}

			// Argument: ligand pdbqt file name
			if (argcmp("lfile", argv[i], 'L'))
			{
				lfile_given = 1;
				mypars->ligandfile = strdup(argv[i+1]);
			}
		}

		// Argument: flexible residue pdbqt file name
		if (argcmp("flexres", argv[i], 'F'))
		{
			mypars->flexresfile = strdup(argv[i+1]);
		}

		// Argument: unbound model to be used.
		// 0 means the bound, 1 means the extended, 2 means the compact ...
		// model's free energy coefficients will be used during docking.
		if (argcmp("ubmod", argv[i], 'u'))
		{
			sscanf(argv[i+1], "%ld", &tempint);
			switch(tempint){
				case 0:
					mypars->unbound_model = 0;
					mypars->coeffs = unbound_models[mypars->unbound_model];
					break;
				case 1:
					mypars->unbound_model = 1;
					mypars->coeffs = unbound_models[mypars->unbound_model];
					break;
				case 2:
					mypars->unbound_model = 2;
					mypars->coeffs = unbound_models[mypars->unbound_model];
					break;
				default:
					printf("Error: Value of --ubmod (-u) argument can only be 0 (unbound same as bound), 1 (extended), or 2 (compact).\n");
					return 1;
			}
		}
	}

	if (ffile_given == 0 && !multiple_files)
	{
		printf("Error: grid fld file was not defined. Use --ffile (-M) argument!\n");
		print_options(argv[0]);
		return 1; // we'll never get here - but we might in the future again ...
	}

	if (lfile_given == 0 && !multiple_files)
	{
		printf("Error: ligand pdbqt file was not defined. Use --lfile (-L) argument!\n");
		print_options(argv[0]);
		return 1; // we'll never get here - but we might in the future again ...
	}

	return 0;
}

void print_options(
                   const char* program_name
                  )
{
	printf("Command line options:\n\n");
	printf(" Argument              | Description                                           | Default value\n");
	printf("-----------------------|-------------------------------------------------------|------------------\n");
	printf("--lfile             -L | Ligand pdbqt file                                     | no default\n");
	printf("--ffile             -M | Grid map files descriptor fld file                    | no default\n");
	printf("--flexres           -F | Flexible residue pdbqt file                           | no default\n");
	printf("--filelist          -B | Batch file                                            | no default\n");
	printf("--import_dpf        -I | Import AD4-type dpf input file (only partial support) | no default\n");
	printf("--resnam            -N | Name for docking output log                           | ligand basename\n");
	printf("--xraylfile         -R | reference ligand file for RMSD analysis               | ligand file\n");
	printf("--devnum            -D | OpenCL/Cuda device number (counting starts at 1)      | 1\n");
	printf("--derivtype         -T | Derivative atom types (e.g. C1,C2,C3=C/S4=S/H5=HD)    | no default\n");
	printf("--modpair           -P | Modify vdW pair params (e.g. C1:S4,1.60,1.200,13,7)   | no default\n");
	printf("--heuristics        -H | Ligand-based automatic search method and # evals      | 1 (yes)\n");
	printf("--heurmax           -E | Asymptotic heuristics # evals limit (smooth limit)    | 12000000\n");
	printf("--autostop          -A | Automatic stopping criterion based on convergence     | 1 (yes)\n");
	printf("--asfreq            -a | AutoStop testing frequency (in # of generations)      | 5\n");
	printf("--contact_analysis  -C | Perform distance-based analysis (description below)   | 0 (no)\n");
	printf("--xml2dlg           -X | One (or many) AD-GPU xml file(s) to convert to dlg(s) | no default\n");
	printf("--xmloutput         -x | Specify if xml output format is wanted                | 1 (yes)\n");
	printf("--loadxml           -c | Load initial population from xml results file         | no default\n");
	printf("--dlgoutput         -d | Control if dlg output is created                      | 1 (yes)\n");
	printf("--dlg2stdout        -2 | Write dlg file output to stdout (if not OVERLAP=ON)   | 0 (no)\n");
	printf("--seed              -s | Random number seeds (up to three comma-sep. integers) | time, process id\n");
	printf("--ubmod             -u | Unbound model: 0 (bound), 1 (extended), 2 (compact)   | 0 (same as bound)\n");
	printf("--nrun              -n | # LGA runs                                            | 20\n");
	printf("--nev               -e | # Score evaluations (max.) per LGA run                | 2500000\n");
	printf("--ngen              -g | # Generations (max.) per LGA run                      | 42000\n");
	printf("--lsmet             -l | Local-search method                                   | ad (ADADELTA)\n");
	printf("--lsit              -i | # Local-search iterations (max.)                      | 300\n");
	printf("--psize             -p | Population size                                       | 150\n");
	printf("--mrat                 | Mutation rate                                         | 2   (%%)\n");
	printf("--crat                 | Crossover rate                                        | 80  (%%)\n");
	printf("--lsrat                | Local-search rate                                     | 100 (%%)\n");
	printf("--trat                 | Tournament (selection) rate                           | 60  (%%)\n");
	printf("--rlige                | Print reference ligand energies                       | 0 (no)\n");
	printf("--hsym                 | Handle symmetry in RMSD calc.                         | 1 (yes)\n");
	printf("--rmstol               | RMSD clustering tolerance                             | 2 (Å)\n");
	printf("--dmov                 | Maximum LGA movement delta                            | 6 (Å)\n");
	printf("--dang                 | Maximum LGA angle delta                               | 90 (°)\n");
	printf("--rholb                | Solis-Wets lower bound of rho parameter               | 0.01\n");
	printf("--lsmov                | Solis-Wets movement delta                             | 2 (Å)\n");
	printf("--lsang                | Solis-Wets angle delta                                | 75 (°)\n");
	printf("--cslim                | Solis-Wets cons. success/failure limit to adjust rho  | 4\n");
	printf("--smooth               | Smoothing parameter for vdW interactions              | 0.5 (Å)\n");
	printf("--elecmindist          | Min. electrostatic potential distance (w/ dpf: 0.5 Å) | 0.01 (Å)\n");
	printf("--modqp                | Use modified QASP from VirtualDrug or AD4 original    | 0 (no, use AD4)\n");
	printf("--cgmaps               | Use individual maps for CG-G0 instead of the same one | 0 (no, same map)\n");
	printf("--stopstd              | AutoStop energy standard deviation tolerance          | 0.15 (kcal/mol)\n");
	printf("--initswgens           | Initial # generations of Solis-Wets instead of -lsmet | 0 (no)\n");
	printf("--gfpop                | Output all poses from all populations of each LGA run | 0 (no)\n");
	printf("--npdb                 | # pose pdbqt files from populations of each LGA run   | 0\n");
	printf("--gbest                | Output single best pose as pdbqt file                 | 0 (no)\n");

	printf("\nAutodock-GPU requires a ligand and a set of grid maps as well as optionally a flexible residue to\n");
	printf("perform a docking calculation. These could be specified directly (--lfile, --ffile, and --flexres),\n");
	printf("as part of a filelist text file (see README.md for format), or as an AD4-style dpf.\n");

	printf("\nExamples:\n");
	printf("   * Dock ligand.pdbqt to receptor.maps.fld using 50 LGA runs:\n");
	printf("        %s --lfile ligand.pdbqt --ffile receptor.maps.fld --nrun 50\n",program_name);
	printf("   * Convert ligand.xml to dlg, perform contact analysis, and output dlg to stdout:\n");
	printf("        %s --xml2dlg ligand.xml --contact_analysis 1 --dlg2stdout 1\n",program_name);
	printf("   * Dock ligands and map specified in file.lst with flexres flex.pdbqt:\n");
	printf("        %s --filelist file.lst --flexres flex.pdbqt\n",program_name);
	printf("   * Dock ligands, map, and (optional) flexres specified in docking.dpf on device #2:\n");
	printf("        %s --import_dpf docking.dpf --devnum 2\n\n",program_name);
	
	exit(0);
}

bool argcmp(
            const char* arg,
            const char* cmd,
            const char  shortarg
           )
{
	int length=strlen(cmd);
	int offset=1;
	if(length>1){
		if(cmd[0]!='-') return false;
		if(cmd[1]=='-') offset++;
		if(length-offset<1) return false;
		if((length-offset==1) && (shortarg!='\0')){ // short argument
			return (cmd[offset]==shortarg);
		}
		return (strcmp(arg,cmd+offset)==0);
	} else return false;
}

int get_commandpars(
                    const int*      argc,
                          char**    argv,
                          double*   spacing,
                          Dockpars* mypars,
                    const bool      late_call
                   )
// The function processes the command line arguments given with the argc and argv parameters,
// and fills the proper fields of mypars according to that. If a parameter was not defined
// in the command line, the default value will be assigned. The mypars' fields will contain
// the data in the same format as it is required for writing it to algorithm defined registers.
{
	int   i;
	int   tempint;
	float tempfloat;
	int   arg_recognized = 0;
	int   arg_set = 1;
	if(late_call){
		// ------------------------------------------
		// default values
		mypars->abs_max_dmov        = 6.0/(*spacing);             // +/-6A
		mypars->base_dmov_mul_sqrt3 = 2.0/(*spacing)*sqrt(3.0);   // 2 A
		mypars->xrayligandfile      = strdup(mypars->ligandfile); // By default xray-ligand file is the same as the randomized input ligand
		if(mypars->xml2dlg){
			if(strlen(mypars->load_xml)>4){ // .xml = 4 chars
				i=strlen(mypars->load_xml)-4;
				mypars->resname = (char*)malloc((i+1)*sizeof(char));
				strncpy(mypars->resname,mypars->load_xml,i);    // Default is ligand file basename
				mypars->resname[i]='\0';
			} else if(!mypars->resname) mypars->resname = strdup("docking"); // Fallback to old default
		} else{
			if(!mypars->resname){ // only need to set if it's not set yet
				if(strlen(mypars->ligandfile)>6){ // .pdbqt = 6 chars
					i=strlen(mypars->ligandfile)-6;
					mypars->resname = (char*)malloc((i+1)*sizeof(char));
					strncpy(mypars->resname,mypars->ligandfile,i);    // Default is ligand file basename
					mypars->resname[i]='\0';
				} else mypars->resname = strdup("docking");               // Fallback to old default
			}
		}
		// ------------------------------------------
	}

	// overwriting values which were defined as a command line argument
	for (i=1; i<(*argc)-1; i+=2)
	{
		arg_recognized = 0;

		// Argument: number of energy evaluations. Must be a positive integer.
		if (argcmp("nev", argv[i], 'e'))
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%d", &tempint);

			if ((tempint > 0) && (tempint < 0x7FFFFFFF)){
				mypars->num_of_energy_evals = (unsigned long) tempint;
				mypars->nev_provided = true;
			} else{
				printf("Error: Value of --nev (-e) argument must be between 0 and 2^31-1.\n");
				return -1;
			}
		}

		if (argcmp("seed", argv[i], 's'))
		{
			arg_recognized = 1;
			mypars->seed[0] = 0; mypars->seed[1] = 0; mypars->seed[2] = 0;
			tempint = sscanf(argv[i+1], "%u,%u,%u", &(mypars->seed[0]), &(mypars->seed[1]), &(mypars->seed[2]));
		}

		// Argument: number of generations. Must be a positive integer.
		if (argcmp("ngen", argv[i], 'g'))
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%d", &tempint);

			if ((tempint > 0) && (tempint < 16250000)){
				mypars->num_of_generations = (unsigned long) tempint;
			} else{
				printf("Error: Value of --ngen (-g) argument must be between 0 and 16250000.\n");
				return -1;
			}
		}

		// Argument: initial sw number of generations. Must be a positive integer.
		if (argcmp("initswgens", argv[i]))
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%d", &tempint);

			if ((tempint >= 0) && (tempint <= 16250000)){
				mypars->initial_sw_generations = (unsigned long) tempint;
			} else{
				printf("Error: Value of --initswgens argument must be between 0 and 16250000.\n");
				return -1;
			}
		}

		// ----------------------------------
		// Argument: Use Heuristics for number of evaluations (can be overwritten with -nev)
		if (argcmp("heuristics", argv [i], 'H'))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->use_heuristics = false;
			else
				mypars->use_heuristics = true;
		}
		// ----------------------------------

		// Argument: Upper limit for heuristics that's reached asymptotically
		if (argcmp("heurmax", argv[i], 'E'))
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%d", &tempint);

			if ((tempint > 0) && (tempint <= 1625000000)){
				mypars->heuristics_max = (unsigned long) tempint;
			} else{
				printf("Error: Value of --heurmax (-E) argument must be between 1 and 1625000000.\n");
				return -1;
			}
		}

		// Argument: maximal delta movement during mutation. Must be an integer between 1 and 16.
		// N means that the maximal delta movement will be +/- 2^(N-10)*grid spacing Angström.
		if (argcmp("dmov", argv[i]))
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%f", &tempfloat);

			if ((tempfloat > 0) && (tempfloat < 10)){
				mypars->abs_max_dmov = tempfloat/(*spacing);
			} else{
				printf("Error: Value of --dmov argument must be a float between 0 and 10.\n");
				return -1;
			}
		}

		// Argument: maximal delta angle during mutation. Must be an integer between 1 and 17.
		// N means that the maximal delta angle will be +/- 2^(N-8)*180/512 degrees.
		if (argcmp("dang", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat > 0) && (tempfloat < 180)){
				mypars->abs_max_dang = tempfloat;
			} else{
				printf("Error: Value of --dang argument must be a float between 0 and 180.\n");
				return -1;
			}
		}

		// Argument: mutation rate. Must be a float between 0 and 100.
		// Means the rate of mutations (cca) in percent.
		if (argcmp("mrat", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.0) && (tempfloat < 100.0)){
				mypars->mutation_rate = tempfloat;
			} else{
				printf("Error: Value of --mrat argument must be a float between 0 and 100.\n");
				return -1;
			}
		}

		// Argument: crossover rate. Must be a float between 0 and 100.
		// Means the rate of crossovers (cca) in percent.
		if (argcmp("crat", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.0) && (tempfloat <= 100.0)){
				mypars->crossover_rate = tempfloat;
			} else{
				printf("Error: Value of --crat argument must be a float between 0 and 100.\n");
				return -1;
			}
		}

		// Argument: local search rate. Must be a float between 0 and 100.
		// Means the rate of local search (cca) in percent.
		if (argcmp("lsrat", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			/*
			if ((tempfloat >= 0.0) && (tempfloat < 100.0))
			*/
			if ((tempfloat >= 0.0) && (tempfloat <= 100.0)){
				mypars->lsearch_rate = tempfloat;
			} else{
				printf("Error: Value of --lrat argument must be a float between 0 and 100.\n");
				return -1;
			}
		}

		// Smoothed pairwise potentials
		if (argcmp("smooth", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			// smooth is measured in Angstrom
			if ((tempfloat >= 0.0f) && (tempfloat <= 0.5f)){
				mypars->smooth = tempfloat;
			} else{
				printf("Error: Value of --smooth argument must be a float between 0 and 0.5.\n");
				return -1;
			}
		}

		// Argument: local search method:
		// "sw": Solis-Wets
		// "sd": Steepest-Descent
		// "fire": FIRE
		// "ad": ADADELTA
		// "adam": ADAM
		if (argcmp("lsmet", argv [i], 'l'))
		{
			arg_recognized = 1;

			char* temp = strdup(argv [i+1]);

			if (strcmp(temp, "sw") == 0) {
				strcpy(mypars->ls_method, temp);
				//mypars->max_num_of_iters = 300;
			}
			else if (strcmp(temp, "sd") == 0) {
				strcpy(mypars->ls_method, temp);
				//mypars->max_num_of_iters = 30;
			}
			else if (strcmp(temp, "fire") == 0) {
				strcpy(mypars->ls_method, temp);
				//mypars->max_num_of_iters = 30;
			}
			else if (strcmp(temp, "ad") == 0) {
				strcpy(mypars->ls_method, temp);
				//mypars->max_num_of_iters = 30;
			}
			else if (strcmp(temp, "adam") == 0) {
				strcpy(mypars->ls_method, temp);
				//mypars->max_num_of_iters = 30;
			}
			else {
				printf("Error: Value of --lsmet must be a valid string: \"sw\", \"sd\", \"fire\", \"ad\", or \"adam\".\n");
				return -1;
			}
			
			free(temp);
		}

		// Argument: tournament rate. Must be a float between 50 and 100.
		// Means the probability that the better entity wins the tournament round during selectin
		if (argcmp("trat", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= /*5*/0.0) && (tempfloat <= 100.0)){
				mypars->tournament_rate = tempfloat;
			} else{
				printf("Error: Value of --trat argument must be a float between 0 and 100.\n");
				return -1;
			}
		}


		// Argument: rho lower bound. Must be a float between 0 and 1.
		// Means the lower bound of the rho parameter (possible stop condition for local search).
		if (argcmp("rholb", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.0) && (tempfloat < 1.0)){
				mypars->rho_lower_bound = tempfloat;
			} else{
				printf("Error: Value of --rholb argument must be a float between 0 and 1.\n");
				return -1;
			}
		}

		// Argument: local search delta movement. Must be a float between 0 and grid spacing*64 A.
		// Means the spread of unifily distributed delta movement of local search.
		if (argcmp("lsmov", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat > 0.0) && (tempfloat < (*spacing)*64/sqrt(3.0))){
				mypars->base_dmov_mul_sqrt3 = tempfloat/(*spacing)*sqrt(3.0);
			} else{
				printf("Error: Value of --lsmov argument must be a float between 0 and %lf.\n", 64*(*spacing));
				return -1;
			}
		}

		// Argument: local search delta angle. Must be a float between 0 and 103°.
		// Means the spread of unifily distributed delta angle of local search.
		if (argcmp("lsang", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat > 0.0) && (tempfloat < 103.0)){
				mypars->base_dang_mul_sqrt3 = tempfloat*sqrt(3.0);
			} else{
				printf("Error: Value of --lsang argument must be a float between 0 and 103.\n");
				return -1;
			}
		}

		// Argument: consecutive success/failure limit. Must be an integer between 1 and 255.
		// Means the number of consecutive successes/failures after which value of rho have to be doubled/halved.
		if (argcmp("cslim", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if ((tempint > 0) && (tempint < 256)){
				mypars->cons_limit = (unsigned long) (tempint);
			} else{
				printf("Error: Value of --cslim argument must be an integer between 1 and 255.\n");
				return -1;
			}
		}

		// Argument: maximal number of iterations for local search. Must be an integer between 1 and 262143.
		// Means the number of iterations after which the local search algorithm has to terminate.
		if (argcmp("lsit", argv [i], 'i'))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if ((tempint > 0) && (tempint < 262144)){
				mypars->max_num_of_iters = (unsigned long) tempint;
			} else{
				printf("Error: Value of --lsit (-i) argument must be an integer between 1 and 262143.\n");
				return -1;
			}
		}

		// Argument: size of population. Must be an integer between 32 and CPU_MAX_POP_SIZE.
		// Means the size of the population in the genetic algorithm.
		if (argcmp("psize", argv [i], 'p'))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if ((tempint >= 2) && (tempint <= MAX_POPSIZE)){
				mypars->pop_size = (unsigned long) (tempint);
			} else{
				printf("Error: Value of --psize (-p) argument must be an integer between 2 and %d.\n", MAX_POPSIZE);
				return -1;
			}
		}

		// Argument: load initial population from xml file instead of generating one.
		if (argcmp("loadxml", argv [i], 'c'))
		{
			arg_recognized = 1;
			mypars->load_xml = strdup(argv[i+1]);
		}

		// Argument: load initial data from xml file and reconstruct dlg, then finish
		if (argcmp("xml2dlg", argv [i], 'X'))
		{
			arg_recognized = 1;
			i += mypars->xml_files-1; // skip ahead
		}

		// Argument: wether to perform a distance-based pose contact analysis or not
		if (argcmp("contact_analysis", argv [i], 'C'))
		{
			arg_recognized = 1;
		}

		// Argument: print dlg output to stdout instead of to a file
		if (argcmp("dlg2stdout", argv [i], '2'))
		{
			arg_recognized = 1;
		}

		// Argument: number of pdb files to be generated.
		// The files will include the best docking poses from the final population.
		if (argcmp("npdb", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if ((tempint < 0) || (tempint > MAX_POPSIZE)){
				printf("Error: Value of --npdb argument must be an integer between 0 and %d.\n", MAX_POPSIZE);
				return -1;
			} else mypars->gen_pdbs = tempint;
		}

		// ---------------------------------
		// UPDATED in : get_filelist()
		// ---------------------------------
		// Argument: name of file containing file list
		if (argcmp("filelist", argv [i], 'B'))
			arg_recognized = 1;

		// ---------------------------------
		// UPDATED in : preparse_dpf()
		// ---------------------------------
		// Argument: name of file containing file list
		if (argcmp("import_dpf", argv [i], 'I'))
			arg_recognized = 1;

		// ---------------------------------
		// MISSING: char* fldfile
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		// Argument: name of grid parameter file.
		if (argcmp("ffile", argv [i], 'M')){
			arg_recognized = 1;
			arg_set = 0;
		}

		// ---------------------------------
		// MISSING: char* ligandfile
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		// Argument: name of ligand pdbqt file
		if (argcmp("lfile", argv [i], 'L')){
			arg_recognized = 1;
			arg_set = 0;
		}

		// ---------------------------------
		// MISSING: char* flexresfile
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		// Argument: name of ligand pdbqt file
		if (argcmp("flexres", argv [i], 'F')){
			arg_recognized = 1;
			arg_set = 0;
		}

		// Argument: derivate atom types
		// - has already been tested for in
		//   main.cpp, as it's needed at grid
		//   creation time not after (now)
		if (argcmp("derivtype", argv [i], 'T'))
		{
			arg_recognized = 1;
		}

		// Argument: modify pairwise atom type parameters (LJ only at this point)
		// - has already been tested for in
		//   main.cpp, as it's needed at grid
		//   creation time not after (now)
		if (argcmp("modpair", argv [i], 'P'))
		{
			arg_recognized = 1;
		}

		// ---------------------------------
		// MISSING: devnum
		// UPDATED in : main
		// ----------------------------------
		// Argument: OpenCL/Cuda device number to use
		if (argcmp("devnum", argv [i], 'D'))
		{
			arg_recognized = 1;
			arg_set = 0;
			if(!late_call){
				arg_set = 1;
				sscanf(argv [i+1], "%d", &tempint);
				if ((tempint >= 1) && (tempint <= 65536)){
					mypars->devnum = (unsigned long) tempint-1;
				} else{
					printf("Error: Value of --devnum (-D) argument must be an integer between 1 and 65536.\n");
					return -1;
				}
			}
		}
		// ----------------------------------

		// ----------------------------------
		// Argument: Multiple CG-G0 maps or not
		// - has already been tested for in
		//   main.cpp, as it's needed at grid
		//   creation time not after (now)
		if (argcmp("cgmaps", argv [i]))
		{
			arg_recognized = 1; // stub to not complain about an unknown parameter
		}
		// ----------------------------------

		// ----------------------------------
		// Argument: Automatic stopping criterion (1) or not (0)
		if (argcmp("autostop", argv [i], 'A'))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->autostop = 0;
			else
				mypars->autostop = 1;
		}
		// ----------------------------------

		// ----------------------------------
		// Argument: Test frequency for auto-stopping criterion
		if (argcmp("asfreq", argv [i], 'a'))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);
			if ((tempint >= 1) && (tempint <= 100)){
				mypars->as_frequency = (unsigned int) tempint;
			} else{
				printf("Error: Value of --asfreq (-a) argument must be an integer between 1 and 100.\n");
				return -1;
			}
		}
		// ----------------------------------

		// ----------------------------------
		// Argument: Stopping criterion standard deviation.. Must be a float between 0.01 and 2.0;
		// Means the energy standard deviation of the best candidates after which to stop evaluation when autostop is 1..
		if (argcmp("stopstd", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.01) && (tempfloat < 2.0)){
				mypars->stopstd = tempfloat;
			} else{
				printf("Error: Value of --stopstd argument must be a float between 0.01 and 2.0.\n");
				return -1;
			}
		}
		// ----------------------------------

		// ----------------------------------
		// Argument: Minimum electrostatic pair potential distance .. Must be a float between 0.0 and 2.0;
		// This will cut the electrostatics interaction to the value at that distance below it. (default: 0.01)
		if (argcmp("elecmindist", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.0) && (tempfloat < 2.0)){
				mypars->elec_min_distance = tempfloat;
			} else{
				printf("Error: Value of --elecmindist argument must be a float between 0.0 and 2.0.\n");
				return -1;
			}
		}
		// ----------------------------------

		// Argument: number of runs. Must be an integer between 1 and 1000.
		// Means the number of required runs
		if (argcmp("nrun", argv [i], 'n'))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if ((tempint >= 1) && (tempint <= MAX_NUM_OF_RUNS)){
				mypars->num_of_runs = (int) tempint;
			} else{
				printf("Error: Value of --nrun (-n) argument must be an integer between 1 and %d.\n", MAX_NUM_OF_RUNS);
				return -1;
			}
		}

		// Argument: energies of reference ligand required.
		// If the value is not zero, energy values of the reference ligand is required.
		if (argcmp("rlige", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->reflig_en_required = false;
			else
				mypars->reflig_en_required = true;
		}

		// ---------------------------------
		// MISSING: char unbound_model
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		// Argument: unbound model to be used.
		if (argcmp("ubmod", argv [i], 'u')){
			arg_recognized = 1;
			arg_set = 0;
		}

		// Argument: handle molecular symmetry during rmsd calculation
		// If the value is not zero, molecular syymetry will be taken into account during rmsd calculation and clustering.
		if (argcmp("hsym", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->handle_symmetry = false;
			else
				mypars->handle_symmetry = true;
		}

		// Argument: generate final population result files.
		// If the value is zero, result files containing the final populations won't be generated, otherwise they will.
		if (argcmp("gfpop", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->gen_finalpop = false;
			else
				mypars->gen_finalpop = true;
		}

		// Argument: generate best.pdbqt
		// If the value is zero, best.pdbqt file containing the coordinates of the best result found during all of the runs won't be generated, otherwise it will
		if (argcmp("gbest", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->gen_best = false;
			else
				mypars->gen_best = true;
		}

		// Argument: name of result files.
		if (argcmp("resnam", argv [i], 'N'))
		{
			arg_recognized = 1;
			free(mypars->resname); // as we assign a default value dynamically created to it
			mypars->resname = strdup(argv [i+1]);
		}

		// Argument: use modified QASP (from VirtualDrug) instead of original one used by AutoDock
		// If the value is not zero, the modified parameter will be used.
		if (argcmp("modqp", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->qasp = 0.01097f; // original AutoDock QASP parameter
			else
				mypars->qasp = 0.00679f; // from VirtualDrug
		}

		// Argument: rmsd tolerance for clustering.
		// This will be used during clustering for the tolerance distance.
		if (argcmp("rmstol", argv [i]))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if (tempfloat > 0.0){
				mypars->rmsd_tolerance = tempfloat;
			} else{
				printf("Error: Value of --rmstol argument must be a double greater than 0.\n");
				return -1;
			}
		}

		// Argument: choose wether to output DLG or not
		// If the value is 1, DLG output will be generated
		// DLG output won't be generated if 0 is specified
		if (argcmp("dlgoutput", argv [i], 'd'))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);
			
			if (tempint == 0)
				mypars->output_dlg = false;
			else
				mypars->output_dlg = true;
			if(!mypars->output_dlg && (mypars->xml2dlg || mypars->dlg2stdout)){
				printf("Note: Value of --dlgoutput (-d) ignored. Arguments --xml2dlg (-X) or --dlg2stdout (-2) require dlg output.\n");
				mypars->output_dlg = true;
			}
		}

		// Argument: choose wether to output XML or not
		// If the value is 1, XML output will be generated
		// XML output won't be generated if 0 is specified
		if (argcmp("xmloutput", argv [i], 'x'))
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);
			
			if (tempint == 0)
				mypars->output_xml = false;
			else
				mypars->output_xml = true;
		}

		// ----------------------------------
		// Argument: ligand xray pdbqt file name
		if (argcmp("xraylfile", argv[i], 'R'))
		{
			arg_recognized = 1;
			free(mypars->xrayligandfile);
			mypars->xrayligandfile = strdup(argv[i+1]);
			mypars->given_xrayligandfile = true;
			printf("Info: using --xraylfile (-R) value as X-ray ligand.\n");
		}
		// ----------------------------------

		if (arg_recognized != 1){
			printf("Error: Unknown argument '%s'.\n", argv [i]);
			print_options(argv[0]);
			return -1; // we won't get here - maybe we will in the future though ...
		}
	}

	// validating some settings
	if ((long)mypars->pop_size < mypars->gen_pdbs)
	{
		printf("Error: Value of --npdb argument (%d) cannot be greater than the population size (%lu).\n", mypars->gen_pdbs, mypars->pop_size);
//		mypars->gen_pdbs = 1;
		return -1;
	}
	
	return arg_recognized + (arg_set<<1);
}

typedef struct{
	unsigned int atom_id;
	unsigned int grid_id;
} atom_and_grid_id;

bool compare_aagid(atom_and_grid_id a, atom_and_grid_id b)
{
	return (a.grid_id<b.grid_id);
}

std::vector<ReceptorAtom> read_receptor_atoms(
                                              const char* receptor_name
                                             )
{
	std::ifstream file(receptor_name);
	if(file.fail()){
		printf("Error: Can't open receptor/flex-res file %s.\n", receptor_name);
		exit(1);
	}
	std::string line;
	char tempstr[256];
	std::vector<ReceptorAtom> atoms;
	std::vector<unsigned int> HD_ids, heavy_ids;
	ReceptorAtom current;
	while(std::getline(file, line))
	{
		sscanf(line.c_str(),"%255s",tempstr);
		if ((strcmp(tempstr, "HETATM") == 0) || (strcmp(tempstr, "ATOM") == 0))
		{
			sscanf(&line.c_str()[30], "%f %f %f", &(current.x), &(current.y), &(current.z));
			sscanf(&line.c_str()[77], "%3s", current.atom_type);
			line[27]='\0';
			sscanf(line.c_str(), "%*s %d %4s %3s %1s %d", &(current.id), current.name, current.res_name, current.chain_id, &(current.res_id));
			// assign H-bond acceptors (is going to fail for flexres with modified atom types)
			current.acceptor = is_H_acceptor(current.atom_type);
			// initialize/setup H-bond donors (is going to fail for flexres with modified atom types)
			current.donor = false;
			if(strcmp(current.atom_type,"HD")==0) HD_ids.push_back(atoms.size());
			char heavy=current.atom_type[0];
			if((heavy=='O') || (heavy=='N') || (heavy=='S')) heavy_ids.push_back(atoms.size());
			atoms.push_back(current);
		}
		if(strcmp(tempstr, "TER") == 0) break;
	}
	ReceptorAtom heavy, HD;
	// assign H-donor heavy atoms
	for(unsigned int i=0; i<HD_ids.size(); i++){
		HD=atoms[HD_ids[i]];
		double mindist2=100.0;
		int heavy_id=-1;
		for(unsigned int j=0; j<heavy_ids.size(); j++){
			heavy=atoms[heavy_ids[j]];
			double dist2 = (heavy.x-HD.x)*(heavy.x-HD.x)+(heavy.y-HD.y)*(heavy.y-HD.y)+(heavy.z-HD.z)*(heavy.z-HD.z);
			if(dist2<mindist2){
				if((heavy.atom_type[0]=='N' && dist2<=1.041*1.041) ||
				   (heavy.atom_type[0]=='O' && dist2<=1.0289*1.0289) ||
				   (heavy.atom_type[0]=='S' && dist2<=1.3455*1.3455)){
					mindist2=dist2;
					heavy_id=heavy_ids[j];
				}
			}
		}
		if(heavy_id>=0)
			atoms[heavy_id].donor=true;
		else
			atoms[HD_ids[i]].donor=true;
	}
//	printf("%d %s %s %s %d %f %f %f %s\n", current.id, current.name, current.res_name, current.chain_id, current.res_id, current.x, current.y, current.z, current.atom_type);
	return atoms;
}

std::vector<ReceptorAtom> read_receptor(
                                        const char* receptor_name,
                                        Gridinfo* mygrid,
                                        unsigned int* &in_reach_map,
                                        unsigned int* &atom_map_list,
                                        double cutoff
                                       )
{
	std::vector<ReceptorAtom> atoms = read_receptor_atoms(receptor_name);
	std::vector<ReceptorAtom> atoms_in_reach;
	cutoff /= mygrid->spacing; // convert cutoff to grid indices
	cutoff += 0.5*sqrt(3.0); // add distance to center from edge (longest possible distance to center)
	double cutoff2 = cutoff*cutoff;
	unsigned int count;
	unsigned int g1 = mygrid->size_xyz[0];
	unsigned int g2 = g1*mygrid->size_xyz[1];
	std::vector<atom_and_grid_id> atom_and_grid_ids;
	// go over receptor atoms
	for (unsigned int i=0; i<atoms.size(); i++){
		// turn atom coordinates into grid coordinates
		atoms[i].x -= mygrid->origo_real_xyz[0];
		atoms[i].y -= mygrid->origo_real_xyz[1];
		atoms[i].z -= mygrid->origo_real_xyz[2];
		atoms[i].x /= mygrid->spacing;
		atoms[i].y /= mygrid->spacing;
		atoms[i].z /= mygrid->spacing;
		// find grid boxes that are touched by the cutoff around the atom
		count=0;
		for(int z=std::max(0,(int)floor(atoms[i].z-cutoff)); z<std::min(mygrid->size_xyz[2]-1,(int)ceil(atoms[i].z+cutoff)); z++)
			for(int y=std::max(0,(int)floor(atoms[i].y-cutoff)); y<std::min(mygrid->size_xyz[1]-1,(int)ceil(atoms[i].y+cutoff)); y++)
				for(int x=std::max(0,(int)floor(atoms[i].x-cutoff)); x<std::min(mygrid->size_xyz[0]-1,(int)ceil(atoms[i].x+cutoff)); x++)
				{
					// calculate (square) distance from current grid box center to current receptor atom
					double dist2 = (atoms[i].x-(x+0.5))*(atoms[i].x-(x+0.5))+
					               (atoms[i].y-(y+0.5))*(atoms[i].y-(y+0.5))+
					               (atoms[i].z-(z+0.5))*(atoms[i].z-(z+0.5));
					if(dist2<=cutoff2){ // we're within the cutoff + longest distance to center
						atom_and_grid_id aagid;
						aagid.atom_id = atoms_in_reach.size(); aagid.grid_id = x  + y*g1  + z*g2;
						atom_and_grid_ids.push_back(aagid);
						count++;
					}
				}
		if(count==0) continue; // atom is not within reach of any grid spaces -- moving on
		atoms_in_reach.push_back(atoms[i]);
	}
	// sort so we can assign index list for atoms_in_reach to grid map
	std::sort(atom_and_grid_ids.begin(), atom_and_grid_ids.end(), compare_aagid);
	in_reach_map = (unsigned int*)malloc(sizeof(unsigned int)*
	                                     (mygrid->size_xyz[0])*
	                                     (mygrid->size_xyz[1])*
	                                     (mygrid->size_xyz[2]));
	memset(in_reach_map,0,sizeof(unsigned int)*
	                      (mygrid->size_xyz[0])*
	                      (mygrid->size_xyz[1])*
	                      (mygrid->size_xyz[2]));
	unsigned int current_gid=atom_and_grid_ids[0].grid_id;
	unsigned count_idx=0;
	in_reach_map[current_gid]=count_idx;
	count = 0;
	std::vector<unsigned int> folded_atom_list;
	folded_atom_list.push_back(0); // placeholder to be filled with the number of atoms
	unsigned int grid_boxes=1;
	for(unsigned int i=0; i<atom_and_grid_ids.size(); i++){
		if(current_gid!=atom_and_grid_ids[i].grid_id){ // new grid box
			current_gid = atom_and_grid_ids[i].grid_id;
			folded_atom_list[count_idx] = count; // fill in the current counts
			count_idx = folded_atom_list.size(); // new idx pointing to counter (start of atom list) for map
			folded_atom_list.push_back(0); // to be filled out when we know the counts
			in_reach_map[current_gid]=count_idx;
			count = 0; // reset counter for this box
			grid_boxes++;
		}
		folded_atom_list.push_back(atom_and_grid_ids[i].atom_id);
		count++;
	}
	folded_atom_list[count_idx] = count; // last one needs to be taken care of
	atom_map_list = (unsigned int*)malloc(sizeof(unsigned int)*folded_atom_list.size());
	memcpy(atom_map_list, folded_atom_list.data(), sizeof(unsigned int)*folded_atom_list.size());
//	printf("total: %d atoms in %d grid boxes\n", folded_atom_list.size(), grid_boxes);
	return atoms_in_reach;
}

void read_xml_filenames(
                        char* xml_filename,
                        char* &dpf_filename,
                        char* &grid_filename,
                        char* &ligand_filename,
                        char* &flexres_filename,
                        unsigned int &list_nr,
                        uint32_t seed[3]
                       )
{
	std::ifstream file(xml_filename);
	if(file.fail()){
		printf("\nError: Could not open xml file %s. Check path and permissions.\n", xml_filename);
		exit(3);
	}
	int error=0;
	bool grid_found=false;
	bool ligand_found=false;
	std::string line;
	char tmpstr[256];
	size_t line_nr=0;
	list_nr=0;
	seed[0]=0; seed[1]=0; seed[2]=0;
	while(std::getline(file, line)) {
		line_nr++;
		trim(line); // Remove leading and trailing whitespace
		if(line.find("<runs>")==0){ // this is where the filename info ends in the xml file -- good time to stop ;-)
			break;
		}
		if(line.find("<dpf>")==0){
			if(!sscanf(line.c_str(),"<dpf>%255[^<]/dpf>",tmpstr)){
				error = 1;
				break;
			}
			if(dpf_filename==NULL) dpf_filename=strdup(tmpstr);
		}
		if(line.find("<grid>")==0){
			if(!sscanf(line.c_str(),"<grid>%255[^<]/grid>",tmpstr)){
				error = 1;
				break;
			}
			if(grid_filename==NULL) grid_filename=strdup(tmpstr);
			grid_found = true;
		}
		if(line.find("<ligand>")==0){
			if(!sscanf(line.c_str(),"<ligand>%255[^<]/ligand>",tmpstr)){
				error = 2;
				break;
			}
			if(ligand_filename==NULL) ligand_filename=strdup(tmpstr);
			ligand_found = true;
		}
		if(line.find("<flexres>")==0){
			if(!sscanf(line.c_str(),"<flexres>%255[^<]/flexres>",tmpstr)){
				error = 3;
				break;
			}
			if(flexres_filename==NULL) flexres_filename=strdup(tmpstr);
		}
		if(line.find("<seed>")==0){
			if(!sscanf(line.c_str(),"<seed>%d %d %d</seed>",&seed[0],&seed[1],&seed[2])){
				error = 4;
				break;
			}
		}
		if(line.find("<list_nr>")==0){
			if(!sscanf(line.c_str(),"<list_nr>%d</list_nr>",&list_nr)){
				error = 5;
				break;
			}
		}
	}
	if(!grid_found || !ligand_found) error |= 16;
	if(error){
		printf("Error: XML file is not in AutoDock-GPU format (error #%d in line %lu).\n",error,line_nr);
		exit(error);
	}
}

std::vector<float> read_xml_genomes(
                                    char* xml_filename,
                                    float grid_spacing,
                                    int &nrot,
                                    bool store_axisangle
                                   )
{
	std::vector<float> result;
	std::ifstream file(xml_filename);
	if(file.fail()){
		printf("\nError: Could not open xml file %s. Check path and permissions.\n", xml_filename);
		exit(3);
	}
	std::string line, items;
	size_t found;
	int count=0;
	int run_nr=-1;
	bool set_nrot=false;
	int curr_nrot=-1;
	int error=0;
	size_t found_genome=0;
	float *gene, theta, phi, genrot;
	size_t line_nr=0;
	while(std::getline(file, line)) {
		line_nr++;
		trim(line); // Remove leading and trailing whitespace
		if(line.find("<run id=")==0){
			if(!sscanf(line.c_str(),"<run id=\"%d\">",&run_nr)){
				error=1;
				break;
			};
			if(found_genome!=0){ // indicates that no </run> is in xml so read items don't reset
				error=2;
				break;
			}
			if(run_nr>count){
				count=run_nr;
				result.resize(GENOTYPE_LENGTH_IN_GLOBMEM*count);
			}
		}
		if(line.find("</run>")==0){
			run_nr=-1;
			curr_nrot=-1;
			if(found_genome!=3){
				error=3;
				break;
			}
			found_genome=0;
		}
		if(line.find("<ndihe>")==0){
			if(run_nr>0){
				if(!sscanf(line.c_str(),"<ndihe>%d</ndihe>",&curr_nrot)){
					error=4;
					break;
				}
				if(set_nrot){
					if(curr_nrot!=nrot){
						error=5;
						break;
					}
				} else{
					nrot=curr_nrot;
					set_nrot=true;
				}
			} else{
				error=6;
				break;
			}
		}
		if(line.find("<tran0>")==0){
			if(run_nr>0){
				gene = result.data() + (run_nr-1)*GENOTYPE_LENGTH_IN_GLOBMEM;
				found=sscanf(line.c_str(),"<tran0>%f %f %f</tran0>",gene,gene+1,gene+2);
				if(found!=3){
					error=7;
					break;
				}
				*gene/=grid_spacing;
				*(gene+1)/=grid_spacing;
				*(gene+2)/=grid_spacing;
				found_genome++;
			} else{
				error=8;
				break;
			}
		}
		if(line.find("<axisangle0>")==0){
			if(run_nr>0){
				gene = result.data() + (run_nr-1)*GENOTYPE_LENGTH_IN_GLOBMEM + 3;
				found=sscanf(line.c_str(),"<axisangle0>%f %f %f %f</axisangle0>",gene,gene+1,gene+2,&genrot);
				if(found!=4){
					error=9;
					break;
				}
				if(!store_axisangle){
					theta=acos(*(gene+2)/sqrt((*gene)*(*gene)+(*(gene+1))*(*(gene+1))+(*(gene+2))*(*(gene+2))));
					phi=atan2(*(gene+1),*gene);
					*gene = phi / DEG_TO_RAD;
					*(gene+1) = theta / DEG_TO_RAD;
					*(gene+2) = genrot;
				} else result[run_nr*GENOTYPE_LENGTH_IN_GLOBMEM-1] = genrot;
				found_genome++;
			} else{
				error=10;
				break;
			}
		}
		if(line.find("<dihe0>")==0){
			if((run_nr>0) && (curr_nrot>=0)){
				if(curr_nrot>0){
					gene = result.data() + (run_nr-1)*GENOTYPE_LENGTH_IN_GLOBMEM + 6;
					items=line.substr(7);
					for(int gene_id=0; gene_id<curr_nrot; gene_id++){
						found=sscanf(items.c_str(),"%f",gene+gene_id);
						if(!found){
							error=11;
							break;
						}
						items=items.substr(items.find(" ")+1);
					}
				}
				found_genome++;
			} else{
				error=12;
				break;
			}
		}
	}
	if(error){
		printf("Error: XML file is not in AutoDock-GPU format (error #%d in line %lu).\n",error,line_nr);
		exit(error);
	}
	return result;
}

void gen_initpop_and_reflig(
                                  Dockpars*   mypars,
                                  float*      init_populations,
                                  Liganddata* myligand,
                            const Gridinfo*   mygrid
                           )
// The function generates a random initial population
// Each contiguous GENOTYPE_LENGTH_IN_GLOBMEM pieces of floats in init_population corresponds to a genotype
{
	unsigned int entity_id, gene_id;
	double movvec_to_origo[3];

	int pop_size = mypars->pop_size;

	float u1, u2, u3; // to generate random quaternion
	float qw, qx, qy, qz; // random quaternion
	float x, y, z, s; // convert quaternion to angles
	float phi, theta, rotangle;

	// Local random numbers for thread safety/reproducibility
	LocalRNG r(mypars->seed);
	
	// Generating initial population
	unsigned int nr_genomes_loaded=0;
	if(mypars->load_xml){ // read population data from previously output xml file
		int nrot;
		std::vector<float> genome = read_xml_genomes(mypars->load_xml, mygrid->spacing, nrot);
		if(nrot!=myligand->num_of_rotbonds){
			printf("Error: XML genome contains %d rotatable bonds but current ligand has %d.\n",nrot,myligand->num_of_rotbonds);
			exit(2);
		}
		nr_genomes_loaded = std::min(genome.size()/GENOTYPE_LENGTH_IN_GLOBMEM, mypars->num_of_runs);
		if(nr_genomes_loaded < mypars->num_of_runs){
			printf("Note: XML contains %d genomes but %lu runs are requested, randomizing other runs.\n",nr_genomes_loaded, mypars->num_of_runs);
		}
		// copy to rest of population
		float *src = genome.data();
		printf("Initializing %d runs from specified xml file.\n",nr_genomes_loaded);
		nr_genomes_loaded *= pop_size;
		for (entity_id=0; entity_id<nr_genomes_loaded; entity_id++){
			if(entity_id % pop_size == 0) src = genome.data() + (entity_id/pop_size)*GENOTYPE_LENGTH_IN_GLOBMEM;
			memcpy(&(init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM]),src,sizeof(float)*GENOTYPE_LENGTH_IN_GLOBMEM);
		}
	}
	for (entity_id=nr_genomes_loaded; entity_id<pop_size*mypars->num_of_runs; entity_id++)
	{
		// randomize location and convert to grid coordinates
		for (gene_id=0; gene_id<3; gene_id++)
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id] = r.random_float()*(mygrid->size_xyz_angstr[gene_id])/mygrid->spacing;
		
		// generate random quaternion
		u1 = r.random_float();
		u2 = r.random_float();
		u3 = r.random_float();
		qw = sqrt(1.0 - u1) * sin(PI_TIMES_2 * u2);
		qx = sqrt(1.0 - u1) * cos(PI_TIMES_2 * u2);
		qy = sqrt(      u1) * sin(PI_TIMES_2 * u3);
		qz = sqrt(      u1) * cos(PI_TIMES_2 * u3);
		
		// convert to angle representation
		rotangle = 2.0 * acos(qw);
		s = sqrt(1.0 - (qw * qw));
		if (s < 0.001){ // rotangle too small
			x = qx;
			y = qy;
			z = qz;
		} else {
			x = qx / s;
			y = qy / s;
			z = qz / s;
		}
		theta = acos(z);
		phi = atan2(y, x);
		
		init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+3] = phi / DEG_TO_RAD;
		init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+4] = theta / DEG_TO_RAD;
		init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+5] = rotangle / DEG_TO_RAD;
		for (gene_id=6; gene_id<MAX_NUM_OF_ROTBONDS+6; gene_id++) {
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id] = r.random_float()*360;
		}
	}
	get_movvec_to_origo(myligand, movvec_to_origo);
	double flex_vec[3];
	for (unsigned int i=0; i<3; i++)
		flex_vec [i] = -mygrid->origo_real_xyz [i];
	move_ligand(myligand, movvec_to_origo, flex_vec);
	scale_ligand(myligand, 1.0/mygrid->spacing);
	get_moving_and_unit_vectors(myligand);
}
