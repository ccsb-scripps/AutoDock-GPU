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


#include <cstdint>
#include "getparameters.h"
#include <fstream>
#include <algorithm> 
#include <cctype>
#include <locale>

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

int get_filelist(const int* argc,
                      char** argv,
		   FileList& filelist)
//The function checks if a filelist has been provided according to the proper command line arguments.
//If it is, it loads the .fld, .pdbqt, and resname files into vectors
{
        for (int i=1; i<(*argc)-1; i++)
        {
                //Argument: file name that contains list of files.
                if (strcmp("-filelist", argv[i]) == 0)
                {
                        filelist.used = true;
                        strcpy(filelist.filename, argv[i+1]);
                }
        }

	if (filelist.used){
		filelist.preload_maps = true; // By default, preload maps if filelist used
		std::ifstream file(filelist.filename);
		if(file.fail()){
			printf("\nError: Could not open filelist %s. Check path and permissions.",filelist.filename);
			return 1;
		}
		std::string line;
		bool prev_line_was_fld=false;
		while(std::getline(file, line)) {
			trim(line); // Remove leading and trailing whitespace
			int len = line.size();
			if (len>=4 && line.compare(len-4,4,".fld") == 0){
				if (prev_line_was_fld){ // Overwrite the previous fld file if two in a row
					filelist.fld_files[filelist.fld_files.size()] = line;
					printf("\n\nWarning: a listed .fld file was not used!");
				} else {
					// Add the .fld file
					filelist.fld_files.push_back(line);
					prev_line_was_fld=true;

					// If more than one unique protein, cant do map preloading yet
					if (filelist.fld_files.size()>1){
						filelist.preload_maps=false;
					}
				}
			} else if (len>=6 && line.compare(len-6,6,".pdbqt") == 0){
				// Add the .pdbqt
				filelist.ligand_files.push_back(line);
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

		if (filelist.ligand_files.size() != filelist.resnames.size() && filelist.resnames.size()>0)
			{printf("\n\nError: Inconsistent number of resnames!"); return 1;}
	}

	return 0;
}

int get_filenames_and_ADcoeffs(const int* argc,
			       char** argv,
			       Dockpars* mypars,
			       const bool multiple_files)
//The function fills the file name and coeffs fields of mypars parameter
//according to the proper command line arguments.
{
	int i;
	int ffile_given, lfile_given;
	long tempint;

	//AutoDock 4 free energy coefficients
	const double coeff_elec_scale_factor = 332.06363;

	//this model assumes the BOUND conformation is the SAME as the UNBOUND, default in AD4.2
	const AD4_free_energy_coeffs coeffs_bound = {0.1662,
						     0.1209,
						     coeff_elec_scale_factor*0.1406,
						     0.1322,
						     0.2983};

	//this model assumes the unbound conformation is EXTENDED, default if AD4.0
	const AD4_free_energy_coeffs coeffs_extended = {0.1560,
						        0.0974,
							coeff_elec_scale_factor*0.1465,
							0.1159,
							0.2744};

	//this model assumes the unbound conformation is COMPACT
	const AD4_free_energy_coeffs coeffs_compact = {0.1641,
						       0.0531,
						       coeff_elec_scale_factor*0.1272,
						       0.0603,
						       0.2272};

	mypars->coeffs = coeffs_bound;	//default coeffs
	mypars->unbound_model = 0;

	ffile_given = 0;
	lfile_given = 0;

	for (i=1; i<(*argc)-1; i++)
	{
		if (!multiple_files){
			//Argument: grid parameter file name.
			if (strcmp("-ffile", argv[i]) == 0)
			{
				ffile_given = 1;
				strcpy(mypars->fldfile, argv[i+1]);
			}

			//Argument: ligand pdbqt file name
			if (strcmp("-lfile", argv[i]) == 0)
			{
				lfile_given = 1;
				strcpy(mypars->ligandfile, argv[i+1]);
			}
		}

		//Argument: unbound model to be used.
		//0 means the bound, 1 means the extended, 2 means the compact ...
		//model's free energy coefficients will be used during docking.
		if (strcmp("-ubmod", argv[i]) == 0)
		{
			sscanf(argv[i+1], "%ld", &tempint);

			if (tempint == 0)
			{
				mypars->coeffs = coeffs_bound;
				mypars->unbound_model = 0;
			}
			else
				if (tempint == 1)
				{
					mypars->coeffs = coeffs_extended;
					mypars->unbound_model = 1;
				}
				else
				{
					mypars->coeffs = coeffs_compact;
					mypars->unbound_model = 2;
				}
		}
	}

	if (ffile_given == 0 && !multiple_files)
	{
		printf("Error: grid fld file was not defined. Use -ffile argument!\n");
		return 1;
	}

	if (lfile_given == 0 && !multiple_files)
	{
		printf("Error: ligand pdbqt file was not defined. Use -lfile argument!\n");
		return 1;
	}

	return 0;
}

void get_commandpars(const int* argc,
		         char** argv,
		        double* spacing,
		      Dockpars* mypars)
//The function processes the command line arguments given with the argc and argv parameters,
//and fills the proper fields of mypars according to that. If a parameter was not defined
//in the command line, the default value will be assigned. The mypars' fields will contain
//the data in the same format as it is required for writing it to algorithm defined registers.
{
	int   i;
	long  tempint;
	float tempfloat;
	int   arg_recognized;

	// ------------------------------------------
	//default values
	mypars->num_of_energy_evals	= 2500000;
	mypars->num_of_generations	= 27000;
	mypars->nev_provided		= false;
	mypars->use_heuristics		= false; // Flag if we want to use Diogo's heuristics
	mypars->max_num_of_energy_evals = 50000000; // By default, dont let heuristics set nev > 50M
	mypars->abs_max_dmov		= 6.0/(*spacing); 	// +/-6A
	mypars->abs_max_dang		= 90; 		// +/- 90°
	mypars->mutation_rate		= 2; 		// 2%
	mypars->crossover_rate		= 80;		// 80%
	mypars->lsearch_rate		= 80;		// 80%
    mypars->adam_beta1          = 0.9f;
    mypars->adam_beta2          = 0.999f;
    mypars->adam_epsilon        = 1.0e-8f;    
				    // unsigned long num_of_ls

	strcpy(mypars->ls_method, "sw");		// "sw": Solis-Wets, 
							// "sd": Steepest-Descent
							// "fire": FIRE, https://www.math.uni-bielefeld.de/~gaehler/papers/fire.pdf
							// "ad": ADADELTA, https://arxiv.org/abs/1212.5701
                            // "adam": ADAM, 
	mypars->smooth			= 0.5f;
	mypars->tournament_rate		= 60;		// 60%
	mypars->rho_lower_bound		= 0.01;		// 0.01
	mypars->base_dmov_mul_sqrt3	= 2.0/(*spacing)*sqrt(3.0);	// 2 A
	mypars->base_dang_mul_sqrt3	= 75.0*sqrt(3.0);		// 75°
	mypars->cons_limit		= 4;		// 4
	mypars->max_num_of_iters	= 300;
	mypars->pop_size		= 150;
	mypars->initpop_gen_or_loadfile	= 0;
	mypars->gen_pdbs		= 0;
				    // char fldfile [128]
				    // char ligandfile [128]
				    // float ref_ori_angles [3]
	mypars->devnum			= 0;
	mypars->autostop		= 0;
	mypars->stopstd			= 0.15;
	mypars->num_of_runs		= 1;
	mypars->reflig_en_reqired	= 0;
				    // char unbound_model
				    // AD4_free_energy_coeffs coeffs
	mypars->handle_symmetry		= 1;
	mypars->gen_finalpop		= 0;
	mypars->gen_best		= 0;
	strcpy(mypars->resname, "docking");
	mypars->qasp			= 0.01097f;
	mypars->rmsd_tolerance 		= 2.0;			//2 Angstroem
	strcpy(mypars->xrayligandfile, mypars->ligandfile);	// By default xray-ligand file is the same as the randomized input ligand
	mypars->given_xrayligandfile	= false;		// That is, not given (explicitly by the user)
	// ------------------------------------------

	//overwriting values which were defined as a command line argument
	for (i=1; i<(*argc)-1; i+=2)
	{
		arg_recognized = 0;

		//Argument: number of energy evaluations. Must be a positive integer.
		if (strcmp("-nev", argv[i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%ld", &tempint);

			if ((tempint > 0) && (tempint < 260000000)){
				mypars->num_of_energy_evals = (unsigned long) tempint;
				mypars->nev_provided = true;
			} else
				printf("Warning: value of -nev argument ignored. Value must be between 0 and 260000000.\n");
		}

		//Argument: number of generations. Must be a positive integer.
		if (strcmp("-ngen", argv[i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%ld", &tempint);

			if ((tempint > 0) && (tempint < 16250000))
				mypars->num_of_generations = (unsigned long) tempint;
			else
				printf("Warning: value of -ngen argument ignored. Value must be between 0 and 16250000.\n");
		}

		// ----------------------------------
		//Argument: Use Heuristics for number of evaluations (can be overwritten with -nev)
		if (strcmp("-heuristics", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if (tempint == 0)
				mypars->use_heuristics = false;
			else
				mypars->use_heuristics = true;
		}
		// ----------------------------------

		//Argument: number of energy evaluations. Must be a positive integer.
		if (strcmp("-maxnev", argv[i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%ld", &tempint);

			if ((tempint > 0) && (tempint < 260000000)){
				mypars->max_num_of_energy_evals = (unsigned long) tempint;
			} else
				printf("Warning: value of -maxnev argument ignored. Value must be between 0 and 260000000.\n");
		}


		//Argument: maximal delta movement during mutation. Must be an integer between 1 and 16.
		//N means that the maximal delta movement will be +/- 2^(N-10)*grid spacing angström.
		if (strcmp("-dmov", argv[i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%f", &tempfloat);

			if ((tempfloat > 0) && (tempfloat < 10))
				mypars->abs_max_dmov = tempfloat/(*spacing);
			else
				printf("Warning: value of -dmov argument ignored. Value must be a float between 0 and 10.\n");
		}

		//Argument: maximal delta angle during mutation. Must be an integer between 1 and 17.
		//N means that the maximal delta angle will be +/- 2^(N-8)*180/512 degrees.
		if (strcmp("-dang", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat > 0) && (tempfloat < 180))
				mypars->abs_max_dang = tempfloat;
			else
				printf("Warning: value of -dang argument ignored. Value must be a float between 0 and 180.\n");
		}

		//Argument: mutation rate. Must be a float between 0 and 100.
		//Means the rate of mutations (cca) in percent.
		if (strcmp("-mrat", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.0) && (tempfloat < 100.0))
				mypars->mutation_rate = tempfloat;
			else
				printf("Warning: value of -mrat argument ignored. Value must be a float between 0 and 100.\n");
		}

		//Argument: crossover rate. Must be a float between 0 and 100.
		//Means the rate of crossovers (cca) in percent.
		if (strcmp("-crat", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.0) && (tempfloat <= 100.0))
				mypars->crossover_rate = tempfloat;
			else
				printf("Warning: value of -crat argument ignored. Value must be a float between 0 and 100.\n");
		}

		//Argument: local search rate. Must be a float between 0 and 100.
		//Means the rate of local search (cca) in percent.
		if (strcmp("-lsrat", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			/*
			if ((tempfloat >= 0.0) && (tempfloat < 100.0))
			*/
			if ((tempfloat >= 0.0) && (tempfloat <= 100.0))
				mypars->lsearch_rate = tempfloat;
			else
				printf("Warning: value of -lrat argument ignored. Value must be a float between 0 and 100.\n");
		}

		// ---------------------------------
		// MISSING: unsigned long num_of_ls
		// ---------------------------------

		// Smoothed pairwise potentials
		if (strcmp("-smooth", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			// smooth is measured in Angstrom
			if ((tempfloat >= 0.0f) && (tempfloat <= 0.5f))
				mypars->smooth = tempfloat;
			else
				printf("Warning: value of -smooth argument ignored. Value must be a float between 0 and 0.5.\n");
		}

		//Argument: local search method: 
		// "sw": Solis-Wets
		// "sd": Steepest-Descent
		// "fire": FIRE
		// "ad": ADADELTA
        // "adam" : ADAM
		if (strcmp("-lsmet", argv [i]) == 0)
		{
			arg_recognized = 1;

			char temp[128];

			strcpy(temp, argv [i+1]);

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
			} else if (strcmp(temp, "adam") == 0) {
				strcpy(mypars->ls_method, temp);
				//mypars->max_num_of_iters = 30;
			}
			else {
				printf("Warning: value of -lsmet argument ignored. Value must be a valid string: \"sw\", \"sd\", \"fire\", \"ad\".\n");
			}
		}

		//Argument: tournament rate. Must be a float between 50 and 100.
		//Means the probability that the better entity wins the tournament round during selectin
		if (strcmp("-trat", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= /*5*/0.0) && (tempfloat <= 100.0))
				mypars->tournament_rate = tempfloat;
			else
				printf("Warning: value of -trat argument ignored. Value must be a float between 0 and 100.\n");
		}


		//Argument: rho lower bound. Must be a float between 0 and 1.
		//Means the lower bound of the rho parameter (possible stop condition for local search).
		if (strcmp("-rholb", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.0) && (tempfloat < 1.0))
				mypars->rho_lower_bound = tempfloat;
			else
				printf("Warning: value of -rholb argument ignored. Value must be a float between 0 and 1.\n");
		}

		//Argument: local search delta movement. Must be a float between 0 and grid spacing*64 A.
		//Means the spread of unifily distributed delta movement of local search.
		if (strcmp("-lsmov", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat > 0.0) && (tempfloat < (*spacing)*64/sqrt(3.0)))
				mypars->base_dmov_mul_sqrt3 = tempfloat/(*spacing)*sqrt(3.0);
			else
				printf("Warning: value of -lsmov argument ignored. Value must be a float between 0 and %lf.\n", 64*(*spacing));
		}

		//Argument: local search delta angle. Must be a float between 0 and 103°.
		//Means the spread of unifily distributed delta angle of local search.
		if (strcmp("-lsang", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat > 0.0) && (tempfloat < 103.0))
				mypars->base_dang_mul_sqrt3 = tempfloat*sqrt(3.0);
			else
				printf("Warning: value of -lsang argument ignored. Value must be a float between 0 and 103.\n");
		}

		//Argument: consecutive success/failure limit. Must be an integer between 1 and 255.
		//Means the number of consecutive successes/failures after which value of rho have to be doubled/halved.
		if (strcmp("-cslim", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if ((tempint > 0) && (tempint < 256))
				mypars->cons_limit = (unsigned long) (tempint);
			else
				printf("Warning: value of -cslim argument ignored. Value must be an integer between 1 and 255.\n");
		}

		//Argument: maximal number of iterations for local search. Must be an integer between 1 and 262143.
		//Means the number of iterations after which the local search algorithm has to terminate.
		if (strcmp("-lsit", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if ((tempint > 0) && (tempint < 262144))
				mypars->max_num_of_iters = (unsigned long) tempint;
			else
				printf("Warning: value of -lsit argument ignored. Value must be an integer between 1 and 262143.\n");
		}

		//Argument: size of population. Must be an integer between 32 and CPU_MAX_POP_SIZE.
		//Means the size of the population in the genetic algorithm.
		if (strcmp("-psize", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if ((tempint >= 2) && (tempint <= MAX_POPSIZE))
				mypars->pop_size = (unsigned long) (tempint);
			else
				printf("Warning: value of -psize argument ignored. Value must be an integer between 2 and %d.\n", MAX_POPSIZE);
		}

		//Argument: load initial population from file instead of generating one.
		//If the value is zero, the initial population will be generated randomly, otherwise it will be loaded from a file.
		if (strcmp("-pload", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if (tempint == 0)
				mypars->initpop_gen_or_loadfile = 0;
			else
				mypars->initpop_gen_or_loadfile = 1;
		}

		//Argument: number of pdb files to be generated.
		//The files will include the best docking poses from the final population.
		if (strcmp("-npdb", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if ((tempint < 0) || (tempint > MAX_POPSIZE))
				printf("Warning: value of -npdb argument ignored. Value must be an integer between 0 and %d.\n", MAX_POPSIZE);
			else
				mypars->gen_pdbs = tempint;
		}

		// ---------------------------------
		// UPDATED in : get_filelist()
		// ---------------------------------
		// Argument: name of file containing file list
		if (strcmp("-filelist", argv [i]) == 0)
			arg_recognized = 1;

		// ---------------------------------
		// MISSING: char fldfile [128]
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		//Argument: name of grid parameter file.
		if (strcmp("-ffile", argv [i]) == 0)
			arg_recognized = 1;

		// ---------------------------------
		// MISSING: char ligandfile [128]
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		//Argument: name of ligand pdbqt file
		if (strcmp("-lfile", argv [i]) == 0)
			arg_recognized = 1;

		// ---------------------------------
		// MISSING: float ref_ori_angles [3]
		// UPDATED in : gen_initpop_and_reflig()
		// ---------------------------------

		// ----------------------------------
		//Argument: OpenCL device number to use
		if (strcmp("-devnum", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%lu", &tempint);
			if ((tempint >= 1) && (tempint <= 256))
				mypars->devnum = (unsigned long) tempint-1;
			else
				printf("Warning: value of -devnum argument ignored. Value must be an integer between 1 and 256.\n");
		}
		// ----------------------------------

		// ----------------------------------
		//Argument: Multiple CG-G0 maps or not
		// - has already been tested for in
		//   main.cpp, as it's needed at grid
		//   creation time not after (now)
		if (strcmp("-cgmaps", argv [i]) == 0)
		{
			arg_recognized = 1; // stub to not complain about an unknown parameter
		}
		// ----------------------------------

		// ----------------------------------
		//Argument: Automatic stopping criterion (1) or not (0)
		if (strcmp("-autostop", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if (tempint == 0)
				mypars->autostop = 0;
			else
				mypars->autostop = 1;
		}
		// ----------------------------------

		// ----------------------------------
		//Argument: Stopping criterion standard deviation.. Must be a float between 0.01 and 2.0;
		//Means the energy standard deviation of the best candidates after which to stop evaluation when autostop is 1..
		if (strcmp("-stopstd", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.01) && (tempfloat < 2.0))
				mypars->stopstd = tempfloat;
			else
				printf("Warning: value of -stopstd argument ignored. Value must be a float between 0.01 and 2.0.\n");
		}
		// ----------------------------------
		
		//Argument: number of runs. Must be an integer between 1 and 1000.
		//Means the number of required runs
		if (strcmp("-nrun", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if ((tempint >= 1) && (tempint <= MAX_NUM_OF_RUNS))
				mypars->num_of_runs = (int) tempint;
			else
				printf("Warning: value of -nrun argument ignored. Value must be an integer between 1 and %d.\n", MAX_NUM_OF_RUNS);
		}

		//Argument: energies of reference ligand required.
		//If the value is not zero, energy values of the reference ligand is required.
		if (strcmp("-rlige", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if (tempint == 0)
				mypars->reflig_en_reqired = 0;
			else
				mypars->reflig_en_reqired = 1;
		}

		// ---------------------------------
		// MISSING: char unbound_model
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		//Argument: unbound model to be used.
		if (strcmp("-ubmod", argv [i]) == 0)
			arg_recognized = 1;

		// ---------------------------------
		// MISSING: AD4_free_energy_coeffs coeffs
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------

		//Argument: handle molecular symmetry during rmsd calculation
		//If the value is not zero, molecular syymetry will be taken into account during rmsd calculation and clustering.
		if (strcmp("-hsym", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if (tempint == 0)
				mypars->handle_symmetry = 0;
			else
				mypars->handle_symmetry = 1;
		}

		//Argument: generate final population result files.
		//If the value is zero, result files containing the final populations won't be generatied, otherwise they will.
		if (strcmp("-gfpop", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if (tempint == 0)
				mypars->gen_finalpop = 0;
			else
				mypars->gen_finalpop = 1;
		}

		//Argument: generate best.pdbqt
		//If the value is zero, best.pdbqt file containing the coordinates of the best result found during all of the runs won't be generated, otherwise it will
		if (strcmp("-gbest", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if (tempint == 0)
				mypars->gen_best = 0;
			else
				mypars->gen_best = 1;
		}

		//Argument: name of result files.
		if (strcmp("-resnam", argv [i]) == 0)
		{
			arg_recognized = 1;
			strcpy(mypars->resname, argv [i+1]);
		}

		//Argument: use modified QASP (from VirtualDrug) instead of original one used by AutoDock
		//If the value is not zero, the modified parameter will be used.
		if (strcmp("-modqp", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if (tempint == 0)
				mypars->qasp = 0.01097f;		//original AutoDock QASP parameter
			else
				mypars->qasp = 0.00679f;		//from VirtualDrug
		}

		//Argument: rmsd tolerance for clustering.
		//This will be used during clustering for the tolerance distance.
		if (strcmp("-rmstol", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if (tempfloat > 0.0)
				mypars->rmsd_tolerance = tempfloat;
			else
				printf("Warning: value of -rmstol argument ignored. Value must be a double greater than 0.\n");
		}

		// ----------------------------------
		//Argument: ligand xray pdbqt file name
		if (strcmp("-xraylfile", argv[i]) == 0)
		{
			arg_recognized = 1;
			strcpy(mypars->xrayligandfile, argv[i+1]);
			mypars->given_xrayligandfile = true;
			printf("Info: using -xraylfile value as X-ray ligand.");
		}
		// ----------------------------------


		if (arg_recognized != 1)
			printf("Warning: unknown argument '%s'.\n", argv [i]);
	}

	//validating some settings

	if (mypars->pop_size < mypars->gen_pdbs)
	{
		printf("Warning: value of -npdb argument igonred. Value mustn't be greater than the population size.\n");
		mypars->gen_pdbs = 1;
	}

}

void gen_initpop_and_reflig(Dockpars*       mypars,
			    float*          init_populations,
			    float*          ref_ori_angles,
			    Liganddata*     myligand,
			    const Gridinfo* mygrid)
//The function generates a random initial population
//(or alternatively, it reads from an external file according to mypars),
//and the angles of the reference orientation.
//The parameters mypars, myligand and mygrid describe the current docking.
//The pointers init_population and ref_ori_angles have to point to
//two allocated memory regions with proper size which the function will fill with random values.
//Each contiguous GENOTYPE_LENGTH_IN_GLOBMEM pieces of floats in init_population corresponds to a genotype,
//and each contiguous three pieces of floats in ref_ori_angles corresponds to
//the phi, theta and angle genes of the reference orientation.
//In addition, as part of reference orientation handling,
//the function moves myligand to origo and scales it according to grid spacing.
{
	int entity_id, gene_id;
	int gen_pop, gen_seeds;
	FILE* fp;
	int i;
	float init_orientation[MAX_NUM_OF_ROTBONDS+6];
	double movvec_to_origo[3];

	int pop_size = mypars->pop_size;

    float u1, u2, u3; // to generate random quaternion
    float qw, qx, qy, qz; // random quaternion
    float x, y, z, s; // convert quaternion to angles
    float phi, theta, rotangle;

	//initial population
	gen_pop = 0;

	//Reading initial population from file if only 1 run was requested
	if (mypars->initpop_gen_or_loadfile == 1)
	{
		if (mypars->num_of_runs != 1)
		{
			printf("Warning: more than 1 run was requested. New populations will be generated \ninstead of being loaded from initpop.txt\n");
			gen_pop = 1;
		}
		else
		{
			fp = fopen("initpop.txt","rb"); // fp = fopen("initpop.txt","r");
			if (fp == NULL)
			{
				printf("Warning: can't find initpop.txt. A new population will be generated.\n");
				gen_pop = 1;
			}
			else
			{
				for (entity_id=0; entity_id<pop_size; entity_id++)
					for (gene_id=0; gene_id<MAX_NUM_OF_ROTBONDS+6; gene_id++)
						fscanf(fp, "%f", &(init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id]));

				//reading reference orienation angles from file
				fscanf(fp, "%f", &(mypars->ref_ori_angles[0]));
				fscanf(fp, "%f", &(mypars->ref_ori_angles[1]));
				fscanf(fp, "%f", &(mypars->ref_ori_angles[2]));

				fclose(fp);
			}
		}
	}
	else
		gen_pop = 1;

	// Local random numbers for thread safety/reproducibility
	LocalRNG r;

	//Generating initial population
	if (gen_pop == 1)
	{
		for (entity_id=0; entity_id<pop_size*mypars->num_of_runs; entity_id++) {
			for (gene_id=0; gene_id<3; gene_id++) {
				init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id] = r.random_float()*(mygrid->size_xyz_angstr[gene_id]);
            }

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

			//printf("angles = %8.2f, %8.2f, %8.2f\n", phi / DEG_TO_RAD, theta / DEG_TO_RAD, rotangle/DEG_TO_RAD);
            
            /*
            init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+3] = (float) r.random_float() * 360.0;
            init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+4] = (float) r.random_float() * 360.0;
            init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+5] = (float) r.random_float() * 360.0;
            init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+3] = (float) r.random_float() * 360;
            init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+4] = (float) r.random_float() * 180;
            init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+5] = (float) r.random_float() * 360;
            */

			for (gene_id=6; gene_id<MAX_NUM_OF_ROTBONDS+6; gene_id++) {
					init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id] = r.random_float()*360;
			}
        }

		//generating reference orientation angles
		// mypars->ref_ori_angles[0] = (float) floor(r.random_float()*360*100)/100.0;
		// mypars->ref_ori_angles[1] = (float) floor(r.random_float()*/*360*/180*100)/100.0;
		// mypars->ref_ori_angles[2] = (float) floor(r.random_float()*360*100)/100.0;

		// mypars->ref_ori_angles[0] = 0.0;
		// mypars->ref_ori_angles[1] = 0.0;
		// mypars->ref_ori_angles[2] = 0.0;

		//Writing first initial population to initpop.txt
		fp = fopen("initpop.txt", "w");
		if (fp == NULL)
			printf("Warning: can't create initpop.txt.\n");
		else
		{
			for (entity_id=0; entity_id<pop_size; entity_id++)
				for (gene_id=0; gene_id<MAX_NUM_OF_ROTBONDS+6; gene_id++)
					fprintf(fp, "%f ", init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id]);

			//writing reference orientation angles to initpop.txt
			fprintf(fp, "%f ", mypars->ref_ori_angles[0]);
			fprintf(fp, "%f ", mypars->ref_ori_angles[1]);
			fprintf(fp, "%f ", mypars->ref_ori_angles[2]);

			fclose(fp);
		}
	}

	//genotypes should contain x, y and z genes in grid spacing instead of Angstroms
	//(but was previously generated in Angstroms since fdock does the same)

	for (entity_id=0; entity_id<pop_size*mypars->num_of_runs; entity_id++)
		for (gene_id=0; gene_id<3; gene_id++)
			init_populations [entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id] = init_populations [entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id]/mygrid->spacing;

	//changing initial orientation of reference ligand
	/*for (i=0; i<38; i++)
		switch (i)
		{
		case 3: init_orientation [i] = mypars->ref_ori_angles [0];
				break;
		case 4: init_orientation [i] = mypars->ref_ori_angles [1];
				break;
		case 5: init_orientation [i] = mypars->ref_ori_angles [2];
				break;
		default: init_orientation [i] = 0;
		}

	change_conform_f(myligand, init_orientation, 0);*/

	//initial orientation will be calculated during docking,
	//only the required angles are generated here,
	//but the angles possibly read from file are ignored

	for (uint32_t i=0; i<mypars->num_of_runs; i++)
	{
		// Enable only for debugging.
		// These specific values of rotational genes (in axis-angle space)
		// correspond to a quaternion for NO rotation.

		// ref_ori_angles[3*i]   = 0.0f;
		// ref_ori_angles[3*i+1] = 0.0f;
		// ref_ori_angles[3*i+2] = 0.0f;

		// Enable for release.
		// ref_ori_angles[3*i]   = (float) (r.random_float()*360.0); 	//phi
		// ref_ori_angles[3*i+1] = (float) (r.random_float()*180.0);	//theta
		// ref_ori_angles[3*i+2] = (float) (r.random_float()*360.0);	//angle

        // uniform distr.
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

		ref_ori_angles[3*i]   = phi / DEG_TO_RAD;
		ref_ori_angles[3*i+1] = theta / DEG_TO_RAD;
		ref_ori_angles[3*i+2] = rotangle / DEG_TO_RAD;
        
	}


#if 0
	for (i=0; i<mypars->num_of_runs; i++)
	{
//#if defined (REPRO)

		// These specific values for the rotation genes (in Shoemake space)
		// correspond to a quaternion for NO rotation.
		//ref_ori_angles[3*i]   = 0.0f;
		//ref_ori_angles[3*i+1] = 0.25f;
		//ref_ori_angles[3*i+2] = 0.0f;
//#else
		ref_ori_angles[3*i]   = ((float) rand()/ (float) RAND_MAX); 	// u1
		ref_ori_angles[3*i+1] = ((float) rand()/ (float) RAND_MAX);	// u2
		ref_ori_angles[3*i+2] = ((float) rand()/ (float) RAND_MAX);	// u3
		//printf("u1, u2, u3: %10f %10f %10f \n", ref_ori_angles[3*i], ref_ori_angles[3*i+1], ref_ori_angles[3*i+2]);
//#endif
	}
#endif

	get_movvec_to_origo(myligand, movvec_to_origo);
	move_ligand(myligand, movvec_to_origo);
	scale_ligand(myligand, 1.0/mygrid->spacing);
	get_moving_and_unit_vectors(myligand);

	/*
	printf("ligand: movvec_to_origo: %f %f %f\n", movvec_to_origo[0], movvec_to_origo[1], movvec_to_origo[2]);
	*/

}
