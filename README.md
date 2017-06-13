odock

# Old commands
# This will execute 1 run
./ofdock_amd -ffile ./input_data/1hvr_vegl.maps.fld -lfile ./input_data/1hvrl.pdbqt

# This will execute 10 runs
./ofdock_amd -ffile ./input_data/1hvr_vegl.maps.fld -lfile ./input_data/1hvrl.pdbqt -nrun 10

# Updated commands for open-source release
./odock_64wi -ffile ./input_data/1stp/derived/1stp_protein.maps.fld -lfile ./input_data/1stp/derived/1stp_ligand.pdbqt -nrun 10
./odock_64wi -ffile ./input_data/3ce3/derived/3ce3_protein.maps.fld -lfile ./input_data/3ce3/derived/3ce3_ligand.pdbqt -nrun 10


odock

getparameters.cpp -> contains default values of docking parameters:

//default values					 |		|      name of parameter flags
mypars->num_of_energy_evals = 2500000;			 |		| -nev:   number of energy evaluations
mypars->num_of_generations  = 27000;			 |		| -ngen:  number of generations
mypars->abs_max_dmov        = 6.0/(*spacing); 		 |// +/-6A	| -dmov:  max delta movement during mutation
mypars->abs_max_dang        = 90; 			 |// +/- 90°	| -dang:  max delta angle during mutation
mypars->mutation_rate 	    = 2; 			 |// 2%		| -mrat:  mutation rate
mypars->crossover_rate 	    = 80;			 |// 80%	| -crat:  crossover rate
mypars->lsearch_rate 	    = 6;			 |// 6%		| -lsrat: local search rate
                              // unsigned long num_of_ls |		|
mypars->tournament_rate     = 60;			 |// 60%	| -trat:  tournament rate
mypars->rho_lower_bound     = 0.01;			 |// 0.01	| -rholb: rho lower bound
mypars->base_dmov_mul_sqrt3 = 2.0/(*spacing)*sqrt(3.0);	 |// 2 A	| -lsmov: local serach delta movement
mypars->base_dang_mul_sqrt3 = 75.0*sqrt(3.0);		 |// 75°	| -lsang: local search delat angle
mypars->cons_limit 	    = 4;			 |// 4		| -cslim: consecutive succ/failure limit
mypars->max_num_of_iters    = 300;			 |		| -lsit:  max num it for local search
mypars->pop_size            = 150;			 |		| -psize: size of population
mypars->initpop_gen_or_loadfile = 0;			 |		| -pload: load init pop from file instead
							 |		|         of generating a new one
mypars->gen_pdbs 	    = 0;			 |		| -npdb:  num of pdb files to be generated
			    // char fldfile [128]	 |		|
	           	    // char ligandfile [128]	 |		|
	      	    	    // float ref_ori_angles [3]	 |		|
mypars->num_of_runs 	    = 1;			 |		| -nrun:  number of runs
mypars->reflig_en_reqired   = 0;			 |		| -rlige: energy of ref ligand required
			    // char unbound_model	 |		|
			    // AD4_free_energy_coeffs coeffs		|
mypars->handle_symmetry     = 0;			 |		| -hsym:  handle molecular symmetry
							 |		|	  during rmsd calculation
							 |		|
mypars->gen_finalpop        = 0;			 |		| -gfpop: generate final population results
							 |		|         file
mypars->gen_best            = 0;			 |		| -gbest: generate best.pdbqt
strcpy(mypars->resname, "docking");			 |		| -resname: name the result file
mypars->qasp 		    = 0.01097f;			 |		| -modqp: use modified QASP
mypars->rmsd_tolerance      = 2.0;			 |//2 Angström	| -rmstol:rmsd tolerance for clustering
							 |		|
							 |		|
							 |		| -ubmod: unbound model to be used
							 |		|
