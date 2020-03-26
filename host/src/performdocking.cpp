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

#include <vector>
#include <Kokkos_Core.hpp>

#include "defines.h"
#include "correct_grad_axisangle.h"
#include "autostop.hpp"
#include "performdocking.h"

// From ./kokkos
#include "kokkos_settings.hpp"
#include "dockingparams.hpp"
#include "geneticparams.hpp"
#include "kernelconsts.hpp"
#include "generation.hpp"
#include "prepare_const_fields.hpp"
#include "calc_init_pop.hpp"
#include "sum_evals.hpp"
#include "genetic_alg_eval_new.hpp"
#include "gradient_minAD.hpp"

inline void checkpoint(const char* input)
{
#ifdef DOCK_DEBUG
	printf("\n");
	printf(input);
	fflush(stdout);
#endif
}

int docking_with_gpu(const Gridinfo*		mygrid,
		 /*const*/ float*		cpu_floatgrids,
                           Dockpars*		mypars,
		     const Liganddata*		myligand_init,
		     const Liganddata*		myxrayligand,
		     const int*			argc,
			   char**		argv,
			   clock_t		clock_start_program)
/* The function performs the docking algorithm and generates the corresponding result files.
parameter mygrid:
		describes the grid
		filled with get_gridinfo()
parameter cpu_floatgrids:
		points to the memory region containing the grids
		filled with get_gridvalues_f()
parameter mypars:
		describes the docking parameters
		filled with get_commandpars()
parameter myligand_init:
		describes the ligands
		filled with get_liganddata()
parameter myxrayligand:
		describes the xray ligand
		filled with get_xrayliganddata()
parameters argc and argv:
		are the corresponding command line arguments parameter clock_start_program:
		contains the state of the clock tick counter at the beginning of the program
filled with clock() */
{
	//------------------------------- SETUP --------------------------------------//

	// Note - Kokkos views initialized to 0 by default
	Kokkos::View<float*,HostType> populations_h(  "populations_h",   mypars->num_of_runs * mypars->pop_size * GENOTYPE_LENGTH_IN_GLOBMEM);
	Kokkos::View<float*,HostType> energies_h(     "energies_h",      mypars->num_of_runs * mypars->pop_size);
	Kokkos::View<int*,  HostType> evals_of_runs_h("evals_of_runs_h", mypars->num_of_runs);

	std::vector<Ligandresult> cpu_result_ligands(mypars->num_of_runs); // Ligand results
	std::vector<float> cpu_ref_ori_angles(mypars->num_of_runs*3); // Reference orientation angles

	//generating initial populations and random orientation angles of reference ligand
	//(ligand will be moved to origo and scaled as well)
	Liganddata myligand_reference = *myligand_init;
	gen_initpop_and_reflig(mypars, populations_h.data(), cpu_ref_ori_angles.data(), &myligand_reference, mygrid);

	//genseed(time(NULL));	//initializing seed generator
	genseed(0u);    // TEMPORARY: removing randomness for consistent debugging - ALS

	// Initialize GeneticParams (broken out of docking params since they relate to the genetic algorithm, not the docking per se
	GeneticParams genetic_params(mypars);

	// Initialize the objects containing the two alternating generations
	Generation<DeviceType> odd_generation(mypars->pop_size * mypars->num_of_runs);
	Generation<DeviceType> even_generation(mypars->pop_size * mypars->num_of_runs);

	// Odd generation gets the initial population copied in
	Kokkos::deep_copy(odd_generation.conformations, populations_h);

	// Evals of runs on device (for kernel2)
	Kokkos::View<int*,DeviceType> evals_of_runs("evals_of_runs",mypars->num_of_runs);

	// Declare the constant arrays on host and device
	Constants<HostType> consts_h;
	Constants<DeviceType> consts;

	// Initialize host constants
	// WARNING - Changes myligand_reference !!! - ALS
	if (kokkos_prepare_const_fields(myligand_reference, mypars, cpu_ref_ori_angles.data(),
                                         consts_h.interintra, consts_h.intracontrib, consts_h.intra, consts_h.rotlist, consts_h.conform, consts_h.grads) == 1) {
                return 1;
        }
	kokkos_prepare_axis_correction(angle, dependence_on_theta, dependence_on_rotangle,
                                        consts_h.axis_correction);

	// Copy constants to device
	consts.deep_copy(consts_h);

	// Initialize DockingParams
        DockingParams<DeviceType> docking_params(myligand_reference, mygrid, mypars, cpu_floatgrids);

	// Input notes
	if (strcmp(mypars->ls_method, "ad") == 0) {
                printf("\nLocal-search chosen method is ADADELTA (ad) because that is the only one available so far in the Kokkos version.");
        } else {
                printf("\nOnly one local-search method available. Please set -lsmet ad\n\n"); return 1;
        }
        printf("\nUsing NUM_OF_THREADS_PER_BLOCK = %d ", NUM_OF_THREADS_PER_BLOCK);


	// Autostop / Progress bar
	AutoStop autostop(mypars->pop_size, mypars->num_of_runs, mypars->stopstd);
        if (mypars->autostop)
        {
		autostop.print_intro(mypars->num_of_generations, mypars->num_of_energy_evals);
        }
        else
        {
                printf("\nExecuting docking runs:\n");
                printf("        20%%        40%%       60%%       80%%       100%%\n");
                printf("---------+---------+---------+---------+---------+\n");
        }

	//----------------------------- EXECUTION ------------------------------------//
        printf("\nExecution starts:\n\n");
	clock_t clock_start_docking = clock();

	// Get the energy of the initial population (formerly kernel1)
	checkpoint("K_INIT");
	kokkos_calc_init_pop(odd_generation, mypars, docking_params, consts);
	Kokkos::fence();
	checkpoint(" ... Finished\n");

	// Reduction on the number of evaluations (formerly kernel2)
	checkpoint("K_EVAL");
	kokkos_sum_evals(mypars, docking_params, evals_of_runs);
	Kokkos::fence();
	checkpoint(" ... Finished\n");

	Kokkos::deep_copy(evals_of_runs_h, evals_of_runs);

	int generation_cnt = 0; // Counter of while loop
	int curr_progress_cnt = 0;
	double progress;
	unsigned long total_evals;
	while ((progress = check_progress(evals_of_runs_h.data(), generation_cnt, mypars->num_of_energy_evals, mypars->num_of_generations, mypars->num_of_runs, total_evals)) < 100.0)
	{
		if (mypars->autostop) {
			if (generation_cnt % 10 == 0) {
				Kokkos::deep_copy(energies_h,odd_generation.energies);
				bool finished = autostop.check_if_satisfactory(generation_cnt, energies_h.data(), total_evals);
				if (finished) break; // Exit loop
			}
		} else {
			//update progress bar (bar length is 50)
			int new_progress_cnt = (int) (progress/2.0+0.5);
			if (new_progress_cnt > 50)
				new_progress_cnt = 50;
			while (curr_progress_cnt < new_progress_cnt) {
				curr_progress_cnt++;
#ifndef DOCK_DEBUG
				printf("*");
#endif
				fflush(stdout);
			}
		}

		// Get the next generation via the genetic algorithm (formerly kernel4)
		checkpoint("K_GA_GENERATION");
		if (generation_cnt % 2 == 0) { // Since we need 2 generations at any time, just alternate btw 2 mem allocations
			kokkos_gen_alg_eval_new(odd_generation, even_generation, mypars, docking_params, genetic_params, consts);
		} else {
			kokkos_gen_alg_eval_new(even_generation, odd_generation, mypars, docking_params, genetic_params, consts);
		}
                Kokkos::fence();
		checkpoint(" ... Finished\n");

		// Refine conformations to minimize energies
		if (docking_params.lsearch_rate != 0.0f) {
			if (strcmp(mypars->ls_method, "ad") == 0) {
				// Use ADADELTA gradient descent (formerly kernel7)
				checkpoint("K_LS_GRAD_ADADELTA");
				if (generation_cnt % 2 == 0){
					kokkos_gradient_minAD(even_generation, mypars, docking_params, consts);
				} else {
					kokkos_gradient_minAD(odd_generation, mypars, docking_params, consts);
				}
				Kokkos::fence();
				checkpoint(" ... Finished\n");
			} else {
				// sw, sd, and fire are NOT SUPPORTED in the Kokkos version (yet)
			}
		}

		// Reduction on the number of evaluations (formerly kernel2)
		checkpoint("K_EVAL");
		kokkos_sum_evals(mypars, docking_params, evals_of_runs);
		Kokkos::fence();
		checkpoint(" ... Finished\n");

		// Copy evals back to CPU
		Kokkos::deep_copy(evals_of_runs_h, evals_of_runs);

		generation_cnt++;
	}

	clock_t clock_stop_docking = clock();
	if (mypars->autostop==0)
	{
		//update progress bar (bar length is 50)mem_num_of_rotatingatoms_per_rotbond_const
		while (curr_progress_cnt < 50) {
			curr_progress_cnt++;
			printf("*");
			fflush(stdout);
		}
	}
	printf("\n\n");

	//----------------------------- PROCESSING ------------------------------------//

	// Pull results back to CPU
	if (generation_cnt % 2 == 0) {
		Kokkos::deep_copy(populations_h,odd_generation.conformations);
		Kokkos::deep_copy(energies_h,odd_generation.energies);
	}
	else {
		Kokkos::deep_copy(populations_h,even_generation.conformations);
		Kokkos::deep_copy(energies_h,even_generation.energies);
	}

	// Arrange results and make res files
	for (unsigned long run_cnt=0; run_cnt < mypars->num_of_runs; run_cnt++)
	{
		arrange_result(populations_h.data()+run_cnt*mypars->pop_size*GENOTYPE_LENGTH_IN_GLOBMEM, energies_h.data()+run_cnt*mypars->pop_size, mypars->pop_size);
		make_resfiles(populations_h.data()+run_cnt*mypars->pop_size*GENOTYPE_LENGTH_IN_GLOBMEM, 
			      energies_h.data()+run_cnt*mypars->pop_size, 
			      &myligand_reference,
			      myligand_init,
			      myxrayligand, 
			      mypars, 
			      evals_of_runs_h[run_cnt], 
			      generation_cnt, 
			      mygrid, 
			      cpu_floatgrids, 
			      cpu_ref_ori_angles.data()+3*run_cnt, 
			      argc, 
			      argv, 
                              /*1*/0,
			      run_cnt, 
			      &(cpu_result_ligands [run_cnt]));
	}

	// Clustering analysis, generate .dlg output
	clock_t clock_stop_program_before_clustering = clock();
	clusanal_gendlg(cpu_result_ligands.data(), mypars->num_of_runs, myligand_init, mypars,
					 mygrid, argc, argv, ELAPSEDSECS(clock_stop_docking, clock_start_docking)/mypars->num_of_runs,
					 ELAPSEDSECS(clock_stop_program_before_clustering, clock_start_program),generation_cnt,total_evals/mypars->num_of_runs);
	clock_stop_docking = clock();

	return 0;
}

double check_progress(int* evals_of_runs, int generation_cnt, int max_num_of_evals, int max_num_of_gens, int num_of_runs, unsigned long &total_evals)
//The function checks if the stop condition of the docking is satisfied, returns 0 if no, and returns 1 if yes. The fitst
//parameter points to the array which stores the number of evaluations performed for each run. The second parameter stores
//the generations used. The other parameters describe the maximum number of energy evaluations, the maximum number of
//generations, and the number of runs, respectively. The stop condition is satisfied, if the generations used is higher
//than the maximal value, or if the average number of evaluations used is higher than the maximal value.
{
	//Stops if the sum of evals of every run reached the sum of the total number of evals

	double evals_progress;
	double gens_progress;

	//calculating progress according to number of runs
	total_evals = 0;
	for (int i=0; i<num_of_runs; i++)
		total_evals += evals_of_runs[i];

	evals_progress = (double)total_evals/((double) num_of_runs)/max_num_of_evals*100.0;

	//calculating progress according to number of generations
	gens_progress = ((double) generation_cnt)/((double) max_num_of_gens)*100.0; //std::cout<< "gens_progress: " << gens_progress <<std::endl;

	if (evals_progress > gens_progress)
		return evals_progress;
	else
		return gens_progress;
}
