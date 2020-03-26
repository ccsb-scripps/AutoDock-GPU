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

inline float average(float* average_sd2_N)
{
	if(average_sd2_N[2]<1.0f)
		return 0.0;
	return average_sd2_N[0]/average_sd2_N[2];
}

inline float stddev(float* average_sd2_N)
{
	if(average_sd2_N[2]<1.0f)
		return 0.0;
	float sq = average_sd2_N[1]*average_sd2_N[2]-average_sd2_N[0]*average_sd2_N[0];
	if((fabs(sq)<=0.000001) || (sq<0.0)) return 0.0;
	return sqrt(sq)/average_sd2_N[2];
}

inline void checkpoint(const char* input)
{
#ifdef DOCK_DEBUG
	printf("\n");
	printf(input);
	fflush(stdout);
#endif
}

int docking_with_gpu(const Gridinfo*  	 	mygrid,
	         /*const*/ float*      		cpu_floatgrids,
                           Dockpars*   		mypars,
		     const Liganddata* 		myligand_init,
		     const Liganddata* 		myxrayligand,
		     const int*        		argc,
		           char**      		argv,
		           clock_t     		clock_start_program)
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
	Liganddata myligand_reference;

	size_t n_populations = mypars->num_of_runs * mypars->pop_size * GENOTYPE_LENGTH_IN_GLOBMEM;
	size_t n_energies = mypars->pop_size * mypars->num_of_runs;
	size_t n_prng_seeds;

	unsigned long run_cnt;	/* int run_cnt; */
	int generation_cnt;
	double progress;

	int curr_progress_cnt;
	int new_progress_cnt;

	clock_t clock_start_docking, clock_stop_docking, clock_stop_program_before_clustering;

	std::vector<float> cpu_populations(n_populations); // Populations
	std::fill(cpu_populations.begin(), cpu_populations.end(), 0); // Initialize to 0 - Isnt this overwritten? - ALS

	std::vector<float> cpu_energies(n_energies); // Energies
	std::vector<Ligandresult> cpu_result_ligands(mypars->num_of_runs); // Ligand results
	std::vector<float> cpu_ref_ori_angles(mypars->num_of_runs*3); // Reference orientation angles

	//generating initial populations and random orientation angles of reference ligand
	//(ligand will be moved to origo and scaled as well)
	myligand_reference = *myligand_init;
	gen_initpop_and_reflig(mypars, cpu_populations.data(), cpu_ref_ori_angles.data(), &myligand_reference, mygrid);

	//allocating memory in CPU for pseudorandom number generator seeds and
	//generating them (seed for each thread during GA)
	n_prng_seeds = mypars->pop_size * mypars->num_of_runs * NUM_OF_THREADS_PER_BLOCK;
	std::vector<unsigned int> cpu_prng_seeds(n_prng_seeds);

	//genseed(time(NULL));	//initializing seed generator
	genseed(0u);    // TEMPORARY: removing randomness for consistent debugging - ALS

	for (int i=0; i<mypars->pop_size * mypars->num_of_runs*NUM_OF_THREADS_PER_BLOCK; i++)
#if defined (REPRO)
		cpu_prng_seeds[i] = 1u;
#else
		cpu_prng_seeds[i] = genseed(0u);
#endif

	//allocating memory in CPU for evaluation counters
	std::vector<int> cpu_evals_of_runs(mypars->num_of_runs);
	std::fill(cpu_evals_of_runs.begin(), cpu_evals_of_runs.end(), 0);

	if (strcmp(mypars->ls_method, "ad") == 0) {
		printf("\nLocal-search chosen method is ADADELTA (ad) because that is the only one available so far in the Kokkos version.");
	} else {
		printf("\nOnly one local-search method available. Please set -lsmet ad\n\n"); return 1;
	}
	printf("\nUsing NUM_OF_THREADS_PER_BLOCK = %d ", NUM_OF_THREADS_PER_BLOCK);

	clock_start_docking = clock();

	// Progress bar
	if (mypars->autostop)
	{
		printf("\nExecuting docking runs, stopping automatically after either reaching %.2f kcal/mol standard deviation\nof the best molecules, %lu generations, or %lu evaluations, whichever comes first:\n\n",mypars->stopstd,mypars->num_of_generations,mypars->num_of_energy_evals);
		printf("Generations |  Evaluations |     Threshold    |  Average energy of best 10%%  | Samples |    Best energy\n");
		printf("------------+--------------+------------------+------------------------------+---------+-------------------\n");
	}
	else
	{
		printf("\nExecuting docking runs:\n");
		printf("        20%%        40%%       60%%       80%%       100%%\n");
		printf("---------+---------+---------+---------+---------+\n");
	}

	printf("\nExecution starts:\n\n");

	// Initialize GeneticParams (broken out of docking params since they relate to the genetic algorithm, not the docking per se
	GeneticParams genetic_params(mypars);

	// Initialize the structs containing the two alternating generations
	// Odd generation gets the initial population copied in
	Generation<DeviceType> odd_generation(mypars->pop_size * mypars->num_of_runs, cpu_populations.data());
	Generation<DeviceType> even_generation(mypars->pop_size * mypars->num_of_runs);

	// Evals of runs on device (for kernel2)
	Kokkos::View<int*,DeviceType> evals_of_runs("evals_of_runs",mypars->num_of_runs);

	// Wrap the C style arrays with an unmanaged kokkos view for easy deep copies (done after view initializations for easy sizing)
        IntView1D evals_of_runs_view(cpu_evals_of_runs.data(), evals_of_runs.extent(0)); // Note this array was prexisting
	FloatView1D energies_view(cpu_energies.data(), odd_generation.energies.extent(0));
	FloatView1D final_populations_view(cpu_populations.data(), odd_generation.conformations.extent(0));

	// Declare these constant arrays on host
	InterIntra<HostType> interintra_h;
        IntraContrib<HostType> intracontrib_h;
        Intra<HostType> intra_h;
        RotList<HostType> rotlist_h;
        Conform<HostType> conform_h;
	Grads<HostType> grads_h;
	AxisCorrection<HostType> axis_correction_h;

	// Initialize them
	// WARNING - Changes myligand_reference !!! - ALS
	if (kokkos_prepare_const_fields(myligand_reference, mypars, cpu_ref_ori_angles.data(),
                                         interintra_h, intracontrib_h, intra_h, rotlist_h, conform_h, grads_h) == 1) {
                return 1;
        }
	kokkos_prepare_axis_correction(angle, dependence_on_theta, dependence_on_rotangle,
                                        axis_correction_h);

	// Declare on device
        InterIntra<DeviceType> interintra;
        IntraContrib<DeviceType> intracontrib;
        Intra<DeviceType> intra;
        RotList<DeviceType> rotlist;
        Conform<DeviceType> conform;
	Grads<DeviceType> grads;
	AxisCorrection<DeviceType> axis_correction;

	// Copy to device
	interintra.deep_copy(interintra_h);
	intracontrib.deep_copy(intracontrib_h);
	intra.deep_copy(intra_h);
	rotlist.deep_copy(rotlist_h);
	conform.deep_copy(conform_h);
	grads.deep_copy(grads_h);
	axis_correction.deep_copy(axis_correction_h);

	// Initialize DockingParams
        DockingParams<DeviceType> docking_params(myligand_reference, mygrid, mypars, cpu_floatgrids, cpu_prng_seeds.data());

	// Perform the kernel formerly known as kernel1
	checkpoint("K_INIT");
	kokkos_calc_init_pop(odd_generation, mypars, docking_params, conform, rotlist, intracontrib, interintra, intra);
	Kokkos::fence();
	checkpoint(" ... Finished\n");

	// Perform sum_evals, formerly known as kernel2
	checkpoint("K_EVAL");
	kokkos_sum_evals(mypars, docking_params, evals_of_runs);
	Kokkos::fence();
	checkpoint(" ... Finished\n");

	Kokkos::deep_copy(evals_of_runs_view, evals_of_runs);

	generation_cnt = 0;
	curr_progress_cnt = 0;
	bool first_time = true;
	float* energies;
	float threshold = 1<<24;
	float threshold_used;
	float thres_stddev = threshold;
	float curr_avg = -(1<<24);
	float curr_std = thres_stddev;
	float prev_avg = 0.0;
	unsigned int roll_count = 0;
	float rolling[4*3];
	float rolling_stddev;
	memset(&rolling[0],0,12*sizeof(float));
	unsigned int bestN = 1;
	unsigned int Ntop = mypars->pop_size;
	unsigned int Ncream = Ntop / 10;
	float delta_energy = 2.0 * thres_stddev / Ntop;
	float overall_best_energy;
	unsigned int avg_arr_size = (Ntop+1)*3;
	float average_sd2_N[avg_arr_size];
	unsigned long total_evals;
	while ((progress = check_progress(cpu_evals_of_runs.data(), generation_cnt, mypars->num_of_energy_evals, mypars->num_of_generations, mypars->num_of_runs, total_evals)) < 100.0)
	{
		if (mypars->autostop)
		{
			if (generation_cnt % 10 == 0) {
				Kokkos::deep_copy(energies_view,odd_generation.energies);
				for(unsigned int count=0; (count<1+8*(generation_cnt==0)) && (fabs(curr_avg-prev_avg)>0.00001); count++)
				{
					threshold_used = threshold;
					overall_best_energy = 1<<24;
					memset(&average_sd2_N[0],0,avg_arr_size*sizeof(float));
					for (run_cnt=0; run_cnt < mypars->num_of_runs; run_cnt++)
					{
						energies = cpu_energies.data()+run_cnt*mypars->pop_size;
						for (unsigned int i=0; i<mypars->pop_size; i++)
						{
							float energy = energies[i];
							if(energy < overall_best_energy)
								overall_best_energy = energy;
							if(energy < threshold)
							{
								average_sd2_N[0] += energy;
								average_sd2_N[1] += energy * energy;
								average_sd2_N[2] += 1.0;
								for(unsigned int m=0; m<Ntop; m++)
									if(energy < (threshold-2.0*thres_stddev)+m*delta_energy)
									{
										average_sd2_N[3*(m+1)] += energy;
										average_sd2_N[3*(m+1)+1] += energy*energy;
										average_sd2_N[3*(m+1)+2] += 1.0;
										break; // only one entry per bin
									}
							}
						}
					}
					if(first_time)
					{
						curr_avg = average(&average_sd2_N[0]);
						curr_std = stddev(&average_sd2_N[0]);
						bestN = average_sd2_N[2];
						thres_stddev = curr_std;
						threshold = curr_avg + thres_stddev;
						delta_energy = 2.0 * thres_stddev / (Ntop-1);
						first_time = false;
					}
					else
					{
						curr_avg = average(&average_sd2_N[0]);
						curr_std = stddev(&average_sd2_N[0]);
						bestN = average_sd2_N[2];
						average_sd2_N[0] = 0.0;
						average_sd2_N[1] = 0.0;
						average_sd2_N[2] = 0.0;
						unsigned int lowest_energy = 0;
						for(unsigned int m=0; m<Ntop; m++)
						{
							if((average_sd2_N[3*(m+1)+2]>=1.0) && (lowest_energy<Ncream))
							{
								if((average_sd2_N[2]<4.0) || fabs(average(&average_sd2_N[0])-average(&average_sd2_N[3*(m+1)]))<2.0*mypars->stopstd)
								{
//									printf("Adding %f +/- %f (%i)\n",average(&average_sd2_N[3*(m+1)]),stddev(&average_sd2_N[3*(m+1)]),(unsigned int)average_sd2_N[3*(m+1)+2]);
									average_sd2_N[0] += average_sd2_N[3*(m+1)];
									average_sd2_N[1] += average_sd2_N[3*(m+1)+1];
									average_sd2_N[2] += average_sd2_N[3*(m+1)+2];
									lowest_energy++;
								}
							}
						}
//						printf("---\n");
						if(lowest_energy>0)
						{
							curr_avg = average(&average_sd2_N[0]);
							curr_std = stddev(&average_sd2_N[0]);
							bestN = average_sd2_N[2];
						}
						if(curr_std<0.5f*mypars->stopstd)
							thres_stddev = mypars->stopstd;
						else
							thres_stddev = curr_std;
						threshold = curr_avg + Ncream * thres_stddev / bestN;
						delta_energy = 2.0 * thres_stddev / (Ntop-1);
					}
				}
				printf("%11u | %12lu |%8.2f kcal/mol |%8.2f +/-%8.2f kcal/mol |%8i |%8.2f kcal/mol\n",generation_cnt,total_evals/mypars->num_of_runs,threshold_used,curr_avg,curr_std,bestN,overall_best_energy);
				fflush(stdout);
				rolling[3*roll_count] = curr_avg * bestN;
				rolling[3*roll_count+1] = (curr_std*curr_std + curr_avg*curr_avg)*bestN;
				rolling[3*roll_count+2] = bestN;
				roll_count = (roll_count + 1) % 4;
				average_sd2_N[0] = rolling[0] + rolling[3] + rolling[6] + rolling[9];
				average_sd2_N[1] = rolling[1] + rolling[4] + rolling[7] + rolling[10];
				average_sd2_N[2] = rolling[2] + rolling[5] + rolling[8] + rolling[11];
				// Finish when the std.dev. of the last 4 rounds is below 0.1 kcal/mol
				if((stddev(&average_sd2_N[0])<mypars->stopstd) && (generation_cnt>30))
				{
					printf("------------+--------------+------------------+------------------------------+---------+-------------------\n");
					printf("\n%43s evaluation after reaching\n%40.2f +/-%8.2f kcal/mol combined.\n%34i samples, best energy %8.2f kcal/mol.\n","Finished",average(&average_sd2_N[0]),stddev(&average_sd2_N[0]),(unsigned int)average_sd2_N[2],overall_best_energy);
					fflush(stdout);
					break;
				}
			}
		}
		else
		{
#ifdef DOCK_DEBUG
			ite_cnt++;
			printf("\nLGA iteration # %u\n", ite_cnt);
			fflush(stdout);
#endif
			//update progress bar (bar length is 50)
			new_progress_cnt = (int) (progress/2.0+0.5);
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

		// Perform the genetic algorithm formerly known as kernel4
		checkpoint("K_GA_GENERATION");
		if (generation_cnt % 2 == 0) { // Since we need 2 generations at any time, just alternate btw 2 mem allocations
			kokkos_gen_alg_eval_new(odd_generation, even_generation, mypars, docking_params, genetic_params,
					        conform, rotlist, intracontrib, interintra, intra);
		} else {
			kokkos_gen_alg_eval_new(even_generation, odd_generation, mypars, docking_params, genetic_params,
                                                conform, rotlist, intracontrib, interintra, intra);
		}
                Kokkos::fence();
		checkpoint(" ... Finished\n");

		if (docking_params.lsearch_rate != 0.0f) {
			if (strcmp(mypars->ls_method, "ad") == 0) {
				// Perform the ADADELTA gradient descent, formerly known as kernel7
				checkpoint("K_LS_GRAD_ADADELTA");
				if (generation_cnt % 2 == 0){
					kokkos_gradient_minAD(even_generation, mypars, docking_params,
							      conform, rotlist, intracontrib, interintra, intra, grads, axis_correction);
				} else {
					kokkos_gradient_minAD(odd_generation, mypars, docking_params,
							      conform, rotlist, intracontrib, interintra, intra, grads, axis_correction);
				}
				Kokkos::fence();
				checkpoint(" ... Finished\n");
			} else {
				// sw, sd, and fire are NOT SUPPORTED in the Kokkos version (yet)
			}
		}

		// Perform sum_evals, formerly known as kernel2
		checkpoint("K_EVAL");
	        kokkos_sum_evals(mypars, docking_params, evals_of_runs);
	        Kokkos::fence();
		checkpoint(" ... Finished\n");

		// Copy evals back to CPU
	        Kokkos::deep_copy(evals_of_runs_view, evals_of_runs);

		generation_cnt++;
	}

	clock_stop_docking = clock();
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

	// Pull results back to CPU
	if (generation_cnt % 2 == 0) {
		Kokkos::deep_copy(final_populations_view,odd_generation.conformations);
		Kokkos::deep_copy(energies_view,odd_generation.energies);
	}
	else {
		Kokkos::deep_copy(final_populations_view,even_generation.conformations);
		Kokkos::deep_copy(energies_view,even_generation.energies);
	}

	// Process results
	for (run_cnt=0; run_cnt < mypars->num_of_runs; run_cnt++)
	{
		arrange_result(cpu_populations.data()+run_cnt*mypars->pop_size*GENOTYPE_LENGTH_IN_GLOBMEM, cpu_energies.data()+run_cnt*mypars->pop_size, mypars->pop_size);
		make_resfiles(cpu_populations.data()+run_cnt*mypars->pop_size*GENOTYPE_LENGTH_IN_GLOBMEM, 
			      cpu_energies.data()+run_cnt*mypars->pop_size, 
			      &myligand_reference,
			      myligand_init,
			      myxrayligand, 
			      mypars, 
			      cpu_evals_of_runs[run_cnt], 
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
	clock_stop_program_before_clustering = clock();
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
