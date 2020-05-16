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




#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef USE_PIPELINE
#include <omp.h>
#endif
#include <vector>

#include "processgrid.h"
#include "processligand.h"
#include "getparameters.h"
#include "performdocking.h"
#include "filelist.hpp"
#include "setup.hpp"
#include "profile.hpp"
#include "simulation_state.hpp"
#include "GpuData.h"

#ifndef _WIN32
// Time measurement
#include <sys/time.h>
#endif

template<typename T>
inline double seconds_since(T& time_start)
{
#ifndef _WIN32
	timeval time_end;
	gettimeofday(&time_end,NULL);
        double num_sec     = time_end.tv_sec  - time_start.tv_sec;
        double num_usec    = time_end.tv_usec - time_start.tv_usec;
        return (num_sec + (num_usec/1000000));
#else
	return 0.0;
#endif
}

template<typename T>
inline void start_timer(T& time_start)
{
#ifndef _WIN32
	gettimeofday(&time_start,NULL);
#endif
}

int main(int argc, char* argv[])
{
	// CPU thread setup
	int nthreads=1;
#ifdef USE_PIPELINE
	#pragma omp parallel
	{nthreads = omp_get_num_threads();}
#endif
	int execution_thread=nthreads-1; // Last thread does the execution
        int nqueues=std::max(nthreads-1,1); // The other threads each get a queue

	// Timer initializations
#ifndef _WIN32
	timeval time_start, idle_timer, setup_timer, exec_timer, processing_timer;
	start_timer(time_start);
	start_timer(idle_timer);
#endif
	double total_setup_time=0;
	double total_processing_time=0;
	double total_exec_time=0;
	double idle_time;

	// File list setup if -filelist option is on
	FileList filelist;
	int n_files;
	if (get_filelist(&argc, argv, filelist) != 0)
		      return 1;
	if (filelist.used){
		n_files = filelist.nfiles;
		printf("\nRunning %d jobs in pipeline mode ", n_files);
		printf("\nUsing %d threads", nthreads);
	} else {
		n_files = 1;
	}

	// Setup master map set (one for now, nthreads-1 for general case)
	std::vector<Map> all_maps;

	// Objects that are arguments of docking_with_gpu
    Dockpars   mypars[nqueues];
    Liganddata myligand_init[nqueues];
    Gridinfo   mygrid[nqueues];
    Liganddata myxrayligand[nqueues];
    std::vector<float> floatgrids[nqueues];
	SimulationState sim_state[nqueues];
	GpuData cData;
	GpuTempData tData;

	// Set up run profiles for timing
	bool get_profiles = true; // hard-coded switch to use ALS's job profiler
	Profiler profiler;
	for (int i=0;i<n_files;i++){
		profiler.p.push_back(Profile(i));
		if (!get_profiles) break; // still create 1 if off
	}

	// Print version info
	printf("\nAutoDock-GPU version: %s\n", VERSION);

	bool finished_all=false;
	std::vector<int> job_in_queue(nqueues,-1);
	enum Stage { Setup, Launch, Processing }; 
	std::vector<Stage> stage(nqueues,Setup); // Each queue can be in one of three stages
	int n_finished_jobs=0;
	int next_job_to_setup=0;
	int err = 0;

	setup_gpu_for_docking(cData,tData);

#ifdef USE_PIPELINE
	#pragma omp parallel
	{
	int t_id = omp_get_thread_num();
#else
	{
	int t_id = 0;
#endif
	while (!finished_all){
		if(t_id!=execution_thread || nthreads==1) { // This thread handles setup and processing
			if (stage[t_id]==Setup && next_job_to_setup<n_files){ // If setup needed
				int i_job;
				// Grab the next job in atomic capture so two threads don't set up the same job
#ifdef USE_PIPELINE
				#pragma omp atomic capture
#endif
				{ i_job=next_job_to_setup; next_job_to_setup++; }

				// Setup the next file in the queue
				if (i_job<n_files) {
					printf ("\n(Thread %d is setting up Job %d)",t_id,i_job); fflush(stdout);
					job_in_queue[t_id]=i_job;
					start_timer(setup_timer);
					// Load files, read inputs, prepare arrays for docking stage
					if (setup(all_maps,mygrid[t_id], floatgrids[t_id], mypars[t_id], myligand_init[t_id], myxrayligand[t_id], filelist, tData.pMem_fgrids, i_job, argc, argv) != 0) {
						// If error encountered: Set error flag to 1; Add to count of finished jobs
						// Keep in setup stage rather than moving to launch stage so a different job will be set up
						printf("\n\nError in setup of Job #%d:", i_job);
						if (filelist.used){
                		                	printf("\n(   Field file: %s )",  filelist.fld_files[i_job].c_str());
		                                	printf("\n(   Ligand file: %s )", filelist.ligand_files[i_job].c_str()); fflush(stdout);
						}
						err = 1;
#ifdef USE_PIPELINE
						#pragma omp atomic update
#endif
						n_finished_jobs+=1;
					} else { // Successful setup
#ifdef USE_PIPELINE
						#pragma omp atomic update
#endif
						total_setup_time+=seconds_since(setup_timer);
						stage[t_id]=Launch; // Indicate this queue is ready for use
					}
				}
			}
			if (stage[t_id]==Processing){ // If ready for processing
				int i_job = job_in_queue[t_id];
				printf ("\n(Thread %d is processing Job %d)",t_id,i_job); fflush(stdout);

				start_timer(processing_timer);
                                process_result(&(mygrid[t_id]), floatgrids[t_id].data(), &(mypars[t_id]), &(myligand_init[t_id]), &(myxrayligand[t_id]), &argc,argv, sim_state[t_id]);
#ifdef USE_PIPELINE
				#pragma omp atomic update
#endif
				total_processing_time+=seconds_since(processing_timer);
				stage[t_id]=Setup; // Indicate this queue is ready for use
#ifdef USE_PIPELINE
				#pragma omp atomic update
#endif
				n_finished_jobs+=1;
                        }
		}

		if(t_id==execution_thread) { // This thread handles the GPU
			// Check if there is a job ready to launch
			int i_queue=-1;
			for (int i=0;i<stage.size();i++){
				if (stage[i]==Launch){
					i_queue=i;
					break;
				}
			}
			if (i_queue>=0){
				int i_job = job_in_queue[i_queue];
				sim_state[i_queue].idle_time = seconds_since(idle_timer);
				start_timer(exec_timer);
				printf("\nRunning Job #%d: ", i_job);
				if (filelist.used){
                                	printf("\n   Fields from: %s",  filelist.fld_files[i_job].c_str());
                                	printf("\n   Ligands from: %s", filelist.ligand_files[i_job].c_str()); fflush(stdout);
				}
				// Starting Docking
				if (docking_with_gpu(&(mygrid[i_queue]), floatgrids[i_queue].data(), &(mypars[i_queue]), &(myligand_init[i_queue]), &(myxrayligand[i_queue]), profiler.p[(get_profiles ? i_job : 0)], &argc, argv, sim_state[i_queue], cData, tData, filelist.only_one_protein ) != 0){

					// If error encountered: Set error flag to 1; Add to count of finished jobs
					// Set back to setup stage rather than moving to processing stage so a different job will be set up
					printf("\n\nError in docking_with_gpu, stopped Job %d.",i_job);
					err = 1;
#ifdef USE_PIPELINE
					#pragma omp atomic update
#endif
					n_finished_jobs+=1;
					stage[i_queue]=Setup;
				} else { // Successful run
#ifndef _WIN32
					sim_state[i_queue].exec_time = seconds_since(exec_timer);
					total_exec_time+=sim_state[i_queue].exec_time;
					printf("\nJob #%d took %.3f sec after waiting %.3f sec for setup", i_job, sim_state[i_queue].exec_time, sim_state[i_queue].idle_time);
					start_timer(idle_timer);
					if (get_profiles && filelist.used){
		                        	// Detailed timing information to .timing
		                        	profiler.p[i_job].exec_time = sim_state[i_queue].exec_time;
					}
#endif
					stage[i_queue]=Processing; // Indicate this queue is ready for a new setup
				}
			} else { // No job is ready to launch
				// Wait
			}
		}
		if (n_finished_jobs==n_files) finished_all=true;
	} // end of while loop
	} // end of parallel section

	finish_gpu_from_docking(cData,tData);

#ifndef _WIN32
	// Total time measurement
	printf("\nRun time of entire job set (%d files): %.3f sec", n_files, seconds_since(time_start));
	printf("\nSavings from multithreading: %.3f sec",(total_setup_time+total_processing_time+total_exec_time) - seconds_since(time_start));
	//if (filelist.used && filelist.only_one_protein) printf("\nSavings from receptor reuse: %.3f sec * avg_maps_used/n_maps",receptor_reuse_time*n_files);
	printf("\nIdle time of execution thread: %.3f sec",seconds_since(time_start) - total_exec_time);
	if (get_profiles && filelist.used) profiler.write_profiles_to_file(filelist.filename);
#endif
	if (err==1){
		printf("\nWARNING: Not all jobs were successful. Search output for 'Error' for details.");
	} else {
		printf("\nAll jobs ran without errors.");
	}

	return 0;
}
