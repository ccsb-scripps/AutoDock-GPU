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
	} else {
		n_files = 1;
	}

	// Setup master map set (one for now, nthreads-1 for general case)
	std::vector<Map> all_maps;

	// Objects that are arguments of docking_with_gpu
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

	int err = 0;

	setup_gpu_for_docking(cData,tData);

#ifdef USE_PIPELINE
	#pragma omp parallel
	{
		int t_id = omp_get_thread_num();
		#pragma omp master
		{printf("\nUsing %d OpenMP threads", omp_get_num_threads());}
#else
	{
		int t_id = 0;
#endif
		Dockpars   mypars;
		Liganddata myligand_init;
		Gridinfo   mygrid;
		Liganddata myxrayligand;
		std::vector<float> floatgrids;
	        SimulationState sim_state;

#ifdef USE_PIPELINE
		#pragma omp for schedule(dynamic,1)
#endif
		for(int i_job=0; i_job<n_files; i_job++){
			// Setup the next file in the queue
			printf ("\n(Thread %d is setting up Job %d)",t_id,i_job); fflush(stdout);
			start_timer(setup_timer);
			// Load files, read inputs, prepare arrays for docking stage
			if (setup(all_maps,mygrid, floatgrids, mypars, myligand_init, myxrayligand, filelist, tData.pMem_fgrids, i_job, argc, argv) != 0) {
				// If error encountered: Set error flag to 1; Add to count of finished jobs
				// Keep in setup stage rather than moving to launch stage so a different job will be set up
				printf("\n\nError in setup of Job #%d:", i_job);
				if (filelist.used){
					printf("\n(   Field file: %s )",  filelist.fld_files[i_job].c_str());
					printf("\n(   Ligand file: %s )", filelist.ligand_files[i_job].c_str()); fflush(stdout);
				}
				err = 1;
				continue;
			} else { // Successful setup
#ifdef USE_PIPELINE
				#pragma omp atomic update
#endif
				total_setup_time+=seconds_since(setup_timer);
			}

			sim_state.idle_time = seconds_since(idle_timer);
			start_timer(exec_timer);
			printf("\nRunning Job #%d: ", i_job);
			if (filelist.used){
	                     	printf("\n   Fields from: %s",  filelist.fld_files[i_job].c_str());
	                      	printf("\n   Ligands from: %s", filelist.ligand_files[i_job].c_str()); fflush(stdout);
			}

			// Starting Docking
			int error_in_docking;
			// Critical section to only let one thread access GPU at a time
#ifdef USE_PIPELINE
			#pragma omp critical
#endif
			{
				error_in_docking = docking_with_gpu(&(mygrid), floatgrids.data(), &(mypars), &(myligand_init), &(myxrayligand), profiler.p[(get_profiles ? i_job : 0)], &argc, argv, sim_state, cData, tData, filelist.preload_maps);
			}

			if (error_in_docking!=0){
				// If error encountered: Set error flag to 1; Add to count of finished jobs
				// Set back to setup stage rather than moving to processing stage so a different job will be set up
				printf("\n\nError in docking_with_gpu, stopped Job %d.",i_job);
				err = 1;
				continue;
			} else { // Successful run
#ifndef _WIN32
				sim_state.exec_time = seconds_since(exec_timer);
#ifdef USE_PIPELINE
                                #pragma omp atomic update
#endif
				total_exec_time+=sim_state.exec_time;
				printf("\nJob #%d took %.3f sec after waiting %.3f sec for setup", i_job, sim_state.exec_time, sim_state.idle_time);
				start_timer(idle_timer);
				if (get_profiles && filelist.used){
	                        	// Detailed timing information to .timing
	                        	profiler.p[i_job].exec_time = sim_state.exec_time;
				}
#endif
			}

			// Post-processing
	                printf ("\n(Thread %d is processing Job %d)",t_id,i_job); fflush(stdout);
	                start_timer(processing_timer);
	                process_result(&(mygrid), floatgrids.data(), &(mypars), &(myligand_init), &(myxrayligand), &argc,argv, sim_state);
#ifdef USE_PIPELINE
	                #pragma omp atomic update
#endif
	                total_processing_time+=seconds_since(processing_timer);
		} // end of for loop
	} // end of parallel section

	finish_gpu_from_docking(cData,tData);

#ifndef _WIN32
	// Total time measurement
	printf("\nRun time of entire job set (%d files): %.3f sec", n_files, seconds_since(time_start));
	printf("\nSavings from multithreading: %.3f sec",(total_setup_time+total_processing_time+total_exec_time) - seconds_since(time_start));
	//if (filelist.preload_maps) printf("\nSavings from receptor reuse: %.3f sec * avg_maps_used/n_maps",receptor_reuse_time*n_files);
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
