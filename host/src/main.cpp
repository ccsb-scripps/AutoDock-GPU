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
	timeval time_start, idle_timer;
	start_timer(time_start);
	start_timer(idle_timer);
#else
	// Dummy variables if timers off
	double time_start, idle_timer;
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
		printf("Running %d jobs in pipeline mode\n", n_files);
	} else {
		n_files = 1;
	}

	// Setup master map set (one for now, nthreads-1 for general case)
	std::vector<Map> all_maps;

	// Objects that are arguments of docking_with_gpu
	GpuData cData;
	GpuTempData tData;

	cData.devnum=-1;
	// Get device number to run on
	for (unsigned int i=1; i<argc-1; i+=2)
	{
		if (strcmp("-devnum", argv [i]) == 0)
		{
			unsigned int tempint;
			sscanf(argv [i+1], "%u", &tempint);
			if ((tempint >= 1) && (tempint <= 65536))
				cData.devnum = (unsigned long) tempint-1;
			else
				printf("Warning: value of -devnum argument ignored. Value must be an integer between 1 and 65536.\n");
			break;
		}
	}
	cData.preload_gridsize = preload_gridsize(filelist);
	// Set up run profiles for timing
	bool get_profiles = true; // hard-coded switch to use ALS's job profiler
	Profiler profiler;
	for (int i=0;i<n_files;i++){
		profiler.p.push_back(Profile(i));
		if (!get_profiles) break; // still create 1 if off
	}

	// Error flag for each ligand
	std::vector<int> err(n_files,0);

	// Print version info
	printf("AutoDock-GPU version: %s\n", VERSION);

	setup_gpu_for_docking(cData,tData);
	total_setup_time+=seconds_since(time_start);

#ifdef USE_PIPELINE
	#pragma omp parallel
	{
		int t_id = omp_get_thread_num();
		#pragma omp master
		{printf("\nUsing %d OpenMP threads\n", omp_get_num_threads());}
		#pragma omp barrier
#else
	{
		int t_id = 0;
#endif
		Dockpars   mypars;
		if(filelist.used){ // otherwise it gets created from command line arguments
			mypars.fldfile = (char*)malloc((filelist.max_len+1)*sizeof(char));
			mypars.ligandfile = (char*)malloc((filelist.max_len+1)*sizeof(char));
		}
		Liganddata myligand_init;
		Gridinfo   mygrid;
		Liganddata myxrayligand;
		std::vector<float> floatgrids;
		SimulationState sim_state;
#ifndef _WIN32
	        timeval setup_timer, exec_timer, processing_timer;
#else
		double setup_timer, exec_timer, processing_timer;
#endif
#ifdef USE_PIPELINE
		#pragma omp for schedule(dynamic,1)
#endif
		for(int i_job=0; i_job<n_files; i_job++){
			// Setup the next file in the queue
			printf ("(Thread %d is setting up Job %d)\n",t_id,i_job); fflush(stdout);
			start_timer(setup_timer);
			// Load files, read inputs, prepare arrays for docking stage
			if (setup(all_maps, mygrid, floatgrids, mypars, myligand_init, myxrayligand, filelist, i_job, argc, argv) != 0) {
				// If error encountered: Set error flag to 1; Add to count of finished jobs
				// Keep in setup stage rather than moving to launch stage so a different job will be set up
				printf("\n\nError in setup of Job #%d:\n", i_job);
				if (filelist.used){
					printf("(   Field file: %s )\n",  filelist.fld_files[i_job].c_str());
					printf("(   Ligand file: %s )\n", filelist.ligand_files[i_job].c_str()); fflush(stdout);
				}
				err[i_job] = 1;
				continue;
			} else { // Successful setup
				// Copy preloaded maps to GPU
#ifdef USE_PIPELINE
				#pragma omp critical
#endif
				{
					if(filelist.preload_maps && filelist.load_maps_gpu){
						int size_of_one_map = 4*mygrid.size_xyz[0]*mygrid.size_xyz[1]*mygrid.size_xyz[2];
						for (int t=0; t < all_maps.size(); t++)
							copy_map_to_gpu(tData,all_maps,t,size_of_one_map);
						filelist.load_maps_gpu=false;
					}
				}
#ifdef USE_PIPELINE
				#pragma omp atomic update
#endif
				total_setup_time+=seconds_since(setup_timer);
			}

			// Starting Docking
			int error_in_docking;
			// Critical section to only let one thread access GPU at a time
#ifdef USE_PIPELINE
			#pragma omp critical
#endif
			{
				printf("\nRunning Job #%d:\n", i_job);
				if (filelist.used){
					printf("   Fields from: %s\n",  filelist.fld_files[i_job].c_str());
				 	printf("   Ligands from: %s\n", filelist.ligand_files[i_job].c_str()); fflush(stdout);
				}
				// End idling timer, start exec timer
				sim_state.idle_time = seconds_since(idle_timer);
				start_timer(exec_timer);
				// Dock
				error_in_docking = docking_with_gpu(&(mygrid), floatgrids.data(), &(mypars), &(myligand_init), &(myxrayligand), profiler.p[(get_profiles ? i_job : 0)], &argc, argv, sim_state, cData, tData, filelist.preload_maps);
				// End exec timer, start idling timer
				sim_state.exec_time = seconds_since(exec_timer);
				start_timer(idle_timer);
			}

			if (error_in_docking!=0){
				// If error encountered: Set error flag to 1; Add to count of finished jobs
				// Set back to setup stage rather than moving to processing stage so a different job will be set up
				printf("\n\nError in docking_with_gpu, stopped Job %d.\n",i_job);
				err[i_job] = 1;
				continue;
			} else { // Successful run
#ifndef _WIN32
#ifdef USE_PIPELINE
                                #pragma omp atomic update
#endif
				total_exec_time+=sim_state.exec_time;
				printf("\nJob #%d took %.3f sec after waiting %.3f sec for setup\n", i_job, sim_state.exec_time, sim_state.idle_time);
				if (get_profiles && filelist.used){
					// Detailed timing information to .timing
					profiler.p[i_job].exec_time = sim_state.exec_time;
				}
#endif
			}

			// Post-processing
			printf ("\n(Thread %d is processing Job %d)\n",t_id,i_job); fflush(stdout);
			start_timer(processing_timer);
			process_result(&(mygrid), floatgrids.data(), &(mypars), &(myligand_init), &(myxrayligand), &argc,argv, sim_state);
#ifdef USE_PIPELINE
	                #pragma omp atomic update
#endif
			total_processing_time+=seconds_since(processing_timer);
		} // end of for loop
		// Clean up memory dynamically allocated to not leak
		if(mypars.fldfile) free(mypars.fldfile); // although those strings should be allocated, it doesn't hurt to make sure
		if(mypars.ligandfile) free(mypars.ligandfile);
		if(mypars.flexresfile) free(mypars.flexresfile);
		if(mypars.xrayligandfile) free(mypars.xrayligandfile);
		if(mypars.resname) free(mypars.resname);
		if(mygrid.grid_file_path) free(mygrid.grid_file_path);
		if(mygrid.receptor_name) free(mygrid.receptor_name);
		if(mygrid.map_base_name) free(mygrid.map_base_name);
	} // end of parallel section

#ifndef _WIN32
	// Total time measurement
	printf("\nRun time of entire job set (%d file%s): %.3f sec", n_files, n_files>1?"s":"", seconds_since(time_start));
#ifdef USE_PIPELINE
	printf("\nSavings from multithreading: %.3f sec",(total_setup_time+total_processing_time+total_exec_time) - seconds_since(time_start));
	//if (filelist.preload_maps) printf("\nSavings from receptor reuse: %.3f sec * avg_maps_used/n_maps",receptor_reuse_time*n_files);
	printf("\nIdle time of execution thread: %.3f sec",seconds_since(time_start) - total_exec_time);
	if (get_profiles && filelist.used) profiler.write_profiles_to_file(filelist.filename);
#else
	printf("\nProcessing time: %.3f sec",total_processing_time);
#endif
#endif

	finish_gpu_from_docking(cData,tData);

	// Alert user to ligands that failed to complete
	int n_errors=0;
	for (int i=0; i<n_files; i++){
		if (err[i]==1){
			if (filelist.used){
				if (n_errors==0) printf("\nWARNING: The following jobs were not successful:");
				printf("\nJob %d: %s", i, filelist.ligand_files[i].c_str());
			} else {
				printf("\nThe job was not successful.");
			}
			n_errors+=1;
		}
	}
	if (n_errors==0) printf("\nAll jobs ran without errors.\n");

	return 0;
}
