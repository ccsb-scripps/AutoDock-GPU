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
	Dockpars initial_pars;
	Gridinfo initial_grid;
	if (preparse_dpf(&argc, argv, &initial_pars, &initial_grid, filelist) != 0)
		return 1;
	int n_files;
	if (get_filelist(&argc, argv, &initial_pars, &initial_grid, filelist) != 0)
		return 1;
	if (filelist.used){
		n_files = filelist.nfiles;
		printf("Running %d jobs in pipeline mode\n", n_files);
	} else {
		n_files = 1;
	}
	int pl_gridsize = preload_gridsize(filelist);

	// Setup master map set (one for now, nthreads-1 for general case)
	std::vector<Map> all_maps;

	bool xml2dlg = false;
	int devnum=-1;
	int nr_devices=initial_pars.devices_requested;
	// Get device number to run on
	for (unsigned int i=1; i<argc-1; i+=2)
	{
		if (strcmp("-devnum", argv [i]) == 0)
		{
			unsigned int tempint;
			sscanf(argv [i+1], "%d", &tempint);
			if ((tempint >= 1) && (tempint <= 65536))
				devnum = (unsigned long) tempint-1;
			else
				printf("Warning: value of -devnum argument ignored. Value must be an integer between 1 and 65536.\n");
			break;
		}
	}
	if(devnum>=0){ // user-specified argument on command line has precedence
		if(initial_pars.devices_requested>1)
			printf("Using (single GPU) -devnum specified as command line option.\n");
		nr_devices=1;
	} else devnum=initial_pars.devnum;

	if(nr_devices<1) nr_devices=1;
#ifndef USE_PIPELINE
	if(nr_devices>1) printf("Info: Parallelization over multiple GPUs is only available if OVERLAP=ON is specified when AD-GPU is build.\n");
#endif
	for(unsigned int i=0; i<nr_devices; i++)
		filelist.load_maps_gpu.push_back(true);
	
	// Objects that are arguments of docking_with_gpu
	GpuData cData[nr_devices];
	GpuTempData tData[nr_devices];
	
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

	if(!initial_pars.xml2dlg){
		if(nr_devices==1){
			cData[0].devnum = devnum;
			cData[0].preload_gridsize = pl_gridsize;
			setup_gpu_for_docking(cData[0],tData[0]);
		}
	}

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
		Dockpars   mypars = initial_pars;
		Liganddata myligand_init;
		Gridinfo   mygrid = initial_grid;
		Liganddata myxrayligand;
		std::vector<float> floatgrids;
		SimulationState sim_state;
		int dev_nr = 0;
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
			printf ("(Thread %d is setting up Job #%d)\n",t_id,i_job+1); fflush(stdout);
			if(filelist.used){
				mypars = filelist.mypars[i_job];
				mygrid = filelist.mygrids[i_job];
			}
			if(!mypars.xml2dlg){
#ifdef USE_PIPELINE
				#pragma omp critical
#endif
				{
					if(nr_devices>1){
						dev_nr = mypars.devices_requested-1;
						if(cData[dev_nr].devnum>-2){
							cData[dev_nr].devnum = mypars.devnum;
							cData[dev_nr].preload_gridsize = pl_gridsize;
							setup_gpu_for_docking(cData[dev_nr],tData[dev_nr]);
						}
					}
				}
			}
			start_timer(setup_timer);
			// Load files, read inputs, prepare arrays for docking stage
			if (setup(all_maps, mygrid, floatgrids, mypars, myligand_init, myxrayligand, filelist, i_job, argc, argv) != 0) {
				// If error encountered: Set error flag to 1; Add to count of finished jobs
				// Keep in setup stage rather than moving to launch stage so a different job will be set up
				printf("\n\nError in setup of Job #%d:\n", i_job+1);
				if (filelist.used){
					printf("(   Field file: %s )\n",  filelist.fld_files[i_job].c_str());
					printf("(   Ligand file: %s )\n", filelist.ligand_files[i_job].c_str()); fflush(stdout);
				}
				err[i_job] = 1;
				continue;
			} else { // Successful setup
#ifdef USE_PIPELINE
				#pragma omp atomic update
#endif
				total_setup_time+=seconds_since(setup_timer); // can't count waiting to enter the critical section -AT
				// Copy preloaded maps to GPU
				if(!mypars.xml2dlg){
#ifdef USE_PIPELINE
					#pragma omp critical
#endif
					{
						start_timer(setup_timer);
						if(filelist.preload_maps && filelist.load_maps_gpu[dev_nr]){
							int size_of_one_map = 4*mygrid.size_xyz[0]*mygrid.size_xyz[1]*mygrid.size_xyz[2];
							for (int t=0; t < all_maps.size(); t++){
								copy_map_to_gpu(tData[dev_nr],all_maps,t,size_of_one_map);
							}
							filelist.load_maps_gpu[dev_nr]=false;
						}
						total_setup_time+=seconds_since(setup_timer);
					}
				}
			}
			
			// Starting Docking or loading results
			if(mypars.xml2dlg){
				start_timer(setup_timer);
				// allocating CPU memory for initial populations
				mypars.output_xml = false;
				unsigned int nr_genomes_loaded=0;
				unsigned int nrot;
				sim_state.cpu_populations = read_xml_genomes(mypars.load_xml, mygrid.spacing, nrot);
				if(nrot!=myligand_init.num_of_rotbonds){
					printf("Error: XML genome contains %d rotatable bonds but current ligand has %d.\n",nrot,myligand_init.num_of_rotbonds);
					exit(2);
				}
				double movvec_to_origo[3];
				sim_state.myligand_reference = myligand_init;
				get_movvec_to_origo(&(sim_state.myligand_reference), movvec_to_origo);
				double flex_vec[3];
				for (unsigned int i=0; i<3; i++)
					flex_vec [i] = -mygrid.origo_real_xyz [i];
				move_ligand(&(sim_state.myligand_reference), movvec_to_origo, flex_vec);
				scale_ligand(&(sim_state.myligand_reference), 1.0/mygrid.spacing);
				get_moving_and_unit_vectors(&(sim_state.myligand_reference));
				mypars.pop_size = 1;
				mypars.num_of_runs = sim_state.cpu_populations.size()/GENOTYPE_LENGTH_IN_GLOBMEM;
				// allocating CPU memory for results
				size_t size_energies = mypars.pop_size * mypars.num_of_runs * sizeof(float);
				sim_state.cpu_energies.resize(size_energies);
				// allocating memory in CPU for evaluation counters
				size_t size_evals_of_runs = mypars.num_of_runs*sizeof(int);
				sim_state.cpu_evals_of_runs.resize(size_evals_of_runs);
				memset(sim_state.cpu_evals_of_runs.data(), 0, size_evals_of_runs);
				total_setup_time+=seconds_since(setup_timer);
			} else{
				int error_in_docking;
				// Critical section to only let one thread access GPU at a time
#ifdef USE_PIPELINE
				#pragma omp critical
#endif
				{
					printf("\nRunning Job #%d:\n", i_job+1);
					if (filelist.used){
						printf("   Fields from: %s\n",  filelist.fld_files[i_job].c_str());
						printf("   Ligands from: %s\n", filelist.ligand_files[i_job].c_str()); fflush(stdout);
					}
					// End idling timer, start exec timer
					sim_state.idle_time = seconds_since(idle_timer);
					start_timer(exec_timer);
					// Dock
					error_in_docking = docking_with_gpu(&(mygrid), floatgrids.data(), &(mypars), &(myligand_init), &(myxrayligand), profiler.p[(get_profiles ? i_job : 0)], &argc, argv, sim_state, cData[dev_nr], tData[dev_nr], filelist.preload_maps);
					// End exec timer, start idling timer
					sim_state.exec_time = seconds_since(exec_timer);
					start_timer(idle_timer);
				}

				if (error_in_docking!=0){
					// If error encountered: Set error flag to 1; Add to count of finished jobs
					// Set back to setup stage rather than moving to processing stage so a different job will be set up
					printf("\n\nError in docking_with_gpu, stopped Job #%d.\n",i_job+1);
					err[i_job] = 1;
					continue;
				} else { // Successful run
#ifndef _WIN32
#ifdef USE_PIPELINE
					#pragma omp atomic update
#endif
					total_exec_time+=sim_state.exec_time;
					printf("\nJob #%d took %.3f sec after waiting %.3f sec for setup\n", i_job+1, sim_state.exec_time, sim_state.idle_time);
					if (get_profiles && filelist.used){
						// Detailed timing information to .timing
						profiler.p[i_job].exec_time = sim_state.exec_time;
					}
#endif
				}
			}

			// Post-processing
			printf ("\n(Thread %d is processing Job #%d)\n",t_id,i_job+1); fflush(stdout);
			start_timer(processing_timer);
			process_result(&(mygrid), floatgrids.data(), &(mypars), &(myligand_init), &(myxrayligand), &argc,argv, sim_state);
#ifdef USE_PIPELINE
			#pragma omp atomic update
#endif
			total_processing_time+=seconds_since(processing_timer);
			if(filelist.used){
				// Clean up memory dynamically allocated to not leak
				if(mypars.fldfile) free(mypars.fldfile);
				if(mypars.ligandfile) free(mypars.ligandfile);
				if(mypars.flexresfile) free(mypars.flexresfile);
				if(mypars.xrayligandfile) free(mypars.xrayligandfile);
				if(mypars.resname) free(mypars.resname);
				if(mygrid.grid_file_path) free(mygrid.grid_file_path);
				if(mygrid.receptor_name) free(mygrid.receptor_name);
				if(mygrid.map_base_name) free(mygrid.map_base_name);
			}
		} // end of for loop
		if(!filelist.used){
			// Clean up memory dynamically allocated to not leak
			if(mypars.fldfile) free(mypars.fldfile);
			if(mypars.ligandfile) free(mypars.ligandfile);
			if(mypars.flexresfile) free(mypars.flexresfile);
			if(mypars.xrayligandfile) free(mypars.xrayligandfile);
			if(mypars.resname) free(mypars.resname);
			if(mygrid.grid_file_path) free(mygrid.grid_file_path);
			if(mygrid.receptor_name) free(mygrid.receptor_name);
			if(mygrid.map_base_name) free(mygrid.map_base_name);
		}
	} // end of parallel section

#ifndef _WIN32
	// Total time measurement
	printf("\nRun time of entire job set (%d file%s): %.3f sec", n_files, n_files>1?"s":"", seconds_since(time_start));
#ifdef USE_PIPELINE
	printf("\n%.3f %.3f %.3f\n",total_setup_time,total_processing_time,total_exec_time);
	printf("\nSavings from multithreading: %.3f sec",(total_setup_time+total_processing_time+total_exec_time) - seconds_since(time_start));
	//if (filelist.preload_maps) printf("\nSavings from receptor reuse: %.3f sec * avg_maps_used/n_maps",receptor_reuse_time*n_files);
	printf("\nIdle time of execution thread: %.3f sec",seconds_since(time_start) - total_exec_time);
	if (get_profiles && filelist.used) profiler.write_profiles_to_file(filelist.filename);
#else
	printf("\nProcessing time: %.3f sec",total_processing_time);
#endif
#endif

	if(!initial_pars.xml2dlg)
		for(unsigned int i=0; i<nr_devices; i++)
			finish_gpu_from_docking(cData[i],tData[i]);

	// Alert user to ligands that failed to complete
	int n_errors=0;
	for (int i=0; i<n_files; i++){
		if (err[i]==1){
			if (filelist.used){
				if (n_errors==0) printf("\nWARNING: The following jobs were not successful:");
				printf("\nJob %d: %s\n", i, filelist.ligand_files[i].c_str());
			} else {
				printf("\nThe job was not successful.\n");
			}
			n_errors+=1;
		}
	}
	if (n_errors==0) printf("\nAll jobs ran without errors.\n");

	return 0;
}
