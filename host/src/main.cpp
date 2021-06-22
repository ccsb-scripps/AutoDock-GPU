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
#include "processresult.h"
#include "getparameters.h"

#ifndef TOOLMODE
#include "performdocking.h"
#endif

#include "filelist.hpp"
#include "setup.hpp"
#include "profile.hpp"
#include "simulation_state.hpp"

#ifndef TOOLMODE
#include "GpuData.h"
#endif

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
	// Print version info
	printf("AutoDock-GPU version: %s\n\n", VERSION);
	// Print help screen if no parameters were specified
	// (or if last parameter is "-help"; parameters in
	//  between will be caught in initial_commandpars later)
	if((argc<2) || (argcmp("help", argv[argc-1], 'h')))
		print_options(argv[0]);
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

	// File list setup if -filelist option is on
	FileList filelist;
	Dockpars initial_pars;
	Gridinfo initial_grid;
	if (initial_commandpars(&argc, argv, &initial_pars, &initial_grid, filelist) != 0)
		return 1;
	if (get_filelist(&argc, argv, &initial_pars, &initial_grid, filelist) != 0)
		return 1;
	
	int n_files;
	if (filelist.used){
		n_files = filelist.nfiles;
	} else {
		n_files = 1;
	}
#ifdef USE_PIPELINE
	if(n_files>1) printf("Using %d OpenMP threads\n\n", std::min(omp_get_max_threads(),n_files));
	if(initial_pars.dlg2stdout && (std::min(omp_get_max_threads(),n_files)>1)){
		printf("Note: Parallel pipeline does not currently support dlg\n");
		printf("      to stdout, redirecting to respective file output.\n\n"); fflush(stdout);
		initial_pars.dlg2stdout = false;
		for(unsigned int i=0; i<n_files; i++) // looks dangerous, but n_files>1 is only possible with the filelist
			filelist.mypars[i].dlg2stdout = false;
	}
#endif
#ifdef TOOLMODE
	if(!initial_pars.xml2dlg){
		printf("Error: Code has been compiled without GPU support and only supports xml2dlg mode.\n");
		exit(-1);
	}
	int nr_devices=0;
#else
	int devnum=-1;
	// Get device number to run on
	for (int i=1; i<argc-1; i+=2)
	{
		if (argcmp("filelist", argv[i], 'B'))
			i+=initial_pars.filelist_files-1; // skip ahead in case there are multiple entries here
		
		if (argcmp("xml2dlg", argv[i], 'X'))
			i+=initial_pars.xml_files-1; // skip ahead in case there are multiple entries here
		
		if (argcmp("devnum", argv [i], 'D'))
		{
			if(stricmp(argv[i+1],"all")==0){
				initial_pars.dev_pool.clear();
				initial_pars.dev_pool = get_gpu_pool();
				devnum = -1;
			} else if(stricmp(argv[i+1],"auto")==0){
				initial_pars.dev_pool.clear();
				devnum = -1;
			} else{
				initial_pars.dev_pool.clear();
				unsigned int tempint;
				char* val=argv[i+1];
				bool multiple=false;
				do{
					sscanf(val, "%d", &tempint);
					if ((tempint >= 1) && (tempint <= 65536)){
						devnum = (unsigned long) tempint-1;
					} else{
						printf("Error: Value(s) of --devnum (-D) argument must be an integer between 1 and 65536 (examples: -D 2 or -D 1,3,5).\n");
						exit(-1);
					}
					val=strchr(val,','); // find next entry
					if(val){
						val++; // move past the comma
						multiple=true;
					}
					if(multiple) initial_pars.dev_pool.push_back(devnum);
				} while(val);
				if(multiple) devnum=-1; // needed to automatically load the right values from the pool
			}
		}
	}
	int nr_devices=initial_pars.dev_pool.size();
	if(devnum>=0){ // user-specified argument on command line has precedence
		if(initial_pars.dev_pool.size()>1)
			printf("Using (single GPU) --devnum (-D) specified as command line option.\n\n");
		nr_devices=1;
		initial_pars.dev_pool.clear();
	} else devnum=initial_pars.devnum;

	if(nr_devices<1){
		nr_devices=1;
		initial_pars.dev_pool.clear();
	}
#ifndef USE_PIPELINE
	if(nr_devices>1) printf("Info: Parallelization over multiple GPUs is only available if OVERLAP=ON is specified when AD-GPU is build.\n\n");
#endif
#endif
	if(initial_pars.xml2dlg){
		if(initial_pars.contact_analysis)
			printf("Analyzing ");
		else
			printf("Converting ");
		printf("%d xml file",n_files);
		if(n_files>1) printf("s");
		if(initial_pars.contact_analysis)
			printf(" (contact analysis cutoffs: R=%.1f Å, H=%.1f Å, V=%.1f Å)\n", initial_pars.R_cutoff, initial_pars.H_cutoff, initial_pars.V_cutoff);
		else
			printf(" to dlg\n");
	} else{
		printf("Running %d docking calculation",n_files);
		if(n_files>1){
			printf("s");
			if(nr_devices>1) printf(" on %d devices",std::min(n_files,nr_devices));
		}
		if(initial_pars.contact_analysis)
			printf(" (contact analysis cutoffs: R=%.1f Å, H=%.1f Å, V=%.1f Å)\n", initial_pars.R_cutoff, initial_pars.H_cutoff, initial_pars.V_cutoff);
		else
			printf("\n");
	}
	printf("\n");
	int max_preallocated_gridsize = preallocated_gridsize(filelist);

#ifndef TOOLMODE
	// Objects that are arguments of docking_with_gpu
	GpuData cData[nr_devices];
	GpuTempData tData[nr_devices];
#ifdef USE_PIPELINE
	omp_lock_t gpu_locks[nr_devices];
#endif
	for(int i=0; i<nr_devices; i++){
		filelist.load_maps_gpu.push_back(true);
		if(initial_pars.dev_pool.size()>0)
			cData[i].devnum=initial_pars.dev_pool[i];
		else
			cData[i].devnum=devnum;
		cData[i].preallocated_gridsize = max_preallocated_gridsize;
		tData[i].pMem_fgrids=NULL; // in case setup fails this is needed to make sure we don't segfault trying to deallocate it
		tData[i].device_busy=false;
#ifdef USE_PIPELINE
		omp_init_lock(&gpu_locks[i]);
#endif
	}
#endif
	// Set up run profiles for timing
	bool get_profiles = true; // hard-coded switch to use ALS's job profiler
	Profiler profiler;
	for (int i=0;i<n_files;i++){
		profiler.p.push_back(Profile(i));
		if (!get_profiles) break; // still create 1 if off
	}

	// Error flag for each ligand
	std::vector<int> err(n_files,0);

#ifndef TOOLMODE
	if(!initial_pars.xml2dlg && (nr_devices==1))
		setup_gpu_for_docking(cData[0],tData[0]);
#endif
	total_setup_time+=seconds_since(time_start);
	
	if(initial_pars.xml2dlg && !initial_pars.dlg2stdout){
		if(n_files>100){ // output progress bar
			printf("0%%      20%%       40%%       60%%       80%%     100%%\n");
			printf("---------+---------+---------+---------+---------+\n");
		}
	}

#ifdef USE_PIPELINE
	#pragma omp parallel
	{
		char outbuf[256];
		int t_id = omp_get_thread_num();
#else
	{
#endif
		Dockpars   mypars = initial_pars;
		Liganddata myligand_init;
		Gridinfo*  mygrid = &initial_grid;
		Liganddata myxrayligand;
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
			if(filelist.used){
				mypars = filelist.mypars[i_job];
				mygrid = &filelist.mygrids[mypars.filelist_grid_idx];
			}
			if(mypars.contact_analysis){
				if(filelist.preload_maps){ // use preloaded data for receptor
					mypars.receptor_atoms    = initial_pars.receptor_atoms;
					mypars.nr_receptor_atoms = mypars.receptor_atoms.size();
					mypars.receptor_map      = initial_pars.receptor_map;
					mypars.receptor_map_list = initial_pars.receptor_map_list;
				}
			}
			if(mypars.xml2dlg){
				if(!mypars.dlg2stdout && (n_files>100))
					if((50*(i_job+1)) % n_files < 50){
						printf("*"); fflush(stdout);
					}
			}
#ifndef TOOLMODE
			else
			{
#ifdef USE_PIPELINE
				printf ("(Thread %d is setting up Job #%d)\n",t_id,i_job+1); fflush(stdout);
				#pragma omp critical
#endif
				{
					if(nr_devices>1){
						if(mypars.dev_pool_nr<0){ // assign next available GPU
							dev_nr=-1;
							for(unsigned int i=0; i<nr_devices; i++){
								if(!tData[i].device_busy){ // found an available GPU
									dev_nr=i;
									break;
								}
							}
							// if no GPU is available, assign one based on the job nr
							if(dev_nr<0) dev_nr = i_job % nr_devices;
						} else dev_nr = mypars.dev_pool_nr; // this is set when specific GPU is requested
						tData[dev_nr].device_busy = true;
						setup_gpu_for_docking(cData[dev_nr],tData[dev_nr]);
						fflush(stdout);
					}
				}
			}
#endif
			start_timer(setup_timer);
			// Load files, read inputs, prepare arrays for docking stage
			if (setup(mygrid, &mypars, myligand_init, myxrayligand, filelist, i_job, argc, argv) != 0) {
				// If error encountered: Set error flag to 1; Add to count of finished jobs
				// Keep in setup stage rather than moving to launch stage so a different job will be set up
#ifdef USE_PIPELINE
				#pragma omp critical
#endif
				{
					printf("\nError in setup of Job #%d", i_job+1);
					if (filelist.used){
						printf(":\n");
						printf("(   Grid map file: %s )\n",  mypars.fldfile);
						printf("(   Ligand file: %s )\n", mypars.ligandfile); fflush(stdout);
						if(mypars.flexresfile)
							printf("(   Flexible residue: %s )\n", mypars.flexresfile);
						fflush(stdout);
					} else printf("\n");
				}
				err[i_job] = 1;
				continue;
			} else { // Successful setup
#ifdef USE_PIPELINE
				#pragma omp atomic update
#endif
				total_setup_time+=seconds_since(setup_timer);
			}
			
			// Starting Docking or loading results
			if(mypars.xml2dlg){
				start_timer(setup_timer);
				// allocating CPU memory for initial populations
				mypars.output_xml = false;
				int nrot;
				sim_state.cpu_populations = read_xml_genomes(mypars.load_xml, mygrid->spacing, nrot, true);
				if(nrot!=myligand_init.num_of_rotbonds){
					printf("\nError: XML genome contains %d rotatable bonds but current ligand has %d.\n",nrot,myligand_init.num_of_rotbonds);
					exit(2);
				}
				double movvec_to_origo[3];
				sim_state.myligand_reference = myligand_init;
				get_movvec_to_origo(&(sim_state.myligand_reference), movvec_to_origo);
				double flex_vec[3];
				for (unsigned int i=0; i<3; i++)
					flex_vec [i] = -mygrid->origo_real_xyz [i];
				move_ligand(&(sim_state.myligand_reference), movvec_to_origo, flex_vec);
				scale_ligand(&(sim_state.myligand_reference), 1.0/mygrid->spacing);
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
				sim_state.idle_time = 0.0;
				sim_state.exec_time = 0.0;

			}
#ifndef TOOLMODE
			else
			{
				int error_in_docking;
				// Lock to only let one thread access a given GPU at a time
				std::string* output = NULL;
#ifdef USE_PIPELINE
				omp_set_lock(&gpu_locks[dev_nr]);
				if(nr_devices>1) output = new std::string;
#endif
				para_printf("\nRunning Job #%d", i_job+1);
				if (filelist.used){
					para_printf(":\n");
					para_printf("    Device: %s\n", tData[dev_nr].device_name);
					para_printf("    Grid map file: %s\n",  mypars.fldfile);
					para_printf("    Ligand file: %s\n", mypars.ligandfile); fflush(stdout);
					if(mypars.flexresfile)
						para_printf("    Flexible residue: %s\n", mypars.flexresfile);
					fflush(stdout);
				} else para_printf("\n");
				// End idling timer, start exec timer
				sim_state.idle_time = seconds_since(idle_timer);
				start_timer(exec_timer);
				// Dock
				error_in_docking = docking_with_gpu(mygrid, &(mypars), &(myligand_init), &(myxrayligand), profiler.p[(get_profiles ? i_job : 0)], &argc, argv, sim_state, cData[dev_nr], tData[dev_nr], output);
				// End exec timer, start idling timer
				sim_state.exec_time = seconds_since(exec_timer);
				start_timer(idle_timer);
#ifdef USE_PIPELINE
				omp_unset_lock(&gpu_locks[dev_nr]);
#endif
				if (error_in_docking!=0){
					// If error encountered: Set error flag to 1; Add to count of finished jobs
					// Set back to setup stage rather than moving to processing stage so a different job will be set up
					para_printf("\nError in docking_with_gpu, stopped Job #%d.\n",i_job+1);
					err[i_job] = 1;
					continue;
				} else { // Successful run
#ifndef _WIN32
#ifdef USE_PIPELINE
					#pragma omp atomic update
#endif
					total_exec_time+=sim_state.exec_time;
					para_printf("\nJob #%d took %.3f sec after waiting %.3f sec for setup\n\n", i_job+1, sim_state.exec_time, sim_state.idle_time);
					if (get_profiles && filelist.used){
						// Detailed timing information to .timing
						profiler.p[i_job].exec_time = sim_state.exec_time;
					}
#endif
				}
#ifdef USE_PIPELINE
				if(nr_devices>1){
					#pragma omp critical
					{
						printf("%s", output->c_str());
						fflush(stdout);
					}
					delete output;
				}
#endif
			}
#endif
			// Post-processing
#ifdef USE_PIPELINE
			if(!mypars.xml2dlg){
#ifndef TOOLMODE
				if(nr_devices>1) tData[dev_nr].device_busy = false;
#endif
				printf ("(Thread %d is processing Job #%d)\n",t_id,i_job+1); fflush(stdout);
			}
#endif
			start_timer(processing_timer);
			process_result(mygrid, &(mypars), &(myligand_init), &(myxrayligand), &argc,argv, sim_state);
#ifdef USE_PIPELINE
			#pragma omp atomic update
#endif
			total_processing_time+=seconds_since(processing_timer);
			if(filelist.used){
				// Clean up memory dynamically allocated to not leak
				mypars.receptor_atoms.clear();
				if(mypars.fldfile) free(mypars.fldfile);
				if(mypars.ligandfile) free(mypars.ligandfile);
				if(mypars.flexresfile) free(mypars.flexresfile);
				if(mypars.xrayligandfile) free(mypars.xrayligandfile);
				if(mypars.resname) free(mypars.resname);
			}
		} // end of for loop
		if(!filelist.used){
			// Clean up memory dynamically allocated to not leak
			mypars.receptor_atoms.clear();
			if(mypars.fldfile) free(mypars.fldfile);
			if(mypars.ligandfile) free(mypars.ligandfile);
			if(mypars.flexresfile) free(mypars.flexresfile);
			if(mypars.xrayligandfile) free(mypars.xrayligandfile);
			if(mypars.resname) free(mypars.resname);
		}
	} // end of parallel section
	if(initial_pars.xml2dlg && !initial_pars.dlg2stdout && (n_files>100)) printf("\n\n"); // finish progress bar
	
#ifndef _WIN32
	// Total time measurement
	printf("Run time of entire job set (%d file%s): %.3f sec\n", n_files, n_files>1?"s":"", seconds_since(time_start));
#ifdef USE_PIPELINE
	if(n_files>1){
		printf("Savings from multithreading: %.3f sec\n",(total_setup_time+total_processing_time+total_exec_time) - seconds_since(time_start));
		if(!initial_pars.xml2dlg) // in xml2dlg mode, there's only "idle time" (aka overlapped processing)
			printf("Idle time of execution thread: %.3f sec\n",seconds_since(time_start) - total_exec_time);
		if (get_profiles && filelist.used && !initial_pars.xml2dlg) // output profile with filelist name or dpf file name (depending on what is available)
			profiler.write_profiles_to_file((filelist.filename!=NULL) ? filelist.filename : initial_pars.dpffile);
	} else printf("Processing time: %.3f sec\n",total_processing_time);
#else
	printf("Processing time: %.3f sec\n",total_processing_time);
#endif
#endif
#ifndef TOOLMODE
	for(int i=0; i<nr_devices; i++){
#ifdef USE_PIPELINE
		omp_destroy_lock(&gpu_locks[i]);
#endif
		if(!initial_pars.xml2dlg)
			finish_gpu_from_docking(cData[i],tData[i]);
	}
#endif
	// Alert user to ligands that failed to complete
	int n_errors=0;
	for (int i=0; i<n_files; i++){
		if (err[i]==1){
			if (filelist.used){
				if (n_errors==0) printf("\nWarning: The following jobs were not successful:\n");
				printf("         Job %d: %s\n", i, filelist.ligand_files[i].c_str());
			} else {
				printf("\nThe job was not successful.\n");
			}
			n_errors++;
		}
	}
	if (n_errors==0) printf("\nAll jobs ran without errors.\n");

	return 0;
}
