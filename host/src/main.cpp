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
	timeval time_start, idle_timer, setup_timer, exec_timer;
	start_timer(time_start);
	start_timer(idle_timer);
#endif
	double total_setup_time=0;
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

	// Objects that are arguments of docking_with_gpu
        Dockpars   mypars[nqueues];
        Liganddata myligand_init[nqueues];
        Gridinfo   mygrid[nqueues];
        Liganddata myxrayligand[nqueues];
        std::vector<float> floatgrids[nqueues];

	// Set up run profiles for timing
	bool get_profiles = false; // hard-coded switch to use ALS's job profiler
	std::vector<Profile> profiles;
	for (int i=0;i<n_files;i++){
		profiles.push_back(Profile(i));
		if (!get_profiles) break; // still create 1 if off
	}

	// Print version info
	printf("\nAutoDock-GPU version: %s\n", VERSION);

	bool finished_all=false;
	std::vector<int> job_in_queue(nqueues,-1);
	std::vector<bool> ready_to_launch(nqueues,false);
	int n_finished_jobs=0;
	int next_job_to_setup=0;
	int err = 0;
#ifdef USE_PIPELINE
	#pragma omp parallel
	{
	int t_id = omp_get_thread_num();
#else
	{
	int t_id = 0;
#endif
	while (!finished_all && err==0){
		if(t_id!=execution_thread || nthreads==1) { // This thread handles setup
			if (!ready_to_launch[t_id] && next_job_to_setup<n_files){ // If not ready for execution
				int i_job;
				// Grab the next job in atomic capture so two threads don't set up the same job
#ifdef USE_PIPELINE
				#pragma omp atomic capture
#endif
				{ i_job=next_job_to_setup; next_job_to_setup++; }

				// Setup the next file in the queue
				if (i_job<n_files) {
					printf ("\n(Thread %d is setting up job %d)",t_id,i_job); fflush(stdout);
					job_in_queue[t_id]=i_job;
					start_timer(setup_timer);
					// Load files, read inputs, prepare arrays for docking stage
					if (setup(mygrid[t_id], floatgrids[t_id], mypars[t_id], myligand_init[t_id], myxrayligand[t_id], filelist, i_job, argc, argv) != 0) {
						printf("\n\nError in setup of Job #%d:", i_job);
                		                printf("\n(   Field file: %s )",  filelist.fld_files[i_job].c_str());
		                                printf("\n(   Ligand file: %s )", filelist.ligand_files[i_job].c_str()); fflush(stdout);
						err = 1;
					}
#ifdef USE_PIPELINE
					#pragma omp atomic update
#endif
					total_setup_time+=seconds_since(setup_timer);
					ready_to_launch[t_id]=true; // Indicate this queue is ready for use
				}
			} else { // Job is ready to go, just waiting for the GPU to free up
				// Wait
			}
		}

		if(t_id==execution_thread) { // This thread handles the GPU
			// Check if there is a job ready to launch
			int i_queue=-1;
			int i_job = -1;
			for (int i=0;i<ready_to_launch.size();i++){
				if (ready_to_launch[i]){
					i_queue=i;
					i_job=job_in_queue[i_queue];
					break;
				}
			}
			if (i_queue>=0){ 
				idle_time = seconds_since(idle_timer);
				start_timer(exec_timer);
				printf("\nRunning Job #%d: ", i_job);
                                printf("\n   Fields from: %s",  filelist.fld_files[i_job].c_str());
                                printf("\n   Ligands from: %s", filelist.ligand_files[i_job].c_str()); fflush(stdout);
				// Starting Docking
				if (docking_with_gpu(&(mygrid[i_queue]), floatgrids[i_queue].data(), &(mypars[i_queue]), &(myligand_init[i_queue]), &(myxrayligand[i_queue]), profiles[(get_profiles ? i_job : 0)], &argc, argv ) != 0)
					{printf("\n\nError in docking_with_gpu, stopped job."); err = 1;}


				ready_to_launch[i_queue]=false; // Indicate this queue is ready for a new setup
                                n_finished_jobs++;
				if (n_finished_jobs==n_files) finished_all=true;

#ifndef _WIN32
				double exec_time = seconds_since(exec_timer);
				total_exec_time+=exec_time;
				printf("\nJob took %.3f sec after waiting %.3f sec for setup", exec_time, idle_time);
				start_timer(idle_timer);
	                        // Append time information to .dlg file
	                        char report_file_name[256];
	                        strcpy(report_file_name, mypars[i_queue].resname);
	                        strcat(report_file_name, ".dlg");
	                        FILE* fp = fopen(report_file_name, "a");
	                        fprintf(fp, "\n\n");
				fprintf(fp, "\nRun time %.3f sec", exec_time);
				fprintf(fp, "\nIdle time %.3f sec\n", idle_time);
	                        fclose(fp);

				if (get_profiles){
	                        	// Detailed timing information to .timing
	                        	profiles[i_job].exec_time = exec_time;
	                        	profiles[i_job].write_to_file(filelist.filename);
				}
#endif
			} else { // No job is ready to launch
				// Wait
			}
		} 
	} // end of while loop
	} // end of parallel section

#ifndef _WIN32
	// Total time measurement
	printf("\nRun time of entire job set (%d files): %.3f sec", n_files, seconds_since(time_start));
	printf("\nSavings from pipelining: %.3f sec",(total_setup_time+total_exec_time) - seconds_since(time_start));
#endif

	return 0;
}
