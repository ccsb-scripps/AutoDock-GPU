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




#ifndef _WIN32
#define STRINGIZE2(s) #s
#define STRINGIZE(s)	STRINGIZE2(s)
#define KRNL_FILE STRINGIZE(KRNL_SOURCE)
#define KRNL_FOLDER STRINGIZE(KRNL_DIRECTORY)
#define KRNL_COMMON STRINGIZE(KCMN_DIRECTORY)
#define KRNL1 STRINGIZE(K1)
#define KRNL2 STRINGIZE(K2)
#define KRNL3 STRINGIZE(K3)
#define KRNL4 STRINGIZE(K4)
#define KRNL5 STRINGIZE(K5)
#define KRNL6 STRINGIZE(K6)
#define KRNL7 STRINGIZE(K7)

#else
#define KRNL_FILE KRNL_SOURCE
#define KRNL_FOLDER KRNL_DIRECTORY
#define KRNL_COMMON KCMN_DIRECTORY
#define KRNL1 K1
#define KRNL2 K2
#define KRNL3 K3
#define KRNL4 K4
#define KRNL5 K5
#define KRNL6 K6
#define KRNL7 K7
#endif

#define INC " -I " KRNL_FOLDER " -I " KRNL_COMMON

#if defined (N1WI)
	#define KNWI " -DN1WI "
#elif defined (N2WI)
	#define KNWI " -DN2WI "
#elif defined (N4WI)
	#define KNWI " -DN4WI "
#elif defined (N8WI)
	#define KNWI " -DN8WI "
#elif defined (N16WI)
	#define KNWI " -DN16WI "
#elif defined (N32WI)
	#define KNWI " -DN32WI "
#elif defined (N64WI)
	#define KNWI " -DN64WI "
#elif defined (N128WI)
	#define KNWI " -DN128WI "
#elif defined (N256WI)
		#define KNWI " -DN256WI "
#else
	#define KNWI	" -DN64WI "
#endif

#if defined (REPRO)
	#define REP " -DREPRO "
#else
	#define REP " "
#endif


#ifdef __APPLE__
	#define KGDB_GPU	" -g -cl-opt-disable "
#else
	#define KGDB_GPU	" -g -O0 -Werror -cl-opt-disable "
#endif
#define KGDB_CPU	" -g3 -Werror -cl-opt-disable "
// Might work in some (Intel) devices " -g -s " KRNL_FILE

#if defined (DOCK_DEBUG)
	#if defined (CPU_DEVICE)
		#define KGDB KGDB_CPU
	#elif defined (GPU_DEVICE)
		#define KGDB KGDB_GPU
	#endif
#else
	#define KGDB " -cl-mad-enable"
#endif


#define OPT_PROG INC KNWI REP KGDB

#include <Kokkos_Core.hpp>

#include "performdocking.h"
#include "stringify.h"
#include "correct_grad_axisangle.h"

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
// =======================================================================
// OpenCL Host Setup
// =======================================================================
	cl_platform_id*  platform_id;
	cl_device_id*    device_ids;
	cl_device_id     device_id;
	cl_context       context;
	cl_command_queue command_queue;
	cl_program       program;

#ifdef _WIN32
	const char *filename = KRNL_FILE;
	printf("\n%-40s %-40s\n", "Kernel source file: ", filename);  fflush(stdout);
#else
	printf("\n%-40s %-40s\n", "Kernel source used for development: ", "./device/calcenergy.cl");  fflush(stdout);
	printf(  "%-40s %-40s\n", "Kernel string used for building: ",    "./host/inc/stringify.h");  fflush(stdout);
#endif

	const char* options_program = OPT_PROG;
	printf("%-40s %-40s\n", "Kernel compilation flags: ", options_program); fflush(stdout);

	cl_kernel kernel1; const char *name_k1 = KRNL1;
	size_t kernel1_gxsize, kernel1_lxsize;

	cl_kernel kernel2; const char *name_k2 = KRNL2;
	size_t kernel2_gxsize, kernel2_lxsize;

	cl_kernel kernel3; const char *name_k3 = KRNL3;
	size_t kernel3_gxsize, kernel3_lxsize;

	cl_kernel kernel4; const char *name_k4 = KRNL4;
	size_t kernel4_gxsize, kernel4_lxsize;

	cl_kernel kernel5; const char *name_k5 = KRNL5;
	size_t kernel5_gxsize, kernel5_lxsize;

	cl_kernel kernel6; const char *name_k6 = KRNL6;
	size_t kernel6_gxsize, kernel6_lxsize;

	cl_kernel kernel7; const char *name_k7 = KRNL7;
	size_t kernel7_gxsize, kernel7_lxsize;

	cl_uint platformCount;
	cl_uint deviceCount;

	// Times
	cl_ulong time_start_kernel;
	cl_ulong time_end_kernel;

	// Get all available platforms
	if (getPlatforms(&platform_id,&platformCount) != 0) return 1;

	// Get all devices of first platform
	if (getDevices(platform_id[0],platformCount,&device_ids,&deviceCount) != 0) return 1;
	if (mypars->devnum>=deviceCount)
	{
		printf("Warning: user specified OpenCL device number does not exist, using first device.\n");
		mypars->devnum=0;
	}
	device_id=device_ids[mypars->devnum];

	// Create context from first platform
	if (createContext(platform_id[0],1,&device_id,&context) != 0) return 1;

	// Create command queue for first device
	if (createCommandQueue(context,device_id,&command_queue) != 0) return 1;

	// Create program from source 
#ifdef _WIN32
	if (ImportSource(filename, name_k1, &device_id, context, options_program, &kernel1) != 0) return 1;
	if (ImportSource(filename, name_k2, &device_id, context, options_program, &kernel2) != 0) return 1;
	if (ImportSource(filename, name_k3, &device_id, context, options_program, &kernel3) != 0) return 1;
	if (ImportSource(filename, name_k4, &device_id, context, options_program, &kernel4) != 0) return 1;
	if (ImportSource(filename, name_k5, &device_id, context, options_program, &kernel5) != 0) return 1;
	if (ImportSource(filename, name_k6, &device_id, context, options_program, &kernel6) != 0) return 1;
	if (ImportSource(filename, name_k7, &device_id, context, options_program, &kernel7) != 0) return 1;
#else
	if (ImportSourceToProgram(calcenergy_ocl, &device_id, context, &program, options_program) != 0) return 1;
#endif

	// Create kernels
	if (createKernel(&device_id, &program, name_k1, &kernel1) != 0) return 1;
	if (createKernel(&device_id, &program, name_k2, &kernel2) != 0) return 1;
	if (createKernel(&device_id, &program, name_k3, &kernel3) != 0) return 1;
	if (createKernel(&device_id, &program, name_k4, &kernel4) != 0) return 1;
	if (createKernel(&device_id, &program, name_k5, &kernel5) != 0) return 1;
	if (createKernel(&device_id, &program, name_k6, &kernel6) != 0) return 1;
	if (createKernel(&device_id, &program, name_k7, &kernel7) != 0) return 1;

// End of OpenCL Host Setup
// =======================================================================

	Liganddata myligand_reference;

	// TEMPORARY - ALS
	float* cpu_energies_kokkos;
	int* cpu_new_entities_kokkos;
	unsigned int* cpu_prng_kokkos;
	float* cpu_conforms_kokkos;
	float* cpu_Nenergies_kokkos;
	float* cpu_Nconforms_kokkos;

	float* cpu_init_populations;
	float* cpu_final_populations;
	float* cpu_energies;
	Ligandresult* cpu_result_ligands;
	unsigned int* cpu_prng_seeds;
	int* cpu_evals_of_runs;
	float* cpu_ref_ori_angles;

	size_t size_floatgrids;
	size_t size_populations;
	size_t size_energies;
	size_t size_prng_seeds;
	size_t size_evals_of_new_entities;
	size_t size_evals_of_runs;

	int threadsPerBlock;
	int blocksPerGridForEachEntity;
	int blocksPerGridForEachRun;

	unsigned long run_cnt;	/* int run_cnt; */
	int generation_cnt;
	int i;
	double progress;

	int curr_progress_cnt;
	int new_progress_cnt;

	clock_t clock_start_docking;
	clock_t	clock_stop_docking;
	clock_t clock_stop_program_before_clustering;

	//setting number of blocks and threads
	threadsPerBlock = NUM_OF_THREADS_PER_BLOCK;
	blocksPerGridForEachEntity = mypars->pop_size * mypars->num_of_runs;
	blocksPerGridForEachRun = mypars->num_of_runs;

	//allocating CPU memory for initial populations
	size_populations = mypars->num_of_runs * mypars->pop_size * GENOTYPE_LENGTH_IN_GLOBMEM*sizeof(float);
	cpu_init_populations = (float*) malloc(size_populations);
	memset(cpu_init_populations, 0, size_populations);

	//allocating CPU memory for results
	size_energies = mypars->pop_size * mypars->num_of_runs * sizeof(float);
	cpu_energies = (float*) malloc(size_energies);
	cpu_result_ligands = (Ligandresult*) malloc(sizeof(Ligandresult)*(mypars->num_of_runs));
	cpu_final_populations = cpu_init_populations;

	//allocating memory in CPU for reference orientation angles
	cpu_ref_ori_angles = (float*) malloc(mypars->num_of_runs*3*sizeof(float));

	//generating initial populations and random orientation angles of reference ligand
	//(ligand will be moved to origo and scaled as well)
	myligand_reference = *myligand_init;
	gen_initpop_and_reflig(mypars, cpu_init_populations, cpu_ref_ori_angles, &myligand_reference, mygrid);

	//allocating memory in CPU for pseudorandom number generator seeds and
	//generating them (seed for each thread during GA)
	size_prng_seeds = blocksPerGridForEachEntity * threadsPerBlock * sizeof(unsigned int);
	cpu_prng_seeds = (unsigned int*) malloc(size_prng_seeds);

	//genseed(time(NULL));	//initializing seed generator
	genseed(0u);    // TEMPORARY: removing randomness for consistent debugging - ALS

	for (i=0; i<blocksPerGridForEachEntity*threadsPerBlock; i++)
#if defined (REPRO)
		cpu_prng_seeds[i] = 1u;
#else
		cpu_prng_seeds[i] = genseed(0u);
#endif

	//Set size for evals_of_new_entities
	size_evals_of_new_entities = mypars->pop_size*mypars->num_of_runs*sizeof(int);

	// TEMPORARY - ALS
	cpu_energies_kokkos = (float*) malloc(size_energies);
	cpu_new_entities_kokkos = (int*) malloc(size_evals_of_new_entities);
	cpu_prng_kokkos = (unsigned int*) malloc(size_prng_seeds);
	cpu_conforms_kokkos = (float *) malloc(size_populations);
	cpu_Nenergies_kokkos = (float*) malloc(size_energies);
	cpu_Nconforms_kokkos = (float *) malloc(size_populations);

	//allocating memory in CPU for evaluation counters
	size_evals_of_runs = mypars->num_of_runs*sizeof(int);
	cpu_evals_of_runs = (int*) malloc(size_evals_of_runs);
	memset(cpu_evals_of_runs, 0, size_evals_of_runs);

	//preparing the constant data fields for the GPU
	// ----------------------------------------------------------------------
	// The original function does CUDA calls initializing const Kernel data.
	// We create a struct to hold those constants
	// and return them <here> (<here> = where prepare_const_fields_for_gpu() is called),
	// so we can send them to Kernels from <here>, instead of from calcenergy.cpp as originally.
	// ----------------------------------------------------------------------
	// Constant struct

/*
	kernelconstant KerConst;

	if (prepare_const_fields_for_gpu(&myligand_reference, mypars, cpu_ref_ori_angles, &KerConst) == 1)
		return 1;
*/

	kernelconstant_interintra	KerConst_interintra;
	kernelconstant_intracontrib	KerConst_intracontrib;
	kernelconstant_intra		KerConst_intra;
	kernelconstant_rotlist		KerConst_rotlist;
	kernelconstant_conform		KerConst_conform;
	kernelconstant_grads		KerConst_grads;

	if (prepare_const_fields_for_gpu(&myligand_reference, mypars, cpu_ref_ori_angles, 
					 &KerConst_interintra, 
					 &KerConst_intracontrib, 
					 &KerConst_intra, 
					 &KerConst_rotlist, 
					 &KerConst_conform,
					 &KerConst_grads) == 1) {
		return 1;
	}

	// Constant data holding struct data
	// Created because structs containing array
	// are not supported as OpenCL kernel args
/*
  	cl_mem mem_atom_charges_const;
	cl_mem mem_atom_types_const;
  	cl_mem mem_intraE_contributors_const;
  	cl_mem mem_reqm_const;
  	cl_mem mem_reqm_hbond_const;
  	cl_mem mem_atom1_types_reqm_const;
  	cl_mem mem_atom2_types_reqm_const;
  	cl_mem mem_VWpars_AC_const;
  	cl_mem mem_VWpars_BD_const;
  	cl_mem mem_dspars_S_const;
  	cl_mem mem_dspars_V_const;
  	cl_mem mem_rotlist_const;
  	cl_mem mem_ref_coords_x_const;
  	cl_mem mem_ref_coords_y_const;
  	cl_mem mem_ref_coords_z_const;
  	cl_mem mem_rotbonds_moving_vectors_const;
  	cl_mem mem_rotbonds_unit_vectors_const;
  	cl_mem mem_ref_orientation_quats_const;
*/

	cl_mem mem_interintra_const;
	cl_mem mem_intracontrib_const;
	cl_mem mem_intra_const;
	cl_mem mem_rotlist_const;
	cl_mem mem_conform_const;

	size_t sz_interintra_const	= MAX_NUM_OF_ATOMS*sizeof(float) + 
					  MAX_NUM_OF_ATOMS*sizeof(char);

	size_t sz_intracontrib_const	= 3*MAX_INTRAE_CONTRIBUTORS*sizeof(char);

	size_t sz_intra_const		= ATYPE_NUM*sizeof(float) + 
					  ATYPE_NUM*sizeof(float) + 
					  ATYPE_NUM*sizeof(unsigned int) + 
					  ATYPE_NUM*sizeof(unsigned int) + 
				          MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) + 
					  MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) + 
					  MAX_NUM_OF_ATYPES*sizeof(float) + 
					  MAX_NUM_OF_ATYPES*sizeof(float);

	size_t sz_rotlist_const		= MAX_NUM_OF_ROTATIONS*sizeof(int);

	size_t sz_conform_const		= 3*MAX_NUM_OF_ATOMS*sizeof(float) + 
					  3*MAX_NUM_OF_ROTBONDS*sizeof(float) + 
					  3*MAX_NUM_OF_ROTBONDS*sizeof(float) + 
					  4*MAX_NUM_OF_RUNS*sizeof(float);

  	cl_mem mem_rotbonds_const;
  	cl_mem mem_rotbonds_atoms_const;
  	cl_mem mem_num_rotating_atoms_per_rotbond_const;

	// Constant data for correcting axisangle gradients
	cl_mem mem_angle_const;
	cl_mem mem_dependence_on_theta_const;
	cl_mem mem_dependence_on_rotangle_const;

	// These constants are allocated in global memory since
	// there is a limited number of constants that can be passed
	// as arguments to kernel
/*
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*sizeof(float),                         &mem_atom_charges_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*sizeof(char),                          &mem_atom_types_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,3*MAX_INTRAE_CONTRIBUTORS*sizeof(char),                 &mem_intraE_contributors_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,ATYPE_NUM*sizeof(float),				    &mem_reqm_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,ATYPE_NUM*sizeof(float),				    &mem_reqm_hbond_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,ATYPE_NUM*sizeof(unsigned int),			    &mem_atom1_types_reqm_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,ATYPE_NUM*sizeof(unsigned int),                         &mem_atom2_types_reqm_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float),      &mem_VWpars_AC_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float),      &mem_VWpars_BD_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATYPES*sizeof(float),                        &mem_dspars_S_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATYPES*sizeof(float),                        &mem_dspars_V_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ROTATIONS*sizeof(int),                       &mem_rotlist_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*sizeof(float),                         &mem_ref_coords_x_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*sizeof(float),                         &mem_ref_coords_y_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*sizeof(float),                         &mem_ref_coords_z_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,3*MAX_NUM_OF_ROTBONDS*sizeof(float),                    &mem_rotbonds_moving_vectors_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,3*MAX_NUM_OF_ROTBONDS*sizeof(float),                    &mem_rotbonds_unit_vectors_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,4*MAX_NUM_OF_RUNS*sizeof(float),                        &mem_ref_orientation_quats_const);
*/

	mallocBufferObject(context,CL_MEM_READ_ONLY,sz_interintra_const,	&mem_interintra_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,sz_intracontrib_const,   	&mem_intracontrib_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,sz_intra_const,             &mem_intra_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,sz_rotlist_const,           &mem_rotlist_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,sz_conform_const,           &mem_conform_const);

	mallocBufferObject(context,CL_MEM_READ_ONLY,2*MAX_NUM_OF_ROTBONDS*sizeof(int),                      &mem_rotbonds_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS*sizeof(int),       &mem_rotbonds_atoms_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ROTBONDS*sizeof(int),      		    &mem_num_rotating_atoms_per_rotbond_const);

	mallocBufferObject(context,CL_MEM_READ_ONLY,1000*sizeof(float),&mem_angle_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,1000*sizeof(float),&mem_dependence_on_theta_const);
	mallocBufferObject(context,CL_MEM_READ_ONLY,1000*sizeof(float),&mem_dependence_on_rotangle_const);
   	
/*
	memcopyBufferObjectToDevice(command_queue,mem_atom_charges_const,			false,  &KerConst.atom_charges_const,           MAX_NUM_OF_ATOMS*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_atom_types_const,           		false,	&KerConst.atom_types_const,             MAX_NUM_OF_ATOMS*sizeof(char));
	memcopyBufferObjectToDevice(command_queue,mem_intraE_contributors_const,  		false,  &KerConst.intraE_contributors_const,    3*MAX_INTRAE_CONTRIBUTORS*sizeof(char));
	memcopyBufferObjectToDevice(command_queue,mem_reqm_const,         			false,  &KerConst.reqm_const,           	ATYPE_NUM*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_reqm_hbond_const,         		false,	&KerConst.reqm_hbond_const,           	ATYPE_NUM*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_atom1_types_reqm_const,         		false,  &KerConst.atom1_types_reqm_const,       ATYPE_NUM*sizeof(unsigned int));
	memcopyBufferObjectToDevice(command_queue,mem_atom2_types_reqm_const,         		false,	&KerConst.atom2_types_reqm_const,       ATYPE_NUM*sizeof(unsigned int));
	memcopyBufferObjectToDevice(command_queue,mem_VWpars_AC_const,            		false,	&KerConst.VWpars_AC_const,              MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_VWpars_BD_const,            		false,	&KerConst.VWpars_BD_const,              MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_dspars_S_const,             		false,	&KerConst.dspars_S_const,               MAX_NUM_OF_ATYPES*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_dspars_V_const,             		false,	&KerConst.dspars_V_const,               MAX_NUM_OF_ATYPES*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_rotlist_const,              		false,	&KerConst.rotlist_const,                MAX_NUM_OF_ROTATIONS*sizeof(int));
	memcopyBufferObjectToDevice(command_queue,mem_ref_coords_x_const,         		false,	&KerConst.ref_coords_x_const,           MAX_NUM_OF_ATOMS*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_ref_coords_y_const,         		false,	&KerConst.ref_coords_y_const,           MAX_NUM_OF_ATOMS*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_ref_coords_z_const,         		false,	&KerConst.ref_coords_z_const,           MAX_NUM_OF_ATOMS*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_rotbonds_moving_vectors_const,		false,	&KerConst.rotbonds_moving_vectors_const,3*MAX_NUM_OF_ROTBONDS*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_rotbonds_unit_vectors_const,  		false,	&KerConst.rotbonds_unit_vectors_const,  3*MAX_NUM_OF_ROTBONDS*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_ref_orientation_quats_const, 		false,	&KerConst.ref_orientation_quats_const,  4*MAX_NUM_OF_RUNS*sizeof(float));
*/
	
	memcopyBufferObjectToDevice(command_queue,mem_interintra_const,		false,	&KerConst_interintra,	sz_interintra_const);
	memcopyBufferObjectToDevice(command_queue,mem_intracontrib_const,      	false,	&KerConst_intracontrib, sz_intracontrib_const);
	memcopyBufferObjectToDevice(command_queue,mem_intra_const,       	false,	&KerConst_intra,       	sz_intra_const);
	memcopyBufferObjectToDevice(command_queue,mem_rotlist_const,     	false,	&KerConst_rotlist,     	sz_rotlist_const);
	memcopyBufferObjectToDevice(command_queue,mem_conform_const,     	false,	&KerConst_conform,     	sz_conform_const);

	memcopyBufferObjectToDevice(command_queue,mem_rotbonds_const,  	      			false,	&KerConst_grads.rotbonds,  			2*MAX_NUM_OF_ROTBONDS*sizeof(int));
	memcopyBufferObjectToDevice(command_queue,mem_rotbonds_atoms_const,  	      		false,	&KerConst_grads.rotbonds_atoms,  		MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS*sizeof(int));
	memcopyBufferObjectToDevice(command_queue,mem_num_rotating_atoms_per_rotbond_const, 	false,	&KerConst_grads.num_rotating_atoms_per_rotbond, MAX_NUM_OF_ROTBONDS*sizeof(int));

  	memcopyBufferObjectToDevice(command_queue,mem_angle_const,				false,  &angle,           		1000*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_dependence_on_theta_const,		false,  &dependence_on_theta,           1000*sizeof(float));
	memcopyBufferObjectToDevice(command_queue,mem_dependence_on_rotangle_const,		false,  &dependence_on_rotangle,        1000*sizeof(float));
	// ----------------------------------------------------------------------

 	//allocating GPU memory for populations, floatgirds,
	//energies, evaluation counters and random number generator states
	size_floatgrids = 4 * (sizeof(float)) * (mygrid->num_of_atypes+2) * (mygrid->size_xyz[0]) * (mygrid->size_xyz[1]) * (mygrid->size_xyz[2]);

	cl_mem mem_dockpars_fgrids;
	cl_mem mem_dockpars_conformations_current;
	cl_mem mem_dockpars_energies_current;
	cl_mem mem_dockpars_conformations_next;
	cl_mem mem_dockpars_energies_next;
	cl_mem mem_dockpars_evals_of_new_entities;
	cl_mem mem_gpu_evals_of_runs;
	cl_mem mem_dockpars_prng_states;

	mallocBufferObject(context,CL_MEM_READ_ONLY,size_floatgrids,         			&mem_dockpars_fgrids);
	mallocBufferObject(context,CL_MEM_READ_WRITE,size_populations,        			&mem_dockpars_conformations_current);
	mallocBufferObject(context,CL_MEM_READ_WRITE,size_energies,           			&mem_dockpars_energies_current);
	mallocBufferObject(context,CL_MEM_READ_WRITE,size_populations,        			&mem_dockpars_conformations_next);
	mallocBufferObject(context,CL_MEM_READ_WRITE,size_energies,    	      			&mem_dockpars_energies_next);
	mallocBufferObject(context,CL_MEM_READ_WRITE,size_evals_of_new_entities,		&mem_dockpars_evals_of_new_entities);

	// -------- Replacing with memory maps! ------------
	mallocBufferObject(context,CL_MEM_READ_WRITE,size_evals_of_runs,	  			&mem_gpu_evals_of_runs);

	// -------- Replacing with memory maps! ------------

	mallocBufferObject(context,CL_MEM_READ_WRITE,size_prng_seeds,  	      				&mem_dockpars_prng_states);

	memcopyBufferObjectToDevice(command_queue,mem_dockpars_fgrids,                	false, cpu_floatgrids,  	size_floatgrids);
 	memcopyBufferObjectToDevice(command_queue,mem_dockpars_conformations_current, 	false, cpu_init_populations, 	size_populations);
	memcopyBufferObjectToDevice(command_queue,mem_gpu_evals_of_runs, 		false, cpu_evals_of_runs, 	size_evals_of_runs);
	memcopyBufferObjectToDevice(command_queue,mem_dockpars_prng_states,     	false, cpu_prng_seeds,      	size_prng_seeds);
	
	printf("Local-search chosen method is ADADELTA (ad) because that is the only one available so far in the Kokkos version.");

	clock_start_docking = clock();

	//print progress bar
#ifndef DOCK_DEBUG
	if (mypars->autostop)
	{
		printf("\nExecuting docking runs, stopping automatically after either reaching %.2f kcal/mol standard deviation\nof the best molecules, %u generations, or %u evaluations, whichever comes first:\n\n",mypars->stopstd,mypars->num_of_generations,mypars->num_of_energy_evals);
		printf("Generations |  Evaluations |     Threshold    |  Average energy of best 10%%  | Samples |    Best energy\n");
		printf("------------+--------------+------------------+------------------------------+---------+-------------------\n");
	}
	else
	{
		printf("\nExecuting docking runs:\n");
		printf("        20%%        40%%       60%%       80%%       100%%\n");
		printf("---------+---------+---------+---------+---------+\n");
	}
#else
	printf("\n");
#endif
	curr_progress_cnt = 0;

#ifdef DOCK_DEBUG
	// Main while-loop iterarion counter
	unsigned int ite_cnt = 0;
#endif

	// Kernel1
	printf("\nExecution starts:\n\n");
	printf("%-25s", "\tK_INIT");fflush(stdout);

	// KOKKOS kernel1: kokkos_calc_initpop
	// Initialize DockingParams
	DockingParams<DeviceType> docking_params(myligand_reference, mygrid, mypars, cpu_floatgrids, cpu_prng_seeds);

	// Initialize GeneticParams (broken out of docking params since they relate to the genetic algorithm, not the docking per se
	GeneticParams genetic_params(mypars);

	// Initialize the structs containing the two alternating generations
	// Odd generation gets the initial population copied in
	Generation<DeviceType> odd_generation(mypars->pop_size * mypars->num_of_runs, cpu_init_populations);
	Generation<DeviceType> even_generation(mypars->pop_size * mypars->num_of_runs);

	// Evals of runs on device (for kernel2)
	Kokkos::View<int*,DeviceType> evals_of_runs("evals_of_runs",mypars->num_of_runs);

	// Wrap the C style arrays with an unmanaged kokkos view for easy deep copies (done after view initializations for easy sizing)
        FloatView1D energies_view(cpu_energies_kokkos, odd_generation.energies.extent(0));
        IntView1D new_entities_view(cpu_new_entities_kokkos, docking_params.evals_of_new_entities.extent(0));
        FloatView1D conforms_view(cpu_conforms_kokkos, odd_generation.conformations.extent(0));
	FloatView1D Nenergies_view(cpu_Nenergies_kokkos, odd_generation.energies.extent(0));
        FloatView1D Nconforms_view(cpu_Nconforms_kokkos, odd_generation.conformations.extent(0));
        UnsignedIntView1D prng_view(cpu_prng_kokkos, docking_params.prng_states.extent(0));
        IntView1D evals_of_runs_view(cpu_evals_of_runs, evals_of_runs.extent(0)); // Note this array was prexisting

	// Declare these constant arrays on host
	InterIntra<HostType> interintra_h;
        IntraContrib<HostType> intracontrib_h;
        Intra<HostType> intra_h;
        RotList<HostType> rotlist_h;
        Conform<HostType> conform_h;
	Grads<HostType> grads_h;
	AxisCorrection<HostType> axis_correction_h;

	// Initialize them
	if (kokkos_prepare_const_fields(&myligand_reference, mypars, cpu_ref_ori_angles,
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

	// Perform the kernel formerly known as kernel1
	kokkos_calc_init_pop(odd_generation, mypars, docking_params, conform, rotlist, intracontrib, interintra, intra);
	Kokkos::fence();

	// Copy back from device
        Kokkos::deep_copy(energies_view, odd_generation.energies);
        Kokkos::deep_copy(new_entities_view, docking_params.evals_of_new_entities);

	// Copy from temporary cpu array back to gpu for the remaining openCL kernels
	memcopyBufferObjectToDevice(command_queue,mem_dockpars_energies_current,true,cpu_energies_kokkos,size_energies);

	printf("%15s" ," ... Finished\n");fflush(stdout); // Finished kernel1

	// Kernel2
	printf("%-25s", "\tK_EVAL");fflush(stdout);

	// Perform sum_evals, formerly known as kernel2
	kokkos_sum_evals(mypars, docking_params, evals_of_runs);
	Kokkos::fence();

	printf("%15s" ," ... Finished\n");fflush(stdout);

	Kokkos::deep_copy(evals_of_runs_view, evals_of_runs);

	generation_cnt = 0;
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
	// -------- Replacing with memory maps! ------------
	while ((progress = check_progress(cpu_evals_of_runs, generation_cnt, mypars->num_of_energy_evals, mypars->num_of_generations, mypars->num_of_runs, total_evals)) < 100.0)
	// -------- Replacing with memory maps! ------------
	{
		if (mypars->autostop)
		{
			if (generation_cnt % 10 == 0) {
				memcopyBufferObjectFromDevice(command_queue,cpu_energies,mem_dockpars_energies_current,size_energies);
				for(unsigned int count=0; (count<1+8*(generation_cnt==0)) && (fabs(curr_avg-prev_avg)>0.00001); count++)
				{
					threshold_used = threshold;
					overall_best_energy = 1<<24;
					memset(&average_sd2_N[0],0,avg_arr_size*sizeof(float));
					for (run_cnt=0; run_cnt < mypars->num_of_runs; run_cnt++)
					{
						energies = cpu_energies+run_cnt*mypars->pop_size;
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
				printf("%11u | %12u |%8.2f kcal/mol |%8.2f +/-%8.2f kcal/mol |%8i |%8.2f kcal/mol\n",generation_cnt,total_evals/mypars->num_of_runs,threshold_used,curr_avg,curr_std,bestN,overall_best_energy);
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

		// Perform gen_alg_eval_new, the genetic algorithm formerly known as kernel4
		memcopyBufferObjectFromDevice(command_queue,cpu_conforms_kokkos,mem_dockpars_conformations_current,size_populations);
                        memcopyBufferObjectFromDevice(command_queue,cpu_energies_kokkos,mem_dockpars_energies_current,size_energies);
			memcopyBufferObjectFromDevice(command_queue,cpu_Nconforms_kokkos,mem_dockpars_conformations_next,size_populations);
                        memcopyBufferObjectFromDevice(command_queue,cpu_Nenergies_kokkos,mem_dockpars_energies_next,size_energies);
		if (generation_cnt % 2 == 0){
			Kokkos::deep_copy(odd_generation.conformations, conforms_view);
	                Kokkos::deep_copy(odd_generation.energies, energies_view);
		} else {
			Kokkos::deep_copy(odd_generation.conformations, Nconforms_view);
	                Kokkos::deep_copy(odd_generation.energies, Nenergies_view);
		}

		printf("%-25s", "\tK_GA_GENERATION");fflush(stdout);


			kokkos_gen_alg_eval_new(odd_generation, even_generation, mypars, docking_params, genetic_params,
					        conform, rotlist, intracontrib, interintra, intra);
                Kokkos::fence();
		printf("%15s", " ... Finished\n");fflush(stdout);

		if (docking_params.lsearch_rate != 0.0f) {
			if (strcmp(mypars->ls_method, "ad") == 0) {
				printf("%-25s", "\tK_LS_GRAD_ADADELTA");fflush(stdout);

				// Perform gradient_minAD, formerly known as kernel7
				kokkos_gradient_minAD(even_generation, mypars, docking_params, conform, rotlist, intracontrib, interintra, intra, grads, axis_correction);
				Kokkos::fence();

                		// Copy kokkos output from CPU to OpenCL format
				if (generation_cnt % 2 == 0){
					Kokkos::deep_copy(Nconforms_view,even_generation.conformations);
                                	Kokkos::deep_copy(Nenergies_view,even_generation.energies);
//					memcopyBufferObjectToDevice(command_queue,mem_dockpars_conformations_next,true,cpu_Nconforms_kokkos,size_populations);
//					memcopyBufferObjectToDevice(command_queue,mem_dockpars_energies_next,true,cpu_Nenergies_kokkos,size_energies);
				}
				if (generation_cnt % 2 == 1){
					Kokkos::deep_copy(conforms_view,even_generation.conformations);
                                	Kokkos::deep_copy(energies_view,even_generation.energies);
//					memcopyBufferObjectToDevice(command_queue,mem_dockpars_conformations_current,true,cpu_conforms_kokkos,size_populations);
//                                        memcopyBufferObjectToDevice(command_queue,mem_dockpars_energies_current,true,cpu_energies_kokkos,size_energies);
				}
				memcopyBufferObjectToDevice(command_queue,mem_dockpars_conformations_next,true,cpu_Nconforms_kokkos,size_populations);
                                        memcopyBufferObjectToDevice(command_queue,mem_dockpars_energies_next,true,cpu_Nenergies_kokkos,size_energies);
					memcopyBufferObjectToDevice(command_queue,mem_dockpars_conformations_current,true,cpu_conforms_kokkos,size_populations);
                                        memcopyBufferObjectToDevice(command_queue,mem_dockpars_energies_current,true,cpu_energies_kokkos,size_energies);

				printf("%15s" ," ... Finished\n");fflush(stdout);
			} else {
				// sw, sd, and fire are NOT SUPPORTED in the Kokkos version (yet)
			}
		}

		// Perform sum_evals, formerly known as kernel2
		printf("%-25s", "\tK_EVAL");fflush(stdout);

	        kokkos_sum_evals(mypars, docking_params, evals_of_runs);
	        Kokkos::fence();
		printf("%15s" ," ... Finished\n");fflush(stdout);

		// Copy output from kokkos kernel2 to CPU
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
		memcopyBufferObjectFromDevice(command_queue,cpu_final_populations,mem_dockpars_conformations_current,size_populations);
		memcopyBufferObjectFromDevice(command_queue,cpu_energies,mem_dockpars_energies_current,size_energies);
	}
	else {
		memcopyBufferObjectFromDevice(command_queue,cpu_final_populations,mem_dockpars_conformations_next,size_populations);
		memcopyBufferObjectFromDevice(command_queue,cpu_energies,mem_dockpars_energies_next,size_energies);
	}

	// Process results
	for (run_cnt=0; run_cnt < mypars->num_of_runs; run_cnt++)
	{
		arrange_result(cpu_final_populations+run_cnt*mypars->pop_size*GENOTYPE_LENGTH_IN_GLOBMEM, cpu_energies+run_cnt*mypars->pop_size, mypars->pop_size);
		make_resfiles(cpu_final_populations+run_cnt*mypars->pop_size*GENOTYPE_LENGTH_IN_GLOBMEM, 
			      cpu_energies+run_cnt*mypars->pop_size, 
			      &myligand_reference,
			      myligand_init,
			      myxrayligand, 
			      mypars, 
			      cpu_evals_of_runs[run_cnt], 
			      generation_cnt, 
			      mygrid, 
			      cpu_floatgrids, 
			      cpu_ref_ori_angles+3*run_cnt, 
			      argc, 
			      argv, 
                              /*1*/0,
			      run_cnt, 
			      &(cpu_result_ligands [run_cnt]));
	}
	clock_stop_program_before_clustering = clock();
	clusanal_gendlg(cpu_result_ligands, mypars->num_of_runs, myligand_init, mypars,
					 mygrid, argc, argv, ELAPSEDSECS(clock_stop_docking, clock_start_docking)/mypars->num_of_runs,
					 ELAPSEDSECS(clock_stop_program_before_clustering, clock_start_program),generation_cnt,total_evals/mypars->num_of_runs);
	clock_stop_docking = clock();

	clReleaseMemObject(mem_interintra_const);
	clReleaseMemObject(mem_intracontrib_const);
	clReleaseMemObject(mem_intra_const);
	clReleaseMemObject(mem_rotlist_const);
	clReleaseMemObject(mem_conform_const);

	clReleaseMemObject(mem_rotbonds_const);
	clReleaseMemObject(mem_rotbonds_atoms_const);
	clReleaseMemObject(mem_num_rotating_atoms_per_rotbond_const);

	clReleaseMemObject(mem_dockpars_fgrids);
	clReleaseMemObject(mem_dockpars_conformations_current);
	clReleaseMemObject(mem_dockpars_energies_current);
	clReleaseMemObject(mem_dockpars_conformations_next);
	clReleaseMemObject(mem_dockpars_energies_next);
	clReleaseMemObject(mem_dockpars_evals_of_new_entities);
	clReleaseMemObject(mem_dockpars_prng_states);
	clReleaseMemObject(mem_gpu_evals_of_runs);

	clReleaseMemObject(mem_angle_const);
	clReleaseMemObject(mem_dependence_on_theta_const);
	clReleaseMemObject(mem_dependence_on_rotangle_const);

	// Release all kernels,
	// regardless of the chosen local-search method for execution.
	// Otherwise, memory leak in clCreateKernel()
	clReleaseKernel(kernel1);
	clReleaseKernel(kernel2);
	clReleaseKernel(kernel3);
	clReleaseKernel(kernel4);
	clReleaseKernel(kernel5);
	clReleaseKernel(kernel6);
	clReleaseKernel(kernel7);

	clReleaseProgram(program);
	
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(device_ids);
	free(platform_id);

	free(cpu_init_populations);
	free(cpu_energies);
	free(cpu_result_ligands);
	free(cpu_prng_seeds);
	free(cpu_evals_of_runs);
	free(cpu_ref_ori_angles);

	// TEMPORARY - ALS
        free(cpu_energies_kokkos);
        free(cpu_new_entities_kokkos);
	free(cpu_Nenergies_kokkos);
	free(cpu_Nconforms_kokkos);
        free(cpu_prng_kokkos);
        free(cpu_conforms_kokkos);
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

	int i;
	double evals_progress;
	double gens_progress;

	//calculating progress according to number of runs
	total_evals = 0;
	for (i=0; i<num_of_runs; i++)
		total_evals += evals_of_runs[i];

	evals_progress = (double)total_evals/((double) num_of_runs)/max_num_of_evals*100.0;

	//calculating progress according to number of generations
	gens_progress = ((double) generation_cnt)/((double) max_num_of_gens)*100.0; //std::cout<< "gens_progress: " << gens_progress <<std::endl;

	if (evals_progress > gens_progress)
		return evals_progress;
	else
		return gens_progress;
}
