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
	#define KNWI " -DN64WI "
#endif

#if defined (REPRO)
	#define REP " -DREPRO "
#else
	#define REP " "
#endif


#ifdef __APPLE__
	#define KGDB_GPU " -g -cl-opt-disable "
#else
	#define KGDB_GPU " -g -O0 -Werror -cl-opt-disable "
#endif
#define KGDB_CPU " -g3 -Werror -cl-opt-disable "
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

#include "autostop.hpp"
#include "performdocking.h"
#include "stringify.h"
#include "correct_grad_axisangle.h"

void copy_map_to_gpu(
                     GpuTempData&      tData,
                     std::vector<Map>& all_maps,
                     int               t,
                     int               size_of_one_map
                    )
{
	cl_int err;
	err = clEnqueueWriteBuffer(tData.command_queue,tData.pMem_fgrids,true,t*size_of_one_map*sizeof(float),size_of_one_map*sizeof(float),all_maps[t].grid.data(),0,NULL,NULL);
	if (err != CL_SUCCESS){
		printf("pMem_fgrids: failed to upload maps to GPU memory. %d\n", err);
		fflush(stdout);
		exit(-1);
	}
}

void setup_gpu_for_docking(
                           GpuData& cData,
                           GpuTempData& tData
                          )
{
	if(cData.devnum<-1) return; // device already setup
// =======================================================================
// OpenCL Host Setup
// =======================================================================
	cl_platform_id*  platform_id;
	cl_device_id*    device_ids;
	cl_device_id     device_id;

#ifdef _WIN32
	const char *filename = KRNL_FILE;
	printf("%-40s %-40s\n", "Kernel source file: ", filename);  fflush(stdout);
#else
	printf("%-40s %-40s\n", "Kernel source used for development: ", "./device/calcenergy.cl");  fflush(stdout);
	printf("%-40s %-40s\n", "Kernel string used for building: ",    "./host/inc/stringify.h");  fflush(stdout);
#endif

	const char* options_program = OPT_PROG;
	printf("%-40s %-40s\n", "Kernel compilation flags: ", options_program); fflush(stdout);

	const char *name_k1 = KRNL1;
	const char *name_k2 = KRNL2;
	const char *name_k3 = KRNL3;
	const char *name_k4 = KRNL4;
	const char *name_k5 = KRNL5;
	const char *name_k6 = KRNL6;
	const char *name_k7 = KRNL7;

	cl_uint platformCount;
	cl_uint deviceCount;

	// Get all available platforms
	if (getPlatforms(&platform_id,&platformCount) != 0) exit(-1);
	if (cData.devnum<0) // this is for Cuda to not set the device (in here for compatibility)
		cData.devnum=0;
	unsigned int platform_nr=cData.devnum>>8;
	cData.devnum&=0xFF;
	if(platform_nr>=platformCount){
		platform_nr=0;
		size_t name_size;
		clGetPlatformInfo(platform_id[platform_nr], CL_PLATFORM_NAME, 0, NULL, &name_size);
		char* platform_name = (char*) malloc(name_size);
		clGetPlatformInfo(platform_id[platform_nr], CL_PLATFORM_NAME, name_size, platform_name, NULL);
		printf("Info: User specified OpenCL platform number does not exist, using first platform (\"%s\").\n",platform_name);
		free(platform_name);
	}

	// Get all devices of first platform
	if (getDevices(platform_id[platform_nr],platformCount,&device_ids,&deviceCount) != 0){
		printf("Info: Specified OpenCL platform %i does not have a suitable device",platform_nr);
		platform_nr=0;
		cl_int err = 1;
		while((err != CL_SUCCESS) && (platform_nr<platformCount)){
#if defined ALL_DEVICE
			err = clGetDeviceIDs(platform_id[platform_nr], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
#elif defined GPU_DEVICE
			err = clGetDeviceIDs(platform_id[platform_nr], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
#elif defined FPGA_DEVICE
			err = clGetDeviceIDs(platform_id[platform_nr], CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &deviceCount);
#elif defined CPU_DEVICE
			err = clGetDeviceIDs(platform_id[platform_nr], CL_DEVICE_TYPE_CPU, 0, NULL, &deviceCount);
#else
			exit(-1);
#endif
			if(err != CL_SUCCESS)
				platform_nr++;
		}
		if(platform_nr>=platformCount){
			printf(".\nError: No suitable OpenCL platform found.\n");
			exit(-1);
		} else{
			size_t name_size;
			clGetPlatformInfo(platform_id[platform_nr], CL_PLATFORM_NAME, 0, NULL, &name_size);
			char* platform_name = (char*) malloc(name_size);
			clGetPlatformInfo(platform_id[platform_nr], CL_PLATFORM_NAME, name_size, platform_name, NULL);
			printf(", using OpenCL platform %i (\"%s\") instead.\n",platform_nr,platform_name);
			free(platform_name);
		}
		device_ids = (cl_device_id*)malloc(sizeof(cl_device_id)*deviceCount);
#if defined ALL_DEVICE
		err = clGetDeviceIDs(platform_id[platform_nr], CL_DEVICE_TYPE_ALL, deviceCount, device_ids, NULL);
#elif defined GPU_DEVICE
		err = clGetDeviceIDs(platform_id[platform_nr], CL_DEVICE_TYPE_GPU, deviceCount, device_ids, NULL);
#elif defined FPGA_DEVICE
		err = clGetDeviceIDs(platform_id[platform_nr], CL_DEVICE_TYPE_ACCELERATOR, deviceCount, device_ids, NULL);
#elif defined CPU_DEVICE
		err = clGetDeviceIDs(platform_id[platform_nr], CL_DEVICE_TYPE_CPU, deviceCount, device_ids, NULL);
#endif
		if(err != CL_SUCCESS){
			printf("Error: clGetDeviceIDs() %i\n",err);
			exit(-1);
		}
	}
	
	if (cData.devnum>=(int)deviceCount)
	{
		printf("Info: User specified OpenCL device number does not exist, using first device.\n");
		cData.devnum=0;
	}
	device_id=device_ids[cData.devnum];
	size_t dev_name_size;
	clGetDeviceInfo(device_ids[cData.devnum], CL_DEVICE_NAME, 0, NULL, &dev_name_size);
	char* device_name = (char*) malloc(dev_name_size);
	clGetDeviceInfo(device_ids[cData.devnum], CL_DEVICE_NAME, dev_name_size, device_name, NULL);
	printf("OpenCL device:                           %s",device_name);
	if(deviceCount>1) printf(" (#%d / %d)",cData.devnum+1,deviceCount);
	printf("\n");
	cData.devnum=-2;

	free(device_name);

	// Create context from first platform
	if (createContext(platform_id[platform_nr],1,&device_id,&tData.context) != 0) exit(-1);

	// Create command queue for first device
	if (createCommandQueue(tData.context,device_id,&tData.command_queue) != 0) exit(-1);

	// Create program from source
#ifdef _WIN32
	if (ImportSource(filename, name_k1, &device_id, tData.context, options_program, &tData.kernel1) != 0) exit(-1);
	if (ImportSource(filename, name_k2, &device_id, tData.context, options_program, &tData.kernel2) != 0) exit(-1);
	if (ImportSource(filename, name_k3, &device_id, tData.context, options_program, &tData.kernel3) != 0) exit(-1);
	if (ImportSource(filename, name_k4, &device_id, tData.context, options_program, &tData.kernel4) != 0) exit(-1);
	if (ImportSource(filename, name_k5, &device_id, tData.context, options_program, &tData.kernel5) != 0) exit(-1);
	if (ImportSource(filename, name_k6, &device_id, tData.context, options_program, &tData.kernel6) != 0) exit(-1);
	if (ImportSource(filename, name_k7, &device_id, tData.context, options_program, &tData.kernel7) != 0) exit(-1);
#else
	if (ImportSourceToProgram(calcenergy_ocl, &device_id, tData.context, &tData.program, options_program) != 0) exit(-1);
#endif

	// Create kernels
	if (createKernel(&device_id, &tData.program, name_k1, &tData.kernel1) != 0) exit(-1);
	if (createKernel(&device_id, &tData.program, name_k2, &tData.kernel2) != 0) exit(-1);
	if (createKernel(&device_id, &tData.program, name_k3, &tData.kernel3) != 0) exit(-1);
	if (createKernel(&device_id, &tData.program, name_k4, &tData.kernel4) != 0) exit(-1);
	if (createKernel(&device_id, &tData.program, name_k5, &tData.kernel5) != 0) exit(-1);
	if (createKernel(&device_id, &tData.program, name_k6, &tData.kernel6) != 0) exit(-1);
	if (createKernel(&device_id, &tData.program, name_k7, &tData.kernel7) != 0) exit(-1);

	free(device_ids);
	free(platform_id);
// End of OpenCL Host Setup
// =======================================================================

	size_t sz_interintra_const   = MAX_NUM_OF_ATOMS*sizeof(float) + 
	                               MAX_NUM_OF_ATOMS*sizeof(uint32_t) +
	                               MAX_NUM_OF_ATOMS*sizeof(uint32_t) +
	                               MAX_NUM_OF_ATOMS*sizeof(char);

	size_t sz_intracontrib_const = 2*MAX_INTRAE_CONTRIBUTORS*sizeof(uint32_t);

	size_t sz_intra_const        = MAX_NUM_OF_ATYPES*sizeof(unsigned int) +
	                               MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(unsigned short) +
	                               MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) +
	                               MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) +
	                               MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) +
	                               MAX_NUM_OF_ATYPES*sizeof(float) +
	                               MAX_NUM_OF_ATYPES*sizeof(float);

	size_t sz_rotlist_const      = MAX_NUM_OF_ROTATIONS*sizeof(int);

	size_t sz_conform_const      = 3*MAX_NUM_OF_ATOMS*sizeof(float) +
	                               3*MAX_NUM_OF_ROTBONDS*sizeof(float) +
	                               3*MAX_NUM_OF_ROTBONDS*sizeof(float) +
	                               4*MAX_NUM_OF_RUNS*sizeof(float);

	// These constants are allocated in global memory since
	// there is a limited number of constants that can be passed
	// as arguments to kernel
	mallocBufferObject(tData.context,CL_MEM_READ_ONLY,sz_interintra_const,        &cData.mem_interintra_const);
	mallocBufferObject(tData.context,CL_MEM_READ_ONLY,sz_intracontrib_const,      &cData.mem_intracontrib_const);
	mallocBufferObject(tData.context,CL_MEM_READ_ONLY,sz_intra_const,             &cData.mem_intra_const);
	mallocBufferObject(tData.context,CL_MEM_READ_ONLY,sz_rotlist_const,           &cData.mem_rotlist_const);
	mallocBufferObject(tData.context,CL_MEM_READ_ONLY,sz_conform_const,           &cData.mem_conform_const);

	mallocBufferObject(tData.context,CL_MEM_READ_ONLY,2*MAX_NUM_OF_ROTBONDS*sizeof(int),                &cData.mem_rotbonds_const);
	mallocBufferObject(tData.context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS*sizeof(int), &cData.mem_rotbonds_atoms_const);
	mallocBufferObject(tData.context,CL_MEM_READ_ONLY,MAX_NUM_OF_ROTBONDS*sizeof(int),                  &cData.mem_num_rotating_atoms_per_rotbond_const);

	mallocBufferObject(tData.context,CL_MEM_READ_ONLY,1000*sizeof(float),&cData.mem_angle_const);
	mallocBufferObject(tData.context,CL_MEM_READ_ONLY,1000*sizeof(float),&cData.mem_dependence_on_theta_const);
	mallocBufferObject(tData.context,CL_MEM_READ_ONLY,1000*sizeof(float),&cData.mem_dependence_on_rotangle_const);

	memcopyBufferObjectToDevice(tData.command_queue,cData.mem_angle_const,                  false,  &angle,                  1000*sizeof(float));
	memcopyBufferObjectToDevice(tData.command_queue,cData.mem_dependence_on_theta_const,    false,  &dependence_on_theta,    1000*sizeof(float));
	memcopyBufferObjectToDevice(tData.command_queue,cData.mem_dependence_on_rotangle_const, false,  &dependence_on_rotangle, 1000*sizeof(float));

	if(cData.preload_gridsize>0)
		mallocBufferObject(tData.context,CL_MEM_READ_ONLY,cData.preload_gridsize* (sizeof(float)) * (ATYPE_NUM+2),&(tData.pMem_fgrids));
}

void finish_gpu_from_docking(
                             GpuData&     cData,
                             GpuTempData& tData
                            )
{
	clReleaseMemObject(cData.mem_interintra_const);
	clReleaseMemObject(cData.mem_intracontrib_const);
	clReleaseMemObject(cData.mem_intra_const);
	clReleaseMemObject(cData.mem_rotlist_const);
	clReleaseMemObject(cData.mem_conform_const);

	clReleaseMemObject(cData.mem_rotbonds_const);
	clReleaseMemObject(cData.mem_rotbonds_atoms_const);
	clReleaseMemObject(cData.mem_num_rotating_atoms_per_rotbond_const);

	clReleaseMemObject(cData.mem_angle_const);
	clReleaseMemObject(cData.mem_dependence_on_theta_const);
	clReleaseMemObject(cData.mem_dependence_on_rotangle_const);

	if(tData.pMem_fgrids) clReleaseMemObject(tData.pMem_fgrids);

	// Release all kernels,
	// regardless of the chosen local-search method for execution.
	// Otherwise, memory leak in clCreateKernel()
	clReleaseKernel(tData.kernel1);
	clReleaseKernel(tData.kernel2);
	clReleaseKernel(tData.kernel3);
	clReleaseKernel(tData.kernel4);
	clReleaseKernel(tData.kernel5);
	clReleaseKernel(tData.kernel6);
	clReleaseKernel(tData.kernel7);

	clReleaseProgram(tData.program);
	clReleaseCommandQueue(tData.command_queue);
	clReleaseContext(tData.context);
}

int docking_with_gpu(
                     const Gridinfo*        mygrid,
                     /*const*/ float*       cpu_floatgrids,
                           Dockpars*        mypars,
                     const Liganddata*      myligand_init,
                     const Liganddata*      myxrayligand,
                           Profile&         profile,
                     const int*             argc,
                           char**           argv,
                           SimulationState& sim_state,
                           GpuData&         cData,
                           GpuTempData&     tData,
                           bool             floatgrids_preloaded
                    )
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
		are the corresponding command line arguments parameter
*/
{
	// Times
	cl_ulong time_start_kernel;
	cl_ulong time_end_kernel;

	size_t kernel1_gxsize, kernel1_lxsize;
	size_t kernel2_gxsize, kernel2_lxsize;
	size_t kernel3_gxsize = 0;
	size_t kernel3_lxsize = 0;
	size_t kernel4_gxsize, kernel4_lxsize;
	size_t kernel5_gxsize = 0;
	size_t kernel5_lxsize = 0;
	size_t kernel6_gxsize = 0;
	size_t kernel6_lxsize = 0;
	size_t kernel7_gxsize = 0;
	size_t kernel7_lxsize = 0;

	Liganddata myligand_reference;

	float* cpu_init_populations;
	float* cpu_final_populations;
	unsigned int* cpu_prng_seeds;

	Dockparameters dockpars;
	size_t size_floatgrids;
	size_t size_populations;
	size_t size_energies;
	size_t size_prng_seeds;
	size_t size_evals_of_runs;

	int threadsPerBlock;
	int blocksPerGridForEachEntity;
	int blocksPerGridForEachRun;
	int blocksPerGridForEachLSEntity = 0;
	int blocksPerGridForEachGradMinimizerEntity = 0;

	int generation_cnt;
	int i;
	double progress;

	int curr_progress_cnt;
	int new_progress_cnt;

	clock_t clock_start_docking;
	clock_t clock_stop_docking;

	// setting number of blocks and threads
	threadsPerBlock = NUM_OF_THREADS_PER_BLOCK;
	blocksPerGridForEachEntity = mypars->pop_size * mypars->num_of_runs;
	blocksPerGridForEachRun = mypars->num_of_runs;

	// allocating CPU memory for initial populations
	size_populations = mypars->num_of_runs * mypars->pop_size * GENOTYPE_LENGTH_IN_GLOBMEM*sizeof(float);
	sim_state.cpu_populations.resize(size_populations);
	memset(sim_state.cpu_populations.data(), 0, size_populations);

	// allocating CPU memory for results
	size_energies = mypars->pop_size * mypars->num_of_runs * sizeof(float);
	sim_state.cpu_energies.resize(size_energies);
	cpu_init_populations = sim_state.cpu_populations.data();
	cpu_final_populations = sim_state.cpu_populations.data();

	// generating initial populations and random orientation angles of reference ligand
	// (ligand will be moved to origo and scaled as well)
	myligand_reference = *myligand_init;
	gen_initpop_and_reflig(mypars, cpu_init_populations, &myligand_reference, mygrid);

	// allocating memory in CPU for pseudorandom number generator seeds and
	// generating them (seed for each thread during GA)
	size_prng_seeds = blocksPerGridForEachEntity * threadsPerBlock * sizeof(unsigned int);
	cpu_prng_seeds = (unsigned int*) malloc(size_prng_seeds);

	LocalRNG r(mypars->seed);

	for (i=0; i<blocksPerGridForEachEntity*threadsPerBlock; i++)
		cpu_prng_seeds[i] = r.random_uint();

	// allocating memory in CPU for evaluation counters
	size_evals_of_runs = mypars->num_of_runs*sizeof(int);
	sim_state.cpu_evals_of_runs.resize(size_evals_of_runs);
	memset(sim_state.cpu_evals_of_runs.data(), 0, size_evals_of_runs);

	// preparing the constant data fields for the GPU
	kernelconstant_interintra	KerConst_interintra;
	kernelconstant_intracontrib	KerConst_intracontrib;
	kernelconstant_intra		KerConst_intra;
	kernelconstant_rotlist		KerConst_rotlist;
	kernelconstant_conform		KerConst_conform;
	kernelconstant_grads		KerConst_grads;

	if (prepare_const_fields_for_gpu(&myligand_reference, mypars,
	                                 &KerConst_interintra,
	                                 &KerConst_intracontrib,
	                                 &KerConst_intra,
	                                 &KerConst_rotlist,
	                                 &KerConst_conform,
	                                 &KerConst_grads) == 1) {
		return 1;
	}

	size_t sz_interintra_const   = MAX_NUM_OF_ATOMS*sizeof(float) +
	                               MAX_NUM_OF_ATOMS*sizeof(uint32_t) +
	                               MAX_NUM_OF_ATOMS*sizeof(uint32_t) +
	                               MAX_NUM_OF_ATOMS*sizeof(char);

	size_t sz_intracontrib_const = 2*MAX_INTRAE_CONTRIBUTORS*sizeof(uint32_t);

	size_t sz_intra_const        = MAX_NUM_OF_ATYPES*sizeof(unsigned int) +
	                               MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(unsigned short) +
	                               MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) +
	                               MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) +
	                               MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) +
	                               MAX_NUM_OF_ATYPES*sizeof(float) +
	                               MAX_NUM_OF_ATYPES*sizeof(float);

	size_t sz_rotlist_const      = MAX_NUM_OF_ROTATIONS*sizeof(int);

	size_t sz_conform_const      = 3*MAX_NUM_OF_ATOMS*sizeof(float) +
	                               3*MAX_NUM_OF_ROTBONDS*sizeof(float) +
	                               3*MAX_NUM_OF_ROTBONDS*sizeof(float) +
	                               4*MAX_NUM_OF_RUNS*sizeof(float);

	memcopyBufferObjectToDevice(tData.command_queue,cData.mem_interintra_const,   false, &KerConst_interintra,   sz_interintra_const);
	memcopyBufferObjectToDevice(tData.command_queue,cData.mem_intracontrib_const, false, &KerConst_intracontrib, sz_intracontrib_const);
	memcopyBufferObjectToDevice(tData.command_queue,cData.mem_intra_const,        false, &KerConst_intra,        sz_intra_const);
	memcopyBufferObjectToDevice(tData.command_queue,cData.mem_rotlist_const,      false, &KerConst_rotlist,      sz_rotlist_const);
	memcopyBufferObjectToDevice(tData.command_queue,cData.mem_conform_const,      false, &KerConst_conform,      sz_conform_const);

	memcopyBufferObjectToDevice(tData.command_queue,cData.mem_rotbonds_const,                       false, &KerConst_grads.rotbonds,                       2*MAX_NUM_OF_ROTBONDS*sizeof(int));
	memcopyBufferObjectToDevice(tData.command_queue,cData.mem_rotbonds_atoms_const,                 false, &KerConst_grads.rotbonds_atoms,                 MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS*sizeof(int));
	memcopyBufferObjectToDevice(tData.command_queue,cData.mem_num_rotating_atoms_per_rotbond_const, false, &KerConst_grads.num_rotating_atoms_per_rotbond, MAX_NUM_OF_ROTBONDS*sizeof(int));

	// ----------------------------------------------------------------------

 	// allocating GPU memory for populations, floatgirds,
	// energies, evaluation counters and random number generator states
	size_floatgrids = 4 * (sizeof(float)) * (mygrid->num_of_atypes+2) * (mygrid->size_xyz[0]) * (mygrid->size_xyz[1]) * (mygrid->size_xyz[2]);

	cl_mem mem_dockpars_conformations_current;
	cl_mem mem_dockpars_energies_current;
	cl_mem mem_dockpars_conformations_next;
	cl_mem mem_dockpars_energies_next;
	cl_mem mem_dockpars_evals_of_new_entities;
	cl_mem mem_gpu_evals_of_runs;
	cl_mem mem_dockpars_prng_states;

	if(cData.preload_gridsize==0)
		mallocBufferObject(tData.context,CL_MEM_READ_ONLY,size_floatgrids, &tData.pMem_fgrids);
	mallocBufferObject(tData.context,CL_MEM_READ_WRITE,size_populations, &mem_dockpars_conformations_current);
	mallocBufferObject(tData.context,CL_MEM_READ_WRITE,size_energies,    &mem_dockpars_energies_current);
	mallocBufferObject(tData.context,CL_MEM_READ_WRITE,size_populations, &mem_dockpars_conformations_next);
	mallocBufferObject(tData.context,CL_MEM_READ_WRITE,size_energies,    &mem_dockpars_energies_next);
	mallocBufferObject(tData.context,CL_MEM_READ_WRITE,mypars->pop_size*mypars->num_of_runs*sizeof(int), &mem_dockpars_evals_of_new_entities);

	// -------- Replacing with memory maps! ------------
#if defined (MAPPED_COPY)
	mallocBufferObject(tData.context,CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,size_evals_of_runs, &mem_gpu_evals_of_runs);
#else
	mallocBufferObject(tData.context,CL_MEM_READ_WRITE,size_evals_of_runs, &mem_gpu_evals_of_runs);
#endif
	// -------- Replacing with memory maps! ------------

	mallocBufferObject(tData.context,CL_MEM_READ_WRITE,size_prng_seeds, &mem_dockpars_prng_states);

	if(!floatgrids_preloaded)
		memcopyBufferObjectToDevice(tData.command_queue,tData.pMem_fgrids,          false, cpu_floatgrids,                     size_floatgrids);
	memcopyBufferObjectToDevice(tData.command_queue,mem_dockpars_conformations_current, false, cpu_init_populations,               size_populations);
	memcopyBufferObjectToDevice(tData.command_queue,mem_gpu_evals_of_runs,              false, sim_state.cpu_evals_of_runs.data(), size_evals_of_runs);
	memcopyBufferObjectToDevice(tData.command_queue,mem_dockpars_prng_states,           false, cpu_prng_seeds,                     size_prng_seeds);

	// preparing parameter struct
	dockpars.num_of_atoms      = ((int)  myligand_reference.num_of_atoms);
	dockpars.true_ligand_atoms = ((int)  myligand_reference.true_ligand_atoms);
	dockpars.num_of_atypes     = ((int)  myligand_reference.num_of_atypes);
	dockpars.num_of_map_atypes = ((int)  mygrid->num_of_map_atypes);
	dockpars.num_of_intraE_contributors = ((int) myligand_reference.num_of_intraE_contributors);
	dockpars.gridsize_x        = ((int)  mygrid->size_xyz[0]);
	dockpars.gridsize_y        = ((int)  mygrid->size_xyz[1]);
	dockpars.gridsize_z        = ((int)  mygrid->size_xyz[2]);
	dockpars.grid_spacing      = ((float) mygrid->spacing);
	dockpars.rotbondlist_length= ((int) NUM_OF_THREADS_PER_BLOCK*(myligand_reference.num_of_rotcyc));
	dockpars.coeff_elec        = ((float) mypars->coeffs.scaled_AD4_coeff_elec);
	dockpars.elec_min_distance = ((float) mypars->elec_min_distance);
	dockpars.coeff_desolv      = ((float) mypars->coeffs.AD4_coeff_desolv);
	dockpars.pop_size          = mypars->pop_size;
	dockpars.num_of_genes      = myligand_reference.num_of_rotbonds + 6;
	// Notice: dockpars.tournament_rate, dockpars.crossover_rate, dockpars.mutation_rate
	// were scaled down to [0,1] in host to reduce number of operations in device
	dockpars.tournament_rate   = mypars->tournament_rate/100.0f;
	dockpars.crossover_rate    = mypars->crossover_rate/100.0f;
	dockpars.mutation_rate     = mypars->mutation_rate/100.f;
	dockpars.abs_max_dang      = mypars->abs_max_dang;
	dockpars.abs_max_dmov      = mypars->abs_max_dmov;
	dockpars.qasp              = mypars->qasp;
	dockpars.smooth            = mypars->smooth;
	unsigned int g2            = dockpars.gridsize_x * dockpars.gridsize_y;
	unsigned int g3            = dockpars.gridsize_x * dockpars.gridsize_y * dockpars.gridsize_z;

	dockpars.lsearch_rate      = mypars->lsearch_rate;
	dockpars.adam_beta1        = mypars->adam_beta1;
	dockpars.adam_beta2        = mypars->adam_beta2;
	dockpars.adam_epsilon      = mypars->adam_epsilon;

	if (dockpars.lsearch_rate != 0.0f)
	{
		dockpars.num_of_lsentities = (unsigned int) (mypars->lsearch_rate/100.0*mypars->pop_size + 0.5);
		dockpars.rho_lower_bound   = mypars->rho_lower_bound;
		dockpars.base_dmov_mul_sqrt3 = mypars->base_dmov_mul_sqrt3;
		dockpars.base_dang_mul_sqrt3 = mypars->base_dang_mul_sqrt3;
		dockpars.cons_limit        = (unsigned int) mypars->cons_limit;
		dockpars.max_num_of_iters  = (unsigned int) mypars->max_num_of_iters;

		// The number of entities that undergo Solis-Wets minimization,
		blocksPerGridForEachLSEntity = dockpars.num_of_lsentities*mypars->num_of_runs;

		// The number of entities that undergo any gradient-based minimization,
		// by default, it is the same as the number of entities that undergo the Solis-Wets minimizer
		blocksPerGridForEachGradMinimizerEntity = dockpars.num_of_lsentities*mypars->num_of_runs;

		// Enable only for debugging
		// Only one entity per reach run, undergoes gradient minimization
		//blocksPerGridForEachGradMinimizerEntity = mypars->num_of_runs;
	}
	
	unsigned long min_as_evals = 0; // no minimum w/o heuristics
	if(mypars->use_heuristics){
		unsigned long heur_evals;
		unsigned long nev=mypars->num_of_energy_evals;
		if(strcmp(mypars->ls_method,"sw")==0){
//			heur_evals = (unsigned long)ceil(1000 * pow(2.0,1.3 * myligand_init->num_of_rotbonds + 3.5));
			heur_evals = (unsigned long)ceil(1000 * pow(2.0,1.3 * myligand_init->num_of_rotbonds + 4.0));
		} else{
			if(strcmp(mypars->ls_method,"ad")==0){
//			heur_evals = (unsigned long)ceil(1000 * pow(2.0,0.4 * myligand_init->num_of_rotbonds + 7.0));
//			heur_evals = (unsigned long)ceil(1000 * pow(2.0,0.5 * myligand_init->num_of_rotbonds + 6.0));
			heur_evals = (unsigned long)ceil(64000 * pow(2.0, (0.5 - 0.2 * myligand_init->num_of_rotbonds/(20.0f + myligand_init->num_of_rotbonds)) * myligand_init->num_of_rotbonds));
			} else{
				printf("\nError: LS method \"%s\" is not supported by heuristics.\n       Please choose Solis-Wets (sw), Adadelta (ad),\n       or switch off the heuristics.\n",mypars->ls_method);
				exit(-1);
			}
		}
		if(heur_evals<500000) heur_evals=500000;
		heur_evals *= 50.0f/mypars->num_of_runs;
		// e*hm/(hm+e) = 0.95*e => hm/(hm+e) = 0.95
		// => 0.95*hm + 0.95*e = hm => 0.95*e = 0.05 * hm
		// => e = 1/19*hm
		// at hm = 50 M => e0 = 2.63 M where e becomes less than 95% (about 11 torsions)
		mypars->num_of_energy_evals = (unsigned long)ceil(heur_evals*(float)mypars->heuristics_max/(mypars->heuristics_max+heur_evals));
		printf("    Using heuristics: (capped) number of evaluations set to %lu\n",mypars->num_of_energy_evals);
		if (mypars->nev_provided && (mypars->num_of_energy_evals>nev)){
			printf("    Overriding heuristics, setting number of evaluations to --nev = %lu instead.\n",nev);
			mypars->num_of_energy_evals = nev;
			profile.capped = true;
		}
		float cap_fraction = (float)mypars->num_of_energy_evals/heur_evals;
		float a = 27.0/26.0; // 10% at cap_fraction of 50%
//		float a = 12.0/11.0; // 20% at cap_fraction of 50%
		float min_frac = a/(1+cap_fraction*cap_fraction*(a/(a-1.0f)-1.0f))+1.0f-a;
		min_as_evals = (unsigned long)ceil(mypars->num_of_energy_evals*min_frac)*mypars->num_of_runs;
		if(cap_fraction<0.5f){
			printf("    Warning: The set number of evals is %.2f%% of the uncapped heuristics estimate of %lu evals.\n",cap_fraction*100.0f,heur_evals);
			printf("             This means this docking may not be able to converge. Increasing ");
			if (mypars->nev_provided && (mypars->num_of_energy_evals>nev))
				printf("--nev");
			else
				printf("--heurmax");
			printf(" may improve\n             convergence but will also increase runtime.\n");
			if(mypars->autostop) printf("             AutoStop will not stop before %.2f%% (%lu) of the set number of evaluations.\n",min_frac*100.0f,min_as_evals/mypars->num_of_runs);
		}
	}
	
	char method_chosen[64]; // 64 chars will be enough for this message as mypars->ls_method is 4 chars at the longest
	if(strcmp(mypars->ls_method, "sw") == 0){
		strcpy(method_chosen,"Solis-Wets (sw)");
	}
	else if(strcmp(mypars->ls_method, "sd")  == 0){
		strcpy(method_chosen,"Steepest-Descent (sd)");
	}
	else if(strcmp(mypars->ls_method, "fire") == 0){
		strcpy(method_chosen,"FIRE (fire)");
	}
	else if(strcmp(mypars->ls_method, "ad") == 0){
		strcpy(method_chosen,"ADADELTA (ad)");
	}
	else{
		printf("\nError: LS method %s is not (yet) supported in the OpenCL version.\n",mypars->ls_method);
		exit(-1);
	}
	printf("    Local-search chosen method is: %s\n", (dockpars.lsearch_rate == 0.0f)? "GA" : method_chosen);

	if((mypars->initial_sw_generations>0) && (strcmp(mypars->ls_method, "sw") != 0))
		printf("    Using Solis-Wets (sw) for the first %d generations.\n",mypars->initial_sw_generations);

        // Get profile for timing
	profile.adadelta=(strcmp(mypars->ls_method, "ad")==0);
	profile.n_evals = mypars->num_of_energy_evals;
	profile.num_atoms = myligand_reference.num_of_atoms;
	profile.num_rotbonds = myligand_init->num_of_rotbonds;

	/*
	printf("dockpars.num_of_intraE_contributors:%u\n", dockpars.num_of_intraE_contributors);
	printf("dockpars.rotbondlist_length:%u\n", dockpars.rotbondlist_length);
	*/

	clock_start_docking = clock();

#ifdef DOCK_DEBUG
	printf("\n");
	// Main while-loop iterarion counter
	unsigned int ite_cnt = 0;
#endif

	/*
	// Added for printing intracontributor_pairs (autodockdevpy)
	for (unsigned int intrapair_cnt=0; 
			  intrapair_cnt<dockpars.num_of_intraE_contributors;
			  intrapair_cnt++) {
		if (intrapair_cnt == 0) {
			printf("%-10s %-10s %-10s\n", "#pair", "#atom1", "#atom2");
		}

		printf ("%-10u %-10u %-10u\n", intrapair_cnt,
					    KerConst.intraE_contributors_const[3*intrapair_cnt],
					    KerConst.intraE_contributors_const[3*intrapair_cnt+1]);
	}
	*/

	// Kernel1
	setKernelArg(tData.kernel1,0, sizeof(dockpars.num_of_atoms),                  &dockpars.num_of_atoms);
	setKernelArg(tData.kernel1,1, sizeof(dockpars.true_ligand_atoms),             &dockpars.true_ligand_atoms);
	setKernelArg(tData.kernel1,2, sizeof(dockpars.num_of_atypes),                 &dockpars.num_of_atypes);
	setKernelArg(tData.kernel1,3, sizeof(dockpars.num_of_map_atypes),             &dockpars.num_of_map_atypes);
	setKernelArg(tData.kernel1,4, sizeof(dockpars.num_of_intraE_contributors),    &dockpars.num_of_intraE_contributors);
	setKernelArg(tData.kernel1,5, sizeof(dockpars.gridsize_x),                    &dockpars.gridsize_x);
	setKernelArg(tData.kernel1,6, sizeof(dockpars.gridsize_y),                    &dockpars.gridsize_y);
	setKernelArg(tData.kernel1,7, sizeof(dockpars.gridsize_z),                    &dockpars.gridsize_z);
	setKernelArg(tData.kernel1,8, sizeof(g2),                                     &g2);
	setKernelArg(tData.kernel1,9, sizeof(g3),                                     &g3);
	setKernelArg(tData.kernel1,10,sizeof(dockpars.grid_spacing),                  &dockpars.grid_spacing);
	setKernelArg(tData.kernel1,11,sizeof(tData.pMem_fgrids),                      &tData.pMem_fgrids);
	setKernelArg(tData.kernel1,12,sizeof(dockpars.rotbondlist_length),            &dockpars.rotbondlist_length);
	setKernelArg(tData.kernel1,13,sizeof(dockpars.coeff_elec),                    &dockpars.coeff_elec);
	setKernelArg(tData.kernel1,14,sizeof(dockpars.elec_min_distance),             &dockpars.elec_min_distance);
	setKernelArg(tData.kernel1,15,sizeof(dockpars.coeff_desolv),                  &dockpars.coeff_desolv);
	setKernelArg(tData.kernel1,16,sizeof(mem_dockpars_conformations_current),     &mem_dockpars_conformations_current);
	setKernelArg(tData.kernel1,17,sizeof(mem_dockpars_energies_current),          &mem_dockpars_energies_current);
	setKernelArg(tData.kernel1,18,sizeof(mem_dockpars_evals_of_new_entities),     &mem_dockpars_evals_of_new_entities);
	setKernelArg(tData.kernel1,19,sizeof(dockpars.pop_size),                      &dockpars.pop_size);
	setKernelArg(tData.kernel1,20,sizeof(dockpars.qasp),                          &dockpars.qasp);
	setKernelArg(tData.kernel1,21,sizeof(dockpars.smooth),                        &dockpars.smooth);
	setKernelArg(tData.kernel1,22,sizeof(cData.mem_interintra_const),             &cData.mem_interintra_const);
	setKernelArg(tData.kernel1,23,sizeof(cData.mem_intracontrib_const),           &cData.mem_intracontrib_const);
	setKernelArg(tData.kernel1,24,sizeof(cData.mem_intra_const),                  &cData.mem_intra_const);
	setKernelArg(tData.kernel1,25,sizeof(cData.mem_rotlist_const),                &cData.mem_rotlist_const);
	setKernelArg(tData.kernel1,26,sizeof(cData.mem_conform_const),                &cData.mem_conform_const);
	kernel1_gxsize = blocksPerGridForEachEntity * threadsPerBlock;
	kernel1_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	printf("%-25s %10s %8u %10s %4u\n", "K_INIT", "gSize: ", kernel1_gxsize, "lSize: ", kernel1_lxsize); fflush(stdout);
#endif
	// End of Kernel1

	// Kernel2
	setKernelArg(tData.kernel2,0,sizeof(dockpars.pop_size),                       &dockpars.pop_size);
	setKernelArg(tData.kernel2,1,sizeof(mem_dockpars_evals_of_new_entities),      &mem_dockpars_evals_of_new_entities);
	setKernelArg(tData.kernel2,2,sizeof(mem_gpu_evals_of_runs),                   &mem_gpu_evals_of_runs);
	kernel2_gxsize = blocksPerGridForEachRun * threadsPerBlock;
	kernel2_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	printf("%-25s %10s %8u %10s %4u\n", "K_EVAL", "gSize: ", kernel2_gxsize, "lSize: ",  kernel2_lxsize); fflush(stdout);
#endif
	// End of Kernel2

	// Kernel4
	setKernelArg(tData.kernel4,0, sizeof(dockpars.num_of_atoms),                  &dockpars.num_of_atoms);
	setKernelArg(tData.kernel4,1, sizeof(dockpars.true_ligand_atoms),             &dockpars.true_ligand_atoms);
	setKernelArg(tData.kernel4,2, sizeof(dockpars.num_of_atypes),                 &dockpars.num_of_atypes);
	setKernelArg(tData.kernel4,3, sizeof(dockpars.num_of_map_atypes),             &dockpars.num_of_map_atypes);
	setKernelArg(tData.kernel4,4, sizeof(dockpars.num_of_intraE_contributors),    &dockpars.num_of_intraE_contributors);
	setKernelArg(tData.kernel4,5, sizeof(dockpars.gridsize_x),                    &dockpars.gridsize_x);
	setKernelArg(tData.kernel4,6, sizeof(dockpars.gridsize_y),                    &dockpars.gridsize_y);
	setKernelArg(tData.kernel4,7, sizeof(dockpars.gridsize_z),                    &dockpars.gridsize_z);
	setKernelArg(tData.kernel4,8, sizeof(g2),                                     &g2);
	setKernelArg(tData.kernel4,9, sizeof(g3),                                     &g3);
	setKernelArg(tData.kernel4,10,sizeof(dockpars.grid_spacing),                  &dockpars.grid_spacing);
	setKernelArg(tData.kernel4,11,sizeof(tData.pMem_fgrids),                      &tData.pMem_fgrids);
	setKernelArg(tData.kernel4,12,sizeof(dockpars.rotbondlist_length),            &dockpars.rotbondlist_length);
	setKernelArg(tData.kernel4,13,sizeof(dockpars.coeff_elec),                    &dockpars.coeff_elec);
	setKernelArg(tData.kernel4,14,sizeof(dockpars.elec_min_distance),             &dockpars.elec_min_distance);
	setKernelArg(tData.kernel4,15,sizeof(dockpars.coeff_desolv),                  &dockpars.coeff_desolv);
	setKernelArg(tData.kernel4,16,sizeof(mem_dockpars_conformations_current),     &mem_dockpars_conformations_current);
	setKernelArg(tData.kernel4,17,sizeof(mem_dockpars_energies_current),          &mem_dockpars_energies_current);
	setKernelArg(tData.kernel4,18,sizeof(mem_dockpars_conformations_next),        &mem_dockpars_conformations_next);
	setKernelArg(tData.kernel4,19,sizeof(mem_dockpars_energies_next),             &mem_dockpars_energies_next);
	setKernelArg(tData.kernel4,20,sizeof(mem_dockpars_evals_of_new_entities),     &mem_dockpars_evals_of_new_entities);
	setKernelArg(tData.kernel4,21,sizeof(mem_dockpars_prng_states),               &mem_dockpars_prng_states);
	setKernelArg(tData.kernel4,22,sizeof(dockpars.pop_size),                      &dockpars.pop_size);
	setKernelArg(tData.kernel4,23,sizeof(dockpars.num_of_genes),                  &dockpars.num_of_genes);
	setKernelArg(tData.kernel4,24,sizeof(dockpars.tournament_rate),               &dockpars.tournament_rate);
	setKernelArg(tData.kernel4,25,sizeof(dockpars.crossover_rate),                &dockpars.crossover_rate);
	setKernelArg(tData.kernel4,26,sizeof(dockpars.mutation_rate),                 &dockpars.mutation_rate);
	setKernelArg(tData.kernel4,27,sizeof(dockpars.abs_max_dmov),                  &dockpars.abs_max_dmov);
	setKernelArg(tData.kernel4,28,sizeof(dockpars.abs_max_dang),                  &dockpars.abs_max_dang);
	setKernelArg(tData.kernel4,29,sizeof(dockpars.qasp),                          &dockpars.qasp);
	setKernelArg(tData.kernel4,30,sizeof(dockpars.smooth),                        &dockpars.smooth);

	setKernelArg(tData.kernel4,31,sizeof(cData.mem_interintra_const),             &cData.mem_interintra_const);
	setKernelArg(tData.kernel4,32,sizeof(cData.mem_intracontrib_const),           &cData.mem_intracontrib_const);
	setKernelArg(tData.kernel4,33,sizeof(cData.mem_intra_const),                  &cData.mem_intra_const);
	setKernelArg(tData.kernel4,34,sizeof(cData.mem_rotlist_const),                &cData.mem_rotlist_const);
	setKernelArg(tData.kernel4,35,sizeof(cData.mem_conform_const),                &cData.mem_conform_const);
	kernel4_gxsize = blocksPerGridForEachEntity * threadsPerBlock;
	kernel4_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	printf("%-25s %10s %8u %10s %4u\n", "K_GA_GENERATION", "gSize: ",  kernel4_gxsize, "lSize: ", kernel4_lxsize); fflush(stdout);
#endif
	// End of Kernel4


	if (dockpars.lsearch_rate != 0.0f) {

		if ((strcmp(mypars->ls_method, "sw") == 0) || (mypars->initial_sw_generations>0)) {
			// Kernel3
			setKernelArg(tData.kernel3,0, sizeof(dockpars.num_of_atoms),                  &dockpars.num_of_atoms);
			setKernelArg(tData.kernel3,1, sizeof(dockpars.true_ligand_atoms),             &dockpars.true_ligand_atoms);
			setKernelArg(tData.kernel3,2, sizeof(dockpars.num_of_atypes),                 &dockpars.num_of_atypes);
			setKernelArg(tData.kernel3,3, sizeof(dockpars.num_of_map_atypes),             &dockpars.num_of_map_atypes);
			setKernelArg(tData.kernel3,4, sizeof(dockpars.num_of_intraE_contributors),    &dockpars.num_of_intraE_contributors);
			setKernelArg(tData.kernel3,5, sizeof(dockpars.gridsize_x),                    &dockpars.gridsize_x);
			setKernelArg(tData.kernel3,6, sizeof(dockpars.gridsize_y),                    &dockpars.gridsize_y);
			setKernelArg(tData.kernel3,7, sizeof(dockpars.gridsize_z),                    &dockpars.gridsize_z);
			setKernelArg(tData.kernel3,8, sizeof(g2),                                     &g2);
			setKernelArg(tData.kernel3,9, sizeof(g3),                                     &g3);
			setKernelArg(tData.kernel3,10,sizeof(dockpars.grid_spacing),                  &dockpars.grid_spacing);
			setKernelArg(tData.kernel3,11,sizeof(tData.pMem_fgrids),                      &tData.pMem_fgrids);
			setKernelArg(tData.kernel3,12,sizeof(dockpars.rotbondlist_length),            &dockpars.rotbondlist_length);
			setKernelArg(tData.kernel3,13,sizeof(dockpars.coeff_elec),                    &dockpars.coeff_elec);
			setKernelArg(tData.kernel3,14,sizeof(dockpars.elec_min_distance),             &dockpars.elec_min_distance);
			setKernelArg(tData.kernel3,15,sizeof(dockpars.coeff_desolv),                  &dockpars.coeff_desolv);
			setKernelArg(tData.kernel3,16,sizeof(mem_dockpars_conformations_next),        &mem_dockpars_conformations_next);
			setKernelArg(tData.kernel3,17,sizeof(mem_dockpars_energies_next),             &mem_dockpars_energies_next);
			setKernelArg(tData.kernel3,18,sizeof(mem_dockpars_evals_of_new_entities),     &mem_dockpars_evals_of_new_entities);
			setKernelArg(tData.kernel3,19,sizeof(mem_dockpars_prng_states),               &mem_dockpars_prng_states);
			setKernelArg(tData.kernel3,20,sizeof(dockpars.pop_size),                      &dockpars.pop_size);
			setKernelArg(tData.kernel3,21,sizeof(dockpars.num_of_genes),                  &dockpars.num_of_genes);
			setKernelArg(tData.kernel3,22,sizeof(dockpars.lsearch_rate),                  &dockpars.lsearch_rate);
			setKernelArg(tData.kernel3,23,sizeof(dockpars.num_of_lsentities),             &dockpars.num_of_lsentities);
			setKernelArg(tData.kernel3,24,sizeof(dockpars.rho_lower_bound),               &dockpars.rho_lower_bound);
			setKernelArg(tData.kernel3,25,sizeof(dockpars.base_dmov_mul_sqrt3),           &dockpars.base_dmov_mul_sqrt3);
			setKernelArg(tData.kernel3,26,sizeof(dockpars.base_dang_mul_sqrt3),           &dockpars.base_dang_mul_sqrt3);
			setKernelArg(tData.kernel3,27,sizeof(dockpars.cons_limit),                    &dockpars.cons_limit);
			setKernelArg(tData.kernel3,28,sizeof(dockpars.max_num_of_iters),              &dockpars.max_num_of_iters);
			setKernelArg(tData.kernel3,29,sizeof(dockpars.qasp),                          &dockpars.qasp);
			setKernelArg(tData.kernel3,30,sizeof(dockpars.smooth),                        &dockpars.smooth);

			setKernelArg(tData.kernel3,31,sizeof(cData.mem_interintra_const),             &cData.mem_interintra_const);
			setKernelArg(tData.kernel3,32,sizeof(cData.mem_intracontrib_const),           &cData.mem_intracontrib_const);
			setKernelArg(tData.kernel3,33,sizeof(cData.mem_intra_const),                  &cData.mem_intra_const);
			setKernelArg(tData.kernel3,34,sizeof(cData.mem_rotlist_const),                &cData.mem_rotlist_const);
			setKernelArg(tData.kernel3,35,sizeof(cData.mem_conform_const),                &cData.mem_conform_const);
			kernel3_gxsize = blocksPerGridForEachLSEntity * threadsPerBlock;
			kernel3_lxsize = threadsPerBlock;
			#ifdef DOCK_DEBUG
			printf("%-25s %10s %8u %10s %4u\n", "K_LS_SOLISWETS", "gSize: ", kernel3_gxsize, "lSize: ", kernel3_lxsize); fflush(stdout);
			#endif
			// End of Kernel3
		}
		if (strcmp(mypars->ls_method, "sd") == 0) {
			// Kernel5
			setKernelArg(tData.kernel5,0, sizeof(dockpars.num_of_atoms),                   &dockpars.num_of_atoms);
			setKernelArg(tData.kernel5,1, sizeof(dockpars.true_ligand_atoms),              &dockpars.true_ligand_atoms);
			setKernelArg(tData.kernel5,2, sizeof(dockpars.num_of_atypes),                  &dockpars.num_of_atypes);
			setKernelArg(tData.kernel5,3, sizeof(dockpars.num_of_map_atypes),              &dockpars.num_of_map_atypes);
			setKernelArg(tData.kernel5,4, sizeof(dockpars.num_of_intraE_contributors),     &dockpars.num_of_intraE_contributors);
			setKernelArg(tData.kernel5,5, sizeof(dockpars.gridsize_x),                     &dockpars.gridsize_x);
			setKernelArg(tData.kernel5,6, sizeof(dockpars.gridsize_y),                     &dockpars.gridsize_y);
			setKernelArg(tData.kernel5,7, sizeof(dockpars.gridsize_z),                     &dockpars.gridsize_z);
			setKernelArg(tData.kernel5,8, sizeof(g2),                                      &g2);
			setKernelArg(tData.kernel5,9, sizeof(g3),                                      &g3);
			setKernelArg(tData.kernel5,10,sizeof(dockpars.grid_spacing),                   &dockpars.grid_spacing);
			setKernelArg(tData.kernel5,11,sizeof(tData.pMem_fgrids),                       &tData.pMem_fgrids);
			setKernelArg(tData.kernel5,12,sizeof(dockpars.rotbondlist_length),             &dockpars.rotbondlist_length);
			setKernelArg(tData.kernel5,13,sizeof(dockpars.coeff_elec),                     &dockpars.coeff_elec);
			setKernelArg(tData.kernel5,14,sizeof(dockpars.elec_min_distance),              &dockpars.elec_min_distance);
			setKernelArg(tData.kernel5,15,sizeof(dockpars.coeff_desolv),                   &dockpars.coeff_desolv);
			setKernelArg(tData.kernel5,16,sizeof(mem_dockpars_conformations_next),         &mem_dockpars_conformations_next);
			setKernelArg(tData.kernel5,17,sizeof(mem_dockpars_energies_next),              &mem_dockpars_energies_next);
			setKernelArg(tData.kernel5,18,sizeof(mem_dockpars_evals_of_new_entities),      &mem_dockpars_evals_of_new_entities);
			setKernelArg(tData.kernel5,19,sizeof(mem_dockpars_prng_states),                &mem_dockpars_prng_states);
			setKernelArg(tData.kernel5,20,sizeof(dockpars.pop_size),                       &dockpars.pop_size);
			setKernelArg(tData.kernel5,21,sizeof(dockpars.num_of_genes),                   &dockpars.num_of_genes);
			setKernelArg(tData.kernel5,22,sizeof(dockpars.lsearch_rate),                   &dockpars.lsearch_rate);
			setKernelArg(tData.kernel5,23,sizeof(dockpars.num_of_lsentities),              &dockpars.num_of_lsentities);
			setKernelArg(tData.kernel5,24,sizeof(dockpars.max_num_of_iters),               &dockpars.max_num_of_iters);
			setKernelArg(tData.kernel5,25,sizeof(dockpars.qasp),                           &dockpars.qasp);
			setKernelArg(tData.kernel5,26,sizeof(dockpars.smooth),                         &dockpars.smooth);

			setKernelArg(tData.kernel5,27,sizeof(cData.mem_interintra_const),              &cData.mem_interintra_const);
			setKernelArg(tData.kernel5,28,sizeof(cData.mem_intracontrib_const),            &cData.mem_intracontrib_const);
			setKernelArg(tData.kernel5,29,sizeof(cData.mem_intra_const),                   &cData.mem_intra_const);
			setKernelArg(tData.kernel5,30,sizeof(cData.mem_rotlist_const),                 &cData.mem_rotlist_const);
			setKernelArg(tData.kernel5,31,sizeof(cData.mem_conform_const),                 &cData.mem_conform_const);

			setKernelArg(tData.kernel5,32,sizeof(cData.mem_rotbonds_const),                &cData.mem_rotbonds_const);
			setKernelArg(tData.kernel5,33,sizeof(cData.mem_rotbonds_atoms_const),          &cData.mem_rotbonds_atoms_const);
			setKernelArg(tData.kernel5,34,sizeof(cData.mem_num_rotating_atoms_per_rotbond_const), &cData.mem_num_rotating_atoms_per_rotbond_const);
			setKernelArg(tData.kernel5,35,sizeof(cData.mem_angle_const),                   &cData.mem_angle_const);
			setKernelArg(tData.kernel5,36,sizeof(cData.mem_dependence_on_theta_const),     &cData.mem_dependence_on_theta_const);
			setKernelArg(tData.kernel5,37,sizeof(cData.mem_dependence_on_rotangle_const),  &cData.mem_dependence_on_rotangle_const);

			kernel5_gxsize = blocksPerGridForEachGradMinimizerEntity * threadsPerBlock;
			kernel5_lxsize = threadsPerBlock;
			#ifdef DOCK_DEBUG
			printf("%-25s %10s %8u %10s %4u\n", "K_LS_GRAD_SDESCENT", "gSize: ", kernel5_gxsize, "lSize: ", kernel5_lxsize); fflush(stdout);
			#endif
			// End of Kernel5
		}
		if (strcmp(mypars->ls_method, "fire") == 0) {
			// Kernel6
			setKernelArg(tData.kernel6,0, sizeof(dockpars.num_of_atoms),                   &dockpars.num_of_atoms);
			setKernelArg(tData.kernel6,1, sizeof(dockpars.true_ligand_atoms),              &dockpars.true_ligand_atoms);
			setKernelArg(tData.kernel6,2, sizeof(dockpars.num_of_atypes),                  &dockpars.num_of_atypes);
			setKernelArg(tData.kernel6,3, sizeof(dockpars.num_of_map_atypes),              &dockpars.num_of_map_atypes);
			setKernelArg(tData.kernel6,4, sizeof(dockpars.num_of_intraE_contributors),     &dockpars.num_of_intraE_contributors);
			setKernelArg(tData.kernel6,5, sizeof(dockpars.gridsize_x),                     &dockpars.gridsize_x);
			setKernelArg(tData.kernel6,6, sizeof(dockpars.gridsize_y),                     &dockpars.gridsize_y);
			setKernelArg(tData.kernel6,7, sizeof(dockpars.gridsize_z),                     &dockpars.gridsize_z);
			setKernelArg(tData.kernel6,8, sizeof(g2),                                      &g2);
			setKernelArg(tData.kernel6,9, sizeof(g3),                                      &g3);
			setKernelArg(tData.kernel6,10,sizeof(dockpars.grid_spacing),                   &dockpars.grid_spacing);
			setKernelArg(tData.kernel6,11,sizeof(tData.pMem_fgrids),                       &tData.pMem_fgrids);
			setKernelArg(tData.kernel6,12,sizeof(dockpars.rotbondlist_length),             &dockpars.rotbondlist_length);
			setKernelArg(tData.kernel6,13,sizeof(dockpars.coeff_elec),                     &dockpars.coeff_elec);
			setKernelArg(tData.kernel6,14,sizeof(dockpars.elec_min_distance),              &dockpars.elec_min_distance);
			setKernelArg(tData.kernel6,15,sizeof(dockpars.coeff_desolv),                   &dockpars.coeff_desolv);
			setKernelArg(tData.kernel6,16,sizeof(mem_dockpars_conformations_next),         &mem_dockpars_conformations_next);
			setKernelArg(tData.kernel6,17,sizeof(mem_dockpars_energies_next),              &mem_dockpars_energies_next);
			setKernelArg(tData.kernel6,18,sizeof(mem_dockpars_evals_of_new_entities),      &mem_dockpars_evals_of_new_entities);
			setKernelArg(tData.kernel6,19,sizeof(mem_dockpars_prng_states),                &mem_dockpars_prng_states);
			setKernelArg(tData.kernel6,20,sizeof(dockpars.pop_size),                       &dockpars.pop_size);
			setKernelArg(tData.kernel6,21,sizeof(dockpars.num_of_genes),                   &dockpars.num_of_genes);
			setKernelArg(tData.kernel6,22,sizeof(dockpars.lsearch_rate),                   &dockpars.lsearch_rate);
			setKernelArg(tData.kernel6,23,sizeof(dockpars.num_of_lsentities),              &dockpars.num_of_lsentities);
			setKernelArg(tData.kernel6,24,sizeof(dockpars.max_num_of_iters),               &dockpars.max_num_of_iters);
			setKernelArg(tData.kernel6,25,sizeof(dockpars.qasp),                           &dockpars.qasp);
			setKernelArg(tData.kernel6,26,sizeof(dockpars.smooth),                         &dockpars.smooth);
			setKernelArg(tData.kernel6,27,sizeof(cData.mem_interintra_const),              &cData.mem_interintra_const);
			setKernelArg(tData.kernel6,28,sizeof(cData.mem_intracontrib_const),            &cData.mem_intracontrib_const);
			setKernelArg(tData.kernel6,29,sizeof(cData.mem_intra_const),                   &cData.mem_intra_const);
			setKernelArg(tData.kernel6,30,sizeof(cData.mem_rotlist_const),                 &cData.mem_rotlist_const);
			setKernelArg(tData.kernel6,31,sizeof(cData.mem_conform_const),                 &cData.mem_conform_const);

			setKernelArg(tData.kernel6,32,sizeof(cData.mem_rotbonds_const),                &cData.mem_rotbonds_const);
			setKernelArg(tData.kernel6,33,sizeof(cData.mem_rotbonds_atoms_const),          &cData.mem_rotbonds_atoms_const);
			setKernelArg(tData.kernel6,34,sizeof(cData.mem_num_rotating_atoms_per_rotbond_const), &cData.mem_num_rotating_atoms_per_rotbond_const);
			setKernelArg(tData.kernel6,35,sizeof(cData.mem_angle_const),                   &cData.mem_angle_const);
			setKernelArg(tData.kernel6,36,sizeof(cData.mem_dependence_on_theta_const),     &cData.mem_dependence_on_theta_const);
			setKernelArg(tData.kernel6,37,sizeof(cData.mem_dependence_on_rotangle_const),  &cData.mem_dependence_on_rotangle_const);
			kernel6_gxsize = blocksPerGridForEachGradMinimizerEntity * threadsPerBlock;
			kernel6_lxsize = threadsPerBlock;
			#ifdef DOCK_DEBUG
			printf("%-25s %10s %8u %10s %4u\n", "K_LS_GRAD_FIRE", "gSize: ", kernel6_gxsize, "lSize: ", kernel6_lxsize); fflush(stdout);
			#endif
			// End of Kernel6
		}
		if (strcmp(mypars->ls_method, "ad") == 0) {
			// Kernel7
			setKernelArg(tData.kernel7,0, sizeof(dockpars.num_of_atoms),                   &dockpars.num_of_atoms);
			setKernelArg(tData.kernel7,1, sizeof(dockpars.true_ligand_atoms),              &dockpars.true_ligand_atoms);
			setKernelArg(tData.kernel7,2, sizeof(dockpars.num_of_atypes),                  &dockpars.num_of_atypes);
			setKernelArg(tData.kernel7,3, sizeof(dockpars.num_of_map_atypes),              &dockpars.num_of_map_atypes);
			setKernelArg(tData.kernel7,4, sizeof(dockpars.num_of_intraE_contributors),     &dockpars.num_of_intraE_contributors);
			setKernelArg(tData.kernel7,5, sizeof(dockpars.gridsize_x),                     &dockpars.gridsize_x);
			setKernelArg(tData.kernel7,6, sizeof(dockpars.gridsize_y),                     &dockpars.gridsize_y);
			setKernelArg(tData.kernel7,7, sizeof(dockpars.gridsize_z),                     &dockpars.gridsize_z);
			setKernelArg(tData.kernel7,8, sizeof(g2),                                      &g2);
			setKernelArg(tData.kernel7,9, sizeof(g3),                                      &g3);
			setKernelArg(tData.kernel7,10,sizeof(dockpars.grid_spacing),                   &dockpars.grid_spacing);
			setKernelArg(tData.kernel7,11,sizeof(tData.pMem_fgrids),                       &tData.pMem_fgrids);
			setKernelArg(tData.kernel7,12,sizeof(dockpars.rotbondlist_length),             &dockpars.rotbondlist_length);
			setKernelArg(tData.kernel7,13,sizeof(dockpars.coeff_elec),                     &dockpars.coeff_elec);
			setKernelArg(tData.kernel7,14,sizeof(dockpars.elec_min_distance),              &dockpars.elec_min_distance);
			setKernelArg(tData.kernel7,15,sizeof(dockpars.coeff_desolv),                   &dockpars.coeff_desolv);
			setKernelArg(tData.kernel7,16,sizeof(mem_dockpars_conformations_next),         &mem_dockpars_conformations_next);
			setKernelArg(tData.kernel7,17,sizeof(mem_dockpars_energies_next),              &mem_dockpars_energies_next);
			setKernelArg(tData.kernel7,18,sizeof(mem_dockpars_evals_of_new_entities),      &mem_dockpars_evals_of_new_entities);
			setKernelArg(tData.kernel7,19,sizeof(mem_dockpars_prng_states),                &mem_dockpars_prng_states);
			setKernelArg(tData.kernel7,20,sizeof(dockpars.pop_size),                       &dockpars.pop_size);
			setKernelArg(tData.kernel7,21,sizeof(dockpars.num_of_genes),                   &dockpars.num_of_genes);
			setKernelArg(tData.kernel7,22,sizeof(dockpars.lsearch_rate),                   &dockpars.lsearch_rate);
			setKernelArg(tData.kernel7,23,sizeof(dockpars.num_of_lsentities),              &dockpars.num_of_lsentities);
			setKernelArg(tData.kernel7,24,sizeof(dockpars.max_num_of_iters),               &dockpars.max_num_of_iters);
			setKernelArg(tData.kernel7,25,sizeof(dockpars.qasp),                           &dockpars.qasp);
			setKernelArg(tData.kernel7,26,sizeof(dockpars.smooth),                         &dockpars.smooth);

			setKernelArg(tData.kernel7,27,sizeof(cData.mem_interintra_const),              &cData.mem_interintra_const);
			setKernelArg(tData.kernel7,28,sizeof(cData.mem_intracontrib_const),            &cData.mem_intracontrib_const);
			setKernelArg(tData.kernel7,29,sizeof(cData.mem_intra_const),                   &cData.mem_intra_const);
			setKernelArg(tData.kernel7,30,sizeof(cData.mem_rotlist_const),                 &cData.mem_rotlist_const);
			setKernelArg(tData.kernel7,31,sizeof(cData.mem_conform_const),                 &cData.mem_conform_const);

			setKernelArg(tData.kernel7,32,sizeof(cData.mem_rotbonds_const),                &cData.mem_rotbonds_const);
			setKernelArg(tData.kernel7,33,sizeof(cData.mem_rotbonds_atoms_const),          &cData.mem_rotbonds_atoms_const);
			setKernelArg(tData.kernel7,34,sizeof(cData.mem_num_rotating_atoms_per_rotbond_const), &cData.mem_num_rotating_atoms_per_rotbond_const);
			setKernelArg(tData.kernel7,35,sizeof(cData.mem_angle_const),                   &cData.mem_angle_const);
			setKernelArg(tData.kernel7,36,sizeof(cData.mem_dependence_on_theta_const),     &cData.mem_dependence_on_theta_const);
			setKernelArg(tData.kernel7,37,sizeof(cData.mem_dependence_on_rotangle_const),  &cData.mem_dependence_on_rotangle_const);
			kernel7_gxsize = blocksPerGridForEachGradMinimizerEntity * threadsPerBlock;
			kernel7_lxsize = threadsPerBlock;
			#ifdef DOCK_DEBUG
			printf("%-25s %10s %8u %10s %4u\n", "K_LS_GRAD_ADADELTA", "gSize: ", kernel7_gxsize, "lSize: ", kernel7_lxsize); fflush(stdout);
			#endif
			// End of Kernel7
		}
	} // End if (dockpars.lsearch_rate != 0.0f)

	// Kernel1
	#ifdef DOCK_DEBUG
		printf("\nExecution starts:\n\n");
		printf("%-25s", "\tK_INIT");fflush(stdout);
	#endif
	runKernel1D(tData.command_queue,tData.kernel1,kernel1_gxsize,kernel1_lxsize,&time_start_kernel,&time_end_kernel);
	#ifdef DOCK_DEBUG
		printf("%15s" ," ... Finished\n");fflush(stdout);
	#endif
	// End of Kernel1

	// Kernel2
	#ifdef DOCK_DEBUG
		printf("%-25s", "\tK_EVAL");fflush(stdout);
	#endif
	runKernel1D(tData.command_queue,tData.kernel2,kernel2_gxsize,kernel2_lxsize,&time_start_kernel,&time_end_kernel);
	#ifdef DOCK_DEBUG
		printf("%15s" ," ... Finished\n");fflush(stdout);
	#endif
	// End of Kernel2
	// ===============================================================================

	// -------- Replacing with memory maps! ------------
#if defined (MAPPED_COPY)
	int* map_cpu_evals_of_runs;
	map_cpu_evals_of_runs = (int*) memMap(tData.command_queue, mem_gpu_evals_of_runs, CL_MAP_READ, size_evals_of_runs);
#else
	memcopyBufferObjectFromDevice(tData.command_queue,sim_state.cpu_evals_of_runs.data(),mem_gpu_evals_of_runs,size_evals_of_runs);
#endif
	// -------- Replacing with memory maps! ------------
	#if 0
	generation_cnt = 1;
	#endif
	generation_cnt = 0;
	unsigned long total_evals;
	// print progress bar
	AutoStop autostop(mypars->pop_size, mypars->num_of_runs, mypars->stopstd, mypars->as_frequency);
#ifndef DOCK_DEBUG
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
#endif
	curr_progress_cnt = 0;

	// -------- Replacing with memory maps! ------------
#if defined (MAPPED_COPY)
	while ((progress = check_progress(map_cpu_evals_of_runs, generation_cnt, mypars->num_of_energy_evals, mypars->num_of_generations, mypars->num_of_runs, total_evals)) < 100.0)
#else
	while ((progress = check_progress(sim_state.cpu_evals_of_runs.data(), generation_cnt, mypars->num_of_energy_evals, mypars->num_of_generations, mypars->num_of_runs, total_evals)) < 100.0)
#endif
	// -------- Replacing with memory maps! ------------
	{
		if (mypars->autostop)
		{
			if (generation_cnt % mypars->as_frequency == 0) {
				if (generation_cnt % 2 == 0)
					memcopyBufferObjectFromDevice(tData.command_queue,sim_state.cpu_energies.data(),mem_dockpars_energies_current,size_energies);
				else
					memcopyBufferObjectFromDevice(tData.command_queue,sim_state.cpu_energies.data(),mem_dockpars_energies_next,size_energies);
				if (autostop.check_if_satisfactory(generation_cnt, sim_state.cpu_energies.data(), total_evals))
					if (total_evals>min_as_evals)
						break; // Exit loop when all conditions are satisfied
			}
		}
		else
		{
#ifdef DOCK_DEBUG
			ite_cnt++;
			printf("\nLGA iteration # %u\n", ite_cnt);
			fflush(stdout);
#endif
			// update progress bar (bar length is 50)
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
		// Kernel4
		#ifdef DOCK_DEBUG
			printf("%-25s", "\tK_GA_GENERATION");fflush(stdout);
		#endif
		runKernel1D(tData.command_queue,tData.kernel4,kernel4_gxsize,kernel4_lxsize,&time_start_kernel,&time_end_kernel);
		#ifdef DOCK_DEBUG
			printf("%15s", " ... Finished\n");fflush(stdout);
		#endif
		// End of Kernel4
		if (dockpars.lsearch_rate != 0.0f) {
			if ((strcmp(mypars->ls_method, "sw") == 0) || ((strcmp(mypars->ls_method, "ad") == 0) && (generation_cnt<mypars->initial_sw_generations))) {
				// Kernel3
				#ifdef DOCK_DEBUG
					printf("%-25s", "\tK_LS_SOLISWETS");fflush(stdout);
				#endif
				runKernel1D(tData.command_queue,tData.kernel3,kernel3_gxsize,kernel3_lxsize,&time_start_kernel,&time_end_kernel);
				#ifdef DOCK_DEBUG
					printf("%15s" ," ... Finished\n");fflush(stdout);
				#endif
				// End of Kernel3
			} else if (strcmp(mypars->ls_method, "sd") == 0) {
				// Kernel5
				#ifdef DOCK_DEBUG
					printf("%-25s", "\tK_LS_GRAD_SDESCENT");fflush(stdout);
				#endif
				runKernel1D(tData.command_queue,tData.kernel5,kernel5_gxsize,kernel5_lxsize,&time_start_kernel,&time_end_kernel);
				#ifdef DOCK_DEBUG
					printf("%15s" ," ... Finished\n");fflush(stdout);
				#endif
				// End of Kernel5
			} else if (strcmp(mypars->ls_method, "fire") == 0) {
				// Kernel6
				#ifdef DOCK_DEBUG
					printf("%-25s", "\tK_LS_GRAD_FIRE");fflush(stdout);
				#endif
				runKernel1D(tData.command_queue,tData.kernel6,kernel6_gxsize,kernel6_lxsize,&time_start_kernel,&time_end_kernel);
				#ifdef DOCK_DEBUG
					printf("%15s" ," ... Finished\n");fflush(stdout);
				#endif
				// End of Kernel6
			} else if (strcmp(mypars->ls_method, "ad") == 0) {
				// Kernel7
				#ifdef DOCK_DEBUG
					printf("%-25s", "\tK_LS_GRAD_ADADELTA");fflush(stdout);
				#endif
				runKernel1D(tData.command_queue,tData.kernel7,kernel7_gxsize,kernel7_lxsize,&time_start_kernel,&time_end_kernel);
				#ifdef DOCK_DEBUG
					printf("%15s" ," ... Finished\n");fflush(stdout);
				#endif
				// End of Kernel7
			}
		} // End if (dockpars.lsearch_rate != 0.0f)
		// -------- Replacing with memory maps! ------------
		#if defined (MAPPED_COPY)
		unmemMap(tData.command_queue,mem_gpu_evals_of_runs,map_cpu_evals_of_runs);
		#endif
		// -------- Replacing with memory maps! ------------
		// Kernel2
		#ifdef DOCK_DEBUG
			printf("%-25s", "\tK_EVAL");fflush(stdout);
		#endif
		runKernel1D(tData.command_queue,tData.kernel2,kernel2_gxsize,kernel2_lxsize,&time_start_kernel,&time_end_kernel);
		#ifdef DOCK_DEBUG
			printf("%15s" ," ... Finished\n");fflush(stdout);
		#endif
		// End of Kernel2
		// ===============================================================================
		// -------- Replacing with memory maps! ------------
#if defined (MAPPED_COPY)
		map_cpu_evals_of_runs = (int*) memMap(tData.command_queue, mem_gpu_evals_of_runs, CL_MAP_READ, size_evals_of_runs);
#else
		memcopyBufferObjectFromDevice(tData.command_queue,sim_state.cpu_evals_of_runs.data(),mem_gpu_evals_of_runs,size_evals_of_runs);
#endif
		// -------- Replacing with memory maps! ------------
		generation_cnt++;
		// ----------------------------------------------------------------------
		// ORIGINAL APPROACH: switching conformation and energy pointers
		// CURRENT APPROACH:  copy data from one buffer to another, pointers are kept the same
		// IMPROVED CURRENT APPROACH
		// Kernel arguments are changed on every iteration
		// No copy from dev glob memory to dev glob memory occurs
		// Use generation_cnt as it evolves with the main loop
		// No need to use tempfloat
		// No performance improvement wrt to "CURRENT APPROACH"

		// Kernel args exchange regions they point to
		// But never two args point to the same region of dev memory
		// NO ALIASING -> use restrict in Kernel
		if (generation_cnt % 2 == 0) { // In this configuration the program starts with generation_cnt = 0
			// Kernel 4
			setKernelArg(tData.kernel4,16,sizeof(mem_dockpars_conformations_current),             &mem_dockpars_conformations_current);
			setKernelArg(tData.kernel4,17,sizeof(mem_dockpars_energies_current),                  &mem_dockpars_energies_current);
			setKernelArg(tData.kernel4,18,sizeof(mem_dockpars_conformations_next),                &mem_dockpars_conformations_next);
			setKernelArg(tData.kernel4,19,sizeof(mem_dockpars_energies_next),                     &mem_dockpars_energies_next);
			if (dockpars.lsearch_rate != 0.0f) {
				if ((strcmp(mypars->ls_method, "sw") == 0) || ((strcmp(mypars->ls_method, "ad") == 0) && (generation_cnt<mypars->initial_sw_generations))){
					// Kernel 3
					setKernelArg(tData.kernel3,16,sizeof(mem_dockpars_conformations_next),&mem_dockpars_conformations_next);
					setKernelArg(tData.kernel3,17,sizeof(mem_dockpars_energies_next),     &mem_dockpars_energies_next);
				} else if (strcmp(mypars->ls_method, "sd") == 0) {
					// Kernel 5
					setKernelArg(tData.kernel5,16,sizeof(mem_dockpars_conformations_next),&mem_dockpars_conformations_next);
					setKernelArg(tData.kernel5,17,sizeof(mem_dockpars_energies_next),     &mem_dockpars_energies_next);
				} else if (strcmp(mypars->ls_method, "fire") == 0) {
					// Kernel 6
					setKernelArg(tData.kernel6,16,sizeof(mem_dockpars_conformations_next),&mem_dockpars_conformations_next);
					setKernelArg(tData.kernel6,17,sizeof(mem_dockpars_energies_next),     &mem_dockpars_energies_next);
				} else if (strcmp(mypars->ls_method, "ad") == 0) {
					// Kernel 7
					setKernelArg(tData.kernel7,16,sizeof(mem_dockpars_conformations_next),&mem_dockpars_conformations_next);
					setKernelArg(tData.kernel7,17,sizeof(mem_dockpars_energies_next),     &mem_dockpars_energies_next);
				}
			} // End if (dockpars.lsearch_rate != 0.0f)
		}
		else {  // Program switches pointers the first time when generation_cnt becomes 1 (as it starts from 0)
			// Kernel 4
			setKernelArg(tData.kernel4,16,sizeof(mem_dockpars_conformations_next),                   &mem_dockpars_conformations_next);
			setKernelArg(tData.kernel4,17,sizeof(mem_dockpars_energies_next),                        &mem_dockpars_energies_next);
			setKernelArg(tData.kernel4,18,sizeof(mem_dockpars_conformations_current),                &mem_dockpars_conformations_current);
			setKernelArg(tData.kernel4,19,sizeof(mem_dockpars_energies_current),                     &mem_dockpars_energies_current);
			if (dockpars.lsearch_rate != 0.0f) {
				if ((strcmp(mypars->ls_method, "sw") == 0) || ((strcmp(mypars->ls_method, "ad") == 0) && (generation_cnt<mypars->initial_sw_generations))){
					// Kernel 3
					setKernelArg(tData.kernel3,16,sizeof(mem_dockpars_conformations_current),&mem_dockpars_conformations_current);
					setKernelArg(tData.kernel3,17,sizeof(mem_dockpars_energies_current),     &mem_dockpars_energies_current);
				} else if (strcmp(mypars->ls_method, "sd") == 0) {
					// Kernel 5
					setKernelArg(tData.kernel5,16,sizeof(mem_dockpars_conformations_current),&mem_dockpars_conformations_current);
					setKernelArg(tData.kernel5,17,sizeof(mem_dockpars_energies_current),     &mem_dockpars_energies_current);
				} else if (strcmp(mypars->ls_method, "fire") == 0) {
					// Kernel 6
					setKernelArg(tData.kernel6,16,sizeof(mem_dockpars_conformations_current),&mem_dockpars_conformations_current);
					setKernelArg(tData.kernel6,17,sizeof(mem_dockpars_energies_current),     &mem_dockpars_energies_current);
				} else if (strcmp(mypars->ls_method, "ad") == 0){
					// Kernel 7
					setKernelArg(tData.kernel7,16,sizeof(mem_dockpars_conformations_current),&mem_dockpars_conformations_current);
			 		setKernelArg(tData.kernel7,17,sizeof(mem_dockpars_energies_current),     &mem_dockpars_energies_current);
				}
			} // End if (dockpars.lsearch_rate != 0.0f)
		}
		// ----------------------------------------------------------------------
		#ifdef DOCK_DEBUG
			printf("\tProgress %.3f %%\n", progress);
			fflush(stdout);
		#endif
	} // End of while-loop

	// Profiler
	profile.nev_at_stop = total_evals/mypars->num_of_runs;
	profile.autostopped = autostop.did_stop();

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
	// ===============================================================================
	// Modification based on:
	// http://www.cc.gatech.edu/~vetter/keeneland/tutorial-2012-02-20/08-opencl.pdf
	// ===============================================================================
	// processing results
	if (generation_cnt % 2 == 0) {
		memcopyBufferObjectFromDevice(tData.command_queue,cpu_final_populations,mem_dockpars_conformations_current,size_populations);
		memcopyBufferObjectFromDevice(tData.command_queue,sim_state.cpu_energies.data(),mem_dockpars_energies_current,size_energies);
	}
	else { 
		memcopyBufferObjectFromDevice(tData.command_queue,cpu_final_populations,mem_dockpars_conformations_next,size_populations);
		memcopyBufferObjectFromDevice(tData.command_queue,sim_state.cpu_energies.data(),mem_dockpars_energies_next,size_energies);
	}

	// Final autostop statistics output
	if (mypars->autostop) autostop.output_final_stddev(generation_cnt, sim_state.cpu_energies.data(), total_evals);

	printf("\n");
#if defined (DOCK_DEBUG)
	for (int cnt_pop=0;cnt_pop<size_populations/sizeof(float);cnt_pop++)
		printf("total_num_pop: %u, cpu_final_populations[%u]: %f\n",(unsigned int)(size_populations/sizeof(float)),cnt_pop,cpu_final_populations[cnt_pop]);
	for (int cnt_pop=0;cnt_pop<size_energies/sizeof(float);cnt_pop++)
		printf("total_num_energies: %u, sim_state.cpu_energies.data()[%u]: %f\n",    (unsigned int)(size_energies/sizeof(float)),cnt_pop,sim_state.cpu_energies.data()[cnt_pop]);
#endif
	// ===============================================================================
	// Assign simulation results to sim_state
	sim_state.myligand_reference = myligand_reference;
	sim_state.generation_cnt = generation_cnt;
	sim_state.sec_per_run = ELAPSEDSECS(clock_stop_docking, clock_start_docking)/mypars->num_of_runs;
	sim_state.total_evals = total_evals;

	clReleaseMemObject(mem_dockpars_conformations_current);
	clReleaseMemObject(mem_dockpars_energies_current);
	clReleaseMemObject(mem_dockpars_conformations_next);
	clReleaseMemObject(mem_dockpars_energies_next);
	clReleaseMemObject(mem_dockpars_evals_of_new_entities);
	clReleaseMemObject(mem_dockpars_prng_states);
	clReleaseMemObject(mem_gpu_evals_of_runs);

	free(cpu_prng_seeds);

	return 0;
}

double check_progress(
                      int* evals_of_runs,
                      int generation_cnt,
                      int max_num_of_evals,
                      int max_num_of_gens,
                      int num_of_runs,
                      unsigned long& total_evals
                     )
// The function checks if the stop condition of the docking is satisfied, returns 0 if no, and returns 1 if yes. The fitst
// parameter points to the array which stores the number of evaluations performed for each run. The second parameter stores
// the generations used. The other parameters describe the maximum number of energy evaluations, the maximum number of
// generations, and the number of runs, respectively. The stop condition is satisfied, if the generations used is higher
// than the maximal value, or if the average number of evaluations used is higher than the maximal value.
{
	// Stops if the sum of evals of every run reached the sum of the total number of evals

	int i;
	double evals_progress;
	double gens_progress;

	// calculating progress according to number of runs
	total_evals = 0;
	for (i=0; i<num_of_runs; i++)
		total_evals += evals_of_runs[i];

	evals_progress = (double)total_evals/((double) num_of_runs)/max_num_of_evals*100.0;

	// calculating progress according to number of generations
	gens_progress = ((double) generation_cnt)/((double) max_num_of_gens)*100.0; //std::cout<< "gens_progress: " << gens_progress <<std::endl;

	if (evals_progress > gens_progress)
		return evals_progress;
	else
		return gens_progress;
}
