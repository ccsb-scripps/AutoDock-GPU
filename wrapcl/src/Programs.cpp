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


#include "Programs.h"

#ifdef PROGRAM_INFO_DISPLAY
int getProgramInfo(cl_program program)
{
	cl_uint i;
	cl_int err;

	cl_context*     context_in_program;
	cl_uint*        num_dev_in_program;
	cl_device_id*   device_ids_program;
	char*           program_source;
	size_t*         program_bin_size;
	unsigned char** program_binaries;
	cl_uint*        ref_count_program;

	size_t sizeParam;

	// ----------------------------------------------------------------------------
	// Query Context in program
	printf("\n-----------------------------------------------------------------------\n");
	err = clGetProgramInfo(program,CL_PROGRAM_CONTEXT,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	context_in_program = (cl_context*) malloc(sizeof(cl_context) * sizeParam);
	err = clGetProgramInfo(program,CL_PROGRAM_CONTEXT,sizeParam,context_in_program,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %x \n", "CL_PROGRAM_CONTEXT", *context_in_program);
	free(context_in_program);

	// ----------------------------------------------------------------------------
	// Query Number of devices targeted by the program

	err = clGetProgramInfo(program,CL_PROGRAM_NUM_DEVICES,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	num_dev_in_program = (cl_uint*) malloc(sizeof(cl_uint) * sizeParam);
	err = clGetProgramInfo(program,CL_PROGRAM_NUM_DEVICES,sizeParam,num_dev_in_program,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %u \n", "CL_PROGRAM_NUM_DEVICES", *num_dev_in_program);

	// Store num devices
	cl_uint num_dev_tmp = *num_dev_in_program;
	free(num_dev_in_program);

	// ----------------------------------------------------------------------------
	// Query Devices targeted by the program

	err = clGetProgramInfo(program,CL_PROGRAM_DEVICES,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	device_ids_program = (cl_device_id*) malloc(sizeof(cl_device_id) * sizeParam);
	err = clGetProgramInfo(program,CL_PROGRAM_DEVICES,sizeParam,device_ids_program,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	// No assumption on the num of dev
	for(i=0; i<num_dev_tmp; i++){
		printf("  %-45s: %x \n", "CL_PROGRAM_DEVICES", device_ids_program[i]);
	}
	free(device_ids_program);

	// ----------------------------------------------------------------------------
	// Query Source code of program

	err = clGetProgramInfo(program,CL_PROGRAM_SOURCE,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	program_source = (char*) malloc(sizeof(char) * sizeParam);
	err = clGetProgramInfo(program,CL_PROGRAM_SOURCE,sizeParam,program_source,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

/*
	https://www.khronos.org/registry/cl/specs/opencl-1.0.pdf
	CL_PROGRAM_SOURCE
	Returns the program source code specified by clCreateProgramWithSource.

	(This is added for the sake of completeness).
	Our program was created with  clCreateProgramWithBinary.
	So calling this api function, in this case, is  mostlikely pointless.

	NOTE:
	The value returned in sizeParam was 1!!!
*/

//	printf("  %-45s: %s \n", "CL_PROGRAM_SOURCE", *program_source);

	free(program_source);

	// ----------------------------------------------------------------------------
	// Query Size of each program's binary buffer (i.e. for each device)
	// Commented out as binaries are not used in this program
	// https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clGetProgramInfo.html
/*
	err = clGetProgramInfo(program,CL_PROGRAM_BINARY_SIZES,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	program_bin_size = (size_t*) malloc(sizeof(size_t) * sizeParam);
	err = clGetProgramInfo(program,CL_PROGRAM_BINARY_SIZES,sizeParam,program_bin_size,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	// No assumption on the num of dev
	for(i=0; i<num_dev_tmp; i++){
		printf("  %-45s: %lu \n", "CL_PROGRAM_BINARY_SIZES (#bytes)", program_bin_size[i]);
	}

	free(program_bin_size);
*/
	// ----------------------------------------------------------------------------
/*
	// REQUIRES REVISION

	// Query Binary of program

	err = clGetProgramInfo(program,CL_PROGRAM_BINARIES,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){printf("Error: clGetProgramInfo() %d\n",err); return EXIT_FAILURE;}

	program_binaries = (unsigned char**) malloc(sizeof(unsigned char*) * sizeParam);
	err = clGetProgramInfo(program,CL_PROGRAM_BINARIES,sizeParam,program_binaries,NULL);
	if (err != CL_SUCCESS){printf("Error: clGetProgramInfo() %d\n",err); return EXIT_FAILURE;}

//  https://www.khronos.org/registry/cl/specs/opencl-1.0.pdf
//  CL_PROGRAM_BINARIES
//  Returns the program binaries for all devices associated with program.
//  The binary can be the one specified by clCreateProgramWithBinary or clBuildProgram.

//  NOTE:
//  The value returned in sizeParam was 8!!!

	// No assumption on the num of dev
	for(i=0; i<num_dev_tmp; i++){
		printf("%-40s: %u \n", "Program's binary", program_binaries[i]);
	}
	free(program_binaries);
*/

	// ----------------------------------------------------------------------------
	// Query Program' reference count

	/*
	https://www.khronos.org/registry/cl/specs/opencl-1.0.pdf
	CL_PROGRAM_REFERENCE_COUNT
	The returned value should be considered stale (pasado, viejo).
	It is unsuitable for general use in applications.
	This feature is provided for identifying memory leaks.
	*/

	err = clGetProgramInfo(program,CL_PROGRAM_REFERENCE_COUNT,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	ref_count_program = (cl_uint*) malloc(sizeof(cl_uint) * sizeParam);
	err = clGetProgramInfo(program,CL_PROGRAM_REFERENCE_COUNT,sizeParam,ref_count_program,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %u \n", "CL_PROGRAM_REFERENCE_COUNT", *ref_count_program);
	free(ref_count_program);

	return CL_SUCCESS;
}
#endif

#ifdef PROGRAM_BUILD_INFO_DISPLAY
int getprogramBuildInfo(
                        cl_program program,
                        cl_device_id device
                       )
{
	cl_int err;

	cl_build_status* program_build_status;
	char*            program_build_options;
	char*            program_build_log;

	size_t sizeParam;

	// ----------------------------------------------------------------------------
	// Query Program build status
	printf("\n-----------------------------------------------------------------------\n");
	err = clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_STATUS,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramBuildInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	program_build_status = (cl_build_status*) malloc(sizeof(cl_build_status) * sizeParam);
	err = clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_STATUS,sizeParam,program_build_status,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramBuildInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %s \n", "CL_PROGRAM_BUILD_STATUS", (*program_build_status == 0)? "CL_BUILD_SUCCESS":(*program_build_status == -1)? "CL_BUILD_NONE":(*program_build_status == -2)? "CL_BUILD_ERROR":(*program_build_status == -3)? "CL_BUILD_IN_PROGRESS":"UNKNOWN");  
	free(program_build_status);

	// ----------------------------------------------------------------------------
	// Query Options used to configure the program

	err = clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_OPTIONS,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramBuildInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	program_build_options = (char*) malloc(sizeof(char) * sizeParam);
	err = clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_OPTIONS,sizeParam,program_build_options,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramBuildInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %s \n", "CL_PROGRAM_BUILD_OPTIONS", program_build_options);
	free(program_build_options);

	// ----------------------------------------------------------------------------
	// Query Build log - compiler's output

	err = clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramBuildInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	program_build_log = (char*) malloc(sizeof(char) * sizeParam);
	err = clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,sizeParam,program_build_log,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetProgramBuildInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %s \n", "CL_PROGRAM_BUILD_LOG", program_build_log);
	free(program_build_log);

	return CL_SUCCESS;
}
#endif
