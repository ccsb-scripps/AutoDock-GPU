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


#include "Kernels.h"


int createKernel(
                       cl_device_id* device_id,
                       cl_program*   program,
                 const char*         kernel_name,
                       cl_kernel*    kernel
                )
{
	cl_int err;

	// Create the compute kernel in the program we wish to run
	*kernel = clCreateKernel(*program, kernel_name, &err);

	if ((! *kernel) || (err != CL_SUCCESS)){
		printf("Error: clCreateKernel() %s %d\n", kernel_name, err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

#ifdef KERNEL_INFO_DISPLAY
	err = getKernelInfo(*kernel);
	if (err != CL_SUCCESS){
		printf("Error: getKernelInfo() %s\n", kernel_name);
		fflush(stdout);
		return EXIT_FAILURE;
	}
#endif

#ifdef KERNEL_WORK_GROUP_INFO_DISPLAY
	err = getKernelWorkGroupInfo(*kernel, device_id[0]);
	if (err != CL_SUCCESS){
		printf("Error: getKernelWorkGroupInfo() %s\n", kernel_name);
		fflush(stdout);
		return EXIT_FAILURE;
	}
#endif

	return CL_SUCCESS;
}

#ifdef KERNEL_INFO_DISPLAY
int getKernelInfo(cl_kernel kernel)
{
	cl_int err;
	char*        kernel_funct_name;
	cl_uint*     kernel_num_args;
	cl_uint*     kernel_ref_count;
	cl_context*  assoc_context; // context associated with the kernel
	cl_program*  assoc_program; // program from which the kernel was created

	size_t sizeParam;

	// ----------------------------------------------------------------------------
	// Query Function name
	printf("\n-----------------------------------------------------------------------\n");
	err = clGetKernelInfo(kernel,CL_KERNEL_FUNCTION_NAME,0,NULL,&sizeParam);

	if (err != CL_SUCCESS){
		printf("Error: clGetKernelInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	kernel_funct_name = (char*) malloc(sizeof(char) * sizeParam);
	err = clGetKernelInfo(kernel,CL_KERNEL_FUNCTION_NAME,sizeParam,kernel_funct_name,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %s \n", "CL_KERNEL_FUNCTION_NAME", kernel_funct_name);
	free(kernel_funct_name);

	// ----------------------------------------------------------------------------
	// Query Number of arguments to kernel
	err = clGetKernelInfo(kernel,CL_KERNEL_NUM_ARGS,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	kernel_num_args = (cl_uint*) malloc(sizeof(cl_uint) * sizeParam);
	err = clGetKernelInfo(kernel,CL_KERNEL_NUM_ARGS,sizeParam,kernel_num_args,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %u \n", "CL_KERNEL_NUM_ARGS", *kernel_num_args);
	free(kernel_num_args);

	// ----------------------------------------------------------------------------
	// Query Reference count of kernel
	/*
	https://www.khronos.org/registry/cl/specs/opencl-1.0.pdf
	CL_KERNEL_REFERENCE_COUNT
	The returned value should be considered stale (pasado, viejo).
	It is unsuitable for general use in applications.
	This feature is provided for identifying memory leaks.
	*/
	err = clGetKernelInfo(kernel,CL_KERNEL_REFERENCE_COUNT,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelInfo() %d\n",err);
		fflush(stdout);
	 	return EXIT_FAILURE;
	}

	kernel_ref_count = (cl_uint*) malloc(sizeof(cl_uint) * sizeParam);
	err = clGetKernelInfo(kernel,CL_KERNEL_REFERENCE_COUNT,sizeParam,kernel_ref_count,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %u \n", "CL_KERNEL_REFERENCE_COUNT", *kernel_ref_count);
	free(kernel_ref_count);

	// ----------------------------------------------------------------------------
	// Query Context associated with kernel
	err = clGetKernelInfo(kernel,CL_KERNEL_CONTEXT,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	assoc_context = (cl_context*) malloc(sizeof(cl_context) * sizeParam);
	err = clGetKernelInfo(kernel,CL_KERNEL_CONTEXT,sizeParam,assoc_context,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %x \n", "CL_KERNEL_CONTEXT", *assoc_context);
	free(assoc_context);

	// ----------------------------------------------------------------------------
	// Query Program from which the kernel was created
	err = clGetKernelInfo(kernel,CL_KERNEL_PROGRAM,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	assoc_program = (cl_program*) malloc(sizeof(cl_program) * sizeParam);
	err = clGetKernelInfo(kernel,CL_KERNEL_PROGRAM,sizeParam,assoc_program,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %x \n", "CL_KERNEL_PROGRAM", *assoc_program);
	free(assoc_program);

	return CL_SUCCESS;
}
#endif

#ifdef KERNEL_WORK_GROUP_INFO_DISPLAY
int getKernelWorkGroupInfo(
                           cl_kernel kernel,
                           cl_device_id device
                          )
{
	cl_int err;
	size_t*   ker_wg_size;
	size_t*   ker_pref_wg_size_multiple;
	cl_ulong* ker_loc_mem_size;
	cl_ulong* ker_pri_mem_size;
	size_t*   ker_comp_wg_size;

	size_t sizeParam;

	// ----------------------------------------------------------------------------
	// Query Kernel work group size
	printf("\n-----------------------------------------------------------------------\n");
	err = clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_WORK_GROUP_SIZE,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelWorkGroupInfo() %d\n",err);
		fflush(stdout);
	 	return EXIT_FAILURE;
	}

	ker_wg_size = (size_t*) malloc(sizeof(size_t) * sizeParam);
	err = clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_WORK_GROUP_SIZE,sizeParam,ker_wg_size,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelWorkGroupInfo() %d\n",err);
		fflush(stdout);
	 	return EXIT_FAILURE;
	}

	printf("  %-45s: %lu \n", "CL_KERNEL_WORK_GROUP_SIZE", *ker_wg_size);
	free(ker_wg_size);

/*
	// CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE is supported from OpenCL 2.0
	// ----------------------------------------------------------------------------
	// Query Kernel preferred work group size multiple
	err = clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){printf("Error: clGetKernelWorkGroupInfo() %d\n",err); return EXIT_FAILURE;}

	ker_pref_wg_size_multiple = (size_t*) malloc(sizeof(size_t) * sizeParam);
	err = clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,sizeParam,ker_pref_wg_size_multiple,NULL);
	if (err != CL_SUCCESS){printf("Error: clGetKernelWorkGroupInfo() %d\n",err); return EXIT_FAILURE;}

	printf("  %-45s: %lu \n", "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE", ker_pref_wg_size_multiple);  
	free(ker_pref_wg_size_multiple);
*/

	// ----------------------------------------------------------------------------
	// Query Kernel local mem size
	err = clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_LOCAL_MEM_SIZE,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelWorkGroupInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	ker_loc_mem_size = (cl_ulong*) malloc(sizeof(size_t) * sizeParam);
	err = clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_LOCAL_MEM_SIZE,sizeParam,ker_loc_mem_size,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelWorkGroupInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %llu \n", "CL_KERNEL_LOCAL_MEM_SIZE (#bytes)", *ker_loc_mem_size);  
	free(ker_loc_mem_size);

	// ----------------------------------------------------------------------------
	// Query Kernel private mem size
	err = clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_PRIVATE_MEM_SIZE,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelWorkGroupInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	ker_pri_mem_size = (cl_ulong*) malloc(sizeof(size_t) * sizeParam);
	err = clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_PRIVATE_MEM_SIZE,sizeParam,ker_pri_mem_size,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelWorkGroupInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %lu \n", "CL_KERNEL_PRIVATE_MEM_SIZE (#bytes)", *ker_pri_mem_size);  
	free(ker_pri_mem_size);

	// ----------------------------------------------------------------------------
	// Query Kernel compiler work group size
	err = clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_COMPILE_WORK_GROUP_SIZE,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelWorkGroupInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	ker_comp_wg_size = (size_t*) malloc(sizeof(size_t) * sizeParam);
	err = clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_COMPILE_WORK_GROUP_SIZE,sizeParam,ker_comp_wg_size,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetKernelWorkGroupInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %lu %lu %lu\n", "CL_KERNEL_COMPILE_WORK_GROUP_SIZE", ker_comp_wg_size[0],  ker_comp_wg_size[1],  ker_comp_wg_size[2]);  
	free(ker_comp_wg_size);

	return CL_SUCCESS;
}
#endif

int setKernelArg(
                 cl_kernel kernel,
                 cl_uint num,
                 size_t size,
                 const void *ptr
                )
{
	cl_int err;
	err = clSetKernelArg(kernel,num,size,ptr);
	if (err != CL_SUCCESS){
		printf("Error: clSetKernelArg() %d\n", err);
		fflush(stdout);
		return EXIT_FAILURE;
	}
	return CL_SUCCESS;
}

int runKernel1D(
                cl_command_queue cmd_queue,
                cl_kernel        kernel,
                size_t           gxDimSize,
                size_t           lxDimSize,
                cl_ulong*        time_start,
                cl_ulong*        time_stop
               )
{
	cl_int   err;
#ifdef CMD_QUEUE_PROFILING_ENABLE
	cl_event event;
#endif
	//cl_ulong start;
	//cl_ulong stop;
	size_t gsize = gxDimSize;
	size_t lsize = lxDimSize;

	// Enqueue kernel
#ifdef CMD_QUEUE_PROFILING_ENABLE
	err = clEnqueueNDRangeKernel(cmd_queue,kernel,1,NULL,&gsize,&lsize,0,NULL,&event);
#else
	err = clEnqueueNDRangeKernel(cmd_queue,kernel,1,NULL,&gsize,&lsize,0,NULL,NULL);
#endif
	if (err != CL_SUCCESS){
		printf("Error: clEnqueueNDRangeKernel() %d\n", err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	// Ensure to have executed all enqueued tasks

	// clFinish commented out, as this causes slow down
	// Our command queue is in-order and this is not needed
	// clFinish(cmd_queue);

#ifdef CMD_QUEUE_PROFILING_ENABLE
	// Get start and stop time
	clWaitForEvents(1,&event);
	cl_ulong start;
	cl_ulong stop;
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL);
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,  sizeof(stop),&stop, NULL);
	printf("queue %p, event %p here: %llu -> %llu (%llu)\n",cmd_queue,event,start,stop,stop-start);
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_QUEUED,sizeof(start),&start,NULL);
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_SUBMIT,  sizeof(stop),&stop, NULL);
	printf("queue %p, event %p there: %llu -> %llu\n",cmd_queue,event,start,stop);

	// Pass kernel exec time to calling function
	*time_start = start;
	*time_stop  = stop;

	clReleaseEvent(event);
#endif
	return CL_SUCCESS;
}


int runKernel2D(
                cl_command_queue cmd_queue,
                cl_kernel        kernel,
                size_t           gxDimSize,
                size_t           gyDimSize,
                size_t           lxDimSize,
                size_t           lyDimSize,
                cl_ulong*        time_start,
                cl_ulong*        time_stop
               )
{
	cl_int   err;
#ifdef CMD_QUEUE_PROFILING_ENABLE
	cl_event event;
#endif
	//cl_ulong start;
	//cl_ulong stop;
	size_t gsize[2] = {gxDimSize,gyDimSize};
	size_t lsize[2] = {lxDimSize,lyDimSize};

	// Enqueue kernel
#ifdef CMD_QUEUE_PROFILING_ENABLE
	err = clEnqueueNDRangeKernel(cmd_queue,kernel,2,NULL,gsize,lsize,0,NULL,&event);
#else
	err = clEnqueueNDRangeKernel(cmd_queue,kernel,2,NULL,gsize,lsize,0,NULL,NULL);
#endif
	if (err != CL_SUCCESS){
		printf("Error: clEnqueueNDRangeKernel() %d\n", err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	// Ensure to have executed all enqueued tasks

	// clFinish commented out, as this causes slow down
	// Our command queue is in-order and this is not needed
	//clFinish(cmd_queue);

#ifdef CMD_QUEUE_PROFILING_ENABLE
	// Get start and stop time
	cl_ulong start;
	cl_ulong stop;
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL);
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,  sizeof(stop),&stop, NULL);

	// Pass kernel exec time to calling function
	*time_start = start;
	*time_stop  = stop;

	clReleaseEvent(event);
#endif
	return CL_SUCCESS;
}


int runKernel3D(
                cl_command_queue cmd_queue,
                cl_kernel        kernel,
                size_t           gxDimSize,
                size_t           gyDimSize,
                size_t           gzDimSize,
                size_t           lxDimSize,
                size_t           lyDimSize,
                size_t           lzDimSize,
                cl_ulong*        time_start,
                cl_ulong*        time_stop
               )
{
	cl_int   err;
#ifdef CMD_QUEUE_PROFILING_ENABLE
	cl_event event;
#endif
	//cl_ulong start;
	//cl_ulong stop;
	size_t gsize[3] = {gxDimSize,gyDimSize,gzDimSize};
	size_t lsize[3] = {lxDimSize,lyDimSize,lzDimSize};

	// Enqueue kernel
#ifdef CMD_QUEUE_PROFILING_ENABLE
	err = clEnqueueNDRangeKernel(cmd_queue,kernel,3,NULL,gsize,lsize,0,NULL,&event);
#else
	err = clEnqueueNDRangeKernel(cmd_queue,kernel,3,NULL,gsize,lsize,0,NULL,NULL);
#endif
	if (err != CL_SUCCESS){
		printf("Error: clEnqueueNDRangeKernel() %d\n", err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	// Ensure to have executed all enqueued tasks

	// clFinish commented out, as this causes slow down
	// Our command queue is in-order and this is not needed
	//clFinish(cmd_queue);

#ifdef CMD_QUEUE_PROFILING_ENABLE
	// Get start and stop time
	cl_ulong start;
	cl_ulong stop;
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL);
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,  sizeof(stop),&stop, NULL);

	// Pass kernel exec time to calling function
	*time_start = start;
	*time_stop  = stop;

	clReleaseEvent(event);
#endif
	return CL_SUCCESS;
}
