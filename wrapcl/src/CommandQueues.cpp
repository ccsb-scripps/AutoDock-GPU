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


#include "CommandQueues.h"

int createCommandQueue(
                       cl_context        context,
                       cl_device_id      device_id,
                       cl_command_queue* command_queue
                      )
{
	cl_int err;
	cl_command_queue local_cmd_queue;

/*
	// Enhanced: Profiling included
#if defined CMD_QUEUE_PROFILING_ENABLE
	local_cmd_queue = clCreateCommandQueue(context,device_id,CL_QUEUE_PROFILING_ENABLE,&err);

	// Enhanced: Out of Order Exec. included
#elif defined CMD_QUEUE_OUTORDER_ENABLE
	local_cmd_queue = clCreateCommandQueue(context,device_id,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&err);

	// Default: No Prof. No Out of Order Exec.
#else
	local_cmd_queue = clCreateCommandQueue(context,device_id, 0, &err);
#endif
*/

// -----------------------------------------------------------
// This CORRECTS above implementation.
// The command queue properties are bit-fields and therefore
// both can be enabled simultaneously. 
// -----------------------------------------------------------
	cl_command_queue_properties cmd_queue_properties = 0;

// Enabling Profiling
#if defined CMD_QUEUE_PROFILING_ENABLE
	cmd_queue_properties |= CL_QUEUE_PROFILING_ENABLE;
#endif

// Enabling Out-Of-Order
#if defined CMD_QUEUE_OUTORDER_ENABLE
	cmd_queue_properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
#endif

	local_cmd_queue = clCreateCommandQueue(context,device_id,cmd_queue_properties,&err);
// -----------------------------------------------------------

	if ((!local_cmd_queue) || (err!=CL_SUCCESS)){
		printf("Error: clCreateCommandQueue() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

#ifdef CMD_QUEUE_INFO_DISPLAY
	err = getCommandQueueInfo(local_cmd_queue);
	if (err!=CL_SUCCESS){
		printf("Error: getCommandQueueInfo() \n");
		fflush(stdout);
		return EXIT_FAILURE;
	}
#endif
	*command_queue = local_cmd_queue;
	return CL_SUCCESS;
}

#ifdef CMD_QUEUE_INFO_DISPLAY
int getCommandQueueInfo(cl_command_queue cmd_queue)
{
	cl_int err;

	cl_context*                  cmdqueue_context;
	cl_device_id*                cmdqueue_device;
	cl_uint*                     ref_count_cmdqueue;
	cl_command_queue_properties* properties_cmdqueue;

	size_t sizeParam;

	// ----------------------------------------------------------------------------
	// Query Context specified when the cmd-queue is created
	printf("\n-----------------------------------------------------------------------\n");
	err = clGetCommandQueueInfo(cmd_queue,CL_QUEUE_CONTEXT,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetCommandQueueInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	cmdqueue_context = (cl_context*) malloc(sizeof(cl_context) * sizeParam);
	err = clGetCommandQueueInfo(cmd_queue,CL_QUEUE_CONTEXT,sizeParam,cmdqueue_context,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetCommandQueueInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %x \n", "CL_QUEUE_CONTEXT", *cmdqueue_context);
	free(cmdqueue_context);

	// ----------------------------------------------------------------------------
	// Query Device specified when cmd-queue is created
	err = clGetCommandQueueInfo(cmd_queue,CL_QUEUE_DEVICE,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetCommandQueueInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	cmdqueue_device = (cl_device_id*) malloc(sizeof(cl_device_id) * sizeParam);
	err = clGetCommandQueueInfo(cmd_queue,CL_QUEUE_DEVICE,sizeParam,cmdqueue_device,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetCommandQueueInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	// Calculate the number of devices
	cl_uint i;
	cl_uint num_devices;
	num_devices = sizeof(cmdqueue_device)/sizeof(cl_device_id);

	// In case of many devices
	for(i=0; i<num_devices; i++){
		printf("  %-45s: %x \n", "CL_QUEUE_DEVICES", cmdqueue_device[i]);
	}
	free(cmdqueue_device);

	// ----------------------------------------------------------------------------
	// Query Cmd-queue reference count
	
	//https://www.khronos.org/registry/cl/specs/opencl-1.0.pdf
	//CL_QUEUE_REFERENCE_COUNT
	//The returned value should be considered immediately stale (pasado, viejo).
	//It is unsuitable for general use in applications.
	//This feature is provided for identifying memory leaks.

	err = clGetCommandQueueInfo(cmd_queue,CL_QUEUE_REFERENCE_COUNT,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetCommandQueueInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	ref_count_cmdqueue = (cl_uint*) malloc(sizeof(cl_uint) * sizeParam);
	err = clGetCommandQueueInfo(cmd_queue,CL_QUEUE_REFERENCE_COUNT,sizeParam,ref_count_cmdqueue,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetCommandQueueInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %u \n", "CL_QUEUE_REFERENCE_COUNT", *ref_count_cmdqueue);
	free(ref_count_cmdqueue);

	// ----------------------------------------------------------------------------
	// Query the currently specified properties for the cmd-queue
	// These properties are specified by properties argument in clCreateCommandQueue()
	// and can be changed in clSetCommandQueueProperty()

	err = clGetCommandQueueInfo(cmd_queue,CL_QUEUE_PROPERTIES,0,NULL,&sizeParam);
	if (err != CL_SUCCESS){
		printf("Error: clGetCommandQueueInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	properties_cmdqueue = (cl_command_queue_properties*) malloc(sizeof(cl_command_queue_properties) * sizeParam);
	err = clGetCommandQueueInfo(cmd_queue,CL_QUEUE_PROPERTIES,sizeParam,properties_cmdqueue,NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetCommandQueueInfo() %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

	printf("  %-45s: %x \n", "CL_QUEUE_PROPERTIES (bit-field)", *properties_cmdqueue);
	free(properties_cmdqueue);

	return CL_SUCCESS;
}
#endif
