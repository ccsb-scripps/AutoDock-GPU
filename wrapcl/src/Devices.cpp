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


#include "Devices.h"

int getDevices(
               cl_platform_id platform_id,
               cl_uint        platformCount,
               cl_device_id** device_id,
               cl_uint*       deviceCount
              )
{
	cl_int err;

	cl_device_id* local_device_id;
	cl_uint       local_deviceCount;

	/* WHEN ALL DEVICES ARE ACCESSED */

	// Access first platforms, get all devices
#if defined (DEVICE_ATTRIBUTES_DISPLAY)
	printf("\n-----------------------------------------------------------------------\n");
#endif

#if defined ALL_DEVICE
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &local_deviceCount);
#elif defined GPU_DEVICE
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &local_deviceCount);
#elif defined FPGA_DEVICE
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &local_deviceCount);
#elif defined CPU_DEVICE
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 0, NULL, &local_deviceCount);
#else
	err = 1;
	printf("The device-type specified is not recognized.\n");
	fflush(stdout);
#endif
	if (err != CL_SUCCESS){
		printf("Error: clGetDeviceIDs(): %d\n", err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

#if defined (DEVICE_ATTRIBUTES_DISPLAY)
	printf("Number of available OpenCL devices: %d \n",local_deviceCount);
#endif


	local_device_id = (cl_device_id *) malloc(sizeof(cl_device_id) * local_deviceCount);

#if defined ALL_DEVICE
	err = clGetDeviceIDs (platform_id,CL_DEVICE_TYPE_ALL,local_deviceCount,local_device_id,NULL);
#elif defined GPU_DEVICE
	err = clGetDeviceIDs (platform_id,CL_DEVICE_TYPE_GPU,local_deviceCount,local_device_id,NULL);
#elif defined FPGA_DEVICE
	err = clGetDeviceIDs (platform_id,CL_DEVICE_TYPE_ACCELERATOR,local_deviceCount,local_device_id,NULL);
#elif CPU_DEVICE
	err = clGetDeviceIDs (platform_id,CL_DEVICE_TYPE_CPU,local_deviceCount,local_device_id,NULL);
#else
	err = 1;
	printf("The device-type specified is not recognized.\n");
	fflush(stdout);
#endif

	if (err != CL_SUCCESS){
		printf("Error: clGetDevices(): %d\n", err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

#ifdef DEVICE_ATTRIBUTES_DISPLAY
	err = getDeviceAttributes(local_device_id, local_deviceCount);
	if (err != CL_SUCCESS){
		printf("Error: getDeviceAttributes() \n");
		fflush(stdout);
		return EXIT_FAILURE;
	}
#endif

	*device_id =  local_device_id;
	*deviceCount = local_deviceCount;
	//free(local_device_ids);

	return CL_SUCCESS;
}

#ifdef DEVICE_ATTRIBUTES_DISPLAY
int getDeviceAttributes(cl_device_id* device_id, cl_uint deviceCount)
{
	cl_int err;
	cl_uint j;
	cl_uint k;

/* ---------------------------------------------------------------------
	http://www.caam.rice.edu/~timwar/HPC12/OpenCL/cl_stuff.c (REVIEW)
	size_t workitemSize[3];
*/
	// Stores different info types from devices
	cl_device_type*              infoDevType;
	size_t*                      vsize;
	char*                        info;
	cl_uint*                     value;
	cl_ulong*                    valuelong;
	// size_t vsize is used again
	cl_bool*                     valuebool;
	cl_device_fp_config*         vfpconfig;
	cl_device_mem_cache_type*    vmemcachetype;
	cl_device_local_mem_type*    vlocalmemtype;
	cl_device_exec_capabilities* vexeccap;
	cl_command_queue_properties* vcmdqueueprop;

	size_t sizeParam;

	extern const char*            attributeDevNames[6];
	extern const cl_device_info   attributeDevTypes[6];
	extern const unsigned int     attributeDevCount;

	extern const char*            attributeUIntDevNames[18];
	extern const cl_device_info   attributeUIntDevTypes[18];
	extern const unsigned int     attributeUIntDevCount;

	extern const char*            attributeULongDevNames[5];
	extern const cl_device_info   attributeULongDevTypes[5];
	extern const unsigned int     attributeULongDevCount;

	extern const char*            attributeSizeTDevNames[8];
	extern const cl_device_info   attributeSizeTDevTypes[8];
	extern const unsigned int     attributeSizeTDevCount;

	extern const char*            attributeBoolDevNames[5];
	extern const cl_device_info   attributeBoolDevTypes[5];
	extern const unsigned int     attributeBoolDevCount;

	// For each device print attributes
	for (j=0; j<deviceCount; j++){
		printf("\n  Device number (DevN): %d\n", j+1);

		// ----------------------------------------------------------------------------
		// Query explicitly device type

		// Get device attribute value size
		err = clGetDeviceInfo(device_id[j],CL_DEVICE_TYPE,0,NULL,&sizeParam);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo(): %d\n", err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		// Allocate space for that size
		infoDevType = (cl_device_type*) malloc(sizeof(cl_device_type) * sizeParam);

		// Get device attribute value
		err =  clGetDeviceInfo(device_id[j],CL_DEVICE_TYPE,sizeParam,infoDevType,NULL);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo(): %d\n", err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		// Print device and corresponding device'attributes
		printf("   %d %-45s: %s \n", j+1, "CL_DEVICE_TYPE", (*infoDevType == 1)? "CL_DEVICE_TYPE_DEFAULT":(*infoDevType == 2)? "CL_DEVICE_TYPE_CPU":(*infoDevType == 4)? "CL_DEVICE_TYPE_GPU":(*infoDevType == 8)? "CL_DEVICE_TYPE_ACCELERATOR":(*infoDevType == 16)? "CL_DEVICE_TYPE_CUSTOM":"UNKNOWN");
		free(infoDevType);

		// ----------------------------------------------------------------------------
		// Query CL_DEVICE_MAX_WORK_ITEM_SIZES attribute

		err = clGetDeviceInfo(device_id[j],CL_DEVICE_MAX_WORK_ITEM_SIZES,0,NULL,&sizeParam);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		vsize = (size_t*) malloc(sizeof(size_t) * sizeParam);

		err = clGetDeviceInfo(device_id[j],CL_DEVICE_MAX_WORK_ITEM_SIZES,sizeParam,vsize,NULL);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		printf("   %d %-45s: %lu %lu %lu\n", j+1, "CL_DEVICE_MAX_WORK_ITEM_SIZES", vsize[0], vsize[1], vsize[2]);
		//free(vsize);
		// ----------------------------------------------------------------------------
		// Query CHAR attribute
		printf("  Char attributes ...\n");
		for (k=0; k<attributeDevCount; k++){
			err = clGetDeviceInfo(device_id[j],attributeDevTypes[k],0,NULL,&sizeParam);
			if (err != CL_SUCCESS){
				printf("Error: clGetDeviceInfo(): %d\n",err);
				fflush(stdout);
				return EXIT_FAILURE;
			}

			info = (char*) malloc(sizeParam);

			err = clGetDeviceInfo(device_id[j],attributeDevTypes[k],sizeParam,info,NULL);
			if (err != CL_SUCCESS){
				printf("Error: clGetDeviceInfo(): %d\n",err);
				fflush(stdout);
				return EXIT_FAILURE;
			}

			printf("   %d %-45s: %s \n", j+1, attributeDevNames[k], info);
			free(info);
		} // End k-for (CHAR attributes)

		// ----------------------------------------------------------------------------
		// Query UINT attribute
		printf("  UInt attributes ...\n");
		for (k=0; k<attributeUIntDevCount; k++){
			err = clGetDeviceInfo(device_id[j],attributeUIntDevTypes[k],0,NULL,&sizeParam);
			if (err != CL_SUCCESS){
				printf("Error: clGetDeviceInfo() %d\n",err);
				fflush(stdout);
				return EXIT_FAILURE;
			}

			value = (cl_uint*) malloc(sizeof(cl_uint) * sizeParam);

			err = clGetDeviceInfo(device_id[j],attributeUIntDevTypes[k],sizeParam,value,NULL);
			if (err != CL_SUCCESS){
				printf("Error: clGetDeviceInfo() %d\n",err);
				fflush(stdout);
				return EXIT_FAILURE;
			}
			printf("   %d %-45s: %u \n", j+1, attributeUIntDevNames[k], *value);
			free(value);
		} // End k-for (UINT attributes)

		// ----------------------------------------------------------------------------
		// Query ULONG attribute
		printf("  ULong attributes ...\n");
		for (k=0; k<attributeULongDevCount; k++){
			err = clGetDeviceInfo(device_id[j],attributeULongDevTypes[k],0,NULL,&sizeParam);
			if (err != CL_SUCCESS){
				printf("Error: clGetDeviceInfo() %d\n",err);
				fflush(stdout);
				return EXIT_FAILURE;
			}

			valuelong = (cl_ulong*) malloc(sizeof(cl_ulong) * sizeParam);

			err = clGetDeviceInfo(device_id[j],attributeULongDevTypes[k],sizeParam,valuelong,NULL);
			if (err != CL_SUCCESS){
				printf("Error: clGetDeviceInfo() %d\n",err);
				fflush(stdout);
 				return EXIT_FAILURE;
			}

			printf("   %d %-45s: %lu \n", j+1, attributeULongDevNames[k], *valuelong);
			free(valuelong);
		} // End k-for (ULONG attributes)

		// ----------------------------------------------------------------------------
		// Query SIZE_T attribute
		printf("  Size_T attributes ...\n");
		for (k=0; k<attributeSizeTDevCount; k++){
			err = clGetDeviceInfo(device_id[j],attributeSizeTDevTypes[k],0,NULL,&sizeParam);
			if (err != CL_SUCCESS){
				printf("Error: clGetDeviceInfo() %d\n",err);
				fflush(stdout);
				return EXIT_FAILURE;
			}

			vsize = (size_t*) malloc(sizeof(size_t) * sizeParam);

			err = clGetDeviceInfo(device_id[j],attributeSizeTDevTypes[k],sizeParam,vsize,NULL);
			if (err != CL_SUCCESS){
				printf("Error: clGetDeviceInfo() %d\n",err);
				fflush(stdout);
				return EXIT_FAILURE;
			}

			printf("   %d %-45s: %lu \n", j+1, attributeSizeTDevNames[k], *vsize);
			free(vsize);
		} // End k-for (SIZE_T attributes)

		// ----------------------------------------------------------------------------
		// Query BOOL attribute
		printf("  Bool attributes ...\n");
		for (k=0; k<attributeBoolDevCount; k++){
			err = clGetDeviceInfo(device_id[j],attributeBoolDevTypes[k],0,NULL,&sizeParam);
			if (err != CL_SUCCESS){
				printf("Error: clGetDeviceInfo() %d\n",err);
				fflush(stdout);
				return EXIT_FAILURE;
			}

			valuebool = (cl_bool*) malloc(sizeof(cl_bool) * sizeParam);

			err = clGetDeviceInfo(device_id[j],attributeBoolDevTypes[k],sizeParam,valuebool,NULL);
			if (err != CL_SUCCESS){
				printf("Error: clGetDeviceInfo() %d\n",err);
				fflush(stdout);
				return EXIT_FAILURE;
			}

			printf("   %d %-45s: %s \n", j+1, attributeBoolDevNames[k], *valuebool ? "true" : "false");
			free(valuebool);
		} // End k-for (BOOL attributes)

		// ----------------------------------------------------------------------------
		// Query cl_device_fp_config attribute
		printf("  cl_device_fp_config attributes ...\n");
		err = clGetDeviceInfo(device_id[j],CL_DEVICE_SINGLE_FP_CONFIG,0,NULL,&sizeParam);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		vfpconfig = (cl_device_fp_config*) malloc(sizeof(cl_device_fp_config) * sizeParam);

		err = clGetDeviceInfo(device_id[j],CL_DEVICE_SINGLE_FP_CONFIG,sizeParam,vfpconfig,NULL);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		printf("   %d %-45s: %lu \n", j+1, "CL_DEVICE_SINGLE_FP_CONFIG (bit-field)", *vfpconfig);
		free(vfpconfig);

		// ----------------------------------------------------------------------------
		// Query cl_device_mem_cache_type attribute
		printf("  cl_device_mem_cache_type attributes ...\n");
		err = clGetDeviceInfo(device_id[j],CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,0,NULL,&sizeParam);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		vmemcachetype = (cl_device_mem_cache_type*) malloc(sizeof(cl_device_mem_cache_type) * sizeParam);

		err = clGetDeviceInfo(device_id[j],CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,sizeParam,vmemcachetype,NULL);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		//printf(" %d.%d.%d %-30s: %u \n", i+1, j+1, k+1, "Type global mem cache supported", *vmemcachetype);
		printf("   %d %-45s: %s \n", j+1, "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE", (*vmemcachetype == 0)? "CL_NONE":(*vmemcachetype == 1)? "CL_READ_ONLY_CACHE":(*vmemcachetype == 2)? "CL_READ_WRITE_CACHE":"UNKNOWN");
		free(vmemcachetype);

		// ----------------------------------------------------------------------------
		// Query cl_device_local_mem_type attribute
		printf("  cl_device_local_mem_type attributes ...\n");
		err = clGetDeviceInfo(device_id[j],CL_DEVICE_LOCAL_MEM_TYPE,0,NULL,&sizeParam);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		vlocalmemtype = (cl_device_local_mem_type*) malloc(sizeof(cl_device_local_mem_type) * sizeParam);

		err = clGetDeviceInfo(device_id[j],CL_DEVICE_LOCAL_MEM_TYPE,sizeParam,vlocalmemtype,NULL);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		printf("   %d %-45s: %s \n", j+1, "CL_DEVICE_LOCAL_MEM_TYPE", (*vlocalmemtype == 1)? "CL_LOCAL":(*vlocalmemtype == 2)? "CL_GLOBAL":"UNKNOWN");
		free(vlocalmemtype);

		// ----------------------------------------------------------------------------
		// Query cl_device_exec_capabilities attribute
		printf("  cl_device_exec_capabilities attributes ...\n");
		err = clGetDeviceInfo(device_id[j],CL_DEVICE_EXECUTION_CAPABILITIES,0,NULL,&sizeParam);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		vexeccap = (cl_device_exec_capabilities*) malloc(sizeof(cl_device_exec_capabilities) * sizeParam);

		err = clGetDeviceInfo(device_id[j],CL_DEVICE_EXECUTION_CAPABILITIES,sizeParam,vexeccap,NULL);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		printf("   %d %-45s: %lu \n", j+1, "CL_DEVICE_EXECUTION_CAPABILITIES (bit-field)", *vexeccap);
		free(vexeccap);

		// ----------------------------------------------------------------------------
		// Query cl_command_queue_properties attribute
		printf("  cl_command_queue_properties attributes ...\n");
		err = clGetDeviceInfo(device_id[j],CL_DEVICE_QUEUE_PROPERTIES,0,NULL,&sizeParam);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		vcmdqueueprop = (cl_command_queue_properties*) malloc(sizeof(cl_command_queue_properties) * sizeParam);

		err = clGetDeviceInfo(device_id[j],CL_DEVICE_QUEUE_PROPERTIES,sizeParam,vcmdqueueprop,NULL);
		if (err != CL_SUCCESS){
			printf("Error: clGetDeviceInfo() %d\n",err);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		printf("   %d %-45s: %lu \n", j+1, "CL_DEVICE_QUEUE_PROPERTIES (bit-field)", *vcmdqueueprop);
		free(vcmdqueueprop);

		// ----------------------------------------------------------------------------
	} // End j-for (devices)

	return CL_SUCCESS;
}
#endif
