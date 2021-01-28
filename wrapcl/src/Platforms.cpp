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


#include "Platforms.h"

int getPlatforms(
                 cl_platform_id** platform_id,
                 cl_uint*         platformCount
                )
{
	cl_int err;

	cl_platform_id* local_platform_id;
	cl_uint         local_platformCount;

	err = clGetPlatformIDs(0, NULL, &local_platformCount);
	if (err != CL_SUCCESS){
		printf("Error: clGetPlatformIDs(): %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

#if defined (PLATFORM_ATTRIBUTES_DISPLAY)
	printf("\n-----------------------------------------------------------------------\n");
	printf("Number of available OpenCL platforms: %d\n",local_platformCount);
#endif

	local_platform_id = (cl_platform_id*) malloc(sizeof(cl_platform_id) * local_platformCount);

	err = clGetPlatformIDs(local_platformCount, local_platform_id, NULL);
	if (err != CL_SUCCESS){
		printf("Error: clGetPlatformIDs(): %d\n",err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

#ifdef PLATFORM_ATTRIBUTES_DISPLAY
	err = getPlatformAttributes (local_platform_id, local_platformCount);
	if (err != CL_SUCCESS){
		printf("Error: getPlatformAttributes() \n");
		fflush(stdout);
		return EXIT_FAILURE;
	}
#endif

	/*http://stackoverflow.com/questions/1698660/when-i-change-a-parameter-inside-a-function-does-it-change-for-the-caller-too*/
	*platform_id   = local_platform_id;
	*platformCount = local_platformCount;

	return CL_SUCCESS;
}

#ifdef PLATFORM_ATTRIBUTES_DISPLAY
int getPlatformAttributes(
                          cl_platform_id* platform_id,
                          cl_uint platformCount
                         )
{
	cl_int err;
	cl_uint i;
	cl_uint j;
	char*  info; // Stores info (strings) from platforms and devices
	size_t infoSize;

	extern const char*            attributePlatNames[5];
	extern const cl_platform_info attributePlatTypes[5];
	extern const unsigned int     attributePlatCount;

	// Print attributes for each platform
	for (i=0; i<platformCount; i++){
		printf("  Platform number: %d\n", i+1);
		for (j=0; j<attributePlatCount; j++){
			// Get platform attribute value size
			err = clGetPlatformInfo(platform_id[i],attributePlatTypes[j],0,NULL,&infoSize);
			if (err != CL_SUCCESS){
				printf("Error: clGetPlatformInfo(): %d\n",err);
				fflush(stdout);
				return EXIT_FAILURE;
			}

			info = (char*) malloc(infoSize);

			// Get platform attribute value
			err = clGetPlatformInfo(platform_id[i],attributePlatTypes[j],infoSize,info,NULL);
			if (err != CL_SUCCESS){
				printf("Error: clGetPlatformInfo(): %d\n",err);
				fflush(stdout);
				return EXIT_FAILURE;
			}

			// Print platform and corresponding attributes
			printf("  %-45s: %s\n", attributePlatNames[j], info);
			free(info);
		} // End j-for (attributes)
	} // End i-for (platforms)

	return CL_SUCCESS;
}
#endif
