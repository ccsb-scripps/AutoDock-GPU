

#include "Platforms.h"

int getPlatforms(cl_platform_id** platform_id, cl_uint* platformCount){
  cl_int err;

  cl_platform_id*               local_platform_id;
  cl_uint                       local_platformCount;

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
  *platform_id =  local_platform_id;
  *platformCount = local_platformCount;

  return CL_SUCCESS;
}

#ifdef PLATFORM_ATTRIBUTES_DISPLAY
int getPlatformAttributes(cl_platform_id* platform_id, cl_uint platformCount){
  cl_int err;
  cl_uint i;
  cl_uint j;
  char*  info;                  // Stores info (strings) from platforms and devices
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
