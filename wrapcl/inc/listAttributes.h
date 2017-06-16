#ifndef LIST_ATTRIBUTES_H
#define LIST_ATTRIBUTES_H

  // L30nardoSV
  //#include <CL/opencl.h>
  #include "commonMacros.h"
  
  // Platforms
  extern const char*            attributePlatNames[5];
  extern const cl_platform_info attributePlatTypes[5];
  extern const unsigned int     attributePlatCount;

  // Devices
  extern const char*            attributeDevNames[6];
  extern const cl_device_info   attributeDevTypes[6];
  extern const unsigned int     attributeDevCount;

  extern const char* 		attributeUIntDevNames[18];
  extern const cl_device_info 	attributeUIntDevTypes[18];
  extern const unsigned int	attributeUIntDevCount;

  extern const char* 		attributeULongDevNames[5];
  extern const cl_device_info   attributeULongDevTypes[5];
  extern const int unsigned     attributeULongDevCount;

  extern const char* 		attributeSizeTDevNames[8];
  extern const cl_device_info 	attributeSizeTDevTypes[8];
  extern const unsigned int 	attributeSizeTDevCount;

  extern const char* 		attributeBoolDevNames[5];
  extern const cl_device_info 	attributeBoolDevTypes[5];
  extern const unsigned int 	attributeBoolDevCount;

#endif /* LIST_ATTRIBUTES_H */
