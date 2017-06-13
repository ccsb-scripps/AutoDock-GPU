#ifndef CONTEXTS_H
#define CONTEXTS_H

  // L30nardoSV
  //#include <stdio.h>
  //#include <CL/opencl.h>
  #include "commonMacros.h"

/*
Create context
Inputs:
       	platform_id -
       	num_devices -
        device_id -
Outputs:
       	context -
*/
  int createContext(
		  /*
		  const cl_context_properties* properties,
		  */
                  cl_platform_id        platform_id,
                  cl_uint               num_devices,
                  cl_device_id*   	device_id,
		  /*
                  void (*pfn_notify)(const char* errinfo,
                                     const void* private_info,
                                     size_t      cb,
                                     void*       user_data)
                  void* user_data,
		  */
		  /*
                  cl_int*               errcode_ret,
                  */
		  cl_context*           context);


/*
Get context info
Inputs:
        context -
Outputs:
        none
*/
  int getContextInfo(cl_context context);

#endif /* CONTEXTS_H */

