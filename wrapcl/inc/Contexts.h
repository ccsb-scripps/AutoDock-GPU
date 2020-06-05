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




























#ifndef CONTEXTS_H
#define CONTEXTS_H

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

