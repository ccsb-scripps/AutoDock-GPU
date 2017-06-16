#ifndef KERNELS_H
#define KERNELS_H

  // L30nardoSV
  //#include <stdio.h>
  //#include <CL/opencl.h>
  #include "commonMacros.h"

/*

*/
  int getKernelInfo(cl_kernel kernel);

/*

*/
  int getKernelWorkGroupInfo(cl_kernel    kernel,
                             cl_device_id device);

/*

*/
  int setKernelArg(cl_kernel   kernel,
		   cl_uint     num,
		   size_t      size,
		   const void* ptr);
/*

*/
  int runKernel1D(cl_command_queue cmd_queue,
		  cl_kernel        kernel,
		  size_t 	   gxDimSize,
		  size_t 	   lxDimSize,
		  cl_ulong* 	   time_start,
		  cl_ulong* 	   time_stop);

  int runKernel2D(cl_command_queue cmd_queue,
                  cl_kernel        kernel,
                  size_t           gxDimSize,
                  size_t           gyDimSize,
                  size_t           lxDimSize,
                  size_t           lyDimSize,
                  cl_ulong*        time_start,
                  cl_ulong*        time_stop);

  int runKernel3D(cl_command_queue cmd_queue,
                  cl_kernel        kernel,
                  size_t           gxDimSize,
                  size_t           gyDimSize,
                  size_t           gzDimSize,
                  size_t           lxDimSize,
                  size_t           lyDimSize,
                  size_t           lzDimSize,
                  cl_ulong*        time_start,
                  cl_ulong*        time_stop);

#endif /* KERNELS_H */
