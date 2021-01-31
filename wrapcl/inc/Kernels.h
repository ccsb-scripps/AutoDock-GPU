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


#ifndef KERNELS_H
#define KERNELS_H

#include "commonMacros.h"

int createKernel(
                 cl_device_id* device_id,
                 cl_program*   program,
                 const char*   kernel_name,
                 cl_kernel*    kernel
                );

int getKernelInfo(cl_kernel kernel);

int getKernelWorkGroupInfo(
                           cl_kernel    kernel,
                           cl_device_id device
                          );

int setKernelArg(
                       cl_kernel kernel,
                       cl_uint   num,
                       size_t    size,
                 const void*     ptr
                );

int runKernel1D(
                cl_command_queue cmd_queue,
                cl_kernel        kernel,
                size_t           gxDimSize,
                size_t           lxDimSize,
                cl_ulong*        time_start,
                cl_ulong*        time_stop
               );

int runKernel2D(
                cl_command_queue cmd_queue,
                cl_kernel        kernel,
                size_t           gxDimSize,
                size_t           gyDimSize,
                size_t           lxDimSize,
                size_t           lyDimSize,
                cl_ulong*        time_start,
                cl_ulong*        time_stop
               );

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
               );

#endif /* KERNELS_H */
