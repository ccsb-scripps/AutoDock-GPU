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


#ifndef BUFFER_OBJECTS_H
#define BUFFER_OBJECTS_H

#include "commonMacros.h"

/*
  Memory objects are represented by cl_mem objects.
  They come in two types:
  - buffer objects
  - image  objects
*/
int mallocBufferObject(
                       cl_context   context,
                       cl_mem_flags flags,
                       size_t       size,
                       cl_mem*      mem
                      );

int getBufferObjectInfo(cl_mem object);

int memcopyBufferObjectToDevice(
                                cl_command_queue cmd_queue,
                                cl_mem           dest,
                                bool             blocking,
                                void*            src,
                                size_t           size
                               );

int memcopyBufferObjectFromDevice(
                                  cl_command_queue cmd_queue,
                                  void*            dest,
                                  cl_mem           src,
                                  size_t           size
                                 );

int memcopyBufferObjectToBufferObject(
                                      cl_command_queue cmd_queue,
                                      cl_mem           dest,
                                      cl_mem           src,
                                      size_t           size
                                     );

void* memMap(
             cl_command_queue cmd_queue,
             cl_mem           dev_mem,
             cl_map_flags     flags,
             size_t           size
            );

int unmemMap(
             cl_command_queue cmd_queue,
             cl_mem           dev_mem,
             void*            host_ptr
            );

#endif /* BUFFER_OBJECTS_H */
