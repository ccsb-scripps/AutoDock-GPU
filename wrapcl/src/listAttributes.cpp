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


#include "listAttributes.h"

/*extern*/

const char* attributePlatNames[5] = {"CL_PLATFORM_NAME","CL_PLATFORM_VENDOR","CL_PLATFORM_VERSION","CL_PLATFORM_PROFILE","CL_PLATFORM_EXTENSIONS"};

const char* attributeDevNames[6]  = {"CL_DEVICE_NAME","CL_DEVICE_VENDOR","CL_DRIVER_VERSION","CL_DEVICE_PROFILE","CL_DEVICE_VERSION","CL_DEVICE_EXTENSIONS"};

const char* attributeUIntDevNames[18] = {"CL_DEVICE_VENDOR_ID","CL_DEVICE_MAX_COMPUTE_UNITS","CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS","CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR",
                                         "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT","CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT","CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG","CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT",
                                         "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE","CL_DEVICE_MAX_CLOCK_FREQUENCY","CL_DEVICE_ADDRESS_BITS",
                                         "CL_DEVICE_MAX_READ_IMAGE_ARGS","CL_DEVICE_MAX_WRITE_IMAGE_ARGS","CL_DEVICE_MAX_SAMPLERS","CL_DEVICE_MEM_BASE_ADDR_ALIGN",
                                         "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE","CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE","CL_DEVICE_MAX_CONSTANT_ARGS"};

const char* attributeULongDevNames[5] = {"CL_DEVICE_MAX_MEM_ALLOC_SIZE","CL_DEVICE_GLOBAL_MEM_CACHE_SIZE","CL_DEVICE_GLOBAL_MEM_SIZE","CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE","CL_DEVICE_LOCAL_MEM_SIZE"};

const char* attributeSizeTDevNames[8] = {"CL_DEVICE_MAX_WORK_GROUP_SIZE","CL_DEVICE_IMAGE2D_MAX_WIDTH","CL_DEVICE_IMAGE2D_MAX_HEIGHT",
                                         "CL_DEVICE_IMAGE3D_MAX_WIDTH","CL_DEVICE_IMAGE3D_MAX_HEIGHT","CL_DEVICE_IMAGE3D_MAX_DEPTH",
                                         "CL_DEVICE_MAX_PARAMETER_SIZE","CL_DEVICE_PROFILING_TIMER_RESOLUTION"};

const char* attributeBoolDevNames[5]  = {"CL_DEVICE_IMAGE_SUPPORT","CL_DEVICE_ERROR_CORRECTION_SUPPORT",
                                         "CL_DEVICE_ENDIAN_LITTLE","CL_DEVICE_AVAILABLE","CL_DEVICE_COMPILER_AVAILABLE"};

const cl_platform_info attributePlatTypes[5] = {CL_PLATFORM_NAME,
                                                CL_PLATFORM_VENDOR,
                                                CL_PLATFORM_VERSION,
                                                CL_PLATFORM_PROFILE,
                                                CL_PLATFORM_EXTENSIONS};

const cl_device_info attributeDevTypes[6] = {CL_DEVICE_NAME,
                                             CL_DEVICE_VENDOR,
                                             CL_DRIVER_VERSION,
                                             CL_DEVICE_PROFILE,
                                             CL_DEVICE_VERSION,
                                             CL_DEVICE_EXTENSIONS};

const cl_device_info attributeUIntDevTypes[18] = {CL_DEVICE_VENDOR_ID,
                                                  CL_DEVICE_MAX_COMPUTE_UNITS,
                                                  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                                                  CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                                  CL_DEVICE_ADDRESS_BITS,
                                                  CL_DEVICE_MAX_READ_IMAGE_ARGS,
                                                  CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
                                                  CL_DEVICE_MAX_SAMPLERS,
                                                  CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                                                  CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
                                                  CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                                                  CL_DEVICE_MAX_CONSTANT_ARGS};

const cl_device_info attributeULongDevTypes[5] = {CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                                  CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                                                  CL_DEVICE_GLOBAL_MEM_SIZE,
                                                  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                                                  CL_DEVICE_LOCAL_MEM_SIZE};

const cl_device_info attributeSizeTDevTypes[8] = {CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                                  CL_DEVICE_IMAGE2D_MAX_WIDTH,
                                                  CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                                                  CL_DEVICE_IMAGE3D_MAX_WIDTH,
                                                  CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                                                  CL_DEVICE_IMAGE3D_MAX_DEPTH,
                                                  CL_DEVICE_MAX_PARAMETER_SIZE,
                                                  CL_DEVICE_PROFILING_TIMER_RESOLUTION}; /* Resolution of the device timer. Measured in nanoseconds. */

const cl_device_info attributeBoolDevTypes[5] = {CL_DEVICE_IMAGE_SUPPORT,
                                                 CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                                                 CL_DEVICE_ENDIAN_LITTLE,
                                                 CL_DEVICE_AVAILABLE,
                                                 CL_DEVICE_COMPILER_AVAILABLE};

const unsigned int attributePlatCount     = sizeof(attributePlatNames)/sizeof(char*);
const unsigned int attributeDevCount      = sizeof(attributeDevNames)/sizeof(char*);
const unsigned int attributeUIntDevCount  = sizeof(attributeUIntDevNames)/sizeof(cl_uint*);
const unsigned int attributeULongDevCount = sizeof(attributeULongDevNames)/sizeof(cl_ulong*);
const unsigned int attributeSizeTDevCount = sizeof(attributeSizeTDevNames)/sizeof(size_t*);
const unsigned int attributeBoolDevCount  = sizeof(attributeBoolDevNames)/sizeof(cl_bool*);

