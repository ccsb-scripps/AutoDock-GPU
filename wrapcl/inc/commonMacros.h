/*

OCLADock, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.

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


#ifndef COMMON_MACROS_H
#define COMMON_MACROS_H

	//#include <malloc.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>

	//#define EXIT_FAILURE -1

// ===============================================
// Platforms -> opencl_lvs/Platforms.cpp
// ===============================================
// #define PLATFORM_ATTRIBUTES_DISPLAY

// ===============================================
// Devices -> opencl_lvs/Devices.cpp
// ===============================================
// #define DEVICE_ATTRIBUTES_DISPLAY

// ===============================================
// Contexts -> opencl_lvs/Contexts.cpp
// ===============================================
// #define CONTEXT_INFO_DISPLAY

// ===============================================
// Commands -> opencl_lvs/CommandQueue.cpp
// ===============================================
// #define CMD_QUEUE_INFO_DISPLAY
// #define CMD_QUEUE_PROFILING_ENABLE
// #define CMD_QUEUE_OUTORDER_ENABLE

// ===============================================
// Programs -> opencl_lvs/Programs.cpp
// ===============================================
// #define PROGRAM_INFO_DISPLAY
// #define PROGRAM_BUILD_INFO_DISPLAY

// ===============================================
// Kernels -> opencl_lvs/Kernels.cpp
// ===============================================
// #define KERNEL_INFO_DISPLAY
// #define KERNEL_WORK_GROUP_INFO_DISPLAY

// ===============================================
// Buffer Objects -> opencl_lvs/BufferObjects.cpp
// ===============================================
// #define BUFFER_OBJECT_INFO_DISPLAY

#endif

