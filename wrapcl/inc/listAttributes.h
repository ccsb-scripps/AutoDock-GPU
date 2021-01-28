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


#ifndef LIST_ATTRIBUTES_H
#define LIST_ATTRIBUTES_H

#include "commonMacros.h"

// Platforms
extern const char*            attributePlatNames[5];
extern const cl_platform_info attributePlatTypes[5];
extern const unsigned int     attributePlatCount;

// Devices
extern const char*            attributeDevNames[6];
extern const cl_device_info   attributeDevTypes[6];
extern const unsigned int     attributeDevCount;

extern const char*            attributeUIntDevNames[18];
extern const cl_device_info   attributeUIntDevTypes[18];
extern const unsigned int     attributeUIntDevCount;

extern const char*            attributeULongDevNames[5];
extern const cl_device_info   attributeULongDevTypes[5];
extern const int unsigned     attributeULongDevCount;

extern const char*            attributeSizeTDevNames[8];
extern const cl_device_info   attributeSizeTDevTypes[8];
extern const unsigned int     attributeSizeTDevCount;

extern const char*            attributeBoolDevNames[5];
extern const cl_device_info   attributeBoolDevTypes[5];
extern const unsigned int     attributeBoolDevCount;

#endif /* LIST_ATTRIBUTES_H */
