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


#ifndef DEFINES_H_
#define DEFINES_H_

#if defined (N16WI)
	#define NUM_OF_THREADS_PER_BLOCK 16
#elif defined (N32WI)
	#define NUM_OF_THREADS_PER_BLOCK 32
#elif defined (N64WI)
	#define NUM_OF_THREADS_PER_BLOCK 64
#elif defined (N128WI)
	#define NUM_OF_THREADS_PER_BLOCK 128
#elif defined (N256WI)
	#define NUM_OF_THREADS_PER_BLOCK 256
#else
	#define NUM_OF_THREADS_PER_BLOCK 64
#endif

#define MAX_NUM_OF_ATOMS 	256
#define MAX_NUM_OF_ATYPES 	14
#define MAX_NUM_OF_ROTBONDS 	32
#define MAX_INTRAE_CONTRIBUTORS 4096 // ideally =MAX_NUM_OF_ATOMS * MAX_NUM_OF_ATOMS, 
				     // but too large for __local in gradient calculation
#define MAX_NUM_OF_ROTATIONS 	(MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS)
#define MAX_POPSIZE 		2048
#define MAX_NUM_OF_RUNS 	1000

// Must be bigger than MAX_NUM_OF_ROTBONDS+6
#define GENOTYPE_LENGTH_IN_GLOBMEM 64
#define ACTUAL_GENOTYPE_LENGTH	(MAX_NUM_OF_ROTBONDS+6)

#define LS_EXP_FACTOR 		2.0f
#define LS_CONT_FACTOR 		0.5f

// Improvements over Pechan's implementation
#define MAPPED_COPY





// TODO: convert this into a program arg
//#define GRADIENT_ENABLED





#endif /* DEFINES_H_ */
