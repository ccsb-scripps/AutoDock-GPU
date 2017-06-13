/*
 * (C) 2013. Evopro Innovation Kft.
 *
 * defines.h
 *
 *  Created on: 2009.05.29.
 *      Author: pechan.imre
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
#else
	#define NUM_OF_THREADS_PER_BLOCK 64
#endif

#define MAX_NUM_OF_ATOMS 				90
#define MAX_NUM_OF_ATYPES 			14
#define MAX_INTRAE_CONTRIBUTORS 8128
#define MAX_NUM_OF_ROTATIONS 		4096
#define MAX_NUM_OF_ROTBONDS 		32
#define MAX_POPSIZE 						2048
#define MAX_NUM_OF_RUNS 				100

// Must be bigger than MAX_NUM_OF_ROTBONDS+6
#define GENOTYPE_LENGTH_IN_GLOBMEM 64
#define ACTUAL_GENOTYPE_LENGTH	(MAX_NUM_OF_ROTBONDS+6)

#define LS_EXP_FACTOR 					2.0f
#define LS_CONT_FACTOR 					0.5f

// Improvements over Pechan's implementation
#define NATIVE_PRECISION
#define ASYNC_COPY
#define IMPROVE_GRID
#define RESTRICT_ARGS
#define MAPPED_COPY

#endif /* DEFINES_H_ */
