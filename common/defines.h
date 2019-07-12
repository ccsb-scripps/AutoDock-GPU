

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

#define ATYPE_NUM 		22	// initial: 22
#define ATYPE_GETBONDS		16	// initial: 16
#define MAX_NUM_OF_ATOMS 	256
#define MAX_NUM_OF_ATYPES 	14
#define MAX_NUM_OF_ROTBONDS 	32
#define MAX_INTRAE_CONTRIBUTORS (MAX_NUM_OF_ATOMS * MAX_NUM_OF_ATOMS)
#define MAX_NUM_OF_ROTATIONS 	(MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS)
#define MAX_POPSIZE 		2048
#define MAX_NUM_OF_RUNS 	1000
#define MAX_NUM_GRIDPOINTS      256

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
