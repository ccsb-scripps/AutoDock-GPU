#ifndef CALCENERGY_BASIC_H_
#define CALCENERGY_BASIC_H_

#include "defines.h"

#define RLIST_ATOMID_MASK 	 0x000000FF
#define RLIST_RBONDID_MASK 	 0x0000FF00
#define RLIST_RBONDID_SHIFT  8
#define RLIST_FIRSTROT_MASK  0x00010000
#define RLIST_GENROT_MASK 	 0x00020000
#define RLIST_DUMMY_MASK 	 	 0x00040000
#define DEG_TO_RAD 					 0.0174533f

// LCG: linear congruential generator constants
#define RAND_A 							1103515245u
#define RAND_C 							12345u
// WARNING: it is supposed that unsigned int is 32 bit long
#define MAX_UINT 						4294967296.0f

// Macro for capturing grid values
	// Original
	#define GETGRIDVALUE(mempoi,gridsize_x,gridsize_y,gridsize_z,t,z,y,x)   *(mempoi + gridsize_x*(y + gridsize_y*(z + gridsize_z*t)) + x)

	// Optimization 1
	// #define GETGRIDVALUE_OPT(mempoi,gridsize_x,gridsize_y,mul_tmp,z,y,x)   *(mempoi + gridsize_x*(y + gridsize_y*(z + mul_tmp)) + x)

	// Optimization 2
	// Implemented direclty in the kernel code: calcenergy_fourkernels_intel.cl

// Macro for trilinear interpolation
#define TRILININTERPOL(cube, weights) (cube[0][0][0]*weights[0][0][0] +cube[1][0][0]*weights[1][0][0] +	\
				       cube[0][1][0]*weights[0][1][0] +cube[1][1][0]*weights[1][1][0] + \
				       cube[0][0][1]*weights[0][0][1] +cube[1][0][1]*weights[1][0][1] + \
				       cube[0][1][1]*weights[0][1][1] +cube[1][1][1]*weights[1][1][1])

#endif /* CALCENERGY_BASIC_H_ */
