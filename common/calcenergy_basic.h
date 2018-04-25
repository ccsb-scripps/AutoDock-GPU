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


#ifndef CALCENERGY_BASIC_H_
#define CALCENERGY_BASIC_H_

#include "defines.h"

#define RLIST_ATOMID_MASK 	0x000000FF
#define RLIST_RBONDID_MASK 	0x0000FF00
#define RLIST_RBONDID_SHIFT  	8
#define RLIST_FIRSTROT_MASK  	0x00010000
#define RLIST_GENROT_MASK 	0x00020000
#define RLIST_DUMMY_MASK 	0x00040000
#define DEG_TO_RAD 		0.0174533f

// LCG: linear congruential generator constants
#define RAND_A 			1103515245u
#define RAND_C 			12345u
// WARNING: it is supposed that unsigned int is 32 bit long
#define MAX_UINT 		4294967296.0f

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

// Constants for dielelectric term of the 
// electrostatic component of the intramolecular energy/gradient
#define DIEL_A 			-8.5525f
#define DIEL_WAT 		78.4f
#define DIEL_B 			(DIEL_WAT - DIEL_A)
#define DIEL_LAMBDA		0.003627f
#define DIEL_H			DIEL_LAMBDA
#define DIEL_K			7.7839f
#define DIEL_B_TIMES_H		(DIEL_B * DIEL_H)
#define DIEL_B_TIMES_H_TIMES_K	(DIEL_B_TIMES_H * DIEL_K)

// Used for Shoemake to quaternion transformation
#define PI_TIMES_2 		(float)(2.0f*M_PI)


// -------------------------------------------
// Gradient-related defines
// -------------------------------------------

#define INFINITESIMAL_RADIAN		1E-5
#define HALF_INFINITESIMAL_RADIAN 	(0.5f * INFINITESIMAL_RADIAN)
#define INV_INFINITESIMAL_RADIAN	(1/INFINITESIMAL_RADIAN)
#define COS_HALF_INFINITESIMAL_RADIAN	cos(HALF_INFINITESIMAL_RADIAN)
#define SIN_HALF_INFINITESIMAL_RADIAN	sin(HALF_INFINITESIMAL_RADIAN)


// Enable printf statements for debugging
// the gradient-based minimizer
#define DEBUG_MINIMIZER


/*
#define TRANGENE_ALPHA 1E-3
#define ROTAGENE_ALPHA 1E-8
#define TORSGENE_ALPHA 1E-13
*/

#define STEP_INCREASE 		1.2f
#define STEP_DECREASE 		0.2f
#define STEP_START		1E3		// Might look gigantic, but will cap
#define MAX_DEV_TRANSLATION	2.0f		// 2 Angstrom, but must be divided by the gridspacing
#define MAX_DEV_ROTATION	0.2f		// Shoemake range [0, 1]
#define MAX_DEV_TORSION		0.5f/DEG_TO_RAD	// 0.5f RAD





#endif /* CALCENERGY_BASIC_H_ */
