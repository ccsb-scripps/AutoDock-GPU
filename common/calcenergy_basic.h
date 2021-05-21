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


#ifndef CALCENERGY_BASIC_H_
#define CALCENERGY_BASIC_H_

#include "defines.h"

#define RLIST_ATOMID_MASK       0x000000FF
#define RLIST_RBONDID_MASK      0x0000FF00
#define RLIST_RBONDID_SHIFT     8
#define RLIST_FIRSTROT_MASK     0x00010000
#define RLIST_GENROT_MASK       0x00020000
#define RLIST_DUMMY_MASK        0x00040000

#define DEG_TO_RAD              0.0174533f

// LCG: linear congruential generator constants
#define RAND_A                  1103515245u
#define RAND_C                  12345u
// WARNING: it is supposed that unsigned int is 32 bit long
#define MAX_UINT                4294967296.0f

// Sticking to array boundaries
#define stick_to_bounds(x,a,b) x + (x <= a)*(a-x) + (x >= b)*(b-x)

// e^2/4pie0 in kcal/mol
#define ELEC_SCALE_FACTOR        332.06363f

// Constants for dielelectric term of the
// electrostatic component of the intramolecular energy/gradient
#define DIEL_A                   -8.5525f
#define DIEL_WAT                 78.4f
#define DIEL_B                   (DIEL_WAT - DIEL_A)
#define DIEL_LAMBDA              0.003627f
#define DIEL_H                   DIEL_LAMBDA
#define DIEL_K                   7.7839f
#define DIEL_B_TIMES_H           (DIEL_B * DIEL_H)
#define DIEL_B_TIMES_H_TIMES_K   (DIEL_B_TIMES_H * DIEL_K)

// Used for Shoemake to quaternion transformation
#if defined(M_PI)
	#define PI_FLOAT (float)(M_PI)
#else
	#define PI_FLOAT 3.14159265359f
#endif
#define PI_TIMES_2       2.0f*PI_FLOAT

// -------------------------------------------
// Gradient-related defines
// -------------------------------------------

#define INFINITESIMAL_RADIAN            1E-3f
#define HALF_INFINITESIMAL_RADIAN       (float)(0.5f * INFINITESIMAL_RADIAN)
#define INV_INFINITESIMAL_RADIAN        (1.0f/INFINITESIMAL_RADIAN)
#define COS_HALF_INFINITESIMAL_RADIAN   cos(HALF_INFINITESIMAL_RADIAN)
#define SIN_HALF_INFINITESIMAL_RADIAN   sin(HALF_INFINITESIMAL_RADIAN)
#define inv_angle_delta                 500.0f / PI_FLOAT

/*
#define TRANGENE_ALPHA 1E-3
#define ROTAGENE_ALPHA 1E-8
#define TORSGENE_ALPHA 1E-13
*/

#define STEP_INCREASE           1.2f
#define STEP_DECREASE           0.2f
#define STEP_START              1000.0f         // Starting step size. This might look gigantic but will cap
#define MAX_DEV_TRANSLATION     2.0f            // 2 Angstrom, but must be divided by the gridspacing (store in variable)
//#define MAX_DEV_ROTATION      0.2f            // Shoemake range [0, 1]
#define MAX_DEV_ROTATION        0.5f/DEG_TO_RAD // 0.5f RAD
#define MAX_DEV_TORSION         0.5f/DEG_TO_RAD // 0.5f RAD


#endif /* CALCENERGY_BASIC_H_ */
