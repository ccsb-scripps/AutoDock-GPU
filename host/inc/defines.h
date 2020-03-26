/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

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

// If the number of threads per block isn't specified, set to 1 for now //let Kokkos choose
#ifndef NUM_OF_THREADS_PER_BLOCK
//#define NUM_OF_THREADS_PER_BLOCK Kokkos::AUTO()
#define NUM_OF_THREADS_PER_BLOCK 1
#endif

// Indexes of atomic types used in
// host/src/processligand.cpp/get_VWpars(),
// and kernel energy & gradient calculation.
#define ATYPE_NUM		28	// 22 (initial) + 2 (CG & G0 for handling flexrings) + W (waters) + CX + NX + OX

#define ATYPE_CG_IDX		22
#define ATYPE_G0_IDX		23
#define ATYPE_W_IDX		24
#define ATYPE_CX_IDX		25
#define ATYPE_NX_IDX		26
#define ATYPE_OX_IDX		27

// Indexes of atomic types used in
// host/src/processligand.cpp/get_bonds().
// Added definition to support flexrings.
#define ATYPE_GETBONDS		20      // + CX [ Nx / Ox already accounted for ]

#define MAX_NUM_OF_ATOMS	256
#define MAX_NUM_OF_ATYPES	14
#define MAX_NUM_OF_ROTBONDS	32
#define MAX_INTRAE_CONTRIBUTORS (MAX_NUM_OF_ATOMS * MAX_NUM_OF_ATOMS)
#define MAX_NUM_OF_ROTATIONS	(MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS)
#define MAX_POPSIZE		2048
#define MAX_NUM_OF_RUNS		1000
#define MAX_NUM_GRIDPOINTS      256

// Must be bigger than MAX_NUM_OF_ROTBONDS+6
#define GENOTYPE_LENGTH_IN_GLOBMEM 64
#define ACTUAL_GENOTYPE_LENGTH	(MAX_NUM_OF_ROTBONDS+6)

#define LS_EXP_FACTOR		2.0f
#define LS_CONT_FACTOR		0.5f

// Resolution of axis correction
#define NUM_AXIS_CORRECTION     1000

// Improvements over Pechan's implementation
#define MAPPED_COPY


// Coefficients for CG-G0 pairs used in
// host/src/processligand.cpp/calc_intraE_f(),
// and in kernel energy and gradient calculation.
// Added definition to support flexrings.
#define G 50



// TODO: convert this into a program arg
//#define GRADIENT_ENABLED





#endif /* DEFINES_H_ */
