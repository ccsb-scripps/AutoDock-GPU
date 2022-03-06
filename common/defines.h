/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.
Copyright (C) 2022 Intel Corporation

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


#ifndef DEFINES_H_
#define DEFINES_H_

#if defined (N1WI)
	#define NUM_OF_THREADS_PER_BLOCK 1
#elif defined (N2WI)
	#define NUM_OF_THREADS_PER_BLOCK 2
#elif defined (N4WI)
	#define NUM_OF_THREADS_PER_BLOCK 4
#elif defined (N8WI)
	#define NUM_OF_THREADS_PER_BLOCK 8
#elif defined (N16WI)
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
	#define NUM_OF_THREADS_PER_BLOCK 16
#endif

typedef enum
{
	idx_000 = 0,
	idx_010 = 1,
	idx_001 = 2,
	idx_011 = 3,
	idx_100 = 4,
	idx_110 = 5,
	idx_101 = 6,
	idx_111 = 7
} indices;

enum {C=0,N=1,O=2,H=3,XX=4,P=5,S=6};  // see "bond_index" in the "AD4.1_bound.dat" or "AD4_parameters.dat" file.
#define NUM_ENUM_ATOMTYPES 7 // this should be the length of the enumerated atom types above

// Indexes of atomic types used in
// host/src/processligand.cpp/get_VWpars(),
// and kernel energy & gradient calculation.
#define ATYPE_NUM                  30 // 22 (initial + Si + B) + 2 (CG & G0 for handling flexrings) + W (waters) + CX + NX + OX

#define ATYPE_CG_IDX               24
#define ATYPE_G0_IDX               25
#define ATYPE_W_IDX                26
#define ATYPE_CX_IDX               27
#define ATYPE_NX_IDX               28
#define ATYPE_OX_IDX               29

// Indexes of atomic types used in
// host/src/processligand.cpp/get_bonds().
// Added definition to support flexrings.
#define ATYPE_GETBONDS             22 // + CX [ Nx / Ox already accounted for ] + Si + B

#define MAX_NUM_OF_ATOMS           256
#define MAX_NUM_OF_ATYPES          32
// 57+6 (+1 for eventual angle in axisangle representation) = 63 or 64
#define MAX_NUM_OF_ROTBONDS        57
#define MAX_INTRAE_CONTRIBUTORS    (MAX_NUM_OF_ATOMS * MAX_NUM_OF_ATOMS)
#define MAX_NUM_OF_ROTATIONS       (MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS)
#define MAX_POPSIZE                2048
#define MAX_NUM_OF_RUNS            1000
#define MAX_NUM_GRIDPOINTS         256

// Must be larger than or equal to MAX_NUM_OF_ROTBONDS+6
#define GENOTYPE_LENGTH_IN_GLOBMEM 64
#define ACTUAL_GENOTYPE_LENGTH     (MAX_NUM_OF_ROTBONDS+6)

#define LS_EXP_FACTOR              2.0f
#define LS_CONT_FACTOR             0.5f

// Improvements over Pechan's implementation
#define MAPPED_COPY


// Coefficients for CG-G0 pairs used in
// host/src/processligand.cpp/calc_intraE_f(),
// and in kernel energy and gradient calculation.
// Added definition to support flexrings.
#define G_AD 50.0f

// Enables full floating point gradient calculation.
// Use is not advised as:
// - the determinism gradients (aka integer gradients) are much faster *and*
// - speed up the local search convergence
// Please only use for debugging
// #define FLOAT_GRADIENTS

// Use one more coefficient in the fit to the Mehler-Solmajer dielectric in energrad implementation
// Although this improves the fit (particularly for the gradient), it costs a little bit more and
// does not return better accuracy overall (default: commented out, don't use)
// #define DIEL_FIT_ABC

// Output for the -derivtype keyword
// #define DERIVTYPE_INFO

// Output for the -modpair keyword
// #define MODPAIR_INFO

#endif /* DEFINES_H_ */
