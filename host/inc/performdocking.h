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


#ifndef PERFORMDOCKING_H_
#define PERFORMDOCKING_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <math.h>

#include "processgrid.h"
#include "miscellaneous.h"
#include "processligand.h"
#include "getparameters.h"
#include "calcenergy.h"
#include "processresult.h"

#include <CL/opencl.h>
#include "commonMacros.h"
#include "listAttributes.h"
#include "Platforms.h"
#include "Devices.h"
#include "Contexts.h"
#include "CommandQueues.h"
#include "Programs.h"
#include "Kernels.h"
#include "ImportBinary.h"
#include "ImportSource.h"
#include "BufferObjects.h"

#define ELAPSEDSECS(stop,start) ((float) stop-start)/((float) CLOCKS_PER_SEC)

// Experimental TSRI gradient-based minimizer kernel argument
// Setup here (temporarily?) the gradient-based minimizer and associated parameters.
// This should be ultimately configurable by the user as program exec. flags.

typedef struct {
	float tolerance;
	unsigned int max_num_of_iters;
	float alpha;
	float h;
	float conformation_min_perturbation [ACTUAL_GENOTYPE_LENGTH];
} Gradientparameters;

int docking_with_gpu(const Gridinfo* 	mygrid,
         	     /*const*/ float* 	cpu_floatgrids,
		           Dockpars*	mypars,
		     const Liganddata* 	myligand_init,
		     const int* 	argc,
		     char**		argv,
		           clock_t 	clock_start_program);

double check_progress(int* evals_of_runs,
		      int generation_cnt,
		      int max_num_of_evals,
		      int max_num_of_gens,
		      int num_of_runs);

#endif /* PERFORMDOCKING_H_ */
