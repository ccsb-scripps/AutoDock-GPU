/*
 * (C) 2013. Evopro Innovation Kft.
 *
 * performdocking.h
 *
 *  Created on: 2010.05.26.
 *      Author: pechan.imre
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
