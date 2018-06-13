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


//#include <stdio.h>
//#include <stdlib.h>
#include <time.h>

#include "processgrid.h"
//include "processresult.h"
#include "processligand.h"
#include "getparameters.h"
#include "performdocking.h"

#ifndef _WIN32
// ------------------------
// Time measurement
#include <sys/time.h>
// ------------------------
#endif

int main(int argc, char* argv[])
{
	Gridinfo 	 mygrid;
	Liganddata myligand_init;
	Dockpars   mypars;
	float*     floatgrids;

	clock_t clock_start_program, clock_stop_program;
	clock_start_program = clock();

#ifndef _WIN32
	// ------------------------
	// Time measurement
	double num_sec, num_usec, elapsed_sec;
	timeval time_start,time_end;
	gettimeofday(&time_start,NULL);
	// ------------------------
#endif
	//------------------------------------------------------------
	// Capturing names of grid parameter file and ligand pdbqt file
	//------------------------------------------------------------

	// Filling the filename and coeffs fields of mypars according to command line arguments
	if (get_filenames_and_ADcoeffs(&argc, argv, &mypars) != 0)
		return 1;

	//------------------------------------------------------------
	// Processing receptor and ligand files
	//------------------------------------------------------------

	// Filling mygrid according to the gpf file
	if (get_gridinfo(mypars.fldfile, &mygrid) != 0)
		return 1;

	// Filling the atom types filed of myligand according to the grid types
	if (init_liganddata(mypars.ligandfile, &myligand_init, &mygrid) != 0)
		return 1;

	// Filling myligand according to the pdbqt file
	if (get_liganddata(mypars.ligandfile, &myligand_init, mypars.coeffs.AD4_coeff_vdW, mypars.coeffs.AD4_coeff_hb) != 0)
		return 1;

	//Reading the grid files and storing values in the memory region pointed by floatgrids
	if (get_gridvalues_f(&mygrid, &floatgrids) != 0)
		return 1;

	//------------------------------------------------------------
	// Capturing algorithm parameters (command line args)
	//------------------------------------------------------------
	get_commandpars(&argc, argv, &(mygrid.spacing), &mypars);

	//------------------------------------------------------------
	// Calculating energies of reference ligand if required
	//------------------------------------------------------------
#if 0
	if (mypars.reflig_en_reqired == 1)
		print_ref_lig_energies_f(myligand_init, mygrid, floatgrids, mypars.coeffs.scaled_AD4_coeff_elec, mypars.coeffs.AD4_coeff_desolv, mypars.qasp);
#endif

	if (mypars.reflig_en_reqired == 1) {
		print_ref_lig_energies_f(myligand_init,
					 mypars.smooth,
					 mygrid,
					 floatgrids,
					 mypars.coeffs.scaled_AD4_coeff_elec,
					 mypars.coeffs.AD4_coeff_desolv,
					 mypars.qasp);
	}

	//------------------------------------------------------------
	// Starting Docking
	//------------------------------------------------------------
	if (docking_with_gpu(&mygrid, floatgrids, &mypars, &myligand_init, &argc, argv, clock_start_program) != 0)
		return 1;

	free(floatgrids);

#ifndef _WIN32
	// ------------------------
	// Time measurement
	gettimeofday(&time_end,NULL);
	num_sec     = time_end.tv_sec  - time_start.tv_sec;
	num_usec    = time_end.tv_usec - time_start.tv_usec;
	elapsed_sec = num_sec + (num_usec/1000000);
	printf("Program run time %.3f sec \n\n", elapsed_sec);
	//// ------------------------
#endif
	return 0;
}
