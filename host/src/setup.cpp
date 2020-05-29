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



#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "filelist.hpp"
#include "processgrid.h"
#include "processligand.h"
#include "getparameters.h"
#include "setup.hpp"

int setup(std::vector<Map>& all_maps,
	  Gridinfo&            mygrid,
	  std::vector<float>& floatgrids,
	  Dockpars&            mypars,
	  Liganddata&          myligand_init,
	  Liganddata&          myxrayligand,
	  FileList&            filelist,
	  float* fgrids_device,
	  int i_file,
	  int argc, char* argv[])
{
	//------------------------------------------------------------
	// Capturing names of grid parameter file and ligand pdbqt file
	//------------------------------------------------------------

	if(filelist.used){
		strcpy(mypars.fldfile, filelist.fld_files[i_file].c_str());
		strcpy(mypars.ligandfile, filelist.ligand_files[i_file].c_str());
	}

	// Filling the filename and coeffs fields of mypars according to command line arguments
	if (get_filenames_and_ADcoeffs(&argc, argv, &mypars, filelist.used) != 0)
		{printf("\n\nError in get_filenames_and_ADcoeffs, stopped job."); return 1;}

	//------------------------------------------------------------
	// Testing command line arguments for cgmaps parameter
	// since we need it at grid creation time
	//------------------------------------------------------------
	mypars.cgmaps = 0; // default is 0 (use one maps for every CGx or Gx atom types, respectively)
	for (unsigned int i=1; i<argc-1; i+=2)
	{
		// ----------------------------------
		//Argument: Use individual maps for CG-G0 instead of the same one
		if (strcmp("-cgmaps", argv [i]) == 0)
		{
			int tempint;
			sscanf(argv [i+1], "%d", &tempint);
			if (tempint == 0)
				mypars.cgmaps = 0;
			else
				mypars.cgmaps = 1;
		}
	}

	//------------------------------------------------------------
	// Processing receptor and ligand files
	//------------------------------------------------------------

	// Filling mygrid according to the gpf file
	if (get_gridinfo(mypars.fldfile, &mygrid) != 0)
		{printf("\n\nError in get_gridinfo, stopped job."); return 1;}

	// Filling the atom types filed of myligand according to the grid types
	if (init_liganddata(mypars.ligandfile, &myligand_init, &mygrid, mypars.cgmaps) != 0)
		{printf("\n\nError in init_liganddata, stopped job."); return 1;}

	// Filling myligand according to the pdbqt file
	if (get_liganddata(mypars.ligandfile, &myligand_init, mypars.coeffs.AD4_coeff_vdW, mypars.coeffs.AD4_coeff_hb) != 0)
		{printf("\n\nError in get_liganddata, stopped job."); return 1;}

	// Resize grid
	floatgrids.resize(4*(mygrid.num_of_atypes+2)*mygrid.size_xyz[0]*mygrid.size_xyz[1]*mygrid.size_xyz[2]);

	if (filelist.preload_maps){
		if (!filelist.maps_are_loaded) { // maps not yet loaded
			bool got_error = false;
#ifdef USE_PIPELINE
			#pragma omp critical
#endif
			{
				if (!filelist.maps_are_loaded) { // maps not yet loaded (but in critical, so only one thread will ever enter this)
					// Load maps to all_maps
					if (load_all_maps(mypars.fldfile, &mygrid, all_maps, mypars.cgmaps,fgrids_device) != 0)
                        			{got_error = true;}
					filelist.maps_are_loaded = true;
				}
			}
			// Return must be outside pragma
			if (got_error) {printf("\n\nError in load_all_maps, stopped job."); return 1;}
		}

		// Copy maps from all_maps
		if (copy_from_all_maps(&mygrid, floatgrids.data(), all_maps) != 0)
                        {printf("\n\nError in copy_from_all_maps, stopped job."); return 1;}

		// Specify total number of maps that will be on GPU
		mygrid.num_of_map_atypes = all_maps.size()-2; // For the two extra maps
		// Map atom_types used for ligand processing to all_maps so all the maps can stay on GPU
		if(map_to_all_maps(&mygrid, &myligand_init, all_maps) !=0)
			{printf("\n\nError in map_to_all_maps, stopped job."); return 1;}
	} else {
		//Reading the grid files and storing values in the memory region pointed by floatgrids
		if (get_gridvalues_f(&mygrid, floatgrids.data(), mypars.cgmaps) != 0)
			{printf("\n\nError in get_gridvalues_f, stopped job."); return 1;}
	}

	//------------------------------------------------------------
	// Capturing algorithm parameters (command line args)
	//------------------------------------------------------------
	get_commandpars(&argc, argv, &(mygrid.spacing), &mypars);

	if (filelist.resnames.size()>0){ // Overwrite resname with specified filename if specified in file list
		strcpy(mypars.resname, filelist.resnames[i_file].c_str());
	} else if (filelist.used) { // otherwise add the index to existing name distinguish the files if multiple
		std::string if_str = std::to_string(i_file);
		strcat(mypars.resname, if_str.c_str());
	}

	Gridinfo mydummygrid;
	// if -lxrayfile provided, then read xray ligand data
	if (mypars.given_xrayligandfile == true) {
		if (init_liganddata(mypars.xrayligandfile, &myxrayligand, &mydummygrid, mypars.cgmaps) != 0)
			{printf("\n\nError in init_liganddata, stopped job."); return 1;}

		if (get_liganddata(mypars.xrayligandfile, &myxrayligand, mypars.coeffs.AD4_coeff_vdW, mypars.coeffs.AD4_coeff_hb) != 0)
			{printf("\n\nError in get_liganddata, stopped job."); return 1;}
	}

	//------------------------------------------------------------
	// Calculating energies of reference ligand if required
	//------------------------------------------------------------
	if (mypars.reflig_en_reqired == 1) {
		print_ref_lig_energies_f(myligand_init,
					 mypars.smooth,
					 mygrid,
					 floatgrids.data(),
					 mypars.coeffs.scaled_AD4_coeff_elec,
					 mypars.coeffs.AD4_coeff_desolv,
					 mypars.qasp);
	}

	return 0;
}
