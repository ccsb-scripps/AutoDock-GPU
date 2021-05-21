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


#ifndef PROCESSGRID_H_
#define PROCESSGRID_H_


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#include <fstream>
#include <sstream>

#ifndef _WIN32

#include <libgen.h>

// libgen.h contains basename() and dirname() from a fullpath name
// Specific: to open correctly grid map field fiels and associated files
// http://ask.systutorials.com/681/get-the-directory-path-and-file-name-from-absolute-path-linux
#endif

#include "defines.h"
#include "miscellaneous.h"

// Struct containing all the important information coming from .gpf and .xyz files.
typedef struct _Gridinfo
{
	std::string fld_name; // keep track of fld filename
	std::string grid_file_path; // Added to store the full path of the grid file
	std::string receptor_name;
	std::string map_base_name;
	int         size_xyz           [3];
	double      spacing;
	double      size_xyz_angstr    [3];
	char        ligand_grid_types  [MAX_NUM_OF_ATYPES+2][4]; // The additional two are the electrostatic and the desolvation types
	bool        fld_relative       = true; // By default (and until further notice) map file names are relative to the fld file
	int         num_of_atypes;
	int         num_of_map_atypes;
	double      origo_real_xyz     [3];
	std::vector<std::string> grid_mapping; // stores the atom types and associated map filenames from the fld file
} Gridinfo;

struct Map
{
	std::string        atype;
	std::vector<float> grid;
	Map(std::string atype) : atype(atype){}
};

int get_gridinfo(
                 const char*,
                       Gridinfo*
                );

int get_gridvalues_f(
                     const Gridinfo* mygrid,
                           float**   fgrids
                    );

int get_gridvalues_f(
                     const Gridinfo* mygrid,
                           float*    fgrids
                    );

#endif /* PROCESSGRID_H_ */
