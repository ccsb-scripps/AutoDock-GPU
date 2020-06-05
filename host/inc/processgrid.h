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

// libgen.h contains basename() and dir() from a fullpath name
// Specific: to open correctly grid map field fiels and associated files
// http://ask.systutorials.com/681/get-the-directory-path-and-file-name-from-absolute-path-linux
#endif

#include "defines.h"
#include "miscellaneous.h"

#define getvalue_4Darr(mempoi, grinf, t, z, y, x)                  *(mempoi + 4*((grinf).size_xyz[0] * (y + (grinf).size_xyz[1] * (z + (grinf).size_xyz[2]*t)) + x))
#define getvalue_4Darr_withsize(mempoi, gridsize_xyz, t, z, y, x)  *(mempoi + 4*(gridsize_xyz[0]*(y + gridsize_xyz[1] * (z + gridsize_xyz[2]*t)) + x))
//The macro helps to access the grid point values
//which were read from external files with get_gridvalues function.
//The first parameter is a pointer which points to the memory area storing the data.
//The second one is the corresponding grid info (parameter of get_gridinfo function).
//The other parameters are the type index, z, y and x coordinates of the grid point.

typedef struct
//Struct which can contain all the important informations which derives from .gpf and .xyz files.
{

#ifndef _WIN32
	char*  grid_file_path;	  // Added to store the full path of the grid file
#else
	char  grid_file_path[2*_MAX_DIR];	  // Added to store the full path of the grid file
#endif
	char   receptor_name [64];
	int    size_xyz [3];
	double spacing;
	double size_xyz_angstr [3];
	char   grid_types [MAX_NUM_OF_ATYPES+2][4]; // The additional two are the electrostatic and the desolvation types
	int    num_of_atypes;
	int    num_of_map_atypes;
	double origo_real_xyz [3];
} Gridinfo;

struct Map
{
	std::string atype;
	std::vector<float> grid;

	Map(std::string atype) : atype(atype){}
};

int get_gridinfo(const char*, Gridinfo*);

int get_gridvalues_f(const Gridinfo* mygrid,
		     float* fgrids,
		     bool cgmaps);

int load_all_maps (const char* fldfilename, const Gridinfo* mygrid, std::vector<Map>& all_maps, bool cgmaps,float* fgrids_device);
int copy_from_all_maps (const Gridinfo* mygrid, float* fgrids, std::vector<Map>& all_maps );

#endif /* PROCESSGRID_H_ */
