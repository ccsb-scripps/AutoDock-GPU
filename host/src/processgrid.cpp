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


#include "processgrid.h"

int get_gridinfo(
                 const char*     fldfilename,
                       Gridinfo* mygrid
                )
{
	if(strcmp(fldfilename,mygrid->fld_name.c_str())==0)
		return 0; // already successfully read this grid's information

	if(mygrid->fld_name.size()) // clear grid mapping if information from a previous fld file exists
		mygrid->grid_mapping.clear();

	std::ifstream fp;
	std::string line;
	char   tempstr [256];
	int    gpoints_even[3];
	double center[3];
	int grid_types=0;

	// ----------------------------------------------------
	// Getting full path fo the grid file
	// Getting father directory name
	mygrid->grid_file_path = get_filepath(fldfilename);
	if(mygrid->grid_file_path==".") mygrid->grid_file_path="";
	// ----------------------------------------------------

	// Processing fld file
	fp.open(fldfilename);
	if (fp.fail())
	{
		printf("Error: Can't open fld file %s.\n", fldfilename);
		return 1;
	}

	const char* ext = strstr(fldfilename,".maps");
	if(ext){
		int len=ext-fldfilename;
		mygrid->map_base_name.assign(fldfilename,len);
	} else{
		mygrid->map_base_name = fldfilename;
	}
	
	bool have_e=false;
	bool have_d=false;
	while(std::getline(fp, line))
	{
		sscanf(line.c_str(),"%255s",tempstr);
		// -----------------------------------
		// Reorder according to file *.maps.fld
		// -----------------------------------
		// Grid spacing
		if (strcmp(tempstr, "#SPACING") == 0)
		{
			sscanf(&line.c_str()[8], "%lf", &(mygrid->spacing));
			if (mygrid->spacing > 1)
			{
				printf("Error: Grid spacing is larger than 1 Å.\n");
				return 1;
			}
		}

		// capturing number of grid points
		if (strcmp(tempstr, "#NELEMENTS") == 0)
		{
			sscanf(&line.c_str()[10], "%d %d %d", &(gpoints_even[0]), &(gpoints_even[1]), &(gpoints_even[2]));
			// plus one gridpoint in each dimension
			mygrid->size_xyz[0] = gpoints_even[0] + 1;
			mygrid->size_xyz[1] = gpoints_even[1] + 1;
			mygrid->size_xyz[2] = gpoints_even[2] + 1;

			// If the grid is too big, send message and change the value of truncated_size_xyz
			if ((mygrid->size_xyz [0] > MAX_NUM_GRIDPOINTS) || (mygrid->size_xyz [1] > MAX_NUM_GRIDPOINTS) || (mygrid->size_xyz [2] > MAX_NUM_GRIDPOINTS))
			{
				printf("Error: Each dimension of the grid must be below %i.\n", MAX_NUM_GRIDPOINTS);
				return 1;
			}
		}

		// Capturing center
		if (strcmp(tempstr, "#CENTER") == 0)
		{
			sscanf(&line.c_str()[7], "%lf %lf %lf", &(center[0]), &(center[1]), &(center[2]));
		}

		// Name of the receptor and corresponding files
		if (strcmp(tempstr, "#MACROMOLECULE") == 0)
		{
			sscanf(&line.c_str()[14], "%255s", tempstr);
			if(strrchr(tempstr,'.')!=NULL){
				tempstr[strrchr(tempstr,'.')-tempstr] = '\0';
			}
			mygrid->receptor_name = tempstr;
		}

		if (line.find("label=") == 0)
		{
			sscanf(&line.c_str()[6],"%255s", tempstr);
			char* typesep = strchr(tempstr,'-'); // <atom type>-affinity ...
			if(typesep!=NULL){
				typesep[0]='\0'; // tempstr is now just the atom type
			} else{
				tempstr[1]='\0'; // tempstr is now either E(lectrostatics) or D(esolvation)
				tempstr[0]=tolower(tempstr[0]); // lower-case it
			}
			if(tempstr[0]=='e') have_e=true;
			if(tempstr[0]=='d') have_d=true;
			mygrid->grid_mapping.push_back(tempstr);
			grid_types++;
		}
		
		if (strcmp(tempstr, "variable") == 0)
		{
			size_t fidx = line.find("file=");
			if(fidx==std::string::npos){
				printf("Error: Grid map file names cannot be read.\n");
				return 1;
			}
			sscanf(&line.c_str()[fidx+5],"%255s", tempstr);
			mygrid->grid_mapping.push_back(tempstr);
		}
	}
	
	if(mygrid->grid_mapping.size() != 2*grid_types){
		printf("Error: Number of grid map labels (%d) and filenames (%d) mismatched in fld file.\n", grid_types, mygrid->grid_mapping.size()-grid_types);
		return 1;
	}
	if(!have_e){
		printf("Error: Grid map does not contain an (e)lectrostatics map.\n");
		return 1;
	}
	if(!have_d){
		printf("Error: Grid map does not contain a (d)esolvation map.\n");
		return 1;
	}
	mygrid->num_of_map_atypes = grid_types-2; // w/o e and d maps

	// calculating grid size
	mygrid->size_xyz_angstr[0] = (mygrid->size_xyz[0]-1)*(mygrid->spacing);
	mygrid->size_xyz_angstr[1] = (mygrid->size_xyz[1]-1)*(mygrid->spacing);
	mygrid->size_xyz_angstr[2] = (mygrid->size_xyz[2]-1)*(mygrid->spacing);

	if((center[0] + 0.5f * mygrid->size_xyz_angstr[0] > 9999.0f) || (center[0] - 0.5f * mygrid->size_xyz_angstr[0] < -999.0f) ||
	   (center[1] + 0.5f * mygrid->size_xyz_angstr[1] > 9999.0f) || (center[1] - 0.5f * mygrid->size_xyz_angstr[1] < -999.0f) ||
	   (center[2] + 0.5f * mygrid->size_xyz_angstr[2] > 9999.0f) || (center[2] - 0.5f * mygrid->size_xyz_angstr[2] < -999.0f)){
		printf("Error: Grid box needs to be within [-999,9999] Å for each dimension to ensure result ligand coordinates are compatible with the pdbqt format.\n");
		return 1;
	}

	// calculating coordinates of origo
	mygrid->origo_real_xyz[0] = center[0] - (((double) gpoints_even[0])*0.5*(mygrid->spacing));
	mygrid->origo_real_xyz[1] = center[1] - (((double) gpoints_even[1])*0.5*(mygrid->spacing));
	mygrid->origo_real_xyz[2] = center[2] - (((double) gpoints_even[2])*0.5*(mygrid->spacing));

	fp.close();
	mygrid->fld_name = fldfilename;

	return 0;
}

int get_gridvalues(Gridinfo* mygrid)
// The function reads the grid point values from the .map files
// that correspond to the receptor given by the first parameter.
// It allocates the proper amount of memory and stores the data
// in mygrid->grids
// If there are any errors, it returns 1, otherwise
// the return value is 0.
{
	if(mygrid->grids.size()>0) return 0; // we already read the grid maps
	mygrid->grids.resize(2*mygrid->grid_mapping.size()*
	                      (mygrid->size_xyz[0])*
	                      (mygrid->size_xyz[1])*
	                      (mygrid->size_xyz[2]));
	int t, ti, x, y, z;
	std::ifstream fp;
	std::string fn, line;
	float* mypoi = mygrid->grids.data();

	unsigned int g1 = mygrid->size_xyz[0];
	unsigned int g2 = g1*mygrid->size_xyz[1];

	for (t=0; t < mygrid->grid_mapping.size()/2; t++)
	{
		ti = t + mygrid->grid_mapping.size()/2;
		if(mygrid->fld_relative){ // this is always true (unless changed)
			fn=mygrid->grid_file_path;
			if(mygrid->grid_file_path.size()>0) fn+="/";
			fn+=mygrid->grid_mapping[ti];
//			printf("Atom type %d (%s) uses map: %s\n",t,mygrid->grid_mapping[t].c_str(),fn.c_str());
			fp.open(fn);
		}
		if (fp.fail())
		{
			printf("Error: Can't open grid map %s.\n", fn.c_str());
			return 1;
		}

		// seeking to first data
		do std::getline(fp, line);
		while (line.find("CENTER") != 0);

		// reading values
		for (z=0; z < mygrid->size_xyz[2]; z++)
			for (y=0; y < mygrid->size_xyz[1]; y++)
				for (x=0; x < mygrid->size_xyz[0]; x++)
				{
					std::getline(fp, line); // sscanf(line.c_str(), "%f", mypoi);
					*mypoi = map2float(line.c_str());
					// fill in duplicate data for linearized memory access in kernel
					if(y>0) *(mypoi-4*g1+1) = *mypoi;
					if(z>0) *(mypoi-4*g2+2) = *mypoi;
					if(y>0 && z>0) *(mypoi-4*(g2+g1)+3) = *mypoi;
					mypoi+=4;
				}
		fp.close();
	}
	return 0;
}

