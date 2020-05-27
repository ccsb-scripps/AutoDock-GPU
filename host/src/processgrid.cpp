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




#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cassert>

#include "processgrid.h"

int get_gridinfo(const char* fldfilename, Gridinfo* mygrid)
{
	FILE*  fp;
	char   tempstr [256];
	int    gpoints_even[3];
	int    recnamelen;
	double center[3];

	// ----------------------------------------------------
	// Getting full path fo the grid file
	// Getting father directory name
	//char* dir = dirname(ts1);
	//char* filename = basename(ts1);

	#ifndef _WIN32
	char* ts1 = strdup(fldfilename);
	mygrid->grid_file_path = dirname(ts1);
	#else
	char* ts1 = strdup(fldfilename);
	char drive_tmp[_MAX_DRIVE];
	char path_tmp[_MAX_DIR];
	_splitpath(ts1, drive_tmp, path_tmp, NULL, NULL);
	
	char result[2*_MAX_DIR];
	strcpy(result, drive_tmp);
	strcpy(result, path_tmp);
	for (unsigned int i=0; i<2*_MAX_DIR; i++) {
		mygrid->grid_file_path[i] = result[i];
	}
	#endif
	// ----------------------------------------------------

	//Processing fld file
	fp = fopen(fldfilename, "rb"); // fp = fopen(fldfilename, "r");
	if (fp == NULL)
	{
		printf("Error: can't open fld file %s!\n", fldfilename);
		return 1;
	}

	while (fscanf(fp, "%s", tempstr) != EOF)
	{
		// -----------------------------------
		// Reorder according to file *.maps.fld
		// -----------------------------------
		//Grid spacing
		if (strcmp(tempstr, "#SPACING") == 0)
		{
			fscanf(fp, "%lf", &(mygrid->spacing));
			if (mygrid->spacing > 1)
			{
				printf("Error: grid spacing is too big!\n");
				fclose(fp);
				return 1;
			}
		}

		//capturing number of grid points
		if (strcmp(tempstr, "#NELEMENTS") == 0)
		{
			fscanf(fp, "%d%d%d", &(gpoints_even[0]), &(gpoints_even[1]), &(gpoints_even[2]));
			//plus one gridpoint in each dimension
			mygrid->size_xyz[0] = gpoints_even[0] + 1;
			mygrid->size_xyz[1] = gpoints_even[1] + 1;
			mygrid->size_xyz[2] = gpoints_even[2] + 1;

			//If the grid is too big, send message and change the value of truncated_size_xyz
			if ((mygrid->size_xyz [0] > MAX_NUM_GRIDPOINTS) || (mygrid->size_xyz [1] > MAX_NUM_GRIDPOINTS) || (mygrid->size_xyz [2] > MAX_NUM_GRIDPOINTS))
			{
				printf("Error: each dimension of the grid must be below %i.\n", MAX_NUM_GRIDPOINTS);
				fclose(fp);
				return 1;
			}
		}

		//Capturing center
		if (strcmp(tempstr, "#CENTER") == 0)
		{
			fscanf(fp, "%lf%lf%lf", &(center[0]), &(center[1]), &(center[2]));
		}

		//Name of the receptor and corresponding files
		if (strcmp(tempstr, "#MACROMOLECULE") == 0)
		{
			fscanf(fp, "%s", tempstr);
			recnamelen = strcspn(tempstr,".");
			tempstr[recnamelen] = '\0';
			strcpy(mygrid->receptor_name, tempstr);
		}

		// -----------------------------------
		// MISSING: similar section corresponding to
		// #GRID_PARAMETER_FILE
		// -----------------------------------
	}

	//calculating grid size
	mygrid->size_xyz_angstr[0] = (mygrid->size_xyz[0]-1)*(mygrid->spacing);
	mygrid->size_xyz_angstr[1] = (mygrid->size_xyz[1]-1)*(mygrid->spacing);
	mygrid->size_xyz_angstr[2] = (mygrid->size_xyz[2]-1)*(mygrid->spacing);

	//calculating coordinates of origo
	mygrid->origo_real_xyz[0] = center[0] - (((double) gpoints_even[0])*0.5*(mygrid->spacing));
	mygrid->origo_real_xyz[1] = center[1] - (((double) gpoints_even[1])*0.5*(mygrid->spacing));
	mygrid->origo_real_xyz[2] = center[2] - (((double) gpoints_even[2])*0.5*(mygrid->spacing));

	fclose(fp);

	return 0;
}

int get_gridvalues_f(const Gridinfo* mygrid, float* fgrids, bool cgmaps)
//The function reads the grid point values from the .map files
//that correspond to the receptor given by the first parameter.
//It allocates the proper amount of memory and stores the data there,
//which can be accessed with the fgrids pointer.
//If there are any errors, it returns 1, otherwise
//the return value is 0.
{
	int t, x, y, z;
	FILE* fp;
	char tempstr [128];
	float* mypoi;

	mypoi = fgrids;

	for (t=0; t < mygrid->num_of_atypes+2; t++)
	{
		//opening corresponding .map file
		//-------------------------------------
		// Added the complete path of associated grid files.
		strcpy(tempstr,mygrid->grid_file_path);
		strcat(tempstr, "/");
		strcat(tempstr, mygrid->receptor_name);
		
		//strcpy(tempstr, mygrid->receptor_name);
		//-------------------------------------
		strcat(tempstr, ".");
		strcat(tempstr, mygrid->grid_types[t]);
		strcat(tempstr, ".map");
		fp = fopen(tempstr, "rb"); // fp = fopen(tempstr, "r");
		if (fp == NULL)
		{
			printf("Error: can't open %s!\n", tempstr);
			if ((strncmp(mygrid->grid_types[t],"CG",2)==0) ||
			    (strncmp(mygrid->grid_types[t],"G",1)==0))
			{
				if(cgmaps)
					printf("-> Expecting an individual map for each CGx and Gx (x=0..9) atom type.\n");
				else
					printf("-> Expecting one map file, ending in .CG.map and .G0.map, for CGx and Gx atom types, respectively.\n");
			}
			return 1;
		}

		//seeking to first data
		do    fscanf(fp, "%s", tempstr);
		while (strcmp(tempstr, "CENTER") != 0);
		fscanf(fp, "%s", tempstr);
		fscanf(fp, "%s", tempstr);
		fscanf(fp, "%s", tempstr);

		unsigned int g1 = mygrid->size_xyz[0];
		unsigned int g2 = g1*mygrid->size_xyz[1];
		//reading values
		for (z=0; z < mygrid->size_xyz[2]; z++)
			for (y=0; y < mygrid->size_xyz[1]; y++)
				for (x=0; x < mygrid->size_xyz[0]; x++)
				{
					fscanf(fp, "%f", mypoi);
					// fill in duplicate data for linearized memory access in kernel
					if(y>0) *(mypoi-4*g1+1) = *mypoi;
					if(z>0) *(mypoi-4*g2+2) = *mypoi;
					if(y>0 && z>0) *(mypoi-4*(g2+g1)+3) = *mypoi;
					mypoi+=4;
				}
		fclose(fp);
	}

	return 0;
}

int fill_maplist(const char* fldfilename, std::vector<Map>& all_maps)
{
	std::ifstream file(fldfilename);
        if(file.fail()){
                printf("\nError: Could not open %s. Check path and permissions.",fldfilename);
                return 1;
        }
	std::string line;
	bool prev_line_was_fld=false;
	while(std::getline(file, line)) {
		std::stringstream sline(line.c_str());
		// Split line by spaces:
		std::string word;
		bool is_variable_line=false;
		while(std::getline(sline, word, ' ')){
			// Check if first word is "variable"
			if (word.compare("variable") == 0) is_variable_line=true;
			int len = word.size();
                        if (is_variable_line && len>=4 && word.compare(len-4,4,".map") == 0){ // Found a word that ends in "map"
				// Split the map into segments e.g. protein.O.map -> "protein", "O", "map"
				std::stringstream mapword(word.c_str());
				std::string segment;
				std::vector<std::string> seglist;
				while(std::getline(mapword, segment, '.')) seglist.push_back(segment);

				// Create a new map with the atom name
				all_maps.push_back(Map(seglist[seglist.size()-2]));
			}
		}
	}
	return 0;
}

int load_all_maps (const char* fldfilename, const Gridinfo* mygrid, std::vector<Map>& all_maps, bool cgmaps,float* fgrids_device)
{
	// First, parse .fld file to get map names
	if(fill_maplist(fldfilename,all_maps)==1) return 1;

	// Now fill the maps
        int t, x, y, z;
        FILE* fp;
        char tempstr [128];
	int size_of_one_map = 4*mygrid->size_xyz[0]*mygrid->size_xyz[1]*mygrid->size_xyz[2];

        for (t=0; t < all_maps.size(); t++)
        {
		all_maps[t].grid.resize(size_of_one_map);
		float* mypoi = all_maps[t].grid.data();
                //opening corresponding .map file
                //-------------------------------------
                // Added the complete path of associated grid files.
                strcpy(tempstr,mygrid->grid_file_path);
                strcat(tempstr, "/");
                strcat(tempstr, mygrid->receptor_name);

                //strcpy(tempstr, mygrid->receptor_name);
                //-------------------------------------
                strcat(tempstr, ".");
                strcat(tempstr, all_maps[t].atype.c_str());
                strcat(tempstr, ".map");
                fp = fopen(tempstr, "rb"); // fp = fopen(tempstr, "r");
                if (fp == NULL)
                {
                        printf("Error: can't open %s!\n", tempstr);
                        if ((strncmp(all_maps[t].atype.c_str(),"CG",2)==0) ||
                            (strncmp(all_maps[t].atype.c_str(),"G",1)==0))
                        {
                                if(cgmaps)
                                        printf("-> Expecting an individual map for each CGx and Gx (x=0..9) atom type.\n");
                                else
                                        printf("-> Expecting one map file, ending in .CG.map and .G0.map, for CGx and Gx atom types, respectively.\n");
                        }
                        return 1;
                }

                //seeking to first data
                do    fscanf(fp, "%s", tempstr);
                while (strcmp(tempstr, "CENTER") != 0);
                fscanf(fp, "%s", tempstr);
                fscanf(fp, "%s", tempstr);
                fscanf(fp, "%s", tempstr);

                unsigned int g1 = mygrid->size_xyz[0];
                unsigned int g2 = g1*mygrid->size_xyz[1];
                //reading values
                for (z=0; z < mygrid->size_xyz[2]; z++)
                        for (y=0; y < mygrid->size_xyz[1]; y++)
                                for (x=0; x < mygrid->size_xyz[0]; x++)
                                {
                                        fscanf(fp, "%f", mypoi);
                                        // fill in duplicate data for linearized memory access in kernel
                                        if(y>0) *(mypoi-4*g1+1) = *mypoi;
                                        if(z>0) *(mypoi-4*g2+2) = *mypoi;
                                        if(y>0 && z>0) *(mypoi-4*(g2+g1)+3) = *mypoi;
                                        mypoi+=4;
                                }

		fclose(fp);

		// Copy to GPU
		cudaError_t status = cudaMemcpy(fgrids_device+t*size_of_one_map,all_maps[t].grid.data(),sizeof(float)*size_of_one_map, cudaMemcpyHostToDevice);
    		if (status != cudaSuccess) { 
        		printf("%s %s\n", "pMem_fgrids: failed to upload maps to GPU memory.\n", cudaGetErrorString(status)); 
        		assert(0); 
        		cudaDeviceReset(); 
        		exit(-1); 
    		}
        }

        return 0;
}

int copy_from_all_maps (const Gridinfo* mygrid, float* fgrids, std::vector<Map>& all_maps)
{
	int size_of_one_map = 4*mygrid->size_xyz[0]*mygrid->size_xyz[1]*mygrid->size_xyz[2];
        for (int t=0; t < mygrid->num_of_atypes+2; t++) {
		// Look in all_maps for desired map
		int i_map = -1;
		for (int i_atype=0; i_atype < all_maps.size(); i_atype++){
			if (strcmp(mygrid->grid_types[t],all_maps[i_atype].atype.c_str())==0){
				i_map = i_atype; // Found the map!
				break;
			}
		}
		if (i_map == -1){ // Didnt find the map
			printf("\nError: The %s map needed for the ligand was not found in the .fld file!", mygrid->grid_types[t]);
			return 1;
		}

		// Copy from all_maps into fgrids
		memcpy(fgrids+t*size_of_one_map,all_maps[i_map].grid.data(),sizeof(float)*all_maps[i_map].grid.size());
        }

        return 0;
}
