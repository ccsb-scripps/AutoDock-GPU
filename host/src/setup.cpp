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


#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "filelist.hpp"
#include "processgrid.h"
#include "processligand.h"
#include "getparameters.h"
#include "setup.hpp"

int preload_gridsize(FileList& filelist)
{
	if(!filelist.used) return 0;
	int gridsize=0;
	for(unsigned int i_file=0; i_file<filelist.fld_files.size(); i_file++){
		// Filling mygrid according to the gpf file
		if (get_gridinfo(filelist.fld_files[i_file].c_str(), &filelist.mygrids[i_file]) != 0)
			{printf("\n\nError in get_gridinfo, stopped job."); return 1;}
		int curr_size = 4*filelist.mygrids[i_file].size_xyz[0]*filelist.mygrids[i_file].size_xyz[1]*filelist.mygrids[i_file].size_xyz[2];
		if(curr_size>gridsize)
			gridsize=curr_size;
	}
	return gridsize;
}

int setup(
          std::vector<Map>&   all_maps,
          Gridinfo&           mygrid,
          std::vector<float>& floatgrids,
          Dockpars&           mypars,
          Liganddata&         myligand_init,
          Liganddata&         myxrayligand,
          FileList&           filelist,
          int                 i_file,
          int                 argc,
          char*               argv[]
         )
{
	// Filling the filename and coeffs fields of mypars according to command line arguments
	if (get_filenames_and_ADcoeffs(&argc, argv, &mypars, filelist.used) != 0)
		{printf("\n\nError in get_filenames_and_ADcoeffs, stopped job."); return 1;}

	//------------------------------------------------------------
	// Testing command line arguments for cgmaps parameter,
	// for derived atom types, and modified atom type pairs
	// since they will be needed at ligand and grid creation
	//------------------------------------------------------------
	for (int i=1; i<argc-1; i+=2)
	{
		if (argcmp("xml2dlg", argv[i], 'X'))
			i+=mypars.xml_files-1; // skip ahead in case there are multiple entries here

		// ----------------------------------
		// Argument: Use individual maps for CG-G0 instead of the same one
		if (argcmp("cgmaps", argv [i]))
		{
			int tempint;
			sscanf(argv [i+1], "%d", &tempint);
			if (tempint == 0)
				mypars.cgmaps = 0;
			else
				mypars.cgmaps = 1;
		}
		// ----------------------------------
		// Argument: derivate atom types
		if (argcmp("derivtype", argv [i], 'T'))
		{
			if(mypars.nr_deriv_atypes==0){
				mypars.deriv_atypes=(deriv_atype*)malloc(sizeof(deriv_atype));
				if(mypars.deriv_atypes==NULL){
					printf("Error: Cannot allocate memory for --derivtype (-T).\n");
					exit(1);
				}
			}
			bool success=true;
			char* tmp=argv[i+1];
			
			while(success && (*tmp!='\0')){
				bool base_exists=false;
				char* start_block=tmp;
				int nr_start=mypars.nr_deriv_atypes;
				// count nr of derivative atom types first
				while((*tmp!='\0') && (*tmp!='/')){ // do one block at a time
					if(*(tmp++)==','){ // this works here as the first character is not a ','
						if(base_exists){
							printf("Error in --derivtype (-T) %s: only one base name is allowed.\n",argv[i+1]);
							success=false;
							break;
						}
						if(tmp-start_block-1>0){ // make sure there is a name (at least one char, we'll test later if it's taken already)
							if(!add_deriv_atype(&mypars,start_block,tmp-start_block-1)){
								printf("Error in --derivtype (-T) %s: derivative names can only be upto 3 characters long.\n",argv[i+1]);
								success=false;
								break;
							}
							start_block=tmp;
						} else{
							printf("Error in --derivtype (-T) %s: derivative names have to be at least one character long.\n",argv[i+1]);
							success=false;
							break;
						}
					}
					if((*tmp=='=') && ((*(tmp+1)!='\0') || (*(tmp+1)!='/'))){
						if(tmp-start_block>0){ // make sure there is a name (at least one char, we'll test later if it's taken already)
							if(!add_deriv_atype(&mypars,start_block,tmp-start_block)){
								printf("Error in --derivtype (-T) %s: derivative names can only be upto 3 characters long.\n",argv[i+1]);
								success=false;
								break;
							}
						} else{
							printf("Error in --derivtype (-T) %s: derivative names have to be at least one character long.\n",argv[i+1]);
							success=false;
							break;
						}
						start_block=tmp+1;
						base_exists=true;
					}
				}
				for(int idx=nr_start; idx<mypars.nr_deriv_atypes; idx++){
					int length=tmp-start_block;
					if(length<4){
						strncpy(mypars.deriv_atypes[idx].base_name,start_block,length);
						mypars.deriv_atypes[idx].base_name[length]='\0';
					} else{
						printf("Error in --derivtype (-T) %s: base names can only be upto 3 characters long.\n",argv[i+1]);
						success=false;
						break;
					}
#ifdef DERIVTYPE_INFO
					printf("%i: %s=%s\n",mypars.deriv_atypes[idx].nr,mypars.deriv_atypes[idx].deriv_name,mypars.deriv_atypes[idx].base_name);
#endif
				}
				if(*tmp=='/') // need to go to next char otherwise the two loops will infinite loop (ask me how I knooooooooooooooooooooooooooooooooooooooo
					tmp++;
			}
			if(!success){
				printf("Example syntax: --derivtype C1,C2,C3=C/S4=S/H5=HD.\n");
				exit(12);
			}
		}

		// Argument: modify pairwise atom type parameters (LJ only at this point)
		if (argcmp("modpair", argv [i], 'P'))
		{
			bool success=true;
			char* tmp=argv[i+1];
			
			while(success && (*tmp!='\0')){
				mypars.nr_mod_atype_pairs++;
				if(mypars.nr_mod_atype_pairs==1)
					mypars.mod_atype_pairs=(pair_mod*)malloc(sizeof(pair_mod));
				else
					mypars.mod_atype_pairs=(pair_mod*)realloc(mypars.mod_atype_pairs, mypars.nr_mod_atype_pairs*sizeof(pair_mod));
				if(mypars.mod_atype_pairs==NULL){
					printf("Error: Cannot allocate memory for --modpair (-P).\n");
					exit(1);
				}
				pair_mod* curr_pair=&mypars.mod_atype_pairs[mypars.nr_mod_atype_pairs-1];
				// find atom type pair to modify
				char* first_comma=strchr(tmp,',');
				if(first_comma==NULL){
					printf("Error in --modpair (-P) %s: no parameters specified (not even a first comma).\n",argv[i+1]);
					success=false;
					break;
				}
				char* colon=strchr(tmp,':');
				if(colon==NULL){
					printf("Error in --modpair (-P) %s: Could not find pair atom type name separator (\":\").\n",argv[i+1]);
					success=false;
					break;
				}
				int Alen = colon-tmp;
				colon++; // we don't want to include the colon
				int Blen = first_comma-colon;
				if ((Alen>0) && (Blen>0)){ // we have data between the start, colon, and comma (good start)
					if ((Alen<4) && (Blen<4)){
						strncpy(curr_pair->A,tmp,Alen);
						curr_pair->A[Alen]='\0';
						strncpy(curr_pair->B,colon,Blen);
						curr_pair->B[Blen]='\0';
					} else{
						printf("Error in --modpair (-P) %s: pair atom type name(s) are too long (>3 characters).\n",argv[i+1]);
						success=false;
						break;
					}
				} else{
					printf("Error in --modpair (-P) %s: pair atom type name(s) not specified.\n",argv[i+1]);
					success=false;
					break;
				}
				tmp=first_comma+1;
				char* start_block=tmp;
				curr_pair->nr_parameters=0;
				curr_pair->parameters=NULL;
				// count nr of derivative atom types first
				while((*tmp!='\0') && (*tmp!='/')){ // do one block at a time
					tmp++;
					if((*tmp==',') || (*tmp=='\0') || (*tmp=='/')){
						if(tmp-start_block>0){ // make sure there is a name (at least one char, we'll test later if it's taken already)
							float tmpfloat;
							sscanf(start_block, "%f", &tmpfloat);
							curr_pair->nr_parameters++;
							curr_pair->parameters=(float*)realloc(curr_pair->parameters,curr_pair->nr_parameters*sizeof(float));
							if(curr_pair->parameters==NULL){
								printf("Error: Cannot allocate memory for --modpair (-P).\n");
								exit(1);
							}
							curr_pair->parameters[curr_pair->nr_parameters-1]=tmpfloat;
							start_block=tmp+1;
						} else{
							printf("Error in --modpair (-P) %s: force field parameters should be at least one number long.\n",argv[i+1]);
							success=false;
							break;
						}
					}
				}
				if(*tmp=='/') // need to go to next char otherwise the two loops will infinite loop (ask me how I knooooooooooooooooooooooooooooooooooooooo
					tmp++;
#ifdef MODPAIR_INFO
				printf("%i: %s:%s",mypars.nr_mod_atype_pairs,curr_pair->A,curr_pair->B);
				for(unsigned int idx=0; idx<curr_pair->nr_parameters; idx++)
					printf(",%f",curr_pair->parameters[idx]);
				printf("\n");
#endif
			}
			if(!success){
				printf("Example syntax: --modpair C1:S4,1.60,1.200,13,7/C1:C3,1.20,0.025.\n");
				exit(12);
			}
		}
	}

	//------------------------------------------------------------
	// Processing receptor and ligand files
	//------------------------------------------------------------

	// Filling mygrid according to the fld file
	if (get_gridinfo(mypars.fldfile, &mygrid) != 0)
	{
		printf("\n\nError in get_gridinfo, stopped job.");
		return 1;
	}

	// Filling the atom types field of myligand according to the grid types
	if (init_liganddata(mypars.ligandfile,
	                    mypars.flexresfile,
	                    &myligand_init,
	                    &mygrid,
	                    mypars.nr_deriv_atypes,
	                    mypars.deriv_atypes,
	                    mypars.cgmaps) != 0)
	{
		printf("\n\nError in init_liganddata, stopped job.");
		return 1;
	}

	// Filling myligand according to the pdbqt file
	if (get_liganddata(mypars.ligandfile,
	                   mypars.flexresfile,
	                   &myligand_init,
	                   mypars.coeffs.AD4_coeff_vdW,
	                   mypars.coeffs.AD4_coeff_hb,
	                   mypars.nr_deriv_atypes,
	                   mypars.deriv_atypes,
	                   mypars.nr_mod_atype_pairs,
	                   mypars.mod_atype_pairs) != 0)
	{
		printf("\n\nError in get_liganddata, stopped job.");
		return 1;
	}

	// Adding receptor atom information needed for analysis
	if (mypars.contact_analysis && (mypars.flexresfile!=NULL)){
		std::vector<ReceptorAtom> flexresatoms = read_receptor_atoms(mypars.flexresfile);
		mypars.receptor_atoms.insert(mypars.receptor_atoms.end(), flexresatoms.begin(), flexresatoms.end());
		for(int i=myligand_init.true_ligand_atoms; i<myligand_init.num_of_atoms; i++){
			mypars.receptor_atoms[mypars.nr_receptor_atoms+i-myligand_init.true_ligand_atoms].acceptor=myligand_init.acceptor[i];
			mypars.receptor_atoms[mypars.nr_receptor_atoms+i-myligand_init.true_ligand_atoms].donor=myligand_init.donor[i];
		}
	}

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
					if (load_all_maps(mypars.fldfile,
					                  &mygrid,
					                  all_maps,
					                  mypars.cgmaps) != 0)
					{
						got_error = true;
					}
					filelist.maps_are_loaded = true;
				}
			}
			// Return must be outside pragma
			if (got_error) {
				printf("\n\nError in load_all_maps, stopped job.");
				return 1;
			}
		}

		// Copy maps from all_maps
		if (copy_from_all_maps(&mygrid,
		                       floatgrids.data(),
		                       all_maps) != 0)
		{
			printf("\n\nError in copy_from_all_maps, stopped job.");
			return 1;
		}

		// Specify total number of maps that will be on GPU
		mygrid.num_of_map_atypes = all_maps.size()-2; // For the two extra maps
		// Map atom_types used for ligand processing to all_maps so all the maps can stay on GPU
		if (map_to_all_maps(&mygrid,
		                    &myligand_init,
		                    all_maps) !=0)
		{
			printf("\n\nError in map_to_all_maps, stopped job.");
			return 1;
		}
	} else {
		// read receptor in case contact analysis is requested and we haven't done so already (in the preload case above)
		std::string receptor_name=mygrid.grid_file_path;
		if(strlen(mygrid.grid_file_path)>0) receptor_name+="/";
		receptor_name += mygrid.receptor_name;
		receptor_name += ".pdbqt";
		mypars.receptor_atoms = read_receptor(receptor_name.c_str(),&mygrid,mypars.receptor_map,mypars.receptor_map_list);
		mypars.nr_receptor_atoms = mypars.receptor_atoms.size();
		// Reading the grid files and storing values in the memory region pointed by floatgrids
		if (get_gridvalues_f(&mygrid,
		                     floatgrids.data(),
		                     mypars.cgmaps) != 0)
		{
			printf("\n\nError in get_gridvalues_f, stopped job.");
			return 1;
		}
	}

	//------------------------------------------------------------
	// Capturing algorithm parameters (command line args)
	//------------------------------------------------------------
	char* orig_resn = mypars.resname;
	if(get_commandpars(&argc, argv, &(mygrid.spacing), &mypars)<0)
		return 1;

	// command-line specified resname with more than one file
	if (!mypars.xml2dlg){ // if the user specified an xml file, that's the one we want to use
		if ((orig_resn!=mypars.resname) && (filelist.nfiles>1)){ // add an index to existing name distinguish the files
			char* tmp = strdup(mypars.resname);
			char* nrtmp = strdup(std::to_string(i_file+1).c_str());
			if(mypars.resname) free(mypars.resname);
			mypars.resname = (char*)malloc((strlen(tmp)+strlen(nrtmp)+2)*sizeof(char));
			strcpy(mypars.resname, tmp);
			strcat(mypars.resname,"_");
			strcat(mypars.resname, nrtmp);
			free(tmp);
			free(nrtmp);
		}
	}

	Gridinfo mydummygrid;
	// if -lxrayfile provided, then read xray ligand data
	if (mypars.given_xrayligandfile == true)
	{
		if (init_liganddata(mypars.xrayligandfile,
		                    "\0",
		                    &myxrayligand,
		                    &mydummygrid,
		                    0,
		                    NULL,
		                    mypars.cgmaps) != 0)
		{
			printf("\n\nError in init_liganddata, stopped job.");
			return 1;
		}

		if (get_liganddata(mypars.xrayligandfile,
		                   "\0",
		                   &myxrayligand,
		                   mypars.coeffs.AD4_coeff_vdW,
		                   mypars.coeffs.AD4_coeff_hb,
		                   mypars.nr_deriv_atypes,
		                   mypars.deriv_atypes,
		                   mypars.nr_mod_atype_pairs,
		                   mypars.mod_atype_pairs) != 0)
		{
			printf("\n\nError in get_liganddata, stopped job.");
			return 1;
		}
	}

	//------------------------------------------------------------
	// Calculating energies of reference ligand if required
	//------------------------------------------------------------
	if (mypars.reflig_en_required) {
		print_ref_lig_energies_f(myligand_init,
		                         mypars.smooth,
		                         mygrid,
		                         floatgrids.data(),
		                         mypars.coeffs.scaled_AD4_coeff_elec,
		                         mypars.elec_min_distance,
		                         mypars.coeffs.AD4_coeff_desolv,
		                         mypars.qasp,
		                         mypars.nr_mod_atype_pairs,
		                         mypars.mod_atype_pairs);
	}

	return 0;
}

int fill_maplist(
                 const char*             fldfilename,
                       std::vector<Map>& all_maps)
{
	std::ifstream file(fldfilename);
	if(file.fail()){
		printf("\nError: Could not open %s. Check path and permissions.",fldfilename);
		return 1;
	}
	std::string line;
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

int load_all_maps(
                  const char*             fldfilename,
                  const Gridinfo*         mygrid,
                        std::vector<Map>& all_maps,
                        bool              cgmaps
                 )
{
	// First, parse .fld file to get map names
	if(fill_maplist(fldfilename,all_maps)==1) return 1;

	// Now fill the maps
	int x, y, z;
	FILE* fp;
	size_t len = strlen(mygrid->grid_file_path)+strlen(mygrid->receptor_name)+1;
	if(strlen(mygrid->map_base_name)>len)
		len = strlen(mygrid->map_base_name);
	len += 10; // "..map\0" = 6 entries + 4 at most for grid type
	if(len<128) len=128;
	char* tempstr = (char*)malloc(len*sizeof(char));
	int size_of_one_map = 4*mygrid->size_xyz[0]*mygrid->size_xyz[1]*mygrid->size_xyz[2];

	for (unsigned int t=0; t < all_maps.size(); t++)
	{
		all_maps[t].grid.resize(size_of_one_map);
		float* mypoi = all_maps[t].grid.data();
		// opening corresponding .map file
		strcpy(tempstr,mygrid->map_base_name);
		strcat(tempstr, ".");
		strcat(tempstr, all_maps[t].atype.c_str());
		strcat(tempstr, ".map");
		fp = fopen(tempstr, "rb"); // fp = fopen(tempstr, "r");
		if (fp == NULL){ // try again with the receptor name in the .maps.fld file
			strcpy(tempstr,mygrid->grid_file_path);
			strcat(tempstr, "/");
			strcat(tempstr, mygrid->receptor_name);
			strcat(tempstr, ".");
			strcat(tempstr, all_maps[t].atype.c_str());
			strcat(tempstr, ".map");
			fp = fopen(tempstr, "rb"); // fp = fopen(tempstr, "r");
		}
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

		// seeking to first data
		do    fscanf(fp, "%127s", tempstr);
		while (strcmp(tempstr, "CENTER") != 0);
		fscanf(fp, "%127s", tempstr);
		fscanf(fp, "%127s", tempstr);
		fscanf(fp, "%127s", tempstr);

		unsigned int g1 = mygrid->size_xyz[0];
		unsigned int g2 = g1*mygrid->size_xyz[1];
		// reading values
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
	free(tempstr);
	return 0;
}

int copy_from_all_maps(
                       const Gridinfo*         mygrid,
                             float*            fgrids,
                             std::vector<Map>& all_maps
                      )
{
	int size_of_one_map = 4*mygrid->size_xyz[0]*mygrid->size_xyz[1]*mygrid->size_xyz[2];
	for (int t=0; t < mygrid->num_of_atypes+2; t++) {
		// Look in all_maps for desired map
		int i_map = -1;
		for (unsigned int i_atype=0; i_atype < all_maps.size(); i_atype++){
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

