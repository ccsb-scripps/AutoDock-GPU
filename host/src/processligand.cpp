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


// Use AD4 minimum and maximum bond distances instead of the Frankenstein we had in AD-GPU
#define AD4_BOND_DISTS

// Add the desolvation energy to the vdW per atom energies in the output (like AD4 does)
#define AD4_desolv_peratom_vdW

// Output showing the CG-G0 virtual bonds and pairs
// #define CG_G0_INFO

// Show information about the atom types in each ligand
// #define TYPE_INFO

#include "processligand.h"

int init_liganddata(
                    const char*        ligfilename,
                    const char*        flexresfilename,
                          Liganddata*  myligand,
                          Gridinfo*    mygrid,
                          int          nr_deriv_atypes,
                          deriv_atype* deriv_atypes,
                          bool         cgmaps
                   )
// The functions first parameter is an empty Liganddata, the second a variable of
// Gridinfo type. The function fills the num_of_atypes and atom_types fields of
// myligand according to the num_of_atypes and grid_types fields of mygrid. In
// this case it is supposed, that the ligand and receptor described by the two
// parameters correspond to each other.
// If the operation was successful, the function returns 0, if not, it returns 1.
{
	std::ifstream fp;
	int num_of_atypes, new_type, num_of_base_atypes;
	char atom_types [MAX_NUM_OF_ATOMS][4];
	char base_atom_types [MAX_NUM_OF_ATOMS][4];
	memset(myligand->base_type_idx,0,MAX_NUM_OF_ATOMS*sizeof(int));
	char tempstr [256];
	std::string line;

	unsigned int lnr=1;
	if ( flexresfilename!=NULL) {
		if ( strlen(flexresfilename)>0 )
			lnr++;
	}

	num_of_atypes = 0;
	num_of_base_atypes = 0;
	unsigned int atom_cnt = 0;
	for (unsigned int l=0; l<lnr; l++)
	{
		if(l==0)
			fp.open(ligfilename);
		else
			fp.open(flexresfilename);
		if (fp.fail())
		{
			if(l==0)
				printf("Error: can't open ligand data file %s!\n", ligfilename);
			else
				printf("Error: can't open flexibe residue data file %s!\n", flexresfilename);
			return 1;
		}
		// reading the whole ligand pdbqt file
		while(std::getline(fp, line))
		{
			sscanf(line.c_str(),"%255s",tempstr);
			if ((strcmp(tempstr, "HETATM") == 0) || (strcmp(tempstr, "ATOM") == 0))
			{
				new_type = 1; // supposing this will be a new atom type
				if ((strcmp(tempstr, "HETATM") == 0)) // seeking to the first coordinate value
				line[17]='\0';
				sscanf(&line.c_str()[77], "%3s", tempstr); // reading atom type
				tempstr[3] = '\0'; //just to be sure strcpy wont fail even if something is wrong with position
				line[17]='\0';
				sscanf(&line.c_str()[13], "%4s", myligand->atom_names[atom_cnt]);
				atom_cnt++;

				// checking if this atom has been already found
				for (int i=0; i<num_of_atypes; i++)
				{
					if (strcmp(atom_types[i], tempstr) == 0)
						new_type = 0; // this is not a new type
				}

				if (new_type == 1) // if new type, copying string...
				{
					// checking if atom type number doesn't exceed 14
					if (num_of_atypes >= MAX_NUM_OF_ATYPES)
					{
						printf("Error: too many types of ligand atoms!\n");
						return 1;
					}

					strcpy(atom_types[num_of_atypes], tempstr);
					strcpy(base_atom_types[num_of_atypes], tempstr);
					// assumption is that we are adding a base atom type to the list
					myligand->base_type_idx[num_of_atypes] = num_of_base_atypes;
					num_of_atypes++;
					bool deriv_type = false;
					// find base type in case there are derivative types
					// - this will find the derived type this atom type is, and then
					// - search for it's base type name in the current list of base atom types,
					// - if found, the index is changed to the found one,
					// - if not, this is a new base type and the index above is correct :-)
					for (int i=0; i<nr_deriv_atypes; i++){
						if (strcmp(deriv_atypes[i].deriv_name, tempstr) == 0){
							strcpy(base_atom_types[num_of_atypes-1],deriv_atypes[i].base_name); // copy the actual base atom type
							// find the base idx (if it's not found the above is correct)
							num_of_base_atypes++;
							for(int j=0; j<num_of_atypes-1; j++)
								if(strcmp(base_atom_types[j],base_atom_types[num_of_atypes-1]) == 0){
									myligand->base_type_idx[num_of_atypes-1]=myligand->base_type_idx[j];
									num_of_base_atypes--;
									break;
								}
							deriv_type = true;
							break;
						}
					}
					if(!deriv_type) // this tracks num_of_atypes with no derived types
						num_of_base_atypes++;
				}
			}
		}
		// copying field to ligand and grid data
		myligand->num_of_atypes = num_of_atypes;
		mygrid->num_of_atypes   = num_of_base_atypes;
		mygrid->num_of_map_atypes   = num_of_base_atypes;
		fp.close();
	}
#if defined(CG_G0_INFO)
	if (cgmaps)
	{
		printf("Expecting individual maps for CGx and Gx atom types (x=0..9).\n");
	}
	else
	{
		printf("Using one map file, .CG.map and .G0.map, for CGx and Gx atom types, respectively.\n");
	}
#endif
#ifdef TYPE_INFO
	printf("Ligand contains %i base types and %i derived types.\n",num_of_base_atypes,num_of_atypes-num_of_base_atypes);
#endif
	for (int i=0; i<num_of_atypes; i++)
	{
		strcpy(myligand->atom_types[i], atom_types[i]);
		strcpy(myligand->base_atom_types[i], base_atom_types[i]);
		strcpy(mygrid->grid_types[myligand->base_type_idx[i]], base_atom_types[i]);
		if(strncmp(base_atom_types[i],"CG",2)+strncmp(base_atom_types[i],"G",1)==0){
			memcpy(mygrid->grid_types[myligand->base_type_idx[i]], base_atom_types[i],2*sizeof(char));
			mygrid->grid_types[myligand->base_type_idx[i]][2] = '\0'; // make sure CG0..9 results in CG
			if (isdigit(mygrid->grid_types[myligand->base_type_idx[i]][1])) // make sure G0..9 results in G0
				mygrid->grid_types[myligand->base_type_idx[i]][1] = '0';
		}
#ifdef TYPE_INFO
		printf("Atom type %i -> %s -> %s (grid type %i)\n",i,myligand->atom_types[i],mygrid->grid_types[myligand->base_type_idx[i]],myligand->base_type_idx[i]);
#endif
	}

	// adding the two other grid types to mygrid
	strcpy(mygrid->grid_types[num_of_base_atypes],   "e");
	strcpy(mygrid->grid_types[num_of_base_atypes+1], "d");

	return 0;
}

int set_liganddata_typeid(
                                Liganddata* myligand,
                                int         atom_id,
                          const char*       typeof_new_atom
                         )
// The function sets the type index of the atom_id-th atom of myligand (in atom_idxyzq field),
// that is, it looks for the row in the atom_types field of myligand which is the same as
// typeof_new_atom, and sets the type index according to the row index.
// If the operation was successful, the function returns 0, if not, it returns 1.
{
	int i;
	int type;

	type = myligand->num_of_atypes; // setting type to an invalid index
	for (i=0; i < myligand->num_of_atypes; i++)
	{
		if (strcmp(myligand->atom_types[i], typeof_new_atom) == 0)
		{
			type = i;
		}
	}

	if (type < myligand->num_of_atypes)
	{
		myligand->atom_idxyzq[atom_id][0] = type;
		myligand->atom_map_to_fgrids[atom_id] = myligand->base_type_idx[type];
		return 0;
	}
	else // if typeof_new_atom hasn't been found
	{
		printf("Error: no grid for ligand atom type %s!\n", typeof_new_atom);
		return 1;
	}
}

void get_intraE_contributors(Liganddata* myligand)
// The function fills the intraE_contributors field of the myligand parameter according
// to its bonds and rigid_structures field, which must contain valid data when calling
// this function.
{
	int  atom_id1, atom_id2, atom_id3, rotb_id1, rotb_id2;
	unsigned int atom_neighbours [MAX_NUM_OF_ATOMS];
	unsigned int atom_neighbours_temp [MAX_NUM_OF_ATOMS];
	int  atom_id_a, atom_id_b, structure_id_A, structure_id_B;
	int  atom_id_a2, atom_id_b2;

	for (atom_id1=0; atom_id1 < myligand->num_of_atoms; atom_id1++)
		for (atom_id2=atom_id1; atom_id2 < myligand->num_of_atoms; atom_id2++)
			// initially, all the values are 1, that is, all the atom pairs
			if (atom_id1 != atom_id2)
			{
				// are contributors
				myligand->intraE_contributors[atom_id1][atom_id2] = 1;
				myligand->intraE_contributors[atom_id2][atom_id1] = 1;
			}
			// except if they are the same
			else
				myligand->intraE_contributors[atom_id1][atom_id2] = 0;

	// There are 5 cases when the atom pair's energy contribution
	// has not to be included in intramolecular energy calculation
	// (that is, when the distance of the atoms are constant during docking) <- 1-4 interactions do actually change

	// CASE 1
	// if the two atoms are members of the same rigid structure, they aren't contributors
	//printf("\n\n Members of the same rigid structure: \n\n");
	for (atom_id1=0; atom_id1 < myligand->num_of_atoms-1; atom_id1++)
		for (atom_id2=atom_id1+1; atom_id2 < myligand->num_of_atoms; atom_id2++)
			if (myligand->atom_rigid_structures[atom_id1] == myligand->atom_rigid_structures[atom_id2])
			{
				myligand->intraE_contributors[atom_id1][atom_id2] = 0;
				myligand->intraE_contributors[atom_id2][atom_id1] = 0;
				//printf("%d, %d\n", atom_id1+1, atom_id2+1);
			}
		//}

	// CASE2
	// if the atom pair represents a 1-2, 1-3 or 1-4 interaction, they aren't contributors
	// the following algorithm will find the first, second and third neighbours of each atom
	// (so the algorithm is redundant, several atoms will be found more than once)
	for (atom_id1=0; atom_id1 < myligand->num_of_atoms; atom_id1++)
	{
		// if atom_neighbours[i] is one,
		// it will indicate that the atom with id i is the neighbour of the atom with id atom_id1
		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
			if (myligand->bonds[atom_id1][atom_id2] > 0)
				atom_neighbours[atom_id2] = 1; // neighbour
			else
				atom_neighbours[atom_id2] = 0; // not neighbour

		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
			atom_neighbours_temp[atom_id2] = atom_neighbours [atom_id2]; // storing in a temp array as well

		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
			if (atom_neighbours[atom_id2] == 1) // for each neighbour of atom_id1
				for (atom_id3=0; atom_id3 < myligand->num_of_atoms; atom_id3++)
					if (myligand->bonds[atom_id2][atom_id3] > 0) // if atom_id3 is second neighbour of atom_id1
						atom_neighbours_temp[atom_id3] = 1; // changing the temporary array

		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
				atom_neighbours[atom_id2] = atom_neighbours_temp[atom_id2];

		// now ones of atom_neighbours indicate the first and second neighbours of atom_id1

		// the same code as above
		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
			if (atom_neighbours[atom_id2] == 1) // for each neighbour or second neighbour of atom_id1
				for (atom_id3=0; atom_id3 < myligand->num_of_atoms; atom_id3++)
					if (myligand->bonds[atom_id2][atom_id3] > 0) // if atom_id3 is second or third neighbour of atom_id1
						atom_neighbours_temp[atom_id3] = 1;

		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
			atom_neighbours[atom_id2] = atom_neighbours_temp[atom_id2];

		// now atom_neighbours [i] is one for atom_id1, its first, second and third neighbours, pairs consisting of
		// these atoms aren't contributors
		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
			if (atom_neighbours[atom_id2] == 1)
			{
				myligand->intraE_contributors[atom_id1][atom_id2] = 0;
				myligand->intraE_contributors[atom_id2][atom_id1] = 0;
			}
		int atom_typeid1 = myligand->atom_idxyzq[atom_id1][0];
		// take care of CG-G0 atoms and pairs
		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
		{
			int atom_typeid2 = myligand->atom_idxyzq[atom_id2][0];
			// Make sure G0 atoms do not interact with anything (except their respective CG partner as set below)
			if ((strncmp(myligand->base_atom_types[atom_typeid1], "G", 1) == 0) || (strncmp(myligand->base_atom_types[atom_typeid2], "G", 1) == 0)) {
				myligand->intraE_contributors[atom_id1][atom_id2] = 0;
				myligand->intraE_contributors[atom_id2][atom_id1] = 0;
			}
			// first, make sure non-matching ID pairs don't interact with each other (the code above happily allows it)
			if (  (myligand->bonds[atom_id1][atom_id2] == 0) && // for non-bonded CG-G0 atoms
			    (((strncmp(myligand->base_atom_types[atom_typeid1], "CG", 2) == 0) &&
			      (strncmp(myligand->base_atom_types[atom_typeid2], "G", 1) == 0) &&
			      (strcmp(myligand->base_atom_types[atom_typeid1]+2,myligand->base_atom_types[atom_typeid2]+1) != 0)) || // with non-matching ids
			     ((strncmp(myligand->base_atom_types[atom_typeid1], "G", 1) == 0) &&
			      (strncmp(myligand->base_atom_types[atom_typeid2], "CG", 2) == 0) &&
			      (strcmp(myligand->base_atom_types[atom_typeid1]+1,myligand->base_atom_types[atom_typeid2]+2) != 0))))
			{
				myligand->intraE_contributors[atom_id1][atom_id2] = 0;
				myligand->intraE_contributors[atom_id2][atom_id1] = 0;
			}
			// second, let matching ID pair interact
			if (  (myligand->bonds[atom_id1][atom_id2] == 0) && // for non-bonded CG-G0 atoms
			    (((strncmp(myligand->base_atom_types[atom_typeid1], "CG", 2) == 0) &&
			      (strncmp(myligand->base_atom_types[atom_typeid2], "G", 1) == 0) &&
			      (strcmp(myligand->base_atom_types[atom_typeid1]+2,myligand->base_atom_types[atom_typeid2]+1) == 0)) || // with matching ids
			     ((strncmp(myligand->base_atom_types[atom_typeid1], "G", 1) == 0) &&
			      (strncmp(myligand->base_atom_types[atom_typeid2], "CG", 2) == 0) &&
			      (strcmp(myligand->base_atom_types[atom_typeid1]+1,myligand->base_atom_types[atom_typeid2]+2) == 0))))
			{
				myligand->intraE_contributors[atom_id1][atom_id2] = 1;
				myligand->intraE_contributors[atom_id2][atom_id1] = 1;
#if defined(CG_G0_INFO)
				printf("Found CG-G0 pair: atom %i (%s) <-> atom %i (%s)\n",atom_id1+1,myligand->base_atom_types[atom_typeid1],atom_id2+1,myligand->base_atom_types[atom_typeid2]);
#endif
			}
		}
	}

	// CASE3
	// Let atom a and atom b be the endpoints of the same rotatable bond,
	// and A and B the rigid structures connected
	// to the rotatable bond's a and b atoms, respectively.
	// The atom pairs consisting of a and any atom of B aren't contributors.
	// Similarly, atom pairs consisting of b and any atom of A aren't, either.

	for (rotb_id1=0; rotb_id1 < myligand->num_of_rotbonds; rotb_id1++)
	{
		atom_id_a = myligand->rotbonds[rotb_id1][0];
		atom_id_b = myligand->rotbonds[rotb_id1][1];

		structure_id_A = myligand->atom_rigid_structures[atom_id_a];
		structure_id_B = myligand->atom_rigid_structures[atom_id_b];

		for (atom_id1=0; atom_id1 < myligand->num_of_atoms; atom_id1++)
		{
			// if atom_id1 is member of structure A
			if (myligand->atom_rigid_structures[atom_id1] == structure_id_A)
			{
				myligand->intraE_contributors[atom_id1][atom_id_b] = 0;
				myligand->intraE_contributors[atom_id_b][atom_id1] = 0;
			}

			// if atom_id1 is member of structure B
			if (myligand->atom_rigid_structures[atom_id1] == structure_id_B)
			{
				myligand->intraE_contributors[atom_id1][atom_id_a] = 0;
				myligand->intraE_contributors[atom_id_a][atom_id1] = 0;
			}
		}
	}

	// CASE4
	// If one end of two different rotatable bonds are connected to the same rigid structure, the other end, that is,
	// atoms of the bonds aren't contributors.

	for (rotb_id1=0; rotb_id1 < myligand->num_of_rotbonds-1; rotb_id1++)
		for (rotb_id2=rotb_id1+1; rotb_id2 < myligand->num_of_rotbonds; rotb_id2++)
		{
			atom_id_a  = myligand->rotbonds[rotb_id1][0];
			atom_id_b  = myligand->rotbonds[rotb_id1][1];
			atom_id_a2 = myligand->rotbonds[rotb_id2][0];
			atom_id_b2 = myligand->rotbonds[rotb_id2][1];

			if (myligand->atom_rigid_structures[atom_id_a] == myligand->atom_rigid_structures[atom_id_a2])
			{
				myligand->intraE_contributors[atom_id_b][atom_id_b2] = 0;
				myligand->intraE_contributors[atom_id_b2][atom_id_b] = 0;
			}
			if (myligand->atom_rigid_structures[atom_id_a] == myligand->atom_rigid_structures[atom_id_b2])
			{
				myligand->intraE_contributors[atom_id_b][atom_id_a2] = 0;
				myligand->intraE_contributors[atom_id_a2][atom_id_b] = 0;
			}
			if (myligand->atom_rigid_structures[atom_id_b] == myligand->atom_rigid_structures[atom_id_a2])
			{
				myligand->intraE_contributors[atom_id_a][atom_id_b2] = 0;
				myligand->intraE_contributors[atom_id_b2][atom_id_a] = 0;
			}
			if (myligand->atom_rigid_structures[atom_id_b] == myligand->atom_rigid_structures[atom_id_b2])
			{
				myligand->intraE_contributors[atom_id_a][atom_id_a2] = 0;
				myligand->intraE_contributors[atom_id_a2][atom_id_a] = 0;
			}
		}

	// CASE5
	// One of the atoms is a W atom
	for (atom_id1=0; atom_id1 < myligand->num_of_atoms-1; atom_id1++) {
		int atom_id1_type = myligand->atom_idxyzq[atom_id1][0];
		for (atom_id2=atom_id1+1; atom_id2 < myligand->num_of_atoms; atom_id2++) {
			int atom_id2_type = myligand->atom_idxyzq[atom_id2][0];
			if ((strcmp(myligand->base_atom_types[atom_id1_type], "W") == 0) || (strcmp(myligand->base_atom_types[atom_id2_type], "W") == 0))
			{
				myligand->intraE_contributors[atom_id1][atom_id2] = 0;
				myligand->intraE_contributors[atom_id2][atom_id1] = 0;
			}
		}
	}
}

int get_bonds(Liganddata* myligand)
// The function fills the bonds field of myligand based on the distance of the ligand's atoms,
// which can be calculated from the atom_idxyzq field, so this field must contain valid data
// when calling this function.
{
	char atom_names [ATYPE_GETBONDS][3];
	// Values from atomic parameter file AD4.1_bound_dat / "bond_index"
	unsigned int bondtype_id [ATYPE_GETBONDS] = {
	                                     0, // "C"
	                                     0, // "A"
	                                     3, // "Hx"
	                                     1, // "Nx"
	                                     2, // "Ox"
	                                     4, // "F"
	                                     4, // "MG"
	                                     5, // "P"
	                                     6, // "Sx"
	                                     4, // "CL"
	                                     4, // "CA"
	                                     4, // "MN"
	                                     4, // "FE"
	                                     4, // "ZN"
	                                     4, // "BR"
	                                     4, // "I"
	                                     0, // "CG"
	                                     0, // "G0"
	                                     2, // "W" as oxygen, but irrelevant, all bonds containing W will be disabled
	                                     0, // "CX"
	                                     6  // "SI"
	                                    };

	double mindist[NUM_ENUM_ATOMTYPES][NUM_ENUM_ATOMTYPES];
	double maxdist[NUM_ENUM_ATOMTYPES][NUM_ENUM_ATOMTYPES];

	double temp_point1 [3];
	double temp_point2 [3];
	double temp_dist;

	int atom_id1, atom_id2, i, j;
	int atom_typeid1, atom_typeid2;
	int atom_nameid1, atom_nameid2;
	int bondtype_id1, bondtype_id2;

	int W_nameid = 18; // hard-coded (sorry!) id for W atom
	int CG_nameid = 16;

	strcpy(atom_names[0],  "C");
	strcpy(atom_names[1],  "A");
	strcpy(atom_names[2],  "Hx");
	strcpy(atom_names[3],  "Nx");
	strcpy(atom_names[4],  "Ox");
	strcpy(atom_names[5],  "F");
	strcpy(atom_names[6],  "MG");
	strcpy(atom_names[7],  "P");
	strcpy(atom_names[8],  "Sx");
	strcpy(atom_names[9],  "CL");
	strcpy(atom_names[10], "CA");
	strcpy(atom_names[11], "MN");
	strcpy(atom_names[12], "FE");
	strcpy(atom_names[13], "ZN");
	strcpy(atom_names[14], "BR");
	strcpy(atom_names[15], "I");
	strcpy(atom_names[16], "CG");
	strcpy(atom_names[17], "G0");
	strcpy(atom_names[18], "W"); // used to disable all bonds containing W
	strcpy(atom_names[19], "CX"); // used to disable all bonds containing W
	strcpy(atom_names[20], "SI");

#ifdef AD4_BOND_DISTS
	// Set all the mindist and maxdist elements to the defaults for AutoDock versions 1 - 3...
	for (i=0; i<   NUM_ENUM_ATOMTYPES; i++) {
		for (j=0; j<   NUM_ENUM_ATOMTYPES; j++) {
			mindist[i][j] = 0.9 - BOND_LENGTH_TOLERANCE;
			maxdist[i][j] = 2.1 + BOND_LENGTH_TOLERANCE;
		}
	}
	/*
	 * These values, unless otherwise stated,
	 * are taken from "handbook of Chemistry and Physics"
	 * 44th edition(!)
	 */
	set_minmax(C, C, 1.20, 1.545); // mindist[C][C] = 1.20, p. 3510 ; maxdist[C][C] = 1.545, p. 3511
	set_minmax(C, N, 1.1, 1.479); // mindist[C][N] = 1.1, p. 3510 ; maxdist[C][N] = 1.479, p. 3511
	set_minmax(C, O, 1.15, 1.47); // mindist[C][O] = 1.15, p. 3510 ; maxdist[C][O] = 1.47, p. 3512
	set_minmax(C, H, 1.022, 1.12);  // p. 3518, p. 3517
	set_minmax(C, XX, 0.9, 1.545); // mindist[C][XX] = 0.9, AutoDock 3 defaults ; maxdist[C][XX] = 1.545, p. 3511
	set_minmax(C, P, 1.85, 1.89); // mindist[C][P] = 1.85, p. 3510 ; maxdist[C][P] = 1.89, p. 3510
	set_minmax(C, S, 1.55, 1.835); // mindist[C][S] = 1.55, p. 3510 ; maxdist[C][S] = 1.835, p. 3512
	set_minmax(N, N, 1.0974, 1.128); // mindist[N][N] = 1.0974, p. 3513 ; maxdist[N][N] = 1.128, p. 3515
	set_minmax(N, O, 1.0619, 1.25); // mindist[N][O] = 1.0975, p. 3515 ; maxdist[N][O] = 1.128, p. 3515
	set_minmax(N, H, 1.004, 1.041); // mindist[N][H] = 1.004, p. 3516 ; maxdist[N][H] = 1.041, p. 3515
	set_minmax(N, XX, 0.9, 1.041); // mindist[N][XX] = 0.9, AutoDock 3 defaults ; maxdist[N][XX] = 1.041, p. 3515
	set_minmax(N, P, 1.4910, 1.4910); // mindist[N][P] = 1.4910, p. 3515 ; maxdist[N][P] = 1.4910, p. 3515
	set_minmax(N, S, 1.58, 1.672); // mindist[N][S] = 1.58, 1czm.pdb sulfonamide ; maxdist[N][S] = 1.672, J. Chem. SOC., Dalton Trans., 1996, Pages 4063-4069 
	set_minmax(O, O, 1.208, 1.51); // p.3513, p.3515
	set_minmax(O, H, 0.955, 1.0289); // mindist[O][H] = 0.955, p. 3515 ; maxdist[O][H] = 1.0289, p. 3515
	set_minmax(O, XX, 0.955, 2.1); // AutoDock 3 defaults
	set_minmax(O, P, 1.36, 1.67); // mindist[O][P] = 1.36, p. 3516 ; maxdist[O][P] = 1.67, p. 3517
	set_minmax(O, S, 1.41, 1.47); // p. 3517, p. 3515
	set_minmax(H, H, 100.,-100.); // impossible values to prevent such bonds from forming.
	set_minmax(H, XX, 0.9, 1.5); // AutoDock 4 defaults
	set_minmax(H, P, 1.40, 1.44); // mindist[H][P] = 1.40, p. 3515 ; maxdist[H][P] = 1.44, p. 3515
	set_minmax(H, S, 1.325, 1.3455); // mindist[H][S] = 1.325, p. 3518 ; maxdist[H][S] = 1.3455, p. 3516
	set_minmax(XX, XX, 0.9, 2.1); // AutoDock 3 defaults
	set_minmax(XX, P, 0.9, 2.1); // AutoDock 3 defaults
	set_minmax(XX, S, 1.325, 2.1); // mindist[XX][S] = 1.325, p. 3518 ; maxdist[XX][S] = 2.1, AutoDock 3 defaults
	set_minmax(P, P, 2.18, 2.23); // mindist[P][P] = 2.18, p. 3513 ; maxdist[P][P] = 2.23, p. 3513
	set_minmax(P, S, 1.83, 1.88); // mindist[P][S] = 1.83, p. 3516 ; maxdist[P][S] = 1.88, p. 3515
	set_minmax(S, S, 2.03, 2.05); // mindist[S][S] = 2.03, p. 3515 ; maxdist[S][S] = 2.05, p. 3515
	/* end values from Handbook of Chemistry and Physics */
#else
	// Filling the mindist and maxdist tables (as in Autodock, see AD4_parameters.dat and mdist.h).
	// It is supposed that the bond length of atoms with bondtype_id1 and bondtype_id2 is
	// between mindist[bondtype_id1][bondtype_id2] and maxdist[bondtype_id1][bondtype_id2]
	for (i=0; i<7; i++)
	{
		for (j=0; j<7; j++)
		{
			mindist[i][j] = 0.9;
			maxdist[i][j] = 2.1;
		}
	}

	//0=C, 3=H
	mindist[C][H] = 1.07; mindist[H][C] = mindist[C][H];
	maxdist[C][H] = 1.15; maxdist[H][C] = maxdist[C][H];

	//1=N
	mindist[N][H] = 0.99; mindist[H][N] = mindist[N][H];
	maxdist[N][H] = 1.10; maxdist[H][N] = maxdist[N][H];

	//2=O
	mindist[O][H] = 0.94; mindist[H][O] = mindist[O][H];
	maxdist[O][H] = 1.10; maxdist[H][O] = maxdist[O][H];

	//6=S
	mindist[S][H] = 1.316; mindist[H][S] = mindist[S][H];
	maxdist[S][H] = 1.356; maxdist[H][S] = maxdist[S][H];

	//5=P
	mindist[P][H] = 1.35; mindist[H][P] = mindist[P][H];
	maxdist[P][H] = 1.40; maxdist[H][P] = maxdist[P][H];

	mindist[N][O] = 1.11;  // N=O is ~ 1.21 A, minus 0.1A error
	maxdist[N][O] = 1.50;  // N-O is ~ 1.40 A, plus 0.1 A error
	mindist[O][N] = mindist[N][O];  // N=O is ~ 1.21 A, minus 0.1A error
	maxdist[O][N] = maxdist[N][O];  // N-O is ~ 1.40 A, plus 0.1 A error

	// There is no bond between two hydrogens (does not derive from Autodock)
	mindist[H][H] = 2;
	maxdist[H][H] = 1;
#endif
	
	memset(myligand->bonds,0,MAX_NUM_OF_ATOMS*MAX_NUM_OF_ATOMS*sizeof(char));

	bool is_HD1, is_HD2;
	memset(myligand->donor,0,MAX_NUM_OF_ATOMS*sizeof(bool));
	double HD_dists[MAX_NUM_OF_ATOMS];
	int HD_ids[MAX_NUM_OF_ATOMS];
	memset(HD_ids,0xFF,MAX_NUM_OF_ATOMS*sizeof(int));
	// make sure the last ones are initialized too (saves work in second loop)
	myligand->acceptor[myligand->num_of_atoms-1] = is_H_acceptor(myligand->base_atom_types[(int)(myligand->atom_idxyzq[myligand->num_of_atoms-1][0])]);
	char reactnum = myligand->atom_types[(int)(myligand->atom_idxyzq[myligand->num_of_atoms-1][0])][strlen(myligand->atom_types[(int)(myligand->atom_idxyzq[myligand->num_of_atoms-1][0])])-1];
	myligand->reactive[myligand->num_of_atoms-1] = ((reactnum=='1') || (reactnum=='4') || (reactnum=='7'));

	for (atom_id1=0; atom_id1 < myligand->num_of_atoms-1; atom_id1++)
	{
		atom_typeid1 = myligand->atom_idxyzq[atom_id1][0];
		myligand->acceptor[atom_id1] = is_H_acceptor(myligand->base_atom_types[atom_typeid1]);
		reactnum = myligand->atom_types[atom_typeid1][strlen(myligand->atom_types[atom_typeid1])-1];
		myligand->reactive[atom_id1] = ((reactnum=='1') || (reactnum=='4') || (reactnum=='7'));
		is_HD1=(strcmp(myligand->base_atom_types[atom_typeid1],"HD")==0);
		for (atom_id2=atom_id1+1; atom_id2 < myligand->num_of_atoms; atom_id2++)
		{
			atom_typeid2 = myligand->atom_idxyzq[atom_id2][0];
			is_HD2=(strcmp(myligand->base_atom_types[atom_typeid2],"HD")==0);
			temp_point1[0] = myligand->atom_idxyzq[atom_id1][1];
			temp_point1[1] = myligand->atom_idxyzq[atom_id1][2];
			temp_point1[2] = myligand->atom_idxyzq[atom_id1][3];
			temp_point2[0] = myligand->atom_idxyzq[atom_id2][1];
			temp_point2[1] = myligand->atom_idxyzq[atom_id2][2];
			temp_point2[2] = myligand->atom_idxyzq[atom_id2][3];
			temp_dist = distance(temp_point1, temp_point2);
			atom_nameid1 = ATYPE_GETBONDS;
			atom_nameid2 = ATYPE_GETBONDS;
			// identifying atom types
			for (i=0; i<ATYPE_GETBONDS; i++)
			{
				if ((atom_names[i][1] == 'x') || (atom_names[i][1] == '0')) // this catches "G0..9"
				{
					if (atom_names[i][0] == toupper(myligand->base_atom_types[atom_typeid1][0]))
						atom_nameid1 = i;
				}
				else
				{
					if (strincmp(atom_names[i], myligand->base_atom_types[atom_typeid1], 2) == 0)
						atom_nameid1 = i;
				}
			}

			for (i=0; i<ATYPE_GETBONDS; i++)
			{
				if ((atom_names[i][1] == 'x') || (atom_names[i][1] == '0')) // this catches "G0..9"
				{
					if (atom_names[i][0] == toupper(myligand->base_atom_types[atom_typeid2][0]))
						atom_nameid2 = i;
				}
				else
				{
					if (strincmp(atom_names[i], myligand->base_atom_types[atom_typeid2], 2) == 0)
					{
						atom_nameid2 = i;
					}
				}
			}

			if ((atom_nameid1 == ATYPE_GETBONDS) || (atom_nameid2 == ATYPE_GETBONDS))
			{
				printf("Error: Ligand includes atom with unknown type: %s or %s!\n", myligand->base_atom_types[atom_typeid1], myligand->base_atom_types[atom_typeid2]);
				return 1;
			}

			bondtype_id1 = bondtype_id[atom_nameid1];
			bondtype_id2 = bondtype_id[atom_nameid2];

			// W atoms are never bonded to any other atom
			if ((atom_nameid1 == W_nameid) || (atom_nameid2 == W_nameid))
			{
				myligand->bonds [atom_id1][atom_id2] = 0;
				myligand->bonds [atom_id2][atom_id1] = 0;
			}
			else if (((temp_dist >= mindist [bondtype_id1][bondtype_id2]) && (temp_dist <= maxdist [bondtype_id1][bondtype_id2])) || (atom_id1 == atom_id2))
				{
					myligand->bonds [atom_id1][atom_id2] = 1;
					myligand->bonds [atom_id2][atom_id1] = 1;
					if(is_HD1 || is_HD2){ // closest O,N,S is going to be an H-bond donor
						unsigned int HD_id = is_HD1 ? atom_id1 : atom_id2;
						unsigned int heavy_id = is_HD1 ? atom_id2 : atom_id1;
						char heavy=myligand->base_atom_types[is_HD1 ? atom_typeid2 : atom_typeid1][0];
						if((heavy=='O') || (heavy=='N') || (heavy=='S')){
							if((HD_ids[HD_id]<0) ||
							   ((HD_ids[HD_id]>=0) && (temp_dist<HD_dists[HD_id]))){
								HD_ids[HD_id] = heavy_id;
								HD_dists[HD_id] = temp_dist;
							}
						}
					}
				}
				else if ((atom_nameid1 == CG_nameid) && (atom_nameid2 == CG_nameid) && // two CG atoms
					 (strcmp(myligand->base_atom_types[atom_typeid1]+2,myligand->base_atom_types[atom_typeid2]+2) == 0)) // and matching numbers
					{
						myligand->bonds [atom_id1][atom_id2] = 2; // let's call 2 a "virtual" bond to
						myligand->bonds [atom_id2][atom_id1] = 2; // distinguish them if needed later
#if defined(CG_G0_INFO)
						printf("Found virtual CG-CG bond between atom %i (%s) and atom %i (%s).\n",atom_id1+1,myligand->base_atom_types[atom_typeid1],atom_id2+1,myligand->base_atom_types[atom_typeid2]);
#endif
					}
		} // inner for-loop
	} // outer for-loop

	for(int i=0; i<myligand->num_of_atoms; i++){
		if(HD_ids[i]>=0)
			myligand->donor[i]=true;
		else if(strcmp(myligand->base_atom_types[(int)(myligand->atom_idxyzq[i][0])],"HD")==0) myligand->donor[i]=true;
	}

	return 0;
}

pair_mod* is_mod_pair(
                      const char*     A,
                      const char*     B,
                            int       nr_mod_atype_pairs,
                            pair_mod* mod_atype_pairs
                     )
{
	for(int i=0; i<nr_mod_atype_pairs; i++) // need to make sure there isn't a modified pair
		if ( ((strcmp(mod_atype_pairs[i].A, A) == 0) && (strcmp(mod_atype_pairs[i].B, B) == 0)) ||
		     ((strcmp(mod_atype_pairs[i].A, B) == 0) && (strcmp(mod_atype_pairs[i].B, A) == 0)))
			return &mod_atype_pairs[i];
	return NULL;
}

int get_VWpars(
                     Liganddata*  myligand,
               const double       AD4_coeff_vdW,
               const double       AD4_coeff_hb,
                     int          nr_deriv_atypes,
                     deriv_atype* deriv_atypes,
                     int          nr_mod_atype_pairs,
                     pair_mod*    mod_atype_pairs
              )
// The function calculates the Van der Waals parameters for each pair of atom
// types of the ligand given by the first parameter, and fills the VWpars_A, _B,
// _C and _D fields according to the result as well as the solvation parameters
// and atomic volumes (desolv and volume fields) for each atom type.
{
	char atom_names [ATYPE_NUM][3];

	// Initial implementation included 22 atom-types.
	// Handling flexrings requires 2 additional atom-types: "CG" & "G0".
	// All corresponding CG & G0 values are added as last 2 elements
	// in existing atom-types look-up tables.

	// See in ./common/defines.h: ATYPE_CG_IDX and ATYPE_G0_IDX
	// CG (idx=22): a copy of the standard "C" (idx=3) atom-type
	// G0 (idx=23): an invisible atom-type, all atomic parameters are zero
	

	// Sum of vdW radii of two like atoms (A)
	double reqm [ATYPE_NUM] = {
	                           2.00, 2.00, 2.00, 4.00, 4.00,
	                           3.50, 3.50, 3.50, 3.20, 3.20,
	                           3.09, 1.30, 4.20, 4.00, 4.00,
	                           4.09, 1.98, 1.30, 1.30, 1.48,
	                           4.33, 4.72, 4.10,
	                           4.00, // CG
	                           0.00, // G0
	                           0.00, // W
	                           4.00, // CX
	                           3.50, // NX
	                           3.20  // OX
	                          };

	// cdW well depth (kcal/mol)
	double eps [ATYPE_NUM] = {
	                          0.020, 0.020, 0.020, 0.150, 0.150,
	                          0.160, 0.160, 0.160, 0.200, 0.200,
	                          0.080, 0.875, 0.200, 0.200, 0.200,
	                          0.276, 0.550, 0.875, 0.010, 0.550,
	                          0.389, 0.550, 0.200,
	                          0.150, // CG
	                          0.000, // G0
	                          0.000, // W
	                          0.150, // CX
	                          0.160, // NX
	                          0.200  // OX
	                         };

	// Sum of vdW radii of two like atoms (A) in case of hydrogen bond
	double reqm_hbond [ATYPE_NUM] = {
	                                 0.0, 0.0, 0.0, 0.0, 0.0,
	                                 0.0, 1.9, 1.9, 1.9, 1.9,
	                                 0.0, 0.0, 0.0, 2.5, 0.0,
	                                 0.0, 0.0, 0.0, 0.0, 0.0,
	                                 0.0, 0.0, 0.0,
	                                 0.0, // CG
	                                 0.0, // G0
	                                 0.0, // W
	                                 0.0, // CX
	                                 0.0, // NX
	                                 1.9  // OX
	                                };

	// cdW well depth (kcal/mol) in case of hydrogen bond
	double eps_hbond [ATYPE_NUM] = {
	                                0.0, 1.0, 1.0, 0.0, 0.0,         //HD and HS value is 1 so that it is not necessary to decide which atom_typeid
	                                0.0, 5.0, 5.0, 5.0, 5.0,         //corresponds to the hydrogen when reading eps_hbond...
	                                0.0, 0.0, 0.0, 1.0, 0.0,
	                                0.0, 0.0, 0.0, 0.0, 0.0,
	                                0.0, 0.0, 0.0,
	                                0.0, // CG
	                                0.0, // G0
	                                0.0, // W
	                                0.0, // CX
	                                0.0, // NX
	                                5.0  // OX
	                               };

	// volume of atoms
	double volume [ATYPE_NUM] = {
	                              0.0000,  0.0000,  0.0000, 33.5103, 33.5103,
	                             22.4493, 22.4493, 22.4493, 17.1573, 17.1573,
	                             15.4480,  1.5600, 38.7924, 33.5103, 33.5103,
	                             35.8235,  2.7700,  2.1400,  1.8400,  1.7000,
	                             42.5661, 55.0585, 35.8235,
	                             33.5103, // CG
	                              0.0000, // G0
	                              0.0000, // W
	                             33.5103, // CX
	                             22.4493, // NX
	                             17.1573  // OX
	                            };

	// atomic solvation parameters
	double solpar [ATYPE_NUM] = {
	                              0.00051,  0.00051,  0.00051, -0.00143, -0.00052,
	                             -0.00162, -0.00162, -0.00162, -0.00251, -0.00251,
	                             -0.00110, -0.00110, -0.00110, -0.00214, -0.00214,
	                             -0.00110, -0.00110, -0.00110, -0.00110, -0.00110,
	                             -0.00110, -0.00110, -0.00143,
	                             -0.00143, // CG
	                              0.00000, // G0
	                              0.00000, // W
	                             -0.00143, // CX
	                             -0.00162, // NX
	                             -0.00251  // OX
	                            };

	int atom_typeid1, atom_typeid2, VWid_atype1, VWid_atype2, i;
	double eps12, reqm12;
	pair_mod* pm;

	strcpy(atom_names [0], "H");
	strcpy(atom_names [1], "HD");
	strcpy(atom_names [2], "HS");
	strcpy(atom_names [3], "C");
	strcpy(atom_names [4], "A");
	strcpy(atom_names [5], "N");
	strcpy(atom_names [6], "NA");
	strcpy(atom_names [7], "NS");
	strcpy(atom_names [8], "OA");
	strcpy(atom_names [9], "OS");
	strcpy(atom_names [10], "F");
	strcpy(atom_names [11], "MG");
	strcpy(atom_names [12], "P");
	strcpy(atom_names [13], "SA");
	strcpy(atom_names [14], "S");
	strcpy(atom_names [15], "CL");
	strcpy(atom_names [16], "CA");
	strcpy(atom_names [17], "MN");
	strcpy(atom_names [18], "FE");
	strcpy(atom_names [19], "ZN");
	strcpy(atom_names [20], "BR");
	strcpy(atom_names [21], "I");
	strcpy(atom_names [22], "SI");
	strcpy(atom_names [/*23*/ATYPE_CG_IDX], "CG"); // CG
	strcpy(atom_names [/*24*/ATYPE_G0_IDX], "G0"); // G0
	strcpy(atom_names [/*25*/ATYPE_W_IDX], "W"); // W
	strcpy(atom_names [/*26*/ATYPE_CX_IDX], "CX"); // CX
	strcpy(atom_names [/*27*/ATYPE_NX_IDX], "NX"); // NX
	strcpy(atom_names [/*28*/ATYPE_OX_IDX], "OX"); // OX
//	for(unsigned int i=0; i<nr_deriv_atypes; i++) // add derivative type names to get proper type ids
//		strcpy(atom_names[ATYPE_NUM+i],deriv_atypes[i].deriv_name);

	// Using this variable to signal when the CG-CG pair was found.
	// This is further reused to set vdW constant coeffs: "vdWpars_A" and "vdWpars_B".
	// found_CG_CG_pair == true  -> set vdW coeffs to zero
	// found_CG_CG_pair == false -> use vdW default values
	bool found_CG_CG_pair;

	for (atom_typeid1 = 0; atom_typeid1 < myligand->num_of_atypes; atom_typeid1++)
	{
		VWid_atype1 = ATYPE_NUM;
		// identifying atom types
		for (i=0; i<ATYPE_NUM; i++) {
			if (strincmp(atom_names [i], myligand->base_atom_types [atom_typeid1],2) == 0) {
				VWid_atype1 = i;
				myligand->atom_types_reqm [atom_typeid1] = VWid_atype1;
				break;
			}
			else
			{
				if(atom_names[i][1] == '0') {
					if (atom_names[i][0] == toupper(myligand->base_atom_types[atom_typeid1][0])) {
						VWid_atype1 = i;
						myligand->atom_types_reqm [atom_typeid1] = VWid_atype1;
						break;
					}
				}
			}
		}
		if(VWid_atype1==ATYPE_NUM){
			printf("Error: Could not identify ligand atom type %s.\n",myligand->base_atom_types [atom_typeid1]);
			exit(1);
		}
	}
	for (atom_typeid1 = 0; atom_typeid1 < myligand->num_of_atypes; atom_typeid1++)
	{
		VWid_atype1 = myligand->atom_types_reqm [atom_typeid1];
		for (atom_typeid2 = 0; atom_typeid2 < myligand->num_of_atypes; atom_typeid2++)
		{
			VWid_atype2 = myligand->atom_types_reqm [atom_typeid2];

			// Was CG_CG_pair found?
			found_CG_CG_pair = false;

			// Was CG_CG_pair found?
			if ((VWid_atype1 == ATYPE_CG_IDX) && (VWid_atype2 == ATYPE_CG_IDX) &&
			    (strcmp(myligand->base_atom_types[atom_typeid1]+2,myligand->base_atom_types[atom_typeid2]+2) == 0)) { // make sure to only exclude matching IDs
				found_CG_CG_pair = true;
			}
			else {
				found_CG_CG_pair = false;
			}

			if (VWid_atype1 == MAX_NUM_OF_ATYPES)
			{
				printf("Error: Ligand includes atom with unknown type 1: %s!\n", myligand->atom_types [atom_typeid1]);
				return 1;
			}

			if  (VWid_atype2 == MAX_NUM_OF_ATYPES)
			{
				printf("Error: Ligand includes atom with unknown type 2: %s!\n", myligand->atom_types [atom_typeid2]);
				return 1;
			}

			myligand->VWpars_exp  [atom_typeid1][atom_typeid2] = (12 << 8) + 6; // shift first exponent right by 8 bit
			// calculating van der Waals parameters
			if (is_H_bond(myligand->base_atom_types [atom_typeid1], myligand->base_atom_types [atom_typeid2]))
			{
				eps12 = AD4_coeff_hb * eps_hbond [VWid_atype1] * eps_hbond [VWid_atype2]; // The hydrogen's eps is 1, doesn't change the value...
				reqm12 = reqm_hbond [VWid_atype1] + reqm_hbond [VWid_atype2]; // The hydrogen's is 0, doesn't change the value...
				myligand->VWpars_exp  [atom_typeid1][atom_typeid2] = (12 << 8) + 10; // shift first exponent right by 8 bit
				myligand->reqm_AB  [atom_typeid1][atom_typeid2] = reqm12;
				myligand->VWpars_C [atom_typeid1][atom_typeid2] = 5*eps12*pow(reqm12, 12);
				myligand->VWpars_D [atom_typeid1][atom_typeid2] = 6*eps12*pow(reqm12, 10);
				myligand->VWpars_A [atom_typeid1][atom_typeid2] = 0;
				myligand->VWpars_B [atom_typeid1][atom_typeid2] = 0;
			}
			else
			{
				eps12 = AD4_coeff_vdW * sqrt(eps [VWid_atype1]*eps [VWid_atype2]); //weighting with coefficient for van der Waals term
				reqm12 = 0.5*(reqm [VWid_atype1]+reqm [VWid_atype2]);
				myligand->reqm_AB  [atom_typeid1][atom_typeid2] = reqm12;

				// Was CG_CG_pair found?
				if (found_CG_CG_pair == true) { // Zero for CG-CG atomic pair
					myligand->VWpars_A [atom_typeid1][atom_typeid2] = 0.0;
					myligand->VWpars_B [atom_typeid1][atom_typeid2] = 0.0;
				} else { // Using default values for any atomic-pair different than CG-CG
					myligand->VWpars_A [atom_typeid1][atom_typeid2] =   eps12*pow(reqm12, 12);
					myligand->VWpars_B [atom_typeid1][atom_typeid2] = 2*eps12*pow(reqm12, 6);
				}
				myligand->VWpars_C [atom_typeid1][atom_typeid2] = 0;
				myligand->VWpars_D [atom_typeid1][atom_typeid2] = 0;

/*
				// ----------------------------------
				// Smoothing test
				eps12 = AD4_coeff_vdW * sqrt(eps [3]*eps [8]); //weighting with coefficient for van der Waals term
				reqm12 = 0.5*(reqm [3]+reqm [8]);
				
				printf("epsii (C): %f\n", eps [3]);
				printf("epsii (OA): %f\n", eps [8]);
				printf("epsij: %f\n", eps12);
				printf("rij: %f\n", reqm12);
				printf("C12=%f\n", eps12*pow(reqm12, 12));
				printf("C6=%f\n", 2*eps12*pow(reqm12, 6));

				// ----------------------------------
*/				

			}
			if( ( pm = is_mod_pair(	myligand->atom_types[atom_typeid1],
						myligand->atom_types[atom_typeid2],
						nr_mod_atype_pairs,
						mod_atype_pairs) ) )
			{
				reqm12 = pm->parameters[0];
				eps12  = pm->parameters[1];
				int m = (myligand->VWpars_exp [atom_typeid1][atom_typeid2] & 0xFF00) >> 8;
				int n = (myligand->VWpars_exp [atom_typeid1][atom_typeid2] & 0xFF);
				if(pm->nr_parameters>3){ // LJ exponents are specified
					m=(int)pm->parameters[2];
					n=(int)pm->parameters[3];
					myligand->VWpars_exp [atom_typeid1][atom_typeid2] = (m << 8) + n; // shift first exponent right by 8 bit
				}
				eps12 *= 1.0f/float(m-n);
				myligand->reqm_AB  [atom_typeid1][atom_typeid2] = reqm12;
				myligand->VWpars_A [atom_typeid1][atom_typeid2] = eps12*pow(reqm12, m)*n;
				myligand->VWpars_B [atom_typeid1][atom_typeid2] = eps12*pow(reqm12, n)*m;
				myligand->VWpars_C [atom_typeid1][atom_typeid2] = 0;
				myligand->VWpars_D [atom_typeid1][atom_typeid2] = 0;
			}
		}
	}

	for (atom_typeid1 = 0; atom_typeid1 < myligand->num_of_atypes; atom_typeid1++)
	{
		VWid_atype1 = ATYPE_NUM;

		// identifying atom type
		for (i=0; i<ATYPE_NUM; i++) {
			if (strincmp(atom_names [i], myligand->base_atom_types [atom_typeid1], 2) == 0) // captures GG0..9 to CG in tables
			{
				VWid_atype1 = i;
			}
			else
			{
				if(atom_names[i][1] == '0') { // captures G0..9 to G0 in tables
					if (atom_names[i][0] == toupper(myligand->base_atom_types[atom_typeid1][0]))
						VWid_atype1 = i;
				}
			}
		}

		if (VWid_atype1 == ATYPE_NUM)
		{
			printf("Error: Ligand includes atom with unknown type: %s\n", myligand->base_atom_types [atom_typeid1]);
			return 1;
		}

		myligand->volume [atom_typeid1] = volume [VWid_atype1];
		myligand->solpar [atom_typeid1] = solpar [VWid_atype1];
	}

	return 0;
}

int get_moving_and_unit_vectors(Liganddata* myligand)
// The function calculates and fills the
// rotbonds_moving_vectors and rotbonds_unit_vectors fields of the myligand parameter.
{
	int rotb_id, i;
	int atom_id_pointA, atom_id_pointB;
	double origo [3];
	double movvec [3];
	double unitvec [3];
	double pointA [3];
	double pointB [3];
	double dist;


	for (rotb_id=0; rotb_id<myligand->num_of_rotbonds; rotb_id++)
	{
		// capturing unit vector's direction
		atom_id_pointA = myligand->rotbonds [rotb_id][0]; // capturing indexes of the two atoms
		atom_id_pointB = myligand->rotbonds [rotb_id][1];
		for (i=0; i<3; i++) // capturing coordinates of the two atoms
		{
			pointA [i] = myligand->atom_idxyzq [atom_id_pointA][i+1];
			pointB [i] = myligand->atom_idxyzq [atom_id_pointB][i+1];
			unitvec [i] = pointB [i] - pointA [i];
		}

		// normalize unit vector
		dist = distance(pointA, pointB);

		if (dist==0.0){
			printf("Error: Two atoms have the same XYZ coordinates!\n");
                	return 1;
		}

		for (i=0; i<3; i++) // capturing coordinates of the two atoms
		{
			unitvec [i] = unitvec [i]/dist;
			if (unitvec [i] >= 1) // although it is not too probable...
				unitvec [i] = 0.999999;
		}

		for (i=0; i<3; i++)
			origo [i] = 0;

		// capturing moving vector
		vec_point2line(origo, pointA, pointB, movvec);

		for (i=0; i<3; i++)
		{
			myligand->rotbonds_moving_vectors [rotb_id][i] = movvec [i];
			myligand->rotbonds_unit_vectors [rotb_id][i] = unitvec [i];
		}
	}
	return 0;
}

int get_liganddata(
                   const char*        ligfilename,
                   const char*        flexresfilename,
                         Liganddata*  myligand,
                   const double       AD4_coeff_vdW,
                   const double       AD4_coeff_hb,
                         int          nr_deriv_atypes,
                         deriv_atype* deriv_atypes,
                         int          nr_mod_atype_pairs,
                         pair_mod*    mod_atype_pairs
                  )
// The functions second parameter is a Liganddata variable whose num_of_atypes
// and atom_types fields must contain valid data.
// The function opens the file ligfilename, which is supposed to be an AutoDock4 pdbqt file,
// and fills the other fields of myligand according to the content of the file.
// If the operation was successful, the function returns 0, if not, it returns 1.
{
	FILE* fp;
	fpos_t fp_start;
	char tempstr [256];
	int atom_counter;
	int delta_count = 0;
	int branch_counter = 0;
	int atom_rot_start = 0;
	int branch_start;
	int endbranch_counter = 0;
	int branches [MAX_NUM_OF_ROTBONDS][3];
	int i,j,k;
	unsigned int atom_rotbonds_temp [MAX_NUM_OF_ATOMS][MAX_NUM_OF_ROTBONDS];
	memset(atom_rotbonds_temp,0,MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS*sizeof(unsigned int));
	memset(myligand->ignore_inter,0,MAX_NUM_OF_ATOMS*sizeof(bool));
	int current_rigid_struct_id, reserved_highest_rigid_struct_id;
	current_rigid_struct_id = 1;
	reserved_highest_rigid_struct_id = 1;

	atom_counter = 0;
	unsigned int lnr=1;
	if ( flexresfilename!=NULL ) {
		if ( strlen(flexresfilename)>0 )
			lnr++;
	}
	for (unsigned int l=0; l<lnr; l++)
	{
		if(l==0)
			fp = fopen(ligfilename, "rb"); // fp = fopen(ligfilename, "r");
		else
			fp = fopen(flexresfilename, "rb"); // fp = fopen(ligfilename, "r");
		if (fp == NULL)
		{
			if(l==0)
				printf("Error: can't open ligand data file %s!\n", ligfilename);
			else
				printf("Error: can't open flexible residue data file %s!\n", flexresfilename);
			return 1;
		}
		fgetpos (fp, &fp_start);
	
		// reading atomic coordinates, charges and atom types, and writing
		// data to myligand->atom_idxyzq
		while (fscanf(fp, "%255s", tempstr) != EOF)
		{
			if ((strcmp(tempstr, "HETATM") == 0) || (strcmp(tempstr, "ATOM") == 0))
			{
				if (atom_counter > MAX_NUM_OF_ATOMS-1)
				{
					printf("Error: ligand consists of too many atoms'\n");
					printf("Maximal allowed number of atoms is %d!\n", MAX_NUM_OF_ATOMS);
					return 1;
				}
				if ((strcmp(tempstr, "HETATM") == 0)) // seeking to the first coordinate value
					fseek(fp, 25, SEEK_CUR);
				else
					fseek(fp, 27, SEEK_CUR);
				fscanf(fp, "%lf", &(myligand->atom_idxyzq [atom_counter][1]));
				fscanf(fp, "%lf", &(myligand->atom_idxyzq [atom_counter][2]));
				fscanf(fp, "%lf", &(myligand->atom_idxyzq [atom_counter][3]));
				fscanf(fp, "%255s", tempstr); // skipping the next two fields
				fscanf(fp, "%255s", tempstr);
				fscanf(fp, "%lf", &(myligand->atom_idxyzq [atom_counter][4])); // reading charge
				fscanf(fp, "%4s", tempstr); // reading atom type
				if (set_liganddata_typeid(myligand, atom_counter, tempstr) != 0) // the function sets the type index
					return 1;
				atom_counter++;
			}
		}
		
		myligand->num_of_atoms = atom_counter;
		if(l==0){
			myligand->true_ligand_atoms = atom_counter;
			atom_counter = 0; // this looks wrong but is correct as it's increment below again (like above)
			branch_start=0;
		} else{ // example counts 4 - 3 - 6 (lig - flex res - flex res)
			unsigned int tmp = delta_count; // l=1: = 0 ; l=2: = 3
			delta_count = atom_counter - myligand->true_ligand_atoms; // l=1: 7 - 4 = 3 ; l=2: 13 - 4 = 9
			atom_counter -= delta_count - tmp; // l=1: = 7 - (3-0) = 4 ; l=2: 13 - (9-3) = 7
			atom_rot_start = atom_counter;
			branch_start=branch_counter;
		}
		
		fsetpos (fp, &fp_start);
		unsigned int flex_root = atom_rot_start; // takes care of multiple flexible residues in the same file
		
		// reading data for rotbonds and atom_rotbonds fields
		while (fscanf(fp, "%255s", tempstr) != EOF)
		{
			if ((l>0) && (strcmp(tempstr, "ROOT") == 0)){
				flex_root = atom_counter;
			}
			if ((strcmp(tempstr, "HETATM") == 0) || (strcmp(tempstr, "ATOM") == 0)) // if new atom, looking for open rotatable bonds
			{
				for (i=branch_start; i<branch_counter; i++) // for all branches found until now
					if (branches [i][2] == 1) // if it is open, the atom has to be rotated
						atom_rotbonds_temp [atom_counter][i] = 1; // modifying atom_rotbonds_temp
						/* else it is 2, so it is closed, so nothing to be done... */

				myligand->atom_rigid_structures [atom_counter] = current_rigid_struct_id; // using the id of the current rigid structure

				if (l>0)
					if (atom_counter-flex_root<2)
						myligand->ignore_inter [atom_counter] = true;

				atom_counter++;
			}

			if (strcmp(tempstr, "BRANCH") == 0) // if new branch, stroing atom indexes into branches [][]
			{
				if (branch_counter >= MAX_NUM_OF_ROTBONDS)
				{
					if(l==0)
						printf("Error: ligand includes too many rotatable bonds.\n");
					else
						printf("Error: ligand and flexible residue include too many rotatable bonds.\n");
					printf("Maximal allowed number is %d.\n", MAX_NUM_OF_ROTBONDS);
					fclose(fp);
					return 1;
				}
				fscanf(fp, "%d", &(branches [branch_counter][0]));
				fscanf(fp, "%d", &(branches [branch_counter][1]));
				branches [branch_counter][0] += atom_rot_start-1; // atom IDs start from 0 instead of 1
				branches [branch_counter][1] += atom_rot_start-1;
	
				branches [branch_counter][2] = 1; // 1 means the branch is open, atoms will be rotated
	
				branch_counter++;
	
				reserved_highest_rigid_struct_id++; // next ID is reserved
				current_rigid_struct_id = reserved_highest_rigid_struct_id; // New branch means new rigid structure, and a new id as well
			}
	
			if (strcmp(tempstr, "ENDBRANCH") == 0)
			{
				fscanf(fp, "%d", &(myligand->rotbonds [endbranch_counter][0])); // rotatable bonds have to be stored in the order
				fscanf(fp, "%d", &(myligand->rotbonds [endbranch_counter][1])); // of endbranches
				myligand->rotbonds [endbranch_counter][0] += atom_rot_start-1;
				myligand->rotbonds [endbranch_counter][1] += atom_rot_start-1;
				for (i=branch_start; i<branch_counter; i++) // the branch have to be closed
					if ((branches [i][0] == myligand->rotbonds [endbranch_counter][0]) &&
					    (branches [i][1] == myligand->rotbonds [endbranch_counter][1]))
						branches [i][2] = 2;
				endbranch_counter++;
				current_rigid_struct_id--; // probably unnecessary since there is a new branch after every endbranch...
			}
		}
		reserved_highest_rigid_struct_id++;
		current_rigid_struct_id=reserved_highest_rigid_struct_id;
		fclose(fp);
		myligand->num_of_rotbonds = branch_counter;
		if (l==0)
			myligand->true_ligand_rotbonds = branch_counter;
	}
	// Now the rotbonds field contains the rotatable bonds (that is, the corresponding two atom's indexes) in the proper order
	// (this will be the order of rotations if an atom have to be rotated around more then one rotatable bond.) However, the
	// atom_rotbonds_temp, whose column indexes correspond to rotatable bond indexes, contains data according to the order of
	// branches (that is, according to branches [][] array), instead of endbranches. Columns of atom_rotbonds_temp have to be
	// copied now to myligand->atom_rotbonds, but in the proper order.
	for (i=0; i<branch_counter; i++)
		for (j=0; j<branch_counter; j++)
			if ((myligand->rotbonds [i][0] == branches [j][0]) && (myligand->rotbonds [i][1] == branches [j][1]))
				for (k=0; k<myligand->num_of_atoms; k++)
					myligand->atom_rotbonds [k][i] = atom_rotbonds_temp [k][j]; // rearrange the columns
	if (get_bonds(myligand) == 1)
		return 1;
	get_intraE_contributors(myligand);
	if (get_VWpars(	myligand,
			AD4_coeff_vdW,
			AD4_coeff_hb,
			nr_deriv_atypes,
			deriv_atypes,
			nr_mod_atype_pairs,
			mod_atype_pairs) == 1)
		return 1;

	if (get_moving_and_unit_vectors(myligand) == 1)
                return 1;

	return 0;
}

int gen_new_pdbfile(
                    const char* oldpdb,
                    const char* newpdb,
                    const       Liganddata* myligand
                   )
// The function opens old pdb file, which is supposed to be an AutoDock4 pdbqt file, and
// copies it to newpdb file, but switches the coordinate values to the atomic coordinates
// of myligand, so newpdb file will be identical to oldpdb except the coordinate values.
// Myligand has to be the ligand which was originally read from oldpdb.
// If the operation was successful, the function returns 0, if not, it returns 1.
{
	FILE* fp_old;
	FILE* fp_new;
	char tempstr [256];
	char tempstr_short [32];
	int acnt_oldlig;
	int i,j;

	acnt_oldlig = 0;

	fp_old = fopen(oldpdb, "rb"); // fp_old = fopen(oldpdb, "r");
	if (fp_old == NULL)
	{
		printf("Error: can't open old pdb file %s!\n", oldpdb);
		return 1;
	}

	fp_new = fopen(newpdb, "w");
	if (fp_new == NULL)
	{
		printf("Error: can't create new pdb file %s!\n", newpdb);
		fclose(fp_old);
		return 1;
	}

	while (fgets(tempstr, 255, fp_old) != NULL) // reading a whole row from oldpdb
	{
		sscanf(tempstr, "%s", tempstr_short);
		if ((strcmp(tempstr_short, "HETATM") == 0) || (strcmp(tempstr_short, "ATOM") == 0)) // if the row begins with HETATM/ATOM, coordinates must be switched
		{
			if (acnt_oldlig >= myligand->num_of_atoms)
			{
				printf("Error: ligand in old pdb file includes more atoms than new one.\n");
				fclose(fp_old);
				fclose(fp_new);
				return 1;
			}
			for (i=0; i<3; i++)
			{
				sprintf(tempstr_short, "%7.3lf", myligand->atom_idxyzq [acnt_oldlig][1+i]);
				for (j=0; j<7; j++)
					tempstr [31+8*i+j] = tempstr_short [j];
			}
			acnt_oldlig++;
		}
		fprintf(fp_new, "%s", tempstr); // writing the row to newpdb
	}

	if (acnt_oldlig != myligand->num_of_atoms)
	{
		printf("%d %d \n", acnt_oldlig, myligand->num_of_atoms);
		printf("Warning: New ligand consists of more atoms than original one (i.e. w/ flexres).\n");
		printf("         Not all the atoms have been written to file!\n");
	}

	fclose(fp_old);
	fclose(fp_new);

	return 0;
}

void get_movvec_to_origo(
                         const Liganddata* myligand,
                               double      movvec []
                        )
// The function returns the moving vector in the second parameter which moves the ligand
// (that is, its geometrical center point) given by the first parameter to the origo).
{
	double tmp_x, tmp_y, tmp_z;
	int i;

	tmp_x = 0;
	tmp_y = 0;
	tmp_z = 0;

	for (i=0; i < myligand->true_ligand_atoms; i++) // only for the ligand
	{
		tmp_x += myligand->atom_idxyzq [i][1];
		tmp_y += myligand->atom_idxyzq [i][2];
		tmp_z += myligand->atom_idxyzq [i][3];
	}

	movvec [0] = -1*tmp_x/myligand->true_ligand_atoms;
	movvec [1] = -1*tmp_y/myligand->true_ligand_atoms;
	movvec [2] = -1*tmp_z/myligand->true_ligand_atoms;
}

void move_ligand(
                       Liganddata*  myligand,
                 const double       movvec []
                )
{
	move_ligand(myligand, movvec, NULL);
}

void move_ligand(
                       Liganddata*  myligand,
                 const double       movvec [],
                 const double       flexmovvec []
                )
// The function moves the ligand given by the first parameter according to
// the vector given by the second one.
{
	int i;

	for (i=0; i < myligand->true_ligand_atoms; i++) // only for the ligand
	{
		myligand->atom_idxyzq [i][1] += movvec [0];
		myligand->atom_idxyzq [i][2] += movvec [1];
		myligand->atom_idxyzq [i][3] += movvec [2];
	}
	
	if (flexmovvec)
		for (i=myligand->true_ligand_atoms; i < myligand->num_of_atoms; i++) // flexible residue
		{
			myligand->atom_idxyzq [i][1] += flexmovvec [0];
			myligand->atom_idxyzq [i][2] += flexmovvec [1];
			myligand->atom_idxyzq [i][3] += flexmovvec [2];
		}
}

void scale_ligand(
                        Liganddata*  myligand,
                  const double       scale_factor
                 )
// The function scales the ligand given by the first parameter according to the factor
// given by the second (that is, all the ligand atom coordinates will be multiplied by
// scale_factor).
{
	int i,j;

	for (i=0; i < myligand->num_of_atoms; i++){
		for (j=1; j<4; j++)
			myligand->atom_idxyzq [i][j] = myligand->atom_idxyzq [i][j]*scale_factor;
//		if(i>=myligand->true_ligand_atoms)
//			printf("%i: (%f, %f, %f)\n",i-myligand->true_ligand_atoms+1,myligand->atom_idxyzq [i][1],myligand->atom_idxyzq [i][2],myligand->atom_idxyzq [i][3]);
	}
}

double calc_rmsd(
                 const Liganddata* myligand_ref,
                 const Liganddata* myligand,
                 const bool        handle_symmetry
                )
// The function calculates the RMSD value (root mean square deviation of the
// atomic distances for two conformations of the same ligand) and returns it.
// If the handle_symmetry parameter is 0, symmetry is not handled, and the
// distances are calculated between atoms with the same atom id. If it is not
// 0, one atom from myligand will be compared to the closest atom with the same
// type from myligand_ref and this will be accumulated during rmsd calculation
// (which is a silly method but this is applied in AutoDock, too).
// The two positions must be given by the myligand and myligand_ref parameters.
{
	int i,j;
	double sumdist2;
	double mindist2;

	if (myligand_ref->true_ligand_atoms != myligand->true_ligand_atoms)
	{
		printf("Warning: RMSD can't be calculated, atom number mismatch %d (ref) vs. %d!\n",myligand_ref->true_ligand_atoms,myligand->true_ligand_atoms);
		return 100000; // returning unreasonable value
	}

	sumdist2 = 0;

	if (!handle_symmetry)
	{
		for (i=0; i<myligand->true_ligand_atoms; i++)
		{
			sumdist2 += pow(distance(&(myligand->atom_idxyzq [i][1]), &(myligand_ref->atom_idxyzq [i][1])), 2);
		}
	}
	else // handling symmetry with the silly AutoDock method
	{
		for (i=0; i<myligand->true_ligand_atoms; i++)
		{
			mindist2 = 100000; // initial value should be high enough so that it is ensured that lower distances will be found
			for (j=0; j<myligand_ref->num_of_atoms; j++) // looking for the closest atom with same type from the reference
			{
				if (myligand->atom_idxyzq [i][0] == myligand_ref->atom_idxyzq [j][0])
					if (pow(distance(&(myligand->atom_idxyzq [i][1]), &(myligand_ref->atom_idxyzq [j][1])), 2) < mindist2)
						mindist2 = pow(distance(&(myligand->atom_idxyzq [i][1]), &(myligand_ref->atom_idxyzq [j][1])), 2);
			}
			sumdist2 += mindist2;
		}
	}

	return (sqrt(sumdist2/myligand->true_ligand_atoms));
}

double calc_ddd_Mehler_Solmajer(double distance)
// The function returns the value of the distance-dependend dielectric function.
// (Whole function copied from AutoDock...)
{

    double epsilon = 1.0L;
    double lambda = 0.003627L;
    double epsilon0 = 78.4L;
    double A = -8.5525L;
    double B;
    double rk= 7.7839L;
    double lambda_B;

    B = epsilon0 - A;
    lambda_B = -lambda * B;

    epsilon = A + B / (1.0L + rk*exp(lambda_B * distance));

    return epsilon;
}

bool is_H_acceptor(const char* atype)
{
	return ((strcmp(atype, "NA") == 0) ||
	        (strcmp(atype, "NS") == 0) ||
	        (strcmp(atype, "OA") == 0) ||
	        (strcmp(atype, "OS") == 0) ||
	        (strcmp(atype, "SA") == 0) ); // NA NS OA OS or SA are all acceptors
}

bool is_H_bond(
               const char* atype1,
               const char* atype2
              )
// Returns True if a H-bond can exist between the atoms with atom code atype1 and atype2,
// otherwise it returns False.
{
	if ( // H-bond
	    (((strcmp(atype1, "HD") == 0) || (strcmp(atype1, "HS") == 0)) && // HD or HS
	              is_H_acceptor(atype2))
	    ||
	    (((strcmp(atype2, "HD") == 0) || (strcmp(atype2, "HS") == 0)) && // HD or HS
	              is_H_acceptor(atype1))
	   )
		return true;
	else
		return false;
}

void print_ref_lig_energies_f(
                                    Liganddata myligand,
                              const float      smooth,
                                    Gridinfo   mygrid,
                              const float*     fgrids,
                              const float      scaled_AD4_coeff_elec,
                              const float      elec_min_distance,
                              const float      AD4_coeff_desolv,
                              const float      qasp,
                                    int        nr_mod_atype_pairs,
                                    pair_mod*  mod_atype_pairs
                            )
// The function calculates the energies of the ligand given in the first parameter,
// and prints them to the screen.
{
	double temp_vec [3];
	float tmp;
	int i;

	IntraTables tables(&myligand, scaled_AD4_coeff_elec, AD4_coeff_desolv, qasp);
	printf("Intramolecular energy of reference ligand: %lf\n",
	       calc_intraE_f(&myligand, 8, smooth, 0, elec_min_distance, tables, 0, tmp, nr_mod_atype_pairs, mod_atype_pairs));

	for (i=0; i<3; i++)
		temp_vec [i] = -1*mygrid.origo_real_xyz [i];

	move_ligand(&myligand, temp_vec);
	scale_ligand(&myligand, (double) 1.0/mygrid.spacing);

	printf("Intermolecular energy of reference ligand: %lf\n",
	       calc_interE_f(&mygrid, &myligand, fgrids, 0, 0, tmp));
}

//////////////////////////////////
//float functions

void calc_distdep_tables_f(
                                 float r_6_table [],
                                 float r_10_table [],
                                 float r_12_table [],
                                 float r_epsr_table [],
                                 float desolv_table [],
                           const float scaled_AD4_coeff_elec,
                           const float AD4_coeff_desolv
                          )
// The function fills the input arrays with the following functions:
// 1/r^6, 1/r^10, 1/r^12, W_el/(r*eps(r)) and W_des*exp(-r^2/(2sigma^2))
// for distances 0.01, 0.02, ..., 20.48 A
{
	int i;
	float dist;
	const float sigma = 3.6;

	dist = 0;
	for (i=0; i<2048; i++)
	{
		dist += 0.01;
		r_6_table [i] = 1/powf(dist,6);
		r_10_table [i] = 1/powf(dist,10);
		r_12_table [i] = 1/powf(dist,12);
		r_epsr_table [i] = (float) scaled_AD4_coeff_elec/(dist*calc_ddd_Mehler_Solmajer(dist));
		desolv_table [i] = AD4_coeff_desolv*expf(-1*dist*dist/(2*sigma*sigma));
	}
}

void calc_q_tables_f(
                     const Liganddata* myligand,
                           float       qasp,
                           float       q1q2[][MAX_NUM_OF_ATOMS],
                           float       qasp_mul_absq []
                    )
// The function calculates q1*q2 and qasp*abs(q) values
// based on the myligand parameter.
{
	int i, j;

	for (i=0; i < myligand->num_of_atoms; i++)
		for (j=0; j < myligand->num_of_atoms; j++)
			q1q2 [i][j] = (float) myligand->atom_idxyzq [i][4] * myligand->atom_idxyzq [j][4];

	for (i=0; i < myligand->num_of_atoms; i++)
		qasp_mul_absq [i] = qasp*fabs(myligand->atom_idxyzq [i][4]);

}

void change_conform_f(
                            Liganddata* myligand,
                      const Gridinfo*   mygrid,
                      const float       genotype_f [],
                            int         debug
                     )
// The function changes the conformation of myligand according to
// the floating point genotype from GPU
{
	double genotype [ACTUAL_GENOTYPE_LENGTH];
	for (unsigned int i=0; i<ACTUAL_GENOTYPE_LENGTH; i++)
		genotype [i] = genotype_f [i];
	change_conform(myligand,mygrid,genotype,NULL,debug);
}

void change_conform(
                          Liganddata* myligand,
                    const Gridinfo*   mygrid,
                    const double      genotype [],
                    const double      axisangle[4],
                          int         debug
                   )
// The function changes the conformation of myligand according to
// the genotype and (optionally) the general rotation in axisangle
{
	double genrot_movvec [3] = {0, 0, 0};
	double genrot_unitvec [3];
	double genrot_angle;
	double movvec_to_origo [3];
	int atom_id, rotbond_id;

	if(!axisangle){
		double phi = (genotype [3])/180*PI;
		double theta = (genotype [4])/180*PI;
		
		genrot_unitvec [0] = sin(theta)*cos(phi);
		genrot_unitvec [1] = sin(theta)*sin(phi);
		genrot_unitvec [2] = cos(theta);
		genrot_angle = genotype[5];
	} else{
		genrot_unitvec [0] = axisangle[0];
		genrot_unitvec [1] = axisangle[1];
		genrot_unitvec [2] = axisangle[2];
		genrot_angle = axisangle[3];
	}

	get_movvec_to_origo(myligand, movvec_to_origo); // moving ligand to origo
	move_ligand(myligand, movvec_to_origo);

	for (atom_id=0; atom_id < myligand->num_of_atoms; atom_id++) // for each atom of the ligand
	{
		if (debug == 1)
			printf("\n\n\nROTATING atom %d ", atom_id);

		if (myligand->num_of_rotbonds != 0) // if the ligand has rotatable bonds
		{
			for (rotbond_id=0; rotbond_id < myligand->num_of_rotbonds; rotbond_id++) // for each rotatable bond
				if (myligand->atom_rotbonds[atom_id][rotbond_id] != 0) // if the atom has to be rotated around this bond
				{
					if (debug == 1)
						printf("around rotatable bond %d\n", rotbond_id);

					rotate(&(myligand->atom_idxyzq[atom_id][1]),
					       myligand->rotbonds_moving_vectors[rotbond_id],
					       myligand->rotbonds_unit_vectors[rotbond_id],
					       &(genotype [6+rotbond_id]), /*debug*/0); // rotating
				}
		}

		if (atom_id<myligand->true_ligand_atoms)
		{
			rotate(&(myligand->atom_idxyzq[atom_id][1]),
			       genrot_movvec,
			       genrot_unitvec,
			       &genrot_angle, debug); // general rotation
		}
	}

	move_ligand(myligand, genotype);

	if (debug == 1)
		for (atom_id=0; atom_id < myligand->num_of_atoms; atom_id++)
			printf("Moved point (final values) (x,y,z): %lf, %lf, %lf\n", myligand->atom_idxyzq [atom_id][1], myligand->atom_idxyzq [atom_id][2], myligand->atom_idxyzq [atom_id][3]);
}

std::vector<AnalysisData> analyze_ligand_receptor(
                                                  const Gridinfo*     mygrid,
                                                  const Liganddata*   myligand,
                                                  const ReceptorAtom* receptor_atoms,
                                                  const unsigned int* receptor_map,
                                                  const unsigned int* receptor_map_list,
                                                        float         outofgrid_tolerance,
                                                        int           debug,
                                                        float         H_cutoff,
                                                        float         V_cutoff
                                                 )
// The function performs a simple distance based ligand-receptor analysis
{
	int atom_cnt;
	float x, y, z;
	int atomtypeid;
	std::vector<AnalysisData> result;
	H_cutoff /= mygrid->spacing;
	H_cutoff *= H_cutoff;
	V_cutoff /= mygrid->spacing;
	V_cutoff *= V_cutoff;

	unsigned int g1 = mygrid->size_xyz[0];
	unsigned int g2 = g1*mygrid->size_xyz[1];

	const unsigned int* receptor_list;
	AnalysisData datum;

	for (atom_cnt=0; atom_cnt<myligand->true_ligand_atoms; atom_cnt++) // for each atom
	{
		if (myligand->ignore_inter[atom_cnt])
			continue;
		atomtypeid = myligand->base_type_idx[(int)myligand->atom_idxyzq [atom_cnt][0]];
		x = myligand->atom_idxyzq [atom_cnt][1];
		y = myligand->atom_idxyzq [atom_cnt][2];
		z = myligand->atom_idxyzq [atom_cnt][3];

		if ((x < 0) || (x >= mygrid->size_xyz [0]-1) ||
		    (y < 0) || (y >= mygrid->size_xyz [1]-1) ||
		    (z < 0) || (z >= mygrid->size_xyz [2]-1)) // if the atom is outside of the grid
		{
			if (debug == 1)
			{
				printf("\n\nPartial results for atom with id %d:\n", atom_cnt);
				printf("Atom out of grid: ");
				printf("x= %lf, y = %lf, z = %lf\n", x, y, z);
			}

			if (outofgrid_tolerance != 0) // if tolerance is set, try to place atom back into the grid
			{
				if (x < 0)
					x += outofgrid_tolerance;
				if (y < 0)
					y += outofgrid_tolerance;
				if (z < 0)
					z += outofgrid_tolerance;
				if (x >= mygrid->size_xyz [0]-1)
					x -= outofgrid_tolerance;
				if (y >= mygrid->size_xyz [1]-1)
					y -= outofgrid_tolerance;
				if (z >= mygrid->size_xyz [2]-1)
					z -= outofgrid_tolerance;
			}

			if ((x < 0) || (x >= mygrid->size_xyz [0]-1) ||
			    (y < 0) || (y >= mygrid->size_xyz [1]-1) ||
			    (z < 0) || (z >= mygrid->size_xyz [2]-1)) // check again if the atom is outside of the grid
			{
				continue;
			}

			if (debug == 1)
			{
				printf("\n\nAtom was placed back into the grid according to the tolerance value %f:\n", outofgrid_tolerance);
				printf("x= %lf, y = %lf, z = %lf\n", x, y, z);
			}
		}

		receptor_list = receptor_map_list + receptor_map[(int)floor(x)  + ((int)floor(y))*g1  + ((int)floor(z))*g2];
		for(unsigned int rid=1; rid<=receptor_list[0]; rid++)
		{
			const ReceptorAtom* curr = &receptor_atoms[receptor_list[rid]];
			double dist2 = (curr->x-x)*(curr->x-x)+(curr->y-y)*(curr->y-y)+(curr->z-z)*(curr->z-z);
			if((myligand->acceptor[atom_cnt] && curr->donor) ||
			   (myligand->donor[atom_cnt] && curr->acceptor)){
				if(dist2 <= H_cutoff){
					datum.type     = 1; // 0 .. reactive, 1 .. hydrogen bond, 2 .. vdW
					datum.lig_id   = atom_cnt+1;
					datum.lig_name = myligand->atom_names[atom_cnt];
					datum.rec_id   = curr->id;
					datum.rec_name = curr->name;
					datum.residue  = curr->res_name;
					datum.res_id   = curr->res_id;
					datum.chain    = curr->chain_id;
					result.push_back(datum);
				}
			} else{ // vdW
				if((myligand->base_atom_types[atomtypeid][0]!='H') && (curr->atom_type[0]!='H') && // exclude Hydrogens,
				   !myligand->acceptor[atom_cnt] && !myligand->donor[atom_cnt] &&                  // non-H-bond capable atoms on ligand
				   !curr->acceptor && !curr->donor){                                               // ... and receptor
					if(dist2 <= V_cutoff){
						datum.type     = 2; // 0 .. reactive, 1 .. hydrogen bond, 2 .. vdW
						datum.lig_id   = atom_cnt+1;
						datum.lig_name = myligand->atom_names[atom_cnt];
						datum.rec_id   = curr->id;
						datum.rec_name = curr->name;
						datum.residue  = curr->res_name;
						datum.res_id   = curr->res_id;
						datum.chain    = curr->chain_id;
						result.push_back(datum);
					}
				}
			}
		}
	}
	return result;
}


float calc_interE_f(
                    const Gridinfo*   mygrid,
                    const Liganddata* myligand,
                    const float*      fgrids,
                          float       outofgrid_tolerance,
                          int         debug,
                          float&      intraflexE
                   )
// The function calculates the intermolecular energy of a ligand (given by myligand parameter),
// and a receptor (represented as a grid). The grid point values must be stored at the location
// which starts at fgrids, the memory content can be generated with get_gridvalues funciton.
// The mygrid parameter must be the corresponding grid informtaion. If an atom is outside the
// grid, the coordinates will be changed with the value of outofgrid_tolerance, if it remains
// outside, a very high value will be added to the current energy as a penality. If the fifth
// parameter is one, debug messages will be printed to the screen during calculation.
{
	float interE;
	int atom_cnt;
	float x, y, z;
	int atomtypeid;
	int x_low, x_high, y_low, y_high, z_low, z_high;
	float q, x_frac, y_frac, z_frac;
	float cube [2][2][2];
	float weights [2][2][2];
	float dx, dy, dz;

	float val;
	interE = 0;
	intraflexE = 0;

	for (atom_cnt=myligand->num_of_atoms-1; atom_cnt>=0; atom_cnt--) // for each atom
	{
		val = 0.0;
		if (myligand->ignore_inter[atom_cnt])
			continue;
		atomtypeid = myligand->base_type_idx[(int)myligand->atom_idxyzq [atom_cnt][0]];
		x = myligand->atom_idxyzq [atom_cnt][1];
		y = myligand->atom_idxyzq [atom_cnt][2];
		z = myligand->atom_idxyzq [atom_cnt][3];
		q = myligand->atom_idxyzq [atom_cnt][4];

		if ((x < 0) || (x >= mygrid->size_xyz [0]-1) || (y < 0) || (y >= mygrid->size_xyz [1]-1) ||
			(z < 0) || (z >= mygrid->size_xyz [2]-1)) // if the atom is outside of the grid
		{
			if (debug == 1)
			{
				printf("\n\nPartial results for atom with id %d:\n", atom_cnt);
				printf("Atom out of grid: ");
				printf("x= %lf, y = %lf, z = %lf\n", x, y, z);
			}

			if (outofgrid_tolerance != 0) // if tolerance is set, try to place atom back into the grid
			{
				if (x < 0)
					x += outofgrid_tolerance;
				if (y < 0)
					y += outofgrid_tolerance;
				if (z < 0)
					z += outofgrid_tolerance;
				if (x >= mygrid->size_xyz [0]-1)
					x -= outofgrid_tolerance;
				if (y >= mygrid->size_xyz [1]-1)
					y -= outofgrid_tolerance;
				if (z >= mygrid->size_xyz [2]-1)
					z -= outofgrid_tolerance;
			}

			if ((x < 0) || (x >= mygrid->size_xyz [0]-1) ||
			    (y < 0) || (y >= mygrid->size_xyz [1]-1) ||
			    (z < 0) || (z >= mygrid->size_xyz [2]-1)) // check again if the atom is outside of the grid
			{
				//interE = HIGHEST_ENERGY; // return maximal value
				//return interE;
				val += 16777216; // penalty is 2^24 for each atom outside the grid
				if (atom_cnt < myligand->true_ligand_atoms)
					interE += val;
				else
					intraflexE += val;
				continue;
			}

			if (debug == 1)
			{
				printf("\n\nAtom was placed back into the grid according to the tolerance value %f:\n", outofgrid_tolerance);
				printf("x= %lf, y = %lf, z = %lf\n", x, y, z);
			}
		}

		x_low = (int) floor(x);
		y_low = (int) floor(y);
		z_low = (int) floor(z);
		x_high = (int) ceil(x);
		y_high = (int) ceil(y);
		z_high = (int) ceil(z);
		x_frac = x - x_low;
		y_frac = y - y_low;
		z_frac = z - z_low;
		dx = x_frac;
		dy = y_frac;
		dz = z_frac;

		get_trilininterpol_weights_f(weights, &dx, &dy, &dz);

		if (debug == 1)
		{
			printf("\n\nPartial results for atom with id %d:\n", atom_cnt);
			printf("x_low = %d, x_high = %d, x_frac = %lf\n", x_low, x_high, x_frac);
			printf("y_low = %d, y_high = %d, y_frac = %lf\n", y_low, y_high, y_frac);
			printf("z_low = %d, z_high = %d, z_frac = %lf\n\n", z_low, z_high, z_frac);
			printf("coeff(0,0,0) = %lf\n", weights [0][0][0]);
			printf("coeff(1,0,0) = %lf\n", weights [1][0][0]);
			printf("coeff(0,1,0) = %lf\n", weights [0][1][0]);
			printf("coeff(1,1,0) = %lf\n", weights [1][1][0]);
			printf("coeff(0,0,1) = %lf\n", weights [0][0][1]);
			printf("coeff(1,0,1) = %lf\n", weights [1][0][1]);
			printf("coeff(0,1,1) = %lf\n", weights [0][1][1]);
			printf("coeff(1,1,1) = %lf\n", weights [1][1][1]);
		}

		// energy contribution of the current grid type

		cube [0][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_low);
		cube [1][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_high);
		cube [0][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_low);
		cube [1][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_high);
		cube [0][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_low);
		cube [1][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_high);
		cube [0][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_low);
		cube [1][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_high);

		if (debug == 1)
		{
			printf("Interpolation of van der Waals map:\n");
			printf("cube(0,0,0) = %lf\n", cube [0][0][0]);
			printf("cube(1,0,0) = %lf\n", cube [1][0][0]);
			printf("cube(0,1,0) = %lf\n", cube [0][1][0]);
			printf("cube(1,1,0) = %lf\n", cube [1][1][0]);
			printf("cube(0,0,1) = %lf\n", cube [0][0][1]);
			printf("cube(1,0,1) = %lf\n", cube [1][0][1]);
			printf("cube(0,1,1) = %lf\n", cube [0][1][1]);
			printf("cube(1,1,1) = %lf\n", cube [1][1][1]);
		}


		val += trilin_interpol(cube, weights);

		if (debug == 1)
			printf("interpoated value = %lf\n\n", trilin_interpol(cube, weights));

		// energy contribution of the electrostatic grid

		atomtypeid = mygrid->num_of_atypes;

		cube [0][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_low);
		cube [1][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_high);
		cube [0][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_low);
		cube [1][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_high);
		cube [0][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_low);
		cube [1][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_high);
		cube [0][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_low);
		cube [1][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_high);

		if (debug == 1)
		{
			printf("Interpolation of electrostatic map:\n");
			printf("cube(0,0,0) = %lf\n", cube [0][0][0]);
			printf("cube(1,0,0) = %lf\n", cube [1][0][0]);
			printf("cube(0,1,0) = %lf\n", cube [0][1][0]);
			printf("cube(1,1,0) = %lf\n", cube [1][1][0]);
			printf("cube(0,0,1) = %lf\n", cube [0][0][1]);
			printf("cube(1,0,1) = %lf\n", cube [1][0][1]);
			printf("cube(0,1,1) = %lf\n", cube [0][1][1]);
			printf("cube(1,1,1) = %lf\n", cube [1][1][1]);
		}


		val += q * trilin_interpol(cube, weights);

		if (debug == 1)
			printf("interpoated value = %lf, multiplied by q = %lf\n\n", trilin_interpol(cube, weights), q*trilin_interpol(cube, weights));

		// energy contribution of the desolvation grid

		atomtypeid = mygrid->num_of_atypes+1;

		cube [0][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_low);
		cube [1][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_high);
		cube [0][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_low);
		cube [1][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_high);
		cube [0][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_low);
		cube [1][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_high);
		cube [0][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_low);
		cube [1][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_high);

		if (debug == 1)
		{
			printf("Interpolation of desolvation map:\n");
			printf("cube(0,0,0) = %lf\n", cube [0][0][0]);
			printf("cube(1,0,0) = %lf\n", cube [1][0][0]);
			printf("cube(0,1,0) = %lf\n", cube [0][1][0]);
			printf("cube(1,1,0) = %lf\n", cube [1][1][0]);
			printf("cube(0,0,1) = %lf\n", cube [0][0][1]);
			printf("cube(1,0,1) = %lf\n", cube [1][0][1]);
			printf("cube(0,1,1) = %lf\n", cube [0][1][1]);
			printf("cube(1,1,1) = %lf\n", cube [1][1][1]);
		}

		val += fabs(q) * trilin_interpol(cube, weights);

		if (atom_cnt < myligand->true_ligand_atoms)
			interE += val;
		else
			intraflexE += val;

		if (debug == 1)
			printf("interpoated value = %lf, multiplied by abs(q) = %lf\n\n", trilin_interpol(cube, weights), fabs(q) * trilin_interpol(cube, weights));

		if (debug == 1)
			printf("Current value of intermolecular energy = %lf\n\n\n", interE);
	}
	return interE;
}

void calc_interE_peratom_f(
                           const Gridinfo*   mygrid,
                           const Liganddata* myligand,
                           const float*      fgrids,
                                 float       outofgrid_tolerance,
                                 float*      elecE,
                                 float       peratom_vdw [MAX_NUM_OF_ATOMS],
                                 float       peratom_elec [MAX_NUM_OF_ATOMS],
                                 int         debug
                          )
{
	//float interE;
	int atom_cnt;
	float x, y, z;
	int atomtypeid;
	int x_low, x_high, y_low, y_high, z_low, z_high;
	float q, x_frac, y_frac, z_frac;
	float cube [2][2][2];
	float weights [2][2][2];
	float dx, dy, dz;

	//interE = 0;
	*elecE = 0;

	for (atom_cnt=myligand->num_of_atoms-1; atom_cnt>=0; atom_cnt--)		//for each atom
	{
		if (myligand->ignore_inter[atom_cnt])
			continue;
		atomtypeid = myligand->base_type_idx[(int)myligand->atom_idxyzq [atom_cnt][0]];
		x = myligand->atom_idxyzq [atom_cnt][1];
		y = myligand->atom_idxyzq [atom_cnt][2];
		z = myligand->atom_idxyzq [atom_cnt][3];
		q = myligand->atom_idxyzq [atom_cnt][4];

		if ((x < 0) || (x >= mygrid->size_xyz [0]-1) ||
		    (y < 0) || (y >= mygrid->size_xyz [1]-1) ||
		    (z < 0) || (z >= mygrid->size_xyz [2]-1)) // if the atom is outside of the grid
		{
			if (debug == 1)
			{
				printf("\n\nPartial results for atom with id %d:\n", atom_cnt);
				printf("Atom out of grid: ");
				printf("x= %lf, y = %lf, z = %lf\n", x, y, z);
			}

			if (outofgrid_tolerance != 0) // if tolerance is set, try to place atom back into the grid
			{
				if (x < 0)
					x += outofgrid_tolerance;
				if (y < 0)
					y += outofgrid_tolerance;
				if (z < 0)
					z += outofgrid_tolerance;
				if (x >= mygrid->size_xyz [0]-1)
					x -= outofgrid_tolerance;
				if (y >= mygrid->size_xyz [1]-1)
					y -= outofgrid_tolerance;
				if (z >= mygrid->size_xyz [2]-1)
					z -= outofgrid_tolerance;
			}

			if ((x < 0) || (x >= mygrid->size_xyz [0]-1) || (y < 0) || (y >= mygrid->size_xyz [1]-1) ||
						(z < 0) || (z >= mygrid->size_xyz [2]-1)) // check again if the atom is outside of the grid
			{
				//interE = HIGHEST_ENERGY; // return maximal value
				//return interE;
				//interE += 16777216; // penalty is 2^24 for each atom outside the grid
				peratom_vdw[atom_cnt] = 100000;
				peratom_elec[atom_cnt] = 100000;
				continue;
			}

			if (debug == 1)
			{
				printf("\n\nAtom was placed back into the grid according to the tolerance value %f:\n", outofgrid_tolerance);
				printf("x= %lf, y = %lf, z = %lf\n", x, y, z);
			}
		}

		x_low = (int) floor(x);
		y_low = (int) floor(y);
		z_low = (int) floor(z);
		x_high = (int) ceil(x);
		y_high = (int) ceil(y);
		z_high = (int) ceil(z);
		x_frac = x - x_low;
		y_frac = y - y_low;
		z_frac = z - z_low;
		dx = x_frac;
		dy = y_frac;
		dz = z_frac;

		get_trilininterpol_weights_f(weights, &dx, &dy, &dz);

		if (debug == 1)
		{
			printf("\n\nPartial results for atom with id %d:\n", atom_cnt);
			printf("x_low = %d, x_high = %d, x_frac = %lf\n", x_low, x_high, x_frac);
			printf("y_low = %d, y_high = %d, y_frac = %lf\n", y_low, y_high, y_frac);
			printf("z_low = %d, z_high = %d, z_frac = %lf\n\n", z_low, z_high, z_frac);
			printf("coeff(0,0,0) = %lf\n", weights [0][0][0]);
			printf("coeff(1,0,0) = %lf\n", weights [1][0][0]);
			printf("coeff(0,1,0) = %lf\n", weights [0][1][0]);
			printf("coeff(1,1,0) = %lf\n", weights [1][1][0]);
			printf("coeff(0,0,1) = %lf\n", weights [0][0][1]);
			printf("coeff(1,0,1) = %lf\n", weights [1][0][1]);
			printf("coeff(0,1,1) = %lf\n", weights [0][1][1]);
			printf("coeff(1,1,1) = %lf\n", weights [1][1][1]);
		}

		// energy contribution of the current grid type

		cube [0][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_low);
		cube [1][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_high);
		cube [0][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_low);
		cube [1][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_high);
		cube [0][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_low);
		cube [1][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_high);
		cube [0][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_low);
		cube [1][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_high);

		if (debug == 1)
		{
			printf("Interpolation of van der Waals map:\n");
			printf("cube(0,0,0) = %lf\n", cube [0][0][0]);
			printf("cube(1,0,0) = %lf\n", cube [1][0][0]);
			printf("cube(0,1,0) = %lf\n", cube [0][1][0]);
			printf("cube(1,1,0) = %lf\n", cube [1][1][0]);
			printf("cube(0,0,1) = %lf\n", cube [0][0][1]);
			printf("cube(1,0,1) = %lf\n", cube [1][0][1]);
			printf("cube(0,1,1) = %lf\n", cube [0][1][1]);
			printf("cube(1,1,1) = %lf\n", cube [1][1][1]);
		}


		//interE += trilin_interpol(cube, weights);
		peratom_vdw[atom_cnt] = trilin_interpol(cube, weights);

		if (debug == 1)
			printf("interpolated value = %lf\n\n", trilin_interpol(cube, weights));

		// energy contribution of the electrostatic grid

		atomtypeid = mygrid->num_of_atypes;

		cube [0][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_low);
		cube [1][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_high);
		cube [0][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_low);
		cube [1][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_high);
		cube [0][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_low);
		cube [1][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_high);
		cube [0][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_low);
		cube [1][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_high);

		if (debug == 1)
		{
			printf("Interpolation of electrostatic map:\n");
			printf("cube(0,0,0) = %lf\n", cube [0][0][0]);
			printf("cube(1,0,0) = %lf\n", cube [1][0][0]);
			printf("cube(0,1,0) = %lf\n", cube [0][1][0]);
			printf("cube(1,1,0) = %lf\n", cube [1][1][0]);
			printf("cube(0,0,1) = %lf\n", cube [0][0][1]);
			printf("cube(1,0,1) = %lf\n", cube [1][0][1]);
			printf("cube(0,1,1) = %lf\n", cube [0][1][1]);
			printf("cube(1,1,1) = %lf\n", cube [1][1][1]);
		}


		//interE += q * trilin_interpol(cube, weights);
		peratom_elec[atom_cnt] = q * trilin_interpol(cube, weights);
		*elecE += q * trilin_interpol(cube, weights);

		if (debug == 1)
			printf("interpolated value = %lf, multiplied by q = %lf\n\n", trilin_interpol(cube, weights), q*trilin_interpol(cube, weights));

#ifdef AD4_desolv_peratom_vdW
		// energy contribution of the desolvation grid
		atomtypeid = mygrid->num_of_atypes+1;

		cube [0][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_low);
		cube [1][0][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_low, x_high);
		cube [0][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_low);
		cube [1][1][0] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_low, y_high, x_high);
		cube [0][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_low);
		cube [1][0][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_low, x_high);
		cube [0][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_low);
		cube [1][1][1] = getvalue_4Darr(fgrids, *mygrid, atomtypeid, z_high, y_high, x_high);

		if (debug == 1)
		{
			printf("Interpolation of desolvation map:\n");
			printf("cube(0,0,0) = %lf\n", cube [0][0][0]);
			printf("cube(1,0,0) = %lf\n", cube [1][0][0]);
			printf("cube(0,1,0) = %lf\n", cube [0][1][0]);
			printf("cube(1,1,0) = %lf\n", cube [1][1][0]);
			printf("cube(0,0,1) = %lf\n", cube [0][0][1]);
			printf("cube(1,0,1) = %lf\n", cube [1][0][1]);
			printf("cube(0,1,1) = %lf\n", cube [0][1][1]);
			printf("cube(1,1,1) = %lf\n", cube [1][1][1]);
		}

		peratom_vdw[atom_cnt] += fabs(q) * trilin_interpol(cube, weights);

		if (debug == 1)
			printf("interpolated value = %lf, multiplied by abs(q) = %lf\n\n", trilin_interpol(cube, weights), fabs(q) * trilin_interpol(cube, weights));
#endif
	}
}

// Corrected host "calc_intraE_f" function after smoothing was added
float calc_intraE_f(
                    const Liganddata*               myligand,
                          float                     dcutoff,
                          float                     smooth,
                          bool                      ignore_desolv,
                    const float                     elec_min_distance,
                          IntraTables&              tables,
                          int                       debug,
                          float&                    interflexE,
                          int                       nr_mod_atype_pairs,
                          pair_mod*                 mod_atype_pairs,
                          std::vector<AnalysisData> *analysis,
                    const ReceptorAtom*             flexres_atoms,
                          float                     R_cutoff,
                          float                     H_cutoff,
                          float                     V_cutoff
                   )
// The function calculates the intramolecular energy of the ligand given by the first parameter,
// and returns it as a double. The second parameter is the distance cutoff, if the third isn't 0,
// desolvation energy won't be included by the energy value, the fourth indicates if messages
// about partial results are required (if debug=1)
{
	int atom_id1, atom_id2, atom_cnt;
	int type_id1, type_id2;
	float dist;
	int distance_id;
	int smoothed_distance_id;
	float vdW1 = 0.0f;
	float vdW2 = 0.0f;
	float s1, s2, v1, v2;

	float vW, el, desolv;
	bool analyze = (analysis!=NULL);
	bool a_flex, b_flex;
	int atomtypeid;
	bool flex_reactive;
	AnalysisData datum;

	vW = 0.0f;
	el = 0.0f;
	desolv = 0.0f;
	interflexE = 0.0f;
	
	if (debug == 1)
		printf("\n\n\nINTRAMOLECULAR ENERGY CALCULATION\n\n");

	for (atom_id1=0; atom_id1<myligand->num_of_atoms-1; atom_id1++) // for each atom pair
	{
		a_flex = (atom_id1>=myligand->true_ligand_atoms);
		for (atom_id2=atom_id1+1; atom_id2<myligand->num_of_atoms; atom_id2++)
		{
			b_flex = (atom_id2>=myligand->true_ligand_atoms);
			if (myligand->intraE_contributors [atom_id1][atom_id2] == 1) // if they have to be included in intramolecular energy calculation
			{                                                            // the energy contribution has to be calculated
				dist = distance(&(myligand->atom_idxyzq [atom_id1][1]), &(myligand->atom_idxyzq [atom_id2][1]));

				if (debug == 1)
				{
					printf("\n\nCalculating energy contribution of atoms %d and %d\n", atom_id1+1, atom_id2+1);
					printf("Distance: %lf\n", dist);
				}

				// Adding smoothing

				// Getting type ids
				type_id1 = myligand->atom_idxyzq [atom_id1][0];
				type_id2 = myligand->atom_idxyzq [atom_id2][0];

				unsigned int atom1_type_vdw_hb = myligand->atom_types_reqm [type_id1];
				unsigned int atom2_type_vdw_hb = myligand->atom_types_reqm [type_id2];

				// Getting optimum pair distance (opt_distance) from reqm and reqm_hbond
				float opt_distance = myligand->reqm_AB [type_id1][type_id2];

				// Getting smoothed distance
				// smoothed_distance = function(dist, opt_distance)
				float smoothed_distance;
				float delta_distance = 0.5f*smooth;

				if (dist <= (opt_distance - delta_distance)) {
					smoothed_distance = dist + delta_distance;
				}
				else if (dist < (opt_distance + delta_distance)) {
					smoothed_distance = opt_distance;
				}
				else { // else if (dist >= (opt_distance + delta_distance))
					smoothed_distance = dist - delta_distance;
				}

				distance_id = (int) floor((100.0f*dist) + 0.5f) - 1; // +0.5: rounding, -1: r_xx_table [0] corresponds to r=0.01
				if (distance_id < 0) {
					distance_id = 0;
				}

				smoothed_distance_id = (int) floor((100.0f*smoothed_distance) + 0.5f) - 1; // +0.5: rounding, -1: r_xx_table [0] corresponds to r=0.01
				if (smoothed_distance_id < 0) {
					smoothed_distance_id = 0;
				}

				// ------------------------------------------------
				// Required only for flexrings
				// Checking if this is a CG-G0 atomic pair.
				// If so, then adding energy term (E = G * distance).
				// Initial specification required NON-SMOOTHED distance.
				// This interaction is evaluated at any distance,
				// so no cuttoffs considered here!
				// FIXME: accumulated into vW ... is that correct?
				if (((atom1_type_vdw_hb == ATYPE_CG_IDX) && (atom2_type_vdw_hb == ATYPE_G0_IDX)) || 
				    ((atom1_type_vdw_hb == ATYPE_G0_IDX) && (atom2_type_vdw_hb == ATYPE_CG_IDX))) {
					if (((atom_id1<myligand->true_ligand_atoms) && (atom_id2<myligand->true_ligand_atoms)) ||
					    ((atom_id1>=myligand->true_ligand_atoms) && (atom_id2>=myligand->true_ligand_atoms))) // if both atoms are of either a ligand or a flex res it's intra
						vW += G * dist;
					else
						interflexE += G * dist;
					/*printf("OpenCL host - calc_intraE_f: CG-G0 pair found!\n");*/
				}
				// ------------------------------------------------
				if (dist < dcutoff) // but only if the distance is less than distance cutoff value
				{
					pair_mod* pm = is_mod_pair(myligand->atom_types[type_id1], myligand->atom_types[type_id2], nr_mod_atype_pairs, mod_atype_pairs);
					if (tables.is_HB [type_id1][type_id2] && !pm) //H-bond
					{
						vdW1 = myligand->VWpars_C [type_id1][type_id2]*tables.r_12_table [smoothed_distance_id];
						vdW2 = myligand->VWpars_D [type_id1][type_id2]*tables.r_10_table [smoothed_distance_id];
						if (debug == 1) printf("H-bond interaction = ");
					}
					else // normal van der Waals or mod pair
					{
						float r_A = tables.r_12_table [smoothed_distance_id];
						float r_B = tables.r_6_table  [smoothed_distance_id];
						if(pm){
							int m = (myligand->VWpars_exp [type_id1][type_id2] & 0xFF00) >> 8;
							int n = (myligand->VWpars_exp [type_id1][type_id2] & 0xFF);
							float dist = 0.01f + smoothed_distance_id*0.01f;
							if(m!=12) r_A = powf(dist,-m);
							if(n!=6) r_B = powf(dist,-n);
						}
						vdW1 = myligand->VWpars_A [type_id1][type_id2]*r_A;
						vdW2 = myligand->VWpars_B [type_id1][type_id2]*r_B;
						if (debug == 1){
							if(pm)
								printf("Modified pair interaction = ");
							else
								printf("van der Waals interaction = ");
						}
					}
					if ((a_flex + b_flex) & 1){ // if both atoms are of either a ligand or a flex res it's intra
						interflexE += vdW1 - vdW2;
						if (analyze){
							const ReceptorAtom* curr;
							if(a_flex){ // a is flexres, b is ligand
								atomtypeid = myligand->base_type_idx[type_id2];
								atom_cnt = atom_id2;
								curr = &flexres_atoms[atom_id1-myligand->true_ligand_atoms];
								flex_reactive = myligand->reactive[atom_id1];
							} else{ // a is ligand, b is flexres
								atomtypeid = myligand->base_type_idx[type_id1];
								atom_cnt = atom_id1;
								curr = &flexres_atoms[atom_id2-myligand->true_ligand_atoms];
								flex_reactive = myligand->reactive[atom_id2];
							}
							if(myligand->reactive[atom_cnt] && flex_reactive && (dist <= R_cutoff)){
								datum.type     = 0; // 0 .. reactive, 1 .. hydrogen bond, 2 .. vdW
								datum.lig_id   = atom_cnt+1;
								datum.lig_name = myligand->atom_names[atom_cnt];
								datum.rec_id   = curr->id;
								datum.rec_name = curr->name;
								datum.residue  = curr->res_name;
								datum.res_id   = curr->res_id;
								datum.chain    = curr->chain_id;
								analysis->push_back(datum);
							} else{ // HB or vdW
								if((myligand->acceptor[atom_cnt] && curr->donor) ||
								   (myligand->donor[atom_cnt] && curr->acceptor)){
									if(dist <= H_cutoff){
										datum.type     = 1; // 0 .. reactive, 1 .. hydrogen bond, 2 .. vdW
										datum.lig_id   = atom_cnt+1;
										datum.lig_name = myligand->atom_names[atom_cnt];
										datum.rec_id   = curr->id;
										datum.rec_name = curr->name;
										datum.residue  = curr->res_name;
										datum.res_id   = curr->res_id;
										datum.chain    = curr->chain_id;
										analysis->push_back(datum);
									}
								} else{
									if((myligand->base_atom_types[atomtypeid][0]!='H') && (curr->atom_type[0]!='H') && // exclude Hydrogens,
									   !myligand->acceptor[atom_cnt] && !myligand->donor[atom_cnt] &&                  // non-H-bond capable atoms on ligand,
									   !curr->acceptor && !curr->donor){                                               // as well as flexres
										if(dist <= V_cutoff){
											datum.type     = 2; // 0 .. reactive, 1 .. hydrogen bond, 2 .. vdW
											datum.lig_id   = atom_cnt+1;
											datum.lig_name = myligand->atom_names[atom_cnt];
											datum.rec_id   = curr->id;
											datum.rec_name = curr->name;
											datum.residue  = curr->res_name;
											datum.res_id   = curr->res_id;
											datum.chain    = curr->chain_id;
											analysis->push_back(datum);
										}
									}
								}
							}
						}
					} else{ // both atoms are on either a ligand xor a flex res
						vW += vdW1 - vdW2;
					}
				}
				if (dist < 20.48)
				{
					if(dist<elec_min_distance){
						dist=elec_min_distance;
						distance_id = (int) floor((100*dist) + 0.5) - 1; // +0.5: rounding, -1: r_xx_table [0] corresponds to r=0.01
					}
					s1 = (myligand->solpar [type_id1] + tables.qasp_mul_absq [atom_id1]);
					s2 = (myligand->solpar [type_id2] + tables.qasp_mul_absq [atom_id2]);
					v1 = myligand->volume [type_id1];
					v2 = myligand->volume [type_id2];

					if (debug == 1)
						printf(" %lf, electrostatic = %lf, desolv = %lf\n", (vdW1 - vdW2), tables.q1q2[atom_id1][atom_id2] * tables.r_epsr_table [distance_id],
							   (s1*v2 + s2*v1) * tables.desolv_table [distance_id]);

					if ((a_flex + b_flex) & 1){ // if both atoms are of either a ligand or a flex res it's intra
						interflexE += tables.q1q2[atom_id1][atom_id2] * tables.r_epsr_table [distance_id] +
						              (s1*v2 + s2*v1) * tables.desolv_table [distance_id];
					} else{
						el += tables.q1q2[atom_id1][atom_id2] * tables.r_epsr_table [distance_id];
						desolv += (s1*v2 + s2*v1) * tables.desolv_table [distance_id];
					}
				}
			}
		}
	}

	if (debug == 1)
		printf("\nFinal energies: van der Waals = %lf, electrostatic = %lf, desolvation = %lf, total = %lf\n\n", vW, el, desolv, vW + el + desolv);

	if (!ignore_desolv)
		return (vW + el + desolv);
	else
		return (vW + el);
}

int map_to_all_maps(
                    Gridinfo*         mygrid,
                    Liganddata*       myligand,
                    std::vector<Map>& all_maps
                   )
{
	for (int i_atom = 0; i_atom<myligand->num_of_atoms;i_atom++){
		int type = myligand->atom_idxyzq[i_atom][0];
		int type_idx = myligand->base_type_idx[type];
		int map_idx = -1;
		for (unsigned int i_map = 0; i_map<all_maps.size(); i_map++){
			if (strcmp(all_maps[i_map].atype.c_str(),mygrid->grid_types[type_idx])==0){
				map_idx = i_map;
				break;
			}
		}
		if (map_idx == -1) {printf("\nERROR: Did not map to all_maps correctly."); return 1;}

		myligand->atom_map_to_fgrids[i_atom] = map_idx;
//		printf("\nMapping atom %d (type %d, %s) in the ligand to map #%d (%s)",i_atom,type_idx,mygrid->grid_types[type_idx],map_idx,all_maps[map_idx].atype.c_str());
	}

	return 0;
}
