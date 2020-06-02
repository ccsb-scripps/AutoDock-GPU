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



// Output showing the CG-G0 virtual bonds and pairs
// #define CG_G0_INFO

#include "processligand.h"

int init_liganddata(const char* ligfilename,
		    Liganddata* myligand,
		    Gridinfo*   mygrid,
		    bool cgmaps)
//The functions first parameter is an empty Liganddata, the second a variable of
//Gridinfo type. The function fills the num_of_atypes and atom_types fields of
//myligand according to the num_of_atypes and grid_types fields of mygrid. In
//this case it is supposed, that the ligand and receptor described by the two
//parameters correspond to each other.
//If the operation was successful, the function returns 0, if not, it returns 1.
{
	FILE* fp;
	int num_of_atypes, i, new_type;
	char atom_types [MAX_NUM_OF_ATYPES][4];
	char tempstr [256];

	fp = fopen(ligfilename, "rb"); // fp = fopen(ligfilename, "r");
	if (fp == NULL)
	{
		printf("Error: can't open ligand data file %s!\n", ligfilename);
		return 1;
	}

	num_of_atypes = 0;

	//reading the whole ligand pdbqt file
	while (fscanf(fp, "%s", tempstr) != EOF)
	{
		if ((strcmp(tempstr, "HETATM") == 0) || (strcmp(tempstr, "ATOM") == 0))
		{
			new_type = 1;	//supposing this will be a new atom type

			if ((strcmp(tempstr, "HETATM") == 0))	//seeking to the first coordinate value
				fseek(fp, 25, SEEK_CUR);
			else
				fseek(fp, 27, SEEK_CUR);
                        int result;
			result = fscanf(fp, "%*f");		//skipping fields
			result = fscanf(fp, "%*f");
			result = fscanf(fp, "%*f");
			result = fscanf(fp, "%*s");
			result = fscanf(fp, "%*s");
			result = fscanf(fp, "%*f");
			result = fscanf(fp, "%s", tempstr);	//reading atom type

			tempstr[3] = '\0';	//just to be sure strcpy wont fail even if something is wrong with position

			//checking if this atom has been already found
			for (i=0; i<num_of_atypes; i++)
			{
				if (strcmp(atom_types[i], tempstr) == 0)
					new_type = 0;	//this is not a new type
			}

			if (new_type == 1)	//if new type, copying string...
			{
				//checking if atom type number doesn't exceed 14
				if (num_of_atypes >= MAX_NUM_OF_ATYPES)
				{
					printf("Error: too many types of ligand atoms!\n");
					fclose(fp);
					return 1;
				}

				strcpy(atom_types[num_of_atypes], tempstr);
				num_of_atypes++;
			}
		}
	}

	fclose(fp);

	//copying field to ligand and grid data
	myligand->num_of_atypes = num_of_atypes;
	mygrid->num_of_atypes   = num_of_atypes;
	mygrid->num_of_map_atypes = num_of_atypes;
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
	for (i=0; i<num_of_atypes; i++)
	{
		strcpy(myligand->atom_types[i], atom_types[i]);
		if(cgmaps) {
			strcpy(mygrid->grid_types[i], atom_types[i]);
		}
		else
		{
			strncpy(mygrid->grid_types[i], atom_types[i],2);
			mygrid->grid_types[i][2] = '\0'; // make sure CG0..9 results in CG
			if (isdigit(mygrid->grid_types[i][1])) // make sure G0..9 results in G0
				mygrid->grid_types[i][1] = '0';
		}
#if defined(CG_G0_INFO)
		printf("Atom type %i -> %s -> grid type %s\n",i,myligand->atom_types[i],mygrid->grid_types[i]);
#endif
	}

	//adding the two other grid types to mygrid
	strcpy(mygrid->grid_types[num_of_atypes],   "e");
	strcpy(mygrid->grid_types[num_of_atypes+1], "d");

	return 0;
}

int set_liganddata_typeid(Liganddata* myligand,
			  int 	      atom_id,
			  const char* typeof_new_atom)
//The function sets the type index of the atom_id-th atom of myligand (in atom_idxyzq field),
//that is, it looks for the row in the atom_types field of myligand which is the same as
//typeof_new_atom, and sets the type index according to the row index.
//If the operation was successful, the function returns 0, if not, it returns 1.
{
	int i;
	int type;

	type = myligand->num_of_atypes;		//setting type to an invalid index
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
		myligand->atom_map_to_fgrids[atom_id] = type;
		return 0;
	}
	else		//if typeof_new_atom hasn't been found
	{
		printf("Error: no grid for ligand atom type %s!\n", typeof_new_atom);
		return 1;
	}
}

void get_intraE_contributors(Liganddata* myligand)
//The function fills the intraE_contributors field of the myligand parameter according
//to its bonds and rigid_structures field, which must contain valid data when calling
//this function.
{

	int  atom_id1, atom_id2, atom_id3, rotb_id1, rotb_id2;
	char atom_neighbours [MAX_NUM_OF_ATOMS];
	char atom_neighbours_temp [MAX_NUM_OF_ATOMS];
	int  atom_id_a, atom_id_b, structure_id_A, structure_id_B;
	int  atom_id_a2, atom_id_b2;

	for (atom_id1=0; atom_id1 < myligand->num_of_atoms; atom_id1++)
		for (atom_id2=atom_id1; atom_id2 < myligand->num_of_atoms; atom_id2++)
			//initially, all the values are 1, that is, all the atom pairs
			if (atom_id1 != atom_id2)
			{
				//are contributors
				myligand->intraE_contributors[atom_id1][atom_id2] = 1;
				myligand->intraE_contributors[atom_id2][atom_id1] = 1;
			}
			//except if they are the same
			else
				myligand->intraE_contributors[atom_id1][atom_id2] = 0;

	//There are 5 cases when the atom pair's energy contribution
	//has not to be included in intramolecular energy calculation
	//(that is, when the distance of the atoms are constant during docking) <- 1-4 interactions do actually change

	//CASE 1
	//if the two atoms are members of the same rigid structure, they aren't contributors
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

	//CASE2
	//if the atom pair represents a 1-2, 1-3 or 1-4 interaction, they aren't contributors
	//the following algorithm will find the first, second and third neighbours of each atom
	//(so the algorithm is redundant, several atoms will be found more than once)
	for (atom_id1=0; atom_id1 < myligand->num_of_atoms; atom_id1++)
	{
		//if atom_neighbours[i] is one,
		//it will indicate that the atom with id i is the neighbour of the atom with id atom_id1
		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
			if (myligand->bonds[atom_id1][atom_id2] > 0)
				atom_neighbours[atom_id2] = 1;		//neighbour
			else
				atom_neighbours[atom_id2] = 0;		//not neighbour

		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
			atom_neighbours_temp[atom_id2] = atom_neighbours [atom_id2];	//storing in a temp array as well

		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
			if (atom_neighbours[atom_id2] == 1)			//for each neighbour of atom_id1
				for (atom_id3=0; atom_id3 < myligand->num_of_atoms; atom_id3++)
					if (myligand->bonds[atom_id2][atom_id3] > 0)		//if atom_id3 is second neighbour of atom_id1
						atom_neighbours_temp[atom_id3] = 1;			//changing the temporary array

		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
				atom_neighbours[atom_id2] = atom_neighbours_temp[atom_id2];

		//now ones of atom_neighbours indicate the first and second neighbours of atom_id1

		//the same code as above
		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
			if (atom_neighbours[atom_id2] == 1)			//for each neighbour or second neighbour of atom_id1
				for (atom_id3=0; atom_id3 < myligand->num_of_atoms; atom_id3++)
					if (myligand->bonds[atom_id2][atom_id3] > 0)		//if atom_id3 is second or third neighbour of atom_id1
						atom_neighbours_temp[atom_id3] = 1;

		for (atom_id2=0; atom_id2 < myligand->num_of_atoms; atom_id2++)
			atom_neighbours[atom_id2] = atom_neighbours_temp[atom_id2];

		//now atom_neighbours [i] is one for atom_id1, its first, second and third neighbours, pairs consisting of
		//these atoms aren't contributors
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
			if ((strncmp(myligand->atom_types[atom_typeid1], "G", 1) == 0) || (strncmp(myligand->atom_types[atom_typeid2], "G", 1) == 0)) {
				myligand->intraE_contributors[atom_id1][atom_id2] = 0;
				myligand->intraE_contributors[atom_id2][atom_id1] = 0;
			}
			// first, make sure non-matching ID pairs don't interact with each other (the code above happily allows it)
			if (  (myligand->bonds[atom_id1][atom_id2] == 0) && // for non-bonded CG-G0 atoms
			    (((strncmp(myligand->atom_types[atom_typeid1], "CG", 2) == 0) &&
			      (strncmp(myligand->atom_types[atom_typeid2], "G", 1) == 0) &&
			      (strcmp(myligand->atom_types[atom_typeid1]+2,myligand->atom_types[atom_typeid2]+1) != 0)) || // with non-matching ids
			     ((strncmp(myligand->atom_types[atom_typeid1], "G", 1) == 0) &&
			      (strncmp(myligand->atom_types[atom_typeid2], "CG", 2) == 0) &&
			      (strcmp(myligand->atom_types[atom_typeid1]+1,myligand->atom_types[atom_typeid2]+2) != 0))))
			{
				myligand->intraE_contributors[atom_id1][atom_id2] = 0;
				myligand->intraE_contributors[atom_id2][atom_id1] = 0;
			}
			// second, let matching ID pair interact
			if (  (myligand->bonds[atom_id1][atom_id2] == 0) && // for non-bonded CG-G0 atoms
			    (((strncmp(myligand->atom_types[atom_typeid1], "CG", 2) == 0) &&
			      (strncmp(myligand->atom_types[atom_typeid2], "G", 1) == 0) &&
			      (strcmp(myligand->atom_types[atom_typeid1]+2,myligand->atom_types[atom_typeid2]+1) == 0)) || // with matching ids
			     ((strncmp(myligand->atom_types[atom_typeid1], "G", 1) == 0) &&
			      (strncmp(myligand->atom_types[atom_typeid2], "CG", 2) == 0) &&
			      (strcmp(myligand->atom_types[atom_typeid1]+1,myligand->atom_types[atom_typeid2]+2) == 0))))
			{
				myligand->intraE_contributors[atom_id1][atom_id2] = 1;
				myligand->intraE_contributors[atom_id2][atom_id1] = 1;
#if defined(CG_G0_INFO)
				printf("Found CG-G0 pair: atom %i (%s) <-> atom %i (%s)\n",atom_id1+1,myligand->atom_types[atom_typeid1],atom_id2+1,myligand->atom_types[atom_typeid2]);
#endif
			}
		}
	}

	//CASE3
	//Let atom a and atom b be the endpoints of the same rotatable bond,
	//and A and B the rigid structures connected
	//to the rotatable bond's a and b atoms, respectively.
	//The atom pairs consisting of a and any atom of B aren't contributors.
	//Similarly, atom pairs consisting of b and any atom of A aren't, either.

	for (rotb_id1=0; rotb_id1 < myligand->num_of_rotbonds; rotb_id1++)
	{
		atom_id_a = myligand->rotbonds[rotb_id1][0];
		atom_id_b = myligand->rotbonds[rotb_id1][1];

		structure_id_A = myligand->atom_rigid_structures[atom_id_a];
		structure_id_B = myligand->atom_rigid_structures[atom_id_b];

		for (atom_id1=0; atom_id1 < myligand->num_of_atoms; atom_id1++)
		{
			//if atom_id1 is member of structure A
			if (myligand->atom_rigid_structures[atom_id1] == structure_id_A)
			{
				myligand->intraE_contributors[atom_id1][atom_id_b] = 0;
				myligand->intraE_contributors[atom_id_b][atom_id1] = 0;
			}

			//if atom_id1 is member of structure B
			if (myligand->atom_rigid_structures[atom_id1] == structure_id_B)
			{
				myligand->intraE_contributors[atom_id1][atom_id_a] = 0;
				myligand->intraE_contributors[atom_id_a][atom_id1] = 0;
			}
		}
	}

	//CASE4
	//If one end of two different rotatable bonds are connected to the same rigid structure, the other end, that is,
	//atoms of the bonds aren't contributors.

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
			if ((strcmp(myligand->atom_types[atom_id1_type], "W") == 0) || (strcmp(myligand->atom_types[atom_id2_type], "W") == 0))
			{
				myligand->intraE_contributors[atom_id1][atom_id2] = 0;
				myligand->intraE_contributors[atom_id2][atom_id1] = 0;
			}
		}
	}
}

int get_bonds(Liganddata* myligand)
//The function fills the bonds field of myligand based on the distance of the ligand's atoms,
//which can be calculated from the atom_idxyzq field, so this field must contain valid data
//when calling this function.
{
	char atom_names [ATYPE_GETBONDS][3];
	// Values from atomic parameter file AD4.1_bound_dat / "bond_index"
	char bondtype_id [ATYPE_GETBONDS] = {
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
					     0  // "CX"
					    };

	double mindist[7][7];
	double maxdist[7][7];

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

	//Filling the mindist and maxdist tables (as in Autodock, see AD4_parameters.dat and mdist.h).
	//It is supposed that the bond length of atoms with bondtype_id1 and bondtype_id2 is
	//between mindist[bondtype_id1][bondtype_id2] and maxdist[bondtype_id1][bondtype_id2]
	for (i=0; i<7; i++)
	{
		for (j=0; j<7; j++)
		{
			mindist[i][j] = 0.9;
			maxdist[i][j] = 2.1;
		}
	}

	//0=C, 3=H
	mindist[0][3] = 1.07; mindist[3][0] = mindist[0][3];
	maxdist[0][3] = 1.15; maxdist[3][0] = maxdist[0][3];

	//1=N
	mindist[1][3] = 0.99; mindist[3][1] = mindist[1][3];
	maxdist[1][3] = 1.10; maxdist[3][1] = maxdist[1][3];

	//2=O
	mindist[2][3] = 0.94; mindist[3][2] = mindist[2][3];
	maxdist[2][3] = 1.10; maxdist[3][2] = maxdist[2][3];

	//6=S
	mindist[6][3] = 1.316; mindist[3][6] = mindist[6][3];
	maxdist[6][3] = 1.356; maxdist[3][6] = maxdist[6][3];

	//5=P
	mindist[5][3] = 1.35; mindist[3][5] = mindist[5][3];
	maxdist[5][3] = 1.40; maxdist[3][5] = maxdist[5][3];

	mindist[1][2] = 1.11;  // N=O is ~ 1.21 A, minus 0.1A error
	maxdist[1][2] = 1.50;  // N-O is ~ 1.40 A, plus 0.1 A error
	mindist[2][1] = mindist[1][2];  // N=O is ~ 1.21 A, minus 0.1A error
	maxdist[2][1] = maxdist[1][2];  // N-O is ~ 1.40 A, plus 0.1 A error

	//There is no bond between two hydrogenes (does not derive from Autodock)
	mindist[3][3] = 2;
	maxdist[3][3] = 1;

	for (atom_id1=0; atom_id1 < myligand->num_of_atoms-1; atom_id1++)
	{
		atom_typeid1 = myligand->atom_idxyzq[atom_id1][0];
		for (atom_id2=atom_id1; atom_id2 < myligand->num_of_atoms; atom_id2++)
		{
			atom_typeid2 = myligand->atom_idxyzq[atom_id2][0];
			temp_point1[0] = myligand->atom_idxyzq[atom_id1][1];
			temp_point1[1] = myligand->atom_idxyzq[atom_id1][2];
			temp_point1[2] = myligand->atom_idxyzq[atom_id1][3];
			temp_point2[0] = myligand->atom_idxyzq[atom_id2][1];
			temp_point2[1] = myligand->atom_idxyzq[atom_id2][2];
			temp_point2[2] = myligand->atom_idxyzq[atom_id2][3];
			temp_dist = distance(temp_point1, temp_point2);
			atom_nameid1 = ATYPE_GETBONDS;
			atom_nameid2 = ATYPE_GETBONDS;
			//identifying atom types
			for (i=0; i<ATYPE_GETBONDS; i++)
			{
				if ((atom_names[i][1] == 'x') || (atom_names[i][1] == '0')) // this catches "G0..9"
				{
					if (atom_names[i][0] == toupper(myligand->atom_types[atom_typeid1][0]))
						atom_nameid1 = i;
				}
				else
				{
					if (strincmp(atom_names[i], myligand->atom_types[atom_typeid1], 2) == 0)
						atom_nameid1 = i;
				}
			}

			for (i=0; i<ATYPE_GETBONDS; i++)
			{
				if ((atom_names[i][1] == 'x') || (atom_names[i][1] == '0')) // this catches "G0..9"
				{
					if (atom_names[i][0] == toupper(myligand->atom_types[atom_typeid2][0]))
						atom_nameid2 = i;
				}
				else
				{
					if (strincmp(atom_names[i], myligand->atom_types[atom_typeid2], 2) == 0)
					{
						atom_nameid2 = i;
					}
				}
			}

			if ((atom_nameid1 == ATYPE_GETBONDS) || (atom_nameid2 == ATYPE_GETBONDS))
			{
				printf("Error: Ligand includes atom with unknown type: %s or %s!\n", myligand->atom_types[atom_typeid1], myligand->atom_types[atom_typeid2]);
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
				}
				else if ((atom_nameid1 == CG_nameid) && (atom_nameid2 == CG_nameid) && // two CG atoms
					 (strcmp(myligand->atom_types[atom_typeid1]+2,myligand->atom_types[atom_typeid2]+2) == 0)) // and matching numbers
					{
						myligand->bonds [atom_id1][atom_id2] = 2; // let's call 2 a "virtual" bond to
						myligand->bonds [atom_id2][atom_id1] = 2; // distinguish them if needed later
#if defined(CG_G0_INFO)
						printf("Found virtual CG-CG bond between atom %i (%s) and atom %i (%s).\n",atom_id1+1,myligand->atom_types[atom_typeid1],atom_id2+1,myligand->atom_types[atom_typeid2]);
#endif
					}
					else
					{
						myligand->bonds [atom_id1][atom_id2] = 0;
						myligand->bonds [atom_id2][atom_id1] = 0;
					}
		} // inner for-loop
	} // outer for-loop

	return 0;
}

int get_VWpars(Liganddata* myligand, const double AD4_coeff_vdW, const double AD4_coeff_hb)
//The function calculates the Van der Waals parameters for each pair of atom
//types of the ligand given by the first parameter, and fills the VWpars_A, _B,
//_C and _D fields according to the result as well as the solvation parameters
//and atomic volumes (desolv and volume fields) for each atom type.
{
	char atom_names [ATYPE_NUM][3];

	// Initial implementation included 22 atom-types.
	// Handling flexrings requires 2 additional atom-types: "CG" & "G0".
	// All corresponding CG & G0 values are added as last 2 elements
	// in existing atom-types look-up tables.

	// See in ./common/defines.h: ATYPE_CG_IDX and ATYPE_G0_IDX
	// CG (idx=22): a copy of the standard "C" (idx=3) atom-type
	// G0 (idx=23): an invisible atom-type, all atomic parameters are zero
	

	//Sum of vdW radii of two like atoms (A)
	double reqm [ATYPE_NUM] = {
				   2.00, 2.00, 2.00, 4.00, 4.00,
				   3.50, 3.50, 3.50, 3.20, 3.20,
				   3.09, 1.30, 4.20, 4.00, 4.00,
				   4.09, 1.98, 1.30, 1.30, 1.48,
				   4.33, 4.72,
				   4.00, // CG
				   0.00, // G0
				   0.00, // W
				   4.00, // CX
				   3.50, // NX
				   3.20  // OX
				  };

	//cdW well depth (kcal/mol)
	double eps [ATYPE_NUM] = {
				  0.020, 0.020, 0.020, 0.150, 0.150,
				  0.160, 0.160, 0.160, 0.200, 0.200,
				  0.080, 0.875, 0.200, 0.200, 0.200,
				  0.276, 0.550, 0.875, 0.010, 0.550,
				  0.389, 0.550,
				  0.150, // CG
				  0.000, // G0
				  0.000, // W
				  0.150, // CX
				  0.160, // NX
				  0.200  // OX
				 };

	//Sum of vdW radii of two like atoms (A) in case of hydrogen bond
	double reqm_hbond [ATYPE_NUM] = {
					 0.0, 0.0, 0.0, 0.0, 0.0,
					 0.0, 1.9, 1.9, 1.9, 1.9,
					 0.0, 0.0, 0.0, 2.5, 0.0,
					 0.0, 0.0, 0.0, 0.0, 0.0,
					 0.0, 0.0,
					 0.0, // CG
					 0.0, // G0
					 0.0, // W
					 0.0, // CX
					 0.0, // NX
					 1.9  // OX
					};

	//cdW well depth (kcal/mol) in case of hydrogen bond
	double eps_hbond [ATYPE_NUM] = {
					0.0, 1.0, 1.0, 0.0, 0.0, 	//HD and HS value is 1 so that it is not necessary to decide which atom_typeid
					0.0, 5.0, 5.0, 5.0, 5.0, 	//corresponds to the hydrogen when reading eps_hbond...
					0.0, 0.0, 0.0, 1.0, 0.0,
					0.0, 0.0, 0.0, 0.0, 0.0,
					0.0, 0.0,
					0.0, // CG
					0.0, // G0
					0.0, // W
					0.0, // CX
					0.0, // NX
					5.0  // OX
				       };

	//volume of atoms
	double volume [ATYPE_NUM] = {
				      0.0000,  0.0000,  0.0000, 33.5103, 33.5103,
				     22.4493, 22.4493, 22.4493, 17.1573, 17.1573,
				     15.4480,  1.5600, 38.7924, 33.5103, 33.5103,
				     35.8235,  2.7700,  2.1400,  1.8400,  1.7000,
				     42.5661, 55.0585,
				     33.5103, // CG
				      0.0000, // G0
				      0.0000, // W
				     33.5103, // CX
				     22.4493, // NX
				     17.1573  // OX
				    };

	//atomic solvation parameters
	double solpar [ATYPE_NUM] = {
				      0.00051,  0.00051,  0.00051, -0.00143, -0.00052,
				     -0.00162, -0.00162, -0.00162, -0.00251, -0.00251,
				     -0.00110, -0.00110, -0.00110, -0.00214, -0.00214,
				     -0.00110, -0.00110, -0.00110, -0.00110, -0.00110,
				     -0.00110, -0.00110,
				     -0.00143, // CG
				      0.00000, // G0
				      0.00000, // W
				     -0.00143, // CX
				     -0.00162, // NX
				     -0.00251  // OX
				    };

	int atom_typeid1, atom_typeid2, VWid_atype1, VWid_atype2, i;
	double eps12, reqm12;

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
	strcpy(atom_names [/*22*/ATYPE_CG_IDX], "CG"); // CG
	strcpy(atom_names [/*23*/ATYPE_G0_IDX], "G0"); // G0
	strcpy(atom_names [/*24*/ATYPE_W_IDX], "W"); // W
	strcpy(atom_names [/*25*/ATYPE_CX_IDX], "CX"); // CX
	strcpy(atom_names [/*26*/ATYPE_NX_IDX], "NX"); // NX
	strcpy(atom_names [/*27*/ATYPE_OX_IDX], "OX"); // OX

	// Using this variable to signal when the CG-CG pair was found.
	// This is further reused to set vdW constant coeffs: "vdWpars_A" and "vdWpars_B".
	// found_CG_CG_pair == true  -> set vdW coeffs to zero
	// found_CG_CG_pair == false -> use vdW default values
	bool found_CG_CG_pair;

	for (atom_typeid1 = 0; atom_typeid1 < myligand->num_of_atypes; atom_typeid1++)
		for (atom_typeid2 = 0; atom_typeid2 < myligand->num_of_atypes; atom_typeid2++)
		{
			VWid_atype1 = ATYPE_NUM;
			VWid_atype2 = ATYPE_NUM;

			// Was CG_CG_pair found?
			found_CG_CG_pair = false;

			//identifying atom types
			for (i=0; i<ATYPE_NUM; i++) {
				if (strincmp(atom_names [i], myligand->atom_types [atom_typeid1],2) == 0) {
					VWid_atype1 = i;
					myligand->atom1_types_reqm [atom_typeid1] = VWid_atype1;
				}
				else
				{
					if(atom_names[i][1] == '0') {
						if (atom_names[i][0] == toupper(myligand->atom_types[atom_typeid1][0])) {
							VWid_atype1 = i;
							myligand->atom1_types_reqm [atom_typeid1] = VWid_atype1;
						}
					}
				}
			}

			for (i=0; i<ATYPE_NUM; i++) {
				if (strincmp(atom_names[i], myligand->atom_types[atom_typeid2],2) == 0) {
					VWid_atype2 = i;
					myligand->atom2_types_reqm [atom_typeid2] = VWid_atype2;
				}
				else
				{
					if(atom_names[i][1] == '0') {
						if (atom_names[i][0] == toupper(myligand->atom_types[atom_typeid2][0])) {
							VWid_atype2 = i;
							myligand->atom2_types_reqm [atom_typeid2] = VWid_atype2;
						}
					}
				}
			}

			// Was CG_CG_pair found?
			// CG atom-type has a idx=22
			if ((VWid_atype1 == /*22*/ATYPE_CG_IDX) && (VWid_atype2 == /*22*/ATYPE_CG_IDX) &&
			    (strcmp(myligand->atom_types[atom_typeid1]+2,myligand->atom_types[atom_typeid2]+2) == 0)) { // make sure to only exclude matching IDs
				found_CG_CG_pair = true;
			}
			else {
				found_CG_CG_pair = false;
			}

			if (VWid_atype1 == ATYPE_NUM)
			{
				printf("Error: Ligand includes atom with unknown type 1: %s!\n", myligand->atom_types [atom_typeid1]);
				return 1;
			}

			if  (VWid_atype2 == ATYPE_NUM)
			{
				printf("Error: Ligand includes atom with unknown type 2: %s!\n", myligand->atom_types [atom_typeid2]);
				return 1;
			}

			//calculating van der Waals parameters
			if (is_H_bond(myligand->atom_types [atom_typeid1], myligand->atom_types [atom_typeid2]) != 0)
			{
				eps12 = AD4_coeff_hb * eps_hbond [VWid_atype1] * eps_hbond [VWid_atype2];	//The hydrogen's eps is 1, doesn't change the value...
				reqm12 = reqm_hbond [VWid_atype1] + reqm_hbond [VWid_atype2];			//The hydrogen's is 0, doesn't change the value...
				myligand->VWpars_C [atom_typeid1][atom_typeid2] = 5*eps12*pow(reqm12, 12);
				myligand->VWpars_D [atom_typeid1][atom_typeid2] = 6*eps12*pow(reqm12, 10);
				myligand->VWpars_A [atom_typeid1][atom_typeid2] = 0;
				myligand->VWpars_B [atom_typeid1][atom_typeid2] = 0;
			}
			else
			{
				eps12 = AD4_coeff_vdW * sqrt(eps [VWid_atype1]*eps [VWid_atype2]);		//weighting with coefficient for van der Waals term
				reqm12 = 0.5*(reqm [VWid_atype1]+reqm [VWid_atype2]);

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
				eps12 = AD4_coeff_vdW * sqrt(eps [3]*eps [8]);		//weighting with coefficient for van der Waals term
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
		}

	for (atom_typeid1 = 0; atom_typeid1 < ATYPE_NUM/*myligand->num_of_atypes*/; atom_typeid1++) {
		myligand->reqm[atom_typeid1]       = reqm[atom_typeid1];
		myligand->reqm_hbond[atom_typeid1] = reqm_hbond[atom_typeid1];
	}

	for (atom_typeid1 = 0; atom_typeid1 < myligand->num_of_atypes; atom_typeid1++)
	{
		VWid_atype1 = ATYPE_NUM;

		//identifying atom type
		for (i=0; i<ATYPE_NUM; i++) {
			if (strincmp(atom_names [i], myligand->atom_types [atom_typeid1], 2) == 0) // captures GG0..9 to CG in tables
			{
				VWid_atype1 = i;
			}
			else
			{
				if(atom_names[i][1] == '0') { // captures G0..9 to G0 in tables
					if (atom_names[i][0] == toupper(myligand->atom_types[atom_typeid1][0]))
						VWid_atype1 = i;
				}
			}
		}

		if (VWid_atype1 == ATYPE_NUM)
		{
			printf("Error: Ligand includes atom with unknown type: %s\n", myligand->atom_types [atom_typeid1]);
			return 1;
		}

		myligand->volume [atom_typeid1] = volume [VWid_atype1];
		myligand->solpar [atom_typeid1] = solpar [VWid_atype1];
	}

	return 0;
}

int get_moving_and_unit_vectors(Liganddata* myligand)
//The function calculates and fills the
//rotbonds_moving_vectors and rotbonds_unit_vectors fields of the myligand parameter.
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
		//capturing unitvector's direction
		atom_id_pointA = myligand->rotbonds [rotb_id][0];			//capturing indexes of the two atoms
		atom_id_pointB = myligand->rotbonds [rotb_id][1];
		for (i=0; i<3; i++)												//capturing coordinates of the two atoms
		{
			pointA [i] = myligand->atom_idxyzq [atom_id_pointA][i+1];
			pointB [i] = myligand->atom_idxyzq [atom_id_pointB][i+1];
			unitvec [i] = pointB [i] - pointA [i];
		}

		//normalize unitvector
		dist = distance(pointA, pointB);

		if (dist==0.0){
			printf("Error: Two atoms have the same XYZ coordinates!\n");
                	return 1;
		}

		for (i=0; i<3; i++) //capturing coordinates of the two atoms
		{
			unitvec [i] = unitvec [i]/dist;
			if (unitvec [i] >= 1)		//although it is not too probable...
				unitvec [i] = 0.999999;
		}

		for (i=0; i<3; i++)
			origo [i] = 0;

		//capturing moving vector
		vec_point2line(origo, pointA, pointB, movvec);

		for (i=0; i<3; i++)
		{
			myligand->rotbonds_moving_vectors [rotb_id][i] = movvec [i];
			myligand->rotbonds_unit_vectors [rotb_id][i] = unitvec [i];
		}
	}
	return 0;
}

int get_liganddata(const char* ligfilename, Liganddata* myligand, const double AD4_coeff_vdW, const double AD4_coeff_hb)
//The functions second parameter is a Liganddata variable whose num_of_atypes
//and atom_types fields must contain valid data.
//The function opens the file ligfilename, which is supposed to be an AutoDock4 pdbqt file,
//and fills the other fields of myligand according to the content of the file.
//If the operation was successful, the function returns 0, if not, it returns 1.
{
	FILE* fp;
	char tempstr [128];
	int atom_counter;
	int branch_counter;
	int endbranch_counter;
	int branches [MAX_NUM_OF_ROTBONDS][3];
	int i,j,k;
	char atom_rotbonds_temp [MAX_NUM_OF_ATOMS][MAX_NUM_OF_ROTBONDS];
	int current_rigid_struct_id, reserved_highest_rigid_struct_id;

	atom_counter = 0;

	fp = fopen(ligfilename, "rb"); // fp = fopen(ligfilename, "r");
	if (fp == NULL)
	{
		printf("Error: can't open ligand data file %s!\n", ligfilename);
		return 1;
	}

	//reading atomic coordinates, charges and atom types, and writing
	//data to myligand->atom_idxyzq
	while (fscanf(fp, "%s", tempstr) != EOF)
	{
		if ((strcmp(tempstr, "HETATM") == 0) || (strcmp(tempstr, "ATOM") == 0))
		{
			if (atom_counter > MAX_NUM_OF_ATOMS-1)
			{
				printf("Error: ligand consists of too many atoms'\n");
				printf("Maximal allowed number of atoms is %d!\n", MAX_NUM_OF_ATOMS);
				fclose(fp);
				return 1;
			}
			if ((strcmp(tempstr, "HETATM") == 0))	//seeking to the first coordinate value
				fseek(fp, 25, SEEK_CUR);
			else
				fseek(fp, 27, SEEK_CUR);
			fscanf(fp, "%lf", &(myligand->atom_idxyzq [atom_counter][1]));
			fscanf(fp, "%lf", &(myligand->atom_idxyzq [atom_counter][2]));
			fscanf(fp, "%lf", &(myligand->atom_idxyzq [atom_counter][3]));
			fscanf(fp, "%s", tempstr);	//skipping the next two fields
			fscanf(fp, "%s", tempstr);
			fscanf(fp, "%lf", &(myligand->atom_idxyzq [atom_counter][4]));	//reading charge
			fscanf(fp, "%s", tempstr);	//reading atom type
			if (set_liganddata_typeid(myligand, atom_counter, tempstr) != 0){	//the function sets the type index
				fclose(fp);
				return 1;
			}
			atom_counter++;
		}
	}

	myligand->num_of_atoms = atom_counter;

	fclose(fp);

	//filling atom_rotbonds_temp with 0s
	for (i=0; i<myligand->num_of_atoms; i++)
	{
		for (j=0; j<MAX_NUM_OF_ROTBONDS; j++)
			atom_rotbonds_temp [i][j] = 0;
	}

	fp = fopen(ligfilename, "rb"); // fp = fopen(ligfilename, "r");	//re-open the file
	if (fp == NULL)
	{
		printf("Error: can't open ligand data file %s!\n", ligfilename);
		return 1;
	}

	branch_counter = 0;
	atom_counter = 0;
	endbranch_counter = 0;

	current_rigid_struct_id = 1;
	reserved_highest_rigid_struct_id = 1;

	//reading data for rotbonds and atom_rotbonds fields
	while (fscanf(fp, "%s", tempstr) != EOF)
	{
		if ((strcmp(tempstr, "HETATM") == 0) || (strcmp(tempstr, "ATOM") == 0))		//if new atom, looking for open rotatable bonds
		{
			for (i=0; i<branch_counter; i++)	//for all branches found until now
				if (branches [i][2] == 1)	//if it is open, the atom has to be rotated
					atom_rotbonds_temp [atom_counter][i] = 1;	//modifying atom_rotbonds_temp
					/*else it is 2, so it is closed, so nothing to be done...*/

			myligand->atom_rigid_structures [atom_counter] = current_rigid_struct_id;	//using the id of the current rigid structure

			atom_counter++;
		}

		if (strcmp(tempstr, "BRANCH") == 0)	//if new branch, stroing atom indexes into branches [][]
		{
			if (branch_counter >= MAX_NUM_OF_ROTBONDS)
			{
				printf("Error: ligand includes too many rotatable bonds.\n");
				printf("Maximal allowed number is %d.\n", MAX_NUM_OF_ROTBONDS);
				fclose(fp);
				return 1;
			}
			fscanf(fp, "%d", &(branches [branch_counter][0]));
			fscanf(fp, "%d", &(branches [branch_counter][1]));
			(branches [branch_counter][0])--;	//atom IDs start from 0 instead of 1
			(branches [branch_counter][1])--;

			branches [branch_counter][2] = 1;	// 1 means the branch is open, atoms will be rotated

			branch_counter++;

			reserved_highest_rigid_struct_id++;		//next ID is reserved
			current_rigid_struct_id = reserved_highest_rigid_struct_id;		//New branch means new rigid structure, and a new id as well
		}

		if (strcmp(tempstr, "ENDBRANCH") == 0)
		{
			fscanf(fp, "%d", &(myligand->rotbonds [endbranch_counter][0]));	//rotatable bonds have to be stored in the order
			fscanf(fp, "%d", &(myligand->rotbonds [endbranch_counter][1])); //of endbranches
			(myligand->rotbonds [endbranch_counter][0])--;
			(myligand->rotbonds [endbranch_counter][1])--;

			for (i=0; i<branch_counter; i++)	//the branch have to be closed
				if ((branches [i][0] == myligand->rotbonds [endbranch_counter][0]) &&
				    (branches [i][1] == myligand->rotbonds [endbranch_counter][1]))
					branches [i][2] = 2;
			endbranch_counter++;

			current_rigid_struct_id--;	//probably unnecessary since there is a new branch after every endbranch...
		}
	}

	fclose(fp);

	myligand->num_of_rotbonds = branch_counter;

	//Now the rotbonds field contains the rotatable bonds (that is, the corresponding two atom's indexes) in the proper order
	//(this will be the order of rotations if an atom have to be rotated around more then one rotatable bond.) However, the
	//atom_rotbonds_temp, whose column indexes correspond to rotatable bond indexes, contains data according to the order of
	//branches (that is, according to branches [][] array), instead of endbranches. Columns of atom_rotbonds_temp have to be
	//copied now to myligand->atom_rotbonds, but in the proper order.
	for (i=0; i<branch_counter; i++)
		for (j=0; j<branch_counter; j++)
			if ((myligand->rotbonds [i][0] == branches [j][0]) && (myligand->rotbonds [i][1] == branches [j][1]))
				for (k=0; k<myligand->num_of_atoms; k++)
					myligand->atom_rotbonds [k][i] = atom_rotbonds_temp [k][j];		//rearrange the columns

	if (get_bonds(myligand) == 1)
		return 1;

	get_intraE_contributors(myligand);

	if (get_VWpars(myligand, AD4_coeff_vdW, AD4_coeff_hb) == 1)
		return 1;

	if (get_moving_and_unit_vectors(myligand) == 1)
                return 1;

	return 0;
}

int gen_new_pdbfile(const char* oldpdb, const char* newpdb, const Liganddata* myligand)
//The funciton opens old pdb file, which is supposed to be an AutoDock4 pdbqt file, and
//copies it to newpdb file, but switches the coordinate values to the atomic coordinates
//of myligand, so newpdb file will be identical to oldpdb except the coordinate values.
//Myligand has to be the ligand which was originally read from oldpdb.
//If the operation was successful, the function returns 0, if not, it returns 1.
{
	FILE* fp_old;
	FILE* fp_new;
	char tempstr [256];
	char tempstr_short [32];
	int acnt_oldlig, acnt_newlig;
	int i,j;

	acnt_oldlig = 0;
	acnt_newlig = 0;

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

	while (fgets(tempstr, 255, fp_old) != NULL)		//reading a whole row from oldpdb
	{
		sscanf(tempstr, "%s", tempstr_short);
		if ((strcmp(tempstr_short, "HETATM") == 0) || (strcmp(tempstr_short, "ATOM") == 0))	//if the row begins with HETATM/ATOM, coordinates must be switched
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
		fprintf(fp_new, "%s", tempstr);		//writing the row to newpdb
	}

	if (acnt_oldlig != myligand->num_of_atoms)
	{
		printf("%d %d \n", acnt_oldlig, myligand->num_of_atoms);
		printf("Warning: new lingand consists more atoms than old one.\n");
		printf("Not all the atoms have been written to file!\n");
	}

	fclose(fp_old);
	fclose(fp_new);

	return 0;
}

void get_movvec_to_origo(const Liganddata* myligand, double movvec [])
//The function returns the moving vector in the second parameter which moves the ligand
//(that is, its geometrical center point) given by the first parameter to the origo).
{
	double tmp_x, tmp_y, tmp_z;
	int i;

	tmp_x = 0;
	tmp_y = 0;
	tmp_z = 0;

	for (i=0; i < myligand->num_of_atoms; i++)
	{
		tmp_x += myligand->atom_idxyzq [i][1];
		tmp_y += myligand->atom_idxyzq [i][2];
		tmp_z += myligand->atom_idxyzq [i][3];
	}

	movvec [0] = -1*tmp_x/myligand->num_of_atoms;
	movvec [1] = -1*tmp_y/myligand->num_of_atoms;
	movvec [2] = -1*tmp_z/myligand->num_of_atoms;
}

void move_ligand(Liganddata* myligand, const double movvec [])
//The function moves the ligand given by the first parameter according to
//the vector given by the second one.
{
	int i;

	for (i=0; i < myligand->num_of_atoms; i++)
	{
		myligand->atom_idxyzq [i][1] += movvec [0];
		myligand->atom_idxyzq [i][2] += movvec [1];
		myligand->atom_idxyzq [i][3] += movvec [2];
	}
}

void scale_ligand(Liganddata* myligand, const double scale_factor)
//The function scales the ligand given by the first parameter according to the factor
//given by the second (that is, all the ligand atom coordinates will be multiplied by
//scale_factor).
{
	int i,j;

	for (i=0; i < myligand->num_of_atoms; i++)
		for (j=1; j<4; j++)
			myligand->atom_idxyzq [i][j] = myligand->atom_idxyzq [i][j]*scale_factor;
}

double calc_rmsd(const Liganddata* myligand_ref, const Liganddata* myligand, const int handle_symmetry)
//The function calculates the RMSD value (root mean square deviation of the
//atomic distances for two conformations of the same ligand) and returns it.
//If the handle_symmetry parameter is 0, symmetry is not handled, and the
//distances are calculated between atoms with the same atom id. If it is not
//0, one atom from myligand will be compared to the closest atom with the same
//type from myligand_ref and this will be accumulated during rmsd calculation
//(which is a silly method but this is applied in AutoDock, too).
//The two positions must be given by the myligand and myligand_ref parameters.
{
	int i,j;
	double sumdist2;
	double mindist2;

	if (myligand_ref->num_of_atoms != myligand->num_of_atoms)
	{
		printf("Warning: RMSD can't be calculated, atom number mismatch!\n");
		return 100000;	//returning unreasonable value
	}

	sumdist2 = 0;

	if (handle_symmetry == 0)
	{
		for (i=0; i<myligand->num_of_atoms; i++)
		{
			sumdist2 += pow(distance(&(myligand->atom_idxyzq [i][1]), &(myligand_ref->atom_idxyzq [i][1])), 2);
		}
	}
	else	//handling symmetry with the silly AutoDock method
	{
		for (i=0; i<myligand->num_of_atoms; i++)
		{
			mindist2 = 100000;	//initial value should be high enough so that it is ensured that lower distances will be found
			for (j=0; j<myligand_ref->num_of_atoms; j++)	//looking for the closest atom with same type from the reference
			{
				if (myligand->atom_idxyzq [i][0] == myligand_ref->atom_idxyzq [j][0])
					if (pow(distance(&(myligand->atom_idxyzq [i][1]), &(myligand_ref->atom_idxyzq [j][1])), 2) < mindist2)
						mindist2 = pow(distance(&(myligand->atom_idxyzq [i][1]), &(myligand_ref->atom_idxyzq [j][1])), 2);
			}
			sumdist2 += mindist2;
		}
	}

	return (sqrt(sumdist2/myligand->num_of_atoms));
}

double calc_ddd_Mehler_Solmajer(double distance)
//The function returns the value of the distance-dependend dielectric function.
//(Whole function copied from AutoDock...)
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

int is_H_bond(const char* atype1, const char* atype2)
//Returns 1 if a H-bond can exist between the atoms with atom code atype1 and atype2,
//otherwise it returns 0.
{
	if  (	//H-bond
		(((strcmp(atype1, "HD") == 0) || (strcmp(atype1, "HS") == 0)) && //HD or HS
		( (strcmp(atype2, "NA") == 0) ||
		  (strcmp(atype2, "NS") == 0) ||
		  (strcmp(atype2, "OA") == 0) ||
		  (strcmp(atype2, "OS") == 0) ||
		  (strcmp(atype2, "SA") == 0) ))		//NA NS OA OS or SA
		||
		(((strcmp(atype2, "HD") == 0) || (strcmp(atype2, "HS") == 0)) && //HD or HS
		( (strcmp(atype1, "NA") == 0) ||
		  (strcmp(atype1, "NS") == 0) ||
		  (strcmp(atype1, "OA") == 0) ||
		  (strcmp(atype1, "OS") == 0) ||
		  (strcmp(atype1, "SA") == 0) ))		//NA NS OA OS or SA
		)
		return 1;
	else
		return 0;
}

#if 0
void print_ref_lig_energies_f(Liganddata myligand,
			      Gridinfo mygrid,
			      const float* fgrids,
			      const float scaled_AD4_coeff_elec,
			      const float AD4_coeff_desolv,
			      const float qasp)
//The function calculates the energies of the ligand given in the first parameter,
//and prints them to the screen.
{
	double temp_vec [3];
	int i;

	printf("Intramolecular energy of reference ligand: %lf\n",
		calc_intraE_f(&myligand, 8, 0, scaled_AD4_coeff_elec, AD4_coeff_desolv, qasp, 0));

	for (i=0; i<3; i++)
		temp_vec [i] = -1*mygrid.origo_real_xyz [i];

	move_ligand(&myligand, temp_vec);
	scale_ligand(&myligand, (double) 1.0/mygrid.spacing);

	printf("Intermolecular energy of reference ligand: %lf\n",
		calc_interE_f(&mygrid, &myligand, fgrids, 0, 0));
}
#endif

void print_ref_lig_energies_f(Liganddata   myligand,
			      const float  smooth,
			      Gridinfo     mygrid,
			      const float* fgrids,
			      const float  scaled_AD4_coeff_elec,
			      const float  AD4_coeff_desolv,
			      const float  qasp)
//The function calculates the energies of the ligand given in the first parameter,
//and prints them to the screen.
{
	double temp_vec [3];
	int i;

	IntraTables tables(&myligand, scaled_AD4_coeff_elec, AD4_coeff_desolv, qasp);
	printf("Intramolecular energy of reference ligand: %lf\n",
		calc_intraE_f(&myligand, 8, smooth, 0, tables, 0));

	for (i=0; i<3; i++)
		temp_vec [i] = -1*mygrid.origo_real_xyz [i];

	move_ligand(&myligand, temp_vec);
	scale_ligand(&myligand, (double) 1.0/mygrid.spacing);

	printf("Intermolecular energy of reference ligand: %lf\n",
		calc_interE_f(&mygrid, &myligand, fgrids, 0, 0));
}

//////////////////////////////////
//float functions

void calc_distdep_tables_f(float r_6_table [],
			   float r_10_table [],
			   float r_12_table [],
			   float r_epsr_table [],
			   float desolv_table [],
			   const float scaled_AD4_coeff_elec,
			   const float AD4_coeff_desolv)
//The function fills the input arrays with the following functions:
//1/r^6, 1/r^10, 1/r^12, W_el/(r*eps(r)) and W_des*exp(-r^2/(2sigma^2))
//for distances 0.01, 0.02, ..., 20.48 A
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

void calc_q_tables_f(const Liganddata* myligand,
		     float qasp,
		     #if 0
		     float q1q2[][256],
		     #endif
		     float q1q2[][MAX_NUM_OF_ATOMS],
		     float qasp_mul_absq [])
//The function calculates q1*q2 and qasp*abs(q) values
//based on the myligand parameter.
{
	int i, j;

	for (i=0; i < myligand->num_of_atoms; i++)
		for (j=0; j < myligand->num_of_atoms; j++)
			q1q2 [i][j] = (float) myligand->atom_idxyzq [i][4] * myligand->atom_idxyzq [j][4];

	for (i=0; i < myligand->num_of_atoms; i++)
		qasp_mul_absq [i] = qasp*fabs(myligand->atom_idxyzq [i][4]);

}

#if 1
void change_conform_f(Liganddata* myligand,
		      const float genotype_f [],
		      float*      cpu_ref_ori_angles,
		      int         debug)
//The function changes the conformation of myligand according to
//the genotype given by the second parameter.
{
	double genrot_movvec [3] = {0, 0, 0};
	double genrot_unitvec [3];
	double movvec_to_origo [3];
	double phi, theta;
	int atom_id, rotbond_id, i;
	double genotype [40];
	double refori_unitvec [3];
	double refori_angle;

	for (i=0; i<40; i++)
		genotype [i] = genotype_f [i];

	phi = (genotype [3])/180*PI;
	theta = (genotype [4])/180*PI;

	genrot_unitvec [0] = sin(theta)*cos(phi);
	genrot_unitvec [1] = sin(theta)*sin(phi);
	genrot_unitvec [2] = cos(theta);

	phi = (cpu_ref_ori_angles [0])/180*PI;
	theta = (cpu_ref_ori_angles [1])/180*PI;

	refori_unitvec [0] = sin(theta)*cos(phi);
	refori_unitvec [1] = sin(theta)*sin(phi);
	refori_unitvec [2] = cos(theta);
	refori_angle = cpu_ref_ori_angles[2];

// +++++++++++++++++++++++++++++++++++++++
//printf("cpu_ref_ori_angles [0]: %f, cpu_ref_ori_angles [1]: %f, %f\n",cpu_ref_ori_angles [0],cpu_ref_ori_angles [1],PI);
//printf("refori_unitvec [0]:%f, refori_unitvec [1]:%f, refori_unitvec [2]:%f\n",refori_unitvec [0],refori_unitvec [1],refori_unitvec [2]);
// +++++++++++++++++++++++++++++++++++++++

	get_movvec_to_origo(myligand, movvec_to_origo);	//moving ligand to origo
	move_ligand(myligand, movvec_to_origo);


	for (atom_id=0; atom_id < myligand->num_of_atoms; atom_id++)	//for each atom of the ligand
	{
		if (debug == 1)
			printf("\n\n\nROTATING atom %d ", atom_id);

		if (myligand->num_of_rotbonds != 0)			//if the ligand has rotatable bonds
		{
			for (rotbond_id=0; rotbond_id < myligand->num_of_rotbonds; rotbond_id++)	//for each rotatable bond
				if (myligand->atom_rotbonds[atom_id][rotbond_id] != 0)			//if the atom has to be rotated around this bond
				{
					if (debug == 1)
						printf("around rotatable bond %d\n", rotbond_id);

					rotate(&(myligand->atom_idxyzq[atom_id][1]),
					       myligand->rotbonds_moving_vectors[rotbond_id],
					       myligand->rotbonds_unit_vectors[rotbond_id],
					       &(genotype [6+rotbond_id]), /*debug*/0);	//rotating
				}
		}

		if (debug == 1)
			printf("according to general rotation\n");

		rotate(&(myligand->atom_idxyzq[atom_id][1]),
		       genrot_movvec,
		       refori_unitvec,
		       &refori_angle, debug);		//rotating to reference oritentation

		rotate(&(myligand->atom_idxyzq[atom_id][1]),
		       genrot_movvec,
		       genrot_unitvec,
		       &(genotype [5]), debug);		//general rotation
	}

	move_ligand(myligand, genotype);

	if (debug == 1)
		for (atom_id=0; atom_id < myligand->num_of_atoms; atom_id++)
			printf("Moved point (final values) (x,y,z): %lf, %lf, %lf\n", myligand->atom_idxyzq [atom_id][1], myligand->atom_idxyzq [atom_id][2], myligand->atom_idxyzq [atom_id][3]);

}
#endif // End of original change_conform_f()

#if 0
// -------------------------------------------------------------------
// Replacing rotation genes: from spherical space to Shoemake space
// gene [0:2]: translation -> kept as original x, y, z
// gene [3:5]: rotation    -> transformed into Shoemake (u1: adimensional, u2&u3: sexagesimal)
// gene [6:N]: torsions	   -> kept as original angles	(all in sexagesimal)

// Shoemake ranges:
// u1: [0, 1]
// u2: [0: 2PI] or [0: 360]

// Random generator in the host is changed:
// LCG (original, myrand()) -> CPP std (rand())
// -------------------------------------------------------------------
void change_conform_f(Liganddata* myligand,
		      const float genotype_f [],
		      float* cpu_ref_ori_angles,
		      int debug)
//The function changes the conformation of myligand according to
//the genotype given by the second parameter.
{
	double genrot_movvec [3] = {0, 0, 0};

	double shoemake [3] = {0, 0, 0};

// Replaced by shoemake [3]
/*
	double genrot_unitvec [3];
*/

	double movvec_to_origo [3];
/*
	double phi, theta;
*/
	int atom_id, rotbond_id, i;
	double genotype [40];

	double refori_shoemake [3];

// Replaced by refori_shoemake [3]
/*
	double refori_unitvec [3];
*/


/*
	double refori_angle;
*/


	for (i=0; i<40; i++)
		genotype [i] = genotype_f [i];

	shoemake [0] = (genotype [3]);
	shoemake [1] = (genotype [4])*(2*PI);
	shoemake [2] = (genotype [5])*(2*PI);

	refori_shoemake [0] = (cpu_ref_ori_angles [0]);
	refori_shoemake [1] = (cpu_ref_ori_angles [1])*(2*PI);
	refori_shoemake [2] = (cpu_ref_ori_angles [2])*(2*PI);

// +++++++++++++++++++++++++++++++++++++++
//printf("cpu_ref_ori_angles [0]: %f, cpu_ref_ori_angles [1]: %f, %f\n",cpu_ref_ori_angles [0],cpu_ref_ori_angles [1],PI);
//printf("refori_unitvec [0]:%f, refori_unitvec [1]:%f, refori_unitvec [2]:%f\n",refori_unitvec [0],refori_unitvec [1],refori_unitvec [2]);
// +++++++++++++++++++++++++++++++++++++++

	get_movvec_to_origo(myligand, movvec_to_origo);	//moving ligand to origo
	move_ligand(myligand, movvec_to_origo);


	for (atom_id=0; atom_id < myligand->num_of_atoms; atom_id++)						//for each atom of the ligand
	{
		if (debug == 1)
			printf("\n\n\nROTATING atom %d ", atom_id);

		if (myligand->num_of_rotbonds != 0)											//if the ligand has rotatable bonds
		{
			for (rotbond_id=0; rotbond_id < myligand->num_of_rotbonds; rotbond_id++)	//for each rotatable bond
				if (myligand->atom_rotbonds[atom_id][rotbond_id] != 0)				//if the atom has to be rotated around this bond
				{
					if (debug == 1)
						printf("around rotatable bond %d\n", rotbond_id);

					rotate(&(myligand->atom_idxyzq[atom_id][1]),
					       myligand->rotbonds_moving_vectors[rotbond_id],
					       myligand->rotbonds_unit_vectors[rotbond_id],
					       &(genotype [6+rotbond_id]), /*debug*/0);	//rotating
				}
		}

		if (debug == 1)
			printf("according to general rotation\n");

		rotate_shoemake(&(myligand->atom_idxyzq[atom_id][1]),
	  		        genrot_movvec,
		       		refori_shoemake,
		       		debug);		//rotating to reference oritentation

		rotate_shoemake(&(myligand->atom_idxyzq[atom_id][1]),
			       genrot_movvec,
			       shoemake,
			       debug);		//general rotation
	}

	move_ligand(myligand, genotype);

	if (debug == 1)
		for (atom_id=0; atom_id < myligand->num_of_atoms; atom_id++)
			printf("Moved point (final values) (x,y,z): %lf, %lf, %lf\n", myligand->atom_idxyzq [atom_id][1], myligand->atom_idxyzq [atom_id][2], myligand->atom_idxyzq [atom_id][3]);

}
#endif

float calc_interE_f(const Gridinfo*   mygrid,
		    const Liganddata* myligand,
		    const float*      fgrids,
		    float             outofgrid_tolerance,
		    int               debug)
//The function calculates the intermolecular energy of a ligand (given by myligand parameter),
//and a receptor (represented as a grid). The grid point values must be stored at the location
//which starts at fgrids, the memory content can be generated with get_gridvalues funciton.
//The mygrid parameter must be the corresponding grid informtaion. If an atom is outside the
//grid, the coordinates will be changed with the value of outofgrid_tolerance, if it remains
//outside, a very high value will be added to the current energy as a penality. If the fifth
//parameter is one, debug messages will be printed to the screen during calculation.
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

	interE = 0;


	for (atom_cnt=myligand->num_of_atoms-1; atom_cnt>=0; atom_cnt--)		//for each atom
	{
		atomtypeid = myligand->atom_idxyzq [atom_cnt][0];
		x = myligand->atom_idxyzq [atom_cnt][1];
		y = myligand->atom_idxyzq [atom_cnt][2];
		z = myligand->atom_idxyzq [atom_cnt][3];
		q = myligand->atom_idxyzq [atom_cnt][4];

		if ((x < 0) || (x >= mygrid->size_xyz [0]-1) || (y < 0) || (y >= mygrid->size_xyz [1]-1) ||
			(z < 0) || (z >= mygrid->size_xyz [2]-1))		//if the atom is outside of the grid
		{
			if (debug == 1)
			{
				printf("\n\nPartial results for atom with id %d:\n", atom_cnt);
				printf("Atom out of grid: ");
				printf("x= %lf, y = %lf, z = %lf\n", x, y, z);
			}

			if (outofgrid_tolerance != 0)	//if tolerance is set, try to place atom back into the grid
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
						(z < 0) || (z >= mygrid->size_xyz [2]-1))		//check again if the atom is outside of the grid
			{
				//interE = HIGHEST_ENERGY;	//return maximal value
				//return interE;
				interE += 16777216;	//penalty is 2^24 for each atom outside the grid
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

		//energy contribution of the current grid type

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


		interE += trilin_interpol(cube, weights);

		if (debug == 1)
			printf("interpoated value = %lf\n\n", trilin_interpol(cube, weights));

		//energy contribution of the electrostatic grid

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


		interE += q * trilin_interpol(cube, weights);

		if (debug == 1)
			printf("interpoated value = %lf, multiplied by q = %lf\n\n", trilin_interpol(cube, weights), q*trilin_interpol(cube, weights));

		//energy contribution of the desolvation grid

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

		interE += fabs(q) * trilin_interpol(cube, weights);

		if (debug == 1)
			printf("interpoated value = %lf, multiplied by abs(q) = %lf\n\n", trilin_interpol(cube, weights), fabs(q) * trilin_interpol(cube, weights));

		if (debug == 1)
			printf("Current value of intermolecular energy = %lf\n\n\n", interE);
	}

	return interE;
}

void calc_interE_peratom_f(const Gridinfo* mygrid,
	                   const Liganddata* myligand,
			   const float* fgrids,
			   float  outofgrid_tolerance,
			   float* elecE,
			   float  peratom_vdw [MAX_NUM_OF_ATOMS],
			   float  peratom_elec [MAX_NUM_OF_ATOMS],
			   int    debug)
//
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
		atomtypeid = myligand->atom_idxyzq [atom_cnt][0];
		x = myligand->atom_idxyzq [atom_cnt][1];
		y = myligand->atom_idxyzq [atom_cnt][2];
		z = myligand->atom_idxyzq [atom_cnt][3];
		q = myligand->atom_idxyzq [atom_cnt][4];

		if ((x < 0) || (x >= mygrid->size_xyz [0]-1) || (y < 0) || (y >= mygrid->size_xyz [1]-1) ||
			(z < 0) || (z >= mygrid->size_xyz [2]-1))		//if the atom is outside of the grid
		{
			if (debug == 1)
			{
				printf("\n\nPartial results for atom with id %d:\n", atom_cnt);
				printf("Atom out of grid: ");
				printf("x= %lf, y = %lf, z = %lf\n", x, y, z);
			}

			if (outofgrid_tolerance != 0)	//if tolerance is set, try to place atom back into the grid
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
						(z < 0) || (z >= mygrid->size_xyz [2]-1))		//check again if the atom is outside of the grid
			{
				//interE = HIGHEST_ENERGY;	//return maximal value
				//return interE;
				//interE += 16777216;	//penalty is 2^24 for each atom outside the grid
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

		//energy contribution of the current grid type

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
			printf("interpoated value = %lf\n\n", trilin_interpol(cube, weights));

		//energy contribution of the electrostatic grid

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
			printf("interpoated value = %lf, multiplied by q = %lf\n\n", trilin_interpol(cube, weights), q*trilin_interpol(cube, weights));

/*		//energy contribution of the desolvation grid

		typeid = mygrid->num_of_atypes+1;

		cube [0][0][0] = getvalue_4Darr(fgrids, *mygrid, typeid, z_low, y_low, x_low);
		cube [1][0][0] = getvalue_4Darr(fgrids, *mygrid, typeid, z_low, y_low, x_high);
		cube [0][1][0] = getvalue_4Darr(fgrids, *mygrid, typeid, z_low, y_high, x_low);
		cube [1][1][0] = getvalue_4Darr(fgrids, *mygrid, typeid, z_low, y_high, x_high);
		cube [0][0][1] = getvalue_4Darr(fgrids, *mygrid, typeid, z_high, y_low, x_low);
		cube [1][0][1] = getvalue_4Darr(fgrids, *mygrid, typeid, z_high, y_low, x_high);
		cube [0][1][1] = getvalue_4Darr(fgrids, *mygrid, typeid, z_high, y_high, x_low);
		cube [1][1][1] = getvalue_4Darr(fgrids, *mygrid, typeid, z_high, y_high, x_high);

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

		interE += fabs(q) * trilin_interpol(cube, weights);

		if (debug == 1)
			printf("interpoated value = %lf, multiplied by abs(q) = %lf\n\n", trilin_interpol(cube, weights), fabs(q) * trilin_interpol(cube, weights));

		if (debug == 1)
			printf("Current value of intermolecular energy = %lf\n\n\n", interE);*/
	}

	//return interE;
}

// Original host "calc_intraE_f" function
#if 0
float calc_intraE_f(const Liganddata* myligand,
		    float dcutoff,
		    char ignore_desolv,
		    const float scaled_AD4_coeff_elec,
		    const float AD4_coeff_desolv,
		    const float qasp, int debug)
//The function calculates the intramolecular energy of the ligand given by the first parameter,
//and returns it as a double. The second parameter is the distance cutoff, if the third isn't 0,
//desolvation energy won't be included by the energy value, the fourth indicates if messages
//about partial results are required (if debug=1)
{

	int atom_id1, atom_id2;
	int type_id1, type_id2;
	float dist;
	int distance_id;
	float vdW1, vdW2;
	float s1, s2, v1, v2;

	float vW, el, desolv;

	//The following tables will contain the 1/r^6, 1/r^10, 1/r^12, W_el/(r*eps(r)) and W_des*exp(-r^2/(2sigma^2)) functions for
	//distances 0.01:0.01:20.48 A
	static char first_call = 1;
	static float r_6_table [2048];
	static float r_10_table [2048];
	static float r_12_table [2048];
	static float r_epsr_table [2048];
	static float desolv_table [2048];

	//The following arrays will contain the q1*q2 and qasp*abs(q) values for the ligand which is the input parameter when this
	//function is called first time (it is supposed that the energy must always be calculated for this ligand only, that is, there
	//is only one ligand during the run of the program...)
	static float q1q2 [256][256];
	static float qasp_mul_absq [256];

	//when first call, calculating tables
	if (first_call == 1)
	{
		calc_distdep_tables_f(r_6_table, r_10_table, r_12_table, r_epsr_table, desolv_table, scaled_AD4_coeff_elec, AD4_coeff_desolv);
		calc_q_tables_f(myligand, qasp, q1q2, qasp_mul_absq);
		first_call = 0;
	}

	vW = 0;
	el = 0;
	desolv = 0;

	if (debug == 1)
		printf("\n\n\nINTRAMOLECULAR ENERGY CALCULATION\n\n");

	for (atom_id1=0; atom_id1<myligand->num_of_atoms-1; atom_id1++)	//for each atom pair
		for (atom_id2=atom_id1+1; atom_id2<myligand->num_of_atoms; atom_id2++)
		{
			if (myligand->intraE_contributors [atom_id1][atom_id2] == 1)	//if they have to be included in intramolecular energy calculation
			{															//the energy contribution has to be calculated
				dist = distance(&(myligand->atom_idxyzq [atom_id1][1]), &(myligand->atom_idxyzq [atom_id2][1]));

				if (dist <= 1)
				{
					if (debug == 1)
						printf("\n\nToo low distance (%lf) between atoms %d and %d\n", dist, atom_id1, atom_id2);

					//return HIGHEST_ENERGY;	//returning maximal value
					dist = 1;
				}

				if (debug == 1)
				{
					printf("\n\nCalculating energy contribution of atoms %d and %d\n", atom_id1+1, atom_id2+1);
					printf("Distance: %lf\n", dist);
				}

				if ((dist < dcutoff) && (dist < 20.48))	//but only if the distance is less than distance cutoff value and 20.48A (because of the tables)
				{
					type_id1 = myligand->atom_idxyzq [atom_id1][0];
					type_id2 = myligand->atom_idxyzq [atom_id2][0];

					distance_id = (int) floor((100*dist) + 0.5) - 1;	// +0.5: rounding, -1: r_xx_table [0] corresponds to r=0.01
					if (distance_id < 0)
						distance_id = 0;

					if (is_H_bond(myligand->atom_types [type_id1], myligand->atom_types [type_id2]) != 0)	//H-bond
					{
						vdW1 = myligand->VWpars_C [type_id1][type_id2]*r_12_table [distance_id];
						vdW2 = myligand->VWpars_D [type_id1][type_id2]*r_10_table [distance_id];
						if (debug == 1)
							printf("H-bond interaction = ");
					}
					else	//normal van der Waals
					{
						vdW1 = myligand->VWpars_A [type_id1][type_id2]*r_12_table [distance_id];
						vdW2 = myligand->VWpars_B [type_id1][type_id2]*r_6_table [distance_id];
						if (debug == 1)
							printf("van der Waals interaction = ");
					}

					s1 = (myligand->solpar [type_id1] + qasp_mul_absq [atom_id1]);
					s2 = (myligand->solpar [type_id2] + qasp_mul_absq [atom_id2]);
					v1 = myligand->volume [type_id1];
					v2 = myligand->volume [type_id2];

					if (debug == 1)
						printf(" %lf, electrostatic = %lf, desolv = %lf\n", (vdW1 - vdW2), q1q2[atom_id1][atom_id2] * r_epsr_table [distance_id],
							   (s1*v2 + s2*v1) * desolv_table [distance_id]);

					vW += vdW1 - vdW2;
					el += q1q2[atom_id1][atom_id2] * r_epsr_table [distance_id];
					desolv += (s1*v2 + s2*v1) * desolv_table [distance_id];
				}
			}
		}

	if (debug == 1)
		printf("\nFinal energies: van der Waals = %lf, electrostatic = %lf, desolvation = %lf, total = %lf\n\n", vW, el, desolv, vW + el + desolv);

	if (ignore_desolv == 0)
		return (vW + el + desolv);
	else
		return (vW + el);
}
#endif

// Corrected host "calc_intraE_f" function after smoothing was added
float calc_intraE_f(const Liganddata* myligand,
		          float       dcutoff,
		          float       smooth,
		          char        ignore_desolv,
			  IntraTables& tables,
		          int         debug)
//The function calculates the intramolecular energy of the ligand given by the first parameter,
//and returns it as a double. The second parameter is the distance cutoff, if the third isn't 0,
//desolvation energy won't be included by the energy value, the fourth indicates if messages
//about partial results are required (if debug=1)
{

	int atom_id1, atom_id2;
	int type_id1, type_id2;
	float dist;
	int distance_id;
	int smoothed_distance_id;
	float vdW1, vdW2;
	float s1, s2, v1, v2;

	float vW, el, desolv;

	vW = 0;
	el = 0;
	desolv = 0;
	
	if (debug == 1)
		printf("\n\n\nINTRAMOLECULAR ENERGY CALCULATION\n\n");

	for (atom_id1=0; atom_id1<myligand->num_of_atoms-1; atom_id1++)	//for each atom pair
	{
		for (atom_id2=atom_id1+1; atom_id2<myligand->num_of_atoms; atom_id2++)
		{
			if (myligand->intraE_contributors [atom_id1][atom_id2] == 1)	//if they have to be included in intramolecular energy calculation
			{															//the energy contribution has to be calculated
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

				unsigned int atom1_type_vdw_hb = myligand->atom1_types_reqm [type_id1];
				unsigned int atom2_type_vdw_hb = myligand->atom2_types_reqm [type_id2];

				// Getting optimum pair distance (opt_distance) from reqm and reqm_hbond
				// reqm: equilibrium internuclear separation
				//       (sum of the vdW radii of two like atoms (A)) in the case of vdW
				// reqm_hbond: equilibrium internuclear separation
				// 	 (sum of the vdW radii of two like atoms (A)) in the case of hbond
				float opt_distance;

				if (is_H_bond(myligand->atom_types [type_id1], myligand->atom_types [type_id2]) != 0)	//H-bond
				{
					opt_distance = myligand->reqm_hbond [atom1_type_vdw_hb] + myligand->reqm_hbond [atom2_type_vdw_hb];
				}
				else	//normal van der Waals
				{
					opt_distance = 0.5f*(myligand->reqm [atom1_type_vdw_hb] + myligand->reqm [atom2_type_vdw_hb]);
				}

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

				distance_id = (int) floor((100*dist) + 0.5) - 1;	// +0.5: rounding, -1: r_xx_table [0] corresponds to r=0.01
				if (distance_id < 0) {
					distance_id = 0;
				}

				smoothed_distance_id = (int) floor((100*smoothed_distance) + 0.5) - 1;	// +0.5: rounding, -1: r_xx_table [0] corresponds to r=0.01
				if (smoothed_distance_id < 0) {
					smoothed_distance_id = 0;
				}

				if (dist < dcutoff) //but only if the distance is less than distance cutoff value
				{
					if (is_H_bond(myligand->atom_types [type_id1], myligand->atom_types [type_id2]) != 0)	//H-bond
					{
						vdW1 = myligand->VWpars_C [type_id1][type_id2]*tables.r_12_table [smoothed_distance_id];
						vdW2 = myligand->VWpars_D [type_id1][type_id2]*tables.r_10_table [smoothed_distance_id];
						if (debug == 1)
							printf("H-bond interaction = ");
					}
					else	//normal van der Waals
					{
						vdW1 = myligand->VWpars_A [type_id1][type_id2]*tables.r_12_table [smoothed_distance_id];
						vdW2 = myligand->VWpars_B [type_id1][type_id2]*tables.r_6_table  [smoothed_distance_id];
						if (debug == 1)
							printf("van der Waals interaction = ");
					}

					vW += vdW1 - vdW2;
				}

				if (dist < 20.48)
				{
					s1 = (myligand->solpar [type_id1] + tables.qasp_mul_absq [atom_id1]);
					s2 = (myligand->solpar [type_id2] + tables.qasp_mul_absq [atom_id2]);
					v1 = myligand->volume [type_id1];
					v2 = myligand->volume [type_id2];

					if (debug == 1)
						printf(" %lf, electrostatic = %lf, desolv = %lf\n", (vdW1 - vdW2), tables.q1q2[atom_id1][atom_id2] * tables.r_epsr_table [distance_id],
							   (s1*v2 + s2*v1) * tables.desolv_table [distance_id]);


					el += tables.q1q2[atom_id1][atom_id2] * tables.r_epsr_table [distance_id];
					desolv += (s1*v2 + s2*v1) * tables.desolv_table [distance_id];
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
					vW += G * dist;
					/*printf("OpenCL host - calc_intraE_f: CG-G0 pair found!\n");*/
				}
				// ------------------------------------------------
			}
		}
	}

	if (debug == 1)
		printf("\nFinal energies: van der Waals = %lf, electrostatic = %lf, desolvation = %lf, total = %lf\n\n", vW, el, desolv, vW + el + desolv);

	if (ignore_desolv == 0)
		return (vW + el + desolv);
	else
		return (vW + el);
}

int map_to_all_maps(Gridinfo* mygrid, Liganddata* myligand, std::vector<Map>& all_maps){
	for (int i_atom = 0; i_atom<myligand->num_of_atoms;i_atom++){
		int idx = myligand->atom_idxyzq[i_atom][0];
		int map_idx = -1;
		for (int i_map = 0; i_map<all_maps.size(); i_map++){
			if (strcmp(all_maps[i_map].atype.c_str(),mygrid->grid_types[idx])==0){
				map_idx = i_map;
				break;
			}
		}
		if (map_idx == -1) {printf("\nERROR: Did not map to all_maps correctly."); return 1;}

		myligand->atom_map_to_fgrids[i_atom] = map_idx;
		//printf("\nMapping atom %d (type %d) in the ligand to map #%d",i_atom,idx,map_idx);
	}

	return 0;
}
