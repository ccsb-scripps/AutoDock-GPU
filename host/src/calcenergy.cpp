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


#include "calcenergy.h"

void make_reqrot_ordering(char number_of_req_rotations[MAX_NUM_OF_ATOMS],
			  char atom_id_of_numrots[MAX_NUM_OF_ATOMS],
			  int  num_of_atoms)
//The function puts the first array into a descending order and
//performs the same operations on the second array (since element i of
//number_or_req_rotations and element i of atom_id_of_numrots correspond to each other).
//Element i of the former array stores how many rotations have to be perfomed on the atom
//whose atom ID is stored by element i of the latter one. The third parameter has to be equal
//to the number of ligand atoms
{
	int i, j;
	char temp;

	for (j=0; j<num_of_atoms-1; j++)
		for (i=num_of_atoms-2; i>=j; i--)
			if (number_of_req_rotations[i+1] > number_of_req_rotations[i])
			{
				temp = number_of_req_rotations[i];
				number_of_req_rotations[i] = number_of_req_rotations[i+1];
				number_of_req_rotations[i+1] = temp;

				temp = atom_id_of_numrots[i];
				atom_id_of_numrots[i] = atom_id_of_numrots[i+1];
				atom_id_of_numrots[i+1] = temp;
			}

/*	printf("\n\nRotation priority list after re-ordering:\n");
	for (i=0; i<num_of_atoms; i++)
		printf("Roatom_rotbondstation of %d (required rots remaining: %d)\n", atom_id_of_numrots[i], number_of_req_rotations[i]);
	printf("\n\n");*/


}



int gen_rotlist(Liganddata& myligand, int rotlist[MAX_NUM_OF_ROTATIONS])
//The function generates the rotation list which will be stored in the constant memory field rotlist_const by
//prepare_const_fields_for_gpu(). The structure of this array is described at that function.
{
	int atom_id, rotb_id, parallel_rot_id, rotlist_id;
	char number_of_req_rotations[MAX_NUM_OF_ATOMS];
	char atom_id_of_numrots[MAX_NUM_OF_ATOMS];
	char atom_wasnt_rotated_yet[MAX_NUM_OF_ATOMS];
	int new_rotlist_element;
	char rotbond_found;
	char rotbond_candidate;
	char remaining_rots_around_rotbonds;


	myligand.num_of_rotcyc = 0;
	myligand.num_of_rotations_required = 0;


	for (atom_id=0; atom_id<NUM_OF_THREADS_PER_BLOCK; atom_id++)	//handling special case when num_of_atoms<NUM_OF_THREADS_PER_BLOCK
		number_of_req_rotations[atom_id] = 0;

	for (atom_id=0; atom_id<myligand.num_of_atoms; atom_id++)
	{
		atom_id_of_numrots[atom_id] = atom_id;
		atom_wasnt_rotated_yet[atom_id] = 1;

		number_of_req_rotations[atom_id] = 1;

		for (rotb_id=0; rotb_id<myligand.num_of_rotbonds; rotb_id++)
			if (myligand.atom_rotbonds[atom_id][rotb_id] != 0)
				(number_of_req_rotations[atom_id])++;

		myligand.num_of_rotations_required += number_of_req_rotations[atom_id];
	}


	rotlist_id=0;
	make_reqrot_ordering(number_of_req_rotations, atom_id_of_numrots, myligand.num_of_atoms);
	while (number_of_req_rotations[0] != 0)	//if the atom with the most remaining rotations has to be rotated 0 times, done
	{
		if (rotlist_id == MAX_NUM_OF_ROTATIONS)
			return 1;

		//putting the NUM_OF_THREADS_PER_BLOCK pieces of most important rotations to the list
		for (parallel_rot_id=0; parallel_rot_id<NUM_OF_THREADS_PER_BLOCK; parallel_rot_id++)
		{
			if (number_of_req_rotations[parallel_rot_id] == 0)	//if the atom has not to be rotated anymore, dummy rotation
				new_rotlist_element = RLIST_DUMMY_MASK;
			else
			{
				atom_id = atom_id_of_numrots[parallel_rot_id];
				new_rotlist_element = ((int) atom_id) & RLIST_ATOMID_MASK;

				if (number_of_req_rotations[parallel_rot_id] == 1)
					new_rotlist_element |= RLIST_GENROT_MASK;
				else
				{
					rotbond_found = 0;
					rotbond_candidate = myligand.num_of_rotbonds - 1;
					remaining_rots_around_rotbonds = number_of_req_rotations[parallel_rot_id] - 1;	//-1 because of genrot

					while (rotbond_found == 0)
					{
						if (myligand.atom_rotbonds[atom_id][rotbond_candidate] != 0)	//if the atom has to be rotated around current candidate
						{
							if (remaining_rots_around_rotbonds == 1)	//if current value of remaining rots is 1, the proper rotbond is found
								rotbond_found = 1;
							else
								remaining_rots_around_rotbonds--;	//if not, decresing remaining rots (that is, skipping rotations which have to be performed later
						}

						if (rotbond_found == 0)
							rotbond_candidate--;

						if (rotbond_candidate < 0)
							return 1;
					}

					new_rotlist_element |= (((int) rotbond_candidate) << RLIST_RBONDID_SHIFT) & RLIST_RBONDID_MASK;
				}

				if (atom_wasnt_rotated_yet[atom_id] != 0)
					new_rotlist_element |= RLIST_FIRSTROT_MASK;

				//put atom_id's next rotation to rotlist
				atom_wasnt_rotated_yet[atom_id] = 0;
				(number_of_req_rotations[parallel_rot_id])--;
			}


			rotlist[rotlist_id] = new_rotlist_element;

			rotlist_id++;
		}

		make_reqrot_ordering(number_of_req_rotations, atom_id_of_numrots, myligand.num_of_atoms);
		(myligand.num_of_rotcyc)++;
	}
	return 0;
}
