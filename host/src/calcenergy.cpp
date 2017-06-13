/*
 * (C) 2013. Evopro Innovation Kft.
 *
 * calcenergy.cu
 *
 *  Created on: 2010.04.20.
 *      Author: pechan.imre
 */

#include "calcenergy.h"

int prepare_const_fields_for_gpu(Liganddata*     myligand_reference,
                                 Dockpars*       mypars,
                                 float*          cpu_ref_ori_angles,
                                 kernelconstant* KerConst)
//The function fills the constant memory field of the GPU (ADM FPGA)
//defined above (erased from here) and used during GPU docking,
//based on the parameters which describe the ligand,
//the docking parameters and the reference orientation angles.
//Short description of the field is as follows:

//atom_charges_const: stores the ligand atom charges.
//		      Element i corresponds to atom with atom ID i in myligand_reference.
//atom_types_const: stores the ligand atom type IDs according to myligand_reference.
//		    Element i corresponds to atom with ID i in myligand_reference.
//intraE_contributors_const: each three contiguous items describe an intramolecular contributor.
//		         The first two elements store the atom ID of the contributors according to myligand_reference.
//		         The third element is 0, if no H-bond can occur between the two atoms, and 1, if it can.
//VWpars_AC_const: stores the A or C van der Waals parameters.
//                 The element i*MAX_NUM_OF_ATYPES+j and j*MAX_NUM_OF_ATYPES+i corresponds to A or C in case of
//		   H-bond for atoms with type ID i and j (according to myligand_reference).
//VWpars_BD_const: stores the B or D van der Waals parameters similar to VWpars_AC_const.
//dspars_S_const: stores the S desolvation parameters.
//		  The element i corresponds to the S parameter of atom with type ID i
//		  according to myligand_reference.
//rotlist_const: stores the data describing the rotations for conformation calculation.
//		 Each element describes one rotation, and the elements are in a proper order,
//               considering that NUM_OF_THREADS_PER_BLOCK rotations will be performed in
//		 parallel (that is, each block of contiguous NUM_OF_THREADS_PER_BLOCK pieces of elements describe rotations that can
//		 be performed simultaneously).
//		 One element is a 32 bit integer, with bit 0 in the LSB position.
//		 Bit 7-0 describe the atom ID of the atom to be rotated (according to myligand_reference).
//		 Bit 15-7 describe the rotatable bond ID of the bond around which the atom is to be rotated (if this is not a general rotation)
//				 (bond ID is according to myligand_reference).
//		 If bit 16 is 1, this is the first rotation of the atom.
//		 If bit 17 is 1, this is a general rotation (so rotbond ID has to be ignored).
//		 If bit 18 is 1, this is a "dummy" rotation, that is, no rotation can be performed in this cycle
//	         (considering the other rotations which are being carried out in this period).
//ref_coords_x_const: stores the x coordinates of the reference ligand atoms.
//		      Element i corresponds to the x coordinate of
//					  atom with atom ID i (according to myligand_reference).
//ref_coords_y_const: stores the y coordinates of the reference ligand atoms similarly to ref_coords_x_const.
//ref_coords_z_const: stores the z coordinates of the reference ligand atoms similarly to ref_coords_x_const.
//rotbonds_moving_vectors_const: stores the coordinates of rotatable bond moving vectors. Element i, i+1 and i+2 (where i%3=0)
//								 correspond to the moving vector coordinates x, y and z of rotbond ID i, respectively
//								 (according to myligand_reference).
//rotbonds_unit_vectors_const: stores the coordinates of rotatable bond unit vectors similarly to rotbonds_moving_vectors_const.
//ref_orientation_quats_const: stores the quaternions describing the reference orientations for each run. Element i, i+1, i+2
//							   and i+3 (where i%4=0) correspond to the quaternion coordinates q, x, y and z of reference
//							   orientation for run i, respectively.
{
	int i, j;
	int type_id1, type_id2;
	float* floatpoi;
	char* charpoi;
	float phi, theta, genrotangle;

	// ------------------------------
	float atom_charges[MAX_NUM_OF_ATOMS];
	char  atom_types[MAX_NUM_OF_ATOMS];
	char  intraE_contributors[3*MAX_INTRAE_CONTRIBUTORS];
	float VWpars_AC[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
	float VWpars_BD[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
	float dspars_S[MAX_NUM_OF_ATYPES];
	float dspars_V[MAX_NUM_OF_ATYPES];
	int   rotlist[MAX_NUM_OF_ROTATIONS];
	float ref_coords_x[MAX_NUM_OF_ATOMS];
	float ref_coords_y[MAX_NUM_OF_ATOMS];
	float ref_coords_z[MAX_NUM_OF_ATOMS];
	float rotbonds_moving_vectors[3*MAX_NUM_OF_ROTBONDS];
	float rotbonds_unit_vectors[3*MAX_NUM_OF_ROTBONDS];
	float ref_orientation_quats[4*MAX_NUM_OF_RUNS];
	// ------------------------------

	//charges and type id-s
	floatpoi = atom_charges;
	charpoi = atom_types;

	for (i=0; i < myligand_reference->num_of_atoms; i++)
	{
		*floatpoi = (float) myligand_reference->atom_idxyzq[i][4];
		*charpoi = (char) myligand_reference->atom_idxyzq[i][0];
		floatpoi++;
		charpoi++;
	}

	//intramolecular energy contributors
	myligand_reference->num_of_intraE_contributors = 0;
	for (i=0; i<myligand_reference->num_of_atoms-1; i++)
		for (j=i+1; j<myligand_reference->num_of_atoms; j++)
		{
			if (myligand_reference->intraE_contributors[i][j])
				myligand_reference->num_of_intraE_contributors++;
		}

	if (myligand_reference->num_of_intraE_contributors > MAX_INTRAE_CONTRIBUTORS)
	{
		printf("Error: number of intramolecular energy contributor is too high!\n");
		fflush(stdout);
		return 1;
	}

	charpoi = intraE_contributors;
	for (i=0; i<myligand_reference->num_of_atoms-1; i++)
		for (j=i+1; j<myligand_reference->num_of_atoms; j++)
		{
			if (myligand_reference->intraE_contributors[i][j] == 1)
			{
				*charpoi = (char) i;
				charpoi++;
				*charpoi = (char) j;
				charpoi++;

				type_id1 = (int) myligand_reference->atom_idxyzq [i][0];
				type_id2 = (int) myligand_reference->atom_idxyzq [j][0];
				if (is_H_bond(myligand_reference->atom_types[type_id1], myligand_reference->atom_types[type_id2]) != 0)
					*charpoi = (char) 1;
				else
					*charpoi = (char) 0;
				charpoi++;
			}
		}

	//van der Waals parameters
	for (i=0; i<myligand_reference->num_of_atypes; i++)
		for (j=0; j<myligand_reference->num_of_atypes; j++)
		{
			if (is_H_bond(myligand_reference->atom_types[i], myligand_reference->atom_types[j]) != 0)
			{
				floatpoi = VWpars_AC + i*myligand_reference->num_of_atypes + j;
				*floatpoi = (float) myligand_reference->VWpars_C[i][j];
				floatpoi = VWpars_AC + j*myligand_reference->num_of_atypes + i;
				*floatpoi = (float) myligand_reference->VWpars_C[j][i];

				floatpoi = VWpars_BD + i*myligand_reference->num_of_atypes + j;
				*floatpoi = (float) myligand_reference->VWpars_D[i][j];
				floatpoi = VWpars_BD + j*myligand_reference->num_of_atypes + i;
				*floatpoi = (float) myligand_reference->VWpars_D[j][i];
			}
			else
			{
				floatpoi = VWpars_AC + i*myligand_reference->num_of_atypes + j;
				*floatpoi = (float) myligand_reference->VWpars_A[i][j];
				floatpoi = VWpars_AC + j*myligand_reference->num_of_atypes + i;
				*floatpoi = (float) myligand_reference->VWpars_A[j][i];

				floatpoi = VWpars_BD + i*myligand_reference->num_of_atypes + j;
				*floatpoi = (float) myligand_reference->VWpars_B[i][j];
				floatpoi = VWpars_BD + j*myligand_reference->num_of_atypes + i;
				*floatpoi = (float) myligand_reference->VWpars_B[j][i];
			}
		}

	//desolvation parameters
	for (i=0; i<myligand_reference->num_of_atypes; i++)
	{
		dspars_S[i] = myligand_reference->solpar[i];
		dspars_V[i] = myligand_reference->volume[i];
	}

	//generate rotation list
	if (gen_rotlist(myligand_reference, rotlist) != 0)
	{
		printf("Error: number of required rotations is too high!\n");
		return 1;
	}

	//coordinates of reference ligand
	for (i=0; i < myligand_reference->num_of_atoms; i++)
	{
		ref_coords_x[i] = myligand_reference->atom_idxyzq[i][1];
		ref_coords_y[i] = myligand_reference->atom_idxyzq[i][2];
		ref_coords_z[i] = myligand_reference->atom_idxyzq[i][3];
	}

	//rotatable bond vectors
	for (i=0; i < myligand_reference->num_of_rotbonds; i++)
		for (j=0; j<3; j++)
		{
			rotbonds_moving_vectors[3*i+j] = myligand_reference->rotbonds_moving_vectors[i][j];
			rotbonds_unit_vectors[3*i+j] = myligand_reference->rotbonds_unit_vectors[i][j];
		}


	//reference orientation quaternions
	for (i=0; i<mypars->num_of_runs; i++)
	{
		//printf("Pregenerated angles for run %d: %f %f %f\n", i, cpu_ref_ori_angles[3*i], cpu_ref_ori_angles[3*i+1], cpu_ref_ori_angles[3*i+2]);

		phi = cpu_ref_ori_angles[3*i]*DEG_TO_RAD;
		theta = cpu_ref_ori_angles[3*i+1]*DEG_TO_RAD;
		genrotangle = cpu_ref_ori_angles[3*i+2]*DEG_TO_RAD;

		ref_orientation_quats[4*i] = cosf(genrotangle/2.0f);					//q
		ref_orientation_quats[4*i+1] = sinf(genrotangle/2.0f)*sinf(theta)*cosf(phi);		//x
		ref_orientation_quats[4*i+2] = sinf(genrotangle/2.0f)*sinf(theta)*sinf(phi);		//y
		ref_orientation_quats[4*i+3] = sinf(genrotangle/2.0f)*cosf(theta);			//z

		//printf("Precalculated quaternion for run %d: %f %f %f %f\n", i, ref_orientation_quats[4*i], ref_orientation_quats[4*i+1], ref_orientation_quats[4*i+2], ref_orientation_quats[4*i+3]);
	}

	int m;
	for (m=0;m<MAX_NUM_OF_ATOMS;m++){ KerConst->atom_charges_const[m] = atom_charges[m]; }
	for (m=0;m<MAX_NUM_OF_ATOMS;m++){ KerConst->atom_types_const[m]   = atom_types[m]; }
	for (m=0;m<3*MAX_INTRAE_CONTRIBUTORS;m++){ KerConst->intraE_contributors_const[m]   = intraE_contributors[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES;m++){ KerConst->VWpars_AC_const[m]   = VWpars_AC[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES;m++){ KerConst->VWpars_BD_const[m]   = VWpars_BD[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES;m++)		   { KerConst->dspars_S_const[m]    = dspars_S[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES;m++)		   { KerConst->dspars_V_const[m]    = dspars_V[m]; }
	for (m=0;m<MAX_NUM_OF_ROTATIONS;m++)		   { KerConst->rotlist_const[m]     = rotlist[m]; }
	for (m=0;m<MAX_NUM_OF_ATOMS;m++)		   { KerConst->ref_coords_x_const[m]= ref_coords_x[m]; }
	for (m=0;m<MAX_NUM_OF_ATOMS;m++)		   { KerConst->ref_coords_y_const[m]= ref_coords_y[m]; }
	for (m=0;m<MAX_NUM_OF_ATOMS;m++)		   { KerConst->ref_coords_z_const[m]= ref_coords_z[m]; }
	for (m=0;m<3*MAX_NUM_OF_ROTBONDS;m++){ KerConst->rotbonds_moving_vectors_const[m]= rotbonds_moving_vectors[m]; }
	for (m=0;m<3*MAX_NUM_OF_ROTBONDS;m++){ KerConst->rotbonds_unit_vectors_const[m]  = rotbonds_unit_vectors[m]; }
	for (m=0;m<4*MAX_NUM_OF_RUNS;m++)    { KerConst->ref_orientation_quats_const[m]  = ref_orientation_quats[m]; }

	return 0;
}



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
		printf("Rotation of %d (required rots remaining: %d)\n", atom_id_of_numrots[i], number_of_req_rotations[i]);
	printf("\n\n");*/


}



int gen_rotlist(Liganddata* myligand, int rotlist[MAX_NUM_OF_ROTATIONS])
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


	myligand->num_of_rotcyc = 0;
	myligand->num_of_rotations_required = 0;


	for (atom_id=0; atom_id<NUM_OF_THREADS_PER_BLOCK; atom_id++)	//handling special case when num_of_atoms<NUM_OF_THREADS_PER_BLOCK
		number_of_req_rotations[atom_id] = 0;

	for (atom_id=0; atom_id<myligand->num_of_atoms; atom_id++)
	{
		atom_id_of_numrots[atom_id] = atom_id;
		atom_wasnt_rotated_yet[atom_id] = 1;

		number_of_req_rotations[atom_id] = 1;

		for (rotb_id=0; rotb_id<myligand->num_of_rotbonds; rotb_id++)
			if (myligand->atom_rotbonds[atom_id][rotb_id] != 0)
				(number_of_req_rotations[atom_id])++;

		myligand->num_of_rotations_required += number_of_req_rotations[atom_id];
	}


	rotlist_id=0;
	make_reqrot_ordering(number_of_req_rotations, atom_id_of_numrots, myligand->num_of_atoms);
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
					rotbond_candidate = myligand->num_of_rotbonds - 1;
					remaining_rots_around_rotbonds = number_of_req_rotations[parallel_rot_id] - 1;	//-1 because of genrot

					while (rotbond_found == 0)
					{
						if (myligand->atom_rotbonds[atom_id][rotbond_candidate] != 0)	//if the atom has to be rotated around current candidate
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

		make_reqrot_ordering(number_of_req_rotations, atom_id_of_numrots, myligand->num_of_atoms);
		(myligand->num_of_rotcyc)++;
	}

	return 0;
}
