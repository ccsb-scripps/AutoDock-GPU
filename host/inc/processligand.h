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


#ifndef PROCESSLIGAND_H_
#define PROCESSLIGAND_H_

#include "defines.h"
#include "processgrid.h"
#include "miscellaneous.h"

// expand the allowed bond length ranges by the BOND_LENGTH_TOLERANCE
#define BOND_LENGTH_TOLERANCE 0.1
#define set_minmax( a1, a2, min, max)  \
    mindist[(a1)][(a2)] = (min)-BOND_LENGTH_TOLERANCE;\
    maxdist[(a1)][(a2)] = (max)+BOND_LENGTH_TOLERANCE;\
    if((a1) != (a2)){\
        mindist[(a2)][(a1)] = mindist[(a1)][(a2)];\
        maxdist[(a2)][(a1)] = maxdist[(a1)][(a2)];\
    }

// Struct which contains ligand and flexres information.
typedef struct
{
// num_of_atoms:          Number of ligand/flexres atoms.
	int            num_of_atoms;
// true_ligand_atoms      Number of ligand atoms
	int            true_ligand_atoms;
// ignore_inter           Array to specify if moving atoms will interact with receptor
//                        This is used to prevent flexres atoms to interact unfavorably
	bool           ignore_inter          [MAX_NUM_OF_ATOMS];
// num_of_atypes:         Number of different atom types in the ligand/flexres.
	int            num_of_atypes;
// num_of_rotbonds:       Number of rotatable bonds in the ligand/flexres.
	int            num_of_rotbonds;
// true_ligand_rotbonds:  Number of rotatable bonds in the ligand only.
	int            true_ligand_rotbonds;
// atom_names:            Each row (first index) contain the ligand atom name
	char           atom_names            [MAX_NUM_OF_ATOMS][5];
// atom_types:            Each row (first index) contain an atom type (as two characters),
//                        the row index is equal to the atom type code.
	char           atom_types            [MAX_NUM_OF_ATOMS][4]; // there can be at most as many types (base+derived) as there are atoms
// base_atom_types:       Each row (first index) contain an atom base type (for derived types it'll be different from atom_types),
//                        the row index is equal to the atom type code.
	char           base_atom_types       [MAX_NUM_OF_ATOMS][4];
	char           base_atom_names       [MAX_NUM_OF_ATOMS][4];
// atom_map_to_fgrids:    Maps each moving atom to a (pre-loaded) map id
	int            atom_map_to_fgrids    [MAX_NUM_OF_ATOMS];
// atom_idxyzq:           Each row describes one atom of the ligand.
//                        The columns (second index) contain the atom type code, x, y and z coordinate
//                        (in Angstroms) and electrical charge  of the atom.
//                        The row index is equal to the index of the current atom.
	double         atom_idxyzq           [MAX_NUM_OF_ATOMS][5];
// rotbonds:              Each row corresponds to one rotatable bond of the ligand.
//                        The rotatable bond is described with the indexes of the
//                        two atoms which are connected to each other by the bond.
//                        The row index is equal to the index of the rotatable bond.
	int            rotbonds              [MAX_NUM_OF_ROTBONDS][2];
// atom_rotbonds:         The array contains the rotatable bonds - atoms assignment.
//                        If the element atom_rotbonds[atom index][rotatable bond index] is equal to 1,
//                        it means,that the atom must be rotated if the bond rotates.
//                        A 0 means the opposite.
	char           atom_rotbonds         [MAX_NUM_OF_ATOMS][MAX_NUM_OF_ROTBONDS];
// atom_rigid_structures: The array contains the ridig structure ID of the atoms.
//                        If the atom_rigid_structures[atom1 index] = atom_rigid_structures[atom2 index],
//                        it means, that the atoms are in the same rigid molecule fragment.
//                        Rigid structures are seperated by branches in the pdbqt file.
	int            atom_rigid_structures [MAX_NUM_OF_ATOMS];
// bonds:                 If the element bonds[atom index][atom index] is equal to 1,
//                        a bond exists between the atoms.
	char           bonds                 [MAX_NUM_OF_ATOMS][MAX_NUM_OF_ATOMS];
// intraE_contributors:   If the element [atom index][atom index] is equal to 1,
//                        the energy contribution of the two atoms must be included in
//                        intramolecular energy calculation,
//                        which requires that there is at least one rotatable bond between the atoms and
//                        their interaction type is not 1-2, 1-3 or 1-4
	char           intraE_contributors   [MAX_NUM_OF_ATOMS][MAX_NUM_OF_ATOMS];
// base_type_idx:         Base type idx for each atom (is used for derived types)
	int            base_type_idx         [MAX_NUM_OF_ATOMS];
// atom_types_reqm:       Translates system atom types (including derived types) into base atom type indices
//                        This is needed as base atom type indices are static (for example: CG is always #22)
//                        while system type indices dynamic (i.e. in order of appearance in the pdbqt)
	unsigned int   atom_types_reqm       [MAX_NUM_OF_ATYPES];
// reqm_AB:               equilibrium distance (r_eqm) for each pair of system atom types
	double         reqm_AB               [MAX_NUM_OF_ATYPES][MAX_NUM_OF_ATYPES];
// VWpars_[A,B,C,D]:      The matrices contain the A, B, C and D Van der Waals parameters for the ligand atom types.
//                        The VWpars_A [atom type code of atom 1][atom type code of atom 2] element is equal
//                        to the A parameter for these types, the VWpars_B contains the B parameters in the
//                        same manner etc. (C and D are used for hydrogen bonds)
	double         VWpars_A              [MAX_NUM_OF_ATYPES][MAX_NUM_OF_ATYPES];
	double         VWpars_B              [MAX_NUM_OF_ATYPES][MAX_NUM_OF_ATYPES];
	double         VWpars_C              [MAX_NUM_OF_ATYPES][MAX_NUM_OF_ATYPES];
	double         VWpars_D              [MAX_NUM_OF_ATYPES][MAX_NUM_OF_ATYPES];
// VWpars_exp:            Contains the exponents of the vdW potential A/r^m - B/r^n in one
//                        2-byte unsigned integer, the upper byte is m and the lower one n
//                        m = (exp & 0xFF00)>>8
//                        n = (exp & 0xFF)
	unsigned short VWpars_exp            [MAX_NUM_OF_ATYPES][MAX_NUM_OF_ATYPES];
// volume, solpar:        these fields are similar to VWpars_x-s,
//                        they contain the atoms' volume and solvation parameter.
	double         volume                [MAX_NUM_OF_ATYPES];
	double         solpar                [MAX_NUM_OF_ATYPES];
	int            num_of_rotations_required;
	int            num_of_intraE_contributors;
	int            num_of_rotcyc;
// rotbonds_moving_vectors,
// rotbonds_unit_vectors: the vectors required for rotation around the
//                        corresponding rotatable bond (the first index is
//                        the index of the rotatable bond). When rotating
//                        a point around the bond with ID i, first it must be moved
//                        according to rotbonds_moving_vectors [i], then must be rotated
//                        around the line which crosses the origo and is parallel to
//                        rotbonds_unit_vectors [i], then it must be moved back with
//                        -1*rotbonds_moving_vectors [i].
//                        WARNING: after calling calc_conform for a Liganddata structure,
//                                 the orientation of the molecule will be changed, and these
//                                 vectors will be invalid, so they must be calculated again if
//                                 it is necessary.
	double         rotbonds_moving_vectors [MAX_NUM_OF_ROTBONDS][3];
	double         rotbonds_unit_vectors   [MAX_NUM_OF_ROTBONDS][3];
// acceptor, donor indicates if a given atom is a Hydrogen acceptor or donor
	bool           acceptor                [MAX_NUM_OF_ATOMS];
	bool           donor                   [MAX_NUM_OF_ATOMS];
	bool           reactive                [MAX_NUM_OF_ATOMS]; // atoms with 1,4,7 numbered atom types
} Liganddata;

// structure to store relevant receptor atom data
// ATOM      1  N   SER A   1      -2.367   4.481 -16.909  1.00  1.00     0.185 N
typedef struct
{
	unsigned int id;           // 1
	char         name[5];      // "N"
	char         res_name[4];  // "SER"
	char         chain_id[2];  // "A"
	unsigned int res_id;       // 1
	float        x,y,z;        // -2.367, 4.481, -16.909
	char         atom_type[4]; // "N"
	bool         acceptor;
	bool         donor;
} ReceptorAtom;

typedef struct
{
	unsigned int type;     // 0 .. reactive, 1 .. hydrogen bond, 2 .. vdW
	unsigned int lig_id;   // ligand atom id
	const char*  lig_name; // ligand atom name
	unsigned int rec_id;   // receptor/flex res atom id
	const char*  rec_name; // receptor/flex res atom name
	const char*  residue;  // residue name
	unsigned int res_id;   // residue id
	const char*  chain;    // chain id
} AnalysisData;

int init_liganddata(
                    const char*,
                    const char*,
                          Liganddata*,
                          Gridinfo*,
                          int          nr_deriv_atypes,
                          deriv_atype* deriv_atypes,
                          bool         cgmaps
                   );

int set_liganddata_typeid(
                                Liganddata*,
                                int,
                          const char*
                         );

void get_intraE_contributors(Liganddata*);

int get_bonds(Liganddata*);

pair_mod* is_mod_pair(
                      const char* A,
                      const char* B,
                            int       nr_mod_atype_pairs,
                            pair_mod* mod_atype_pairs
                     );

int get_VWpars(
                     Liganddata*,
               const double,
               const double,
                     int          nr_deriv_atypes,
                     deriv_atype* deriv_atypes,
                     int          nr_mod_atype_pairs,
                     pair_mod*    mod_atype_pairs
              );

int get_moving_and_unit_vectors(Liganddata*);

int get_liganddata(
                   const char*,
                   const char*,
                         Liganddata*,
                   const double,
                   const double,
                         int          nr_deriv_atypes,
                         deriv_atype* deriv_atypes,
                         int          nr_mod_atype_pairs,
                         pair_mod*    mod_atype_pairs
                  );

int gen_new_pdbfile(const char*, const char*, const Liganddata*);

void get_movvec_to_origo(const Liganddata*, double []);

void move_ligand(Liganddata*, const double []);

void move_ligand(Liganddata*, const double [], const double []);

void scale_ligand(Liganddata*, const double);

double calc_rmsd(const Liganddata*, const Liganddata*, const bool);

double calc_ddd_Mehler_Solmajer(double);

bool is_H_acceptor(const char* atype);

bool is_H_bond(
               const char* atype1,
               const char* atype2
              );

void print_ref_lig_energies_f(
                                    Liganddata,
                              const float,
                                    Gridinfo,
                              const float*,
                              const float,
                              const float,
                              const float,
                              const float,
                                    int,
                                    pair_mod*
                             );

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
                          );

void calc_q_tables_f(
                     const Liganddata* myligand,
                           float       qasp,
                           float       q1q2 [][MAX_NUM_OF_ATOMS],
                           float       qasp_mul_absq []
                    );

void change_conform_f(
                            Liganddata* myligand,
                      const Gridinfo*   mygrid,
                      const float       genotype_f [],
                            int         debug
                     );

void change_conform(
                          Liganddata* myligand,
                    const Gridinfo*   mygrid,
                    const double      genotype [],
                    const double      axisangle[4],
                          int         debug
                   );

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
                                                 );

float calc_interE_f(
                    const Gridinfo*   mygrid,
                    const Liganddata* myligand,
                    const float*      fgrids,
                          float       outofgrid_tolerance,
                          int         debug,
                          float&      intraflexE
                   );

void calc_interE_peratom_f(
                           const Gridinfo*   mygrid,
                           const Liganddata* myligand,
                           const float*      fgrids,
                                 float       outofgrid_tolerance,
                                 float*      elecE,
                                 float       peratom_vdw [MAX_NUM_OF_ATOMS],
                                 float       peratom_elec[MAX_NUM_OF_ATOMS],
                                 int         debug
                          );

struct IntraTables{
	//The following tables will contain the 1/r^6, 1/r^10, 1/r^12, W_el/(r*eps(r)) and W_des*exp(-r^2/(2sigma^2)) functions for
	//distances 0.01:0.01:20.48 A
	float r_6_table    [2048];
	float r_10_table   [2048];
	float r_12_table   [2048];
	float r_epsr_table [2048];
	float desolv_table [2048];

	//The following arrays will contain the q1*q2 and qasp*abs(q) values for the ligand which is the input parameter when this
	//function is called first time (it is supposed that the energy must always be calculated for this ligand only, that is, there
	//is only one ligand during the run of the program...)
	float q1q2          [MAX_NUM_OF_ATOMS][MAX_NUM_OF_ATOMS];
	float qasp_mul_absq [MAX_NUM_OF_ATOMS];
	bool is_HB          [MAX_NUM_OF_ATYPES] [MAX_NUM_OF_ATYPES];

	// Fill intraE tables
	IntraTables(
	            const Liganddata* myligand,
	            const float       scaled_AD4_coeff_elec,
	            const float       AD4_coeff_desolv,
	            const float       qasp
	           )
	{
		calc_distdep_tables_f(r_6_table, r_10_table, r_12_table, r_epsr_table, desolv_table, scaled_AD4_coeff_elec, AD4_coeff_desolv);
		calc_q_tables_f(myligand, qasp, q1q2, qasp_mul_absq);
		for (int type_id1=0; type_id1<myligand->num_of_atypes; type_id1++)
			for (int type_id2=0; type_id2<myligand->num_of_atypes; type_id2++)
				is_HB [type_id1][type_id2] = (is_H_bond(myligand->atom_types [type_id1],
				                              myligand->atom_types [type_id2]) != 0);
	}
};

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
                          std::vector<AnalysisData> *analysis = NULL,
                    const ReceptorAtom*             flexres_atoms = NULL,
                          float                     R_cutoff = 2.1,
                          float                     H_cutoff = 3.7,
                          float                     V_cutoff = 4.2
                   );

int map_to_all_maps(
                    Gridinfo*         mygrid,
                    Liganddata*       myligand,
                    std::vector<Map>& all_maps
                   );

#endif /* PROCESSLIGAND_H_ */
