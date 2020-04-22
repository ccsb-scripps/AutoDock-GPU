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






#include "processresult.h"


void arrange_result(float* final_population, float* energies, const int pop_size)
//The function arranges the rows of the input array (first array index is considered to be the row
//index) according to the sum of [] [38] and [][39] elements, which can be used for arranging the
//genotypes of the final population according to the sum of energy values. Genotypes with lower
//energies will be placed at lower row indexes. The second parameter must be equal to the size of
//the population, the arrangement will be performed only on the first pop_size part of final_population.
{
	int i,j;
	float temp_genotype[GENOTYPE_LENGTH_IN_GLOBMEM];
	float temp_energy;

	for (j=0; j<pop_size-1; j++)
		for (i=pop_size-2; i>=j; i--)		//arrange according to sum of inter- and intramolecular energies
			if (energies[i] > energies[i+1])
			{
				memcpy(temp_genotype, final_population+i*GENOTYPE_LENGTH_IN_GLOBMEM, GENOTYPE_LENGTH_IN_GLOBMEM*sizeof(float));
				memcpy(final_population+i*GENOTYPE_LENGTH_IN_GLOBMEM, final_population+(i+1)*GENOTYPE_LENGTH_IN_GLOBMEM, GENOTYPE_LENGTH_IN_GLOBMEM*sizeof(float));
				memcpy(final_population+(i+1)*GENOTYPE_LENGTH_IN_GLOBMEM, temp_genotype, GENOTYPE_LENGTH_IN_GLOBMEM*sizeof(float));

				temp_energy = energies[i];
				energies[i] = energies[i+1];
				energies[i+1] = temp_energy;
			}
}


void write_basic_info(FILE* fp, const Liganddata* ligand_ref, const Dockpars* mypars, const Gridinfo* mygrid, const int* argc, char** argv)
//The function writes basic information (such as docking parameters) to the file whose file pointer is the first parameter of the function.
{
	char temp_filename [128];
	int i;

	fprintf(fp, "***********************************\n");
	fprintf(fp, "**    AUTODOCK-GPU REPORT FILE   **\n");
	fprintf(fp, "***********************************\n\n\n");

	//Writing out docking parameters

	fprintf(fp, "         DOCKING PARAMETERS        \n");
	fprintf(fp, "===================================\n\n");

	fprintf(fp, "Ligand file:                               %s\n", mypars->ligandfile);
	fprintf(fp, "Grid fld file:                             %s\n", mypars->fldfile);

	fprintf(fp, "Number of energy evaluations:              %ld\n", mypars->num_of_energy_evals);
	fprintf(fp, "Number of generations:                     %ld\n", mypars->num_of_generations);
	fprintf(fp, "Size of population:                        %ld\n", mypars->pop_size);
	fprintf(fp, "Rate of crossover:                         %lf%%\n", (double) mypars->crossover_rate);
	fprintf(fp, "Tournament selection probability limit:    %lf%%\n", (double) mypars->tournament_rate);
	fprintf(fp, "Rate of mutation:                          %lf%%\n", (double) mypars->mutation_rate);
	fprintf(fp, "Maximal allowed delta movement:            +/- %lfA\n", (double) mypars->abs_max_dmov*mygrid->spacing);
	fprintf(fp, "Maximal allowed delta angle:               +/- %lf\n\n", (double) mypars->abs_max_dang);

	fprintf(fp, "Rate of local search:                      %lf%%\n", mypars->lsearch_rate);

	fprintf(fp, "Maximal number of local search iterations: %ld\n", mypars->max_num_of_iters);
	fprintf(fp, "Rho lower bound:                           %lf\n", (double) mypars->rho_lower_bound);
	fprintf(fp, "Spread of local search delta movement:     %lfA\n", (double) mypars->base_dmov_mul_sqrt3*mygrid->spacing/sqrt(3.0));
	fprintf(fp, "Spread of local search delta angle:        %lf\n", (double) mypars->base_dang_mul_sqrt3/sqrt(3.0));
	fprintf(fp, "Limit of consecutive successes/failures:   %ld\n\n", mypars->cons_limit);

	fprintf(fp, "Unbound model:                             ");
	if (mypars->unbound_model == 0)
		fprintf(fp, "BOUND\n");
	else
		if (mypars->unbound_model == 1)
			fprintf(fp, "EXTENDED\n");
		else
			fprintf(fp, "COMPACT\n");

	fprintf(fp, "Number of pdb files to be generated:       %d\n", mypars->gen_pdbs);

	fprintf(fp, "Initial population:                        ");
	if (mypars->initpop_gen_or_loadfile == 0)
		fprintf(fp, "GENERATE\n");
	else
		fprintf(fp, "LOAD FROM FILE (initpop.txt)\n");

	fprintf(fp, "\n\nProgram call in command line was:          ");
	for (i=0; i<*argc; i++)
		fprintf(fp, "%s ", argv [i]);
	fprintf(fp, "\n\n\n");

	//Writing out receptor parameters

	fprintf(fp, "        RECEPTOR PARAMETERS        \n");
	fprintf(fp, "===================================\n\n");

	fprintf(fp, "Receptor name:                             %s\n", mygrid->receptor_name);
	fprintf(fp, "Number of grid points (x, y, z):           %d, %d, %d\n", mygrid->size_xyz [0], mygrid->size_xyz [1], mygrid->size_xyz [2]);
	fprintf(fp, "Grid size (x, y, z):                       %lf, %lf, %lfA\n", mygrid->size_xyz_angstr [0], mygrid->size_xyz_angstr [1], mygrid->size_xyz_angstr [2]);
	fprintf(fp, "Grid spacing:                              %lfA\n", mygrid->spacing);
	fprintf(fp, "\n\n");

	//Writing out ligand parameters

	strncpy(temp_filename, mypars->ligandfile, strlen(mypars->ligandfile) - 6);
	temp_filename [strlen(mypars->ligandfile) - 6] = '\0';

	fprintf(fp, "         LIGAND PARAMETERS         \n");
	fprintf(fp, "===================================\n\n");

	fprintf(fp, "Ligand name:                               %s\n", temp_filename);
	fprintf(fp, "Number of atoms:                           %d\n", ligand_ref->num_of_atoms);
	fprintf(fp, "Number of rotatable bonds:                 %d\n", ligand_ref->num_of_rotbonds);
	fprintf(fp, "Number of atom types:                      %d\n", ligand_ref->num_of_atypes);

	fprintf(fp, "Number of intraE contributors:             %d\n", ligand_ref->num_of_intraE_contributors);
	fprintf(fp, "Number of required rotations:              %d\n", ligand_ref->num_of_rotations_required);
	fprintf(fp, "Number of rotation cycles:                 %d\n", ligand_ref->num_of_rotcyc);

	fprintf(fp, "\n\n");
}

void write_basic_info_dlg(FILE* fp, const Liganddata* ligand_ref, const Dockpars* mypars, const Gridinfo* mygrid, const int* argc, char** argv)
//The function writes basic information (such as docking parameters) to the file whose file pointer is the first parameter of the function.
{
	char temp_filename [128];
	int i;

	fprintf(fp, "AutoDock-GPU version: %s\n\n", "sd-tsri-195-g25d913d6d79b52a64961714de0f358e0ecc7d4a4");

	fprintf(fp, "**********************************************************\n");
	fprintf(fp, "**    AutoDock-GPU AUTODOCKTOOLS-COMPATIBLE DLG FILE    **\n");
	fprintf(fp, "**********************************************************\n\n\n");

	//Writing out docking parameters

	fprintf(fp, "    DOCKING PARAMETERS\n");
	fprintf(fp, "    ________________________\n\n\n");

	fprintf(fp, "Ligand file:                               %s\n", mypars->ligandfile);
	fprintf(fp, "Grid fld file:                             %s\n\n", mypars->fldfile);

	fprintf(fp, "Number of runs:                            %lu\n", mypars->num_of_runs),
	fprintf(fp, "Number of energy evaluations:              %ld\n", mypars->num_of_energy_evals);
	fprintf(fp, "Number of generations:                     %ld\n", mypars->num_of_generations);
	fprintf(fp, "Size of population:                        %ld\n", mypars->pop_size);
	fprintf(fp, "Rate of crossover:                         %lf%%\n", (double) mypars->crossover_rate);
	fprintf(fp, "Tournament selection probability limit:    %lf%%\n", (double) mypars->tournament_rate);
	fprintf(fp, "Rate of mutation:                          %lf%%\n", (double) mypars->mutation_rate);
	fprintf(fp, "Maximal allowed delta movement:            +/- %lfA\n", (double) mypars->abs_max_dmov*mygrid->spacing);
	fprintf(fp, "Maximal allowed delta angle:               +/- %lf\n\n", (double) mypars->abs_max_dang);

	fprintf(fp, "Rate of local search:                      %lf%%\n", mypars->lsearch_rate);

	fprintf(fp, "Maximal number of local search iterations: %ld\n", mypars->max_num_of_iters);
	fprintf(fp, "Rho lower bound:                           %lf\n", (double) mypars->rho_lower_bound);
	fprintf(fp, "Spread of local search delta movement:     %lfA\n", (double) mypars->base_dmov_mul_sqrt3*mygrid->spacing/sqrt(3.0));
	fprintf(fp, "Spread of local search delta angle:        %lf\n", (double) mypars->base_dang_mul_sqrt3/sqrt(3.0));
	fprintf(fp, "Limit of consecutive successes/failures:   %ld\n\n", mypars->cons_limit);

		fprintf(fp, "Handle symmetry during clustering:         ");
	if (mypars->handle_symmetry != 0)
		fprintf(fp, "YES\n");
	else
		fprintf(fp, "NO\n");

	fprintf(fp, "RMSD tolerance:                            %lfA\n\n", mypars->rmsd_tolerance);

	fprintf(fp, "Program call in command line was:          ");
	for (i=0; i<*argc; i++)
		fprintf(fp, "%s ", argv [i]);
	fprintf(fp, "\n\n\n");

	//Writing out receptor parameters

	fprintf(fp, "    GRID PARAMETERS\n");
	fprintf(fp, "    ________________________\n\n\n");

	fprintf(fp, "Receptor name:                             %s\n", mygrid->receptor_name);
	fprintf(fp, "Number of grid points (x, y, z):           %d, %d, %d\n", mygrid->size_xyz [0],
			mygrid->size_xyz [1], mygrid->size_xyz [2]);
	fprintf(fp, "Grid size (x, y, z):                       %lf, %lf, %lfA\n", mygrid->size_xyz_angstr [0],
			mygrid->size_xyz_angstr [1], mygrid->size_xyz_angstr [2]);
	fprintf(fp, "Grid spacing:                              %lfA\n", mygrid->spacing);
	fprintf(fp, "\n\n");


	//Writing out ligand parameters

	strncpy(temp_filename, mypars->ligandfile, strlen(mypars->ligandfile) - 6);
	temp_filename [strlen(mypars->ligandfile) - 6] = '\0';

	fprintf(fp, "    LIGAND PARAMETERS\n");
	fprintf(fp, "    ________________________\n\n\n");

	fprintf(fp, "Ligand name:                               %s\n", temp_filename);
	fprintf(fp, "Number of atoms:                           %d\n", ligand_ref->num_of_atoms);
	fprintf(fp, "Number of rotatable bonds:                 %d\n", ligand_ref->num_of_rotbonds);
	fprintf(fp, "Number of atom types:                      %d\n\n\n", ligand_ref->num_of_atypes);

	fprintf(fp, "    DUMMY DATA (only for ADT-compatibility)\n");
	fprintf(fp, "    ________________________\n\n\n");
	fprintf(fp, "DPF> outlev 1\n");
	fprintf(fp, "DPF> ga_run %lu\n", mypars->num_of_runs);
	fprintf(fp, "DPF> fld %s.maps.fld\n", mygrid->receptor_name);
	fprintf(fp, "DPF> move %s\n\n\n", mypars->ligandfile);
}

void make_resfiles(float* final_population, 
		   float* energies, 
		   const Liganddata* ligand_ref,
		   const Liganddata* ligand_from_pdb,
		   const Liganddata* ligand_xray,
		   const Dockpars* mypars, 
		         int evals_performed, 
			 int generations_used, 
		   const Gridinfo* mygrid, 
		   const float* grids,
		   float* cpu_ref_ori_angles, 
		   const int* argc, 
		   char** argv, 
		   int debug, 
		   int run_cnt, 
		   Ligandresult* best_result)
//The function writes out final_population generated by get_result
//as well as different parameters about the docking, the receptor and the ligand to a file called fdock_report.txt in a
//readable and understandable format. The ligand_from_pdb parametere must be the Liganddata which includes the original
//ligand conformation as the result conformations will be compared to this one. The structs containing the grid informations
//and docking parameters are requrided as well as the number and values of command line arguments. The ligand_ref parameter
//describes the ligand with the reference orientation (gene values of final_population refer to this one, that is, this can
//be moved and rotated according to the genotype values). The function returns some information about the best result wich
//was found with the best_result parameter.
{
	FILE* fp;
	int i,j;
	double entity_rmsds [MAX_POPSIZE];
	Liganddata temp_docked;
	char temp_filename [128];
	char* name_ext_start;
	float accurate_interE [MAX_POPSIZE];
	float accurate_intraE [MAX_POPSIZE];
	float temp_genotype[GENOTYPE_LENGTH_IN_GLOBMEM];

	static float best_energy_of_all = 1000000000000;

	int pop_size = mypars->pop_size;

	sprintf(temp_filename, "final_population_run%d.txt", run_cnt+1);

	if (mypars->gen_finalpop != 0)	//if final population files are not required, no file will be opened.
	{
		fp = fopen(temp_filename, "w");

		write_basic_info(fp, ligand_ref, mypars, mygrid, argc, argv);	//Write basic information about docking and molecule parameters to file

		fprintf(fp, "           COUNTER STATES           \n");
		fprintf(fp, "===================================\n\n");
		fprintf(fp, "Number of energy evaluations performed:    %d\n", evals_performed);
		fprintf(fp, "Number of generations used:                %d\n", generations_used);
		fprintf(fp, "\n\n");
	}

	//Writing out state of final population

	strcpy(temp_filename, mypars->ligandfile);
	name_ext_start = temp_filename + strlen(mypars->ligandfile) - 6;	//without .pdbqt

	for (i=0; i<pop_size; i++)
	{
		temp_docked = *ligand_ref;

		//if (i==127)
		//	change_conform(&temp_docked, final_population [i], 1);				//calculating the conformation of current entity
		//else
			change_conform_f(&temp_docked, final_population+i*GENOTYPE_LENGTH_IN_GLOBMEM, cpu_ref_ori_angles, debug);

		//if (i==78)
		//	accurate_interE [i] = calc_interE(mygrid, &temp_docked, grids, 0.00, 1);
		//else
			accurate_interE[i] = calc_interE_f(mygrid, &temp_docked, grids, 0.00, debug);	//calculating the intermolecular energy

		if (i == 0)		//additional calculations for ADT-compatible result file, only in case of best conformation
			calc_interE_peratom_f(mygrid, &temp_docked, grids, 0.00, &(best_result->interE_elec), best_result->peratom_vdw, best_result->peratom_elec, debug);

		scale_ligand(&temp_docked, mygrid->spacing);
		//if (i==127)
		//	accurate_intraE [i] = calc_intraE(&temp_docked, 8, 0, mypars->coeffs.scaled_AD4_coeff_elec, mypars->coeffs.AD4_coeff_desolv, 1);				//calculating the intramolecular energy
		//else
			#if 0
			accurate_intraE[i] = calc_intraE_f(&temp_docked, 8, 0, mypars->coeffs.scaled_AD4_coeff_elec, mypars->coeffs.AD4_coeff_desolv, mypars->qasp, debug);
			#endif
			accurate_intraE[i] = calc_intraE_f(&temp_docked, 8, mypars->smooth, 0, mypars->coeffs.scaled_AD4_coeff_elec, mypars->coeffs.AD4_coeff_desolv, mypars->qasp, debug);

		move_ligand(&temp_docked, mygrid->origo_real_xyz);				//moving it according to grid location
		if (mypars->given_xrayligandfile == true) {
			entity_rmsds [i] = calc_rmsd(ligand_xray, &temp_docked, mypars->handle_symmetry);	//calculating rmds compared to original xray file
		}
		else {
			entity_rmsds [i] = calc_rmsd(ligand_from_pdb, &temp_docked, mypars->handle_symmetry);	//calculating rmds compared to original pdb file
		}

		//copying best result to output parameter
		if (i == 0)		//assuming this is the best one (final_population is arranged), however,
		{				//arrangement was made according to the unaccurate values calculated by FPGA
			best_result->interE = accurate_interE [i];
			best_result->intraE = accurate_intraE [i];
			best_result->reslig_realcoord = temp_docked;
			best_result->rmsd_from_ref = entity_rmsds [i];
			best_result->run_number = run_cnt+1;
		}

		//generating best.pdbqt
		if (i == 0)
			if (best_energy_of_all > accurate_interE [i] + accurate_intraE [i])
			{
				best_energy_of_all = accurate_interE [i] + accurate_intraE [i];

				if (mypars->gen_best != 0)
					gen_new_pdbfile(mypars->ligandfile, "best.pdbqt", &temp_docked);
			}

		if (i < mypars->gen_pdbs)											//if it is necessary, making new pdbs for best entities
		{
			sprintf(name_ext_start, "_docked_run%d_entity%d.pdbqt", run_cnt+1, i+1);	//name will be <original pdb filename>_docked_<number starting from 1>.pdb
			gen_new_pdbfile(mypars->ligandfile, temp_filename, &temp_docked);
		}
	}

	if (mypars->gen_finalpop != 0)
	{

		fprintf(fp, "     STATE OF FINAL POPULATION     \n");
		fprintf(fp, "===================================\n\n");

		fprintf(fp, " Entity |      dx [A]      |      dy [A]      |      dz [A]      |     phi []      |    theta []     | alpha_genrot [] |");
		for (i=0; i<ligand_from_pdb->num_of_rotbonds; i++)
			fprintf(fp, " alpha_rotb%2d [] |", i);
		fprintf(fp, " intramolecular energy | intermolecular energy | total energy calculated by CPU / calculated by GPU / difference | RMSD [A] | \n");

		fprintf(fp, "--------+------------------+------------------+------------------+------------------+------------------+------------------+");
		for (i=0; i<ligand_from_pdb->num_of_rotbonds; i++)
			fprintf(fp, "------------------+");
		fprintf(fp, "-----------------------+-----------------------+------------------------------------------------------------------------+----------+ \n");

		for (i=0; i<pop_size; i++)
		{
			fprintf(fp, "  %3d   |", i+1);

			for (j=0; j<3; j++)
				fprintf(fp, "    %10.3f    |", final_population [i*GENOTYPE_LENGTH_IN_GLOBMEM+j]*(mygrid->spacing));
			for (j=3; j<6+ligand_from_pdb->num_of_rotbonds; j++)
				fprintf(fp, "    %10.3f    |", final_population [i*GENOTYPE_LENGTH_IN_GLOBMEM+j]);

			fprintf(fp, " %21.3f |", accurate_intraE [i]);
			fprintf(fp, " %21.3f |", accurate_interE [i]);
			fprintf(fp, "  %21.3f / %21.3f / %21.3f |", accurate_intraE[i] + accurate_interE[i], energies[i], energies[i] - (accurate_intraE[i] + accurate_interE[i]));

			fprintf(fp, " %8.3lf | \n", entity_rmsds [i]);
		}

		fclose(fp);

	}
}

void cluster_analysis(Ligandresult myresults [], int num_of_runs, char* report_file_name, const Liganddata* ligand_ref,
					  const Dockpars* mypars, const Gridinfo* mygrid, const int* argc, char** argv, const double docking_avg_runtime,
					  const double program_runtime)
//The function performs ranked cluster analisys similar to that of AutoDock and creates a file with report_file_name name, the result
//will be written to it.
{
	int i,j;
	Ligandresult temp_ligres;
	int num_of_clusters;
	int current_clust_center;
	double temp_rmsd;
	double cluster_tolerance = 2;
	int result_clustered;
	int subrank;
	FILE* fp;
	int cluster_sizes [1000];
	double sum_energy [1000];
	double best_energy [1000];

	const double AD4_coeff_tors = mypars->coeffs.AD4_coeff_tors;
	double torsional_energy;

	//first of all, let's calculate the constant torsional free energy term
	torsional_energy = AD4_coeff_tors * ligand_ref->num_of_rotbonds;

	//arranging results according to energy, myresults [0] will be the best one (with lowest energy)
	for (j=0; j<num_of_runs-1; j++)
		for (i=num_of_runs-2; i>=j; i--)		//arrange according to sum of inter- and intramolecular energies
			if ((myresults [i]).interE /*+ (myresults [i]).intraE*/ > (myresults [i+1]).interE /*+ (myresults [i+1]).intraE*/)	//mimics the behaviour of AD4 unbound_same_as_bound
			//if ((myresults [i]).interE + (myresults [i]).intraE > (myresults [i+1]).interE + (myresults [i+1]).intraE)
			{
				temp_ligres = myresults [i];
				myresults [i] = myresults [i+1];
				myresults [i+1] = temp_ligres;
			}

	for (i=0; i<num_of_runs; i++)
	{
		(myresults [i]).clus_id = 0;	//indicates that it hasn't been put into cluster yet
	}

	//the best result is the center of the first cluster
	(myresults [0]).clus_id = 1;
	(myresults [0]).rmsd_from_cluscent = 0;
	num_of_clusters = 1;

	for (i=1; i<num_of_runs; i++)	//for each result
	{
		current_clust_center = 0;
		result_clustered = 0;

		for (j=0; j<i; j++)		//results with lower id-s are clustered, look for cluster centers
		{
			if ((myresults [j]).clus_id > current_clust_center)		//it is the center of a new cluster
			{
				current_clust_center = (myresults [j]).clus_id;
				temp_rmsd = calc_rmsd(&((myresults [j]).reslig_realcoord), &((myresults [i]).reslig_realcoord), mypars->handle_symmetry);	//comparing current result with cluster center
				if (temp_rmsd <= cluster_tolerance)		//in this case we put result i to cluster with center j
				{
					(myresults [i]).clus_id = current_clust_center;
					(myresults [i]).rmsd_from_cluscent = temp_rmsd;
					result_clustered = 1;
					break;
				}
			}
		}

		if (result_clustered != 1)		//if no suitable cluster was found, this is the center of a new one
		{
			num_of_clusters++;
			(myresults [i]).clus_id = num_of_clusters;		//new cluster id
			(myresults [i]).rmsd_from_cluscent = 0;
		}

	}

	for (i=1; i<=num_of_clusters; i++)	//printing cluster info to file
	{
		subrank = 0;
		cluster_sizes [i-1] = 0;
		sum_energy [i-1] = 0;
		for (j=0; j<num_of_runs; j++)
			if (myresults [j].clus_id == i)
			{
				subrank++;
				(cluster_sizes [i-1])++;
				sum_energy [i-1] += (myresults [j]).interE + /*(myresults [j]).intraE +*/ torsional_energy;		//intraE can be commented when unbound_same_as_bound
				(myresults [j]).clus_subrank = subrank;
				if (subrank == 1)
					best_energy [i-1] = (myresults [j]).interE + /*(myresults [j]).intraE +*/ torsional_energy;		//intraE can be commented when unbound_same_as_bound
			}
	}

	fp = fopen(report_file_name, "w");

	write_basic_info(fp, ligand_ref, mypars, mygrid, argc, argv);	//Write basic information about docking and molecule parameters to file

	fprintf(fp, "           RUN TIME INFO           \n");
	fprintf(fp, "===================================\n\n");

	fprintf(fp, "Average GPU run time for 1 run:           %lfs\n", docking_avg_runtime);
	fprintf(fp, "Total GPU docking run time:               %fs\n", docking_avg_runtime*mypars->num_of_runs);

	fprintf(fp, "Program run time:                          %lfs\n", program_runtime);
	fprintf(fp, "\n\n");

	fprintf(fp, "       CLUSTERING HISTOGRAM        \n");
	fprintf(fp, "===================================\n\n");
	fprintf(fp, " Cluster rank | Num in cluster |   Best energy   |   Mean energy   |    5    10   15   20   25   30   35\n");
	fprintf(fp, "--------------+----------------+-----------------+-----------------+----+----+----+----+----+----+----+\n");

	for (i=1; i<=num_of_clusters; i++)
	{
		fprintf(fp, "      %3d     |       %3d      | %15.3lf | %15.3lf |", i, cluster_sizes [i-1], best_energy [i-1], sum_energy [i-1]/cluster_sizes [i-1]);

		for (j=0; j<cluster_sizes [i-1]; j++)
			fprintf(fp, "#");

		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");

	fprintf(fp, "              CLUSTERS             \n");
	fprintf(fp, "===================================\n\n");
	fprintf(fp, " Rank | Subrank | Run | Intermolecular E | Intramolecular E | Torsional energy |   Total energy   | Cluster RMSD | Reference RMSD |\n");
	fprintf(fp, "------+---------+-----+------------------+------------------+------------------+------------------+--------------+----------------+\n");

	for (i=1; i<=num_of_clusters; i++)	//printing cluster info to file
	{
		for (j=0; j<num_of_runs; j++)
			if (myresults [j].clus_id == i)
			{
				fprintf(fp, "  %3d |   %3d   | %3d |  %15.3lf |  %15.3lf |  %15.3lf |  %15.3lf |     %4.2lf     |      %4.2lf      |\n", (myresults [j]).clus_id, (myresults [j]).clus_subrank, (myresults [j]).run_number,
						(myresults [j]).interE, (myresults [j]).intraE, torsional_energy, (myresults [j]).interE + /*(myresults [j]).intraE +*/ torsional_energy, (myresults [j]).rmsd_from_cluscent, (myresults [j]).rmsd_from_ref); 	//intraE can be commented when unbound_same_as_bound
			}
	}

	fclose(fp);

}

void clusanal_gendlg(Ligandresult myresults [], int num_of_runs, const Liganddata* ligand_ref,
					 const Dockpars* mypars, const Gridinfo* mygrid, const int* argc, char** argv, const double docking_avg_runtime,
					 unsigned long generations_used, unsigned long evals_performed)
//The function performs ranked cluster analisys similar to that of AutoDock and creates a file with report_file_name name, the result
//will be written to it.
{
	int i, j, atom_cnt;
	Ligandresult temp_ligres;
	int num_of_clusters;
	int current_clust_center;
	double temp_rmsd;
	int result_clustered;
	int subrank;
	FILE* fp;
	FILE* fp_orig;
	FILE* fp_xml;
	int cluster_sizes [1000];
	double sum_energy [1000];
	double best_energy [1000];
	int best_energy_runid [1000];
	char tempstr [256];
	char report_file_name [256];
	char xml_file_name [256];

	double cluster_tolerance = mypars->rmsd_tolerance;
	const double AD4_coeff_tors = mypars->coeffs.AD4_coeff_tors;
	double torsional_energy;

	//first of all, let's calculate the constant torsional free energy term
	torsional_energy = AD4_coeff_tors * ligand_ref->num_of_rotbonds;


	//GENERATING DLG FILE


	strcpy(report_file_name, mypars->resname);
	strcat(report_file_name, ".dlg");
	fp = fopen(report_file_name, "w");


	//writing basic info

	write_basic_info_dlg(fp, ligand_ref, mypars, mygrid, argc, argv);

	fprintf(fp, "           COUNTER STATES           \n");
	fprintf(fp, "___________________________________\n\n");
	fprintf(fp, "Number of energy evaluations performed:    %lu\n", evals_performed);
	fprintf(fp, "Number of generations used:                %lu\n", generations_used);
	fprintf(fp, "\n\n");

	//writing input pdbqt file

	fprintf(fp, "    INPUT LIGAND PDBQT FILE:\n    ________________________\n\n\n");

	fp_orig = fopen(mypars->ligandfile, "rb"); // fp_orig = fopen(mypars->ligandfile, "r");

	while (fgets(tempstr, 255, fp_orig) != NULL)	//reading original ligand pdb line by line
	{
		fprintf(fp, "INPUT-LIGAND-PDBQT: %s", tempstr);
	}

	fprintf(fp, "\n\n");

	fclose(fp_orig);

	//writing docked conformations

	for (i=0; i<num_of_runs; i++)
	{
		fprintf(fp, "    FINAL DOCKED STATE:\n    ________________________\n\n\n");

		fprintf(fp, "Run:   %d / %lu\n", i+1, mypars->num_of_runs);
		fprintf(fp, "Time taken for this run:   %.3lfs\n\n", docking_avg_runtime);

		fprintf(fp, "DOCKED: MODEL        %d\n", i+1);
		fprintf(fp, "DOCKED: USER    Run = %d\n", i+1);
		fprintf(fp, "DOCKED: USER\n");

		fprintf(fp, "DOCKED: USER    Estimated Free Energy of Binding    =");
		PRINT1000(fp, ((float) (myresults[i].interE + torsional_energy)));
		fprintf(fp, " kcal/mol  [=(1)+(2)+(3)-(4)]\n");

		fprintf(fp, "DOCKED: USER\n");

		fprintf(fp, "DOCKED: USER    (1) Final Intermolecular Energy     =");
		PRINT1000(fp, ((float) myresults[i].interE));
		fprintf(fp, " kcal/mol\n");

		fprintf(fp, "DOCKED: USER        vdW + Hbond + desolv Energy     =");
		PRINT1000(fp, ((float) (myresults[i].interE - myresults[i].interE_elec)));
		fprintf(fp, " kcal/mol\n");

		fprintf(fp, "DOCKED: USER        Electrostatic Energy            =");
		PRINT1000(fp, ((float) myresults[i].interE_elec));
		fprintf(fp, " kcal/mol\n");

		fprintf(fp, "DOCKED: USER    (2) Final Total Internal Energy     =");
		PRINT1000(fp, ((float) myresults[i].intraE));
		fprintf(fp, " kcal/mol\n");

		fprintf(fp, "DOCKED: USER    (3) Torsional Free Energy           =");
		PRINT1000(fp, ((float) torsional_energy));
		fprintf(fp, " kcal/mol\n");

		fprintf(fp, "DOCKED: USER    (4) Unbound System's Energy         =");
		PRINT1000(fp, ((float) myresults[i].intraE));
		fprintf(fp, " kcal/mol\n");

		fprintf(fp, "DOCKED: USER\n");
		fprintf(fp, "DOCKED: USER\n");


		fp_orig = fopen(mypars->ligandfile, "rb"); // fp_orig = fopen(mypars->ligandfile, "r");

		atom_cnt = 0;

		while (fgets(tempstr, 255, fp_orig) != NULL)	//reading original ligand pdb line by line
		{
			if ((strncmp("ATOM", tempstr, 4) == 0) || (strncmp("HETATM", tempstr, 6) == 0))
			{
				tempstr[30] = '\0';
				fprintf(fp, "DOCKED: %s", tempstr);
				fprintf(fp, "%8.3lf", myresults[i].reslig_realcoord.atom_idxyzq[atom_cnt][1]);		//x
				fprintf(fp, "%8.3lf", myresults[i].reslig_realcoord.atom_idxyzq[atom_cnt][2]);		//y
				fprintf(fp, "%8.3lf", myresults[i].reslig_realcoord.atom_idxyzq[atom_cnt][3]);		//z
				fprintf(fp, "%+6.2lf", myresults[i].peratom_vdw[atom_cnt]);		//vdw
				fprintf(fp, "%+6.2lf", myresults[i].peratom_elec[atom_cnt]);	//elec
				fprintf(fp, "    %+6.3lf ", myresults[i].reslig_realcoord.atom_idxyzq[atom_cnt][4]);	//q
				fprintf(fp, "%-2s\n", myresults[i].reslig_realcoord.atom_types[((int) myresults[i].reslig_realcoord.atom_idxyzq[atom_cnt][0])]);	//type
				atom_cnt++;
			}
			else
				if (strncmp("ROOT", tempstr, 4) == 0)
				{
					fprintf(fp, "DOCKED: USER                              x       y       z     vdW  Elec       q    Type\n");
					fprintf(fp, "DOCKED: USER                           _______ _______ _______ _____ _____    ______ ____\n");
					fprintf(fp, "DOCKED: %s", tempstr);
				}
				else
					fprintf(fp, "DOCKED: %s", tempstr);
		}

		fclose(fp_orig);

		fprintf(fp, "DOCKED: TER\n");
		fprintf(fp, "DOCKED: ENDMDL\n");
		fprintf(fp, "________________________________________________________________________________\n\n\n");

	}


	//PERFORM CLUSTERING


	//arranging results according to energy, myresults [0] will be the best one (with lowest energy)
	for (j=0; j<num_of_runs-1; j++)
		for (i=num_of_runs-2; i>=j; i--)		//arrange according to sum of inter- and intramolecular energies
			if ((myresults [i]).interE /*+ (myresults [i]).intraE*/ > (myresults [i+1]).interE /*+ (myresults [i+1]).intraE*/)	//mimics the behaviour of AD4 unbound_same_as_bound
			//if ((myresults [i]).interE + (myresults [i]).intraE > (myresults [i+1]).interE + (myresults [i+1]).intraE)
			{
				temp_ligres = myresults [i];
				myresults [i] = myresults [i+1];
				myresults [i+1] = temp_ligres;
			}

	for (i=0; i<num_of_runs; i++)
	{
		(myresults [i]).clus_id = 0;	//indicates that it hasn't been put into cluster yet
	}

	//the best result is the center of the first cluster
	(myresults [0]).clus_id = 1;
	(myresults [0]).rmsd_from_cluscent = 0;
	num_of_clusters = 1;

	for (i=1; i<num_of_runs; i++)	//for each result
	{
		current_clust_center = 0;
		result_clustered = 0;

		for (j=0; j<i; j++)		//results with lower id-s are clustered, look for cluster centers
		{
			if ((myresults [j]).clus_id > current_clust_center)		//it is the center of a new cluster
			{
				current_clust_center = (myresults [j]).clus_id;
				temp_rmsd = calc_rmsd(&((myresults [j]).reslig_realcoord), &((myresults [i]).reslig_realcoord), mypars->handle_symmetry);	//comparing current result with cluster center
				if (temp_rmsd <= cluster_tolerance)		//in this case we put result i to cluster with center j
				{
					(myresults [i]).clus_id = current_clust_center;
					(myresults [i]).rmsd_from_cluscent = temp_rmsd;
					result_clustered = 1;
					break;
				}
			}
		}

		if (result_clustered != 1)		//if no suitable cluster was found, this is the center of a new one
		{
			num_of_clusters++;
			(myresults [i]).clus_id = num_of_clusters;		//new cluster id
			(myresults [i]).rmsd_from_cluscent = 0;
		}

	}

	for (i=1; i<=num_of_clusters; i++)	//printing cluster info to file
	{
		subrank = 0;
		cluster_sizes [i-1] = 0;
		sum_energy [i-1] = 0;
		for (j=0; j<num_of_runs; j++)
			if (myresults [j].clus_id == i)
			{
				subrank++;
				(cluster_sizes [i-1])++;
				sum_energy [i-1] += (myresults [j]).interE + /*(myresults [j]).intraE +*/ torsional_energy;		//intraE can be commented when unbound_same_as_bound
				(myresults [j]).clus_subrank = subrank;
				if (subrank == 1)
				{
					best_energy [i-1] = (myresults [j]).interE + /*(myresults [j]).intraE +*/ torsional_energy;		//intraE can be commented when unbound_same_as_bound
					best_energy_runid  [i-1] = (myresults [j]).run_number;
				}
			}
	}


	//WRITING CLUSTER INFORMATION


	fprintf(fp, "    CLUSTERING HISTOGRAM\n    ____________________\n\n\n");
	fprintf(fp, "________________________________________________________________________________\n");
	fprintf(fp, "     |           |     |           |     |\n");
	fprintf(fp, "Clus | Lowest    | Run | Mean      | Num | Histogram\n");
	fprintf(fp, "-ter | Binding   |     | Binding   | in  |\n");
	fprintf(fp, "Rank | Energy    |     | Energy    | Clus|    5    10   15   20   25   30   35\n");
	fprintf(fp, "_____|___________|_____|___________|_____|____:____|____:____|____:____|____:___\n");

	for (i=0; i<num_of_clusters; i++)
	{
		fprintf(fp, "%4d |", i+1);

		if (best_energy[i] > 999999.99)
			fprintf(fp, "%+10.2e", best_energy[i]);
		else
			fprintf(fp, "%+10.2f", best_energy[i]);

		fprintf(fp, " |%4d |", best_energy_runid[i]);

		if (sum_energy[i]/cluster_sizes[i] > 999999.99)
			fprintf(fp, "%+10.2e |", sum_energy[i]/cluster_sizes[i]);
		else
			fprintf(fp, "%+10.2f |", sum_energy[i]/cluster_sizes[i]);

		fprintf(fp, "%4d |", cluster_sizes [i]);

		for (j=0; j<cluster_sizes [i]; j++)
			fprintf(fp, "#");

		fprintf(fp, "\n");
	}

	fprintf(fp, "_____|___________|_____|___________|_____|______________________________________\n\n\n");

	//writing RMSD table

	fprintf(fp, "    RMSD TABLE\n");
	fprintf(fp, "    __________\n\n\n");

    fprintf(fp, "_____________________________________________________________________\n");
    fprintf(fp, "     |      |      |           |         |                 |\n");
    fprintf(fp, "Rank | Sub- | Run  | Binding   | Cluster | Reference       | Grep\n");
    fprintf(fp, "     | Rank |      | Energy    | RMSD    | RMSD            | Pattern\n");
    fprintf(fp, "_____|______|______|___________|_________|_________________|___________\n" );

	for (i=0; i<num_of_clusters; i++)	//printing cluster info to file
	{
		for (j=0; j<num_of_runs; j++)
			if (myresults [j].clus_id == i+1)
			{
	            if (myresults[j].interE + torsional_energy > 999999.99)
	                fprintf(fp, "%4d   %4d   %4d  %+10.2e  %8.2f  %8.2f           RANKING\n", (myresults [j]).clus_id, (myresults [j]).clus_subrank, (myresults [j]).run_number,
	                		myresults[j].interE + torsional_energy, (myresults [j]).rmsd_from_cluscent, (myresults [j]).rmsd_from_ref);
	            else
	            	fprintf(fp, "%4d   %4d   %4d  %+10.2f  %8.2f  %8.2f           RANKING\n", (myresults [j]).clus_id, (myresults [j]).clus_subrank, (myresults [j]).run_number,
	                		myresults[j].interE + torsional_energy, (myresults [j]).rmsd_from_cluscent, (myresults [j]).rmsd_from_ref);
			}
	}

	fclose(fp);

	//if xml has to be generated

	strcpy(xml_file_name, mypars->resname);
	strcat(xml_file_name, ".xml");
	fp_xml = fopen(xml_file_name, "w");

	fprintf(fp_xml, "<?xml version=\"1.0\" ?>\n");
	fprintf(fp_xml, "<result>\n");

	fprintf(fp_xml, "\t<clustering_histogram>\n");
	for (i=0; i<num_of_clusters; i++)
	{
		fprintf(fp_xml, "\t\t<cluster cluster_rank=\"%d\" lowest_binding_energy=\"%.2lf\" run=\"%d\" mean_binding_energy=\"%.2lf\" num_in_clus=\"%d\" />\n",
				i+1, best_energy[i], best_energy_runid[i], sum_energy[i]/cluster_sizes[i], cluster_sizes [i]);
	}
	fprintf(fp_xml, "\t</clustering_histogram>\n");

	fprintf(fp_xml, "\t<rmsd_table>\n");
	for (i=0; i<num_of_clusters; i++)
	{
		for (j=0; j<num_of_runs; j++)
			if (myresults [j].clus_id == i+1)
			{
	            fprintf(fp_xml, "\t\t<run rank=\"%d\" sub_rank=\"%d\" run=\"%d\" binding_energy=\"%.2lf\" cluster_rmsd=\"%.2lf\" reference_rmsd=\"%.2lf\" />\n",
	            		(myresults [j]).clus_id, (myresults [j]).clus_subrank, (myresults [j]).run_number, myresults[j].interE + torsional_energy, (myresults [j]).rmsd_from_cluscent, (myresults [j]).rmsd_from_ref);
			}
	}
	fprintf(fp_xml, "\t</rmsd_table>\n");

	fprintf(fp_xml, "</result>\n");

	fclose(fp_xml);
}

