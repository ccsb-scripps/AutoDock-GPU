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


#ifndef PROCESSRESULT_H_
#define PROCESSRESULT_H_

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "defines.h"
#include "processligand.h"
#include "getparameters.h"
#include "simulation_state.hpp"

#define PRINT1000(file, x) fprintf(file,  ((fabs((x)) >= 0.0) && ((fabs(x)) <= 1000.)) ? "%+7.2f" : "%+11.2e" , (x));

typedef struct
{
	float*                    genotype; // a pointer here is sufficient and saves lots of memory copies
	double                    atom_idxyzq         [MAX_NUM_OF_ATOMS][5]; // type id .. 0, x .. 1, y .. 2, z .. 3, q ... 4
	float                     interE;
	float                     interflexE;
	float                     interE_elec;
	float                     intraE;
	float                     intraflexE;
	float                     peratom_vdw         [MAX_NUM_OF_ATOMS];
	float                     peratom_elec        [MAX_NUM_OF_ATOMS];
	float                     rmsd_from_ref;
	float                     rmsd_from_cluscent;
	int                       clus_id;
	int                       clus_subrank;
	int                       run_number;
	std::vector<AnalysisData> analysis;
} Ligandresult;


void arrange_result(
                          float* final_population,
                          float* energies,
                    const int    pop_size
                   );

void write_basic_info(
                            FILE*       fp,
                            Liganddata* ligand_ref,
                      const Dockpars*   mypars,
                      const Gridinfo*   mygrid,
                      const int*        argc,
                      char**            argv
                     );

void write_basic_info_dlg(
                                FILE*       fp,
                                Liganddata* ligand_ref,
                          const Dockpars*   mypars,
                          const Gridinfo*   mygrid,
                          const int*        argc,
                                char**      argv
                         );

void make_resfiles(
                         float*        final_population,
                         float*        energies,
                         IntraTables*  tables,
                         Liganddata*   ligand_ref,
                         Liganddata*   ligand_from_pdb,
                   const Liganddata*   ligand_xray,
                   const Dockpars*     mypars,
                         int           evals_performed,
                         int           generations_used,
                   const Gridinfo*     mygrid,
                   const int*          argc,
                         char**        argv,
                         int           debug,
                         int           run_cnt,
                         float&        best_energy_of_all,
                         Ligandresult* best_result
                  );

void ligand_calc_output(
                              FILE*         fp,
                        const char*         prefix,
                              IntraTables*  tables,
                        const Liganddata*   ligand,
                        const Dockpars*     mypars,
                        const Gridinfo*     mygrid,
                              bool          output_analysis,
                              bool          output_energy
                       );

void generate_output(
                           Ligandresult  myresults [],
                           int           num_of_runs,
                           IntraTables*  tables,
                           Liganddata*   ligand_ref,
                     const Liganddata*   ligand_xray,
                     const Dockpars*     mypars,
                     const Gridinfo*     mygrid,
                     const int*          argc,
                           char**        argv,
                     const double        docking_avg_runtime,
                           unsigned long generations_used,
                           unsigned long evals_performed,
                           double        exec_time,
                           double        idle_time
                    );

void process_result(
                    const Gridinfo*        mygrid,
                    const Dockpars*        mypars,
                          Liganddata*      myligand_init,
                    const Liganddata*      myxrayligand,
                    const int*             argc,
                          char**           argv,
                          SimulationState& sim_state
                   );

#endif /* PROCESSRESULT_H_ */
