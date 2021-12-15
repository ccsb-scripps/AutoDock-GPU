

#include "typedefine.h"
#include "GpuData.h"
#include "mathfn.hpp"

#ifndef CALCENERGY_H
#define CALCENERGY_H

#pragma omp declare target
void get_atompos(
            const int atom_id,
            float3struct* calc_coords,
            GpuData& cData);

void rotate_atoms( 
			const int rotation_counter,
			float3struct* calc_coords,
			GpuData& cData,
			GpuDockparameters& dockpars,
			const int run_id, 
			float* pGenotype,
			float4struct genrot_unitvec,
			float4struct genrot_movingvec);

float calc_interenergy(
              const int atom_id,
              GpuData& cDtata,
	      GpuDockparameters& dockpars,
              float3struct* calc_coords );

float calc_intraenergy(
			       const int contributor_counter,
			       GpuData& cData,
			       GpuDockparameters& dockpars,
			       float3struct* calc_coords	);
#pragma omp end declare target

#endif
