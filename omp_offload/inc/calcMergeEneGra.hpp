
#include "typedefine.h"
#include "GpuData.h"
#include "mathfn.hpp"

#ifndef CALCENERGY_GRAD_H
#define CALCENERGY_GRAD_H


#define TERMBITS 10
#define MAXTERM  (float)(1 << (31 - TERMBITS - 8)) // 2^(31 - 10 - 8) = 2^13 = 8192
#define TERMSCALE (float)(1 << TERMBITS)           // 2^10 = 1024
#define ONEOVERTERMSCALE 1.0f / TERMSCALE                  // 1 / 1024 = 0.000977
#define MAXREDUCE (float)(1 << (31 - TERMBITS - 4)) // 2^(31 - 10 - 4) = 2^17 = 131072

#define MAXENERGY FLT_MAX / 100.0f // Used to cap absurd energies so placeholder energy is always skipped in sorts
#define MAXFORCE FLT_MAX / 100.0f // Used to cap absurd gradients

#pragma omp declare target
inline void gpu_calc_energrad(
                                  int threadIdx,
                                  int blockDim,
                                  GpuData& cData,
                                  GpuDockparameters& dockpars,
                                  float*  genotype,
                                  float&  global_energy,
                                  int&    run_id,
                                  float3struct* calc_coords,
#if defined (DEBUG_ENERGY_KERNEL)
                                  float&  interE,
                                  float&  pintraE,
#endif
#ifdef FLOAT_GRADIENTS
                                  float3struct* gradient,
#else
                                  int3struct*   gradient,
#endif
                                  float*  fgradient_genotype,
                                  float*  pFloatAccumulator
                                 );

#pragma omp end declare target

#endif
