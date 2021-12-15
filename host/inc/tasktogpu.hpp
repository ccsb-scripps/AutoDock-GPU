

#ifndef TASKTOGPU_HPP
#define TASKTOGPU_HPP

#include "omp.h"

/*
 * Schedule a task to a GPU randomly
 */
inline unsigned
gpu_scheduler_random(unsigned *occupancies, int ngpus)
{
  const unsigned chosen = rand() % ngpus;
  #pragma omp atomic
  occupancies[chosen]++;
  return chosen;
}

/*
 * Schedule a task to a GPU dynamically - a GPU is chosen based on the availabilty
 */
inline unsigned
gpu_scheduler_dynamic(unsigned *occupancies, int ngpus)
{
  short looking = 1;
  unsigned chosen;
  while (looking) {
    for (unsigned i = 0; i < ngpus; i++) {
      unsigned occ_i;
      #pragma omp atomic read
      occ_i = occupancies[i];
      if (occ_i == 0) {
        chosen = i;
        #pragma omp atomic
        occupancies[chosen]++;
        looking = 0;
        break;
      }
    }
  }
  return chosen;
}

#endif
