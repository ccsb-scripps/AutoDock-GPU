/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian
Genetic Algorithm
Copyright (C) 2022 Intel Corporation

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

#ifndef DPCPP_MIGRATION_H
#define DPCPP_MIGRATION_H
//printf("Wait&throw\n");
#define __threadfence() 
#define cudaDeviceSynchronize()  dpct::get_current_device().queues_wait_and_throw();
#define cudaMemGetInfo(x, y) ad_MemGetInfo(x, y)

inline int ad_MemGetInfo(size_t *freemem, size_t *totalmem)
{ 
    
    dpct::device_info properties; 

    dpct::get_current_device().get_device_info(properties);
    *freemem = *totalmem = properties.get_global_mem_size();
    // printf("Freemem is %zu\n", *freemem);
    // returns 0 if success
    if (*freemem) return 0; else return (*freemem);
}
#endif