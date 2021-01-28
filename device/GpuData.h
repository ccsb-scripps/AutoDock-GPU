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


#ifndef GPUDATADOTH
#define GPUDATADOTH

struct GpuData {
	int devnum;
	int preload_gridsize;
	// Consolidated constants and memory pointers to reduce kernel launch overhead
	// dynamic
	cl_mem mem_interintra_const;
	cl_mem mem_intracontrib_const;
	cl_mem mem_intra_const;
	cl_mem mem_rotlist_const;
	cl_mem mem_conform_const;
	cl_mem mem_rotbonds_const;
	cl_mem mem_rotbonds_atoms_const;
	cl_mem mem_num_rotating_atoms_per_rotbond_const;
	// Constant data for correcting axisangle gradients
	cl_mem mem_angle_const;
	cl_mem mem_dependence_on_theta_const;
	cl_mem mem_dependence_on_rotangle_const;
};

struct GpuTempData {
	cl_context       context;
	cl_command_queue command_queue;
	cl_program       program;
	cl_kernel        kernel1;
	cl_kernel        kernel2;
	cl_kernel        kernel3;
	cl_kernel        kernel4;
	cl_kernel        kernel5;
	cl_kernel        kernel6;
	cl_kernel        kernel7;
	cl_mem           pMem_fgrids;
};
#endif


