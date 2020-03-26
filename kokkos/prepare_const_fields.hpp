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




#ifndef PREPARE_CONST_FIELDS_HPP
#define PREPARE_CONST_FIELDS_HPP

#include <math.h>
#include <stdio.h>

#include "calcenergy_basic.h"
#include "miscellaneous.h"
#include "processligand.h"
#include "getparameters.h"
#include "calcenergy.h"

template<class Device>
int kokkos_prepare_const_fields(Liganddata&			myligand_reference,
				 Dockpars*			mypars,
				 float*				cpu_ref_ori_angles,
				 InterIntra<Device>& interintra,
				 IntraContrib<Device>& intracontrib,
				 Intra<Device>& intra,
				 RotList<Device>& rotlist,
				 Conform<Device>& conform,
				 Grads<Device>& grads);

template<class Device>
void kokkos_prepare_axis_correction( float* angle, float* dependence_on_theta, float* dependence_on_rotangle,
                                 AxisCorrection<Device>& axis_correction);

#include "prepare_const_fields.tpp"

#endif
