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

// Adadelta parameters (TODO: to be moved to header file?)
//#define RHO           0.9f
//#define EPSILON       1e-6
#define RHO             0.8f
#define EPSILON         1e-2

template<class Device>
KOKKOS_INLINE_FUNCTION void genotype_gradient_descent(const member_type& team_member, const DockingParams<Device>& docking_params, float* gradient,
			     float* square_gradient, float* square_delta, float* genotype)
{
        // Get team and league ranks
        int tidx = team_member.team_rank();
        int team_size = team_member.team_size();

	for(int i = tidx;
                 i < docking_params.num_of_genes;
                 i+= team_size) {

                // Accummulating gradient^2 (eq.8 in the paper)
                // square_gradient corresponds to E[g^2]
                square_gradient[i] = RHO * square_gradient[i] + (1.0f - RHO) * gradient[i] * gradient[i];

                // Computing update (eq.9 in the paper)
                float delta = -1.0f * gradient[i] * sqrt( (float)(square_delta[i] + EPSILON) /
						       (float)(square_gradient[i] + EPSILON));

                // Accummulating update^2
                // square_delta corresponds to E[dx^2]
                square_delta[i] = RHO * square_delta[i] + (1.0f - RHO) * delta * delta;

                // Applying update
                genotype[i] += delta;
        }
}
