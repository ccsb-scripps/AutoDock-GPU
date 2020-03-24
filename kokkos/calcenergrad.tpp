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

// The GPU device function calculates the energy's gradient (forces or derivatives) 
// of the entity described by genotype, dockpars and the ligand-data
// arrays in constant memory and returns it in the "gradient" parameter. 
// The parameter "run_id" has to be equal to the ID of the run 
// whose population includes the current entity (which can be determined with get_group_id(0)), 
// since this determines which reference orientation should be used.

//#define PRINT_GRAD_TRANSLATION_GENES
//#define PRINT_GRAD_ROTATION_GENES
//#define PRINT_GRAD_TORSION_GENES

//#define ENABLE_PARALLEL_GRAD_TORSION

// The following is a scaling of gradients.
// Initially all genotypes and gradients
// were expressed in grid-units (translations)
// and sexagesimal degrees (rotation and torsion angles).
// Expressing them using angstroms / radians
// might help gradient-based minimizers.
// This conversion is applied to final gradients.
//#define CONVERT_INTO_ANGSTROM_RADIAN

// Scaling factor to multiply the gradients of 
// the genes expressed in degrees (all genes except the first three) 
// (GRID-SPACING * GRID-SPACING) / (DEG_TO_RAD * DEG_TO_RAD) = 461.644
//#define SCFACTOR_ANGSTROM_RADIAN ((0.375 * 0.375)/(DEG_TO_RAD * DEG_TO_RAD))

#include "calcenergy.hpp"

template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_calc_intermolecular_gradients(const member_type& team_member, const DockingParams<Device>& docking_params,const float *genotype, const InterIntra<Device>& interintra, float& energy, float* gradient)
{
	float partial_energies=0.0f;
        float weights[8];
        float cube[8];
        for (int atom_id = tidx;
                  atom_id < docking_params.num_of_atoms;
                  atom_id+= team_member.team_size())
        {
                float x = calc_coords[atom_id].x;
                float y = calc_coords[atom_id].y;
                float z = calc_coords[atom_id].z;
                float q = interintra.atom_charges_const[atom_id];
                int atom_typeid = interintra.atom_types_const[atom_id];

                if ((x < 0) || (y < 0) || (z < 0) || (x >= docking_params.gridsize_x-1)
                                                  || (y >= docking_params.gridsize_y-1)
                                                  || (z >= docking_params.gridsize_z-1)){
                        x -= 0.5f * docking_params.gridsize_x;
                        y -= 0.5f * docking_params.gridsize_y;
                        z -= 0.5f * docking_params.gridsize_z;
                        partial_energies += 21.0f * (x*x+y*y+z*z); //100000.0f;
                        // Setting gradients (forces) penalties.
                        // The idea here is to push the offending
                        // molecule towards the center rather
                        gradient_inter_x[atom_id] += 42.0f * x;
                        gradient_inter_y[atom_id] += 42.0f * y;
                        gradient_inter_z[atom_id] += 42.0f * z;
                        continue;
                }
                // Getting coordinates
                float x_low  = floor(x);
                float y_low  = floor(y);
                float z_low  = floor(z);

                float dx = x - x_low;
                float omdx = 1.0f - dx;
                float dy = y - y_low;
                float omdy = 1.0f - dy;
                float dz = z - z_low;
                float omdz = 1.0f - dz;

		// Calculating interpolation weights
                weights [idx_000] = omdx*omdy*omdz;
                weights [idx_010] = omdx*dy*omdz;
                weights [idx_001] = omdx*omdy*dz;
                weights [idx_011] = omdx*dy*dz;
                weights [idx_100] = dx*omdy*omdz;
                weights [idx_110] = dx*dy*omdz;
                weights [idx_101] = dx*omdy*dz;
                weights [idx_111] = dx*dy*dz;

		// Grid index at 000
		int grid_ind_000 = (x_low  + y_low*dock_params.gridsize_x  + z_low*dock_params.g2)<<2;
		unsigned long mul_tmp = atom_typeid*dock_params.g3<<2;

		// Calculating affinity energy
                partial_energies += kokkos_trilinear_interp(dock_params.fgrids, grid_ind_000+mul_tmp, weights);

		// -------------------------------------------------------------------
                // Deltas dx, dy, dz are already normalized
                // (by host/src/getparameters.cpp) in AutoDock-GPU.
                // The correspondance between vertices in xyz axes is:
                // 0, 1, 2, 3, 4, 5, 6, 7  and  000, 100, 010, 001, 101, 110, 011, 111
                // -------------------------------------------------------------------
                /*
                    deltas: (x-x0)/(x1-x0), (y-y0...
                    vertices: (000, 100, 010, 001, 101, 110, 011, 111)

                          Z
                          '
                          3 - - - - 6
                         /.        /|
                        4 - - - - 7 |
                        | '       | |
                        | 0 - - - + 2 -- Y
                        '/        |/
                        1 - - - - 5
                       /
                      X
                */

                // -------------------------------------------------------------------
                // Calculating gradients (forces) corresponding to
                // "atype" intermolecular energy
                // Derived from autodockdev/maps.py
                // -------------------------------------------------------------------

                // Vector in x-direction
		gradient_inter_x[atom_id] += omdz * (omdy * (cube [idx_100] - cube [idx_000]) + dy * (cube [idx_110] - cube [idx_010])) +
                                               dz * (omdy * (cube [idx_101] - cube [idx_001]) + dy * (cube [idx_111] - cube [idx_011]));
                // Vector in y-direction
                gradient_inter_y[atom_id] += omdz * (omdx * (cube [idx_010] - cube [idx_000]) + dx * (cube [idx_110] - cube [idx_100])) +
                                               dz * (omdx * (cube [idx_011] - cube [idx_001]) + dx * (cube [idx_111] - cube [idx_101]));
                // Vectors in z-direction
                gradient_inter_z[atom_id] += omdy * (omdx * (cube [idx_001] - cube [idx_000]) + dx * (cube [idx_101] - cube [idx_100])) +
                                               dy * (omdx * (cube [idx_011] - cube [idx_010]) + dx * (cube [idx_111] - cube [idx_110]));
                // -------------------------------------------------------------------
                // Calculating gradients (forces) corresponding to
                // "elec" intermolecular energy
                // Derived from autodockdev/maps.py
                // -------------------------------------------------------------------

                // Capturing electrostatic values
                atom_typeid = docking_params.num_of_atypes;

                mul_tmp = atom_typeid*dock_params.g3<<2; // different atom type id to get charge IA
                // Calculating affinity energy
                partial_energies += q * kokkos_trilinear_interp(dock_params.fgrids, grid_ind_000+mul_tmp, weights);

                // Vector in x-direction
                gradient_inter_x[atom_id] += q * ( omdz * (omdy * (cube [idx_100] - cube [idx_000]) + dy * (cube [idx_110] - cube [idx_010])) +
                                                     dz * (omdy * (cube [idx_101] - cube [idx_001]) + dy * (cube [idx_111] - cube [idx_011])));
                // Vector in y-direction
                gradient_inter_y[atom_id] += q * ( omdz * (omdx * (cube [idx_010] - cube [idx_000]) + dx * (cube [idx_110] - cube [idx_100])) +
                                                     dz * (omdx * (cube [idx_011] - cube [idx_001]) + dx * (cube [idx_111] - cube [idx_101])));
                // Vectors in z-direction
                gradient_inter_z[atom_id] += q * ( omdy * (omdx * (cube [idx_001] - cube [idx_000]) + dx * (cube [idx_101] - cube [idx_100])) +
                                                     dy * (omdx * (cube [idx_011] - cube [idx_010]) + dx * (cube [idx_111] - cube [idx_110])));
                // -------------------------------------------------------------------
                // Calculating gradients (forces) corresponding to
                // "dsol" intermolecular energy
                // Derived from autodockdev/maps.py
                // -------------------------------------------------------------------
                // Need only magnitude of charge from here on down
                q = fabs(q);
                // Capturing desolvation values (atom_typeid+1 compared to above => mul_tmp + g3*4)
                mul_tmp += g3<<2;

                // Calculating affinity energy
                partial_energies += q * kokkos_trilinear_interp(dock_params.fgrids, grid_ind_000+mul_tmp, weights);

                // Vector in x-direction
                gradient_inter_x[atom_id] += q * ( omdz * (omdy * (cube [idx_100] - cube [idx_000]) + dy * (cube [idx_110] - cube [idx_010])) +
                                                     dz * (omdy * (cube [idx_101] - cube [idx_001]) + dy * (cube [idx_111] - cube [idx_011])));
                // Vector in y-direction
                gradient_inter_y[atom_id] += q * ( omdz * (omdx * (cube [idx_010] - cube [idx_000]) + dx * (cube [idx_110] - cube [idx_100])) +
                                                     dz * (omdx * (cube [idx_011] - cube [idx_001]) + dx * (cube [idx_111] - cube [idx_101])));
                // Vectors in z-direction
                gradient_inter_z[atom_id] += q * ( omdy * (omdx * (cube [idx_001] - cube [idx_000]) + dx * (cube [idx_101] - cube [idx_100])) +
                                                     dy * (omdx * (cube [idx_011] - cube [idx_010]) + dx * (cube [idx_111] - cube [idx_110])));
                // -------------------------------------------------------------------
        }
}


template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_calc_intramolecular_gradients(const member_type& team_member, const DockingParams<Device>& docking_params,const float *genotype, const IntraContrib<Device>& intracontrib, const InterIntra<Device>& interintra, const Intra<Device>& intra, float& energy, float* gradient)
{
        float delta_distance = 0.5f*docking_params.smooth;
        float smoothed_distance;

        // ================================================
        // CALCULATING INTRAMOLECULAR GRADIENTS
        // ================================================
        for (int contributor_counter = tidx;
                  contributor_counter < docking_params.num_of_intraE_contributors;
                  contributor_counter+= team_member.team_size())
        {
                // Storing in a private variable
                // the gradient contribution of each contributing atomic pair
                float priv_gradient_per_intracontributor= 0.0f;

                // Getting atom IDs
                int atom1_id = intracontrib.intraE_contributors_const[3*contributor_counter];
                int atom2_id = intracontrib.intraE_contributors_const[3*contributor_counter+1];
                int hbond = (int)(intracontrib.intraE_contributors_const[3*contributor_counter+2] == 1);    // evaluates to 1 in case of H-bond, 0 otherwise

                // Calculating vector components of vector going
                // from first atom's to second atom's coordinates
                float subx = calc_coords[atom1_id].x - calc_coords[atom2_id].x;
                float suby = calc_coords[atom1_id].y - calc_coords[atom2_id].y;
                float subz = calc_coords[atom1_id].z - calc_coords[atom2_id].z;

                // Calculating atomic_distance
                float dist = native_sqrt(subx*subx + suby*suby + subz*subz);
                float atomic_distance = dist*docking_params.grid_spacing;

                // Getting type IDs
                int atom1_typeid = interintra.atom_types_const[atom1_id];
                int atom2_typeid = interintra.atom_types_const[atom2_id];

                int atom1_type_vdw_hb = intra.atom1_types_reqm_const [atom1_typeid];
                int atom2_type_vdw_hb = intra.atom2_types_reqm_const [atom2_typeid];

                // ------------------------------------------------
                // Required only for flexrings
                // Checking if this is a CG-G0 atomic pair.
                // If so, then adding energy term (E = G * distance).
                // Initial specification required NON-SMOOTHED distance.
                // This interaction is evaluated at any distance,
                // so no cuttoffs considered here!
                // vbond is G when calculating flexrings, 0.0 otherwise
                float vbond = G * (float)(((atom1_type_vdw_hb == ATYPE_CG_IDX) && (atom2_type_vdw_hb == ATYPE_G0_IDX)) ||
                                          ((atom1_type_vdw_hb == ATYPE_G0_IDX) && (atom2_type_vdw_hb == ATYPE_CG_IDX)));
                partial_energies[tidx] += vbond * atomic_distance;
                priv_gradient_per_intracontributor += vbond;
                // ------------------------------------------------

                // Calculating energy contributions
                // Cuttoff1: internuclear-distance at 8A only for vdw and hbond.
                if (atomic_distance < 8.0f)
                {
			// Getting optimum pair distance (opt_distance) from reqm and reqm_hbond
                        // reqm: equilibrium internuclear separation
                        //       (sum of the vdW radii of two like atoms (A)) in the case of vdW
                        // reqm_hbond: equilibrium internuclear separation
                        //       (sum of the vdW radii of two like atoms (A)) in the case of hbond
                        float opt_distance = (intra.reqm_const [atom1_type_vdw_hb+ATYPE_NUM*hbond] + intra.reqm_const [atom2_type_vdw_hb+ATYPE_NUM*hbond]);

                        // Getting smoothed distance
                        // smoothed_distance = function(atomic_distance, opt_distance)
                        float opt_dist_delta = opt_distance - atomic_distance;
                        if(fabs(opt_dist_delta)>=delta_distance){
                                smoothed_distance = atomic_distance + copysign(delta_distance,opt_dist_delta);
                        } else smoothed_distance = opt_distance;
                        // Calculating van der Waals / hydrogen bond term
                        int idx = atom1_typeid * docking_params.num_of_atypes + atom2_typeid;
                        float nvbond = 1.0 - vbond;
                        float A = nvbond * native_divide(intra.VWpars_AC_const[idx],native_powr(smoothed_distance,12));
                        float B = nvbond * native_divide(intra.VWpars_BD_const[idx],native_powr(smoothed_distance,6+4*hbond));
                        partial_energies[tidx] += A - B;
                        priv_gradient_per_intracontributor += native_divide ((6.0f+4.0f*hbond) * B - 12.0f * A, smoothed_distance);
                } // if cuttoff1 - internuclear-distance at 8A

                // Calculating energy contributions
                // Cuttoff2: internuclear-distance at 20.48A only for el and sol.
                if (atomic_distance < 20.48f)
                {
                        float q1 = interintra.atom_charges_const[atom1_id];
                        float q2 = interintra.atom_charges_const[atom2_id];
//                      float exp_el = native_exp(DIEL_B_TIMES_H*atomic_distance);
                        float dist2 = atomic_distance*atomic_distance;
                        // Calculating desolvation term
                        // 1/25.92 = 0.038580246913580245
                        float desolv_energy =  ((intra.dspars_S_const[atom1_typeid] +
                                                 docking_params.qasp*fabs(q1)) * intra.dspars_V_const[atom2_typeid] +
                                                (intra.dspars_S_const[atom2_typeid] +
                                                 docking_params.qasp*fabs(q2)) * intra.dspars_V_const[atom1_typeid]) *
                                                native_divide (
                                                                docking_params.coeff_desolv*(12.96f-0.1063f*dist2*(1.0f-0.001947f*dist2)),
                                                                (12.96f+dist2*(0.4137f+dist2*(0.00357f+0.000112f*dist2)))
                                                              );
                        float dist_shift=atomic_distance+1.588f;
                        dist2=dist_shift*dist_shift;
                        float disth_shift=atomic_distance+0.794f;
                        float disth4=disth_shift*disth_shift;
                        disth4*=disth4;
                        float diel = native_divide(1.404f,dist2)+native_divide(0.072f,disth4)+0.00831f;
			float es_energy = native_divide (
                                                          docking_params.coeff_elec * q1 * q2,
                                                          atomic_distance
                                                        );
                        partial_energies[tidx] += diel * es_energy + desolv_energy;

                        // http://www.wolframalpha.com/input/?i=1%2F(x*(A%2B(B%2F(1%2BK*exp(-h*B*x)))))
                        priv_gradient_per_intracontributor +=  -native_divide(es_energy,atomic_distance) * diel - es_energy * (native_divide (2.808f,dist2*dist_shift)+native_divide(0.288f,disth4*disth_shift)) -
                                                                0.0771605f * atomic_distance * desolv_energy; // 1/3.6^2 = 1/12.96 = 0.0771605
                } // if cuttoff2 - internuclear-distance at 20.48A


                // Decomposing "priv_gradient_per_intracontributor"
                // into the contribution of each atom of the pair.
                // Distances in Angstroms of vector that goes from
                // "atom1_id"-to-"atom2_id", therefore - subx, - suby, and - subz are used
                float grad_div_dist = native_divide(-priv_gradient_per_intracontributor,dist);
                float priv_intra_gradient_x = subx * grad_div_dist;
                float priv_intra_gradient_y = suby * grad_div_dist;
                float priv_intra_gradient_z = subz * grad_div_dist;

                // Calculating gradients in xyz components.
                // Gradients for both atoms in a single contributor pair
                // have the same magnitude, but opposite directions
                atomicSub_g_f(&gradient_intra_x[atom1_id], priv_intra_gradient_x);
                atomicSub_g_f(&gradient_intra_y[atom1_id], priv_intra_gradient_y);
                atomicSub_g_f(&gradient_intra_z[atom1_id], priv_intra_gradient_z);

                atomicAdd_g_f(&gradient_intra_x[atom2_id], priv_intra_gradient_x);
                atomicAdd_g_f(&gradient_intra_y[atom2_id], priv_intra_gradient_y);
                atomicAdd_g_f(&gradient_intra_z[atom2_id], priv_intra_gradient_z);
        }
}


template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_acculumate_interintra_gradients(const member_type& team_member, const DockingParams<Device>& docking_params,const float *genotype, float& energy, float* gradient)
{
	for (int atom_cnt = tidx;
                  atom_cnt < docking_params.num_of_atoms;
                  atom_cnt+= team_member.team_size()) {

                // Grid gradients were calculated in the grid space,
                // so they have to be put back in Angstrom.

                // Intramolecular gradients were already in Angstrom,
                // so no scaling for them is required.
                float grad_total_x = native_divide(gradient_inter_x[atom_cnt], docking_params.grid_spacing);
                float grad_total_y = native_divide(gradient_inter_y[atom_cnt], docking_params.grid_spacing);
                float grad_total_z = native_divide(gradient_inter_z[atom_cnt], docking_params.grid_spacing);

                #if defined (PRINT_GRAD_ROTATION_GENES)
                if (atom_cnt == 0) {
                        printf("\n%s\n", "----------------------------------------------------------");
                        printf("%s\n", "Gradients: inter and intra");
                        printf("%10s %13s %13s %13s %5s %13s %13s %13s\n", "atom_id", "grad_intER.x", "grad_intER.y", "grad_intER.z", "|", "grad_intRA.x", "grad_intRA.y", "grad_intRA.z");
                }
                printf("%10u %13.6f %13.6f %13.6f %5s %13.6f %13.6f %13.6f\n", atom_cnt, grad_total_x, grad_total_y, grad_total_z, "|", gradient_intra_x[atom_cnt], gradient_intra_y[atom_cnt], gradient_intra_z[atom_cnt]);
                #endif

                grad_total_x += gradient_intra_x[atom_cnt];
                grad_total_y += gradient_intra_y[atom_cnt];
                grad_total_z += gradient_intra_z[atom_cnt];
                // Re-using "gradient_inter_*" for total gradient (inter+intra)
                gradient_inter_x[atom_cnt] = grad_total_x;
                gradient_inter_y[atom_cnt] = grad_total_y;
                gradient_inter_z[atom_cnt] = grad_total_z;

                // Re-use "gradient_intra_*" for total gradient to do reduction below
                // - need to prepare by doing thread-wise reduction
                gradient_intra_x[tidx] += (float)(atom_cnt==tidx)*(-gradient_intra_x[tidx])+grad_total_x; // We need to start sum from 0 but I don't want an if statement
                gradient_intra_y[tidx] += (float)(atom_cnt==tidx)*(-gradient_intra_y[tidx])+grad_total_y;
                gradient_intra_z[tidx] += (float)(atom_cnt==tidx)*(-gradient_intra_z[tidx])+grad_total_z;

                #if defined (PRINT_GRAD_ROTATION_GENES)
                if (atom_cnt == 0) {
                        printf("\n%s\n", "----------------------------------------------------------");
                        printf("%s\n", "Gradients: total = inter + intra");
                        printf("%10s %13s %13s %13s\n", "atom_id", "grad.x", "grad.y", "grad.z");
                }
                printf("%10u %13.6f %13.6f %13.6f \n", atom_cnt, gradient_inter_x[atom_cnt], gradient_inter_y[atom_cnt], gradient_inter_z[atom_cnt]);
                #endif
        }
}


template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_calc_energy_and_translation_gradients(const member_type& team_member, const DockingParams<Device>& docking_params,const float *genotype, float& energy, float* gradient)
{
	// reduction over partial energies and prepared "gradient_intra_*" values
        for (int off=(team_member.team_size())>>1; off>0; off >>= 1)
        {
                barrier(CLK_LOCAL_MEM_FENCE);
                if (tidx < off)
                {
                        partial_energies[tidx] += partial_energies[tidx+off];
                        gradient_intra_x[tidx] += gradient_intra_x[tidx+off];
                        gradient_intra_y[tidx] += gradient_intra_y[tidx+off];
                        gradient_intra_z[tidx] += gradient_intra_z[tidx+off];
                }
        }
        if (tidx == 0) {
                *energy = partial_energies[0];
                // Scaling gradient for translational genes as
                // their corresponding gradients were calculated in the space
                // where these genes are in Angstrom,
                // but AutoDock-GPU translational genes are within in grids
                gradient_genotype[0] = gradient_intra_x[0] * docking_params.grid_spacing;
                gradient_genotype[1] = gradient_intra_y[0] * docking_params.grid_spacing;
                gradient_genotype[2] = gradient_intra_z[0] * docking_params.grid_spacing;

                #if defined (PRINT_GRAD_TRANSLATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("gradient_x:%f\n", gradient_genotype [0]);
                printf("gradient_y:%f\n", gradient_genotype [1]);
                printf("gradient_z:%f\n", gradient_genotype [2]);
                #endif
        }
}


template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_calc_rotation_gradients(const member_type& team_member, const DockingParams<Device>& docking_params,const float *genotype, const AxisCorrection<Device>& axis_correction, float& energy, float* gradient)
{
	        // Transform gradients_inter_{x|y|z}
        // into local_gradients[i] (with four quaternion genes)
        // Derived from autodockdev/motions.py/forces_to_delta_genes()

        // Transform local_gradients[i] (with four quaternion genes)
        // into local_gradients[i] (with three Shoemake genes)
        // Derived from autodockdev/motions.py/_get_cube3_gradient()
        // ------------------------------------------

        // start by populating "gradient_intra_*" with torque values
        gradient_intra_x[tidx] = 0.0;
        gradient_intra_y[tidx] = 0.0;
        gradient_intra_z[tidx] = 0.0;
        for (int atom_cnt = tidx;
                  atom_cnt < docking_params.num_of_atoms;
                  atom_cnt+= team_member.team_size()) {
                float4 r = (calc_coords[atom_cnt] - genrot_movingvec) * docking_params.grid_spacing;
                // Re-using "gradient_inter_*" for total gradient (inter+intra)
                float4 force;
                force.x = gradient_inter_x[atom_cnt];
                force.y = gradient_inter_y[atom_cnt];
                force.z = gradient_inter_z[atom_cnt];
                force.w = 0.0;
                float4 torque_rot = cross(r, force);
                gradient_intra_x[tidx] += torque_rot.x;
                gradient_intra_y[tidx] += torque_rot.y;
                gradient_intra_z[tidx] += torque_rot.z;
        }
        // do a reduction over the total gradient containing prepared "gradient_intra_*" values
        for (int off=(team_member.team_size())>>1; off>0; off >>= 1)
        {
                barrier(CLK_LOCAL_MEM_FENCE);
                if (tidx < off)
                {
                        gradient_intra_x[tidx] += gradient_intra_x[tidx+off];
                        gradient_intra_y[tidx] += gradient_intra_y[tidx+off];
                        gradient_intra_z[tidx] += gradient_intra_z[tidx+off];
                }
        }
        if (tidx == 0) {
                float4 torque_rot;
                torque_rot.x = gradient_intra_x[0];
                torque_rot.y = gradient_intra_y[0];
                torque_rot.z = gradient_intra_z[0];

                #if defined (PRINT_GRAD_ROTATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("%-20s %-10.6f %-10.6f %-10.6f\n", "final torque: ", torque_rot.x, torque_rot.y, torque_rot.z);
                #endif

                // Derived from rotation.py/axisangle_to_q()
                // genes[3:7] = rotation.axisangle_to_q(torque, rad)
                float torque_length = fast_length(torque_rot);

                #if defined (PRINT_GRAD_ROTATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("%-20s %-10.6f\n", "torque length: ", torque_length);
                #endif

                /*
                // Infinitesimal rotation in radians
                const float infinitesimal_radian = 1E-5;
                */

                // Finding the quaternion that performs
                // the infinitesimal rotation around torque axis
                float4 quat_torque = torque_rot * native_divide (SIN_HALF_INFINITESIMAL_RADIAN, torque_length);
                quat_torque.w = COS_HALF_INFINITESIMAL_RADIAN;

                #if defined (PRINT_GRAD_ROTATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("%-20s %-10.6f\n", "INFINITESIMAL_RADIAN: ", INFINITESIMAL_RADIAN);
                printf("%-20s %-10.6f %-10.6f %-10.6f %-10.6f\n", "quat_torque (w,x,y,z): ", quat_torque.w, quat_torque.x, quat_torque.y, quat_torque.z);
                #endif

                // Converting quaternion gradients into orientation gradients 
                // Derived from autodockdev/motion.py/_get_cube3_gradient
                #if defined (PRINT_GRAD_ROTATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("%-30s %-10.6f %-10.6f %-10.6f %-10.6f\n", "current_q (w,x,y,z): ", genrot_unitvec.w, genrot_unitvec.x, genrot_unitvec.y, genrot_unitvec.z);
                #endif

                // This is where we want to be in quaternion space
                // target_q = rotation.q_mult(q, current_q)
                float4 target_q = quaternion_multiply(quat_torque, genrot_unitvec);

                #if defined (PRINT_GRAD_ROTATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("%-30s %-10.6f %-10.6f %-10.6f %-10.6f\n", "target_q (w,x,y,z): ", target_q.w, target_q.x, target_q.y, target_q.z);
                #endif

		// This is where we are in the orientation axis-angle space
                // Equivalent to "current_oclacube" in autodockdev/motions.py
                float current_phi      = fmod_pi2(PI_TIMES_2 + phi);
                float current_theta    = fmod_pi2(PI_TIMES_2 + theta);
                float current_rotangle = fmod_pi2(PI_TIMES_2 + genrotangle);

                // This is where we want to be in the orientation axis-angle space
                float target_phi, target_theta, target_rotangle;

                // target_oclacube = quaternion_to_oclacube(target_q, theta_larger_than_pi)
                // Derived from autodockdev/motions.py/quaternion_to_oclacube()
                // In our terms means quaternion_to_oclacube(target_q{w|x|y|z}, theta_larger_than_pi)
                target_rotangle = 2.0f * fast_acos(target_q.w); // = 2.0f * ang;
                float sin_ang = native_sqrt(1.0f-target_q.w*target_q.w); // = native_sin(ang);

                target_theta = PI_TIMES_2 + is_theta_gt_pi * fast_acos( native_divide ( target_q.z, sin_ang ) );
                target_phi   = fmod_pi2((atan2( is_theta_gt_pi*target_q.y, is_theta_gt_pi*target_q.x) + PI_TIMES_2));

                #if defined (PRINT_GRAD_ROTATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("%-30s %-10.6f %-10.6f %-10.6f\n", "target_axisangle (1,2,3): ", target_phi, target_theta, target_rotangle);
                #endif

                // The infinitesimal rotation will produce an infinitesimal displacement
                // in shoemake space. This is to guarantee that the direction of
                // the displacement in shoemake space is not distorted.
                // The correct amount of displacement in shoemake space is obtained
                // by multiplying the infinitesimal displacement by shoemake_scaling:
                //float shoemake_scaling = native_divide(torque_length, INFINITESIMAL_RADIAN/*infinitesimal_radian*/);
                float orientation_scaling = torque_length * INV_INFINITESIMAL_RADIAN;

                #if defined (PRINT_GRAD_ROTATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("%-30s %-10.6f\n", "orientation_scaling: ", orientation_scaling);
                #endif

                // Derivates in cube3
                float grad_phi, grad_theta, grad_rotangle;
                grad_phi      = orientation_scaling * (fmod_pi2(target_phi      - current_phi      + PI_FLOAT) - PI_FLOAT);
                grad_theta    = orientation_scaling * (fmod_pi2(target_theta    - current_theta    + PI_FLOAT) - PI_FLOAT);
                grad_rotangle = orientation_scaling * (fmod_pi2(target_rotangle - current_rotangle + PI_FLOAT) - PI_FLOAT);

                #if defined (PRINT_GRAD_ROTATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("%-30s \n", "grad_axisangle (1,2,3) - before empirical scaling: ");
                printf("%-13s %-13s %-13s \n", "grad_phi", "grad_theta", "grad_rotangle");
                printf("%-13.6f %-13.6f %-13.6f\n", grad_phi, grad_theta, grad_rotangle);
                #endif

                // Correcting theta gradients interpolating
                // values from correction look-up-tables
                // (X0,Y0) and (X1,Y1) are known points
                // How to find the Y value in the straight line between Y0 and Y1,
                // corresponding to a certain X?
		/*
                        | dependence_on_theta_const
                        | dependence_on_rotangle_const
                        |
                        |
                        |                        Y1
                        |
                        |             Y=?
                        |    Y0
                        |_________________________________ angle_const
                             X0         X        X1
                */

                // Finding the index-position of "grad_delta" in the "angle_const" array
                int index_theta    = floor((current_theta    - axis_correction.angle(0)) * inv_angle_delta);
                int index_rotangle = floor((current_rotangle - axis_correction.angle(0)) * inv_angle_delta);

                // Interpolating theta values
                // X0 -> index - 1
                // X1 -> index + 1
                // Expresed as weighted average:
                // Y = [Y0 * ((X1 - X) / (X1-X0))] +  [Y1 * ((X - X0) / (X1-X0))]
                // Simplified for GPU (less terms):
                // Y = [Y0 * (X1 - X) + Y1 * (X - X0)] / (X1 - X0)
                // Taking advantage of constant:
                // Y = [Y0 * (X1 - X) + Y1 * (X - X0)] * inv_angle_delta

                float X0, Y0;
                float X1, Y1;
                float dependence_on_theta;      //Y = dependence_on_theta

                // Using interpolation on out-of-bounds elements results in hang
                if ((index_theta <= 0) || (index_theta >= 999))
                {
                        dependence_on_theta = axis_correction.dependence_on_theta(stick_to_bounds(index_theta,0,999));
                } else
                {
                        X0 = axis_correction.angle(index_theta);
                        X1 = axis_correction.angle(index_theta+1);
                        Y0 = axis_correction.dependence_on_theta(index_theta);
                        Y1 = axis_correction.dependence_on_theta(index_theta+1);
                        dependence_on_theta = (Y0 * (X1-current_theta) + Y1 * (current_theta-X0)) * inv_angle_delta;
                }

                #if defined (PRINT_GRAD_ROTATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("%-30s %-10.6f\n", "dependence_on_theta: ", dependence_on_theta);
                #endif

                // Interpolating rotangle values
                float dependence_on_rotangle;   //Y = dependence_on_rotangle
                // Using interpolation on previous and/or next elements results in hang
                // Using interpolation on out-of-bounds elements results in hang
                if ((index_rotangle <= 0) || (index_rotangle >= 999))
                {
                        dependence_on_rotangle = axis_correction.dependence_on_rotangle(stick_to_bounds(index_rotangle,0,999));
                } else
                {
                        X0 = axis_correction.angle(index_rotangle);
                        X1 = axis_correction.angle(index_rotangle+1);
                        Y0 = axis_correction.dependence_on_rotangle(index_rotangle);
                        Y1 = axis_correction.dependence_on_rotangle(index_rotangle+1);
                        dependence_on_rotangle = (Y0 * (X1-current_rotangle) + Y1 * (current_rotangle-X0)) * inv_angle_delta;
                }
		#if defined (PRINT_GRAD_ROTATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("%-30s %-10.6f\n", "dependence_on_rotangle: ", dependence_on_rotangle);
                #endif

                // Setting gradient rotation-related genotypes in cube
                // Multiplicating by DEG_TO_RAD is to make it uniform to DEG (see torsion gradients)
                gradient_genotype[3] = native_divide(grad_phi, (dependence_on_theta * dependence_on_rotangle))  * DEG_TO_RAD;
                gradient_genotype[4] = native_divide(grad_theta, dependence_on_rotangle)                        * DEG_TO_RAD;
                gradient_genotype[5] = grad_rotangle                                                            * DEG_TO_RAD;
                #if defined (PRINT_GRAD_ROTATION_GENES)
                printf("\n%s\n", "----------------------------------------------------------");
                printf("%-30s \n", "grad_axisangle (1,2,3) - after empirical scaling: ");
                printf("%-13s %-13s %-13s \n", "grad_phi", "grad_theta", "grad_rotangle");
                printf("%-13.6f %-13.6f %-13.6f\n", gradient_genotype[3], gradient_genotype[4], gradient_genotype[5]);
                #endif
        }
}


template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_calc_torsion_gradients(const member_type& team_member, const DockingParams<Device>& docking_params,const float *genotype, const Grads<Device>& grads, float& energy, float* gradient)
{
	int num_torsion_genes = docking_params.num_of_genes-6;
        for (int idx = tidx;
                  idx < num_torsion_genes * docking_params.num_of_atoms;
                  idx += team_member.team_size()) {

                int rotable_atom_cnt = idx / num_torsion_genes;
                int rotbond_id = idx - rotable_atom_cnt * num_torsion_genes; // this is a bit cheaper than % (modulo)

                if (rotable_atom_cnt >= grads.num_rotating_atoms_per_rotbond(rotbond_id))
                        continue; // Nothing to do

                // Querying ids of atoms belonging to the rotatable bond in question
                int atom1_id = grads.rotbonds(2*rotbond_id);
                int atom2_id = grads.rotbonds(2*rotbond_id+1);

                float4 atomRef_coords;
                atomRef_coords = calc_coords[atom1_id];
                float4 rotation_unitvec = fast_normalize(calc_coords[atom2_id] - atomRef_coords);

                // Torque of torsions
                int lig_atom_id = grads.rotbonds_atoms(MAX_NUM_OF_ATOMS*rotbond_id + rotable_atom_cnt);
                float4 torque_tor, r, atom_force;

                // Calculating torque on point "A"
                // They are converted back to Angstroms here
                r = (calc_coords[lig_atom_id] - atomRef_coords);

                // Re-using "gradient_inter_*" for total gradient (inter+intra)
                atom_force.x = gradient_inter_x[lig_atom_id];
                atom_force.y = gradient_inter_y[lig_atom_id];
                atom_force.z = gradient_inter_z[lig_atom_id];
                atom_force.w = 0.0f;

                torque_tor = cross(r, atom_force);
                float torque_on_axis = dot(rotation_unitvec, torque_tor) * docking_params.grid_spacing; // it is cheaper to do a scalar multiplication than a vector one

                // Assignment of gene-based gradient
                // - this works because a * (a_1 + a_2 + ... + a_n) = a*a_1 + a*a_2 + ... + a*a_n
                atomicAdd_g_f(&gradient_genotype[rotbond_id+6], torque_on_axis * DEG_TO_RAD); /*(M_PI / 180.0f)*/;
        }
}


template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_calc_energrad(const member_type& team_member, const DockingParams<Device>& docking_params,const float *genotype,const Conform<Device>& conform, const RotList<Device>& rotlist, const IntraContrib<Device>& intracontrib, const InterIntra<Device>& interintra, const Intra<Device>& intra, const Grads<Device>& grads, const AxisCorrection<Device>& axis_correction, float& energy, float* gradient)
{
        // Get team and league ranks
        int tidx = team_member.team_rank();
        int lidx = team_member.league_rank();

        // Determine which run this team is doing
        int run_id;
        if (tidx == 0) {
                run_id = lidx/docking_params.pop_size;
        }

        team_member.team_barrier();

	// Initialize energy - FIX ME - should be array length team_size - ALS
	float partial_energies = 0.0f;

	// Gradient of the intermolecular energy per each ligand atom
        // Also used to store the accummulated gradient per each ligand atom
        float gradient_inter_x[MAX_NUM_OF_ATOMS];
        float gradient_inter_y[MAX_NUM_OF_ATOMS];
        float gradient_inter_z[MAX_NUM_OF_ATOMS];

        // Gradient of the intramolecular energy per each ligand atom
        float gradient_intra_x[MAX_NUM_OF_ATOMS];
        float gradient_intra_y[MAX_NUM_OF_ATOMS];
        float gradient_intra_z[MAX_NUM_OF_ATOMS];

	// Initializing gradients (forces) 
        // Derived from autodockdev/maps.py
        for (int atom_id = tidx; atom_id < MAX_NUM_OF_ATOMS; atom_id+= team_member.team_size()) {
                // Intermolecular gradients
                gradient_inter_x[atom_id] = 0.0f;
                gradient_inter_y[atom_id] = 0.0f;
                gradient_inter_z[atom_id] = 0.0f;
                // Intramolecular gradients
                gradient_intra_x[atom_id] = 0.0f;
                gradient_intra_y[atom_id] = 0.0f;
                gradient_intra_z[atom_id] = 0.0f;
        }

	// Initializing gradient genotypes
        for (int gene_cnt = tidx; gene_cnt < docking_params.num_of_genes; gene_cnt+= team_member.team_size()) {
                gradient[gene_cnt] = 0.0f;
        }

	// GETTING ATOMIC POSITIONS
	// ALS - This view needs to be in scratch space FIX ME
	Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords("calc_coords");
	Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, (int)(docking_params.num_of_atoms)),
	[=] (int& idx) {
		kokkos_get_atom_pos(idx, conform, calc_coords);
	});

	// CALCULATING ATOMIC POSITIONS AFTER ROTATIONS
	// General rotation moving vector
	float4struct genrot_movingvec;
	genrot_movingvec.x = genotype[0];
	genrot_movingvec.y = genotype[1];
	genrot_movingvec.z = genotype[2];
	genrot_movingvec.w = 0.0;

	// Convert orientation genes from sex. to radians
	float phi         = genotype[3] * DEG_TO_RAD;
	float theta       = genotype[4] * DEG_TO_RAD;
	float genrotangle = genotype[5] * DEG_TO_RAD;

	float4struct genrot_unitvec;
	float sin_angle = sin(theta);
	float s2 = sin(genrotangle*0.5f);
	genrot_unitvec.x = s2*sin_angle*cos(phi);
	genrot_unitvec.y = s2*sin_angle*sin(phi);
	genrot_unitvec.z = s2*cos(theta);
	genrot_unitvec.w = cos(genrotangle*0.5f);
	bool is_theta_gt_pi = 1.0-2.0*(float)(sin_angle < 0.0f); // WTF - ALS

	team_member.team_barrier();

	// FIX ME This will break once multi-threading - ALS
	// Loop over the rot bond list and carry out all the rotations
	Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, docking_params.rotbondlist_length),
	[=] (int& idx) {
		kokkos_rotate_atoms(idx, conform, rotlist, run_id, genotype, genrot_movingvec, genrot_unitvec, calc_coords);
	});

	team_member.team_barrier();

	// CALCULATING INTERMOLECULAR GRADIENTS
	kokkos_calc_intermolecular_gradients();

	// CALCULATING INTRAMOLECULAR GRADIENTS
//	kokkos_calc_intramolecular_gradients();

	team_member.team_barrier();

	// ACCUMULATE INTER- AND INTRAMOLECULAR GRADIENTS
//	kokkos_acculumate_interintra_gradients();

	team_member.team_barrier();

	// Obtaining energy and translation-related gradients
//	kokkos_calc_energy_and_translation_gradients();

	team_member.team_barrier();

	// Obtaining rotation-related gradients
//	kokkos_calc_rotation_gradients();

	team_member.team_barrier();

	// Obtaining torsion-related gradients
//	kokkos_calc_torsion_gradients();

#if defined (CONVERT_INTO_ANGSTROM_RADIAN)
	team_member.team_barrier();

        for (int gene_cnt = tidx+3; // Only for gene_cnt > 2 means start gene_cnt at 3
                  gene_cnt < docking_params.num_of_genes;
                  gene_cnt+= team_member.team_size()) {
                gradient[gene_cnt] *= SCFACTOR_ANGSTROM_RADIAN;
        }
#endif

	team_member.team_barrier();
}
