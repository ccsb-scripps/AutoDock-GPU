#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian
Genetic Algorithm Copyright (C) 2017 TU Darmstadt, Embedded Systems and
Applications Group, Germany. All rights reserved. For some of the code,
Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research
Institute.
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

#define ADADELTA_AUTOSTOP

// Gradient-based adadelta minimizer
// https://arxiv.org/pdf/1212.5701.pdf
// Alternative to Solis-Wets / Steepest-Descent / FIRE

// "rho": controls degree of memory of previous gradients
//        ranges between [0, 1[
//        "rho" = 0.9 most popular value
// "epsilon":  to better condition the square root

// Adadelta parameters (TODO: to be moved to header file?)
//#define RHO             0.9f
//#define EPSILON         1e-6
#define RHO             0.8f
#define EPSILON         1e-2f

// Enabling "DEBUG_ENERGY_ADADELTA" requires
// manually enabling "DEBUG_ENERGY_KERNEL" in calcenergy.cl
//#define DEBUG_ENERGY_ADADELTA
//#define PRINT_ADADELTA_ENERGIES
//#define PRINT_ADADELTA_GENES_AND_GRADS
//#define PRINT_ADADELTA_ATOMIC_COORDS
//#define DEBUG_SQDELTA_ADADELTA

// Enable DEBUG_ADADELTA_MINIMIZER for a seeing a detailed ADADELTA evolution
// If only PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION is enabled,
// then a only a simplified ADADELTA evolution will be shown
//#define DEBUG_ADADELTA_MINIMIZER
//#define PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION

// Enable this for debugging ADADELTA from a defined initial genotype
//#define DEBUG_ADADELTA_INITIAL_2BRT

void

gpu_gradient_minAD_kernel(
                          float* pMem_conformations_next,
                          float* pMem_energies_next
                         ,
                          sycl::nd_item<3> item_ct1,
                          uint8_t *dpct_local,
                          GpuData cData,
                          int *entity_id,
                          float *best_energy,
                          float *sFloatAccumulator,
                          float *rho,
                          int *cons_succ,
                          int *cons_fail)
// The GPU global function performs gradient-based minimization on (some) entities of conformations_next.
// The number of OpenCL compute units (CU) which should be started equals to num_of_minEntities*num_of_runs.
// This way the first num_of_lsentities entity of each population will be subjected to local search
// (and each CU carries out the algorithm for one entity).
// Since the first entity is always the best one in the current population,
// it is always tested according to the ls probability, and if it not to be
// subjected to local search, the entity with ID num_of_lsentities is selected instead of the first one (with ID 0).
{
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------

	// Determining entity, and its run, energy, and genotype
        int run_id = item_ct1.get_group(2) / cData.dockpars.num_of_lsentities;
        float energy;
	// Energy may go up, so we keep track of the best energy ever calculated.
	// Then, we return the genotype corresponding 
	// to the best observed energy, i.e. "best_genotype"

        auto sFloatBuff = (float *)dpct_local;

        // Ligand-atom position and partial energies
        sycl::float3 *calc_coords = (sycl::float3 *)sFloatBuff;

        // Gradient of the intermolecular energy per each ligand atom
	// Also used to store the accummulated gradient per each ligand atom
#ifdef FLOAT_GRADIENTS
	float3* cartesian_gradient = (float3*)(calc_coords + cData.dockpars.num_of_atoms);
#else
        sycl::int3 *cartesian_gradient =
            (sycl::int3 *)(calc_coords + cData.dockpars.num_of_atoms);
#endif
	// Genotype pointers
	float* genotype = (float*)(cartesian_gradient + cData.dockpars.num_of_atoms);
	float* best_genotype = genotype + cData.dockpars.num_of_genes;

	// Partial results of the gradient step
	float* gradient = best_genotype + cData.dockpars.num_of_genes;

	// Squared updates E[dx^2]
	float* square_delta = gradient + cData.dockpars.num_of_genes;

	// Vector for storing squared gradients E[g^2]
	float* square_gradient = square_delta + cData.dockpars.num_of_genes;

	// Iteration counter for the minimizer
	uint32_t iteration_cnt = 0;

        if (item_ct1.get_local_id(2) == 0)
        {
		// Since entity 0 is the best one due to elitism,
		// it should be subjected to random selection
                *entity_id = item_ct1.get_group(2) % cData.dockpars.num_of_lsentities;
                if (*entity_id == 0) {
                        // If entity 0 is not selected according to LS-rate,
			// choosing another entity
                        if (100.0f *
                                gpu_randf(cData.pMem_prng_states, item_ct1) >
                            cData.dockpars.lsearch_rate) {
                                *entity_id =
                                    cData.dockpars
                                        .num_of_lsentities; // AT - Should this
                                                            // be
                                                            // (uint)(dockpars_pop_size
                                                            // * gpu_randf(dockpars_prng_states))?
                        }
		}

		#if defined (DEBUG_ADADELTA_MINIMIZER) || defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
		printf("\n");
		printf("-------> Start of ADADELTA minimization cycle\n");
		printf("%20s %6u\n", "run_id: ", run_id);
		printf("%20s %6u\n", "entity_id: ", entity_id);
		printf("\n");
		printf("%20s \n", "LGA genotype: ");
		printf("%20s %.6f\n", "initial energy: ", energy);
		#endif
	}
        /*
        DPCT1007:67: Migration of this CUDA API is not supported by the Intel(R)
        DPC++ Compatibility Tool.
        */
        __threadfence();
        item_ct1.barrier();
        energy = pMem_energies_next[run_id * cData.dockpars.pop_size + *entity_id];

        int offset = (run_id * cData.dockpars.pop_size + *entity_id) *
                     GENOTYPE_LENGTH_IN_GLOBMEM;
        for (int i = item_ct1.get_local_id(2); i < cData.dockpars.num_of_genes;
             i += item_ct1.get_local_range().get(2))
        {
		genotype[i] = pMem_conformations_next[offset + i];
	}

	// -------------------------------------------------------------------
	// Calculate gradients (forces) for intermolecular energy
	// Derived from autodockdev/maps.py
	// -------------------------------------------------------------------

	#if defined (DEBUG_ENERGY_KERNEL)
	float interE;
	float intraE;
	#endif

	// Update vector, i.e., "delta".
	// It is added to the genotype to create the next genotype.
	// E.g. in steepest descent "delta" is -1.0 * stepsize * gradient

	// Asynchronous copy should be finished by here
        /*
        DPCT1007:68: Migration of this CUDA API is not supported by the Intel(R)
        DPC++ Compatibility Tool.
        */
        __threadfence();
        item_ct1.barrier();

        // Initializing vectors
        for (uint32_t i = item_ct1.get_local_id(2);
             i < cData.dockpars.num_of_genes;
             i += item_ct1.get_local_range().get(2))
        {
		gradient[i]        = 0.0f;
		square_gradient[i] = 0.0f;
		square_delta[i]    = 0.0f;
		best_genotype[i] = genotype[i];
	}

	// Initializing best energy
        if (item_ct1.get_local_id(2) == 0) {
                *best_energy = INFINITY;
        }

#ifdef ADADELTA_AUTOSTOP

        if (item_ct1.get_local_id(2) == 0) {
                *rho = 1.0f;
                *cons_succ = 0;
                *cons_fail = 0;
        }
#endif

	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// Perform adadelta iterations

	// The termination criteria is based on 
	// a maximum number of iterations, and
	// the minimum step size allowed for single-floating point numbers 
	// (IEEE-754 single float has a precision of about 6 decimal digits)
	do {
		// Printing number of ADADELTA iterations
		#if defined (DEBUG_ADADELTA_MINIMIZER) || defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
		if (threadIdx.x == 0) {
			#if defined (DEBUG_ADADELTA_MINIMIZER)
			printf("%s\n", "----------------------------------------------------------");
			#endif
			printf("%-15s %-3u ", "# ADADELTA iteration: ", iteration_cnt);
		}
		#endif

		// =============================================================
		// =============================================================
		// =============================================================
		// Calculating energy & gradient
                /*
                DPCT1007:70: Migration of this CUDA API is not supported by the
                Intel(R) DPC++ Compatibility Tool.
                */
                __threadfence();
                item_ct1.barrier();

                gpu_calc_energrad(
                    // Some OpenCL compilers don't allow declaring
                    // local variables within non-kernel functions.
                    // These local variables must be declared in a kernel,
                    // and then passed to non-kernel functions.
                    genotype, energy, run_id, calc_coords,
#if defined (DEBUG_ENERGY_KERNEL)
                    interE, intraE,
		                  #endif
                    // Gradient-related arguments
                    // Calculate gradients (forces) for intermolecular energy
                    // Derived from autodockdev/maps.py
                    cartesian_gradient, gradient, sFloatAccumulator, item_ct1,
                    cData);

		// =============================================================
		// =============================================================
		// =============================================================
		#if defined (DEBUG_ENERGY_ADADELTA)
		if (threadIdx.x == 0) {
			#if defined (PRINT_ADADELTA_ENERGIES)
			printf("\n");
			printf("%-10s %-10.6f \n", "intra: ",  intraE);
			printf("%-10s %-10.6f \n", "grids: ",  interE);
			printf("%-10s %-10.6f \n", "Energy: ", intraE + interE);
			#endif

			#if defined (PRINT_ADADELTA_GENES_AND_GRADS)
			for(uint i = 0; i < cData.dockpars.num_of_genes; i++) {
				if (i == 0) {
					printf("\n%s\n", "----------------------------------------------------------");
					printf("%13s %13s %5s %15s %15s\n", "gene_id", "gene.value", "|", "gene.grad", "(autodockdevpy units)");
				}
				printf("%13u %13.6f %5s %15.6f %15.6f\n", i, genotype[i], "|", gradient[i], (i<3)? (gradient[i]/0.375f):(gradient[i]*180.0f/PI_FLOAT));
			}
			#endif

			#if defined (PRINT_ADADELTA_ATOMIC_COORDS)
			for(uint i = 0; i < cData.dockpars.num_of_atoms; i++) {
				if (i == 0) {
					printf("\n%s\n", "----------------------------------------------------------");
					printf("%s\n", "Coordinates calculated by calcenergy.cl");
					printf("%12s %12s %12s %12s\n", "atom_id", "coords.x", "coords.y", "coords.z");
				}
				printf("%12u %12.6f %12.6f %12.6f\n", i, calc_coords_x[i], calc_coords_y[i], calc_coords_z[i]);
			}
			printf("\n");
			#endif
		}
		__threadfence();
		__syncthreads();
		#endif // DEBUG_ENERGY_ADADELTA

                for (int i = item_ct1.get_local_id(2);
                     i < cData.dockpars.num_of_genes;
                     i += item_ct1.get_local_range().get(2))
                {
                        if (energy <
                            *best_energy) // we need to be careful not to change
                                          // best_energy until we had a chance
                                          // to update the whole array
                                best_genotype[i] = genotype[i];

			// Accumulating gradient^2 (eq.8 in the paper)
			// square_gradient corresponds to E[g^2]
			square_gradient[i] = RHO * square_gradient[i] + (1.0f - RHO) * gradient[i] * gradient[i];

			// Computing update (eq.9 in the paper)
                        float delta =
                            -1.0f * gradient[i] *
                            sycl::sqrt((float)(square_delta[i] + EPSILON) /
                                       (float)(square_gradient[i] + EPSILON));

                        // Accumulating update^2
			// square_delta corresponds to E[dx^2]
			square_delta[i] = RHO * square_delta[i] + (1.0f - RHO) * delta * delta;

			// Applying update
			genotype[i] += delta;
		}
                /*
                DPCT1007:71: Migration of this CUDA API is not supported by the
                Intel(R) DPC++ Compatibility Tool.
                */
                __threadfence();
                item_ct1.barrier();

#if defined (DEBUG_SQDELTA_ADADELTA)
		if (/*(get_group_id(0) == 0) &&*/ (threadIdx.x == 0)) {
			for(int i = 0; i < cData.dockpars.num_of_genes; i++) {
				if (i == 0) {
					printf("\n%s\n", "----------------------------------------------------------");
					printf("%13s %20s %15s %15s %15s\n", "gene", "sq_grad", "delta", "sq_delta", "new.genotype");
				}
				printf("%13u %20.6f %15.6f %15.6f %15.6f\n", i, square_gradient[i], delta[i], square_delta[i], genotype[i]);
			}
		}
		__threadfence();
		__syncthreads();
		#endif

		// Updating number of ADADELTA iterations (energy evaluations)
		iteration_cnt = iteration_cnt + 1;
                if (item_ct1.get_local_id(2) == 0) {
                        if (energy < *best_energy)
                        {
                                *best_energy = energy;
#ifdef ADADELTA_AUTOSTOP
                                (*cons_succ)++;
                                *cons_fail = 0;
#endif
			}
#ifdef ADADELTA_AUTOSTOP
			else
			{
                                *cons_succ = 0;
                                (*cons_fail)++;
                        }
#endif

			#if defined (DEBUG_ADADELTA_MINIMIZER) || defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
			printf("%20s %10.6f\n", "new.energy: ", energy);
			#endif

			#if defined (DEBUG_ENERGY_ADADELTA)
			printf("%-18s [%-5s]---{%-5s}   [%-10.7f]---{%-10.7f}\n", "-ENERGY-KERNEL7-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
			#endif
#ifdef ADADELTA_AUTOSTOP
                        if (*cons_succ >= 4)
                        {
                                *rho *= LS_EXP_FACTOR;
                                *cons_succ = 0;
                        }
			else
			{
                                if (*cons_fail >= 4)
                                {
                                        *rho *= LS_CONT_FACTOR;
                                        *cons_fail = 0;
                                }
			}
#endif
		}
                /*
                DPCT1007:72: Migration of this CUDA API is not supported by the
                Intel(R) DPC++ Compatibility Tool.
                */
                __threadfence();
                item_ct1.barrier(); // making sure that iteration_cnt is up-to-date
#ifdef ADADELTA_AUTOSTOP
        } while ((iteration_cnt < cData.dockpars.max_num_of_iters) && (*rho > 0.01f));
#else
	} while (iteration_cnt < cData.dockpars.max_num_of_iters);
#endif
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// Mapping torsion angles
        for (uint32_t gene_counter = item_ct1.get_local_id(2) + 3;
             gene_counter < cData.dockpars.num_of_genes;
             gene_counter += item_ct1.get_local_range().get(2))
        {
		map_angle(best_genotype[gene_counter]);
	}

	// Updating old offspring in population
        /*
        DPCT1007:69: Migration of this CUDA API is not supported by the Intel(R)
        DPC++ Compatibility Tool.
        */
        __threadfence();
        item_ct1.barrier();

        offset = (run_id * cData.dockpars.pop_size + *entity_id) *
                 GENOTYPE_LENGTH_IN_GLOBMEM;
        for (uint gene_counter = item_ct1.get_local_id(2);
             gene_counter < cData.dockpars.num_of_genes;
             gene_counter += item_ct1.get_local_range().get(2))
        {
		pMem_conformations_next[gene_counter + offset] = best_genotype[gene_counter];
	}

	// Updating eval counter and energy
        if (item_ct1.get_local_id(2) == 0) {
                cData.pMem_evals_of_new_entities[run_id *
                                                     cData.dockpars.pop_size +
                                                 *entity_id] += iteration_cnt;
                pMem_energies_next[run_id * cData.dockpars.pop_size +
                                   *entity_id] = *best_energy;

#if defined (DEBUG_ADADELTA_MINIMIZER) || defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
		printf("\n");
		printf("Termination criteria: ( #adadelta-iters >= %-3u )\n", dockpars_max_num_of_iters);
		printf("-------> End of ADADELTA minimization cycle, num of energy evals: %u, final energy: %.6f\n", iteration_cnt, best_energy);
		#endif
	}
}


void gpu_gradient_minAD(
                        uint32_t blocks,
                        uint32_t threads,
                        float*   pMem_conformations_next,
                        float*   pMem_energies_next
                       )
{
	size_t sz_shared = (2 * sizeof(sycl::float3) * cpuData.dockpars.num_of_atoms) + (5 * cpuData.dockpars.num_of_genes * sizeof(float));
        /*
        DPCT1049:73: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                extern dpct::constant_memory<GpuData, 0> cData;

                cData.init();

                auto cData_ptr_ct1 = cData.get_ptr();

                sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(sz_shared), cgh);
                sycl::accessor<int, 0, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    entity_id_acc_ct1(cgh);
                sycl::accessor<float, 0, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    best_energy_acc_ct1(cgh);
                sycl::accessor<float, 0, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    sFloatAccumulator_acc_ct1(cgh);
                sycl::accessor<float, 0, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    rho_acc_ct1(cgh);
                sycl::accessor<int, 0, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    cons_succ_acc_ct1(cgh);
                sycl::accessor<int, 0, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    cons_fail_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                          sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                            gpu_gradient_minAD_kernel(
                                pMem_conformations_next, pMem_energies_next,
                                item_ct1, dpct_local_acc_ct1.get_pointer(),
                                *cData_ptr_ct1, entity_id_acc_ct1.get_pointer(),
                                best_energy_acc_ct1.get_pointer(),
                                sFloatAccumulator_acc_ct1.get_pointer(),
                                rho_acc_ct1.get_pointer(),
                                cons_succ_acc_ct1.get_pointer(),
                                cons_fail_acc_ct1.get_pointer());
                    });
        });
        /*
        DPCT1001:74: The statement could not be removed.
        */
        LAUNCHERROR("gpu_gradient_minAD_kernel");
}
