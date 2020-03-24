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

template<class Device>
KOKKOS_INLINE_FUNCTION unsigned int my_rand(const member_type& team_member, const DockingParams<Device>& docking_params)
//The GPU device function generates a random int
//with a linear congruential generator.
//Each thread (supposing num_of_runs*pop_size blocks and NUM_OF_THREADS_PER_BLOCK threads per block)
//has its own state which is stored in the global memory area pointed by
//prng_states (thread with ID tx in block with ID bx stores its state in prng_states[bx*NUM_OF_THREADS_PER_BLOCK+$
//The random number generator uses the gcc linear congruential generator constants.
{
        unsigned int state;
	// Global ID
	int gidx = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

#if defined (REPRO)
        state = 1;
#else
        // Current state of the threads own PRNG
        state = docking_params.prng_states(gidx);

        // Calculating next state
        state = (RAND_A*state+RAND_C);
#endif
        // Saving next state to memory
        // prng_states[get_group_id(0)*NUM_OF_THREADS_PER_BLOCK + get_local_id(0)] = state;
        docking_params.prng_states(gidx) = state;

        return state;
}

template<class Device>
KOKKOS_INLINE_FUNCTION float rand_float(const member_type& team_member, const DockingParams<Device>& docking_params)
	//The GPU device function generates a
//random float greater than (or equal to) 0 and less than 1.
{
        float state;

        // State will be between 0 and 1
#if defined (REPRO)
        state = 0.55f; //0.55f;
#else
        state =  (my_rand(team_member, docking_params)/MAX_UINT)*0.999999f;
#endif

        return state;
}
