#ifndef AUTOSTOP_HPP
#define AUTOSTOP_HPP

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

class AutoStop{
	bool first_time;
	bool autostopped;
        float threshold;
        float threshold_used;
        float thres_stddev;
        float curr_avg;
        float curr_std;
        unsigned int roll_count;
        float rolling_stddev;
        unsigned int bestN;
        const unsigned int Ntop;
	const unsigned int pop_size;
	const int num_of_runs;
	const float stopstd;
        const unsigned int Ncream;
        float delta_energy;
        float overall_best_energy;
	std::vector<float> rolling;
	std::vector<float> average_sd2_N;

	inline float average(float* average_sd2_N)
	{
		if(average_sd2_N[2]<1.0f)
			return 0.0;
		return average_sd2_N[0]/average_sd2_N[2];
	}

	inline float stddev(float* average_sd2_N)
	{
		if(average_sd2_N[2]<1.0f)
			return 0.0;
		float sq = average_sd2_N[1]*average_sd2_N[2]-average_sd2_N[0]*average_sd2_N[0];
		if((fabs(sq)<=0.000001) || (sq<0.0)) return 0.0;
		return sqrt(sq)/average_sd2_N[2];
	}

	inline void tabulate_energies(const float* energies){
		overall_best_energy = 1<<24;
		for (unsigned long run_cnt=0; run_cnt < num_of_runs; run_cnt++) {
			for (unsigned int i=0; i<pop_size; i++) {
				float energy = energies[run_cnt*pop_size + i];
				if(energy < overall_best_energy)
					overall_best_energy = energy;
				if(energy < threshold) {
					average_sd2_N[0] += energy;
					average_sd2_N[1] += energy * energy;
					average_sd2_N[2] += 1.0;
					for(unsigned int m=0; m<Ntop; m++)
						if(energy < (threshold-2.0*thres_stddev)+m*delta_energy) {
							average_sd2_N[3*(m+1)] += energy;
							average_sd2_N[3*(m+1)+1] += energy*energy;
							average_sd2_N[3*(m+1)+2] += 1.0;
							break; // only one entry per bin
						}
				}
			}
		}
	}

	inline void set_stats(){
		curr_avg = average(&average_sd2_N[0]);
		curr_std = stddev(&average_sd2_N[0]);
		bestN = average_sd2_N[2];
		average_sd2_N[0] = 0.0;
		average_sd2_N[1] = 0.0;
		average_sd2_N[2] = 0.0;
		unsigned int lowest_energy = 0;
		for(unsigned int m=0; m<Ntop; m++) {
			if((average_sd2_N[3*(m+1)+2]>=1.0) && (lowest_energy<Ncream)) {
				if((average_sd2_N[2]<4.0) || fabs(average(&average_sd2_N[0])-average(&average_sd2_N[3*(m+1)]))<2.0*stopstd) {
					average_sd2_N[0] += average_sd2_N[3*(m+1)];
					average_sd2_N[1] += average_sd2_N[3*(m+1)+1];
					average_sd2_N[2] += average_sd2_N[3*(m+1)+2];
					lowest_energy++;
				}
			}
		}

		if(lowest_energy>0) {
			curr_avg = average(&average_sd2_N[0]);
			curr_std = stddev(&average_sd2_N[0]);
			bestN = average_sd2_N[2];
		}
	}

	public:

	AutoStop(int pop_size_in, int num_of_runs_in, float stopstd_in)
		: rolling(4*3, 0), // Initialize to zero
		  average_sd2_N((pop_size_in+1)*3),
		  num_of_runs(num_of_runs_in),
		  pop_size(pop_size_in),
		  Ntop(pop_size_in),
		  Ncream(pop_size_in / 10),
		  stopstd(stopstd_in)
	{
		first_time = true;
		autostopped = false;
		threshold = 1<<24;
		thres_stddev = threshold;
		curr_avg = -(1<<24);
		curr_std = thres_stddev;
		roll_count = 0;
		bestN = 1;
		delta_energy = 2.0 * thres_stddev / Ntop;
	}

	inline void print_intro(unsigned long num_of_generations, unsigned long num_of_energy_evals)
	{
		printf("\nExecuting docking runs, stopping automatically after either reaching %.2f kcal/mol standard deviation\nof the best molecules, %lu generations, or %lu evaluations, whichever comes first:\n\n",stopstd,num_of_generations,num_of_energy_evals);
                printf("Generations |  Evaluations |     Threshold    |  Average energy of best 10%%  | Samples |    Best energy\n");
                printf("------------+--------------+------------------+------------------------------+---------+-------------------\n");
	}

	inline bool check_if_satisfactory(int generation_cnt, const float* energies, unsigned long total_evals)
	{
		for(unsigned int count=0; (count<1+8*(generation_cnt==0)) && (fabs(curr_avg)>0.00001); count++)
		{
			threshold_used = threshold;
			std::fill(average_sd2_N.begin(), average_sd2_N.end(), 0); // reset to 0
			tabulate_energies(energies); // Fills average_sd2_N and overall_best_energy
			if(first_time)
			{
				curr_avg = average(&average_sd2_N[0]);
				curr_std = stddev(&average_sd2_N[0]);
				bestN = average_sd2_N[2];
				thres_stddev = curr_std;
				threshold = curr_avg + thres_stddev;
				delta_energy = 2.0 * thres_stddev / (Ntop-1);
				first_time = false;
			}
			else
			{
				set_stats(); // set curr_avg, curr_std, bestN, average_sd2_N 

				if(curr_std<0.5f*stopstd)
					thres_stddev = stopstd;
				else
					thres_stddev = curr_std;
				threshold = curr_avg + Ncream * thres_stddev / bestN;
				delta_energy = 2.0 * thres_stddev / (Ntop-1);
			}
		}
		printf("%11u | %12lu |%8.2f kcal/mol |%8.2f +/-%8.2f kcal/mol |%8i |%8.2f kcal/mol\n",generation_cnt,total_evals/num_of_runs,threshold_used,curr_avg,curr_std,bestN,overall_best_energy);
		fflush(stdout);
		rolling[3*roll_count] = curr_avg * bestN;
		rolling[3*roll_count+1] = (curr_std*curr_std + curr_avg*curr_avg)*bestN;
		rolling[3*roll_count+2] = bestN;
		roll_count = (roll_count + 1) % 4;
		average_sd2_N[0] = rolling[0] + rolling[3] + rolling[6] + rolling[9];
		average_sd2_N[1] = rolling[1] + rolling[4] + rolling[7] + rolling[10];
		average_sd2_N[2] = rolling[2] + rolling[5] + rolling[8] + rolling[11];

		// Finish when the std.dev. of the last 4 rounds is below 0.1 kcal/mol
		if((stddev(&average_sd2_N[0])<stopstd) && (generation_cnt>30))
			autostopped = true;

		return autostopped;
	}

	inline void output_final_stddev(int generation_cnt, const float* energies, unsigned long total_evals){
		if (autostopped){
			printf("------------+--------------+------------------+------------------------------+---------+-------------------\n");
			printf("\n%43s evaluation after reaching\n%40.2f +/-%8.2f kcal/mol combined.\n%34i samples, best energy %8.2f kcal/mol.\n","Finished",average(&average_sd2_N[0]),stddev(&average_sd2_N[0]),(unsigned int)average_sd2_N[2],overall_best_energy);
		} else {
			// Stopped without autostop; output stddev statistics regardless

			tabulate_energies(energies);  // Fills average_sd2_N and overall_best_energy
			set_stats(); // set curr_avg, curr_std, bestN, average_sd2_N

			printf("%11u | %12u |%8.2f kcal/mol |%8.2f +/-%8.2f kcal/mol |%8i |%8.2f kcal/mol\n",generation_cnt,total_evals/num_of_runs,threshold,curr_avg,curr_std,bestN,overall_best_energy);
			printf("------------+--------------+------------------+------------------------------+---------+-------------------\n");
			printf("\n%43s evaluation after reaching\n%33u evaluations. Best energy %8.2f kcal/mol.\n","Finished",total_evals/num_of_runs,overall_best_energy);
		}
		fflush(stdout);
	}

	inline bool did_stop(){
		return autostopped;
	}

};

#endif
