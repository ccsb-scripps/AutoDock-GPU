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


#ifndef PROFILE_HPP
#define PROFILE_HPP

#include <string.h>
#include <vector>
#include <cmath>

struct Profile{
	public:
	int id;
	bool adadelta;
	int n_evals;
	bool capped;
	bool autostopped;
	int nev_at_stop;
	int num_atoms;
	int num_rotbonds;
	float exec_time;

	Profile(const int id_in) : id(id_in), capped(false), autostopped(false), exec_time(-1.0f) {}

	void write_to_file(FILE* fp){
		int success = (exec_time>=0.0f ? 1 : 0);
                float real_exec_time = (exec_time>=0.0f ? exec_time : 0.0f);
                fprintf(fp, "\n%d %d %d %d %d %d %d %d %d %.3f", id, adadelta?1:0, n_evals, capped?1:0, autostopped, nev_at_stop, num_atoms, num_rotbonds, success, real_exec_time );
	}
};

class Profiler{
	public:
	std::vector<Profile> p;

	void write_profiles_to_file(char* filename){
		char profile_file_name[256];
	        strcpy(profile_file_name, filename);
	        strcat(profile_file_name, ".timing");
	        FILE* fp = fopen(profile_file_name, "a");
	        fprintf(fp, "ID ADADELTA n_evals capped autostopped nev_at_stop num_atoms num_rotbonds successful exec_time");
		for (int i=0;i<p.size();i++){
		        p[i].write_to_file(fp);
		}
	        fclose(fp);
	}
};
#endif
