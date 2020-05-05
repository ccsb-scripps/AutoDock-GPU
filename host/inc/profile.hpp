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
