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

	Profile(const int id_in) : id(id_in), capped(false), autostopped(false) {}

	void write_to_file(char* filename){
		char profile_file_name[256];
		strcpy(profile_file_name, filename);
		strcat(profile_file_name, ".timing");
		FILE* fp = fopen(profile_file_name, "a");
		if (id==0) fprintf(fp, "ID ADADELTA n_evals capped autostopped nev_at_stop num_atoms num_rotbonds exec_time");
		fprintf(fp, "\n%d %d %d %d %d %d %d %d %.3f", id, adadelta?1:0, n_evals, capped?1:0, autostopped, nev_at_stop, num_atoms, num_rotbonds, exec_time );
		fclose(fp);
	}

};

#endif
