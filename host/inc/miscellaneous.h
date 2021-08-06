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


#ifndef MISCELLANEOUS_H_
#define MISCELLANEOUS_H_

#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <cstring>
#include <limits>
#include <cstdint>
#include <string>

#ifdef _WIN32
#include <processthreadsapi.h>
inline unsigned int processid() { return GetProcessId(); }
#else
// libgen.h contains basename() and dirname() from a fullpath name
// Specific: to open correctly grid map field fiels and associated files
// http://ask.systutorials.com/681/get-the-directory-path-and-file-name-from-absolute-path-linux
#include <libgen.h>
#include <unistd.h>
inline unsigned int processid() { return getpid(); }
#endif

#define PI 3.14159265359

#define PHI 0x9e3779b9

#ifdef USE_PIPELINE
#define para_printf(f_, ...) do { if(output==NULL){ printf((f_), ##__VA_ARGS__); } else { snprintf(outbuf, 256, (f_), ##__VA_ARGS__); *output += outbuf; } } while(false)
#else
#define para_printf(f_, ...) printf((f_), ##__VA_ARGS__)
#endif

typedef struct
{
	int  nr;            // this number starts at 1 and will be used to extend the base atom type nr
	char deriv_name[4]; // name of derivative atom type (3 chars max + 1 for \0)
	char base_name[4];  // name of base type
} deriv_atype;

typedef struct
{
	char   A[4];          // name of one interaction atom type (order is arbitrary)
	char   B[4];          // name of the other interaction atom type
	int    nr_parameters; // number of parameters (the order is unique)
	float* parameters;    // parameter array ([0] = r, [1] = eps, [2] = rep. LJ exponent, [3] = attr. LJ exponent)
} pair_mod;

// Struct which describes a quaternion.
typedef struct
{
	double q;
	double x;
	double y;
	double z;
} Quaternion;

float map2float(const char* c);

int float2fracint(double, int);

long long float2fraclint(double, int);

//double timer_gets(void);

double distance(const double [], const double []);

double distance2(const double [], const double []);

void vec_point2line(const double [], const double [], const double [], double []);

void rotate(double [], const double [], const double [], const double*, int);

std::string get_filepath(const char* filename);

#if 0
// -------------------------------------------------------------------
// Replacing rotation genes: from spherical space to Shoemake space
// gene [0:2]: translation -> kept as original x, y, z
// gene [3:5]: rotation    -> transformed into Shoemake (u1: adimensional, u2&u3: sexagesimal)
// gene [6:N]: torsions	   -> kept as original angles	(all in sexagesimal)

// Shoemake ranges:
// u1: [0, 1]
// u2: [0: 2PI] or [0: 360]

// Random generator in the host is changed:
// LCG (original, myrand()) -> CPP std (rand())
// -------------------------------------------------------------------
void rotate_shoemake(double [], const double [], const double [], int);
#endif

double angle_of_vectors(const double [], const double []);

void vec_crossprod(const double [], const double [], double []);

void print_binary_string(unsigned long long);

#ifndef _WIN32
int stricmp(const char*, const char*);

int strincmp(const char*, const char*, int);
#endif

class LocalRNG
{
	uint32_t Q[4096], i, c; // CMWC4096 variables

public:
	LocalRNG(uint32_t seed[3]){
		if(!seed[2]){
			if(!seed[1]){
				init(seed[0]);
			} else init(seed[0],seed[1]);
		} else init(seed[0],seed[1],seed[2]);
	}

	LocalRNG(){
#if defined (REPRO)
		init(8);
#else
		init(time(NULL),processid());
#endif
	}

	void init(uint32_t x)
	{
		init(x,x+PHI,x+PHI+PHI);
	}

	void init(uint32_t x, uint32_t y)
	{
		init(x,y,y+PHI);
	}

	void init(uint32_t x, uint32_t y, uint32_t z)
	{
		Q[0]=x;
		Q[1]=y+PHI;
		Q[2]=z+PHI;
		for(unsigned int j=3; j<4096; j++) Q[j]=Q[j-3] ^ Q[j-2] ^ PHI ^ j;
		i=4095;
		c=362436;
		do
			c=random_uint();
		while(c >= 809430660);
		i=4095;
	}

	// This function generates random numbers with CMWC4096 between 0..2^32-1
	// G. Marsaglia, JMASM May 2003, Vol 2, No 1, 2-13
	uint32_t random_uint()
	{
		uint64_t const a = 18782LL;
		uint32_t const m = 0xfffffffe;
		uint32_t x;
		uint64_t t;
		i = (i+1)&4095;
		t = a*Q[i]+c;
		c = (t>>32);
		x = t+c;
		if(x<c){
			x++;
			c++;
		}
		return(Q[i] = m-x);
	}

	float random_float(){
		float result = 0.0f;
		// Ensure random number is between 0 and 1
		while (result<=0.0f || result>=1.0f)
			result = (float)random_uint()/4294967295.0f;
		return result;
	}
};

#endif /* MISCELLANEOUS_H_ */
