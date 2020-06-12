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
#include <limits>
#include <cstdint>

#define PI 3.14159265359

#define PHI 0x9e3779b9

typedef struct
//Struct which describes a quaternion.
{
	double q;
	double x;
	double y;
	double z;
} Quaternion;

#define trilin_interpol(cube, weights) (cube[0][0][0]*weights[0][0][0] +cube[1][0][0]*weights[1][0][0] +cube[0][1][0]*weights[0][1][0] +cube[1][1][0]*weights[1][1][0] +cube[0][0][1]*weights[0][0][1] +cube[1][0][1]*weights[1][0][1] +cube[0][1][1]*weights[0][1][1] +cube[1][1][1]*weights[1][1][1])
//macro that calculates the trilinear interpolation,
//the first parameter is a 2*2*2 array of the values of the function
//in the vertices of the cube,
//and the second one is a 2*2*2 array of the interpolation weights

int float2fracint(double, int);

long long float2fraclint(double, int);

//double timer_gets(void);

double distance(const double [], const double []);

void vec_point2line(const double [], const double [], const double [], double []);

void rotate(double [], const double [], const double [], const double*, int);

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

void get_trilininterpol_weights(double [][2][2], const double*, const double*, const double*);

void get_trilininterpol_weights_f(float [][2][2], const float*, const float*, const float*);

void print_binary_string(unsigned long long);

#ifndef _WIN32
int stricmp(const char*, const char*);

int strincmp(const char*, const char*, int);
#endif

class LocalRNG
{
	uint32_t Q[4096], i, c; // CMWC4096 variables

public:
	LocalRNG(){
#if defined (REPRO)
		init(8);
#else
		init(time(NULL));
#endif
	}

	void init(uint32_t x)
	{
		Q[0]=x;
		Q[1]=Q[0]+PHI;
		Q[2]=Q[1]+PHI;
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
