/*
 * (C) 2013. Evopro Innovation Kft.
 *
 * miscallenous.h
 *
 *  Created on: 2008.09.29.
 *      Author: pechan.imre
 */

#ifndef MISCELLANEOUS_H_
#define MISCELLANEOUS_H_

#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>

#define PI 3.1415926535

//Visual C++ linear congruential generator constants
#define RAND_A_GS 214013u
#define RAND_C_GS 2531011u

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

double myrand(void);

unsigned int myrand_int(unsigned int);

double distance(const double [], const double []);

void vec_point2line(const double [], const double [], const double [], double []);

void rotate(double [], const double [], const double [], const double*, int);

double angle_of_vectors(const double [], const double []);

void vec_crossprod(const double [], const double [], double []);

void get_trilininterpol_weights(double [][2][2], const double*, const double*, const double*);

void get_trilininterpol_weights_f(float [][2][2], const float*, const float*, const float*);

void print_binary_string(unsigned long long);

int stricmp(const char*, const char*);


unsigned int genseed(unsigned int init);

#endif /* MISCELLANEOUS_H_ */
