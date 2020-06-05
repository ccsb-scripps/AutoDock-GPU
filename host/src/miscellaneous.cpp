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






#include "miscellaneous.h"

int float2fracint(double toconv, int frac)
//The function converts a float value to a fixed pont fractional number in (32-frac).frac format,
//and returns it as an integer.
{
	if (toconv >= 0)
		return (int) floor(toconv*pow(2.0, frac));
	else
		return (int) ceil(toconv*pow(2.0, frac));
}

long long float2fraclint(double toconv, int frac)
//The function converts a float value to a fixed pont fractional number in (32-frac).frac format,
//and returns it as a long integer.
{
	if (toconv >= 0)
		return (long long) floor(toconv*pow(2.0, frac));
	else
		return (long long) ceil(toconv*pow(2.0, frac));
}

/*double timer_gets(void)
//The function returns the current time in seconds.
{
  struct timeval ts;
  double timesec;

  gettimeofday(&ts, (struct timezone*)0);
  timesec = ((double) ts.tv_sec*1000000000.0 + (double) ts.tv_usec*1000.0)/1000000000.0;
  return timesec;
}*/

double distance(const double point1 [], const double point2 [])
//Returns the distance between point1 and point2.
//The arrays have to store the x, y and z coordinates of the
//point, respectively.
{
	double sub1, sub2, sub3;
	sub1 = point1 [0] - point2 [0];
	sub2 = point1 [1] - point2 [1];
	sub3 = point1 [2] - point2 [2];
	return sqrt(sub1*sub1 + sub2*sub2 + sub3*sub3);
}

void vec_point2line(const double point [], const double line_pointA [], const double line_pointB [], double vec [])
//The function calculates the vector which moves a point given by the first parameter to its perpendicular projection
//on a line given by to of its points (line_pointA and line_pointB parameters). The result vector is the vec parameter.
{
	double posvec_of_line [3];
	double proj_of_point [3];
	double posvec_of_line_length2, temp;
	int i;

	//vector parallel to line
	for (i=0; i<3; i++)
		posvec_of_line[i] = line_pointB[i] - line_pointA[i];

	//length^2 of posvec_of_line
	posvec_of_line_length2 = pow(posvec_of_line[0], 2) +
				 pow(posvec_of_line[1], 2) +
				 pow(posvec_of_line[2], 2);

	temp = 0;
	for (i=0; i<3; i++)
		temp += posvec_of_line [i] * (point [i] - line_pointA [i]);
	temp = temp/posvec_of_line_length2;

	//perpendicular projection of point to the line
	for (i=0; i<3; i++)
		proj_of_point [i] = temp * posvec_of_line [i] + line_pointA [i];

	for (i=0; i<3; i++)
		vec [i] = proj_of_point [i] - point [i];
}

void rotate(double point [], const double movvec [], const double normvec [], const double* angle, int debug)
//The function rotates the point given by the first parameter around an axis
//which is parallel to vector normvec and which
//can be moved to the origo with vector movvec.
//The direction of rotation with angle is considered relative to normvec
//according to right hand rule. If debug is 1, debug messages will be printed to the screen.
{
	Quaternion quatrot_left, quatrot_right, quatrot_temp;
	double anglediv2, cos_anglediv2, sin_anglediv2;

	//the point must be moved according to moving vector
	point [0] = point [0] - movvec [0];
	point [1] = point [1] - movvec [1];
	point [2] = point [2] - movvec [2];

	if (debug == 1)
	{
		printf("Moving vector coordinates (x,y,z): %lf, %lf, %lf\n",
						  movvec [0], movvec [1], movvec [2]);
		printf("Unit vector coordinates (x,y,z): %lf, %lf, %lf\n",
						  normvec [0], normvec [1], normvec [2]);
	}

	//Related equations:
	//q = quater_w+i*quater_x+j*quater_y+k*quater_z
	//v = i*point_x+j*point_y+k*point_z
	//The coordinates of the rotated point can be calculated as:
	//q*v*(q^-1), where
	//q^-1 = quater_w-i*quater_x-j*quater_y-k*quater_z
	//and * is the quaternion multiplication defined as follows:
	//(a1+i*b1+j*c1+k*d1)*(a2+i*b2+j*c2+k*d2) = (a1a2-b1b2-c1c2-d1d2)+
	//i*(a1b2+a2b1+c1d2-c2d1)+
	//j*(a1c2+a2c1+b2d1-b1d2)+
	//k*(a1d2+a2d1+b1c2-b2c1)

	anglediv2 = (*angle)/2/180*PI;
	cos_anglediv2 = cos(anglediv2);
	sin_anglediv2 = sin(anglediv2);

	//rotation quaternion
	quatrot_left.q = cos_anglediv2;
	quatrot_left.x = sin_anglediv2*normvec [0];
	quatrot_left.y = sin_anglediv2*normvec [1];
	quatrot_left.z = sin_anglediv2*normvec [2];

	//inverse of rotation quaternion
	quatrot_right.q = quatrot_left.q;
	quatrot_right.x = -1*quatrot_left.x;
	quatrot_right.y = -1*quatrot_left.y;
	quatrot_right.z = -1*quatrot_left.z;

	if (debug == 1)
	{
		printf("q (w,x,y,z): %lf, %lf, %lf, %lf\n",
			quatrot_left.q, quatrot_left.x, quatrot_left.y, quatrot_left.z);
		printf("q^-1 (w,x,y,z): %lf, %lf, %lf, %lf\n",
			quatrot_right.q, quatrot_right.x, quatrot_right.y, quatrot_right.z);
		printf("v (w,x,y,z): %lf, %lf, %lf, %lf\n",
			0.0, point [0], point [1], point [2]);
	}

	//Quaternion multiplications
	//Since the q field of v is 0 as well as the result's q element,
	//simplifications can be made...
	quatrot_temp.q = 0 -
			 quatrot_left.x*point [0] -
			 quatrot_left.y*point [1] -
			 quatrot_left.z*point [2];
	quatrot_temp.x = quatrot_left.q*point [0] +
			 0 +
			 quatrot_left.y*point [2] -
			 quatrot_left.z*point [1];
	quatrot_temp.y = quatrot_left.q*point [1] -
			 quatrot_left.x*point [2] +
			 0 +
			 quatrot_left.z*point [0];
	quatrot_temp.z = quatrot_left.q*point [2] +
			 quatrot_left.x*point [1] -
			 quatrot_left.y*point [0] +
			0;

	if (debug == 1)
		printf("q*v (w,x,y,z): %lf, %lf, %lf, %lf\n",
		        quatrot_temp.q, quatrot_temp.x, quatrot_temp.y, quatrot_temp.z);

	point [0] = quatrot_temp.q*quatrot_right.x +
		    quatrot_temp.x*quatrot_right.q +
		    quatrot_temp.y*quatrot_right.z -
		    quatrot_temp.z*quatrot_right.y;
	point [1] = quatrot_temp.q*quatrot_right.y -
		    quatrot_temp.x*quatrot_right.z +
		    quatrot_temp.y*quatrot_right.q +
		    quatrot_temp.z*quatrot_right.x;
	point [2] = quatrot_temp.q*quatrot_right.z +
		    quatrot_temp.x*quatrot_right.y -
		    quatrot_temp.y*quatrot_right.x +
		    quatrot_temp.z*quatrot_right.q;

	if (debug == 1)
		printf("q*v*q^-1 (w,x,y,z): %lf, %lf, %lf, %lf\n",
			0.0, point [0], point [1], point [2]);

	//Moving the point back
	point [0] = point [0] + movvec [0];
	point [1] = point [1] + movvec [1];
	point [2] = point [2] + movvec [2];

	if (debug == 1)
		printf("rotated point (x,y,z): %lf, %lf, %lf\n\n",
			point [0], point [1], point [2]);
}

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
void rotate_shoemake(double point [], 
		    const double movvec [], 
		    const double shoemake [],
	//	    const double normvec [], 
	//	    const double* angle, 
		    int debug)
//The function rotates the point given by the first parameter around an axis
//which is parallel to vector normvec and which
//can be moved to the origo with vector movvec.
//The direction of rotation with angle is considered relative to normvec
//according to right hand rule. If debug is 1, debug messages will be printed to the screen.
{
	Quaternion quatrot_left, quatrot_right, quatrot_temp;

/*
	double anglediv2, cos_anglediv2, sin_anglediv2;
*/

	//the point must be moved according to moving vector
	point [0] = point [0] - movvec [0];
	point [1] = point [1] - movvec [1];
	point [2] = point [2] - movvec [2];

	if (debug == 1)
	{
		printf("Moving vector coordinates (x,y,z): %lf, %lf, %lf\n",
						  movvec [0], movvec [1], movvec [2]);
/*
		/printf("Unit vector coordinates (x,y,z): %lf, %lf, %lf\n",
						  normvec [0], normvec [1], normvec [2]);
*/
	}

	//Related equations:
	//q = quater_w+i*quater_x+j*quater_y+k*quater_z
	//v = i*point_x+j*point_y+k*point_z
	//The coordinates of the rotated point can be calculated as:
	//q*v*(q^-1), where
	//q^-1 = quater_w-i*quater_x-j*quater_y-k*quater_z
	//and * is the quaternion multiplication defined as follows:
	//(a1+i*b1+j*c1+k*d1)*(a2+i*b2+j*c2+k*d2) = (a1a2-b1b2-c1c2-d1d2)+
	//i*(a1b2+a2b1+c1d2-c2d1)+
	//j*(a1c2+a2c1+b2d1-b1d2)+
	//k*(a1d2+a2d1+b1c2-b2c1)

/*
	anglediv2 = (*angle)/2/180*PI;
	cos_anglediv2 = cos(anglediv2);
	sin_anglediv2 = sin(anglediv2);

	//rotation quaternion
	quatrot_left.q = cos_anglediv2;
	quatrot_left.x = sin_anglediv2*normvec [0];
	quatrot_left.y = sin_anglediv2*normvec [1];
	quatrot_left.z = sin_anglediv2*normvec [2];
*/
	// Rotation quaternion from Shoemake input (which MUST be already in radians)
	double u1, u2, u3;
	u1 = shoemake[0];
	u2 = shoemake[1];
	u3 = shoemake[2];

	quatrot_left.q = sqrt(1-u1) * sin(u2);
	quatrot_left.x = sqrt(1-u1) * cos(u2);
	quatrot_left.y = sqrt(u1)   * sin(u3);
	quatrot_left.z = sqrt(u1)   * cos(u3);

	//inverse of rotation quaternion
	quatrot_right.q = quatrot_left.q;
	quatrot_right.x = -1*quatrot_left.x;
	quatrot_right.y = -1*quatrot_left.y;
	quatrot_right.z = -1*quatrot_left.z;

	if (debug == 1)
	{
		printf("q (w,x,y,z): %lf, %lf, %lf, %lf\n",
			quatrot_left.q, quatrot_left.x, quatrot_left.y, quatrot_left.z);
		printf("q^-1 (w,x,y,z): %lf, %lf, %lf, %lf\n",
			quatrot_right.q, quatrot_right.x, quatrot_right.y, quatrot_right.z);
		printf("v (w,x,y,z): %lf, %lf, %lf, %lf\n",
			0.0, point [0], point [1], point [2]);
	}

	//Quaternion multiplications
	//Since the q field of v is 0 as well as the result's q element,
	//simplifications can be made...
	quatrot_temp.q = 0 -
			 quatrot_left.x*point [0] -
			 quatrot_left.y*point [1] -
			 quatrot_left.z*point [2];
	quatrot_temp.x = quatrot_left.q*point [0] +
			 0 +
			 quatrot_left.y*point [2] -
			 quatrot_left.z*point [1];
	quatrot_temp.y = quatrot_left.q*point [1] -
			 quatrot_left.x*point [2] +
			 0 +
			 quatrot_left.z*point [0];
	quatrot_temp.z = quatrot_left.q*point [2] +
			 quatrot_left.x*point [1] -
			 quatrot_left.y*point [0] +
			0;

	if (debug == 1)
		printf("q*v (w,x,y,z): %lf, %lf, %lf, %lf\n",
		        quatrot_temp.q, quatrot_temp.x, quatrot_temp.y, quatrot_temp.z);

	point [0] = quatrot_temp.q*quatrot_right.x +
		    quatrot_temp.x*quatrot_right.q +
		    quatrot_temp.y*quatrot_right.z -
		    quatrot_temp.z*quatrot_right.y;
	point [1] = quatrot_temp.q*quatrot_right.y -
		    quatrot_temp.x*quatrot_right.z +
		    quatrot_temp.y*quatrot_right.q +
		    quatrot_temp.z*quatrot_right.x;
	point [2] = quatrot_temp.q*quatrot_right.z +
		    quatrot_temp.x*quatrot_right.y -
		    quatrot_temp.y*quatrot_right.x +
		    quatrot_temp.z*quatrot_right.q;

	if (debug == 1)
		printf("q*v*q^-1 (w,x,y,z): %lf, %lf, %lf, %lf\n",
			0.0, point [0], point [1], point [2]);

	//Moving the point back
	point [0] = point [0] + movvec [0];
	point [1] = point [1] + movvec [1];
	point [2] = point [2] + movvec [2];

	if (debug == 1)
		printf("rotated point (x,y,z): %lf, %lf, %lf\n\n",
			point [0], point [1], point [2]);
}
#endif

double angle_of_vectors(const double vector1 [], const double vector2 [])
//The function's inputs are two position vectors (whose starting point is the origo).
//The function returns the angle between them.
{
	int i;
	double len_vec1, len_vec2, scalmul;
	double zerovec [3] = {0, 0, 0};
	double temp;

	scalmul = 0;

	len_vec1 = distance(vector1, zerovec);
	len_vec2 = distance(vector2, zerovec);

	for (i=0; i<3; i++)
		scalmul += vector1 [i]*vector2 [i];

	temp = scalmul/(len_vec1*len_vec2);

	if (temp > 1)  temp =  1;
	if (temp < -1) temp = -1;

	return (acos(temp)*180/PI);
}

void vec_crossprod(const double vector1 [], const double vector2 [], double crossprodvec [])
//The function calculates the cross product of position vectors vector1 and vector2, and returns
//it in the third parameter.
{
	crossprodvec [0] = vector1 [1]*vector2 [2] - vector1 [2]*vector2 [1];
	crossprodvec [1] = vector1 [2]*vector2 [0] - vector1 [0]*vector2 [2];
	crossprodvec [2] = vector1 [0]*vector2 [1] - vector1 [1]*vector2 [0];
}

void get_trilininterpol_weights(double weights [][2][2], const double* dx, const double* dy, const double* dz)
//The function calculates the weights for trilinear interpolation based on the location of the point inside
//the cube which is given by the second, third and fourth parameters.
{
	weights [0][0][0] = (1-(*dx))*(1-(*dy))*(1-(*dz));
	weights [1][0][0] = (*dx)*(1-(*dy))*(1-(*dz));
	weights [0][1][0] = (1-(*dx))*(*dy)*(1-(*dz));
	weights [1][1][0] = (*dx)*(*dy)*(1-(*dz));
	weights [0][0][1] = (1-(*dx))*(1-(*dy))*(*dz);
	weights [1][0][1] = (*dx)*(1-(*dy))*(*dz);
	weights [0][1][1] = (1-(*dx))*(*dy)*(*dz);
	weights [1][1][1] = (*dx)*(*dy)*(*dz);
}

void get_trilininterpol_weights_f(float weights [][2][2], const float* dx, const float* dy, const float* dz)
//The function calculates the weights for trilinear interpolation based on the location of the point inside
//the cube which is given by the second, third and fourth parameters.
{
	weights [0][0][0] = (1-(*dx))*(1-(*dy))*(1-(*dz));
	weights [1][0][0] = (*dx)*(1-(*dy))*(1-(*dz));
	weights [0][1][0] = (1-(*dx))*(*dy)*(1-(*dz));
	weights [1][1][0] = (*dx)*(*dy)*(1-(*dz));
	weights [0][0][1] = (1-(*dx))*(1-(*dy))*(*dz);
	weights [1][0][1] = (*dx)*(1-(*dy))*(*dz);
	weights [0][1][1] = (1-(*dx))*(*dy)*(*dz);
	weights [1][1][1] = (*dx)*(*dy)*(*dz);
}

void print_binary_string(unsigned long long to_print)
//The function prints out the value of to_print parameter to the standart io as a binary number.
{
	unsigned long long temp;
	int i;

	temp = 1;
	temp = (temp << 63);

	for (i=0; i<64; i++)
	{
		if ((temp & to_print) != 0)
			printf("1");
		else
			printf("0");
	temp = (temp >> 1);
	}
}

#ifndef _WIN32
// This was disabled for Windows
int stricmp(const char* str1, const char* str2)
//The function compares the two input strings and
//returns 0 if they are identical (case-UNsensitive)
//and 1 if not.
{
	const char* c1_poi;
	const char* c2_poi;
	char c1;
	char c2;
	char isdifferent = 0;

	c1_poi = str1;
	c2_poi = str2;

	c1 = *c1_poi;
	c2 = *c2_poi;

	while ((c1 != '\0') && (c2 != '\0'))
	{
		if (toupper(c1) != toupper(c2))
		{
			isdifferent = 1;
			break;
		}

		c1_poi++;
		c2_poi++;

		c1 = *c1_poi;
		c2 = *c2_poi;
	}

	if (toupper(c1) != toupper(c2))
		isdifferent = 1;

	return isdifferent;
}

int strincmp(const char* str1, const char* str2, int num)
//The function compares up to num characters of two input
//strings and returns 0 if they are identical (case-UNsensitive)
//and 1 if not.
{
	const char* c1_poi;
	const char* c2_poi;
	char c1;
	char c2;
	char count = 1; // the test at the end counts too

	c1_poi = str1;
	c2_poi = str2;

	c1 = *c1_poi;
	c2 = *c2_poi;
	
	while ((c1 != '\0') && (c2 != '\0') && (count<num))
	{
		if (toupper(c1) != toupper(c2))
			return 1;

		c1_poi++;
		c2_poi++;

		c1 = *c1_poi;
		c2 = *c2_poi;
		count++;
	}

	if (toupper(c1) != toupper(c2))
		return 1;

	return 0;
}
#endif
