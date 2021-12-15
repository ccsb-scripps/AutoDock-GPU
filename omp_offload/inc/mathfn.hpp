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


//#define DEBUG_ENERGY_KERNEL

// No needed to be included as all kernel sources are stringified

#ifndef MATHFN_H
#define MATHFN_H

#define invpi2 1.0f/(PI_TIMES_2)

#define fast_acos_a  9.78056e-05f
#define fast_acos_b -0.00104588f
#define fast_acos_c  0.00418716f
#define fast_acos_d -0.00314347f
#define fast_acos_e  2.74084f
#define fast_acos_f  0.370388f
#define fast_acos_o -(fast_acos_a+fast_acos_b+fast_acos_c+fast_acos_d)

#pragma omp declare target
inline float norm_3df(float x, float y, float z)
{
        return sqrtf(x*x +y*y +z*z);
}

inline float rnorm_3df(float x, float y, float z)
{
        return (1.0f/norm_3df(x, y, z));
}

inline float fmod_pi2(float x)
{
	return x-(int)(invpi2*x)*PI_TIMES_2;
}

inline float fast_acos(float cosine)
{
	float x=fabs(cosine);
	float x2=x*x;
	float x3=x2*x;
	float x4=x3*x;
	float ac=(((fast_acos_o*x4+fast_acos_a)*x3+fast_acos_b)*x2+fast_acos_c)*x+fast_acos_d+
		 fast_acos_e*sqrt(2.0f-sqrt(2.0f+2.0f*x))-fast_acos_f*sqrt(2.0f-2.0f*x);
	return copysign(ac,cosine) + (cosine<0.0f)*PI_FLOAT;
}

inline float4struct cross(float3struct& u, float3struct& v)
{
    float4struct result;
    result.x = u.y * v.z - v.y * u.z;
    result.y = v.x * u.z - u.x * v.z;
    result.z = u.x * v.y - v.x * u.y;
    result.w = 0.0f;
    return result;
}

inline float4struct cross(float4struct& u, float4struct& v)
{
    float4struct result;
    result.x = u.y * v.z - v.y * u.z;
    result.y = v.x * u.z - u.x * v.z;
    result.z = u.x * v.y - v.x * u.y;
    result.w = 0.0f;
    return result;
}

inline float4struct quaternion_multiply(float4struct a, float4struct b)
{
	float4struct result = { a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, // x
			  a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, // y
			  a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, // z
			  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z }; // w
	return result;
}
inline float4struct quaternion_rotate(float4struct v, float4struct rot)
{
	float4struct result;
	
	float4struct z = cross(rot,v);
    z.x *= 2.0f;
    z.y *= 2.0f;
    z.z *= 2.0f;
    float4struct c = cross(rot, z); 
    result.x = v.x + z.x * rot.w + c.x;
    result.y = v.y + z.y * rot.w + c.y;    
    result.z = v.z + z.z * rot.w + c.z;
    result.w = 0.0f;	
	return result;
}
#pragma omp end declare target

#endif


