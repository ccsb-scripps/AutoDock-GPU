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


struct float4struct
//Quarternion struct for floats (float4)
{
        float x;
        float y;
        float z;
        float w;

        inline float4struct() {}
        inline float4struct(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
        inline float4struct operator + (const float4struct &p)  const { return float4struct(x+p.x, y+p.y, z+p.z, w+p.w); }
        inline float4struct operator - (const float4struct &p)  const { return float4struct(x-p.x, y-p.y, z-p.z, w-p.w); }

};


struct float3struct
//Coordinate struct for floats (float4)
{
        float x;
        float y;
        float z;

        inline float3struct() {}
        inline float3struct(float x, float y, float z) : x(x), y(y), z(z) {}
        inline float3struct operator + (const float3struct &p)  const { return float3struct(x+p.x, y+p.y, z+p.z); }
        inline float3struct operator - (const float3struct &p)  const { return float3struct(x-p.x, y-p.y, z-p.z); }

};

