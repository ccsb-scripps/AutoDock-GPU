#ifndef FLOAT4STRUCT_HPP
#define FLOAT4STRUCT_HPP

struct float4struct
//Quarternion struct for floats (float4 equivalent for kokkos kernels)
{
        float x;
        float y;
        float z;
        float w;

        KOKKOS_INLINE_FUNCTION float4struct() {}
        KOKKOS_INLINE_FUNCTION float4struct(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
        KOKKOS_INLINE_FUNCTION float4struct operator + (const float4struct &p)  const { return float4struct(x+p.x, y+p.y, z+p.z, w+p.w); }
        KOKKOS_INLINE_FUNCTION float4struct operator - (const float4struct &p)  const { return float4struct(x-p.x, y-p.y, z-p.z, w-p.w); }

};

#endif
