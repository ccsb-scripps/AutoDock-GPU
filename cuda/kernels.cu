/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.
For the ROCm/HIP port, Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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


#include <cstdint>
#include <cassert>
#include <cstring>
#include "cuda_to_hip.h"
#include "defines.h"
#include "calcenergy.h"
#include "GpuData.h"

// Warp/wavefront width used by the device-side warp reductions below. NVIDIA
// warps are always 32 lanes; AMD wavefronts are 64 on CDNA (gfx90a/gfx94x,
// __GFX9__) and 32 on RDNA. __GFX9__ is defined only during device compilation.
#if defined(USE_HIP)
#if defined(__GFX9__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
#else
#define WARP_SIZE 32
#endif

__device__ inline uint64_t llitoulli(int64_t l)
{
	uint64_t u;
#if defined(USE_HIP)
	memcpy(&u, &l, sizeof(u)); // HIP has no NVIDIA PTX; this is a pure bit-cast
#else
	asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
#endif
	return u;
}

__device__ inline int64_t ullitolli(uint64_t u)
{
	int64_t l;
#if defined(USE_HIP)
	memcpy(&l, &u, sizeof(l));
#else
	asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
#endif
	return l;
}


// Full-wavefront lane mask for the sync-shuffle/vote intrinsics. CUDA's
// __shfl_sync/__any_sync take a 32-bit mask; HIP requires a 64-bit mask for all
// wavefront-sync intrinsics (it static_asserts sizeof(mask)==8), independent of
// the active wave width.
#if defined(USE_HIP)
#define WARP_FULL_MASK 0xffffffffffffffffULL
#else
#define WARP_FULL_MASK 0xffffffffU
#endif

#define WARPMINIMUMEXCHANGE(tgx, v0, k0, mask) \
	{ \
		float v1    = v0; \
		int k1      = k0; \
		int otgx    = tgx ^ mask; \
		float v2    = __shfl_sync(WARP_FULL_MASK, v0, otgx); \
		int k2      = __shfl_sync(WARP_FULL_MASK, k0, otgx); \
		int flag    = ((v1 < v2) ^ (tgx > otgx)) && (v1 != v2); \
		k0          = flag ? k1 : k2; \
		v0          = flag ? v1 : v2; \
	}

// The last exchange (stride 32) only exists on a 64-lane wavefront; on a 32-lane
// warp lanes are already fully reduced after the stride-16 step.
#if WARP_SIZE > 32
#define WARPMINIMUM_TOP(tgx, v0, k0) WARPMINIMUMEXCHANGE(tgx, v0, k0, 32)
#else
#define WARPMINIMUM_TOP(tgx, v0, k0)
#endif

#define WARPMINIMUM2(tgx, v0, k0) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 1) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 2) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 4) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 8) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 16) \
	WARPMINIMUM_TOP(tgx, v0, k0)

// Stride-32 shuffle-add term: present only on a 64-lane wavefront.
#if WARP_SIZE > 32
#define WARPSUM_TOP(value, tgx) value += __shfl_sync(WARP_FULL_MASK, value, (tgx) ^ 32);
#else
#define WARPSUM_TOP(value, tgx)
#endif

#define REDUCEINTEGERSUM(value, pAccumulator) \
	if (threadIdx.x == 0) \
	{ \
		*pAccumulator = 0; \
	} \
	__threadfence(); \
	__syncthreads(); \
	if (__any_sync(WARP_FULL_MASK, value != 0)) \
	{ \
		uint32_t tgx            = threadIdx.x & cData.warpmask; \
		value                  += __shfl_sync(WARP_FULL_MASK, value, tgx ^ 1); \
		value                  += __shfl_sync(WARP_FULL_MASK, value, tgx ^ 2); \
		value                  += __shfl_sync(WARP_FULL_MASK, value, tgx ^ 4); \
		value                  += __shfl_sync(WARP_FULL_MASK, value, tgx ^ 8); \
		value                  += __shfl_sync(WARP_FULL_MASK, value, tgx ^ 16); \
		WARPSUM_TOP(value, tgx) \
		if (tgx == 0) \
		{ \
			atomicAdd(pAccumulator, value); \
		} \
	} \
	__threadfence(); \
	__syncthreads(); \
	value = *pAccumulator; \
	__syncthreads();

#define ATOMICADDI32(pAccumulator, value) atomicAdd(pAccumulator, (value))
#define ATOMICSUBI32(pAccumulator, value) atomicAdd(pAccumulator, -(value))
#define ATOMICADDF32(pAccumulator, value) atomicAdd(pAccumulator, (value))
#define ATOMICSUBF32(pAccumulator, value) atomicAdd(pAccumulator, -(value))

#ifdef USE_NVTENSOR
/* Begin: Reduction using tensor units */

// Implementation based on M.Sc. thesis by Gabin Schieffer at KTH:
// "Accelerating a Molecular Docking Application by Leveraging Modern Heterogeneous Computing Systemx"
// https://www.diva-portal.org/smash/get/diva2:1786161/FULLTEXT01.pdf

#if defined(__HIP_PLATFORM_AMD__)
	// AMD path: rocWMMA, the fragment-API analog of nvcuda::wmma. Two device
	// branches in reduce_via_tensor_units, keyed on the architecture:
	//   - CDNA (gfx9, __GFX9__): native single-precision MFMA
	//     (amdgcn_mfma_f32_16x16x4f32), so the reduction runs in EXACT fp32 and the
	//     tf32 split-and-correct error compensation the NVIDIA path uses is dropped.
	//     rocWMMA composes the 16x16x16 fragment shape from the native K=4 MFMA.
	//   - RDNA (gfx11+, wave32): WMMA has no fp32->fp32 intrinsic; only f16/bf16
	//     inputs are accepted. fp32 is emulated with bf16 error correction, the same
	//     scheme the NVIDIA tcec path uses: each fp32 operand is split into a hi
	//     bf16 and a lo bf16 residual, and the cross products are accumulated in an
	//     fp32 accumulator. The exact operands here (the all-ones and the identity)
	//     have a zero lo part, so only the data/partial-sum operands are split.
	// On Windows, windows.h defines min/max as macros; rocWMMA's float8.hpp has
	// min()/max() static methods that conflict with them. Undefine here before
	// the include; this is a no-op on Linux.
	#ifdef min
	#undef min
	#endif
	#ifdef max
	#undef max
	#endif
	#include <rocwmma/rocwmma.hpp>
	namespace adwmma = rocwmma;
	using wmma_float = float;
	using wmma_bf16  = rocwmma::bfloat16_t;

	// rocWMMA load/store_matrix_sync are per-lane LDS ops with no implicit
	// barrier (unlike nvcuda's .sync collectives), so every LDS round-trip
	// between fragment ops needs explicit ordering. Only one wavefront runs the
	// reduction, so the barrier must be wavefront-scoped: a workgroup __syncthreads
	// would deadlock when the block has more than one wavefront (e.g. NUMWI=128)
	// because the other wavefronts never enter this branch.
	__device__ __forceinline__ void wavefront_lds_barrier() {
		__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
		__builtin_amdgcn_wave_barrier();
		__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
	}
#else
	/*
	* WMMA Extension for single precision matmul using Tensor Cores
	* and error correction technique (TCEC)
	* https://github.com/wmmae/wmma_extension/blob/main/docs/mma_f32.md
	*/
	#include <wmma_extension/tcec/tcec.hpp>
	using tf32 = nvcuda::wmma::precision::tf32;

/*
 * Tensor Cores
 * https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9
 *
 * Don't forget to compile specifying the architecture, e.g., sm_86.
 * For AutoDock-GPU, this can be done via the TARGETS option.
 * make DEVICE=GPU TESTLS=ad NUMWI=64 TARGETS=86 test
 * https://stackoverflow.com/a/53634598/1616865
 */
#include <mma.h>
using namespace nvcuda;
#endif

#define TILE_SIZE (16 * 16)

constexpr int rowscols_M = 16;	// Number of rows (or cols) in the M dimension
constexpr int rowscols_N = 16;	// Number of rows (or cols) in the N dimension
constexpr int rowscols_K = 16;	// Number of rows (or cols) in the K dimension

__device__ void reduce_via_tensor_units(float *data_to_be_reduced) {
	__syncthreads();

#if defined(__HIP_PLATFORM_AMD__)
#if defined(__GFX9__)
	// CDNA (gfx9): native single-precision MFMA. nvcuda WMMA is a 32-lane (one
	// warp) collective; CDNA MFMA is a 64-lane (full wavefront) collective, so the
	// whole wavefront must enter and the identity tile is filled by index math over
	// the lanes of a wave64, not a wave32.
	if (threadIdx.x < 64) { // One wavefront performs the reduction
		__shared__ __align__ (256) float Q_square[TILE_SIZE]; // 16x16 matrix, reused for the 4x4-tiled I4 matrix

		adwmma::fragment<adwmma::matrix_b, rowscols_M, rowscols_N, rowscols_K, wmma_float, adwmma::col_major> frag_P;
		adwmma::fragment<adwmma::accumulator, rowscols_M, rowscols_N, rowscols_K, wmma_float> frag_V;

		adwmma::fragment<adwmma::matrix_a, rowscols_M, rowscols_N, rowscols_K, wmma_float, adwmma::col_major> frag_Q;
		adwmma::fragment<adwmma::matrix_b, rowscols_M, rowscols_N, rowscols_K, wmma_float, adwmma::col_major> frag_W;
		adwmma::fragment<adwmma::accumulator, rowscols_M, rowscols_N, rowscols_K, wmma_float> frag_C;

		adwmma::fill_fragment(frag_P, 1.0f); // P: only ones
		adwmma::fill_fragment(frag_V, 0.0f); // Output: initialize to zeros
		adwmma::fill_fragment(frag_C, 0.0f); // Final result

		// 1. Accumulate the values: V <- AP + V
		for(uint i = 0; i < (4 * NUM_OF_THREADS_PER_BLOCK)/TILE_SIZE; i++){
			const unsigned int offset = i * TILE_SIZE;

			adwmma::fragment<adwmma::matrix_a, rowscols_M, rowscols_N, rowscols_K, wmma_float, adwmma::col_major> frag_A;
			adwmma::load_matrix_sync(frag_A, data_to_be_reduced + offset, 16);
			adwmma::mma_sync(frag_V, frag_A, frag_P, frag_V);
		}

		// W <- V (required since we need V as a "matrix_b")
		adwmma::store_matrix_sync(Q_square, frag_V, 16, adwmma::mem_col_major);
		wavefront_lds_barrier();
		adwmma::load_matrix_sync(frag_W, Q_square, 16);
		wavefront_lds_barrier(); // W is loaded before the fill below clobbers Q_square

		// 2. Perform line sum: C <- QW + C (zero)
		//    a) build the 4x4-tiled matrix that holds a 4x4 identity in each tile.
		//       Derive each entry from its column-major offset so the fill is
		//       wave-size independent (the wave32 8-per-lane index math the NVIDIA
		//       path uses does not cover a wave64's 256 entries): Q[row][col]=1
		//       iff row%4==col%4.
		for(uint o = threadIdx.x; o < TILE_SIZE; o += 64){
			Q_square[o] = (((o % 16) & 3) == ((o / 16) & 3)) ? 1.0f : 0.0f;
		}
		wavefront_lds_barrier();
		adwmma::load_matrix_sync(frag_Q, Q_square, 16);
		//    b) perform sum
		adwmma::mma_sync(frag_C, frag_Q, frag_W, frag_C);

		// 3. Store result in shared memory
		adwmma::store_matrix_sync(data_to_be_reduced, frag_C, 16, adwmma::mem_col_major);
	}
#else
	// RDNA (gfx11+, wave32): WMMA accepts only f16/bf16 inputs, so fp32 is emulated
	// with bf16 error correction (the same scheme as the NVIDIA tcec path). Each
	// fp32 operand v is split as hi = (bf16)v, lo = (bf16)(v - (float)hi); the
	// cross products hi*other + lo*other accumulate in the fp32 accumulator. The
	// all-ones (frag_P) and the identity (frag_Q) are exact in bf16 (lo == 0), so
	// only the data tiles (stage 1) and the partial-sum tile W (stage 2) are split.
	// WMMA is a 32-lane (single wave32 wavefront) collective; the wavefront gate,
	// the identity-fill stride, and the LDS barriers all use WARP_SIZE so they are
	// correct independent of the wave width.
	if (threadIdx.x < WARP_SIZE) { // One wavefront performs the reduction
		__shared__ __align__ (256) float Q_square[TILE_SIZE]; // 16x16 matrix, reused for the 4x4-tiled I4 matrix

		adwmma::fragment<adwmma::matrix_b, rowscols_M, rowscols_N, rowscols_K, wmma_bf16, adwmma::col_major> frag_P; // ones (exact)
		adwmma::fragment<adwmma::accumulator, rowscols_M, rowscols_N, rowscols_K, wmma_float> frag_V;

		adwmma::fragment<adwmma::matrix_a, rowscols_M, rowscols_N, rowscols_K, wmma_bf16, adwmma::col_major> frag_Q; // identity (exact)
		adwmma::fragment<adwmma::matrix_b, rowscols_M, rowscols_N, rowscols_K, wmma_bf16, adwmma::col_major> frag_W_hi;
		adwmma::fragment<adwmma::matrix_b, rowscols_M, rowscols_N, rowscols_K, wmma_bf16, adwmma::col_major> frag_W_lo;
		adwmma::fragment<adwmma::accumulator, rowscols_M, rowscols_N, rowscols_K, wmma_float> frag_C;

		adwmma::fill_fragment(frag_P, (wmma_bf16)1.0f); // P: only ones
		adwmma::fill_fragment(frag_V, 0.0f); // Output: initialize to zeros
		adwmma::fill_fragment(frag_C, 0.0f); // Final result

		// 1. Accumulate the values: V <- AP + V, with A split hi+lo (P exact).
		for(uint i = 0; i < (4 * NUM_OF_THREADS_PER_BLOCK)/TILE_SIZE; i++){
			const unsigned int offset = i * TILE_SIZE;

			adwmma::fragment<adwmma::matrix_a, rowscols_M, rowscols_N, rowscols_K, wmma_float, adwmma::col_major> frag_A_f32;
			adwmma::load_matrix_sync(frag_A_f32, data_to_be_reduced + offset, 16);

			adwmma::fragment<adwmma::matrix_a, rowscols_M, rowscols_N, rowscols_K, wmma_bf16, adwmma::col_major> frag_A_hi;
			adwmma::fragment<adwmma::matrix_a, rowscols_M, rowscols_N, rowscols_K, wmma_bf16, adwmma::col_major> frag_A_lo;
			for(int e = 0; e < frag_A_f32.num_elements; e++){
				const float v   = frag_A_f32.x[e];
				const wmma_bf16 hi = (wmma_bf16)v;
				frag_A_hi.x[e] = hi;
				frag_A_lo.x[e] = (wmma_bf16)(v - (float)hi);
			}
			adwmma::mma_sync(frag_V, frag_A_hi, frag_P, frag_V);
			adwmma::mma_sync(frag_V, frag_A_lo, frag_P, frag_V);
		}

		// W <- V (required since we need V as a "matrix_b"); V is an fp32 partial
		// sum, so it is split hi+lo for the stage-2 multiply.
		adwmma::store_matrix_sync(Q_square, frag_V, 16, adwmma::mem_col_major);
		wavefront_lds_barrier();
		adwmma::fragment<adwmma::matrix_b, rowscols_M, rowscols_N, rowscols_K, wmma_float, adwmma::col_major> frag_W_f32;
		adwmma::load_matrix_sync(frag_W_f32, Q_square, 16);
		wavefront_lds_barrier(); // W is loaded before the fill below clobbers Q_square
		for(int e = 0; e < frag_W_f32.num_elements; e++){
			const float v   = frag_W_f32.x[e];
			const wmma_bf16 hi = (wmma_bf16)v;
			frag_W_hi.x[e] = hi;
			frag_W_lo.x[e] = (wmma_bf16)(v - (float)hi);
		}

		// 2. Perform line sum: C <- QW + C (zero)
		//    a) build the 4x4-tiled matrix that holds a 4x4 identity in each tile.
		//       Derive each entry from its column-major offset so the fill is
		//       wave-size independent: Q[row][col]=1 iff row%4==col%4.
		for(uint o = threadIdx.x; o < TILE_SIZE; o += WARP_SIZE){
			Q_square[o] = (((o % 16) & 3) == ((o / 16) & 3)) ? 1.0f : 0.0f;
		}
		wavefront_lds_barrier();
		adwmma::fragment<adwmma::matrix_a, rowscols_M, rowscols_N, rowscols_K, wmma_float, adwmma::col_major> frag_Q_f32;
		adwmma::load_matrix_sync(frag_Q_f32, Q_square, 16);
		for(int e = 0; e < frag_Q_f32.num_elements; e++){
			frag_Q.x[e] = (wmma_bf16)frag_Q_f32.x[e]; // identity is exact in bf16
		}
		//    b) perform sum: C <- Q*W_hi + Q*W_lo (Q exact)
		adwmma::mma_sync(frag_C, frag_Q, frag_W_hi, frag_C);
		adwmma::mma_sync(frag_C, frag_Q, frag_W_lo, frag_C);

		// 3. Store result in shared memory
		adwmma::store_matrix_sync(data_to_be_reduced, frag_C, 16, adwmma::mem_col_major);
	}
#endif
#else
	if (threadIdx.x <= 31) { // Only one warp performs reduction
		__shared__ __align__ (256) float Q_square[TILE_SIZE]; // storage for 16x16 matrix and 4x4 tiles of I4 matrix after

		// Declaring and filling fragments - Those are *not* shared
		mtk::wmma::tcec::fragment<wmma::matrix_b, rowscols_M, rowscols_N, rowscols_K, tf32, wmma::col_major> frag_P;
		mtk::wmma::tcec::fragment<wmma::accumulator, rowscols_M, rowscols_N, rowscols_K, tf32> frag_V;

		mtk::wmma::tcec::fragment<wmma::matrix_a, rowscols_M, rowscols_N, rowscols_K, tf32, wmma::col_major> frag_Q;
		mtk::wmma::tcec::fragment<wmma::matrix_b, rowscols_M, rowscols_N, rowscols_K, tf32, wmma::col_major> frag_W;
		mtk::wmma::tcec::fragment<wmma::accumulator, rowscols_M, rowscols_N, rowscols_K, tf32> frag_C;

		mtk::wmma::tcec::fill_fragment(frag_P, 1.0f); // P: only ones
		mtk::wmma::tcec::fill_fragment(frag_V, 0.0f); // Output: initialize to zeros
		mtk::wmma::tcec::fill_fragment(frag_C, 0.0f); // Final result

		// 1. Accumulate the values: V <- AP + V
		for(uint i = 0; i < (4 * NUM_OF_THREADS_PER_BLOCK)/TILE_SIZE; i++){
			const unsigned int offset = i * TILE_SIZE;

			mtk::wmma::tcec::fragment<wmma::matrix_a, rowscols_M, rowscols_N, rowscols_K, tf32, wmma::col_major> frag_A;
			mtk::wmma::tcec::load_matrix_sync(frag_A, data_to_be_reduced + offset, 16);
			mtk::wmma::tcec::mma_sync(frag_V, frag_A, frag_P, frag_V);
		}

		// W <- V (required since we need V as a "wmma::matrix_b")
		mtk::wmma::tcec::store_matrix_sync(Q_square, frag_V, 16, wmma::mem_col_major);
		mtk::wmma::tcec::load_matrix_sync(frag_W, Q_square, 16);

		// 2. Perform line sum: C <- QW + C (zero)
		//    a) create a 4x4 tiled matrix containing 4x4 identity matrix in each tile:
		//       - TENSOR=ON requires NUMWI to be larger than 32, so the following works and neatly gets rid of an additional function:
		const unsigned int k  = (threadIdx.x<<3);
		const unsigned int kk = 16 - (threadIdx.x>>1);
		for(uint i = 0; i < 8; i++) Q_square[k + i] = ((i + kk) & 3) ? 0.0f : 1.0f;
		mtk::wmma::tcec::load_matrix_sync(frag_Q, Q_square, 16);
		//    b) perform sum
		mtk::wmma::tcec::mma_sync(frag_C, frag_Q, frag_W, frag_C);

		// 3. Store result in shared memory
		mtk::wmma::tcec::store_matrix_sync(data_to_be_reduced, frag_C, 16, wmma::mem_col_major);
	}
#endif

	__syncthreads();
}

/* End: Reduction using tensor units */
#endif

#define REDUCEFLOATSUM(value, pAccumulator) \
	if (threadIdx.x == 0) \
	{ \
		*pAccumulator = 0; \
	} \
	__threadfence(); \
	__syncthreads(); \
	if (__any_sync(WARP_FULL_MASK, value != 0.0f)) \
	{ \
		uint32_t tgx            = threadIdx.x & cData.warpmask; \
		value                  += __shfl_sync(WARP_FULL_MASK, value, tgx ^ 1); \
		value                  += __shfl_sync(WARP_FULL_MASK, value, tgx ^ 2); \
		value                  += __shfl_sync(WARP_FULL_MASK, value, tgx ^ 4); \
		value                  += __shfl_sync(WARP_FULL_MASK, value, tgx ^ 8); \
		value                  += __shfl_sync(WARP_FULL_MASK, value, tgx ^ 16); \
		WARPSUM_TOP(value, tgx) \
		if (tgx == 0) \
		{ \
			atomicAdd(pAccumulator, value); \
		} \
	} \
	__threadfence(); \
	__syncthreads(); \
	value = (float)(*pAccumulator); \
	__syncthreads();



static __constant__ GpuData cData;
static GpuData cpuData;

void SetKernelsGpuData(GpuData* pData)
{
	cudaError_t status;
	status = cudaMemcpyToSymbol(cData, pData, sizeof(GpuData));
	RTERROR(status, "SetKernelsGpuData copy to cData failed");
	memcpy(&cpuData, pData, sizeof(GpuData));
}

void GetKernelsGpuData(GpuData* pData)
{
	cudaError_t status;
	status = cudaMemcpyFromSymbol(pData, cData, sizeof(GpuData));
	RTERROR(status, "GetKernelsGpuData copy From cData failed");
}


// Kernel files
#include "calcenergy.cu"
#include "calcMergeEneGra.cu"
#include "auxiliary_genetic.cu"
#include "kernel1.cu"
#include "kernel2.cu"
#include "kernel3.cu"
#include "kernel4.cu"
#include "kernel_ad.cu"
#include "kernel_adam.cu"
