/* cuda_stub.h - just enough CUDA for cppcheck / clang-tidy to parse the Kindi kernel sources.
 *
 * The Windows and WSL boxes used for the Turing verification have no CUDA headers (no nvcc, no toolkit), so the
 * static analysers get the declarations they need from here instead: the execution-space qualifiers, the vector
 * types, the math builtins and the warp/block builtins; the inline PTX is left to the analyser.  Only used for
 * static analysis; never compiled into anything.
 *
 * cppcheck (plain C++, the qualifiers become empty attributes):
 *
 *   cppcheck --enable=warning,style,performance,portability --language=c++ --std=c++14 \
 *            --suppress=missingIncludeSystem --inline-suppr --include=tests/static/cuda_stub.h \
 *            --template='{file}:{line}: {severity}: {message} [{id}]' kindi/tc_turing.cu
 *
 * clang-tidy (real CUDA mode, device pass only, so the target is nvptx64 and the PTX asm constraints - "=f", "h",
 * ... - are the ones that actually apply; NVIDIA headers and libdevice are not needed):
 *
 *   clang-tidy --checks='-*,clang-analyzer-*' --header-filter='^$' \
 *       --extra-arg=-x --extra-arg=cuda --extra-arg=--cuda-device-only --extra-arg=--cuda-gpu-arch=sm_75 \
 *       --extra-arg=-nocudainc --extra-arg=-nocudalib --extra-arg=-std=c++14 \
 *       --extra-arg=-include --extra-arg=tests/static/cuda_stub.h kindi/tc_turing.cu
 *
 * For a_b.cu that is only a part of the concatenated translation unit (tca_turing.cu needs tc_turing.cu and tcf.cu),
 * concatenate them in build order first; tests/static/run_static_analysis.py does that automatically.
 */
#ifndef KINDI_CUDA_STUB_H
#define KINDI_CUDA_STUB_H

/* In CUDA mode clang knows __device__ / __global__ / __shared__ as attributes, but not as the CUDA keywords this
 * build spells them with (no cuda_runtime.h), so map the keywords onto the attributes.  In plain C++ mode they are
 * empty.  __CUDA__ is defined by -x cuda. */
#if defined(__CUDA__)
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#else
#define __device__
#define __global__
#define __host__
#define __shared__
#define __constant__
#define __forceinline__ inline
#define __noinline__
#endif
#define __restrict__ __restrict
#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

struct uint2 { unsigned int x, y; };
struct uint3 { unsigned int x, y, z; };
struct uint4 { unsigned int x, y, z, w; };
struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct int4 { int x, y, z, w; };
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct dim3 { unsigned int x, y, z; };

__device__ inline uint2 make_uint2(unsigned int x, unsigned int y) { uint2 r = {x, y}; return r; }
__device__ inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { uint3 r = {x, y, z}; return r; }
__device__ inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{ uint4 r = {x, y, z, w}; return r; }
__device__ inline float2 make_float2(float x, float y) { float2 r = {x, y}; return r; }
__device__ inline float4 make_float4(float x, float y, float z, float w) { float4 r = {x, y, z, w}; return r; }

extern __device__ dim3 threadIdx, blockIdx, blockDim, gridDim;

/* math builtins (CUDA's <math_functions.h> is not available) */
__device__ inline float fabsf(float x) { return x < 0.0f ? -x : x; }
__device__ inline float fminf(float a, float b) { return a < b ? a : b; }
__device__ inline float fmaxf(float a, float b) { return a > b ? a : b; }
__device__ inline float sqrtf(float x) { (void)x; return 0.0f; }
__device__ inline float rsqrtf(float x) { (void)x; return 0.0f; }
__device__ inline float expf(float x) { (void)x; return 0.0f; }
__device__ inline float rintf(float x) { return x; }
__device__ inline float __expf(float x) { (void)x; return 0.0f; }
__device__ inline float logf(float x) { (void)x; return 0.0f; }
__device__ inline float powf(float a, float b) { (void)a; (void)b; return 0.0f; }
__device__ inline float __fdividef(float a, float b) { (void)a; (void)b; return 0.0f; }
__device__ inline float __saturatef(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
__device__ inline int min(int a, int b) { return a < b ? a : b; }
__device__ inline int max(int a, int b) { return a > b ? a : b; }
__device__ inline unsigned int min(unsigned int a, unsigned int b) { return a < b ? a : b; }
__device__ inline unsigned int max(unsigned int a, unsigned int b) { return a > b ? a : b; }
__device__ inline long long min(long long a, long long b) { return a < b ? a : b; }
__device__ __device__ inline long long max(long long a, long long b) { return a > b ? a : b; }
__device__ inline float log1pf(float x) { (void)x; return 0.0f; }
__device__ inline float tanhf(float x) { (void)x; return 0.0f; }
__device__ inline float erf(float x) { (void)x; return 0.0f; }

template <class T> __device__ inline T __ldg(const T* p) { return *p; }
template <class T> __device__ inline T* __ldg(T* p) { return p; }
template <class T> __device__ inline T __shfl_xor_sync(unsigned int mask, T v, int width) { (void)mask; (void)width; return v; }
template <class T> __device__ inline T __shfl_sync(unsigned int mask, T v, int src) { (void)mask; (void)src; return v; }
template <class T> __device__ inline T __shfl_down_sync(unsigned int mask, T v, int d) { (void)mask; (void)d; return v; }
template <class T> __device__ inline T __shfl_up_sync(unsigned int mask, T v, int d) { (void)mask; (void)d; return v; }
/* __syncthreads is a clang builtin in CUDA mode */
#if !defined(__CUDA__)
inline void __syncthreads(void) {}
#endif
__device__ inline void __syncwarp(unsigned int mask = 0xffffffffu) { (void)mask; }
__device__ inline int __float2int_rn(float f) { return (int)f; }
__device__ inline int __float2int_rd(float f) { return (int)f; }
__device__ inline int __float2int_ru(float f) { return (int)f; }
__device__ inline unsigned int __float_as_uint(float f) { union { float f; unsigned int u; } v; v.f = f; return v.u; }
__device__ inline float __uint_as_float(unsigned int u) { union { float f; unsigned int u; } v; v.u = u; return v.f; }
__device__ inline unsigned int __byte_perm(unsigned int a, unsigned int b, unsigned int s) { (void)a; (void)b; (void)s; return 0u; }
__device__ inline int __popc(unsigned int x) { int n = 0; while (x) { n += (int)(x & 1u); x >>= 1; } return n; }
__device__ inline unsigned int __brev(unsigned int x) { (void)x; return 0u; }
__device__ inline int __clz(unsigned int x) { (void)x; return 0; }
__device__ inline unsigned int __umulhi(unsigned int a, unsigned int b) { return (unsigned int)((unsigned long long)a * b >> 32); }
__device__ inline int __mulhi(int a, int b) { return (int)((long long)a * b >> 32); }
__device__ inline float fmaf(float a, float b, float c) { return a * b + c; }
__device__ inline float __fmaf_rn(float a, float b, float c) { return a * b + c; }
__device__ inline float __fadd_rn(float a, float b) { return a + b; }
__device__ inline float __fmul_rn(float a, float b) { return a * b; }
__device__ inline float __int_as_float(int i) { union { float f; int i; } v; v.i = i; return v.f; }
__device__ inline float __int2float_rn(int i) { return (float)i; }
__device__ inline int __float2int_rz(float f) { return (int)f; }
__device__ inline void __threadfence(void) {}
__device__ inline void __threadfence_block(void) {}
__device__ inline void __threadfence_system(void) {}
__device__ inline float atomicAdd(float* p, float v) { float o = *p; *p += v; return o; }
__device__ inline int atomicAdd(int* p, int v) { int o = *p; *p += v; return o; }
__device__ inline int atomicMax(int* p, int v) { int o = *p; if (v > o) *p = v; return o; }
__device__ inline int atomicMin(int* p, int v) { int o = *p; if (v < o) *p = v; return o; }

#endif
