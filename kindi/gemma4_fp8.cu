/* gemma4_fp8.cu - Gemma 4 prompt GEMM on native FP8 tensor cores (Ada sm_89, Blackwell sm_100/sm_120).

   mma.sync m16n8k32 with E4M3 (GGUF 43) or E5M2 (GGUF 44) operands and fp32 accumulation.  The
   weights are already FP8; the activations are converted per token with an explicit scale so the
   largest magnitude fills the format's range.  The output is scaled back in the epilogue.

   This file has no includes: it is compiled to compute_75 PTX by the old NVRTC and the loader
   raises the PTX version/target to 8.4/sm_89, as gemma4_tc.cu does for the int8 tensor cores.  The
   sm_89 FP8 instruction is a portable (non-'a') instruction, so Blackwell's driver JIT builds the
   same PTX for its own tensor cores. */

#define G4_FP8_MMA_E4(c, a0, a1, a2, a3, b0, b1) \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#define G4_FP8_MMA_E5(c, a0, a1, a2, a3, b0, b1) \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

__device__ __forceinline__ unsigned short g4_fp8_pair_e4(float a, float b) {
    unsigned short r;
    /* the first source operand becomes the high byte, so pass b first to put a in the low byte */
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(b), "f"(a));
    return r;
}
__device__ __forceinline__ unsigned short g4_fp8_pair_e5(float a, float b) {
    unsigned short r;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(r) : "f"(b), "f"(a));
    return r;
}

template<int KIND>
__device__ __forceinline__ void g4_fp8_quant(const float* x, unsigned char* q, float* scale,
                                              int cols, int nt) {
    int t = blockIdx.x;
    if (t >= nt) return;
    const float* xr = x + (size_t)t * cols;
    __shared__ float red[256];
    float m = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, fabsf(xr[i]));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) {
        if (threadIdx.x < d) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + d]);
        __syncthreads();
    }
    const float maxv = KIND == 43 ? 448.0f : 57344.0f;
    float mx = red[0];
    float s = mx > 0.0f ? mx / maxv : 1.0f;
    if (threadIdx.x == 0) scale[t] = s;
    unsigned char* qr = q + (size_t)t * cols;
    for (int i = 2 * threadIdx.x; i < cols; i += 2 * blockDim.x) {
        float a = xr[i] / s, b = xr[i + 1] / s;
        unsigned short p = KIND == 43 ? g4_fp8_pair_e4(a, b) : g4_fp8_pair_e5(a, b);
        *reinterpret_cast<unsigned short*>(qr + i) = p;
    }
}

/* One block computes 64 weight rows x 64 tokens; warp w owns weight rows 16*(w&3)..+15 and
   tokens 32*(w>>2)..+31, four m16n8k32 tiles.  A tile of the weights is the mma A operand
   (row-major) and the matching activation tile is the B operand (col-major, i.e. the same bytes
   in row-major activation order).  K advances in 32-byte steps through shared memory. */
template<int KIND>
__device__ __forceinline__ void g4_fp8mm(const unsigned char* w, const unsigned char* xq,
                                          const float* xscale, float* y, int rows, int cols, int nt) {
    __shared__ unsigned char WS[64][40];
    __shared__ unsigned char XS[64][40];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int gid = lane >> 2, t4 = lane & 3;
    int row0 = blockIdx.x * 64, tok0 = blockIdx.y * 64;
    int mr = 16 * (warp & 3), nc = 32 * (warp >> 2);
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[j][i] = 0.0f;
    for (int k0 = 0; k0 < cols; k0 += 32) {
        /* 64 rows x 32 bytes = 2048 bytes; each of the 256 threads owns one 8-byte chunk */
        int r = tid >> 2, c = (tid & 3) * 8;
        int rr = min(row0 + r, rows - 1);
        const unsigned int* src = reinterpret_cast<const unsigned int*>(w + (size_t)rr * cols + k0 + c);
        unsigned int* dst = reinterpret_cast<unsigned int*>(&WS[r][c]);
        dst[0] = src[0];
        dst[1] = src[1];
        int tt = min(tok0 + r, nt - 1);
        src = reinterpret_cast<const unsigned int*>(xq + (size_t)tt * cols + k0 + c);
        dst = reinterpret_cast<unsigned int*>(&XS[r][c]);
        dst[0] = src[0];
        dst[1] = src[1];
        __syncthreads();
        int ra = mr + gid, rb = ra + 8;
        unsigned int a0 = *reinterpret_cast<const unsigned int*>(&WS[ra][t4 * 4]);
        unsigned int a1 = *reinterpret_cast<const unsigned int*>(&WS[rb][t4 * 4]);
        unsigned int a2 = *reinterpret_cast<const unsigned int*>(&WS[ra][t4 * 4 + 16]);
        unsigned int a3 = *reinterpret_cast<const unsigned int*>(&WS[rb][t4 * 4 + 16]);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int tb = nc + 8 * j + gid;
            unsigned int b0 = *reinterpret_cast<const unsigned int*>(&XS[tb][t4 * 4]);
            unsigned int b1 = *reinterpret_cast<const unsigned int*>(&XS[tb][t4 * 4 + 16]);
            if (KIND == 43) G4_FP8_MMA_E4(acc[j], a0, a1, a2, a3, b0, b1);
            else G4_FP8_MMA_E5(acc[j], a0, a1, a2, a3, b0, b1);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int ta = tok0 + nc + 8 * j + 2 * t4, tb = ta + 1;
        int ra = row0 + mr + gid, rb = ra + 8;
        if (ta < nt && ra < rows) y[(size_t)ta * rows + ra] = acc[j][0] * xscale[ta];
        if (tb < nt && ra < rows) y[(size_t)tb * rows + ra] = acc[j][1] * xscale[tb];
        if (ta < nt && rb < rows) y[(size_t)ta * rows + rb] = acc[j][2] * xscale[ta];
        if (tb < nt && rb < rows) y[(size_t)tb * rows + rb] = acc[j][3] * xscale[tb];
    }
}

extern "C" __global__ void __launch_bounds__(256) k_g4_fp8mm_t43(const unsigned char* w,
        const unsigned char* xq, const float* xscale, float* y, int rows, int cols, int nt) {
    g4_fp8mm<43>(w, xq, xscale, y, rows, cols, nt);
}
extern "C" __global__ void __launch_bounds__(256) k_g4_fp8mm_t44(const unsigned char* w,
        const unsigned char* xq, const float* xscale, float* y, int rows, int cols, int nt) {
    g4_fp8mm<44>(w, xq, xscale, y, rows, cols, nt);
}
extern "C" __global__ void k_g4_fp8_quant_t43(const float* x, unsigned char* q, float* scale,
                                               int cols, int nt) {
    g4_fp8_quant<43>(x, q, scale, cols, nt);
}
extern "C" __global__ void k_g4_fp8_quant_t44(const float* x, unsigned char* q, float* scale,
                                               int cols, int nt) {
    g4_fp8_quant<44>(x, q, scale, cols, nt);
}
