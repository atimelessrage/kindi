/* kcpu.c - Kindi CPU backend (Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only): one token of a Ternary-Bonsai-2 (Qwen3.5 hybrid GDN/attention) model in C.
   The whole forward pass runs inside one OpenMP parallel region (barriers between the stages), so there is no
   Python or thread start-up cost per operation. Arithmetic plan as kindi/engine.py: fp64 norm sums, Hadamard folds
   with the pack's sign vectors, 81-entry fp32 table GEMV for every ternary matrix, fp16 K/V cache. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

/* The engine requires AVX.  Five tiers are buildable: AVX2/FMA/F16C, AVX2/F16C without FMA,
   AVX2/FMA without F16C, AVX2, and AVX-only.  kindi/cpu.py builds the tier the host supports
   and loads that library; kc_isa()/kc_isa_name() below report which tier was compiled, so a
   stale cache entry cannot run instructions the CPU lacks.  FMA is optional (separate
   multiply/add without it), and a build without F16C uses the software fp16 conversion.  Fail
   at compile time with an actionable message instead of a target-mismatch error. */
#if !defined(__AVX__)
#error "kindi CPU engine requires AVX; build with -mavx (docs/CPU-SUPPORT.md)"
#endif

#ifdef _WIN32
#define KC_API __declspec(dllexport)      /* the ctypes loader needs the names in the DLL export table */
#else
#define KC_API
#endif

typedef struct { const uint8_t* W; const float* S; int R, C; } Mat;   /* W [R/8][C/128][64] u32 pair codes, S [R/8][C/128][8] */

typedef struct {
  int is_attn;
  const float *attn_norm, *post_norm;
  /* GDN */
  Mat qkv, gate, ssm_out;
  const float *beta_w, *alpha_w, *dt_b, *ssm_a, *conv_w, *ssm_norm;
  float *M, *conv;                       /* state [NV][128][128], conv [3][CONV_CH] */
  /* attention */
  Mat q, k, v, o;
  const float *q_norm, *k_norm;
  uint16_t *Kc, *Vc;                     /* [ctx][NHKV][HD] fp16 */
  /* FFN */
  Mat ffg, ffu, ffd;
} Layer;

typedef struct {
  int D, NL, NFF, NH, NHKV, HD, NK, HK, NV, CONV_CH, HB, rope_half, ctx, vocab;
  Layer* L;
  Mat emb, head;
  const float *out_norm, *sD, *sV, *sA, *sF, *cos_t, *sin_t;
  /* scratch */
  float *h, *xn, *xp, *y, *qkv, *zv, *cs, *qk, *gout, *outn, *m, *qn, *kr, *vv, *qr, *kro, *sc, *att, *ffg, *ffu,
        *T, *logits, *gb, *gg;
  double* red;                           /* per-thread partial sums */
} Model;

/* ------------------------------------------------------------------------------------------- helpers */
/* stage timers (thread 0, between synchronised points): 0 embed 1 norm+fold 2 gemv 3 gdn-other 4 attn-other 5 ffn-other 6 head */
double kc_times[8];
static double t_last; static int t_cur;
#define TS(cat) do { if(omp_get_thread_num() == 0){ const double _t = omp_get_wtime(); kc_times[t_cur] += _t - t_last; t_last = _t; t_cur = (cat); } } while(0)
static inline void span(long long n, int* a, int* b){
  const int nt = omp_get_num_threads(), id = omp_get_thread_num();
  *a = (int)(n * id / nt); *b = (int)(n * (id + 1) / nt);
}
#if defined(__F16C__)
static inline float h2f(uint16_t h){ return _cvtsh_ss(h); }
static inline uint16_t f2h(float f){ return _cvtss_sh(f, 0); }
#else
/* IEEE binary16 conversions for AVX-only hosts.  f2h is round-to-nearest-even, matching
   _cvtss_sh(f, 0); h2f is exact for every finite, infinite and NaN input. */
static inline float h2f(uint16_t h){
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu, bits;
  if(exp == 0){
    if(mant == 0) bits = sign;
    else {
      int e = -1;
      do { mant <<= 1; e++; } while(!(mant & 0x400u));
      mant &= 0x3ffu;
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if(exp == 31) bits = sign | 0x7f800000u | (mant << 13);
  else bits = sign | ((exp + 112u) << 23) | (mant << 13);
  union { uint32_t u; float f; } v; v.u = bits; return v.f;
}
static inline uint16_t f2h(float f){
  union { float f; uint32_t u; } v; v.f = f;
  const uint32_t sign = (v.u >> 16) & 0x8000u;
  const uint32_t abs = v.u & 0x7fffffffu;
  if(abs >= 0x7f800000u){
    if(abs > 0x7f800000u) return (uint16_t)(sign | 0x7e00u | ((abs >> 13) & 0x3ffu));
    return (uint16_t)(sign | 0x7c00u);
  }
  if(abs >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);   /* 65520 rounds to infinity */
  int exp = (int)(abs >> 23) - 127;
  const uint32_t mant = abs & 0x7fffffu;
  if(exp >= -14){
    uint32_t q = (mant | 0x800000u) >> 13;
    const uint32_t rem = (mant | 0x800000u) & 0x1fffu;
    if(rem > 0x1000u || (rem == 0x1000u && (q & 1u))) q++;
    if(q == 0x800u){ q = 0x400u; exp++; }
    const uint32_t e = (uint32_t)(exp + 15);
    if(e >= 31u) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (e << 10) | (q & 0x3ffu));
  }
  if(exp >= -25){
    const uint32_t sig = mant | 0x800000u;
    const int shift = -exp - 1;
    uint32_t q = sig >> shift;
    const uint32_t rem = sig & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if(rem > half || (rem == half && (q & 1u))) q++;
    return (uint16_t)(sign | q);
  }
  return (uint16_t)sign;
}
#endif
static inline float silu(float x){ return x / (1.0f + expf(-x)); }
static inline float sigm(float x){ return 1.0f / (1.0f + expf(-x)); }

static void fwht_block(float* x, int hb){
  for(int h = 1; h < hb; h <<= 1)
    for(int i = 0; i < hb; i += 2 * h)
      for(int j = i; j < i + h; ++j){ const float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
  const float s = 1.0f / sqrtf((float)hb);
  for(int j = 0; j < hb; ++j) x[j] *= s;
}

/* out = fold(v) = FWHT_blocks(sg * v); in parallel over blocks (must be called by all threads; ends in a barrier) */
static void fold(const Model* m, const float* v, const float* sg, float* out, int n){
  #pragma omp for schedule(static)
  for(int b = 0; b < n / m->HB; ++b){
    float* o = out + (size_t)b * m->HB;
    for(int j = 0; j < m->HB; ++j) o[j] = sg[b * m->HB + j] * v[b * m->HB + j];
    fwht_block(o, m->HB);
  }
}

/* ------------------------------------------------------------------------------------------- ternary GEMV */
/* Weights as pair codes, rows in SIMD lanes: per (8-row block, 128-column group) 64 uint32; word p holds the 8 rows'
   4-bit codes for columns 2p, 2p+1 (0..7 = which of the 8 nonzero (w0, w1) pairs, 8 = both zero; 2 bits per weight).
   Per call, a table of the 8 nonzero pair sums for every column pair (m->T, C/2 x 8 floats); then one vpermps per
   8 rows x 2 columns. Per-group sums are scaled (the 8 rows' scales are contiguous) into a vector accumulator, so a
   row's result needs no horizontal sum. Every pair sum is formed once in fp32; no activation quantisation. */
KC_API void kc_init(void){}
static void pair_tables(const float* x, float* T, int p0, int p1){
  for(int p = p0; p < p1; ++p){
    const float x0 = x[2 * p], x1 = x[2 * p + 1];
    float* t = T + (size_t)p * 8;
    t[0] = -x0 - x1; t[1] = -x1; t[2] = x0 - x1; t[3] = -x0;
    t[4] = x0; t[5] = -x0 + x1; t[6] = x1; t[7] = x0 + x1;
  }
}
#if defined(__AVX2__)
static void rows8(const Mat* A, float* y, const float* T, int b0, int b1){
  const int G = A->C / 128;
  const __m256i sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  const __m256i f = _mm256_set1_epi32(15);
  const uint32_t* P = (const uint32_t*)A->W;
  for(int rb = b0; rb < b1; ++rb){
    __m256 acc = _mm256_setzero_ps();
    for(int g = 0; g < G; ++g){
      const uint32_t* pw = P + ((size_t)rb * G + g) * 64;
      const float* pt = T + (size_t)g * 64 * 8;
      __m256 s0 = _mm256_setzero_ps(), s1 = s0;
      for(int p = 0; p < 64; p += 2){
        const __m256i c0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)pw[p]), sh), f);
        const __m256i c1 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)pw[p + 1]), sh), f);
        const __m256 v0 = _mm256_permutevar8x32_ps(_mm256_load_ps(pt + 8 * p), c0);
        const __m256 v1 = _mm256_permutevar8x32_ps(_mm256_load_ps(pt + 8 * p + 8), c1);
        const __m256 z0 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_slli_epi32(c0, 28), 31));   /* code 8 */
        const __m256 z1 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_slli_epi32(c1, 28), 31));
        s0 = _mm256_add_ps(s0, _mm256_andnot_ps(z0, v0));
        s1 = _mm256_add_ps(s1, _mm256_andnot_ps(z1, v1));
      }
#if defined(__FMA__)
      acc = _mm256_fmadd_ps(_mm256_add_ps(s0, s1), _mm256_loadu_ps(A->S + ((size_t)rb * G + g) * 8), acc);
#else
      /* AVX2/F16C hosts without FMA: keep the AVX2 table/permute path and round the product
         before the add.  This is not bit-identical to FMA, but it is the documented fallback. */
      acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_add_ps(s0, s1),
                                             _mm256_loadu_ps(A->S + ((size_t)rb * G + g) * 8)));
#endif
    }
    _mm256_storeu_ps(y + rb * 8, acc);
  }
}
#else
static void rows8(const Mat* A, float* y, const float* T, int b0, int b1){
  const int G = A->C / 128;
  const __m256 vfour = _mm256_set1_ps(4.0f), veight = _mm256_set1_ps(8.0f);
  for(int rb = b0; rb < b1; ++rb){
    __m256 acc = _mm256_setzero_ps();
    for(int g = 0; g < G; ++g){
      const uint32_t* pw = (const uint32_t*)A->W + ((size_t)rb * G + g) * 64;
      const float* pt = T + (size_t)g * 64 * 8;
      __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
      for(int p = 0; p < 64; p += 2){
        const uint32_t w0 = pw[p], w1 = pw[p + 1];
        float c0[8], c1[8];
        int i0[8], i1[8];
        for(int l = 0; l < 8; ++l){
          const int a = (int)((w0 >> (4 * l)) & 15u);
          const int b = (int)((w1 >> (4 * l)) & 15u);
          c0[l] = (float)a; i0[l] = a & 3;
          c1[l] = (float)b; i1[l] = b & 3;
        }
        const __m256 f0 = _mm256_loadu_ps(c0), f1 = _mm256_loadu_ps(c1);
        const __m256i x0 = _mm256_loadu_si256((const __m256i*)i0);
        const __m256i x1 = _mm256_loadu_si256((const __m256i*)i1);
        const __m256 t0l = _mm256_broadcast_ps((const __m128*)(pt + 8 * p));
        const __m256 t0h = _mm256_broadcast_ps((const __m128*)(pt + 8 * p + 4));
        const __m256 t1l = _mm256_broadcast_ps((const __m128*)(pt + 8 * p + 8));
        const __m256 t1h = _mm256_broadcast_ps((const __m128*)(pt + 8 * p + 12));
        __m256 v0 = _mm256_blendv_ps(_mm256_permutevar_ps(t0l, x0),
                                     _mm256_permutevar_ps(t0h, x0),
                                     _mm256_cmp_ps(f0, vfour, _CMP_GE_OQ));
        __m256 v1 = _mm256_blendv_ps(_mm256_permutevar_ps(t1l, x1),
                                     _mm256_permutevar_ps(t1h, x1),
                                     _mm256_cmp_ps(f1, vfour, _CMP_GE_OQ));
        v0 = _mm256_andnot_ps(_mm256_cmp_ps(f0, veight, _CMP_EQ_OQ), v0);
        v1 = _mm256_andnot_ps(_mm256_cmp_ps(f1, veight, _CMP_EQ_OQ), v1);
        s0 = _mm256_add_ps(s0, v0);
        s1 = _mm256_add_ps(s1, v1);
      }
      acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_add_ps(s0, s1),
                                             _mm256_loadu_ps(A->S + ((size_t)rb * G + g) * 8)));
    }
    _mm256_storeu_ps(y + rb * 8, acc);
  }
}
#endif

/* inside the parallel region: up to 3 matrices with the same input; the threads build the pair tables, then split
   the concatenated rows in 8-row blocks; ends in a barrier */
static void gemv(Model* m, int n, const Mat* const* A, float* const* y, const float* x){
  const int prev = t_cur; TS(2);
  int a, b;
  span(A[0]->C / 2, &a, &b);
  pair_tables(x, m->T, a, b);
  #pragma omp barrier
  long long Ut = 0;
  for(int i = 0; i < n; ++i) Ut += A[i]->R / 8;
  int a0, a1;
  span(Ut, &a0, &a1);
  int base = 0;
  for(int i = 0; i < n; ++i){
    const int U = A[i]->R / 8;
    const int u0 = a0 > base ? a0 - base : 0, u1 = (a1 - base) < U ? (a1 - base) : U;
    base += U;
    if(u1 > u0) rows8(A[i], y[i], m->T, u0, u1);
  }
  #pragma omp barrier
  TS(prev);
}
static void gemv1(Model* m, const Mat* A, float* y, const float* x){ const Mat* a[1] = {A}; float* yy[1] = {y}; gemv(m, 1, a, yy, x); }

/* ------------------------------------------------------------------------------------------- norms */
/* residual add (if add) + RMS norm + fold of m->h into xn / xp; all threads; ends in a barrier */
static void rmsfold(Model* m, const float* w, int add){
  const int D = m->D;
  const int prev = t_cur; TS(1);
  #pragma omp single
  {
    double acc = 0.0;
    for(int e = 0; e < D; ++e){
      const float xf = add ? m->h[e] + m->y[e] : m->h[e];
      m->h[e] = xf;
      acc += (double)xf * (double)xf;
    }
    const float s = 1.0f / sqrtf((float)(acc / D) + 1e-6f);
    for(int e = 0; e < D; ++e) m->xn[e] = (m->h[e] * s) * w[e];
  }
  fold(m, m->xn, m->sD, m->xp, D);
  TS(prev);
}

static void rms_rows(const float* x, const float* w, float* out, int rows, int n){
  for(int r = 0; r < rows; ++r){
    const float* xr = x + (size_t)r * n;
    double acc = 0.0;
    for(int e = 0; e < n; ++e) acc += (double)xr[e] * (double)xr[e];
    const float sc = 1.0f / sqrtf((float)(acc / n) + 1e-6f);
    for(int e = 0; e < n; ++e) out[(size_t)r * n + e] = (xr[e] * sc) * w[e];
  }
}

/* ------------------------------------------------------------------------------------------- blocks */
static void gdn(Model* m, Layer* L, int add){
  TS(3);
  const int D = m->D, NV = m->NV, NK = m->NK, HK = m->HK, CC = m->CONV_CH;
  rmsfold(m, L->attn_norm, add);
  { const Mat* a[2] = {&L->qkv, &L->gate}; float* yy[2] = {m->qkv, m->zv}; gemv(m, 2, a, yy, m->xp); }
  /* gates (per v head) and the causal conv (per channel) */
  #pragma omp for schedule(static) nowait
  for(int h = 0; h < NV; ++h){
    float ab = 0.f, aa = 0.f;
    for(int i = 0; i < D; ++i){ ab += L->beta_w[(size_t)h * D + i] * m->xn[i]; aa += L->alpha_w[(size_t)h * D + i] * m->xn[i]; }
    m->gb[h] = sigm(ab);
    const float a = aa + L->dt_b[h];
    const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
    m->gg[h] = sp * L->ssm_a[h];
  }
  #pragma omp for schedule(static)
  for(int c = 0; c < CC; ++c){
    float* cv = L->conv;
    const float c0 = cv[c], c1 = cv[CC + c], c2 = cv[2 * CC + c], nw = m->qkv[c];
    const float o = c0 * L->conv_w[4 * c] + c1 * L->conv_w[4 * c + 1] + c2 * L->conv_w[4 * c + 2] + nw * L->conv_w[4 * c + 3];
    cv[c] = c1; cv[CC + c] = c2; cv[2 * CC + c] = nw;
    m->cs[c] = silu(o);
  }
  /* L2 norm of the q and k heads */
  #pragma omp for schedule(static)
  for(int r = 0; r < 2 * NK; ++r){
    const float* xr = m->cs + (size_t)r * HK;
    double acc = 0.0;
    for(int e = 0; e < HK; ++e) acc += (double)xr[e] * (double)xr[e];
    const float sc = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
    for(int e = 0; e < HK; ++e) m->qk[(size_t)r * HK + e] = xr[e] * sc;
  }
  /* delta rule per v head, then the per-head RMS norm */
  const float scale = 1.0f / sqrtf((float)HK);
  #pragma omp for schedule(static)
  for(int h = 0; h < NV; ++h){
    const int kh = h % NK;
    const float* ks = m->qk + (size_t)(NK + kh) * HK;
    const float* qs = m->qk + (size_t)kh * HK;
    const float* v = m->cs + 2 * NK * HK + (size_t)h * HK;
    const float e = expf(m->gg[h]);
    float* Mh = L->M + (size_t)h * HK * HK;
    float sk[128], o[128], dd[128];
    for(int j = 0; j < HK; ++j) sk[j] = 0.f;
    for(int i = 0; i < HK; ++i){ const float ki = ks[i]; const float* mr = Mh + i * HK; for(int j = 0; j < HK; ++j) sk[j] += (mr[j] * e) * ki; }
    for(int j = 0; j < HK; ++j){ dd[j] = (v[j] - sk[j]) * m->gb[h]; o[j] = 0.f; }
    for(int i = 0; i < HK; ++i){
      const float ki = ks[i], qi = qs[i];
      float* mr = Mh + i * HK;
      for(int j = 0; j < HK; ++j){ const float mm = mr[j] * e + ki * dd[j]; mr[j] = mm; o[j] += mm * qi; }
    }
    for(int j = 0; j < HK; ++j) m->gout[(size_t)h * HK + j] = o[j] * scale;
    rms_rows(m->gout + (size_t)h * HK, L->ssm_norm, m->outn + (size_t)h * HK, 1, HK);
  }
  /* gated output in the grouped v-head order, then the fold */
  const int rep = NV / NK;
  #pragma omp for schedule(static)
  for(int e = 0; e < NV * HK; ++e){
    int src = e;
    if(rep > 1){ const int hd = e % HK, grp = e / HK; src = hd + HK * (grp / rep + NK * (grp % rep)); }
    m->m[e] = silu(m->zv[src]) * m->outn[src];
  }
  fold(m, m->m, m->sV, m->xp, NV * HK);
  gemv1(m, &L->ssm_out, m->y, m->xp);
}

static void attn(Model* m, Layer* L, int pos, int add){
  TS(4);
  const int NH = m->NH, NKV = m->NHKV, HD = m->HD, rh = m->rope_half, G = NH / NKV, n = pos + 1;
  rmsfold(m, L->attn_norm, add);
  { const Mat* a[3] = {&L->q, &L->k, &L->v}; float* yy[3] = {m->qn, m->kr, m->vv}; gemv(m, 3, a, yy, m->xp); }
  const float* cs = m->cos_t + (size_t)pos * rh;
  const float* sn = m->sin_t + (size_t)pos * rh;
  #pragma omp for schedule(static)
  for(int r = 0; r < NH + NKV; ++r){
    float tmp[512];
    float* dst;
    if(r < NH){ rms_rows(m->qn + (size_t)r * 2 * HD, L->q_norm, tmp, 1, HD); dst = m->qr + (size_t)r * HD; }
    else { rms_rows(m->kr + (size_t)(r - NH) * HD, L->k_norm, tmp, 1, HD); dst = m->kro + (size_t)(r - NH) * HD; }
    for(int j = 0; j < rh; ++j){
      const float a = tmp[j], b = tmp[j + rh];
      dst[j] = a * cs[j] - b * sn[j];
      dst[j + rh] = a * sn[j] + b * cs[j];
    }
    for(int j = 2 * rh; j < HD; ++j) dst[j] = tmp[j];
    if(r >= NH){
      const int kv = r - NH;
      uint16_t* K = L->Kc + ((size_t)pos * NKV + kv) * HD;
      uint16_t* V = L->Vc + ((size_t)pos * NKV + kv) * HD;
      for(int j = 0; j < HD; ++j){ K[j] = f2h(dst[j]); V[j] = f2h(m->vv[(size_t)kv * HD + j]); }
    }
  }
  /* scores sc[head][t] */
  #pragma omp for schedule(static)
  for(int t = 0; t < n; ++t){
    for(int kv = 0; kv < NKV; ++kv){
      const uint16_t* K = L->Kc + ((size_t)t * NKV + kv) * HD;
      float kf[512];
      for(int j = 0; j < HD; ++j) kf[j] = h2f(K[j]);
      for(int g = 0; g < G; ++g){
        const float* q = m->qr + (size_t)(kv * G + g) * HD;
        float a = 0.f;
        for(int j = 0; j < HD; ++j) a += kf[j] * q[j];
        m->sc[(size_t)(kv * G + g) * m->ctx + t] = a * 0.0625f;
      }
    }
  }
  /* softmax per head */
  #pragma omp for schedule(static)
  for(int hh = 0; hh < NH; ++hh){
    float* s = m->sc + (size_t)hh * m->ctx;
    float mx = -1e30f;
    for(int t = 0; t < n; ++t) mx = fmaxf(mx, s[t]);
    float tot = 0.f;
    for(int t = 0; t < n; ++t) tot += expf(s[t] - mx);
    for(int t = 0; t < n; ++t) s[t] = expf(s[t] - mx) / tot;
  }
  /* weighted V, gated; parallel over (head, 32-dim slice) */
  #pragma omp for schedule(static)
  for(int u = 0; u < NH * (HD / 32); ++u){
    const int hh = u / (HD / 32), d0 = (u % (HD / 32)) * 32, kv = hh / G;
    const float* p = m->sc + (size_t)hh * m->ctx;
    float acc[32];
    for(int j = 0; j < 32; ++j) acc[j] = 0.f;
    for(int t = 0; t < n; ++t){
      const uint16_t* V = L->Vc + ((size_t)t * NKV + kv) * HD + d0;
      const float pt = p[t];
      for(int j = 0; j < 32; ++j) acc[j] += pt * h2f(V[j]);
    }
    for(int j = 0; j < 32; ++j) m->att[(size_t)hh * HD + d0 + j] = acc[j] * sigm(m->qn[(size_t)hh * 2 * HD + HD + d0 + j]);
  }
  fold(m, m->att, m->sA, m->xp, NH * HD);
  gemv1(m, &L->o, m->y, m->xp);
}

static void ffn_input(Model* m, Layer* L){
  TS(5);
  rmsfold(m, L->post_norm, 1);
  { const Mat* a[2] = {&L->ffg, &L->ffu}; float* yy[2] = {m->ffg, m->ffu}; gemv(m, 2, a, yy, m->xp); }
  #pragma omp for schedule(static)
  for(int e = 0; e < m->NFF; ++e) m->m[e] = silu(m->ffg[e]) * m->ffu[e];
  fold(m, m->m, m->sF, m->xp, m->NFF);
}

static void ffn_down(Model* m, Layer* L){
  gemv1(m, &L->ffd, m->y, m->xp);
}

static void ffn(Model* m, Layer* L){
  ffn_input(m, L);
  ffn_down(m, L);
}

static void layer_body(Model* m, int il, int pos, int add){
  Layer* L = m->L + il;
  if(L->is_attn) attn(m, L, pos, add); else gdn(m, L, add);
  ffn(m, L);
}

static void head_body(Model* m, int add, int* best){
  TS(6);
  rmsfold(m, m->out_norm, add);
  gemv1(m, &m->head, m->logits, m->xp);
  #pragma omp single
  {
    int bi = 0; float bv = m->logits[0];
    for(int i = 1; i < m->head.R; ++i) if(m->logits[i] > bv){ bv = m->logits[i]; bi = i; }
    *best = bi;
  }
}

static void embed(Model* m, int tok){
  const int C = m->emb.C, R = m->emb.R;
  #pragma omp for schedule(static)
  for(int g = 0; g < C / 128; ++g){
    static const signed char W0[9] = {-1, 0, 1, -1, 1, -1, 0, 1, 0}, W1[9] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};
    const uint32_t* pw = (const uint32_t*)m->emb.W + ((size_t)(tok / 8) * (C / 128) + g) * 64;
    const float d = m->emb.S[((size_t)(tok / 8) * (C / 128) + g) * 8 + (tok & 7)];
    for(int p = 0; p < 64; ++p){
      const int code = (pw[p] >> (4 * (tok & 7))) & 15;
      m->m[g * 128 + 2 * p] = (float)W0[code] * d;
      m->m[g * 128 + 2 * p + 1] = (float)W1[code] * d;
    }
  }
  #pragma omp for schedule(static)
  for(int bl = 0; bl < m->D / m->HB; ++bl){
    float* o = m->m + (size_t)bl * m->HB;
    fwht_block(o, m->HB);
    for(int j = 0; j < m->HB; ++j) m->h[bl * m->HB + j] = m->sD[bl * m->HB + j] * o[j];
  }
}

/* one token; head != 0: also the output head (logits in m->logits); returns the greedy token or -1 */
KC_API int kc_step(Model* m, int tok, int pos, int head){
  int best = -1;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  {
    embed(m, tok);
    for(int il = 0; il < m->NL; ++il) layer_body(m, il, pos, il > 0);
    if(head) head_body(m, 1, &best);
    else {
      #pragma omp single
      { for(int e = 0; e < m->D; ++e) m->h[e] += m->y[e]; }   /* keep h = the residual after the last block */
    }
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return best;
}

/* Per-layer entry point for a CPU/GPU layer split (tandem).  kc_embed() puts the token vector in
   m->h; each kc_layer() call treats m->h as the complete residual entering il and leaves the
   complete residual leaving il in m->h.  The first block's residual is added to h by ffn(), and
   the FFN output is added explicitly at the end, so m->y is never carried between calls.  The
   whole-model kc_step() above stays a single OpenMP region; these calls open one region per part
   and are for handing a layer to or from another backend. */
KC_API void kc_embed(Model* m, int tok){
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  { embed(m, tok); }
  kc_times[t_cur] += omp_get_wtime() - t_last;
}

KC_API int kc_layer(Model* m, int il, int pos){
  if(il < 0 || il >= m->NL || pos < 0 || pos >= m->ctx) return -1;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  {
    #pragma omp single
    { for(int e = 0; e < m->D; ++e) m->y[e] = 0.0f; }
    #pragma omp barrier
    layer_body(m, il, pos, 0);
    #pragma omp single
    { for(int e = 0; e < m->D; ++e) m->h[e] += m->y[e]; }
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return 0;
}

/* The same layer without its FFN-down GEMV.  m->xp is left holding the folded FFN input;
   m->y is zero so a caller can fill the down rows from one or more backends. */
KC_API int kc_layer_ffn_input(Model* m, int il, int pos){
  if(il < 0 || il >= m->NL || pos < 0 || pos >= m->ctx) return -1;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  {
    #pragma omp single
    { for(int e = 0; e < m->D; ++e) m->y[e] = 0.0f; }
    #pragma omp barrier
    Layer* L = m->L + il;
    if(L->is_attn) attn(m, L, pos, 0); else gdn(m, L, 0);
    ffn_input(m, L);
    #pragma omp single
    { for(int e = 0; e < m->D; ++e) m->y[e] = 0.0f; }
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return 0;
}

/* One ternary matrix over an explicit row range, for a CPU/GPU row split.  The caller supplies
   the packed rows and the complete input vector; this entry point does not touch the layer
   schedule or the residual stream. */
KC_API int kc_gemv(Model* m, const Mat* A, const float* x, float* y){
  if(!m || !A || !x || !y) return -1;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  { gemv1(m, A, y, x); }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return 0;
}

/* Q4_0 GEMV for a matrix that is not part of the Model (the Bonsai 2.5 -mtp draft head, stored Q4_0):
   blocks of 32 values, 18 bytes each (one fp16 scale then 16 packed nibbles; element j < 16 is the low
   nibble of qs[j] and element j + 16 its high nibble).  y[r] = sum_c W[r][c] x[c], C a multiple of 32.
   Rows are independent, so the loop is the parallel region; the caller owns the buffers. */
KC_API int kc_q4gemv(const uint8_t* W, const float* x, float* y, int R, int C){
  if(!W || !x || !y || R <= 0 || C <= 0 || C % 32) return -1;
  const int nb = C / 32;
  #pragma omp parallel for schedule(static)
  for(int r = 0; r < R; ++r){
    const uint8_t* p = W + (size_t)r * nb * 18;
    float acc = 0.f;
    for(int b = 0; b < nb; ++b, p += 18){
      const float d = h2f(*(const uint16_t*)p);
      const uint8_t* qs = p + 2;
      float s = 0.f;
      for(int j = 0; j < 16; ++j){
        s += (float)((int)(qs[j] & 15u) - 8) * x[b * 32 + j];
        s += (float)((int)(qs[j] >> 4) - 8) * x[b * 32 + 16 + j];
      }
      acc += d * s;
    }
    y[r] = acc;
  }
  return 0;
}

KC_API int kc_head(Model* m){
  int best = -1;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  { head_body(m, 0, &best); }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return best;
}

/* The compiled tier. The numbers and names match kindi/cpu.py:ISA_ORDER; the loader compares
   kc_isa_name() with the tier it selected before the first model call. */
KC_API int kc_isa(void){
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
  return 4;
#elif defined(__AVX2__) && defined(__F16C__)
  return 3;
#elif defined(__AVX2__) && defined(__FMA__)
  return 2;
#elif defined(__AVX2__)
  return 1;
#else
  return 0;
#endif
}

KC_API const char* kc_isa_name(void){
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
  return "avx2-fma-f16c";
#elif defined(__AVX2__) && defined(__F16C__)
  return "avx2-f16c-no-fma";
#elif defined(__AVX2__) && defined(__FMA__)
  return "avx2-fma";
#elif defined(__AVX2__)
  return "avx2";
#else
  return "avx-only";
#endif
}

KC_API int kc_sizeof_model(void){ return (int)sizeof(Model); }
KC_API int kc_sizeof_layer(void){ return (int)sizeof(Layer); }

/* ==============================================================================================
   Batched T-row verify (roadmap CP1).

   Plain decode streams the whole pack for one row - 6.75 GB of weights per token on the 27B.  A
   verify pass asks the same model for T rows at consecutive positions, so the way to pay for
   drafting is to read each packed weight word once and multiply it by T pair tables.  rows8_multi()
   does that with the same per-row sequence of permutes, adds and FMAs as rows8(), which is what
   keeps a batch's rows equal, bit for bit, to the rows plain decode computes one at a time.

   Everything around the GEMV (norms, folds, attention, the GDN scan) is per row and is the same
   arithmetic in the same order as attn()/gdn()/ffn()/head_body(); the work is spread over rows
   instead of over positions/heads.  Two pieces of model state are shared:

     * K/V cache: row t writes its own position and reads only positions <= its own, and a later
       batch rewrites every position it passes, so a rejected row's K/V is simply overwritten.
     * GDN state (conv taps and the SSM matrix): sequential in the rows, so it cannot be undone by
       overwriting.  kc_step_batch() copies the state before the batch, and kc_batch_rollback(keep)
       restores that copy and replays rows 0..keep-1 of the scan from the per-(row, layer) QKV and
       XN vectors it stored.  A replay reads no weights: undoing a rejected draft costs arithmetic,
       not traffic.
   ============================================================================================== */
#define KC_MAXB 8

typedef struct {
  Model* owner;                    /* the Model this arena was sized for; reuse only for it */
  int nt, NL, D, NFF, CC, NG, NH, NHKV, HD, NK, HK, NV, HB, ctx, vocab;
  int MW, QW, KW, AW;              /* MM and XP / QN / KR|KRO|VV / ATT row widths */
  size_t tabp;                     /* pair tables per row (the widest matrix input) */
  size_t snap_len, *soff;          /* SNAP: the GDN state of every layer, one block each */
  float *H, *Y, *XN, *XP, *QKV, *ZV, *CS, *QK, *GOUT, *OUTN, *MM, *QN, *KR, *VV, *QR, *KRO, *ATT,
        *SC, *FFG, *FFU, *TAB, *LOG, *GB, *GG, *SCL, *R1, *R2, *R3, *STQ, *STX;
  uint8_t* SNAP;
  int keep_n;
  void* keep_p[40];
} Batch;
static Batch B;

#ifdef _WIN32
static void* b_alloc(size_t bytes){ return _aligned_malloc(bytes, 32); }
static void b_release(void* p){ if(p) _aligned_free(p); }
#else
static void* b_alloc(size_t bytes){ void* p = 0; return posix_memalign(&p, 32, bytes) ? 0 : p; }
static void b_release(void* p){ free(p); }
#endif

static float* b_rows(size_t n){
  float* p = (float*)b_alloc((n ? n : 1) * sizeof(float));
  if(p){
    if(B.keep_n >= 40){ b_release(p); return 0; }   /* an untracked block would leak and go unnoticed */
    B.keep_p[B.keep_n++] = p;
  }
  return p;
}

static void batch_drop(void){
  for(int i = 0; i < B.keep_n; ++i) b_release(B.keep_p[i]);
  b_release(B.SNAP);
  free(B.soff);
  memset(&B, 0, sizeof B);
}

/* Allocate for nt rows against this model (releases a smaller set).  Returns 0 on ENOMEM. */
static int batch_prepare(Model* m, int nt){
  /* One arena per process, so reuse only for the model it was sized for. Keying on D+ctx alone
     let a second Model with the same width and context reuse a SNAP/soff table sized for the
     first one's layer count: out-of-bounds copies with no error. */
  if(B.owner == m && B.nt >= nt && B.NL == m->NL && B.D == m->D && B.ctx == m->ctx) return 1;
  batch_drop();
  const int D = m->D, NFF = m->NFF, NH = m->NH, NHKV = m->NHKV, HD = m->HD, NK = m->NK, HK = m->HK,
            NV = m->NV, CC = m->CONV_CH, ctx = m->ctx;
  int NG = 0;
  for(int il = 0; il < m->NL; ++il) if(!m->L[il].is_attn && m->L[il].gate.R > NG) NG = m->L[il].gate.R;
  B.owner = m; B.nt = nt; B.NL = m->NL; B.D = D; B.NFF = NFF; B.CC = CC; B.NG = NG; B.NH = NH; B.NHKV = NHKV; B.HD = HD;
  B.NK = NK; B.HK = HK; B.NV = NV; B.HB = m->HB; B.ctx = ctx; B.vocab = m->head.R;
  B.MW = D;
  if(NFF > B.MW) B.MW = NFF;
  if(NV * HK > B.MW) B.MW = NV * HK;
  if(NH * HD > B.MW) B.MW = NH * HD;
  B.QW = 2 * NH * HD; B.KW = NHKV * HD; B.AW = NH * HD;
  B.tabp = (size_t)((D > NFF ? D : NFF) / 2);
  const size_t n = (size_t)nt, MW = (size_t)B.MW;
  B.H = b_rows(n * D); B.Y = b_rows(n * D); B.XN = b_rows(n * D); B.XP = b_rows(n * MW);
  B.QKV = b_rows(n * CC); B.ZV = b_rows(n * NG); B.CS = b_rows(n * CC); B.QK = b_rows(n * 2 * NK * HK);
  B.GOUT = b_rows(n * NV * HK); B.OUTN = b_rows(n * NV * HK); B.MM = b_rows(n * MW);
  B.QN = b_rows(n * B.QW); B.KR = b_rows(n * B.KW); B.VV = b_rows(n * B.KW);
  B.QR = b_rows(n * B.AW); B.KRO = b_rows(n * B.KW); B.ATT = b_rows(n * B.AW);
  B.SC = b_rows(n * NH * ctx); B.FFG = b_rows(n * NFF); B.FFU = b_rows(n * NFF);
  B.TAB = b_rows(n * B.tabp * 8); B.LOG = b_rows(n * (size_t)B.vocab);
  B.GB = b_rows(n * NV); B.GG = b_rows(n * NV); B.SCL = b_rows(n);
  /* per-thread replay scratch: the layer replay runs one layer per thread */
  B.R1 = b_rows(64 * (CC > 2 * NK * HK ? (size_t)CC : (size_t)2 * NK * HK));
  B.R2 = b_rows(64 * (size_t)2 * NK * HK); B.R3 = b_rows(64 * (size_t)2 * NV);
  B.STQ = b_rows(n * m->NL * CC); B.STX = b_rows(n * m->NL * D);
  B.soff = (size_t*)calloc((size_t)m->NL + 1, sizeof(size_t));
  size_t len = 0;
  for(int il = 0; il < m->NL; ++il){
    B.soff[il] = len;
    if(!m->L[il].is_attn) len += ((size_t)NV * HK * HK + (size_t)3 * CC) * sizeof(float);
  }
  B.soff[m->NL] = len;
  B.snap_len = len;
  if(len) B.SNAP = (uint8_t*)b_alloc(len);
  if(!B.soff || (len && !B.SNAP)) return 0;
  for(int i = 0; i < B.keep_n; ++i) if(!B.keep_p[i]) return 0;
  return 1;
}

/* y[t][r] = sum_c A[r][c] x[t][c] for every t; the packed weights are read once. */
static void rows8_multi(const Mat* A, float* Y, int ystride, int nt, int b0, int b1){
  const int G = A->C / 128;
  const float* T = B.TAB;
  const size_t ts = B.tabp * 8;
#if defined(__AVX2__)
  const __m256i sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  const __m256i f = _mm256_set1_epi32(15);
  const uint32_t* P = (const uint32_t*)A->W;
  for(int rb = b0; rb < b1; ++rb){
    __m256 acc[KC_MAXB];
    for(int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for(int g = 0; g < G; ++g){
      const uint32_t* pw = P + ((size_t)rb * G + g) * 64;
      __m256 s0[KC_MAXB], s1[KC_MAXB];
      for(int t = 0; t < nt; ++t){ s0[t] = _mm256_setzero_ps(); s1[t] = _mm256_setzero_ps(); }
      for(int p = 0; p < 64; p += 2){
        const __m256i c0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)pw[p]), sh), f);
        const __m256i c1 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)pw[p + 1]), sh), f);
        const __m256 z0 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_slli_epi32(c0, 28), 31));
        const __m256 z1 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_slli_epi32(c1, 28), 31));
        for(int t = 0; t < nt; ++t){
          const float* pt = T + (size_t)t * ts + (size_t)g * 512;
          const __m256 v0 = _mm256_permutevar8x32_ps(_mm256_load_ps(pt + 8 * p), c0);
          const __m256 v1 = _mm256_permutevar8x32_ps(_mm256_load_ps(pt + 8 * p + 8), c1);
          s0[t] = _mm256_add_ps(s0[t], _mm256_andnot_ps(z0, v0));
          s1[t] = _mm256_add_ps(s1[t], _mm256_andnot_ps(z1, v1));
        }
      }
      const __m256 sc = _mm256_loadu_ps(A->S + ((size_t)rb * G + g) * 8);
      for(int t = 0; t < nt; ++t){
#if defined(__FMA__)
        acc[t] = _mm256_fmadd_ps(_mm256_add_ps(s0[t], s1[t]), sc, acc[t]);
#else
        acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(_mm256_add_ps(s0[t], s1[t]), sc));
#endif
      }
    }
    for(int t = 0; t < nt; ++t) _mm256_storeu_ps(Y + (size_t)t * ystride + rb * 8, acc[t]);
  }
#else
  /* AVX-only fallback: rows8()'s AVX path with the row loop inside the pair loop. */
  const __m256 vfour = _mm256_set1_ps(4.0f), veight = _mm256_set1_ps(8.0f);
  for(int rb = b0; rb < b1; ++rb){
    __m256 acc[KC_MAXB];
    for(int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for(int g = 0; g < G; ++g){
      const uint32_t* pw = (const uint32_t*)A->W + ((size_t)rb * G + g) * 64;
      __m256 s0[KC_MAXB], s1[KC_MAXB];
      for(int t = 0; t < nt; ++t){ s0[t] = _mm256_setzero_ps(); s1[t] = _mm256_setzero_ps(); }
      for(int p = 0; p < 64; p += 2){
        const uint32_t w0 = pw[p], w1 = pw[p + 1];
        float c0[8], c1[8];
        int i0[8], i1[8];
        for(int l = 0; l < 8; ++l){
          const int a = (int)((w0 >> (4 * l)) & 15u);
          const int b = (int)((w1 >> (4 * l)) & 15u);
          c0[l] = (float)a; i0[l] = a & 3;
          c1[l] = (float)b; i1[l] = b & 3;
        }
        const __m256 f0 = _mm256_loadu_ps(c0), f1 = _mm256_loadu_ps(c1);
        const __m256i x0 = _mm256_loadu_si256((const __m256i*)i0);
        const __m256i x1 = _mm256_loadu_si256((const __m256i*)i1);
        for(int t = 0; t < nt; ++t){
          const float* pt = T + (size_t)t * ts + (size_t)g * 512;
          const __m256 t0l = _mm256_broadcast_ps((const __m128*)(pt + 8 * p));
          const __m256 t0h = _mm256_broadcast_ps((const __m128*)(pt + 8 * p + 4));
          const __m256 t1l = _mm256_broadcast_ps((const __m128*)(pt + 8 * p + 8));
          const __m256 t1h = _mm256_broadcast_ps((const __m128*)(pt + 8 * p + 12));
          __m256 v0 = _mm256_blendv_ps(_mm256_permutevar_ps(t0l, x0), _mm256_permutevar_ps(t0h, x0),
                                       _mm256_cmp_ps(f0, vfour, _CMP_GE_OQ));
          __m256 v1 = _mm256_blendv_ps(_mm256_permutevar_ps(t1l, x1), _mm256_permutevar_ps(t1h, x1),
                                       _mm256_cmp_ps(f1, vfour, _CMP_GE_OQ));
          v0 = _mm256_andnot_ps(_mm256_cmp_ps(f0, veight, _CMP_EQ_OQ), v0);
          v1 = _mm256_andnot_ps(_mm256_cmp_ps(f1, veight, _CMP_EQ_OQ), v1);
          s0[t] = _mm256_add_ps(s0[t], v0);
          s1[t] = _mm256_add_ps(s1[t], v1);
        }
      }
      const __m256 sc = _mm256_loadu_ps(A->S + ((size_t)rb * G + g) * 8);
      for(int t = 0; t < nt; ++t)
        acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(_mm256_add_ps(s0[t], s1[t]), sc));
    }
    for(int t = 0; t < nt; ++t) _mm256_storeu_ps(Y + (size_t)t * ystride + rb * 8, acc[t]);
  }
#endif
}

/* The gemv() of a batch: up to 3 matrices on one input, T rows.  In the parallel region; ends in a
   barrier. */
static void gemv_multi(Model* m, int nt, int n, const Mat* const* A, float* const* ys, const int* ystride,
                       const float* xs, int xstride){
  const int prev = t_cur; TS(2);
  const int pairs = A[0]->C / 2;
  int a, b;
  span((long long)nt * pairs, &a, &b);
  for(int i = a; i < b; ++i){
    const int t = i / pairs, p = i % pairs;
    pair_tables(xs + (size_t)t * xstride, B.TAB + (size_t)t * B.tabp * 8, p, p + 1);
  }
  #pragma omp barrier
  long long Ut = 0;
  for(int i = 0; i < n; ++i) Ut += A[i]->R / 8;
  int a0, a1;
  span(Ut, &a0, &a1);
  int base = 0;
  for(int i = 0; i < n; ++i){
    const int U = A[i]->R / 8;
    const int u0 = a0 > base ? a0 - base : 0, u1 = (a1 - base) < U ? (a1 - base) : U;
    base += U;
    if(u1 > u0) rows8_multi(A[i], ys[i], ystride[i], nt, u0, u1);
  }
  #pragma omp barrier
  TS(prev);
}

/* ---- per-row norms and folds ---- */
static void rmsfold_b(Model* m, const float* w, int add, int nt){
  const int D = m->D, HB = m->HB, nb = D / HB, MW = B.MW;
  const int prev = t_cur; TS(1);
  #pragma omp for schedule(static)
  for(int t = 0; t < nt; ++t){
    float* h = B.H + (size_t)t * D;
    const float* y = B.Y + (size_t)t * D;
    double acc = 0.0;
    for(int e = 0; e < D; ++e){
      const float xf = add ? h[e] + y[e] : h[e];
      h[e] = xf;
      acc += (double)xf * (double)xf;
    }
    B.SCL[t] = 1.0f / sqrtf((float)(acc / D) + 1e-6f);
  }
  #pragma omp for schedule(static)
  for(int t = 0; t < nt; ++t){
    const float* h = B.H + (size_t)t * D;
    float* xn = B.XN + (size_t)t * D;
    const float s = B.SCL[t];
    for(int e = 0; e < D; ++e) xn[e] = (h[e] * s) * w[e];
  }
  #pragma omp for schedule(static)
  for(int i = 0; i < nt * nb; ++i){
    const int t = i / nb, bl = i % nb;
    float* o = B.XP + (size_t)t * MW + (size_t)bl * HB;
    const float* s = B.XN + (size_t)t * D + (size_t)bl * HB;
    for(int j = 0; j < HB; ++j) o[j] = m->sD[bl * HB + j] * s[j];
    fwht_block(o, HB);
  }
  TS(prev);
}

/* fold each row's first `width` entries of src (rows `sstride` apart) into that row's xp */
static void fold_b(const float* src, int sstride, const float* sign, float* dst, int width, int nt){
  const int nb = width / B.HB, MW = B.MW;
  #pragma omp for schedule(static)
  for(int i = 0; i < nt * nb; ++i){
    const int t = i / nb, bl = i % nb;
    float* o = dst + (size_t)t * MW + (size_t)bl * B.HB;
    const float* s = src + (size_t)t * sstride + (size_t)bl * B.HB;
    for(int j = 0; j < B.HB; ++j) o[j] = sign[bl * B.HB + j] * s[j];
    fwht_block(o, B.HB);
  }
}

/* ---- embedding, T rows ---- */
static void embed_b(Model* m, const int* toks, int nt){
  const int C = m->emb.C, D = m->D, HB = m->HB, MW = B.MW;
  static const signed char W0[9] = {-1, 0, 1, -1, 1, -1, 0, 1, 0}, W1[9] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};
  for(int t = 0; t < nt; ++t){
    const int tok = toks[t];
    #pragma omp for schedule(static)
    for(int g = 0; g < C / 128; ++g){
      const uint32_t* pw = (const uint32_t*)m->emb.W + ((size_t)(tok / 8) * (C / 128) + g) * 64;
      const float d = m->emb.S[((size_t)(tok / 8) * (C / 128) + g) * 8 + (tok & 7)];
      float* o = B.MM + (size_t)t * MW;
      for(int p = 0; p < 64; ++p){
        const int code = (pw[p] >> (4 * (tok & 7))) & 15;
        o[g * 128 + 2 * p] = (float)W0[code] * d;
        o[g * 128 + 2 * p + 1] = (float)W1[code] * d;
      }
    }
    #pragma omp for schedule(static)
    for(int bl = 0; bl < D / HB; ++bl){
      float* o = B.MM + (size_t)t * MW + (size_t)bl * HB;
      float* h = B.H + (size_t)t * D + (size_t)bl * HB;
      fwht_block(o, HB);
      for(int j = 0; j < HB; ++j) h[j] = m->sD[bl * HB + j] * o[j];
    }
  }
}

/* ---- attention, T rows ---- */
static void attn_b(Model* m, Layer* L, int nt, int pos0, int add){
  TS(4);
  const int NH = m->NH, NKV = m->NHKV, HD = m->HD, rh = m->rope_half, G = NH / NKV, MW = B.MW;
  const int QW = B.QW, KW = B.KW, AW = B.AW;
  rmsfold_b(m, L->attn_norm, add, nt);
  { const Mat* a[3] = {&L->q, &L->k, &L->v};
    float* yy[3] = {B.QN, B.KR, B.VV};
    int ys[3] = {QW, KW, KW};
    gemv_multi(m, nt, 3, a, yy, ys, B.XP, MW); }
  /* q/k norm + rope, and each row's own K/V write */
  #pragma omp for schedule(static)
  for(int i = 0; i < nt * (NH + NKV); ++i){
    const int t = i / (NH + NKV), r = i % (NH + NKV), pos = pos0 + t;
    float tmp[512];
    float* dst;
    if(r < NH){
      rms_rows(B.QN + (size_t)t * QW + (size_t)r * 2 * HD, L->q_norm, tmp, 1, HD);
      dst = B.QR + (size_t)t * AW + (size_t)r * HD;
    } else {
      rms_rows(B.KR + (size_t)t * KW + (size_t)(r - NH) * HD, L->k_norm, tmp, 1, HD);
      dst = B.KRO + (size_t)t * KW + (size_t)(r - NH) * HD;
    }
    const float* cs = m->cos_t + (size_t)pos * rh;
    const float* sn = m->sin_t + (size_t)pos * rh;
    for(int j = 0; j < rh; ++j){
      const float a = tmp[j], b = tmp[j + rh];
      dst[j] = a * cs[j] - b * sn[j];
      dst[j + rh] = a * sn[j] + b * cs[j];
    }
    for(int j = 2 * rh; j < HD; ++j) dst[j] = tmp[j];
    if(r >= NH){
      const int kv = r - NH;
      uint16_t* K = L->Kc + ((size_t)pos * NKV + kv) * HD;
      uint16_t* V = L->Vc + ((size_t)pos * NKV + kv) * HD;
      const float* vv = B.VV + (size_t)t * KW + (size_t)kv * HD;
      for(int j = 0; j < HD; ++j){ K[j] = f2h(dst[j]); V[j] = f2h(vv[j]); }
    }
  }
  /* scores and softmax: one (row, head) per item */
  #pragma omp for schedule(static)
  for(int i = 0; i < nt * NH; ++i){
    const int t = i / NH, hh = i % NH, pos = pos0 + t, kv = hh / G;
    const float* q = B.QR + (size_t)t * AW + (size_t)hh * HD;
    float* s = B.SC + ((size_t)t * NH + hh) * m->ctx;
    for(int p = 0; p <= pos; ++p){
      const uint16_t* K = L->Kc + ((size_t)p * NKV + kv) * HD;
      float a = 0.f;
      for(int j = 0; j < HD; ++j) a += h2f(K[j]) * q[j];
      s[p] = a * 0.0625f;
    }
    float mx = -1e30f;
    for(int p = 0; p <= pos; ++p) mx = fmaxf(mx, s[p]);
    float tot = 0.f;
    for(int p = 0; p <= pos; ++p) tot += expf(s[p] - mx);
    for(int p = 0; p <= pos; ++p) s[p] = expf(s[p] - mx) / tot;
  }
  /* weighted V, gated */
  #pragma omp for schedule(static)
  for(int i = 0; i < nt * NH * (HD / 32); ++i){
    const int per = NH * (HD / 32);
    const int t = i / per, u = i % per;
    const int hh = u / (HD / 32), d0 = (u % (HD / 32)) * 32, kv = hh / G, pos = pos0 + t;
    const float* p = B.SC + ((size_t)t * NH + hh) * m->ctx;
    float acc[32];
    for(int j = 0; j < 32; ++j) acc[j] = 0.f;
    for(int q = 0; q <= pos; ++q){
      const uint16_t* V = L->Vc + ((size_t)q * NKV + kv) * HD + d0;
      const float pt = p[q];
      for(int j = 0; j < 32; ++j) acc[j] += pt * h2f(V[j]);
    }
    float* dst = B.ATT + (size_t)t * AW + (size_t)hh * HD + d0;
    const float* gate = B.QN + (size_t)t * QW + (size_t)hh * 2 * HD + HD + d0;
    for(int j = 0; j < 32; ++j) dst[j] = acc[j] * sigm(gate[j]);
  }
  fold_b(B.ATT, AW, m->sA, B.XP, AW, nt);
  { const Mat* a[1] = {&L->o}; float* yy[1] = {B.Y}; int ys[1] = {m->D};
    gemv_multi(m, nt, 1, a, yy, ys, B.XP, MW); }
}

/* ---- GDN, T rows ---- */
static void gdn_b(Model* m, Layer* L, int nt, int add){
  TS(3);
  const int D = m->D, NV = m->NV, NK = m->NK, HK = m->HK, CC = m->CONV_CH, MW = B.MW;
  rmsfold_b(m, L->attn_norm, add, nt);
  { const Mat* a[2] = {&L->qkv, &L->gate}; float* yy[2] = {B.QKV, B.ZV}; int ys[2] = {CC, B.NG};
    gemv_multi(m, nt, 2, a, yy, ys, B.XP, MW); }
  /* the layer's own inputs, kept for a state replay after a rollback */
  {
    const size_t nl = (size_t)m->NL, il = (size_t)(L - m->L);
    #pragma omp for schedule(static)
    for(int t = 0; t < nt; ++t){
      memcpy(B.STQ + ((size_t)t * nl + il) * (size_t)CC, B.QKV + (size_t)t * CC, (size_t)CC * sizeof(float));
      memcpy(B.STX + ((size_t)t * nl + il) * (size_t)D, B.XN + (size_t)t * D, (size_t)D * sizeof(float));
    }
  }
  /* beta/alpha gates, per (row, head) */
  #pragma omp for schedule(static)
  for(int i = 0; i < nt * NV; ++i){
    const int t = i / NV, h = i % NV;
    const float* xn = B.XN + (size_t)t * D;
    float ab = 0.f, aa = 0.f;
    for(int j = 0; j < D; ++j){ ab += L->beta_w[(size_t)h * D + j] * xn[j]; aa += L->alpha_w[(size_t)h * D + j] * xn[j]; }
    B.GB[t * NV + h] = sigm(ab);
    const float a = aa + L->dt_b[h];
    const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
    B.GG[t * NV + h] = sp * L->ssm_a[h];
  }
  /* the rows are sequential where the state is: conv taps, then the delta rule */
  for(int t = 0; t < nt; ++t){
    float* cs = B.CS + (size_t)t * CC;
    const float* qk_in = B.QKV + (size_t)t * CC;
    #pragma omp for schedule(static)
    for(int c = 0; c < CC; ++c){
      float* cv = L->conv;
      const float c0 = cv[c], c1 = cv[CC + c], c2 = cv[2 * CC + c], nw = qk_in[c];
      const float o = c0 * L->conv_w[4 * c] + c1 * L->conv_w[4 * c + 1] + c2 * L->conv_w[4 * c + 2] + nw * L->conv_w[4 * c + 3];
      cv[c] = c1; cv[CC + c] = c2; cv[2 * CC + c] = nw;
      cs[c] = silu(o);
    }
    float* qk = B.QK + (size_t)t * 2 * NK * HK;
    #pragma omp for schedule(static)
    for(int r = 0; r < 2 * NK; ++r){
      const float* xr = cs + (size_t)r * HK;
      double acc = 0.0;
      for(int e = 0; e < HK; ++e) acc += (double)xr[e] * (double)xr[e];
      const float sc = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
      for(int e = 0; e < HK; ++e) qk[(size_t)r * HK + e] = xr[e] * sc;
    }
    const float scale = 1.0f / sqrtf((float)HK);
    float* gout = B.GOUT + (size_t)t * NV * HK;
    float* outn = B.OUTN + (size_t)t * NV * HK;
    #pragma omp for schedule(static)
    for(int h = 0; h < NV; ++h){
      const int kh = h % NK;
      const float* ks = qk + (size_t)(NK + kh) * HK;
      const float* qs = qk + (size_t)kh * HK;
      const float* v = cs + 2 * NK * HK + (size_t)h * HK;
      const float e = expf(B.GG[t * NV + h]);
      float* Mh = L->M + (size_t)h * HK * HK;
      float sk[128], o[128], dd[128];
      for(int j = 0; j < HK; ++j) sk[j] = 0.f;
      for(int i = 0; i < HK; ++i){ const float ki = ks[i]; const float* mr = Mh + i * HK; for(int j = 0; j < HK; ++j) sk[j] += (mr[j] * e) * ki; }
      for(int j = 0; j < HK; ++j){ dd[j] = (v[j] - sk[j]) * B.GB[t * NV + h]; o[j] = 0.f; }
      for(int i = 0; i < HK; ++i){
        const float ki = ks[i], qi = qs[i];
        float* mr = Mh + i * HK;
        for(int j = 0; j < HK; ++j){ const float mm = mr[j] * e + ki * dd[j]; mr[j] = mm; o[j] += mm * qi; }
      }
      for(int j = 0; j < HK; ++j) gout[(size_t)h * HK + j] = o[j] * scale;
      rms_rows(gout + (size_t)h * HK, L->ssm_norm, outn + (size_t)h * HK, 1, HK);
    }
  }
  /* gated output in the grouped v-head order, then the fold */
  const int rep = NV / NK;
  {
    float* mm = B.MM;
    #pragma omp for schedule(static)
    for(int i = 0; i < nt * NV * HK; ++i){
      const int t = i / (NV * HK), e = i % (NV * HK);
      int src = e;
      if(rep > 1){ const int hd = e % HK, grp = e / HK; src = hd + HK * (grp / rep + NK * (grp % rep)); }
      mm[(size_t)t * MW + e] = silu(B.ZV[(size_t)t * B.NG + src]) * B.OUTN[(size_t)t * NV * HK + src];
    }
  }
  fold_b(B.MM, MW, m->sV, B.XP, NV * HK, nt);
  { const Mat* a[1] = {&L->ssm_out}; float* yy[1] = {B.Y}; int ys[1] = {D};
    gemv_multi(m, nt, 1, a, yy, ys, B.XP, MW); }
}

/* ---- FFN, T rows ---- */
static void ffn_b(Model* m, Layer* L, int nt){
  const int NFF = m->NFF, MW = B.MW;
  TS(5);
  rmsfold_b(m, L->post_norm, 1, nt);
  { const Mat* a[2] = {&L->ffg, &L->ffu}; float* yy[2] = {B.FFG, B.FFU}; int ys[2] = {NFF, NFF};
    gemv_multi(m, nt, 2, a, yy, ys, B.XP, MW); }
  #pragma omp for schedule(static)
  for(int i = 0; i < nt * NFF; ++i){
    const int t = i / NFF, e = i % NFF;
    B.MM[(size_t)t * MW + e] = silu(B.FFG[(size_t)t * NFF + e]) * B.FFU[(size_t)t * NFF + e];
  }
  fold_b(B.MM, MW, m->sF, B.XP, NFF, nt);
  { const Mat* a[1] = {&L->ffd}; float* yy[1] = {B.Y}; int ys[1] = {m->D};
    gemv_multi(m, nt, 1, a, yy, ys, B.XP, MW); }
}

/* ---- output head, T rows ---- */
static void head_b(Model* m, int nt, int* out){
  TS(6);
  rmsfold_b(m, m->out_norm, 1, nt);
  { const Mat* a[1] = {&m->head}; float* yy[1] = {B.LOG}; int ys[1] = {B.vocab};
    gemv_multi(m, nt, 1, a, yy, ys, B.XP, B.MW); }
  #pragma omp for schedule(static)
  for(int t = 0; t < nt; ++t){
    const float* lg = B.LOG + (size_t)t * B.vocab;
    int bi = 0;
    float bv = lg[0];
    for(int i = 1; i < B.vocab; ++i) if(lg[i] > bv){ bv = lg[i]; bi = i; }
    out[t] = bi;
  }
}

/* One row of the GDN scan, from vectors a batch pass stored: conv taps, the q/k norm and the delta
   rule.  Same expressions and order as gdn(), so a replay lands on the same bits. */
static void gdn_state_row(Model* m, Layer* L, const float* qkv, const float* xn){
  const int D = m->D, NV = m->NV, NK = m->NK, HK = m->HK, CC = m->CONV_CH;
  const size_t slot = (size_t)omp_get_thread_num();      /* the replay runs one layer per thread */
  float* cs = B.R1 + slot * (CC > 2 * NK * HK ? (size_t)CC : (size_t)2 * NK * HK);
  float* qk = B.R2 + slot * (size_t)2 * NK * HK;
  float* g = B.R3 + slot * (size_t)2 * NV;
  for(int h = 0; h < NV; ++h){
    float ab = 0.f, aa = 0.f;
    for(int j = 0; j < D; ++j){ ab += L->beta_w[(size_t)h * D + j] * xn[j]; aa += L->alpha_w[(size_t)h * D + j] * xn[j]; }
    g[h] = sigm(ab);
    const float a = aa + L->dt_b[h];
    const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
    g[NV + h] = sp * L->ssm_a[h];
  }
  for(int c = 0; c < CC; ++c){
    float* cv = L->conv;
    const float c0 = cv[c], c1 = cv[CC + c], c2 = cv[2 * CC + c], nw = qkv[c];
    const float o = c0 * L->conv_w[4 * c] + c1 * L->conv_w[4 * c + 1] + c2 * L->conv_w[4 * c + 2] + nw * L->conv_w[4 * c + 3];
    cv[c] = c1; cv[CC + c] = c2; cv[2 * CC + c] = nw;
    cs[c] = silu(o);
  }
  for(int r = 0; r < 2 * NK; ++r){
    const float* xr = cs + (size_t)r * HK;
    double acc = 0.0;
    for(int e = 0; e < HK; ++e) acc += (double)xr[e] * (double)xr[e];
    const float sc = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
    for(int e = 0; e < HK; ++e) qk[(size_t)r * HK + e] = xr[e] * sc;
  }
  for(int h = 0; h < NV; ++h){
    const int kh = h % NK;
    const float* ks = qk + (size_t)(NK + kh) * HK;
    const float* v = cs + 2 * NK * HK + (size_t)h * HK;
    const float e = expf(g[NV + h]);
    float* Mh = L->M + (size_t)h * HK * HK;
    float sk[128], dd[128];
    for(int j = 0; j < HK; ++j) sk[j] = 0.f;
    for(int i = 0; i < HK; ++i){ const float ki = ks[i]; const float* mr = Mh + i * HK; for(int j = 0; j < HK; ++j) sk[j] += (mr[j] * e) * ki; }
    for(int j = 0; j < HK; ++j) dd[j] = (v[j] - sk[j]) * g[h];
    for(int i = 0; i < HK; ++i){
      const float ki = ks[i];
      float* mr = Mh + i * HK;
      for(int j = 0; j < HK; ++j){ const float mm = mr[j] * e + ki * dd[j]; mr[j] = mm; }
    }
  }
}

/* A batch is only defined on a whole model: a layer this backend does not own has no matrices. */
static int batch_check(Model* m){
  if(!m || m->NL <= 0) return -1;
  for(int il = 0; il < m->NL; ++il){
    Layer* L = m->L + il;
    if(L->is_attn){ if(!L->q.R || !L->k.R || !L->v.R || !L->o.R) return -3; }
    else { if(!L->qkv.R || !L->gate.R || !L->ssm_out.R) return -3; }
    if(!L->ffg.R || !L->ffu.R || !L->ffd.R) return -3;
  }
  if(!m->emb.R || !m->head.R) return -4;
  return 0;
}

/* T rows at consecutive positions: toks[t] is the token at pos0 + t, out[t] the greedy token that
   row predicts (the token plain decode would produce at pos0 + t + 1).  Model state advances through
   all T rows; kc_batch_rollback(keep) undoes the rows past `keep`.  Returns T, or < 0 on an error
   (nothing has been advanced in that case). */
KC_API int kc_step_batch(Model* m, const int* toks, int nt, int pos0, int* out){
  if(!toks || !out || nt < 2 || nt > KC_MAXB || pos0 < 0) return -1;
  if(!m || pos0 + nt > m->ctx) return -1;
  const int bad = batch_check(m);
  if(bad) return bad;
  if(!batch_prepare(m, nt)) return -2;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  {
    /* the GDN state before the batch, for kc_batch_rollback() */
    #pragma omp for schedule(static)
    for(int il = 0; il < m->NL; ++il){
      Layer* L = m->L + il;
      if(L->is_attn) continue;
      const size_t mb = (size_t)m->NV * m->HK * m->HK * sizeof(float);
      const size_t cb = (size_t)3 * m->CONV_CH * sizeof(float);
      memcpy(B.SNAP + B.soff[il], L->M, mb);
      memcpy(B.SNAP + B.soff[il] + mb, L->conv, cb);
    }
    embed_b(m, toks, nt);
    for(int il = 0; il < m->NL; ++il){
      Layer* L = m->L + il;
      if(L->is_attn) attn_b(m, L, nt, pos0, il > 0); else gdn_b(m, L, nt, il > 0);
      ffn_b(m, L, nt);
    }
    head_b(m, nt, out);
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return nt;
}

/* Undo the rows past `keep`: restore the pre-batch GDN state and replay rows 0..keep-1 of the scan
   from the stored vectors (no weights are read).  K/V needs nothing - the next pass rewrites every
   position it reads.  keep == the batch size is a no-op. */
KC_API int kc_batch_rollback(Model* m, int keep){
  if(!m || B.owner != m || B.nt < 2) return -1;   /* the replay reads B.SNAP, which belongs to B.owner */
  if(keep < 0 || keep > B.nt) return -2;
  if(keep == B.nt) return 0;
  t_last = omp_get_wtime(); t_cur = 0;
  const size_t mb = (size_t)m->NV * m->HK * m->HK * sizeof(float);
  const size_t cb = (size_t)3 * m->CONV_CH * sizeof(float);
  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for(int il = 0; il < m->NL; ++il){
      Layer* L = m->L + il;
      if(L->is_attn) continue;
      memcpy(L->M, B.SNAP + B.soff[il], mb);
      memcpy(L->conv, B.SNAP + B.soff[il] + mb, cb);
    }
    #pragma omp for schedule(static)
    for(int il = 0; il < m->NL; ++il){
      Layer* L = m->L + il;
      if(L->is_attn) continue;
      for(int r = 0; r < keep; ++r)
        gdn_state_row(m, L, B.STQ + ((size_t)r * m->NL + il) * m->CONV_CH,
                      B.STX + ((size_t)r * m->NL + il) * m->D);
    }
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return keep;
}

/* The logits of row t from the last batch (for exactness checks against plain decode). */
KC_API const float* kc_batch_logits(Model* m, int t){
  if(!m || B.owner != m || B.nt < 2 || t < 0 || t >= B.nt) return 0;
  return B.LOG + (size_t)t * B.vocab;
}

KC_API int kc_batch_vocab(Model* m){ return (m && B.owner == m && B.nt >= 2) ? B.vocab : 0; }

/* Free the batch arena (SNAP + scratch). Idempotent; the next kc_step_batch re-prepares it, so an
   Engine may call this on teardown even while another Engine in the process still exists. */
KC_API int kc_batch_release(void){ batch_drop(); return 0; }

/* ------------------------------------------------------------------------------------- model state */
/* The GDN state (conv taps and the SSM matrix of every GDN layer) as one flat float block.  A
   drafter that keeps its own K/V and GDN state between rounds uses this to step back to the
   accepted prefix: the drafts it fed sit on top of that prefix, and only the state has to move.
   K/V needs nothing, for the same reason as in a rollback: the next pass rewrites every position
   it reads, in ascending order. */
static int state_layers(Model* m){ return m->NL; }

KC_API int kc_state_len(Model* m){
  if(!m) return -1;
  int n = 0;
  for(int il = 0; il < state_layers(m); ++il)
    if(!m->L[il].is_attn) n += m->NV * m->HK * m->HK + 3 * m->CONV_CH;
  return n;
}

KC_API int kc_state_save(Model* m, float* buf){
  if(!m || !buf) return -1;
  size_t at = 0;
  const size_t mb = (size_t)m->NV * m->HK * m->HK, cb = (size_t)3 * m->CONV_CH;
  for(int il = 0; il < m->NL; ++il){
    Layer* L = m->L + il;
    if(L->is_attn) continue;
    memcpy(buf + at, L->M, mb * sizeof(float)); at += mb;
    memcpy(buf + at, L->conv, cb * sizeof(float)); at += cb;
  }
  return (int)at;
}

/* Length-checked restore: a caller that passes a short buffer gets an error, not an out-of-bounds
   read. kc_state_load() below keeps the old signature for external callers and sizes it itself. */
KC_API int kc_state_load_len(Model* m, const float* buf, int n){
  if(!m || !buf || n != kc_state_len(m)) return -1;
  size_t at = 0;
  const size_t mb = (size_t)m->NV * m->HK * m->HK, cb = (size_t)3 * m->CONV_CH;
  for(int il = 0; il < m->NL; ++il){
    Layer* L = m->L + il;
    if(L->is_attn) continue;
    memcpy(L->M, buf + at, mb * sizeof(float)); at += mb;
    memcpy(L->conv, buf + at, cb * sizeof(float)); at += cb;
  }
  return (int)at;
}

KC_API int kc_state_load(Model* m, const float* buf){
  if(!m || !buf) return -1;
  size_t at = 0;
  const size_t mb = (size_t)m->NV * m->HK * m->HK, cb = (size_t)3 * m->CONV_CH;
  for(int il = 0; il < m->NL; ++il){
    Layer* L = m->L + il;
    if(L->is_attn) continue;
    memcpy(L->M, buf + at, mb * sizeof(float)); at += mb;
    memcpy(L->conv, buf + at, cb * sizeof(float)); at += cb;
  }
  return (int)at;
}

/* The residual of batch row t (the value plain decode keeps in m->h after its head). */
KC_API const float* kc_batch_hidden(Model* m, int t){
  if(!m || B.owner != m || B.nt < 2 || t < 0 || t >= B.nt) return 0;
  return B.H + (size_t)t * B.D;
}
