// Softmax attention, flash style, on the GPU neural accelerators.
//
// Rows: for key/value head g, query row m is token m / G, query head
// g * G + m % G, at position pos0 + m / G; it sees keys 0 .. pos0 + m / G.
// A SIMD-group owns a tile of 16 rows and walks the keys in steps of BK: the
// scores of a step live in registers only, with a running row maximum and sum
// (online softmax), so nothing of size rows x keys is ever written.
//
// The head dimension is split over WN SIMD-groups (DS = HD / WN columns each).
// Each computes partial scores over its columns of Q and K; the partials are
// summed through threadgroup memory in a fixed order, and each SIMD-group then
// accumulates P x V for its own columns. WM such row groups share a threadgroup.
//
// Exactness. A row comes out the same bits whichever pass computes it (a
// prompt chunk, a decode step, a speculative pass), so its arithmetic does not
// depend on the pass: keys are taken in segments of SEG at fixed positions,
// each segment's softmax starts afresh, and segments are merged in order by
// merge(). With many rows (SPLIT 0) a threadgroup runs the segments one after
// the other; with few rows over many keys (SPLIT 1) each segment runs in a
// threadgroup of its own, for parallelism over the keys, and attn_combine
// merges them with the same merge(). A 16-row tile walks the same steps either
// way (up to the keys its own last row sees), and keys or segments a row cannot
// see change nothing for it.
//
// Cache layouts (kernels/ops.metal, attn_prep): K per block of 256 keys as
// [dim][key], which is K^T, the right operand of Q x K^T as it is; V as
// [key][dim].
//
// Register fragments: 16 x 16, eight elements per lane (rows r and r + 8,
// columns c .. c + 3), the operand layout of the accelerator's 16 x 32 x 16
// SIMD-group product on M5/M6-class GPUs (as in MLX's NAX kernels and
// h3metal's kernels/nax.h), so a product's result feeds the next one in
// registers.
//
// Every loop over an array of fragments is fully unrolled: one loop with a run-time index puts the whole
// array in memory, and the products then run 4x slower (20.5 against 5.2 TFLOPS on M6).
//
// Specialization: HD (head dim, a multiple of 64), WN, WM, BK (a multiple of 32), SEG (a multiple of BK), SPLIT.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#include "params.h"

using namespace metal;

#ifndef SPLIT
#define SPLIT 0
#endif

#define KEYS 256  // keys per cache block
#define DS (HD / WN)
#define DF (DS / 16)
#define KF (BK / 16)

namespace frag {

typedef vec<float, 8> f32;
typedef vec<half, 8> f16;

// (column, row) of this lane's first element.
inline short2 coord(ushort lane) {
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  return short2{fn, fm};
}

// Element (r, c) at src[r * ld + c].
inline void load(thread f16 &f, const device half *src, int ld, short2 sc) {
  src += sc.y * ld + sc.x;
  const half4 a = *(const device half4 *)src;
  const half4 b = *(const device half4 *)(src + 8 * ld);
  f = f16(a, b);
}

// c0 | c1 (16 x 32) += a (16 x 16) x b0 | b1 (16 x 16 each).
inline void mma(thread f32 &c0, thread f32 &c1, const thread f16 &a, const thread f16 &b0, const thread f16 &b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
                                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
  auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
  auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
  auto cc = op.get_destination_cooperative_tensor<remove_addrspace_t<decltype(ca)>, remove_addrspace_t<decltype(cb)>,
                                                  float>();
#pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    ca[i] = a[i];
    cb[i] = b0[i];
    cb[8 + i] = b1[i];
    cc[i] = c0[i];
    cc[8 + i] = c1[i];
  }
  op.run(ca, cb, cc);
#pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    c0[i] = cc[i];
    c1[i] = cc[8 + i];
  }
}

// Over the 16 columns of each of the lane's two rows: the four lanes of a row differ in lane bits 0 and 3.
inline float2 row_max(const thread f32 &f) {
  float a = max(max(f[0], f[1]), max(f[2], f[3]));
  float b = max(max(f[4], f[5]), max(f[6], f[7]));
  a = max(a, simd_shuffle_xor(a, ushort(1)));
  a = max(a, simd_shuffle_xor(a, ushort(8)));
  b = max(b, simd_shuffle_xor(b, ushort(1)));
  b = max(b, simd_shuffle_xor(b, ushort(8)));
  return float2(a, b);
}

}  // namespace frag

static inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }

// Running totals (M, L, O) of a row merged with a segment's (m, l, Os); m is finite.
// attn_flash and attn_combine both go through these two, so they round alike.
static inline float2 merge_scales(thread float &M, thread float &L, float m, float l) {
  const float Mn = max(M, m);
  const float a = fast::exp2(M - Mn), b = fast::exp2(m - Mn);
  L = fma(L, a, l * b);
  M = Mn;
  return float2(a, b);
}
static inline float merge_value(float O, float Os, float2 ab) { return fma(O, ab.x, Os * ab.y); }

// The output: normalized, the sigmoid gate, the column scale of o_proj.
static inline half finish(float O, float L, float gate, float cs) { return half(O / L * sigmoidf(gate) * cs); }

kernel void attn_flash(device const half *q [[buffer(0)]],
                       device const half *Kc [[buffer(1)]],
                       device const half *Vc [[buffer(2)]],
                       device const float *proj [[buffer(3)]],
                       device half *out [[buffer(4)]],
                       constant AttnParams &p [[buffer(5)]],
                       device const float *cs [[buffer(6)]],
                       device float *part [[buffer(7)]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]])
{
  const int G = p.H / p.KVH, g = int(tg.z);
  const int rg = sg / WN, dg = sg % WN;
  const int m0 = (int(tg.x) * WM + rg) * 16;           // this SIMD-group's first row
  const short2 sc = frag::coord(lane);
  const int r0 = m0 + sc.y, r1 = r0 + 8;
  const int lim0 = p.pos0 + min(r0, p.M - 1) / G + 1;  // keys row r0 sees (padding rows: as the last row)
  const int lim1 = p.pos0 + min(r1, p.M - 1) / G + 1;
  // Keys this tile sees, and those of the threadgroup's last tile (the loop bound every SIMD-group keeps,
  // for the barriers).
  const int kend = p.pos0 + min(m0 + 15, p.M - 1) / G + 1;
  const int kend_tg = p.pos0 + min((int(tg.x) * WM + WM) * 16 - 1, p.M - 1) / G + 1;
  const float scl = p.scale * 1.4426950408889634f;     // scores in log2 units

  threadgroup float xchg[WM][WN][KF * 8 * 32];
  threadgroup float *mine = xchg[rg][dg] + lane * (KF * 8);

  // This SIMD-group's columns of its 16 query rows, for the whole walk.
  frag::f16 qf[DF];
  {
    const device half *Q = q + (size_t(g) * p.q_rows + m0) * p.D + dg * DS;
#pragma clang loop unroll(full)
    for (short d = 0; d < DF; d++) frag::load(qf[d], Q + d * 16, p.D, sc);
  }
  const device half *Kg = Kc + size_t(g) * p.Lb * KEYS * p.D;
  const device half *Vg = Vc + size_t(g) * p.Lb * KEYS * p.D + dg * DS;

#if SPLIT
  const int s_first = int(tg.y), s_end = s_first + 1;
  if (s_first * SEG >= kend_tg) return;               // the whole threadgroup: past every row's keys
#else
  const int s_first = 0, s_end = (kend_tg + SEG - 1) / SEG;
  float Mt0 = 0.0f, Mt1 = 0.0f, Lt0 = 0.0f, Lt1 = 0.0f;
  frag::f32 ot[DF];
#endif

  for (int s = s_first; s < s_end; s++) {
    float ms0 = -INFINITY, ms1 = -INFINITY, ls0 = 0.0f, ls1 = 0.0f;
    frag::f32 os[DF];
#pragma clang loop unroll(full)
    for (short d = 0; d < DF; d++) os[d] = 0.0f;

    const int k_lo = s * SEG, k_hi = min(k_lo + SEG, kend_tg);
    for (int kb = k_lo; kb < k_hi; kb += BK) {
      const bool live = kb < kend;                     // this tile still has keys here
      // Partial scores over this SIMD-group's columns.
      frag::f32 sp[KF];
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) sp[j] = 0.0f;
      if (live) {
        const device half *Kb = Kg + (size_t(kb / KEYS) * p.D + dg * DS) * KEYS + kb % KEYS;
#pragma clang loop unroll(full)
        for (short j = 0; j < KF; j += 2) {
#pragma clang loop unroll(full)
          for (short d = 0; d < DF; d++) {
            frag::f16 k0, k1;
            frag::load(k0, Kb + d * 16 * KEYS + j * 16, KEYS, sc);
            frag::load(k1, Kb + d * 16 * KEYS + j * 16 + 16, KEYS, sc);
            frag::mma(sp[j], sp[j + 1], qf[d], k0, k1);
          }
        }
      }
      // Sum the WN partials in a fixed order.
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) {
#pragma clang loop unroll(full)
        for (short e = 0; e < 8; e++) mine[j * 8 + e] = sp[j][e];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      frag::f32 sv[KF];
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) {
#pragma clang loop unroll(full)
        for (short e = 0; e < 8; e++) {
          float v = xchg[rg][0][lane * (KF * 8) + j * 8 + e];
#pragma clang loop unroll(full)
          for (short w = 1; w < WN; w++) v += xchg[rg][w][lane * (KF * 8) + j * 8 + e];
          sv[j][e] = v;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (!live) continue;

      // Scale and mask, then the online softmax.
      float n0 = ms0, n1 = ms1;
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) {
#pragma clang loop unroll(full)
        for (short e = 0; e < 8; e++) {
          const int key = kb + j * 16 + sc.x + (e & 3);
          sv[j][e] = key < (e < 4 ? lim0 : lim1) ? sv[j][e] * scl : -INFINITY;
        }
        const float2 mx = frag::row_max(sv[j]);
        n0 = max(n0, mx.x);
        n1 = max(n1, mx.y);
      }
      frag::f16 pf[KF];
      float sum0 = 0.0f, sum1 = 0.0f;
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) {
#pragma clang loop unroll(full)
        for (short e = 0; e < 8; e++) {
          const float v = sv[j][e];
          pf[j][e] = half(v == -INFINITY ? 0.0f : fast::exp2(v - (e < 4 ? n0 : n1)));
        }
        sum0 += (float(pf[j][0]) + float(pf[j][1])) + (float(pf[j][2]) + float(pf[j][3]));
        sum1 += (float(pf[j][4]) + float(pf[j][5])) + (float(pf[j][6]) + float(pf[j][7]));
      }
      sum0 += simd_shuffle_xor(sum0, ushort(1));
      sum0 += simd_shuffle_xor(sum0, ushort(8));
      sum1 += simd_shuffle_xor(sum1, ushort(1));
      sum1 += simd_shuffle_xor(sum1, ushort(8));
      const float f0 = n0 == ms0 ? 1.0f : fast::exp2(ms0 - n0);
      const float f1 = n1 == ms1 ? 1.0f : fast::exp2(ms1 - n1);
      ls0 = fma(ls0, f0, sum0);
      ls1 = fma(ls1, f1, sum1);
      ms0 = n0;
      ms1 = n1;
#pragma clang loop unroll(full)
      for (short d = 0; d < DF; d++) {
#pragma clang loop unroll(full)
        for (short e = 0; e < 8; e++) os[d][e] *= e < 4 ? f0 : f1;
      }
      // O += P x V over this SIMD-group's columns.
      const device half *Vb = Vg + size_t(kb) * p.D;
#pragma clang loop unroll(full)
      for (short d = 0; d < DF; d += 2) {
#pragma clang loop unroll(full)
        for (short j = 0; j < KF; j++) {
          frag::f16 v0, v1;
          frag::load(v0, Vb + j * 16 * p.D + d * 16, p.D, sc);
          frag::load(v1, Vb + j * 16 * p.D + d * 16 + 16, p.D, sc);
          frag::mma(os[d], os[d + 1], pf[j], v0, v1);
        }
      }
    }

#if SPLIT
    // The segment's totals, for attn_combine: [g][s][row] (m, l) and [g][s][row][D].
    const size_t at = (size_t(g) * p.seg_cap + s) * p.rows_cap;
    device float *pml = part + at * 2;
    device float *po = part + size_t(p.KVH) * p.seg_cap * p.rows_cap * 2 + at * p.D;
    if (dg == 0 && (lane & 9) == 0) {                  // one lane per row
      if (r0 < p.M) *(device float2 *)(pml + r0 * 2) = float2(ms0, ls0);
      if (r1 < p.M) *(device float2 *)(pml + r1 * 2) = float2(ms1, ls1);
    }
#pragma clang loop unroll(full)
    for (short d = 0; d < DF; d++) {
      const int col = dg * DS + d * 16 + sc.x;
      if (r0 < p.M) *(device float4 *)(po + size_t(r0) * p.D + col) = os[d].xyzw;
      if (r1 < p.M) *(device float4 *)(po + size_t(r1) * p.D + col) = float4(os[d][4], os[d][5], os[d][6], os[d][7]);
    }
#else
    if (s == 0) {
      Mt0 = ms0, Mt1 = ms1, Lt0 = ls0, Lt1 = ls1;
#pragma clang loop unroll(full)
      for (short d = 0; d < DF; d++) ot[d] = os[d];
    } else {
      // A segment the row cannot see has m = -inf and merges as nothing.
      const float2 ab0 = ms0 != -INFINITY ? merge_scales(Mt0, Lt0, ms0, ls0) : float2(1.0f, 0.0f);
      const float2 ab1 = ms1 != -INFINITY ? merge_scales(Mt1, Lt1, ms1, ls1) : float2(1.0f, 0.0f);
      const bool use0 = ms0 != -INFINITY, use1 = ms1 != -INFINITY;
#pragma clang loop unroll(full)
      for (short d = 0; d < DF; d++) {
#pragma clang loop unroll(full)
        for (short e = 0; e < 8; e++) {
          if (e < 4 ? use0 : use1) ot[d][e] = merge_value(ot[d][e], os[d][e], e < 4 ? ab0 : ab1);
        }
      }
    }
#endif
  }

#if !SPLIT
  // Normalize, gate, column scale; rows beyond M are padding.
#pragma clang loop unroll(full)
  for (short half_ = 0; half_ < 2; half_++) {
    const int r = half_ ? r1 : r0;
    if (r >= p.M) continue;
    const float L = half_ ? Lt1 : Lt0;
    const int t = r / G, hh = g * G + r % G;
#pragma clang loop unroll(full)
    for (short d = 0; d < DF; d++) {
#pragma clang loop unroll(full)
      for (short e = 0; e < 4; e++) {
        const int col = dg * DS + d * 16 + sc.x + e;
        const float gate = proj[size_t(t) * p.proj_stride + hh * 2 * p.D + p.D + col];
        out[size_t(t) * p.out_stride + hh * p.D + col] = finish(ot[d][half_ * 4 + e], L, gate, cs[hh * p.D + col]);
      }
    }
  }
#endif
}

// Few rows over many keys: merges the segments attn_flash (SPLIT 1) left, in order. One SIMD-group per (row,
// key/value head); lane l owns columns l * HD / 32 ...
kernel void attn_combine(device const float *part [[buffer(0)]],
                         device const float *proj [[buffer(1)]],
                         device half *out [[buffer(2)]],
                         constant AttnParams &p [[buffer(3)]],
                         device const float *cs [[buffer(4)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]])
{
  constexpr int J = HD / 32;
  const int G = p.H / p.KVH, m = int(tg.y), g = int(tg.z);
  const int n_seg = (p.pos0 + m / G + 1 + SEG - 1) / SEG;   // segments with keys this row sees
  const device float *pml = part;
  const device float *po = part + size_t(p.KVH) * p.seg_cap * p.rows_cap * 2;
  float M = 0.0f, L = 0.0f, O[J];
  for (int s = 0; s < n_seg; s++) {
    const size_t at = (size_t(g) * p.seg_cap + s) * p.rows_cap + m;
    const float2 ml = *(const device float2 *)(pml + at * 2);
    const device float *os = po + at * p.D + lane * J;
    if (s == 0) {
      M = ml.x, L = ml.y;
      for (int j = 0; j < J; j++) O[j] = os[j];
    } else {
      const float2 ab = merge_scales(M, L, ml.x, ml.y);
      for (int j = 0; j < J; j++) O[j] = merge_value(O[j], os[j], ab);
    }
  }
  const int t = m / G, hh = g * G + m % G;
  for (int j = 0; j < J; j++) {
    const int col = lane * J + j;
    const float gate = proj[size_t(t) * p.proj_stride + hh * 2 * p.D + p.D + col];
    out[size_t(t) * p.out_stride + hh * p.D + col] = finish(O[j], L, gate, cs[hh * p.D + col]);
  }
}
