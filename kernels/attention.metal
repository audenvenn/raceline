// Softmax attention as two matrix products on the GPU neural accelerators.
//
// For each key/value head (queries of G heads share it) and a group of rows:
//
//   attn_scores   S[m][n] = Q[m] . K[n]                 hardware matmul, K = head dim
//   attn_softmax  P[m][n] = exp(scale * (S - max)),     keys past the row's own position get 0
//   attn_pv       out[m]  = (P[m] x V) / sum(P[m])      hardware matmul over the keys, then the output gate
//
// Reading the cache this way costs one pass over K and one over V for the
// whole group of rows, however many query heads and rows share them. A decode
// step, a speculative pair of rows and a prefill chunk all use these kernels,
// and they produce the same bits for a given row: every row's softmax is
// computed alone, and the products are accumulated over fixed blocks of 256
// keys in order, so trailing zeros and neighbouring rows change nothing.
//
// Cache layouts: K is stored per block of 256 keys as [dim][key] (the layout
// the weight kernels use, one contiguous stream per output tile); V as
// [key][dim], which is that same layout seen from the second product, where
// the keys are the inner dimension.
//
// Specialization macros: TM, TN (tile rows/cols), NSG (SIMD-groups per threadgroup).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#include "params.h"

using namespace metal;
using namespace mpp::tensor_ops;

#define KEYS 256  // keys per cache block and per partial product

static inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }

kernel void attn_scores(device half *q [[buffer(0)]],
                        device half *Kc [[buffer(1)]],
                        device float *sc [[buffer(2)]],
                        constant AttnParams &p [[buffer(3)]],
                        uint3 tgid [[threadgroup_position_in_grid]])
{
  const int n0 = int(tgid.x) * TN, m0 = int(tgid.y) * TM, g = int(tgid.z);
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, false, true,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;

  array<int, 2> a_str = {1, p.D};
  array<int, 2> w_str = {1, KEYS};
  array<int, 2> c_str = {1, p.n_stride};
  tensor<device half, dextents<int, 2>, tensor_inline> A(q + size_t(g) * p.q_rows * p.D,
                                                         dextents<int, 2>(p.D, p.kv_rows), a_str);
  tensor<device half, dextents<int, 2>, tensor_inline> W(Kc + (size_t(g) * p.Lb + n0 / KEYS) * p.D * KEYS,
                                                         dextents<int, 2>(KEYS, p.D), w_str);
  tensor<device float, dextents<int, 2>, tensor_inline> C(sc + size_t(g) * p.kv_rows * p.n_stride,
                                                          dextents<int, 2>(p.n_pad, p.kv_rows), c_str);
  auto tA = A.slice(0, m0);
  auto tW = W.slice(n0 % KEYS, 0);
  auto tC = C.slice(n0, m0);
  op.run(tA, tW, tC);
}

// One SIMD-group per (row, key/value head); lane l owns keys l, l+32, ...
// Writes the unnormalized probabilities in half precision and their sum.
kernel void attn_softmax(device const float *sc [[buffer(0)]],
                         device half *pr [[buffer(1)]],
                         device float *lsum [[buffer(2)]],
                         constant AttnParams &p [[buffer(3)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]])
{
  const int m = int(tg.y), g = int(tg.z);
  const int n_keys = p.pos0 + m / (p.H / p.KVH) + 1;
  const device float *s = sc + (size_t(g) * p.kv_rows + m) * p.n_stride;
  device half *o = pr + (size_t(g) * p.kv_rows + m) * p.n_stride;

  float mx = -INFINITY;
  for (int k = int(lane); k < n_keys; k += 32) mx = max(mx, s[k]);
  mx = simd_max(mx);

  float sum = 0.0f;
  int k = int(lane);
  for (; k < n_keys; k += 32) {
    const half h = half(exp((s[k] - mx) * p.scale));
    o[k] = h;
    sum += float(h);
  }
  for (; k < p.n_pad; k += 32) o[k] = half(0.0f);
  sum = simd_sum(sum);
  if (lane == 0) lsum[g * p.kv_rows + m] = sum;
}

kernel void attn_pv(device half *pr [[buffer(0)]],
                    device half *Vc [[buffer(1)]],
                    device const float *lsum [[buffer(2)]],
                    device const float *proj [[buffer(3)]],
                    device half *out [[buffer(4)]],
                    constant AttnParams &p [[buffer(5)]],
                    device const float *cs [[buffer(6)]],
                    uint3 tgid [[threadgroup_position_in_grid]])
{
  const int e0 = int(tgid.x) * TN, m0 = int(tgid.y) * TM, g = int(tgid.z);
  const int G = p.H / p.KVH;
  constexpr auto desc = matmul2d_descriptor(TM, TN, KEYS, false, false, true, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;

  array<int, 2> a_str = {1, p.n_stride};
  array<int, 2> w_str = {1, p.D};
  tensor<device half, dextents<int, 2>, tensor_inline> A(pr + size_t(g) * p.kv_rows * p.n_stride,
                                                         dextents<int, 2>(p.n_pad, p.kv_rows), a_str);
  tensor<device half, dextents<int, 2>, tensor_inline> W(Vc + size_t(g) * p.Lb * KEYS * p.D,
                                                         dextents<int, 2>(p.D, p.n_pad), w_str);
  auto tA = A.slice<KEYS, TM>(0, m0);
  auto tW = W.slice<TN, KEYS>(e0, 0);
  auto acc = op.get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
  auto part = op.get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
#pragma clang loop unroll(full)
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  }

  // Blocks of keys the rows of this tile can see; beyond them P is zero.
  const int last = min(m0 + TM, p.M) - 1;
  const int n_end = min(p.n_pad, (p.pos0 + last / G + KEYS) / KEYS * KEYS);
  for (int b = 0; b < n_end; b += KEYS) {
    tA = A.slice<KEYS, TM>(b, m0);
    tW = W.slice<TN, KEYS>(e0, b);
    op.run(tA, tW, part);
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (acc.is_valid_element(i)) acc[i] += part[i];
    }
  }

  // Normalize, apply the sigmoid output gate and the column scale of o_proj.
#pragma clang loop unroll(full)
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) {
      auto idx = acc.get_multidimensional_index(i);  // {dim, row} within the tile
      const int m = m0 + int(idx[1]);
      if (m < p.M) {
        const int t = m / G, hh = g * G + m % G, e = e0 + int(idx[0]);
        const float gate = proj[size_t(t) * p.proj_stride + hh * 2 * p.D + p.D + e];
        out[size_t(t) * p.out_stride + hh * p.D + e] =
            half(acc[i] / lsum[g * p.kv_rows + m] * sigmoidf(gate) * cs[hh * p.D + e]);
      }
    }
  }
}
