// Quantized linear layer on the GPU neural accelerators.
//
//   C[M][N] = A[M][K] * Wq[N][K]^T          Wq[n][k] = s[n][g] * q[n][k] + b[n][g],  g = k / GROUP
//
// q is 4-bit unsigned and is handed to the accelerator as-is (`uint4b_format`);
// nothing is dequantized in software. The affine scale is applied to each
// group's partial product while it still sits in registers, and the bias term
// collapses to b[n][g] * (sum of A over the group), which the caller provides
// as `gsum`. The result is numerically the same affine 4-bit model MLX runs.
//
// Weight layout ("blocked"): output rows are cut into blocks of BLOCK; a block
// is stored as [K][BLOCK] nibbles, i.e. BLOCK/2 bytes per input index. One
// threadgroup therefore streams one contiguous region, which is what lets a
// single-token pass run close to the memory-bandwidth ceiling. Scales and
// biases use the matching [N/BLOCK][K/GROUP][BLOCK] layout, stored as bfloat16
// (the precision MLX keeps them in, so the values carry over bit for bit).
//
// Specialization macros: TM, TN (tile rows/cols; TN divides BLOCK), NSG
// (SIMD-groups per threadgroup), GROUP, BLOCK, OUT_T (half or float).

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#include "params.h"

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef ACCUM
#define ACCUM 0  // 1: add the result into C (residual connection) instead of overwriting it
#endif

// gsum[m][g] = sum over the g-th group of A[m][*]. One thread per (g, m).
kernel void group_sums(device half *a [[buffer(0)]],
                       device float *gsum [[buffer(1)]],
                       constant LinearParams &p [[buffer(2)]],
                       uint2 tid [[thread_position_in_grid]])
{
  if (int(tid.x) >= p.G || int(tid.y) >= p.M) return;
  const device half *row = a + size_t(tid.y) * p.a_stride + tid.x * GROUP;
  float acc = 0;
  for (int k = 0; k < GROUP; ++k) acc += float(row[k]);
  gsum[size_t(tid.y) * p.G + tid.x] = acc;
}

kernel void linear_q4a(device half *a [[buffer(0)]],
                       device uchar *w [[buffer(1)]],
                       device bfloat *s [[buffer(2)]],
                       device bfloat *b [[buffer(3)]],
                       device float *gsum [[buffer(4)]],
                       device OUT_T *c [[buffer(5)]],
                       constant LinearParams &p [[buffer(6)]],
                       uint3 tgid [[threadgroup_position_in_grid]])
{
  const int n0 = int(tgid.x) * TN;
  const int m0 = int(tgid.y) * TM;
  const int blk = n0 / BLOCK;
  const int col = n0 % BLOCK;

  constexpr auto desc = matmul2d_descriptor(TM, TN, GROUP, false, false, true,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;

  array<int, 2> a_str = {1, p.a_stride};
  array<int, 2> w_str = {1, BLOCK};
  tensor<device half, dextents<int, 2>, tensor_inline> A(a, dextents<int, 2>(p.K, p.M), a_str);
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> W(
      w + size_t(blk) * size_t(p.K) * (BLOCK / 2), dextents<int, 2>(BLOCK, p.K), w_str);

  auto tA = A.slice<GROUP, TM>(0, m0);
  auto tW = W.slice<TN, GROUP>(col, 0);
  auto acc = op.get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
  auto part = op.get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();

#pragma clang loop unroll(full)
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  }

  const device bfloat *sg = s + size_t(blk) * size_t(p.G) * BLOCK + col;
  const device bfloat *bg = b + size_t(blk) * size_t(p.G) * BLOCK + col;
  const device float *gs = gsum + size_t(m0) * p.G;

  for (int g = 0; g < p.G; ++g) {
    tA = A.slice<GROUP, TM>(g * GROUP, m0);
    tW = W.slice<TN, GROUP>(col, g * GROUP);
    op.run(tA, tW, part);

#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (acc.is_valid_element(i)) {
        auto idx = acc.get_multidimensional_index(i);  // {column, row} within the tile
        const float scale = float(sg[idx[0]]);
        const float bias = float(bg[idx[0]]);
        acc[i] += scale * part[i] + bias * gs[size_t(idx[1]) * p.G + g];
      }
    }
    sg += BLOCK;
    bg += BLOCK;
  }

  device OUT_T *out = c + size_t(m0) * p.c_stride + n0;
#pragma clang loop unroll(full)
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) {
      auto idx = acc.get_multidimensional_index(i);
      device OUT_T *dst = out + size_t(idx[1]) * p.c_stride + idx[0];
#if ACCUM
      *dst = OUT_T(float(*dst) + acc[i]);
#else
      *dst = OUT_T(acc[i]);
#endif
    }
  }
}

// ---------------------------------------------------------------------------
// Native format: C[M][N] = s[n] * (A[M][K] * L(q[N][K])^T)
//
// q is a 4-bit code the accelerator decodes itself (W_T: `metal_fp4_e2m1_format`
// or `int4b_format`), in the same blocked layout as above. There is no
// per-group scale, so the whole inner dimension is one hardware matmul with
// nothing done in software until the result is in registers. The scale per
// output row is applied there; the scale per input column is already folded
// into A by whichever kernel produced it.
//
// HALF_STEP = 1 selects the 16-level uniform codebook, whose levels sit at
// q + 0.5: the extra 0.5 * sum_k A[m][k] is supplied per row in `asum`.
#ifndef W_T
#define W_T int4b_format
#endif
#ifndef HALF_STEP
#define HALF_STEP 0
#endif

kernel void linear_q4n(device half *a [[buffer(0)]],
                       device uchar *w [[buffer(1)]],
                       device float *s [[buffer(2)]],
                       device float *asum [[buffer(3)]],
                       device OUT_T *c [[buffer(4)]],
                       constant LinearParams &p [[buffer(5)]],
                       uint3 tgid [[threadgroup_position_in_grid]])
{
  const int n0 = int(tgid.x) * TN;
  const int m0 = int(tgid.y) * TM;
  const int blk = n0 / BLOCK;
  const int col = n0 % BLOCK;

  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, false, true,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;

  array<int, 2> a_str = {1, p.a_stride};
  array<int, 2> w_str = {1, BLOCK};
  tensor<device half, dextents<int, 2>, tensor_inline> A(a, dextents<int, 2>(p.K, p.M), a_str);
  tensor<device W_T, dextents<int, 2>, tensor_inline> W(
      w + size_t(blk) * size_t(p.K) * (BLOCK / 2), dextents<int, 2>(BLOCK, p.K), w_str);

  auto tA = A.slice(0, m0);
  auto tW = W.slice(col, 0);
  auto acc = op.get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
  op.run(tA, tW, acc);

  const device float *sn = s + n0;
  device OUT_T *out = c + size_t(m0) * p.c_stride + n0;
#pragma clang loop unroll(full)
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) {
      auto idx = acc.get_multidimensional_index(i);
#if HALF_STEP
      const float v = sn[idx[0]] * (acc[i] + 0.5f * asum[m0 + idx[1]]);
#else
      const float v = sn[idx[0]] * acc[i];
#endif
      device OUT_T *dst = out + size_t(idx[1]) * p.c_stride + idx[0];
#if ACCUM
      *dst = OUT_T(float(*dst) + v);
#else
      *dst = OUT_T(v);
#endif
    }
  }
}

// Same product without the row scale, written straight to memory by the
// accelerator op: C[M][N] = A[M][K] * L(q[N][K])^T. Kept to measure what the
// in-register epilogue of linear_q4n costs.
kernel void linear_q4n_raw(device half *a [[buffer(0)]],
                           device uchar *w [[buffer(1)]],
                           device float *c [[buffer(4)]],
                           constant LinearParams &p [[buffer(5)]],
                           uint3 tgid [[threadgroup_position_in_grid]])
{
  const int n0 = int(tgid.x) * TN;
  const int m0 = int(tgid.y) * TM;
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, false, true,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  array<int, 2> a_str = {1, p.a_stride};
  array<int, 2> w_str = {1, BLOCK};
  array<int, 2> c_str = {1, p.c_stride};
  tensor<device half, dextents<int, 2>, tensor_inline> A(a, dextents<int, 2>(p.K, p.M), a_str);
  tensor<device W_T, dextents<int, 2>, tensor_inline> W(
      w + size_t(n0 / BLOCK) * size_t(p.K) * (BLOCK / 2), dextents<int, 2>(BLOCK, p.K), w_str);
  tensor<device float, dextents<int, 2>, tensor_inline> C(c, dextents<int, 2>(p.N, p.M), c_str);
  auto tA = A.slice(0, m0);
  auto tW = W.slice(n0 % BLOCK, 0);
  auto tC = C.slice(n0, m0);
  op.run(tA, tW, tC);
}
