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
