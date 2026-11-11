// Everything in a Qwen3.5 / Qwen3.8 layer that is not a weight matmul:
// embedding lookup, RMS norms, the Gated DeltaNet recurrence, softmax
// attention and SwiGLU. These touch little memory compared with the weights;
// they are written for clarity first.

#include <metal_stdlib>
#include <metal_simdgroup>
#include "params.h"

using namespace metal;

static inline float silu(float x) { return x / (1.0f + exp(-x)); }
static inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }

// ---------------------------------------------------------------- embedding
// h[t][k] = scale * q + bias for row tok[t] of a row-major 4-bit matrix.
// One thread dequantizes 8 values (one packed word).
kernel void embed_rows(device const int *tok [[buffer(0)]],
                       device const uchar *w [[buffer(1)]],
                       device const bfloat *s [[buffer(2)]],
                       device const bfloat *b [[buffer(3)]],
                       device float *h [[buffer(4)]],
                       constant EmbedParams &p [[buffer(5)]],
                       uint2 tid [[thread_position_in_grid]])
{
  if (int(tid.x) >= p.K / 8 || int(tid.y) >= p.T) return;
  const size_t row = size_t(tok[tid.y]);
  const device uint *words = (const device uint *)(w + row * size_t(p.K / 2));
  const uint bits = words[tid.x];
  const int k0 = int(tid.x) * 8;
  const float sc = float(s[row * p.G + k0 / 64]);
  const float bi = float(b[row * p.G + k0 / 64]);
  device float *out = h + size_t(tid.y) * p.h_stride + k0;
  for (int i = 0; i < 8; ++i) out[i] = sc * float((bits >> (4 * i)) & 15u) + bi;
}

// Native format: h[t][k] = s[row] * c[k] * L[code]. BOOK_LEVELS is the codebook,
// sixteen comma-separated values in code order.
#ifdef BOOK_LEVELS
constant float kLevel[16] = {BOOK_LEVELS};
kernel void embed_rows_native(device const int *tok [[buffer(0)]],
                              device const uchar *w [[buffer(1)]],
                              device const float *s [[buffer(2)]],
                              device const float *c [[buffer(3)]],
                              device float *h [[buffer(4)]],
                              constant EmbedParams &p [[buffer(5)]],
                              uint2 tid [[thread_position_in_grid]])
{
  if (int(tid.x) >= p.K / 8 || int(tid.y) >= p.T) return;
  const size_t row = size_t(tok[tid.y]);
  const device uint *words = (const device uint *)(w + row * size_t(p.K / 2));
  const uint bits = words[tid.x];
  const int k0 = int(tid.x) * 8;
  const float sc = s[row];
  device float *out = h + size_t(tid.y) * p.h_stride + k0;
  for (int i = 0; i < 8; ++i) out[i] = sc * c[k0 + i] * kLevel[(bits >> (4 * i)) & 15u];
}
#endif

// ----------------------------------------------------------------- RMS norm
// out[t] = x[t] * rsqrt(mean(x[t]^2) + eps) * weight, one threadgroup per row.
//
// NORM_SG selects how many SIMD-groups share a row: 16 for a decode step,
// where the latency of walking the row serially is what costs, and 1 for a
// prefill chunk, where there are rows enough to fill the GPU and extra threads
// only add launch overhead. Both variants add the squares in the same order:
// a sum per group of 64 elements, accumulated into 16 running sums by
// (group mod 16), then those 16 added up. The result is therefore the same
// bits either way, which keeps the model's output independent of how the
// tokens were chunked.
#ifndef NORM_SG
#define NORM_SG 1
#endif
kernel void rms_norm_rows(device const float *x [[buffer(0)]],
                          device const float *wgt [[buffer(1)]],
                          device half *out [[buffer(2)]],
                          constant NormParams &p [[buffer(3)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]])
{
  constexpr int WAYS = 16;
  const device float *row = x + size_t(tg.y) * p.in_stride;
  const int groups = p.D / 64;
  float total = 0.0f;
#if NORM_SG == 1
  float part[WAYS];
  for (int i = 0; i < WAYS; ++i) part[i] = 0.0f;
  for (int g = 0; g < groups; ++g) {
    const float a = row[g * 64 + lane], b = row[g * 64 + 32 + lane];
    part[g % WAYS] += simd_sum(a * a + b * b);
  }
  for (int i = 0; i < WAYS; ++i) total += part[i];
#else
  threadgroup float part[WAYS];
  float acc = 0.0f;
  for (int g = int(sg); g < groups; g += WAYS) {
    const float a = row[g * 64 + lane], b = row[g * 64 + 32 + lane];
    acc += simd_sum(a * a + b * b);
  }
  if (lane == 0) part[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < WAYS; ++i) total += part[i];
#endif
  const float inv = rsqrt(total / float(p.D) + p.eps);
  device half *o = out + size_t(tg.y) * p.out_stride;
  for (int g = int(sg); g < groups; g += NORM_SG) {
    const int i0 = g * 64 + int(lane), i1 = i0 + 32;
    o[i0] = half(row[i0] * inv * wgt[i0]);
    o[i1] = half(row[i1] * inv * wgt[i1]);
  }
}

// ---------------------------------------------------- Gated DeltaNet pieces
// Depthwise causal conv (4 taps) + SiLU over the q|k|v channels. Rows T..T+2
// of the grid write the next conv state (the last three pre-conv inputs).
kernel void lin_conv(device const float *proj [[buffer(0)]],
                     device const float *st_in [[buffer(1)]],
                     device const float *cw [[buffer(2)]],
                     device float *y [[buffer(3)]],
                     device float *st_out [[buffer(4)]],
                     constant ConvParams &p [[buffer(5)]],
                     uint2 tid [[thread_position_in_grid]])
{
  const int c = int(tid.x), r = int(tid.y);
  if (c >= p.C || r >= p.T + 3) return;
  if (r < p.T) {
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
      const int tau = r - 3 + j;
      const float v = tau >= 0 ? proj[size_t(tau) * p.in_stride + c] : st_in[(3 + tau) * p.C + c];
      acc += cw[c * 4 + j] * v;
    }
    y[size_t(r) * p.y_stride + c] = silu(acc);
  } else {
    const int j = r - p.T, tau = p.T - 3 + j;
    st_out[j * p.C + c] = tau >= 0 ? proj[size_t(tau) * p.in_stride + c] : st_in[(3 + tau) * p.C + c];
  }
}

// L2-normalize every query and key head vector (Dk = 128, 4 values per lane);
// queries additionally carry the Dk^-0.5 readout scale.
kernel void lin_prep_qk(device const float *y [[buffer(0)]],
                        device float *qk [[buffer(1)]],
                        constant LinPrepParams &p [[buffer(2)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]])
{
  const int v = int(tg.x);
  const device float *src = y + size_t(tg.y) * p.y_stride + v * p.Dk;
  device float *dst = qk + size_t(tg.y) * p.qk_stride + v * p.Dk;
  float ss = 0.0f;
  for (int i = int(lane); i < p.Dk; i += 32) ss += src[i] * src[i];
  ss = simd_sum(ss);
  float sc = rsqrt(ss + 1e-6f);
  if (v < p.Hk) sc *= rsqrt(float(p.Dk));
  for (int i = int(lane); i < p.Dk; i += 32) dst[i] = src[i] * sc;
}

// beta = sigmoid(b);  g = exp(-exp(A_log) * softplus(a + dt_bias)).
kernel void lin_gates(device const float *proj [[buffer(0)]],
                      device const float *A_log [[buffer(1)]],
                      device const float *dt_bias [[buffer(2)]],
                      device float *bg [[buffer(3)]],
                      constant LinPrepParams &p [[buffer(4)]],
                      uint2 tid [[thread_position_in_grid]])
{
  const int hv = int(tid.x);
  if (hv >= p.Hv || int(tid.y) >= p.T) return;
  const device float *row = proj + size_t(tid.y) * p.proj_stride;
  const float x = row[p.off_a + hv] + dt_bias[hv];
  const float softplus = x > 20.0f ? x : log(1.0f + exp(x));
  device float *o = bg + size_t(tid.y) * p.bg_stride;
  o[hv] = sigmoidf(row[p.off_b + hv]);
  o[p.Hv + hv] = exp(-exp(A_log[hv]) * softplus);
}

// The recurrence itself, for all T tokens of the call. For each value head the
// state is a Dv x Dk matrix S, and per token
//     S <- g S;  d = beta (v - S k);  S <- S + d k^T;  y = S q.
// Rows of S are independent, so a SIMD-group carries 8 rows: 4 lanes per row,
// each lane holding 32 state values in registers for the whole token loop
// (the packing MLX's Gated DeltaNet kernel introduced). Requires Dk == 128.
kernel void gated_delta(device const float *qk [[buffer(0)]],
                        device const float *y [[buffer(1)]],
                        device const float *bg [[buffer(2)]],
                        device float *S [[buffer(3)]],
                        device float *yo [[buffer(4)]],
                        constant DeltaParams &p [[buffer(5)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]])
{
  constexpr int E = 32;
  const int hv = int(tg.y);
  const int hk = hv / (p.Hv / p.Hk);
  const int row = int(tg.x) * 8 + int(lane >> 2);
  const int part = int(lane & 3);

  device float *st = S + (size_t(hv) * p.Dv + row) * p.Dk + part * E;
  float state[E];
  for (int e = 0; e < E; ++e) state[e] = st[e];

  const device float *qp = qk + hk * p.Dk + part * E;
  const device float *kp = qk + (p.Hk + hk) * p.Dk + part * E;
  const device float *vp = y + p.v_off + hv * p.Dv + row;
  device float *op = yo + hv * p.Dv + row;

  for (int t = 0; t < p.T; ++t) {
    const float beta = bg[size_t(t) * p.bg_stride + hv];
    const float g = bg[size_t(t) * p.bg_stride + p.Hv + hv];
    const device float *k = kp + size_t(t) * p.qk_stride;
    const device float *q = qp + size_t(t) * p.qk_stride;

    float kv = 0.0f;
    for (int e = 0; e < E; ++e) {
      state[e] *= g;
      kv += state[e] * k[e];
    }
    kv += simd_shuffle_xor(kv, 1);
    kv += simd_shuffle_xor(kv, 2);

    const float delta = (vp[size_t(t) * p.y_stride] - kv) * beta;
    float out = 0.0f;
    for (int e = 0; e < E; ++e) {
      state[e] += k[e] * delta;
      out += state[e] * q[e];
    }
    out += simd_shuffle_xor(out, 1);
    out += simd_shuffle_xor(out, 2);
    if (part == 0) op[size_t(t) * p.yo_stride] = out;
  }
  for (int e = 0; e < E; ++e) st[e] = state[e];
}

// Per-head RMS norm of the recurrence output, gated by silu(z). `cs` is the
// column scale of the projection that consumes the result (all ones for
// weights that carry their scales themselves).
kernel void lin_out_norm(device const float *yo [[buffer(0)]],
                         device const float *proj [[buffer(1)]],
                         device const float *wgt [[buffer(2)]],
                         device half *out [[buffer(3)]],
                         constant OutNormParams &p [[buffer(4)]],
                         device const float *cs [[buffer(5)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]])
{
  const int hv = int(tg.x);
  cs += hv * p.Dv;
  const device float *src = yo + size_t(tg.y) * p.yo_stride + hv * p.Dv;
  const device float *z = proj + size_t(tg.y) * p.proj_stride + p.off_z + hv * p.Dv;
  device half *dst = out + size_t(tg.y) * p.out_stride + hv * p.Dv;
  float ss = 0.0f;
  for (int i = int(lane); i < p.Dv; i += 32) ss += src[i] * src[i];
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / float(p.Dv) + p.eps);
  for (int i = int(lane); i < p.Dv; i += 32) dst[i] = half(src[i] * inv * wgt[i] * silu(z[i]) * cs[i]);
}

// ------------------------------------------------------------------- SwiGLU
kernel void swiglu(device const float *proj [[buffer(0)]],
                   device half *out [[buffer(1)]],
                   constant SwigluParams &p [[buffer(2)]],
                   device const float *cs [[buffer(3)]],
                   uint2 tid [[thread_position_in_grid]])
{
  if (int(tid.x) >= p.I || int(tid.y) >= p.T) return;
  const device float *row = proj + size_t(tid.y) * p.in_stride;
  out[size_t(tid.y) * p.out_stride + tid.x] = half(silu(row[tid.x]) * row[p.I + tid.x] * cs[tid.x]);
}

// ---------------------------------------------------------------- attention
// One SIMD-group per (vector, token). Vectors 0..H-1 are queries, the next KVH
// are keys, the last KVH are values. Queries and keys get a weighted RMS norm
// and rotary embedding on their first 64 dims; keys and values are appended to
// the cache at position pos0 + t. Lane l owns dims l, l+32, ..., so each
// rotary pair (i, i+32) sits in one lane. Requires D == 256 and 64 rotary dims.
kernel void attn_prep(device const float *proj [[buffer(0)]],
                      device const float *q_w [[buffer(1)]],
                      device const float *k_w [[buffer(2)]],
                      device const float *rope [[buffer(3)]],
                      device float *qn [[buffer(4)]],
                      device half *Kc [[buffer(5)]],
                      device half *Vc [[buffer(6)]],
                      constant AttnParams &p [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]])
{
  constexpr int J = 8;  // D / 32
  const int v = int(tg.x), t = int(tg.y);
  const device float *row = proj + size_t(t) * p.proj_stride;
  const size_t slot = size_t(p.pos0 + t);

  if (v >= p.H + p.KVH) {  // value: plain copy into the cache
    const int kvh = v - p.H - p.KVH;
    const device float *src = row + p.off_v + kvh * p.D;
    device half *dst = Vc + (size_t(kvh) * p.L + slot) * p.D;
    for (int j = 0; j < J; ++j) dst[j * 32 + lane] = half(src[j * 32 + lane]);
    return;
  }

  const bool is_q = v < p.H;
  const device float *src = is_q ? row + v * 2 * p.D : row + p.off_k + (v - p.H) * p.D;
  const device float *wn = is_q ? q_w : k_w;
  float x[J];
  float ss = 0.0f;
  for (int j = 0; j < J; ++j) {
    x[j] = src[j * 32 + lane];
    ss += x[j] * x[j];
  }
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / float(p.D) + p.eps);
  for (int j = 0; j < J; ++j) x[j] *= inv * wn[j * 32 + lane];

  const float c = rope[size_t(t) * 64 + lane], s = rope[size_t(t) * 64 + 32 + lane];
  const float x0 = x[0] * c - x[1] * s;
  const float x1 = x[0] * s + x[1] * c;
  x[0] = x0;
  x[1] = x1;

  if (is_q) {
    device float *dst = qn + size_t(t) * p.qn_stride + v * p.D;
    for (int j = 0; j < J; ++j) dst[j * 32 + lane] = x[j];
  } else {
    device half *dst = Kc + (size_t(v - p.H) * p.L + slot) * p.D;
    for (int j = 0; j < J; ++j) dst[j * 32 + lane] = half(x[j]);
  }
}

// Causal softmax attention with a streaming (online) softmax, one SIMD-group
// per (query head, token), followed by the sigmoid output gate.
kernel void attn_sdpa(device const float *qn [[buffer(0)]],
                      device const half *Kc [[buffer(1)]],
                      device const half *Vc [[buffer(2)]],
                      device const float *proj [[buffer(3)]],
                      device half *out [[buffer(4)]],
                      constant AttnParams &p [[buffer(5)]],
                      device const float *cs [[buffer(6)]],
                      uint3 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]])
{
  constexpr int J = 8;
  const int hh = int(tg.x), t = int(tg.y);
  const int kvh = hh / (p.H / p.KVH);
  const device float *qrow = qn + size_t(t) * p.qn_stride + hh * p.D;
  float q[J];
  for (int j = 0; j < J; ++j) q[j] = qrow[j * 32 + lane];

  const device half *K = Kc + size_t(kvh) * p.L * p.D;
  const device half *V = Vc + size_t(kvh) * p.L * p.D;
  const int n_keys = p.pos0 + t + 1;
  float m = -INFINITY, l = 0.0f;
  float acc[J];
  for (int j = 0; j < J; ++j) acc[j] = 0.0f;
  for (int i = 0; i < n_keys; ++i) {
    const device half *kr = K + size_t(i) * p.D;
    float d = 0.0f;
    for (int j = 0; j < J; ++j) d += q[j] * float(kr[j * 32 + lane]);
    d = simd_sum(d) * p.scale;
    const float m_new = max(m, d);
    const float keep = exp(m - m_new), add = exp(d - m_new);
    l = l * keep + add;
    const device half *vr = V + size_t(i) * p.D;
    for (int j = 0; j < J; ++j) acc[j] = acc[j] * keep + add * float(vr[j * 32 + lane]);
    m = m_new;
  }

  const device float *gate = proj + size_t(t) * p.proj_stride + hh * 2 * p.D + p.D;
  device half *dst = out + size_t(t) * p.out_stride + hh * p.D;
  for (int j = 0; j < J; ++j) {
    const int e = j * 32 + int(lane);
    dst[e] = half(acc[j] / l * sigmoidf(gate[e]) * cs[hh * p.D + e]);
  }
}
