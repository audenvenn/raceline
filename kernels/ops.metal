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

// The model's final norm followed by a second weighted RMS norm of the result,
// in one pass over the row (the hidden-state half of the MTP head's input):
//   t = x * w1 * rsqrt(mean(x^2) + eps);   out = t * w2 * rsqrt(mean(t^2) + eps)
kernel void rms_norm2_rows(device const float *x [[buffer(0)]],
                           device const float *w1 [[buffer(1)]],
                           device const float *w2 [[buffer(2)]],
                           device half *out [[buffer(3)]],
                           constant NormParams &p [[buffer(4)]],
                           uint3 tg [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]])
{
  const device float *row = x + size_t(tg.y) * p.in_stride;
  float a = 0.0f, b = 0.0f;
  for (int i = int(lane); i < p.D; i += 32) {
    const float v = row[i], u = v * w1[i];
    a += v * v;
    b += u * u;
  }
  a = simd_sum(a);
  b = simd_sum(b);
  const float inv1 = rsqrt(a / float(p.D) + p.eps);
  const float inv2 = rsqrt(b * inv1 * inv1 / float(p.D) + p.eps);
  device half *o = out + size_t(tg.y) * p.out_stride;
  for (int i = int(lane); i < p.D; i += 32) o[i] = half(row[i] * w1[i] * inv1 * inv2 * w2[i]);
}

// ----------------------------------------------------------------- rotation
// A model converted with --rotate (tools/quantize.py) has its residual stream
// rotated, x -> x Q with Q = diag(s) * blockdiag(H, ..., H): random signs s
// and normalized Sylvester-Hadamard blocks of ROT_BLOCK values. The rotation
// is folded into every weight of the model itself; only the draft head needs
// it at run time, where a norm sits between the residual stream and a weight.
// Applying Q or its inverse is a sign flip and a fast Walsh-Hadamard transform
// per block. One threadgroup holds one row of ROT_D values.
#if defined(ROT_D) && defined(ROT_BLOCK)
#define ROT_THREADS 256

// In-place, unnormalized Walsh-Hadamard transform of every block of x.
static void fwht_blocks(threadgroup float *x, uint tid)
{
  for (int len = 1; len < ROT_BLOCK; len <<= 1) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int q = int(tid); q < ROT_D / 2; q += ROT_THREADS) {
      const int blk = q / (ROT_BLOCK / 2), j = q % (ROT_BLOCK / 2);
      const int i = blk * ROT_BLOCK + (j / len) * 2 * len + (j % len);
      const float a = x[i], b = x[i + len];
      x[i] = a + b;
      x[i + len] = a - b;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Sum over the threadgroup, in a fixed order (the same bits every time).
static float tg_total(float v, threadgroup float *part, uint sg, uint lane)
{
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (int i = 0; i < ROT_THREADS / 32; ++i) t += part[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// The hidden-state half of the draft head's input, from rows h' of the
// rotated residual stream: undo the rotation, then the model's final norm and
// the head's own input norm (what rms_norm2_rows does for an unrotated model):
//   h = FWHT(h') * s;   z = wf * h / rms(h);   out = z * w2 / rms(z)
// w1 = wf * s / sqrt(ROT_BLOCK) carries the signs and the normalization.
kernel void mtp_hidden_rot(device const float *x [[buffer(0)]],
                           device const float *w1 [[buffer(1)]],
                           device const float *w2 [[buffer(2)]],
                           device half *out [[buffer(3)]],
                           constant RotNormParams &p [[buffer(4)]],
                           uint3 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float v[ROT_D];
  threadgroup float part[ROT_THREADS / 32];
  const device float *row = x + size_t(tg.y) * p.in_stride;
  float ss = 0.0f;
  for (int k = int(tid); k < ROT_D; k += ROT_THREADS) {
    const float a = row[k];
    v[k] = a;
    ss += a * a;
  }
  const float inv1 = rsqrt(tg_total(ss, part, sg, lane) / float(ROT_D) + p.eps);
  fwht_blocks(v, tid);
  float zz = 0.0f;
  for (int k = int(tid); k < ROT_D; k += ROT_THREADS) {
    const float z = v[k] * w1[k] * inv1;
    v[k] = z;
    zz += z * z;
  }
  const float inv2 = rsqrt(tg_total(zz, part, sg, lane) / float(ROT_D) + p.eps);
  device half *o = out + size_t(tg.y) * p.out_stride;
  for (int k = int(tid); k < ROT_D; k += ROT_THREADS) o[k] = half(v[k] * w2[k] * inv2);
}

// Rows given to the model from outside (an image's embeddings) into its
// rotated residual stream, in place: x -> x Q = FWHT(x * s) / sqrt(ROT_BLOCK)
// per block (s2 = s / sqrt(ROT_BLOCK)).
kernel void rotate_rows(device float *x [[buffer(0)]],
                        device const float *s2 [[buffer(1)]],
                        constant CopyParams &p [[buffer(2)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]])
{
  threadgroup float v[ROT_D];
  device float *row = x + size_t(tg.y) * p.D;
  for (int k = int(tid); k < ROT_D; k += ROT_THREADS) v[k] = row[k] * s2[k];
  fwht_blocks(v, tid);
  for (int k = int(tid); k < ROT_D; k += ROT_THREADS) row[k] = v[k];
}

// The draft head's output on its way into the model's output head, which
// expects the rotated stream with the final norm folded in:
//   out = FWHT((u / rms(u)) * g) * c,   g = (wm / wf) * s / sqrt(ROT_BLOCK)
// (wm: the head's own final norm, wf: the model's, c: the output head's
// column scales).
kernel void mtp_out_rot(device const float *x [[buffer(0)]],
                        device const float *g [[buffer(1)]],
                        device const float *c [[buffer(2)]],
                        device half *out [[buffer(3)]],
                        constant RotNormParams &p [[buffer(4)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float v[ROT_D];
  threadgroup float part[ROT_THREADS / 32];
  const device float *row = x + size_t(tg.y) * p.in_stride;
  float ss = 0.0f;
  for (int k = int(tid); k < ROT_D; k += ROT_THREADS) {
    const float a = row[k];
    v[k] = a * g[k];
    ss += a * a;
  }
  const float inv = rsqrt(tg_total(ss, part, sg, lane) / float(ROT_D) + p.eps);
  fwht_blocks(v, tid);
  device half *o = out + size_t(tg.y) * p.out_stride;
  for (int k = int(tid); k < ROT_D; k += ROT_THREADS) o[k] = half(v[k] * inv * c[k]);
}
#endif

// dst[t][i] = src[t][i] for T rows of D floats.
kernel void copy_rows(device const float *src [[buffer(0)]],
                      device float *dst [[buffer(1)]],
                      constant CopyParams &p [[buffer(2)]],
                      uint2 tid [[thread_position_in_grid]])
{
  if (int(tid.x) >= p.D || int(tid.y) >= p.T) return;
  const size_t i = size_t(tid.y) * p.D + tid.x;
  dst[i] = src[i];
}

// ---------------------------------------------------- Gated DeltaNet pieces
// Depthwise causal conv (4 taps) + SiLU over the q|k|v channels. Rows T..T+2
// of the grid write the next conv state (the last three pre-conv inputs).
// With p.spec set, T more rows copy the pre-conv inputs to `keep`, from which
// the state after any prefix of the tokens can be rebuilt (speculative rows
// that turn out wrong are taken back). With p.mark set, three more rows write
// the state as it is before row p.mark to `at_mark` (a snapshot inside the pass).
kernel void lin_conv(device const float *proj [[buffer(0)]],
                     device const float *st_in [[buffer(1)]],
                     device const float *cw [[buffer(2)]],
                     device float *y [[buffer(3)]],
                     device float *st_out [[buffer(4)]],
                     constant ConvParams &p [[buffer(5)]],
                     device float *keep [[buffer(6)]],
                     device float *at_mark [[buffer(7)]],
                     uint2 tid [[thread_position_in_grid]])
{
  const int c = int(tid.x), r = int(tid.y);
  const int base = p.T + 3 + p.T * p.spec;
  if (c >= p.C || r >= base + (p.mark > 0 ? 3 : 0)) return;
  if (r >= base) {
    const int j = r - base, tau = p.mark - 3 + j;
    at_mark[j * p.C + c] = tau >= 0 ? proj[size_t(tau) * p.in_stride + c] : st_in[(3 + tau) * p.C + c];
    return;
  }
  if (r >= p.T + 3) {
    const int j = r - p.T - 3;
    keep[j * p.C + c] = proj[size_t(j) * p.in_stride + c];
    return;
  }
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
// The state is read from S and written to S_out; they may be the same buffer.
kernel void gated_delta(device const float *qk [[buffer(0)]],
                        device const float *y [[buffer(1)]],
                        device const float *bg [[buffer(2)]],
                        device const float *S [[buffer(3)]],
                        device float *yo [[buffer(4)]],
                        constant DeltaParams &p [[buffer(5)]],
                        device float *S_out [[buffer(6)]],
                        device float *S_mark [[buffer(7)]],     // p.mark > 0: the state before that row
                        uint3 tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]])
{
  constexpr int E = 32;
  const int hv = int(tg.y);
  const int hk = hv / (p.Hv / p.Hk);
  const int row = int(tg.x) * 8 + int(lane >> 2);
  const int part = int(lane & 3);

  const size_t at = (size_t(hv) * p.Dv + row) * p.Dk + part * E;
  const device float *st = S + at;
  float state[E];
  for (int e = 0; e < E; ++e) state[e] = st[e];

  const device float *qp = qk + hk * p.Dk + part * E;
  const device float *kp = qk + (p.Hk + hk) * p.Dk + part * E;
  const device float *vp = y + p.v_off + hv * p.Dv + row;
  device float *op = yo + hv * p.Dv + row;

  for (int t = 0; t < p.T; ++t) {
    if (t == p.mark && t > 0) {
      device float *mk = S_mark + at;
      for (int e = 0; e < E; ++e) mk[e] = state[e];
    }
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
  device float *dst = S_out + at;
  for (int e = 0; e < E; ++e) dst[e] = state[e];
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
// Prepares one token's queries, keys and values (the attention itself is in
// attention.metal). One SIMD-group per (vector, token): vectors 0..H-1 are
// queries, the next KVH are keys, the last KVH are values. Queries and keys
// get a weighted RMS norm and rotary embedding on their first 64 dims. Lane l
// owns dims l, l+32, ..., so each rotary pair (i, i+32) sits in one lane.
// Requires D == 256 and 64 rotary dims.
//
// Queries go to q[kv head][t * G + j][D] (G = H / KVH query heads j per
// key/value head), keys and values into the cache at position pos0 + t: keys
// as [block of 256 positions][dim][position], values as [position][dim].
kernel void attn_prep(device const float *proj [[buffer(0)]],
                      device const float *q_w [[buffer(1)]],
                      device const float *k_w [[buffer(2)]],
                      device const float *rope [[buffer(3)]],
                      device half *q [[buffer(4)]],
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
    device half *dst = Vc + (size_t(kvh) * p.Lb * 256 + slot) * p.D;
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
    const int G = p.H / p.KVH;
    device half *dst = q + (size_t(v / G) * p.q_rows + size_t(t) * G + v % G) * p.D;
    for (int j = 0; j < J; ++j) dst[j * 32 + lane] = half(x[j]);
  } else {
    device half *dst = Kc + (size_t(v - p.H) * p.Lb + slot / 256) * p.D * 256 + slot % 256;
    for (int j = 0; j < J; ++j) dst[size_t(j * 32 + lane) * 256] = half(x[j]);
  }
}
