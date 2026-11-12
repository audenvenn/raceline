// Mixture of experts: routing, gathering the routed rows, and combining the
// experts' outputs (the experts' matrices run in linear.metal: linear_q4n_moe).
//
// A MoE MLP sends each token to TOPK of E experts, chosen by the router's
// logits, plus one shared expert that every token uses. Experts are numbered
// 0 .. E-1 and the shared expert is number E, so each token makes S = TOPK + 1
// (token, expert, weight) assignments. The routing kernel sorts them by expert
// (a counting sort), so the rows each expert works on are contiguous; the
// grouped matmuls then run once per expert over just those rows, and the
// combine kernel adds each token's S results back into its residual stream in
// a fixed order. Nothing a token gets depends on the other tokens of the pass,
// so results stay bit-identical however tokens are chunked or drafted.
//
// Specialization macros: TOPK, E_MAX (array bound for E, a multiple of 32),
// TM (row tile of the grouped matmuls).

#include <metal_stdlib>

#include "params.h"

using namespace metal;

#define S (TOPK + 1)

// One threadgroup of 1024 threads for all T tokens of a pass.
//
//   logits [T][stride]  the router's E logits, then the shared expert's gate logit (at E)
//   tok    [T * S]      token of each sorted row
//   inv    [T * S]      sorted row of token t's slot-th assignment (t * S + slot)
//   w      [T * S]      that assignment's weight
//   sel    [T * S]      scratch: the expert of each assignment
//   items  [...]        work list for the grouped matmuls: (expert, first sorted row, rows, 0)
//                       per tile of TM rows
//   args   [2][3]       indirect dispatch sizes: (tiles_gu, items, 1), (tiles_down, items, 1)
//
// A token's slots 0 .. TOPK-1 hold its experts in falling order of logit (on
// equal logits the lower expert first); weights are the softmax over all E
// logits, renormalized over the chosen ones when norm_topk is set. Slot TOPK
// is the shared expert, weighted by sigmoid(gate logit).
kernel void moe_route(device const float *logits [[buffer(0)]],
                      device int *tok [[buffer(1)]],
                      device int *inv [[buffer(2)]],
                      device float *w [[buffer(3)]],
                      device int4 *items [[buffer(4)]],
                      device uint *args [[buffer(5)]],
                      constant MoeRouteParams &p [[buffer(6)]],
                      device int *sel [[buffer(7)]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]])
{
  threadgroup atomic_int cnt[E_MAX + 1];
  threadgroup int off[E_MAX + 2];
  constexpr int EPL = E_MAX / 32;  // logits per lane

  for (int e = int(tid); e <= p.E; e += 1024) atomic_store_explicit(&cnt[e], 0, memory_order_relaxed);

  // Top-k per token, one SIMD-group per token.
  for (int t = int(sg); t < p.T; t += 32) {
    const device float *row = logits + size_t(t) * p.stride;
    float v[EPL];
    for (int j = 0; j < EPL; j++) {
      const int e = int(lane) + 32 * j;
      v[j] = e < p.E ? row[e] : -INFINITY;
    }
    float m = -INFINITY;
    for (int j = 0; j < EPL; j++) m = max(m, v[j]);
    m = simd_max(m);
    float z = 0;                       // softmax denominator over all experts
    for (int j = 0; j < EPL; j++) z += (v[j] > -INFINITY) ? exp(v[j] - m) : 0.0f;
    z = simd_sum(z);

    float top[TOPK];
    int which[TOPK];
    for (int r = 0; r < TOPK; r++) {
      float bv = -INFINITY;
      int be = 1 << 30;
      for (int j = 0; j < EPL; j++) {
        const int e = int(lane) + 32 * j;
        if (v[j] > bv || (v[j] == bv && e < be)) {
          bv = v[j];
          be = e;
        }
      }
      const float mv = simd_max(bv);
      const int me = simd_min(bv == mv ? be : (1 << 30));
      top[r] = mv;
      which[r] = me;
      for (int j = 0; j < EPL; j++)
        if (int(lane) + 32 * j == me) v[j] = -INFINITY;
    }
    if (lane == 0) {
      float ex[TOPK], sum = 0;
      for (int r = 0; r < TOPK; r++) {
        ex[r] = exp(top[r] - m);
        sum += ex[r];
      }
      const float denom = p.norm_topk ? sum : z;
      for (int r = 0; r < TOPK; r++) {
        sel[t * S + r] = which[r];
        w[t * S + r] = ex[r] / denom;
      }
      sel[t * S + TOPK] = p.E;
      w[t * S + TOPK] = 1.0f / (1.0f + exp(-row[p.E]));
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  // Rows per expert.
  const int n = p.T * S;
  for (int a = int(tid); a < n; a += 1024) atomic_fetch_add_explicit(&cnt[sel[a]], 1, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Offsets, the work list and the dispatch sizes (serial: E + 1 is small).
  if (tid == 0) {
    off[0] = 0;
    int k = 0;
    for (int e = 0; e <= p.E; e++) {
      const int c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
      off[e + 1] = off[e] + c;
      for (int r = 0; r < c; r += TM) items[k++] = int4(e, off[e] + r, min(TM, c - r), 0);
      atomic_store_explicit(&cnt[e], off[e], memory_order_relaxed);   // now each expert's next free row
    }
    args[0] = uint(p.tiles_gu);
    args[1] = uint(k);
    args[2] = 1;
    args[3] = uint(p.tiles_down);
    args[4] = uint(k);
    args[5] = 1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Place every assignment. Rows within an expert come in no particular
  // order, which changes nothing: each row is computed on its own.
  for (int a = int(tid); a < n; a += 1024) {
    const int pos = atomic_fetch_add_explicit(&cnt[sel[a]], 1, memory_order_relaxed);
    tok[pos] = a / S;
    inv[a] = pos;
  }
}

// out[r] = x[tok[r]] for the R sorted rows; 8 halves per thread.
kernel void moe_gather(device const half *x [[buffer(0)]],
                       device const int *tok [[buffer(1)]],
                       device half *out [[buffer(2)]],
                       constant MoeGatherParams &p [[buffer(3)]],
                       uint2 tid [[thread_position_in_grid]])
{
  const int k = int(tid.x) * 8, r = int(tid.y);
  if (k >= p.K || r >= p.R) return;
  const device half4 *src = (const device half4 *)(x + size_t(tok[r]) * p.x_stride + k);
  device half4 *dst = (device half4 *)(out + size_t(r) * p.K + k);
  dst[0] = src[0];
  dst[1] = src[1];
}

// h[t] += sum over slots s of w[t * S + s] * y[inv[t * S + s]], the slots in
// order: every token's sum is formed the same way whatever else is in the pass.
kernel void moe_combine(device const float *y [[buffer(0)]],
                        device const int *inv [[buffer(1)]],
                        device const float *w [[buffer(2)]],
                        device float *h [[buffer(3)]],
                        constant MoeCombineParams &p [[buffer(4)]],
                        uint2 tid [[thread_position_in_grid]])
{
  const int d = int(tid.x), t = int(tid.y);
  if (d >= p.H || t >= p.T) return;
  float acc = 0;
  for (int s = 0; s < S; s++) {
    const int a = t * S + s;
    acc += w[a] * y[size_t(inv[a]) * p.y_stride + d];
  }
  h[size_t(t) * p.H + d] += acc;
}
