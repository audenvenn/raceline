// Kernel parameter blocks, shared verbatim between the Metal kernels and the
// C++ host. Plain ints and floats only, so the layout is identical on both sides.
#ifndef RL_PARAMS_H
#define RL_PARAMS_H

struct LinearParams {
  int M;         // rows of A to process (a multiple of the tile height)
  int K;         // input features (multiple of the quantization group)
  int N;         // output features, padded to a multiple of the block
  int a_stride;  // elements between rows of A
  int c_stride;  // elements between rows of C
  int G;         // K / group
};

struct EmbedParams { int T, K, G, h_stride; };
struct NormParams { int T, D, in_stride, out_stride; float eps; };
struct ConvParams { int T, C, in_stride, y_stride; };
struct LinPrepParams { int T, Hk, Dk, Hv, y_stride, qk_stride, proj_stride, off_b, off_a, bg_stride; };
struct DeltaParams { int T, Hk, Dk, Hv, Dv, qk_stride, y_stride, v_off, bg_stride, yo_stride; };
struct OutNormParams { int T, Hv, Dv, yo_stride, proj_stride, off_z, out_stride; float eps; };
struct SwigluParams { int T, I, in_stride, out_stride; };
struct AttnParams {
  int T, H, KVH, D;          // rows, query heads, key/value heads, head dim
  int proj_stride, off_k, off_v;
  int qn_stride, out_stride;
  int pos0, L;               // absolute position of row 0; capacity of the KV cache
  float eps, scale;
};

#endif
