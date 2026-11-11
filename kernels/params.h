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
struct ConvParams { int T, C, in_stride, y_stride, spec; };  // spec: also emit the state one row earlier
struct LinPrepParams { int T, Hk, Dk, Hv, y_stride, qk_stride, proj_stride, off_b, off_a, bg_stride; };
struct DeltaParams { int T, Hk, Dk, Hv, Dv, qk_stride, y_stride, v_off, bg_stride, yo_stride, spec; };
struct OutNormParams { int T, Hv, Dv, yo_stride, proj_stride, off_z, out_stride; float eps; };
struct SwigluParams { int T, I, in_stride, out_stride; };
struct CopyParams { int T, D; };
struct AttnParams {
  int T, H, KVH, D;          // rows, query heads, key/value heads, head dim
  int proj_stride, off_k, off_v;
  int q_rows;                // rows reserved per key/value head in the query buffer
  int out_stride;
  int pos0, Lb;              // absolute position of row 0; KV cache capacity in blocks of 256 keys
  float eps, scale;
  // One group of rows in the matmul kernels (kernels/attention.metal); pos0 is then the group's first row.
  int M;                     // query rows per key/value head: rows * (H / KVH)
  int kv_rows;               // M padded to the tile height: rows between key/value heads in the score buffers
  int n_pad, n_stride;       // keys covered (a multiple of 256); elements between rows of the score buffers
};

#endif
