#include "engine.h"

#include "../kernels/params.h"

#import <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace rl {

struct Engine::LinearLayer {
  id<MTLBuffer> conv[2];  // ping-pong conv state, [3][conv_dim] float
  int cur = 0;
  id<MTLBuffer> S;        // DeltaNet state, [Hv][Dv][Dk] float
};

struct Engine::AttnLayer {
  id<MTLBuffer> K, V;     // [KVH][max_ctx][head_dim] half
};

namespace {

std::string lname(int layer, const char *suffix) { return "layers." + std::to_string(layer) + "." + suffix; }

template <typename P>
void set_params(id<MTLComputeCommandEncoder> enc, const P &p, int index) {
  [enc setBytes:&p length:sizeof(P) atIndex:index];
}

}  // namespace

Engine::Engine(const std::string &model_path, int max_ctx, int max_chunk)
    : model_(new ModelFile(mt_, model_path)), max_ctx_(max_ctx), max_chunk_(max_chunk) {
  const Config &c = config();
  if (c.lin_k_dim != 128 || c.head_dim != 256 || c.rope_dims() != 64 || c.conv_taps != 4 || c.group != 64 ||
      c.block != 256 || c.lin_v_dim % 8 != 0)
    fatal("unsupported model geometry (kernels assume Dk=128, head_dim=256, 64 rotary dims, 4 conv taps)");

  rows_cap_ = (int)Metal::round_up((size_t)max_chunk_, 128);
  x_stride_ = std::max({c.hidden, c.intermediate, c.lin_value_dim(), c.attn_dim()});
  proj_stride_ = 0;
  for (int i = 0; i < c.layers; i++) {
    proj_stride_ = std::max(proj_stride_, model_->q(lname(i, "mlp.gate_up")).n_pad);
    proj_stride_ = std::max(proj_stride_, model_->q(lname(i, c.is_linear_layer(i) ? "lin.in_proj" : "attn.qkv")).n_pad);
  }

  const size_t R = (size_t)rows_cap_;
  tok_ = mt_.alloc(R * sizeof(int32_t));
  h_ = mt_.alloc(R * c.hidden * sizeof(float));
  xn_ = mt_.alloc(R * x_stride_ * sizeof(_Float16));
  gsum_ = mt_.alloc(R * (x_stride_ / c.group) * sizeof(float));
  proj_ = mt_.alloc(R * proj_stride_ * sizeof(float));
  y_ = mt_.alloc(R * c.lin_conv_dim() * sizeof(float));
  qk_ = mt_.alloc(R * 2 * c.lin_key_dim() * sizeof(float));
  bg_ = mt_.alloc(R * 2 * c.lin_v_heads * sizeof(float));
  yo_ = mt_.alloc(R * c.lin_value_dim() * sizeof(float));
  qn_ = mt_.alloc(R * c.attn_dim() * sizeof(float));
  rope_ = mt_.alloc(R * 64 * sizeof(float));
  logits_ = mt_.alloc((size_t)16 * model_->q("lm_head").n_pad * sizeof(float));

  ones_ = mt_.alloc((size_t)x_stride_ * sizeof(float));
  std::fill_n((float *)ones_.contents, x_stride_, 1.0f);
  if (c.native) {
    if (c.book != "fp4" && c.book != "int15") fatal("codebook '%s' is not supported by this build", c.book.c_str());
    auto fold = [&](const FTensor &norm, const QTensor &consumer) {
      id<MTLBuffer> b = mt_.alloc((size_t)c.hidden * sizeof(float));
      const float *cs = (const float *)((const char *)consumer.buf.contents + consumer.c);
      float *o = (float *)b.contents;
      for (int k = 0; k < c.hidden; k++) o[k] = norm.data[k] * cs[k];
      return b;
    };
    for (int i = 0; i < c.layers; i++) {
      norm_in_.push_back(fold(model_->f(lname(i, "in_norm")),
                              model_->q(lname(i, c.is_linear_layer(i) ? "lin.in_proj" : "attn.qkv"))));
      norm_post_.push_back(fold(model_->f(lname(i, "post_norm")), model_->q(lname(i, "mlp.gate_up"))));
    }
    norm_final_ = fold(model_->f("final_norm"), model_->q("lm_head"));
  }

  lin_.resize(c.layers);
  attn_.resize(c.layers);
  for (int i = 0; i < c.layers; i++) {
    if (c.is_linear_layer(i)) {
      auto L = std::make_shared<LinearLayer>();
      L->conv[0] = mt_.alloc((size_t)3 * c.lin_conv_dim() * sizeof(float));
      L->conv[1] = mt_.alloc((size_t)3 * c.lin_conv_dim() * sizeof(float));
      L->S = mt_.alloc((size_t)c.lin_v_heads * c.lin_v_dim * c.lin_k_dim * sizeof(float));
      lin_[i] = L;
    } else {
      auto A = std::make_shared<AttnLayer>();
      size_t bytes = (size_t)c.kv_heads * max_ctx_ * c.head_dim * sizeof(_Float16);
      A->K = mt_.alloc(bytes);
      A->V = mt_.alloc(bytes);
      attn_[i] = A;
    }
  }
}

void Engine::reset() {
  pos_ = 0;
  for (auto &L : lin_) {
    if (!L) continue;
    memset(L->conv[0].contents, 0, L->conv[0].length);
    memset(L->conv[1].contents, 0, L->conv[1].length);
    memset(L->S.contents, 0, L->S.length);
    L->cur = 0;
  }
}

Engine::Tile Engine::tile_for(int rows) const {
  // From tests/test_linear --sweep and end-to-end runs on M6. Tiles taller
  // than 32 rows are ~3x slower in this kernel: the in-register scale step
  // does not scale up with the tile.
  if (rows <= 16) return {16, 64, 2};
  return {32, 64, 2};
}

void Engine::encode_linear(const QTensor &w, int rows, id<MTLBuffer> out, size_t out_offset, int out_stride,
                           bool accum, const char *what) {
  const Tile t = tile_for(rows);
  const int M = (rows + t.tm - 1) / t.tm * t.tm;

  if (w.native) {
    // One hardware matmul over the whole input, row scale applied in registers.
    Macros mac = {{"TM", std::to_string(t.tm)}, {"TN", std::to_string(t.tn)}, {"NSG", std::to_string(t.nsg)},
                  {"GROUP", "64"}, {"BLOCK", "256"}, {"OUT_T", "float"}, {"ACCUM", accum ? "1" : "0"},
                  {"W_T", config().book == "fp4" ? "metal_fp4_e2m1_format" : "int4b_format"}};
    LinearParams p{M, w.k, w.n_pad, x_stride_, out_stride, 0};
    [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_q4n", mac)];
    [enc_ setBuffer:xn_ offset:0 atIndex:0];
    [enc_ setBuffer:w.buf offset:w.w atIndex:1];
    [enc_ setBuffer:w.buf offset:w.s atIndex:2];
    [enc_ setBuffer:gsum_ offset:0 atIndex:3];  // only read by the half-step codebook
    [enc_ setBuffer:out offset:out_offset atIndex:4];
    set_params(enc_, p, 5);
    run_groups(w.n_pad / t.tn, M / t.tm, 32 * t.nsg);
    if (accum && repeating(what)) {  // see below: a repeat must not accumulate twice
      mac["ACCUM"] = "0";
      p.c_stride = proj_stride_;
      [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_q4n", mac)];
      [enc_ setBuffer:proj_ offset:0 atIndex:4];
      set_params(enc_, p, 5);
    }
    mark(what);
    return;
  }

  const Macros mac = {{"TM", std::to_string(t.tm)}, {"TN", std::to_string(t.tn)}, {"NSG", std::to_string(t.nsg)},
                      {"GROUP", "64"}, {"BLOCK", "256"}, {"OUT_T", "float"}, {"ACCUM", accum ? "1" : "0"}};
  const LinearParams p{M, w.k, w.n_pad, x_stride_, out_stride, w.k / 64};

  [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "group_sums", mac)];
  [enc_ setBuffer:xn_ offset:0 atIndex:0];
  [enc_ setBuffer:gsum_ offset:0 atIndex:1];
  set_params(enc_, p, 2);
  run_threads(p.G, M);
  mark("group_sums");

  [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_q4a", mac)];
  [enc_ setBuffer:xn_ offset:0 atIndex:0];
  [enc_ setBuffer:w.buf offset:w.w atIndex:1];
  [enc_ setBuffer:w.buf offset:w.s atIndex:2];
  [enc_ setBuffer:w.buf offset:w.b atIndex:3];
  [enc_ setBuffer:gsum_ offset:0 atIndex:4];
  [enc_ setBuffer:out offset:out_offset atIndex:5];
  set_params(enc_, p, 6);
  run_groups(w.n_pad / t.tn, M / t.tm, 32 * t.nsg);
  if (accum && repeating(what)) {
    // A repeat must not add into the residual stream twice: it reruns into scratch.
    Macros plain = mac;
    plain["ACCUM"] = "0";
    LinearParams p2 = p;
    p2.c_stride = proj_stride_;
    [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_q4a", plain)];
    [enc_ setBuffer:proj_ offset:0 atIndex:5];
    set_params(enc_, p2, 6);
  }
  mark(what);
}

void Engine::encode_norm(int which, int layer, int rows, size_t in_offset) {
  const Config &c = config();
  id<MTLBuffer> wgt;
  size_t wgt_offset = 0;
  if (c.native) {
    wgt = which == 0 ? norm_in_[layer] : which == 1 ? norm_post_[layer] : norm_final_;
  } else {
    const FTensor &f = which == 2 ? model_->f("final_norm") : model_->f(lname(layer, which == 0 ? "in_norm" : "post_norm"));
    wgt = f.buf;
    wgt_offset = f.offset;
  }
  // A decode step pays for per-kernel latency, so the row is spread over 16
  // SIMD-groups; a prefill chunk has enough rows to fill the GPU with one
  // SIMD-group each, and launching 16x the threads only costs time there.
  const int sg = rows <= 16 ? 16 : 1;
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "rms_norm_rows", {{"NORM_SG", std::to_string(sg)}})];
  [enc_ setBuffer:h_ offset:in_offset atIndex:0];
  [enc_ setBuffer:wgt offset:wgt_offset atIndex:1];
  [enc_ setBuffer:xn_ offset:0 atIndex:2];
  set_params(enc_, NormParams{rows, c.hidden, c.hidden, x_stride_, c.norm_eps}, 3);
  run_groups(1, rows, 32 * sg);
  mark("rms_norm");
}

void Engine::bind_column_scale(const QTensor &consumer, int index) {
  if (consumer.native) [enc_ setBuffer:consumer.buf offset:consumer.c atIndex:index];
  else [enc_ setBuffer:ones_ offset:0 atIndex:index];
}

void Engine::encode_mlp(int layer, int T) {
  const Config &c = config();
  encode_norm(1, layer, T, 0);
  encode_linear(model_->q(lname(layer, "mlp.gate_up")), T, proj_, 0, proj_stride_, false, "linear:gate_up");

  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "swiglu")];
  [enc_ setBuffer:proj_ offset:0 atIndex:0];
  [enc_ setBuffer:xn_ offset:0 atIndex:1];
  set_params(enc_, SwigluParams{T, c.intermediate, proj_stride_, x_stride_}, 2);
  bind_column_scale(model_->q(lname(layer, "mlp.down")), 3);
  run_threads(c.intermediate, T);
  mark("swiglu");

  encode_linear(model_->q(lname(layer, "mlp.down")), T, h_, 0, c.hidden, true, "linear:down");
}

void Engine::encode_linear_attn(int layer, int T) {
  const Config &c = config();
  LinearLayer &L = *lin_[layer];
  const QTensor &in_proj = model_->q(lname(layer, "lin.in_proj"));
  const int C = c.lin_conv_dim(), kd = c.lin_key_dim(), vd = c.lin_value_dim();
  const int Hk = c.lin_k_heads, Hv = c.lin_v_heads, Dk = c.lin_k_dim, Dv = c.lin_v_dim;

  encode_norm(0, layer, T, 0);
  encode_linear(in_proj, T, proj_, 0, proj_stride_, false, "linear:lin_in");

  const FTensor &cw = model_->f(lname(layer, "lin.conv"));
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "lin_conv")];
  [enc_ setBuffer:proj_ offset:0 atIndex:0];
  [enc_ setBuffer:L.conv[L.cur] offset:0 atIndex:1];
  [enc_ setBuffer:cw.buf offset:cw.offset atIndex:2];
  [enc_ setBuffer:y_ offset:0 atIndex:3];
  [enc_ setBuffer:L.conv[1 - L.cur] offset:0 atIndex:4];
  set_params(enc_, ConvParams{T, C, proj_stride_, C}, 5);
  run_threads(C, T + 3);
  mark("lin_conv");
  L.cur = 1 - L.cur;

  const LinPrepParams pp{T, Hk, Dk, Hv, C, 2 * kd, proj_stride_, in_proj.part_offset("b"),
                         in_proj.part_offset("a"), 2 * Hv};
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "lin_prep_qk")];
  [enc_ setBuffer:y_ offset:0 atIndex:0];
  [enc_ setBuffer:qk_ offset:0 atIndex:1];
  set_params(enc_, pp, 2);
  run_groups(2 * Hk, T);
  mark("lin_prep_qk");

  const FTensor &A_log = model_->f(lname(layer, "lin.A_log"));
  const FTensor &dt_bias = model_->f(lname(layer, "lin.dt_bias"));
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "lin_gates")];
  [enc_ setBuffer:proj_ offset:0 atIndex:0];
  [enc_ setBuffer:A_log.buf offset:A_log.offset atIndex:1];
  [enc_ setBuffer:dt_bias.buf offset:dt_bias.offset atIndex:2];
  [enc_ setBuffer:bg_ offset:0 atIndex:3];
  set_params(enc_, pp, 4);
  run_threads(Hv, T);
  mark("lin_gates");

  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "gated_delta")];
  [enc_ setBuffer:qk_ offset:0 atIndex:0];
  [enc_ setBuffer:y_ offset:0 atIndex:1];
  [enc_ setBuffer:bg_ offset:0 atIndex:2];
  [enc_ setBuffer:L.S offset:0 atIndex:3];
  [enc_ setBuffer:yo_ offset:0 atIndex:4];
  set_params(enc_, DeltaParams{T, Hk, Dk, Hv, Dv, 2 * kd, C, 2 * kd, 2 * Hv, vd}, 5);
  run_groups(Dv / 8, Hv);
  if (repeating("gated_delta")) {  // the recurrence updates its state in place: repeat on scratch
    if (!scratch_state_) {
      scratch_state_ = mt_.alloc(L.S.length);
      scratch_yo_ = mt_.alloc(yo_.length);
    }
    [enc_ setBuffer:scratch_state_ offset:0 atIndex:3];
    [enc_ setBuffer:scratch_yo_ offset:0 atIndex:4];
  }
  mark("gated_delta");

  const FTensor &nw = model_->f(lname(layer, "lin.norm"));
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "lin_out_norm")];
  [enc_ setBuffer:yo_ offset:0 atIndex:0];
  [enc_ setBuffer:proj_ offset:0 atIndex:1];
  [enc_ setBuffer:nw.buf offset:nw.offset atIndex:2];
  [enc_ setBuffer:xn_ offset:0 atIndex:3];
  set_params(enc_, OutNormParams{T, Hv, Dv, vd, proj_stride_, in_proj.part_offset("z"), x_stride_, c.norm_eps}, 4);
  bind_column_scale(model_->q(lname(layer, "lin.out_proj")), 5);
  run_groups(Hv, T);
  mark("lin_out_norm");

  encode_linear(model_->q(lname(layer, "lin.out_proj")), T, h_, 0, c.hidden, true, "linear:lin_out");
}

void Engine::encode_full_attn(int layer, int T) {
  const Config &c = config();
  AttnLayer &A = *attn_[layer];
  const QTensor &qkv = model_->q(lname(layer, "attn.qkv"));

  encode_norm(0, layer, T, 0);
  encode_linear(qkv, T, proj_, 0, proj_stride_, false, "linear:attn_qkv");

  const AttnParams ap{T, c.heads, c.kv_heads, c.head_dim, proj_stride_, qkv.part_offset("k"),
                      qkv.part_offset("v"), c.attn_dim(), x_stride_, pos_, max_ctx_, c.norm_eps,
                      1.0f / sqrtf((float)c.head_dim)};
  const FTensor &qw = model_->f(lname(layer, "attn.q_norm"));
  const FTensor &kw = model_->f(lname(layer, "attn.k_norm"));
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "attn_prep")];
  [enc_ setBuffer:proj_ offset:0 atIndex:0];
  [enc_ setBuffer:qw.buf offset:qw.offset atIndex:1];
  [enc_ setBuffer:kw.buf offset:kw.offset atIndex:2];
  [enc_ setBuffer:rope_ offset:0 atIndex:3];
  [enc_ setBuffer:qn_ offset:0 atIndex:4];
  [enc_ setBuffer:A.K offset:0 atIndex:5];
  [enc_ setBuffer:A.V offset:0 atIndex:6];
  set_params(enc_, ap, 7);
  run_groups(c.heads + 2 * c.kv_heads, T);
  mark("attn_prep");

  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "attn_sdpa")];
  [enc_ setBuffer:qn_ offset:0 atIndex:0];
  [enc_ setBuffer:A.K offset:0 atIndex:1];
  [enc_ setBuffer:A.V offset:0 atIndex:2];
  [enc_ setBuffer:proj_ offset:0 atIndex:3];
  [enc_ setBuffer:xn_ offset:0 atIndex:4];
  set_params(enc_, ap, 5);
  bind_column_scale(model_->q(lname(layer, "attn.o_proj")), 6);
  run_groups(c.heads, T);
  mark("attn_sdpa");

  encode_linear(model_->q(lname(layer, "attn.o_proj")), T, h_, 0, c.hidden, true, "linear:attn_out");
}

void Engine::run_threads(int w, int h) {
  last_ = {true, MTLSizeMake(w, h, 1), MTLSizeMake(32, 1, 1)};
  redo();
}

// `threads_per_group` threads (a whole number of 32-thread SIMD-groups) per grid cell.
void Engine::run_groups(int w, int h, int threads_per_group) {
  last_ = {false, MTLSizeMake(w, h, 1), MTLSizeMake(threads_per_group, 1, 1)};
  redo();
}

void Engine::redo() {
  if (last_.threads) [enc_ dispatchThreads:last_.grid threadsPerThreadgroup:last_.per_group];
  else [enc_ dispatchThreadgroups:last_.grid threadsPerThreadgroup:last_.per_group];
}

void Engine::mark(const char *what) {
  if (repeating(what))
    for (int i = 0; i < repeat_times_; i++) redo();
}

void Engine::begin() {
  cb_ = [mt_.queue() commandBuffer];
  enc_ = [cb_ computeCommandEncoder];
}

void Engine::flush(bool reopen) {
  [enc_ endEncoding];
  [cb_ commit];
  [cb_ waitUntilCompleted];
  if (cb_.status != MTLCommandBufferStatusCompleted)
    fatal("GPU command buffer failed: %s", cb_.error.localizedDescription.UTF8String);
  gpu_seconds_ += cb_.GPUEndTime - cb_.GPUStartTime;
  cb_ = nil;
  enc_ = nil;
  if (reopen) begin();
}

const float *Engine::forward(const int32_t *tokens, int T) {
  const Config &c = config();
  if (T < 1 || T > max_chunk_) fatal("forward: %d tokens, chunk limit is %d", T, max_chunk_);
  if (pos_ + T > max_ctx_) fatal("context window of %d tokens is full", max_ctx_);
  gpu_seconds_ = 0;

  memcpy(tok_.contents, tokens, (size_t)T * sizeof(int32_t));
  // Rotary angles in double precision: position * theta^(-i/32).
  float *rope = (float *)rope_.contents;
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < 32; i++) {
      const double ang = (double)(pos_ + t) * pow(c.rope_theta, -(double)i / 32.0);
      rope[t * 64 + i] = (float)cos(ang);
      rope[t * 64 + 32 + i] = (float)sin(ang);
    }
  }

  @autoreleasepool {
    begin();
    const QTensor &emb = model_->q("embed");
    if (emb.native) {
      const char *levels = c.book == "fp4" ? "0,0.5,1,1.5,2,3,4,6,-0.0,-0.5,-1,-1.5,-2,-3,-4,-6"
                                           : "0,1,2,3,4,5,6,7,-8,-7,-6,-5,-4,-3,-2,-1";
      [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "embed_rows_native", {{"BOOK_LEVELS", levels}})];
    } else {
      [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "embed_rows")];
    }
    [enc_ setBuffer:tok_ offset:0 atIndex:0];
    [enc_ setBuffer:emb.buf offset:emb.w atIndex:1];
    [enc_ setBuffer:emb.buf offset:emb.s atIndex:2];
    [enc_ setBuffer:emb.buf offset:(emb.native ? emb.c : emb.b) atIndex:3];
    [enc_ setBuffer:h_ offset:0 atIndex:4];
    set_params(enc_, EmbedParams{T, c.hidden, c.hidden / 64, c.hidden}, 5);
    run_threads(c.hidden / 8, T);
    mark("embed");

    for (int i = 0; i < c.layers; i++) {
      if (c.is_linear_layer(i)) encode_linear_attn(i, T);
      else encode_full_attn(i, T);
      encode_mlp(i, T);
      if (!dump_dir_.empty()) {
        flush(true);
        char path[1024];
        snprintf(path, sizeof path, "%s/layer_%02d.f32", dump_dir_.c_str(), i);
        FILE *f = fopen(path, "wb");
        if (!f) fatal("cannot write %s", path);
        fwrite(h_.contents, sizeof(float), (size_t)T * c.hidden, f);
        fclose(f);
      }
    }

    encode_norm(2, 0, 1, (size_t)(T - 1) * c.hidden * sizeof(float));
    const QTensor &head = model_->q("lm_head");
    encode_linear(head, 1, logits_, 0, head.n_pad, false, "linear:lm_head");
    flush(false);
  }

  pos_ += T;
  if (!dump_dir_.empty()) {
    FILE *f = fopen((dump_dir_ + "/logits.f32").c_str(), "wb");
    if (!f) fatal("cannot write logits dump");
    fwrite(logits_.contents, sizeof(float), (size_t)c.vocab, f);
    fclose(f);
  }
  return (const float *)logits_.contents;
}

int Engine::argmax() const {
  float best;
  vDSP_Length idx;
  vDSP_maxvi((const float *)logits_.contents, 1, &best, &idx, (vDSP_Length)config().vocab);
  return (int)idx;
}

}  // namespace rl
