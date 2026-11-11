#include "engine.h"

#include "../kernels/params.h"

#import <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace rl {

struct Engine::LinearLayer {
  id<MTLBuffer> conv[3];  // conv state, [3][conv_dim] float: current, next, and (speculation) one token back
  int cur = 0, prev = 0;
  id<MTLBuffer> S;        // DeltaNet state, [Hv][Dv][Dk] float
  id<MTLBuffer> S_prev;   // the state before a speculative last token
};

namespace {
struct MtpState {  // the head's bookkeeping, as saved by save_state
  int32_t mtp_pos, n_tok, tok[4], draft_next, draft_rows;
};
constexpr int kMtpRows = 4;  // hidden states that can wait for their next token
}  // namespace

struct Engine::AttnLayer {
  id<MTLBuffer> K;        // [KVH][blocks of 256 positions][head_dim][256] half
  id<MTLBuffer> V;        // [KVH][positions][head_dim] half
};

namespace {

template <typename P>
void set_params(id<MTLComputeCommandEncoder> enc, const P &p, int index) {
  [enc setBytes:&p length:sizeof(P) atIndex:index];
}

}  // namespace

Engine::Engine(const std::string &model_path, int max_ctx, int max_chunk)
    : model_(new ModelFile(mt_, model_path)), max_ctx_(max_ctx), max_chunk_(max_chunk) {
  const Config &c = config();
  if (c.lin_k_dim != 128 || c.head_dim != 256 || c.rope_dims() != 64 || c.conv_taps != 4 || c.group != 64 ||
      c.block != 256 || c.lin_v_dim % 8 != 0 || c.heads % c.kv_heads != 0)
    fatal("unsupported model geometry (kernels assume Dk=128, head_dim=256, 64 rotary dims, 4 conv taps)");

  has_mtp_ = mtp_ = c.native && model_->has("mtp.fc");
  const int n_layers = c.layers + (has_mtp_ ? 1 : 0);  // the head's decoder layer is layer c.layers

  // A head pass can carry up to two parked rows on top of a chunk.
  rows_cap_ = (int)Metal::round_up((size_t)max_chunk_ + (has_mtp_ ? 2 : 0), 128);
  x_stride_ = std::max({c.hidden, c.intermediate, c.lin_value_dim(), c.attn_dim(), has_mtp_ ? 2 * c.hidden : 0});
  proj_stride_ = 0;
  for (int i = 0; i < n_layers; i++) {
    proj_stride_ = std::max(proj_stride_, model_->q(lname(i, "mlp.gate_up")).n_pad);
    proj_stride_ = std::max(proj_stride_, model_->q(lname(i, is_linear(i) ? "lin.in_proj" : "attn.qkv")).n_pad);
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
  const int G = c.heads / c.kv_heads;
  kv_blocks_ = (max_ctx_ + 255) / 256;
  q_rows_ = rows_cap_ * G + 32;
  aq_ = mt_.alloc((size_t)c.kv_heads * q_rows_ * c.head_dim * sizeof(_Float16));
  al_ = mt_.alloc((size_t)c.kv_heads * q_rows_ * sizeof(float));
  // Scores for a group of rows: [KVH][rows * G][keys]. A decode step touches a
  // sliver of this; a prefill chunk at long context is cut into groups that fit.
  score_cap_ = std::max((size_t)24 << 20, (size_t)c.kv_heads * 64 * kv_blocks_ * 256);
  as_ = mt_.alloc(score_cap_ * sizeof(float));
  ap_ = mt_.alloc(score_cap_ * sizeof(_Float16));
  rope_ = mt_.alloc(R * 64 * sizeof(float));
  logits_ = mt_.alloc((size_t)16 * model_->q("lm_head").n_pad * sizeof(float));

  ones_ = mt_.alloc((size_t)x_stride_ * sizeof(float));
  std::fill_n((float *)ones_.contents, x_stride_, 1.0f);
  if (c.native) {
    if (c.book != "fp4" && c.book != "int15") fatal("codebook '%s' is not supported by this build", c.book.c_str());
    // Norm weights times the column scales (from column col0 on) of the matrix they feed.
    auto fold = [&](const FTensor &norm, const QTensor &consumer, int col0 = 0) {
      id<MTLBuffer> b = mt_.alloc((size_t)c.hidden * sizeof(float));
      const float *cs = consumer.native ? (const float *)((const char *)consumer.buf.contents + consumer.c) + col0 : nullptr;
      float *o = (float *)b.contents;
      for (int k = 0; k < c.hidden; k++) o[k] = norm.data[k] * (cs ? cs[k] : 1.0f);
      return b;
    };
    for (int i = 0; i < n_layers; i++) {
      norm_in_.push_back(fold(model_->f(lname(i, "in_norm")),
                              model_->q(lname(i, is_linear(i) ? "lin.in_proj" : "attn.qkv"))));
      norm_post_.push_back(fold(model_->f(lname(i, "post_norm")), model_->q(lname(i, "mlp.gate_up"))));
    }
    norm_final_ = fold(model_->f("final_norm"), model_->q("lm_head"));
    if (has_mtp_) {
      mtp_norm_e_ = fold(model_->f("mtp.norm_e"), model_->q("mtp.fc"), 0);
      mtp_norm_h_ = fold(model_->f("mtp.norm_h"), model_->q("mtp.fc"), c.hidden);
      mtp_norm_out_ = fold(model_->f("mtp.norm"), model_->q("lm_head"));
      mtp_h_ = mt_.alloc((size_t)kMtpRows * c.hidden * sizeof(float));
      mtp_tok_ = mt_.alloc(R * sizeof(int32_t));
      mtp_rope_ = mt_.alloc(R * 64 * sizeof(float));
      mtp_logits_ = mt_.alloc((size_t)16 * model_->q("lm_head").n_pad * sizeof(float));
    }
  }

  lin_.resize(n_layers);
  attn_.resize(n_layers);
  for (int i = 0; i < n_layers; i++) {
    if (is_linear(i)) {
      auto L = std::make_shared<LinearLayer>();
      for (auto &b : L->conv) b = mt_.alloc((size_t)3 * c.lin_conv_dim() * sizeof(float));
      L->S = mt_.alloc((size_t)c.lin_v_heads * c.lin_v_dim * c.lin_k_dim * sizeof(float));
      L->S_prev = has_mtp_ ? mt_.alloc(L->S.length) : L->S;  // only written by speculative passes
      lin_[i] = L;
    } else {
      auto A = std::make_shared<AttnLayer>();
      size_t bytes = (size_t)c.kv_heads * kv_blocks_ * 256 * c.head_dim * sizeof(_Float16);
      A->K = mt_.alloc(bytes);
      A->V = mt_.alloc(bytes);
      attn_[i] = A;
    }
  }
}

std::string Engine::lname(int layer, const char *suffix) const {
  if (layer == config().layers) return std::string("mtp.") + suffix;
  return "layers." + std::to_string(layer) + "." + suffix;
}

void Engine::reset() {
  pos_ = 0;
  for (auto &L : lin_) {
    if (!L) continue;
    for (auto &b : L->conv) memset(b.contents, 0, b.length);
    memset(L->S.contents, 0, L->S.length);
    L->cur = L->prev = 0;
  }
  mtp_pos_ = 0;
  draft_next_ = -1;
  draft_rows_ = 0;
  pend_tok_.clear();
  spec_open_ = false;
}

void Engine::set_mtp(bool on) {
  if (on && !has_mtp_) fatal("set_mtp: this model file has no MTP head (tools/add_mtp.py adds one)");
  if (pos_ != 0) fatal("set_mtp: only at position 0");
  mtp_ = on;
  mtp_pos_ = 0;
  draft_next_ = -1;
  draft_rows_ = 0;
  pend_tok_.clear();
}

size_t Engine::state_bytes() const {
  size_t n = has_mtp_ ? sizeof(MtpState) + mtp_h_.length : 0;
  for (const auto &L : lin_)
    if (L) n += L->conv[0].length + L->S.length;
  return n;
}

void Engine::save_state(void *dst) const {
  char *p = (char *)dst;
  for (const auto &L : lin_) {
    if (!L) continue;
    memcpy(p, L->conv[L->cur].contents, L->conv[0].length);
    p += L->conv[0].length;
    memcpy(p, L->S.contents, L->S.length);
    p += L->S.length;
  }
  if (has_mtp_) {
    MtpState st{};
    st.mtp_pos = mtp_pos_;
    st.n_tok = (int32_t)pend_tok_.size();
    for (size_t i = 0; i < pend_tok_.size() && i < 4; i++) st.tok[i] = pend_tok_[i];
    st.draft_next = draft_next_;
    st.draft_rows = draft_rows_;
    memcpy(p, &st, sizeof st);
    memcpy(p + sizeof st, mtp_h_.contents, mtp_h_.length);
  }
}

void Engine::load_state(const void *src, int position) {
  if (position < 0 || position > max_ctx_) fatal("load_state: position %d out of range", position);
  const char *p = (const char *)src;
  for (auto &L : lin_) {
    if (!L) continue;
    memcpy(L->conv[L->cur].contents, p, L->conv[0].length);
    p += L->conv[0].length;
    memcpy(L->S.contents, p, L->S.length);
    p += L->S.length;
  }
  if (has_mtp_) {
    MtpState st;
    memcpy(&st, p, sizeof st);
    memcpy(mtp_h_.contents, p + sizeof st, mtp_h_.length);
    mtp_pos_ = mtp_ ? st.mtp_pos : 0;
    pend_tok_.assign(st.tok, st.tok + st.n_tok);
    draft_next_ = st.draft_next;
    draft_rows_ = st.draft_rows;
  }
  pos_ = position;
  spec_open_ = false;
}

void Engine::reject_draft() {
  if (!spec_open_) fatal("reject_draft: the last call was not forward_draft");
  spec_open_ = false;
  for (auto &L : lin_) {
    if (!L) continue;
    L->cur = L->prev;
    std::swap(L->S, L->S_prev);
  }
  pos_ -= 1;         // the attention entries of the draft row are overwritten by the next token
  pend_tok_.clear();  // one hidden state is waiting again, its next token unknown
}

const float *Engine::logits(int row) const {
  return (const float *)logits_.contents + (size_t)row * model_->q("lm_head").n_pad;
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

  if (w.dense) {
    const Macros mac = {{"TM", std::to_string(t.tm)}, {"TN", std::to_string(t.tn)}, {"NSG", std::to_string(t.nsg)},
                        {"GROUP", "64"}, {"BLOCK", "256"}, {"OUT_T", "float"}, {"ACCUM", accum ? "1" : "0"}};
    [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_f16", mac)];
    [enc_ setBuffer:xn_ offset:0 atIndex:0];
    [enc_ setBuffer:w.buf offset:w.w atIndex:1];
    [enc_ setBuffer:out offset:out_offset atIndex:4];
    set_params(enc_, LinearParams{M, w.k, w.n_pad, x_stride_, out_stride, 0}, 5);
    run_groups(w.n_pad / t.tn, M / t.tm, 32 * t.nsg);
    mark(what);
    return;
  }

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
  if (c.native) {
    encode_rms(h_, in_offset, c.hidden, which == 0 ? norm_in_[layer] : which == 1 ? norm_post_[layer] : norm_final_, 0,
               rows, 0);
  } else {
    const FTensor &f = which == 2 ? model_->f("final_norm") : model_->f(lname(layer, which == 0 ? "in_norm" : "post_norm"));
    encode_rms(h_, in_offset, c.hidden, f.buf, f.offset, rows, 0);
  }
}

// Weighted RMS norm of `rows` rows of hidden floats into xn_ (from byte out_offset on).
void Engine::encode_rms(id<MTLBuffer> in, size_t in_offset, int in_stride, id<MTLBuffer> wgt, size_t wgt_offset,
                        int rows, size_t out_offset) {
  const Config &c = config();
  // A decode step pays for per-kernel latency, so the row is spread over 16
  // SIMD-groups; a prefill chunk has enough rows to fill the GPU with one
  // SIMD-group each, and launching 16x the threads only costs time there.
  const int sg = rows <= 16 ? 16 : 1;
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "rms_norm_rows", {{"NORM_SG", std::to_string(sg)}})];
  [enc_ setBuffer:in offset:in_offset atIndex:0];
  [enc_ setBuffer:wgt offset:wgt_offset atIndex:1];
  [enc_ setBuffer:xn_ offset:out_offset atIndex:2];
  set_params(enc_, NormParams{rows, c.hidden, in_stride, x_stride_, c.norm_eps}, 3);
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

void Engine::encode_linear_attn(int layer, int T, bool spec) {
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
  [enc_ setBuffer:L.conv[(L.cur + 1) % 3] offset:0 atIndex:4];
  [enc_ setBuffer:L.conv[(L.cur + 2) % 3] offset:0 atIndex:6];
  set_params(enc_, ConvParams{T, C, proj_stride_, C, spec ? 1 : 0}, 5);
  run_threads(C, T + 3 + (spec ? 3 : 0));
  mark("lin_conv");
  L.prev = (L.cur + 2) % 3;
  L.cur = (L.cur + 1) % 3;

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
  [enc_ setBuffer:L.S_prev offset:0 atIndex:6];
  set_params(enc_, DeltaParams{T, Hk, Dk, Hv, Dv, 2 * kd, C, 2 * kd, 2 * Hv, vd, spec ? 1 : 0}, 5);
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

void Engine::encode_full_attn(int layer, int T, int pos0, id<MTLBuffer> rope) {
  const Config &c = config();
  AttnLayer &A = *attn_[layer];
  const QTensor &qkv = model_->q(lname(layer, "attn.qkv"));
  const int G = c.heads / c.kv_heads, D = c.head_dim;

  encode_norm(0, layer, T, 0);
  encode_linear(qkv, T, proj_, 0, proj_stride_, false, "linear:attn_qkv");

  AttnParams ap{};
  ap.T = T, ap.H = c.heads, ap.KVH = c.kv_heads, ap.D = D;
  ap.proj_stride = proj_stride_, ap.off_k = qkv.part_offset("k"), ap.off_v = qkv.part_offset("v");
  ap.q_rows = q_rows_, ap.out_stride = x_stride_, ap.pos0 = pos0, ap.Lb = kv_blocks_;
  ap.eps = c.norm_eps, ap.scale = 1.0f / sqrtf((float)D);
  const FTensor &qw = model_->f(lname(layer, "attn.q_norm"));
  const FTensor &kw = model_->f(lname(layer, "attn.k_norm"));
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "attn_prep")];
  [enc_ setBuffer:proj_ offset:0 atIndex:0];
  [enc_ setBuffer:qw.buf offset:qw.offset atIndex:1];
  [enc_ setBuffer:kw.buf offset:kw.offset atIndex:2];
  [enc_ setBuffer:rope offset:0 atIndex:3];
  [enc_ setBuffer:aq_ offset:0 atIndex:4];
  [enc_ setBuffer:A.K offset:0 atIndex:5];
  [enc_ setBuffer:A.V offset:0 atIndex:6];
  set_params(enc_, ap, 7);
  run_groups(c.heads + 2 * c.kv_heads, T);
  mark("attn_prep");

  // Rows are taken in groups whose scores fit the scratch buffers: all of
  // them in one group unless the context is long.
  for (int r0 = 0; r0 < T;) {
    int R = T - r0, kv_rows = 0, n_pad = 0;
    Tile t{};
    for (;; R = (R + 1) / 2) {
      t = tile_for(R * G);
      kv_rows = (R * G + t.tm - 1) / t.tm * t.tm;
      n_pad = (pos0 + r0 + R + 255) / 256 * 256;
      if ((size_t)c.kv_heads * kv_rows * n_pad <= score_cap_ || R == 1) break;
    }
    ap.pos0 = pos0 + r0;
    ap.M = R * G, ap.kv_rows = kv_rows, ap.n_pad = n_pad, ap.n_stride = n_pad;
    const Macros mac = {{"TM", std::to_string(t.tm)}, {"TN", std::to_string(t.tn)}, {"NSG", std::to_string(t.nsg)}};

    [enc_ setComputePipelineState:mt_.pipeline("attention.metal", "attn_scores", mac)];
    [enc_ setBuffer:aq_ offset:(size_t)r0 * G * D * sizeof(_Float16) atIndex:0];
    [enc_ setBuffer:A.K offset:0 atIndex:1];
    [enc_ setBuffer:as_ offset:0 atIndex:2];
    set_params(enc_, ap, 3);
    run_groups(n_pad / t.tn, kv_rows / t.tm, 32 * t.nsg, c.kv_heads);
    mark("attn_scores");

    [enc_ setComputePipelineState:mt_.pipeline("attention.metal", "attn_softmax", mac)];
    [enc_ setBuffer:as_ offset:0 atIndex:0];
    [enc_ setBuffer:ap_ offset:0 atIndex:1];
    [enc_ setBuffer:al_ offset:0 atIndex:2];
    set_params(enc_, ap, 3);
    run_groups(1, R * G, 32, c.kv_heads);
    mark("attn_softmax");

    [enc_ setComputePipelineState:mt_.pipeline("attention.metal", "attn_pv", mac)];
    [enc_ setBuffer:ap_ offset:0 atIndex:0];
    [enc_ setBuffer:A.V offset:0 atIndex:1];
    [enc_ setBuffer:al_ offset:0 atIndex:2];
    [enc_ setBuffer:proj_ offset:(size_t)r0 * proj_stride_ * sizeof(float) atIndex:3];
    [enc_ setBuffer:xn_ offset:(size_t)r0 * x_stride_ * sizeof(_Float16) atIndex:4];
    set_params(enc_, ap, 5);
    bind_column_scale(model_->q(lname(layer, "attn.o_proj")), 6);
    run_groups(D / t.tn, kv_rows / t.tm, 32 * t.nsg, c.kv_heads);
    mark("attn_pv");
    r0 += R;
  }

  encode_linear(model_->q(lname(layer, "attn.o_proj")), T, h_, 0, c.hidden, true, "linear:attn_out");
}

void Engine::run_threads(int w, int h) {
  last_ = {true, MTLSizeMake(w, h, 1), MTLSizeMake(32, 1, 1)};
  redo();
}

// `threads_per_group` threads (a whole number of 32-thread SIMD-groups) per grid cell.
void Engine::run_groups(int w, int h, int threads_per_group, int depth) {
  last_ = {false, MTLSizeMake(w, h, depth), MTLSizeMake(threads_per_group, 1, 1)};
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

// Rotary angles in double precision: position * theta^(-i/32).
void Engine::fill_rope(id<MTLBuffer> buf, int pos0, int T) {
  float *rope = (float *)buf.contents;
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < 32; i++) {
      const double ang = (double)(pos0 + t) * pow(config().rope_theta, -(double)i / 32.0);
      rope[t * 64 + i] = (float)cos(ang);
      rope[t * 64 + 32 + i] = (float)sin(ang);
    }
  }
}

// h_[t] = row tok[t] of the token table.
void Engine::encode_embed(id<MTLBuffer> tok, int T) {
  const Config &c = config();
  const QTensor &emb = model_->q("embed");
  if (emb.native) {
    const char *levels = c.book == "fp4" ? "0,0.5,1,1.5,2,3,4,6,-0.0,-0.5,-1,-1.5,-2,-3,-4,-6"
                                         : "0,1,2,3,4,5,6,7,-8,-7,-6,-5,-4,-3,-2,-1";
    [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "embed_rows_native", {{"BOOK_LEVELS", levels}})];
  } else {
    [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "embed_rows")];
  }
  [enc_ setBuffer:tok offset:0 atIndex:0];
  [enc_ setBuffer:emb.buf offset:emb.w atIndex:1];
  [enc_ setBuffer:emb.buf offset:emb.s atIndex:2];
  [enc_ setBuffer:emb.buf offset:(emb.native ? emb.c : emb.b) atIndex:3];
  [enc_ setBuffer:h_ offset:0 atIndex:4];
  set_params(enc_, EmbedParams{T, c.hidden, c.hidden / 64, c.hidden}, 5);
  run_threads(c.hidden / 8, T);
  mark("embed");
}

void Engine::encode_copy_hidden(int src_row, int dst_row, int rows) {
  const size_t row_bytes = (size_t)config().hidden * sizeof(float);
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "copy_rows")];
  [enc_ setBuffer:h_ offset:src_row * row_bytes atIndex:0];
  [enc_ setBuffer:mtp_h_ offset:dst_row * row_bytes atIndex:1];
  set_params(enc_, CopyParams{rows, config().hidden}, 2);
  run_threads(config().hidden, rows);
  mark("copy_rows");
}

void Engine::encode_mtp(int n_carry, int n_chunk, int keep_row, bool logits) {
  const Config &c = config();
  const int H = c.hidden, M = n_carry + n_chunk;
  const FTensor &fn = model_->f("final_norm");
  // Hidden half of the head's input: rows of `src` -> xn_[first_row ..][H .. 2H).
  auto hidden_half = [&](id<MTLBuffer> src, int rows, int first_row) {
    if (rows <= 0) return;
    [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "rms_norm2_rows")];
    [enc_ setBuffer:src offset:0 atIndex:0];
    [enc_ setBuffer:fn.buf offset:fn.offset atIndex:1];
    [enc_ setBuffer:mtp_norm_h_ offset:0 atIndex:2];
    [enc_ setBuffer:xn_ offset:((size_t)first_row * x_stride_ + H) * sizeof(_Float16) atIndex:3];
    set_params(enc_, NormParams{rows, H, H, x_stride_, c.norm_eps}, 4);
    run_groups(1, rows);
    mark("mtp_norm");
  };
  hidden_half(mtp_h_, n_carry, 0);
  hidden_half(h_, n_chunk, n_carry);
  if (keep_row >= 0) encode_copy_hidden(keep_row, 0, 1);
  if (M == 0) return;

  // From here on h_ is the head's own residual stream.
  encode_embed(mtp_tok_, M);
  encode_rms(h_, 0, H, mtp_norm_e_, 0, M, 0);
  encode_linear(model_->q("mtp.fc"), M, h_, 0, H, false, "linear:mtp_fc");
  encode_full_attn(c.layers, M, mtp_pos_, mtp_rope_);
  encode_mlp(c.layers, M);
  if (logits) {
    encode_rms(h_, (size_t)(M - 1) * H * sizeof(float), H, mtp_norm_out_, 0, 1, 0);
    const QTensor &head = model_->q("lm_head");
    encode_linear(head, 1, mtp_logits_, 0, head.n_pad, false, "linear:lm_head");
  }
}

const float *Engine::forward(const int32_t *tokens, int T) { return run(tokens, T, false); }

const float *Engine::forward_draft(int32_t token, int32_t draft) {
  if (!mtp_) fatal("forward_draft: the model has no MTP head, or it is off");
  const int32_t two[2] = {token, draft};
  return run(two, 2, true);
}

const float *Engine::draft_logits(int32_t next) {
  if (!mtp_) fatal("draft_logits: the model has no MTP head, or it is off");
  const int n = pos_ - mtp_pos_;
  if (n < 1) fatal("draft_logits: no position is waiting for its next token");
  spec_open_ = false;  // a draft row that was not rejected stands
  gpu_seconds_ = 0;

  int32_t *succ = (int32_t *)mtp_tok_.contents;
  int j = 0;
  for (int32_t t : pend_tok_) succ[j++] = t;
  succ[j++] = next;
  fill_rope(mtp_rope_, mtp_pos_, n);
  @autoreleasepool {
    begin();
    encode_mtp(n, 0, -1, true);
    flush(false);
  }
  mtp_pos_ += n;
  pend_tok_.clear();
  draft_next_ = next;
  draft_rows_ = n;
  return (const float *)mtp_logits_.contents;
}

const float *Engine::run(const int32_t *tokens, int T, bool spec) {
  const Config &c = config();
  if (T < 1 || T > max_chunk_) fatal("forward: %d tokens, chunk limit is %d", T, max_chunk_);
  if (pos_ + T > max_ctx_) fatal("context window of %d tokens is full", max_ctx_);
  if (mtp_ && pos_ > 0 && pos_ == mtp_pos_ && tokens[0] != draft_next_) {
    // draft_logits ran the head on the last position for a next token that is
    // not the one arriving (a generation was cut short): that position waits again.
    const size_t row = (size_t)c.hidden * sizeof(float);
    if (draft_rows_ > 1) memmove(mtp_h_.contents, (const char *)mtp_h_.contents + (draft_rows_ - 1) * row, row);
    mtp_pos_ -= 1;
  }
  const int n_carry = mtp_ ? pos_ - mtp_pos_ : 0;  // hidden states waiting for their next token
  if (spec && n_carry != 0) fatal("forward_draft: call draft_logits first");
  spec_open_ = false;
  gpu_seconds_ = 0;

  memcpy(tok_.contents, tokens, (size_t)T * sizeof(int32_t));
  fill_rope(rope_, pos_, T);
  // Head positions whose next token this call supplies: the waiting ones and
  // all rows of this chunk but the last.
  const int head_rows = mtp_ && !spec ? n_carry + T - 1 : 0;
  if (head_rows > 0) {
    int32_t *succ = (int32_t *)mtp_tok_.contents;
    int j = 0;
    for (int32_t t : pend_tok_) succ[j++] = t;
    for (int t = n_carry > 0 ? 0 : 1; t < T; t++) succ[j++] = tokens[t];
    fill_rope(mtp_rope_, mtp_pos_, head_rows);
  }

  @autoreleasepool {
    begin();
    encode_embed(tok_, T);
    for (int i = 0; i < c.layers; i++) {
      if (c.is_linear_layer(i)) encode_linear_attn(i, T, spec);
      else encode_full_attn(i, T, pos_, rope_);
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

    // Logits for the last row, or for both rows of a speculative pair.
    const int out_rows = spec ? 2 : 1;
    encode_norm(2, 0, out_rows, (size_t)(T - out_rows) * c.hidden * sizeof(float));
    const QTensor &head = model_->q("lm_head");
    encode_linear(head, out_rows, logits_, 0, head.n_pad, false, "linear:lm_head");
    if (mtp_) {
      if (spec) encode_copy_hidden(0, 0, 2);
      else encode_mtp(n_carry, T - 1, T - 1, false);
    }
    flush(false);
  }

  pos_ += T;
  if (mtp_) {
    mtp_pos_ += head_rows;
    pend_tok_.clear();
    if (spec) pend_tok_.push_back(tokens[1]);
    draft_next_ = -1;
  }
  spec_open_ = spec;
  if (!dump_dir_.empty()) {
    FILE *f = fopen((dump_dir_ + "/logits.f32").c_str(), "wb");
    if (!f) fatal("cannot write logits dump");
    fwrite(logits_.contents, sizeof(float), (size_t)c.vocab, f);
    fclose(f);
  }
  return (const float *)logits_.contents;
}

int Engine::argmax() const { return argmax_of((const float *)logits_.contents); }

int Engine::argmax_of(const float *logits) const {
  float best;
  vDSP_Length idx;
  vDSP_maxvi(logits, 1, &best, &idx, (vDSP_Length)config().vocab);
  return (int)idx;
}

}  // namespace rl
