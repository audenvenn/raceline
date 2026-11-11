// Forward pass of the Qwen3.5 / Qwen3.8 text model on the GPU.
#pragma once

#include "metal_rt.h"
#include "model_file.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rl {

class Engine {
 public:
  // max_ctx bounds the attention KV cache; max_chunk is the most tokens one
  // forward() call may carry (longer prompts are fed in several calls).
  Engine(const std::string &model_path, int max_ctx = 8192, int max_chunk = 1024);

  const Config &config() const { return model_->config(); }
  int position() const { return pos_; }
  int max_chunk() const { return max_chunk_; }

  // Forgets all context (conv state, DeltaNet state, KV cache).
  void reset();

  // Appends T tokens to the context and returns the logits (vocab floats) for
  // the last one. The pointer stays valid until the next call.
  const float *forward(const int32_t *tokens, int T);

  // Greedy pick from the logits of the last forward().
  int argmax() const;

  // GPU seconds spent in the last forward().
  double last_gpu_seconds() const { return gpu_seconds_; }

  // The recurrent layers' state (conv window and DeltaNet matrices), which is
  // all that is needed to return to an earlier position: attention entries
  // past that position are overwritten as new tokens arrive. A state saved at
  // position p may be loaded for as long as the first p tokens are unchanged.
  size_t state_bytes() const;
  void save_state(void *dst) const;
  void load_state(const void *src, int position);

  // ---- Speculative decoding with the model's own multi-token-prediction head
  // (present in files extended by tools/add_mtp.py). The head guesses the
  // tokens that follow the next one; feeding the guesses along with the next
  // token costs about the same pass over the weights as the next token alone,
  // and every guess the model then confirms is a token that needed no pass of
  // its own. Results are bit-identical to feeding the tokens one at a time.
  //
  //   next = pick(logits)                           // not fed yet
  //   d[0] = pick(draft_logits(next))               // the head's guess for the token after next
  //   d[j] = pick(draft_more(d[j-1]))               // ... and for the ones after that
  //   forward_draft({next, d[0], ..., d[k-1]})      // logits(r): after row r
  //   a = 0; while (a < k && pick(logits(a)) == d[a]) a++;
  //   next = pick(logits(a)); keep_drafts(a);       // rows after the a-th draft are taken back
  static constexpr int kMaxDrafts = 7;
  bool has_mtp() const { return has_mtp_; }
  // On by default when the file has a head. Only at position 0. When on, every
  // forward also runs the head over the new positions (one extra layer).
  void set_mtp(bool on);
  // The head's logits for the token after `next`, the token about to be fed.
  const float *draft_logits(int32_t next);
  // The head applied to its own output: logits for the token after `guess`,
  // the token picked from the previous draft_logits / draft_more call.
  const float *draft_more(int32_t guess);
  // Feeds tokens[0] and, speculatively, tokens[1..n-1] (n <= kMaxDrafts + 1).
  const float *forward_draft(const int32_t *tokens, int n);
  // Keeps the first `accepted` drafts of the last forward_draft and takes the
  // rest back. Must follow forward_draft before any other call.
  void keep_drafts(int accepted);
  // Logits of the last forward: row 0, or row r after forward_draft().
  const float *logits(int row = 0) const;
  int argmax_of(const float *logits) const;

  // Final hidden states [T][hidden] of the last forward(), before the final
  // norm. Only meaningful with the head off (the head reuses the buffer).
  const float *hidden() const { return (const float *)h_.contents; }

  // Differential profiling: every dispatch of the named kernel is repeated
  // `times` extra times. Repeats recompute the same outputs (or write to
  // scratch), so results are unchanged and the added GPU time is that
  // kernel's cost inside the real submission pattern. Per-kernel timers do not
  // work here: separate submissions inflate small kernels, and pass-boundary
  // timestamps of consecutive passes overlap.
  void set_repeat(const std::string &kernel, int times) {
    repeat_ = kernel;
    repeat_times_ = times;
  }

  // Debug: when set, forward() also writes the residual stream after every
  // layer to <dir>/layer_NN.f32 ([T][hidden] float32), plus logits.f32.
  void set_dump_dir(const std::string &dir) { dump_dir_ = dir; }

 private:
  struct Tile { int tm, tn, nsg; };
  struct LinearLayer;  // per-layer recurrent state
  struct AttnLayer;

  Tile tile_for(int rows) const;
  // Tensor name of a layer's part; layer == config().layers is the MTP head's decoder layer.
  std::string lname(int layer, const char *suffix) const;
  bool is_linear(int layer) const { return layer < config().layers && config().is_linear_layer(layer); }
  const float *run(const int32_t *tokens, int T, bool spec);
  void fill_rope(id<MTLBuffer> buf, int pos0, int T);
  void encode_embed(id<MTLBuffer> tok, int T);
  void encode_linear(const QTensor &w, int rows, id<MTLBuffer> out, size_t out_offset, int out_stride,
                     bool accum, const char *what);
  // which: 0 = a layer's input norm, 1 = its post-attention norm, 2 = the final norm.
  void encode_norm(int which, int layer, int rows, size_t in_offset);
  void encode_rms(id<MTLBuffer> in, size_t in_offset, int in_stride, id<MTLBuffer> wgt, size_t wgt_offset,
                  int rows, size_t out_offset);
  void bind_column_scale(const QTensor &consumer, int index);
  void encode_mlp(int layer, int T);
  void encode_linear_attn(int layer, int T, bool spec);
  void encode_full_attn(int layer, int T, int pos0, id<MTLBuffer> rope);
  void encode_copy_rows(id<MTLBuffer> dst, int dst_row, int src_row, int rows);  // rows of h_
  // Hidden half of the head's input: norm_h(w1-norm(rows of src)) -> xn_ rows first_row.., second half.
  void encode_mtp_hidden(id<MTLBuffer> src, const FTensor &w1, int rows, int first_row);
  // The head from its input projection on, for M rows whose next tokens are
  // in mtp_tok_ and whose hidden halves are in xn_; positions pos0.. .
  void encode_mtp_layer(int M, int pos0, bool logits);
  // Runs the head over n_carry rows of mtp_h_ followed by n_chunk rows of h_,
  // after saving row keep_row of h_.
  void encode_mtp(int n_carry, int n_chunk, int keep_row, bool logits);
  void begin();                 // opens a command buffer + compute encoder
  void flush(bool reopen);      // commits, waits, accumulates GPU time
  void run_threads(int w, int h);
  void run_groups(int w, int h, int threads_per_group = 32, int depth = 1);
  void redo();                  // issues the last dispatch again
  void mark(const char *what);  // names the dispatch just issued (for set_repeat)
  bool repeating(const char *what) const { return repeat_times_ > 0 && repeat_ == what; }

  Metal mt_;
  std::unique_ptr<ModelFile> model_;
  int max_ctx_, max_chunk_, rows_cap_;
  int pos_ = 0;
  double gpu_seconds_ = 0;
  std::string dump_dir_;
  std::string repeat_;
  int repeat_times_ = 0;
  struct Dispatch {
    bool threads;
    MTLSize grid, per_group;
  } last_{};
  id<MTLBuffer> scratch_state_ = nil, scratch_yo_ = nil;
  id<MTLCommandBuffer> cb_ = nil;
  id<MTLComputeCommandEncoder> enc_ = nil;

  // Scratch activations, all [rows_cap_][stride].
  int x_stride_, proj_stride_;
  id<MTLBuffer> tok_, h_, xn_, gsum_, proj_, y_, qk_, bg_, yo_, rope_, logits_;
  // Attention: queries [KVH][q_rows_][head_dim] half, and for one group of rows
  // the scores (float), unnormalized probabilities (half) and their row sums.
  id<MTLBuffer> aq_, as_, ap_, al_;
  int q_rows_ = 0, kv_blocks_ = 0;
  size_t score_cap_ = 0;  // elements in as_ / ap_

  // Native-format models scale each matrix's input columns outside the matmul:
  // the norm weights are pre-multiplied by the column scale of the matrix they
  // feed, and the other producers multiply by it explicitly (ones_ otherwise).
  std::vector<id<MTLBuffer>> norm_in_, norm_post_;
  id<MTLBuffer> norm_final_ = nil, ones_ = nil;

  std::vector<std::shared_ptr<LinearLayer>> lin_;
  std::vector<std::shared_ptr<AttnLayer>> attn_;

  // MTP head. Head position i pairs the hidden state of position i with token
  // i+1, so a position can only be processed once its next token is known:
  // positions mtp_pos_ .. pos_-1 are waiting, their hidden states parked in
  // mtp_h_ and the tokens that followed all but the last in pend_tok_.
  bool has_mtp_ = false, mtp_ = false;
  int spec_rows_ = 0;                 // rows of the last forward_draft, until keep_drafts settles them
  int mtp_pos_ = 0;
  int draft_next_ = -1, draft_rows_ = 0, draft_depth_ = 0;  // the draft_logits / draft_more chain in progress
  std::vector<int32_t> pend_tok_;
  id<MTLBuffer> mtp_h_ = nil, mtp_out_ = nil, mtp_tok_ = nil, mtp_rope_ = nil, mtp_logits_ = nil;
  id<MTLBuffer> mtp_norm_e_ = nil, mtp_norm_h_ = nil, mtp_norm_out_ = nil;
};

}  // namespace rl
