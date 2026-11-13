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
  ~Engine();

  const Config &config() const { return model_->config(); }
  int position() const { return pos_; }
  int max_chunk() const { return max_chunk_; }

  // Forgets all context (conv state, DeltaNet state, KV cache).
  void reset();

  // Appends T tokens to the context and returns the logits (vocab floats) for
  // the last one. The pointer stays valid until the next call.
  const float *forward(const int32_t *tokens, int T);

  // Rows of a chunk that carry an image instead of a token's table row: rows
  // [first, first + rows) of the chunk are the vision tower's output rows
  // [offset, offset + rows) of an image `width` merged patches wide. An image
  // continued from the previous chunk (offset > 0) starts the chunk.
  struct ImageRows { int first, rows, offset, width; };
  // forward() with rows given as vectors: `embeds` holds the spans' rows in
  // order, [rows][hidden] floats in the model's own basis (the engine rotates
  // them for a rotated model). Rotary positions follow Qwen3.5's M-RoPE: the
  // patch in row r, column c of an image sits at (time, height, width) =
  // (s, s + r, s + c), s being where a text token would have been, and the
  // text after the image goes on from one past the largest of them.
  const float *forward_embeds(const int32_t *tokens, int T, const ImageRows *spans, int n_spans,
                              const float *embeds);
  // Rotary position (time, height, width) of context position p < position().
  void rope_position(int p, int32_t out[3]) const;
  // What the attention layers rotate context position p by: cos then sin of
  // the 32 rotary frequencies (for checks against a reference).
  void rope_angles(int p, float out[64]) const;

  // Greedy pick from the logits of the last forward().
  int argmax() const;

  // GPU seconds spent in the last forward().
  double last_gpu_seconds() const { return gpu_seconds_; }

  // Mixture of experts on disk (RACELINE_EXPERT_CACHE): slots in the cache, and
  // since the engine opened how many routed experts a layer found loaded and how
  // many it had to read. slots == 0: the experts are all in memory.
  struct ExpertStats {
    int slots, experts;          // slots in the cache; experts in the model (with the shared ones)
    size_t slot_bytes;
    uint64_t hits, loads;
  };
  ExpertStats expert_stats() const;

  // The recurrent layers' state (conv window and DeltaNet matrices), which is
  // all that is needed to return to an earlier position: attention entries
  // past that position are overwritten as new tokens arrive. A state saved at
  // position p may be loaded for as long as the first p tokens are unchanged.
  size_t state_bytes() const;
  void save_state(void *dst) const;
  void load_state(const void *src, int position);

  // Attention entries of positions [from, to) in every attention layer, the
  // draft head's included, in whole blocks of 256 positions (from rounded
  // down, to up), with the positions' rotary positions. With a saved state
  // they set a conversation aside: after
  // another one has used the context, load_kv puts its entries back and
  // load_state continues it, provided the positions before the first block
  // still hold the same tokens.
  size_t kv_bytes(int from, int to) const;
  void save_kv(int from, int to, void *dst) const;
  void load_kv(int from, int to, const void *src);

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
  static constexpr int kMaxDrafts = 15;
  bool has_mtp() const { return has_mtp_; }
  // On by default when the file has a head. Only at position 0. When on, every
  // forward also runs the head over the new positions (one extra layer).
  void set_mtp(bool on);
  // The head's logits for the token after `next`, the token about to be fed.
  const float *draft_logits(int32_t next);
  // The head applied to its own output: logits for the token after `guess`,
  // the token picked from the previous draft_logits / draft_more call.
  const float *draft_more(int32_t guess);
  // Drafts only need the likely tokens: restricts the head's vocabulary pass
  // to token ids below `rows` (rounded up to a block of 256; 0 = all). Lower
  // ids are the more frequent tokens, and the pass over the output matrix is
  // most of what a draft costs. A token outside the range is simply never
  // drafted. draft_logits / draft_more then return that many logits.
  void set_draft_vocab(int rows);
  // The chat template's special tokens sit at the top of the id range, outside
  // any such restriction. set_draft_tail(first) appends the rows from the block
  // of 256 holding token `first` to the end of the vocabulary to every draft's
  // pass (-1: none): a draft call then returns draft_main() logits for ids
  // 0.. followed by the appended rows, from id draft_tail() on.
  void set_draft_tail(int first);
  int draft_vocab() const;   // logits a draft call returns
  int draft_main() const;    // of which the first draft_main() are for ids 0 ..
  int draft_tail() const;    // and the rest for ids draft_tail() .. (-1: no rest)
  // Feeds tokens[0] and, speculatively, tokens[1..n-1] (n <= kMaxDrafts + 1).
  const float *forward_draft(const int32_t *tokens, int n);
  // Keeps the first `accepted` drafts of the last forward_draft and takes the
  // rest back. Must follow forward_draft before any other call.
  void keep_drafts(int accepted);
  // Logits of the last forward: row 0, or row r after forward_draft().
  const float *logits(int row = 0) const;
  int argmax_of(const float *logits, int n = 0) const;  // over the first n entries (0 = the vocabulary)

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
  const float *run(const int32_t *tokens, int T, bool spec, const ImageRows *spans = nullptr, int n_spans = 0);
  // Rotary positions of context positions p0 .. p0+T-1: text, or the spans' image rows (chunk rows).
  void set_positions(int p0, int T, const ImageRows *spans = nullptr, int n_spans = 0);
  int32_t text_position(int p) const;  // a text token's at context position p
  void rope_row(int p, float *out) const;  // cos[32], sin[32] of context position p
  void fill_rope(id<MTLBuffer> buf, int pos0, int T);
  void encode_embed(id<MTLBuffer> tok, int T);
  // Rows of h_ from emb_ for the spans of this pass: chunk row i -> h_ row i + shift, rows < M.
  void encode_image_rows(int shift, int M);
  void encode_linear(const QTensor &w, int rows, id<MTLBuffer> out, size_t out_offset, int out_stride,
                     bool accum, const char *what);
  // which: 0 = a layer's input norm, 1 = its post-attention norm, 2 = the final norm.
  void encode_norm(int which, int layer, int rows, size_t in_offset);
  void encode_rms(id<MTLBuffer> in, size_t in_offset, int in_stride, id<MTLBuffer> wgt, size_t wgt_offset,
                  int rows, size_t out_offset);
  void bind_column_scale(const QTensor &consumer, int index);
  void encode_mlp(int layer, int T);
  // A mixture-of-experts MLP over rows 0..T-1 of h_ (kernels/moe.metal).
  void encode_moe(int layer, int T);
  // The experts' weights, kind 0 (gate/up) or 1 (down), for the grouped matmul:
  // the stacked tensor of the file, or the expert cache, with its slot table.
  void bind_experts(int layer, int kind, const QTensor &stack);
  // The name of a layer's first MLP matrix: "mlp.gate_up", or "moe.gate_up" for a MoE layer.
  std::string mlp_in(int layer) const;
  void encode_linear_attn(int layer, int T, bool spec);
  void encode_full_attn(int layer, int T, int pos0, id<MTLBuffer> rope);
  void encode_copy_rows(id<MTLBuffer> dst, int dst_row, int src_row, int rows);  // rows of h_
  // Hidden half of the head's input: norm_h(w1-norm(rows of src)) -> xn_ rows first_row.., second half.
  void encode_mtp_hidden(id<MTLBuffer> src, const FTensor &w1, int rows, int first_row);
  // The head from its input projection on, for M rows whose next tokens are
  // in mtp_tok_ and whose hidden halves are in xn_; positions pos0.. .
  void encode_mtp_layer(int M, int pos0, bool logits, int image_shift = 0);  // image rows: chunk row i -> head row i + image_shift
  // Runs the head over n_carry rows of mtp_h_ followed by n_chunk rows of h_,
  // after saving row keep_row of h_.
  void encode_mtp(int n_carry, int n_chunk, int keep_row, bool logits);
  void begin();                 // opens a command buffer + compute encoder
  void flush(bool reopen);      // commits, waits, accumulates GPU time
  void run_threads(int w, int h);
  void run_groups(int w, int h, int threads_per_group = 32, int depth = 1);
  // Threadgroups as a GPU-written MTLDispatchThreadgroupsIndirectArguments says.
  void run_indirect(id<MTLBuffer> args, size_t offset, int threads_per_group);
  void redo();                  // issues the last dispatch again
  void mark(const char *what);  // names the dispatch just issued (for set_repeat)
  bool repeating(const char *what) const { return repeat_times_ > 0 && repeat_ == what; }

  Metal mt_;
  std::unique_ptr<ModelFile> model_;
  int max_ctx_, max_chunk_, rows_cap_;
  int pos_ = 0;
  // Rotary position (time, height, width) of every context position, and of
  // the positions drafts may reach past them; all three equal for text.
  std::vector<int32_t> rpos_;
  // The image rows of the pass being encoded (forward_embeds): the spans, and
  // their vectors in emb_ ([max_chunk_][hidden] floats, rotated as the model is).
  std::vector<ImageRows> spans_;
  id<MTLBuffer> emb_ = nil, emb_rot_ = nil;  // emb_rot_: the rotation's signs / sqrt(block)
  double gpu_seconds_ = 0;
  std::string dump_dir_;
  std::string repeat_;
  int repeat_times_ = 0;
  struct Dispatch {
    bool threads;
    MTLSize grid, per_group;
    id<MTLBuffer> indirect = nil;     // set: the grid comes from this buffer
    size_t indirect_offset = 0;
  } last_{};
  id<MTLBuffer> scratch_state_ = nil, scratch_yo_ = nil;
  id<MTLCommandBuffer> cb_ = nil;
  id<MTLComputeCommandEncoder> enc_ = nil;

  // Scratch activations, all [rows_cap_][stride].
  int x_stride_, proj_stride_;
  id<MTLBuffer> tok_, h_, xn_, gsum_, proj_, y_, qk_, bg_, yo_, rope_, logits_;
  // Mixture of experts: router logits [rows][router rows], routing results
  // [rows * slots], the work list and its dispatch sizes, and the sorted rows'
  // inputs and outputs [rows * slots][width].
  id<MTLBuffer> moe_logits_ = nil, moe_tok_ = nil, moe_inv_ = nil, moe_w_ = nil, moe_sel_ = nil, moe_items_ = nil,
                moe_args_ = nil, moe_x_ = nil, moe_gu_ = nil, moe_act_ = nil, moe_y_ = nil;
  id<MTLBuffer> moe_identity_ = nil;               // slot = expert: experts read straight from the file
  // RACELINE_EXPERT_CACHE=<GB>: experts live in the file and the ones in use
  // in a cache of that size (see ExpertCache in engine.mm).
  struct ExpertCache;
  std::unique_ptr<ExpertCache> cache_;
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
  int draft_vocab_ = 0;               // rows of the output matrix a draft looks at (0 = all)
  int draft_tail_ = -1;               // first row of the block of rows appended to them (-1 = none)
  std::vector<int32_t> pend_tok_;
  id<MTLBuffer> mtp_h_ = nil, mtp_out_ = nil, mtp_tok_ = nil, mtp_rope_ = nil, mtp_logits_ = nil;
  id<MTLBuffer> mtp_norm_e_ = nil, mtp_norm_h_ = nil, mtp_norm_out_ = nil;
  // Rotated models (Config::rot_block): weights of the draft head's
  // unrotate-and-normalize step and of its output rotation (kernels/ops.metal).
  id<MTLBuffer> mtp_rot_w1_ = nil, mtp_rot_g_ = nil;
  void encode_rot(const char *kernel, id<MTLBuffer> src, size_t src_offset, id<MTLBuffer> w1, id<MTLBuffer> w2,
                  size_t w2_offset, int rows, size_t out_offset);
};

}  // namespace rl
