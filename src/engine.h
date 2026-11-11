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
  void encode_linear(const QTensor &w, int rows, id<MTLBuffer> out, size_t out_offset, int out_stride,
                     bool accum, const char *what);
  // which: 0 = a layer's input norm, 1 = its post-attention norm, 2 = the final norm.
  void encode_norm(int which, int layer, int rows, size_t in_offset);
  void bind_column_scale(const QTensor &consumer, int index);
  void encode_mlp(int layer, int T);
  void encode_linear_attn(int layer, int T);
  void encode_full_attn(int layer, int T);
  void begin();                 // opens a command buffer + compute encoder
  void flush(bool reopen);      // commits, waits, accumulates GPU time
  void run_threads(int w, int h);
  void run_groups(int w, int h, int threads_per_group = 32);
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
  id<MTLBuffer> tok_, h_, xn_, gsum_, proj_, y_, qk_, bg_, yo_, qn_, rope_, logits_;

  // Native-format models scale each matrix's input columns outside the matmul:
  // the norm weights are pre-multiplied by the column scale of the matrix they
  // feed, and the other producers multiply by it explicitly (ones_ otherwise).
  std::vector<id<MTLBuffer>> norm_in_, norm_post_;
  id<MTLBuffer> norm_final_ = nil, ones_ = nil;

  std::vector<std::shared_ptr<LinearLayer>> lin_;
  std::vector<std::shared_ptr<AttnLayer>> attn_;
};

}  // namespace rl
