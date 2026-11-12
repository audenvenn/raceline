#include "capi.h"

#include "engine.h"

#import <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>

struct rl_engine {
  rl::Engine engine;
  rl_engine(const char *path, int max_ctx, int max_chunk) : engine(path, max_ctx, max_chunk) {}
};

rl_engine *rl_open(const char *model_path, int max_ctx, int max_chunk) {
  return new rl_engine(model_path, max_ctx, max_chunk);
}
void rl_close(rl_engine *e) { delete e; }

int rl_vocab(const rl_engine *e) { return e->engine.config().vocab; }
int rl_position(const rl_engine *e) { return e->engine.position(); }
void rl_reset(rl_engine *e) { e->engine.reset(); }

const float *rl_forward(rl_engine *e, const int32_t *tokens, int n) { return e->engine.forward(tokens, n); }
int rl_argmax(const rl_engine *e) { return e->engine.argmax(); }
double rl_last_gpu_seconds(const rl_engine *e) { return e->engine.last_gpu_seconds(); }

size_t rl_state_bytes(const rl_engine *e) { return e->engine.state_bytes(); }
void rl_save_state(const rl_engine *e, void *dst) { e->engine.save_state(dst); }
void rl_load_state(rl_engine *e, const void *src, int position) { e->engine.load_state(src, position); }

int rl_has_mtp(const rl_engine *e) { return e->engine.has_mtp(); }
void rl_set_mtp(rl_engine *e, int on) { e->engine.set_mtp(on != 0); }
const float *rl_draft_logits(rl_engine *e, int32_t next_token) { return e->engine.draft_logits(next_token); }
const float *rl_draft_more(rl_engine *e, int32_t guess) { return e->engine.draft_more(guess); }
const float *rl_forward_draft(rl_engine *e, const int32_t *tokens, int n) { return e->engine.forward_draft(tokens, n); }
void rl_keep_drafts(rl_engine *e, int accepted) { e->engine.keep_drafts(accepted); }
int rl_max_drafts(const rl_engine *) { return rl::Engine::kMaxDrafts; }
void rl_set_draft_vocab(rl_engine *e, int rows) { e->engine.set_draft_vocab(rows); }
int rl_draft_vocab(const rl_engine *e) { return e->engine.draft_vocab(); }
void rl_set_draft_tail(rl_engine *e, int first) { e->engine.set_draft_tail(first); }
void rl_expert_cache_size(const rl_engine *e, int *experts, unsigned long long *slot_bytes) {
  const rl::Engine::ExpertStats s = e->engine.expert_stats();
  if (experts) *experts = s.experts;
  if (slot_bytes) *slot_bytes = s.slot_bytes;
}
int rl_expert_cache(const rl_engine *e, unsigned long long *hits, unsigned long long *loads) {
  const rl::Engine::ExpertStats s = e->engine.expert_stats();
  if (hits) *hits = s.hits;
  if (loads) *loads = s.loads;
  return s.slots;
}
int rl_draft_main(const rl_engine *e) { return e->engine.draft_main(); }
int rl_draft_tail(const rl_engine *e) { return e->engine.draft_tail(); }
size_t rl_logits_stride(const rl_engine *e) { return (size_t)(e->engine.logits(1) - e->engine.logits(0)); }

const float *rl_hidden(const rl_engine *e) { return e->engine.hidden(); }
int rl_hidden_size(const rl_engine *e) { return e->engine.config().hidden; }

int rl_top_k(const float *x, int n, int k, int32_t *ids, float *vals) {
  k = std::min(k, n);
  if (k <= 0) return 0;
  constexpr int kBlock = 256;
  int m = 0;  // entries held, in descending order
  float thr = -INFINITY;
  for (int b = 0; b < n; b += kBlock) {
    const int len = std::min(kBlock, n - b);
    if (m == k) {  // most blocks hold nothing above the current k-th value: one vector max rules them out
      float mx;
      vDSP_maxv(x + b, 1, &mx, (vDSP_Length)len);
      if (!(mx > thr)) continue;
    }
    for (int i = b; i < b + len; i++) {
      const float v = x[i];
      if (m == k ? !(v > thr) : v != v) continue;
      int j = m < k ? m++ : k - 1;
      for (; j > 0 && vals[j - 1] < v; j--) {
        vals[j] = vals[j - 1];
        ids[j] = ids[j - 1];
      }
      vals[j] = v;
      ids[j] = i;
      if (m == k) thr = vals[k - 1];
    }
  }
  return m;
}
