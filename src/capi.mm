#include "capi.h"

#include "engine.h"

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
