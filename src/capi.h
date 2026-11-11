// C API over the engine, for language bindings.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rl_engine rl_engine;

// Opens a .rlm model. max_ctx bounds the context length, max_chunk the number
// of tokens one rl_forward call may carry. Errors are fatal (the process
// exits with a message): they are setup or usage errors.
rl_engine *rl_open(const char *model_path, int max_ctx, int max_chunk);
void rl_close(rl_engine *e);

int rl_vocab(const rl_engine *e);
int rl_position(const rl_engine *e);
void rl_reset(rl_engine *e);

// Appends n tokens (1 <= n <= max_chunk) and returns the logits of the last
// one: rl_vocab() floats, valid until the next call on this engine.
const float *rl_forward(rl_engine *e, const int32_t *tokens, int n);
int rl_argmax(const rl_engine *e);
double rl_last_gpu_seconds(const rl_engine *e);

// Recurrent state, for returning to an earlier position (see Engine::save_state).
size_t rl_state_bytes(const rl_engine *e);
void rl_save_state(const rl_engine *e, void *dst);
void rl_load_state(rl_engine *e, const void *src, int position);

#ifdef __cplusplus
}
#endif
