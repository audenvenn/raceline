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

// Speculative decoding with the model's MTP head; see Engine in engine.h.
// rl_forward_draft feeds tokens[0] and the drafts tokens[1..n-1]; the logits
// after row r start rl_logits_stride() * r floats into the returned array.
// rl_keep_drafts(accepted) must follow it.
int rl_has_mtp(const rl_engine *e);
int rl_max_drafts(const rl_engine *e);
void rl_set_mtp(rl_engine *e, int on);
const float *rl_draft_logits(rl_engine *e, int32_t next_token);
const float *rl_draft_more(rl_engine *e, int32_t guess);
const float *rl_forward_draft(rl_engine *e, const int32_t *tokens, int n);
void rl_keep_drafts(rl_engine *e, int accepted);
size_t rl_logits_stride(const rl_engine *e);
// Restricts drafts to token ids below `rows` (0 = the whole vocabulary); the
// draft calls then return rl_draft_vocab() logits.
void rl_set_draft_vocab(rl_engine *e, int rows);
int rl_draft_vocab(const rl_engine *e);

// The k largest of n logits in descending order (ties: lowest index first).
// Writes their indices and values and returns how many (min(k, n)). A helper
// for samplers: a full sort or partition of the vocabulary costs more than
// the rest of sampling together.
int rl_top_k(const float *logits, int n, int k, int32_t *ids, float *vals);

// Final hidden states of the last rl_forward, [n][hidden] floats (MTP off).
const float *rl_hidden(const rl_engine *e);
int rl_hidden_size(const rl_engine *e);

#ifdef __cplusplus
}
#endif
