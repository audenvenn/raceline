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
// rl_forward with some rows given as vectors, an image's embeddings: n_spans
// spans of four ints (first row in the chunk, rows, index of the first one in
// its image, the image's width in merged patches; see Engine::ImageRows) and
// the spans' vectors in order, [rows][rl_hidden_size()] floats in the model's
// own basis.
const float *rl_forward_embeds(rl_engine *e, const int32_t *tokens, int n, const int32_t *spans, int n_spans,
                               const float *embeds);
// Rotary position (time, height, width) of context position p.
void rl_rope_position(const rl_engine *e, int p, int32_t *out3);
// What attention rotates context position p by: cos then sin of the 32
// rotary frequencies (64 floats).
void rl_rope_angles(const rl_engine *e, int p, float *out64);
// Debug: every forward also writes the residual stream after each layer to
// <dir>/layer_NN.f32 ([n][hidden] float32) and logits.f32 (NULL or "": off).
void rl_set_dump_dir(rl_engine *e, const char *dir);
int rl_argmax(const rl_engine *e);
double rl_last_gpu_seconds(const rl_engine *e);

// Recurrent state, for returning to an earlier position (see Engine::save_state).
size_t rl_state_bytes(const rl_engine *e);
void rl_save_state(const rl_engine *e, void *dst);
void rl_load_state(rl_engine *e, const void *src, int position);
// Attention entries of positions [from, to) in whole blocks of 256, for setting
// a conversation aside and continuing it later (see Engine::save_kv).
size_t rl_kv_bytes(const rl_engine *e, int from, int to);
void rl_save_kv(const rl_engine *e, int from, int to, void *dst);
void rl_load_kv(rl_engine *e, int from, int to, const void *src);

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
// Appends the rows from the block holding token `first` to the end of the
// vocabulary to every draft (-1: none). Draft logits are then rl_draft_main()
// entries for ids 0.., followed by entries for ids rl_draft_tail()...
void rl_set_draft_tail(rl_engine *e, int first);
// Mixture of experts on disk: slots in the expert cache (0: all experts in
// memory) and, since the engine opened, routed experts found loaded / read.
int rl_expert_cache(const rl_engine *e, unsigned long long *hits, unsigned long long *loads);
// The same cache's size: experts in the model (the shared ones included) and bytes per slot.
void rl_expert_cache_size(const rl_engine *e, int *experts, unsigned long long *slot_bytes);
int rl_draft_main(const rl_engine *e);
int rl_draft_tail(const rl_engine *e);
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
