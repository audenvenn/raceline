// redline-engine: the engine's development command line.
//
//   dump    <model.rlm> <tokens.i32> <outdir>          per-layer activations for verification
//   logits  <model.rlm> <tokens.i32> <out.f32> [--chunk N]
//   draft   <model.rlm> <tokens.i32> <out.f32>         MTP head logits for the token after the last one
//   bench   <model.rlm> <tokens.i32> [--prompt N] [--gen N] [--chunk N] [--ctx N] [--drafts K]
//   spec    <model.rlm> <tokens.i32> [--prompt N] [--gen N] [--drafts K ...]   speculative vs plain: same tokens?
//           (bench and spec: --draft-vocab N restricts drafts to token ids below N)
//   profile <model.rlm> <tokens.i32> [--prompt N] [--gen N] [--times N]
//
// Token files are raw little-endian int32. Tokenization lives outside the
// engine for now.

#include "engine.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using rl::Engine;

static std::vector<int32_t> read_tokens(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) rl::fatal("cannot open %s", path);
  fseek(f, 0, SEEK_END);
  long n = ftell(f) / 4;
  fseek(f, 0, SEEK_SET);
  std::vector<int32_t> t((size_t)n);
  if (fread(t.data(), 4, (size_t)n, f) != (size_t)n) rl::fatal("short read on %s", path);
  fclose(f);
  return t;
}

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int flag(int argc, char **argv, const char *name, int def) {
  for (int i = 0; i + 1 < argc; i++)
    if (!strcmp(argv[i], name)) return atoi(argv[i + 1]);
  return def;
}

// Greedy generation of n tokens after whatever has been fed. With drafts > 0
// the model's MTP head guesses that many tokens ahead and one pass over the
// weights checks them all. Counts those passes and the GPU time.
static std::vector<int32_t> generate(Engine &e, int n, int drafts, int *passes, double *gpu) {
  std::vector<int32_t> out;
  int32_t next = e.argmax();
  while ((int)out.size() < n) {
    const int k = std::min(drafts, n - (int)out.size() - 1);
    if (k < 1) {
      e.forward(&next, 1);
      *gpu += e.last_gpu_seconds();
      ++*passes;
      out.push_back(next);
      next = e.argmax();
      continue;
    }
    std::vector<int32_t> row = {next};
    const float *guess = e.draft_logits(next);
    *gpu += e.last_gpu_seconds();
    for (int j = 0; j < k; j++) {
      row.push_back(e.argmax_of(guess, e.draft_vocab()));
      if (j + 1 < k) {
        guess = e.draft_more(row.back());
        *gpu += e.last_gpu_seconds();
      }
    }
    e.forward_draft(row.data(), (int)row.size());
    *gpu += e.last_gpu_seconds();
    ++*passes;
    out.push_back(next);
    int a = 0;
    while (a < k && e.argmax_of(e.logits(a)) == row[a + 1]) out.push_back(row[++a]);
    next = e.argmax_of(e.logits(a));
    e.keep_drafts(a);
    *gpu += e.last_gpu_seconds();
  }
  return out;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: redline-engine dump|logits|draft|bench|spec|profile <model.rlm> <tokens.i32> ...\n");
    return 2;
  }
  const std::string cmd = argv[1];
  std::vector<int32_t> toks = read_tokens(argv[3]);

  if (cmd == "dump") {
    if (argc < 5) rl::fatal("dump needs an output directory");
    Engine e(argv[2], 8192, (int)toks.size());
    e.set_dump_dir(argv[4]);
    e.forward(toks.data(), (int)toks.size());
    printf("dumped %d layers for %zu tokens; next token (greedy) = %d\n", e.config().layers, toks.size(), e.argmax());
    return 0;
  }

  if (cmd == "logits") {
    if (argc < 5) rl::fatal("logits needs an output file");
    const int chunk = flag(argc, argv, "--chunk", (int)toks.size());
    Engine e(argv[2], 8192, chunk);
    const float *lg = nullptr;
    for (size_t i = 0; i < toks.size(); i += chunk)
      lg = e.forward(toks.data() + i, (int)std::min((size_t)chunk, toks.size() - i));
    FILE *f = fopen(argv[4], "wb");
    if (!f) rl::fatal("cannot write %s", argv[4]);
    fwrite(lg, sizeof(float), (size_t)e.config().vocab, f);
    fclose(f);
    printf("chunk %d: next token (greedy) = %d\n", chunk, e.argmax());
    return 0;
  }

  if (cmd == "draft") {
    if (argc < 5) rl::fatal("draft needs an output file");
    const int n = (int)toks.size();
    Engine e(argv[2], 8192, n);
    if (!e.has_mtp()) rl::fatal("this model file has no MTP head");
    e.forward(toks.data(), n - 1);
    const float *lg = e.draft_logits(toks[n - 1]);
    FILE *f = fopen(argv[4], "wb");
    if (!f) rl::fatal("cannot write %s", argv[4]);
    fwrite(lg, sizeof(float), (size_t)e.config().vocab, f);
    fclose(f);
    printf("draft for the token after the last one = %d\n", e.argmax_of(lg));
    return 0;
  }

  if (cmd == "spec") {
    const int n_prompt = std::min(flag(argc, argv, "--prompt", 512), (int)toks.size());
    const int n_gen = flag(argc, argv, "--gen", 256);
    const int chunk = flag(argc, argv, "--chunk", 1024);
    std::vector<int> modes = {0};  // drafts per pass; 0 = plain decoding
    for (int i = 0; i < argc; i++)
      if (!strcmp(argv[i], "--drafts"))
        for (int j = i + 1; j < argc && argv[j][0] != '-'; j++) modes.push_back(atoi(argv[j]));
    if (modes.size() == 1) modes.insert(modes.end(), {1, 2, 3, 4});
    Engine e(argv[2], n_prompt + n_gen + 16, chunk);
    if (!e.has_mtp()) rl::fatal("this model file has no MTP head");
    e.set_draft_vocab(flag(argc, argv, "--draft-vocab", 0));
    std::vector<int32_t> ref;
    bool same = true;
    for (int drafts : modes) {
      std::vector<int32_t> out;
      for (int rep = 0; rep < 2; rep++) {  // the first run warms pipelines and pages in the weights
        e.reset();
        e.set_mtp(drafts > 0);
        double t0 = now();
        for (int i = 0; i < n_prompt; i += chunk) e.forward(toks.data() + i, std::min(chunk, n_prompt - i));
        const double t_prefill = now() - t0;
        int passes = 0;
        double gpu = 0;
        t0 = now();
        out = generate(e, n_gen, drafts, &passes, &gpu);
        const double dt = now() - t0;
        if (!rep) continue;
        printf("%d draft%s: prefill %7.1f tok/s   decode %d tokens in %3d passes (%.2f per pass), %.3fs = %5.1f tok/s\n",
               drafts, drafts == 1 ? " " : "s", n_prompt / t_prefill, n_gen, passes, (double)n_gen / passes, dt, n_gen / dt);
        fflush(stdout);
      }
      if (drafts == 0) ref = out;
      else same = same && out == ref;
    }
    printf("generated tokens: %s\n", same ? "identical in every mode" : "DIFFERENT");
    return same ? 0 : 1;
  }

  if (cmd == "bench") {
    const int n_prompt = std::min(flag(argc, argv, "--prompt", 1024), (int)toks.size());
    const int n_gen = flag(argc, argv, "--gen", 128);
    const int chunk = flag(argc, argv, "--chunk", 1024);
    const int ctx = flag(argc, argv, "--ctx", n_prompt + n_gen + 16);
    double t0 = now();
    Engine e(argv[2], ctx, chunk);
    printf("load %.2fs\n", now() - t0);
    const int drafts = e.has_mtp() ? std::min(flag(argc, argv, "--drafts", 4), Engine::kMaxDrafts) : 0;
    if (e.has_mtp()) {
      e.set_draft_vocab(flag(argc, argv, "--draft-vocab", 0));
      e.set_mtp(drafts > 0);
      printf("MTP head: %d draft%s per pass\n", drafts, drafts == 1 ? "" : "s");
    }

    for (int rep = 0; rep < 2; rep++) {  // first pass warms pipelines and pages in the weights
      e.reset();
      double gpu = 0;
      t0 = now();
      for (int i = 0; i < n_prompt; i += chunk) {
        e.forward(toks.data() + i, std::min(chunk, n_prompt - i));
        gpu += e.last_gpu_seconds();
      }
      double dt = now() - t0;
      printf("%s prefill: %d tokens in %.3fs = %.1f tok/s (GPU time %.3fs = %.1f tok/s)\n",
             rep ? "warm" : "cold", n_prompt, dt, n_prompt / dt, gpu, n_prompt / gpu);

      gpu = 0;
      int passes = 0;
      t0 = now();
      const std::vector<int32_t> out = generate(e, n_gen, drafts, &passes, &gpu);
      dt = now() - t0;
      printf("%s decode:  %d tokens in %.3fs = %.1f tok/s (GPU time %.3fs = %.1f tok/s; %d passes)\n",
             rep ? "warm" : "cold", n_gen, dt, n_gen / dt, gpu, n_gen / gpu, passes);
      if (rep) {
        printf("tokens:");
        for (size_t i = 0; i < out.size() && i < 24; i++) printf(" %d", out[i]);
        printf("\n");
      }
    }
    return 0;
  }

  if (cmd == "profile") {
    const int n_prompt = std::min(flag(argc, argv, "--prompt", 1024), (int)toks.size());
    const int n_gen = flag(argc, argv, "--gen", 32);
    const int chunk = flag(argc, argv, "--chunk", 1024);
    const int times = flag(argc, argv, "--times", 4);
    Engine e(argv[2], n_prompt + n_gen + 16, chunk);
    if (e.has_mtp()) e.set_mtp(false);  // one token per pass: the head is profiled separately
    // Returns {prefill GPU s, decode GPU s}, best of two runs each.
    auto measure = [&]() {
      double best_p = 1e30, best_d = 1e30;
      for (int rep = 0; rep < 2; rep++) {
        e.reset();
        double gp = 0, gd = 0;
        for (int i = 0; i < n_prompt; i += chunk) {
          e.forward(toks.data() + i, std::min(chunk, n_prompt - i));
          gp += e.last_gpu_seconds();
        }
        int32_t next = e.argmax();
        for (int i = 0; i < n_gen; i++) {
          e.forward(&next, 1);
          gd += e.last_gpu_seconds();
          next = e.argmax();
        }
        best_p = std::min(best_p, gp);
        best_d = std::min(best_d, gd);
      }
      return std::pair<double, double>(best_p, best_d);
    };
    measure();  // compile pipelines, page in weights
    const auto base = measure();
    const char *kernels[] = {"linear:gate_up", "linear:down", "linear:lin_in", "linear:lin_out", "linear:attn_qkv",
                             "linear:attn_out", "linear:lm_head", "group_sums", "rms_norm", "swiglu", "lin_conv",
                             "lin_prep_qk", "lin_gates", "gated_delta", "lin_out_norm", "attn_prep", "attn_scores",
                             "attn_softmax", "attn_pv",
                             "embed"};
    printf("in-situ cost per kernel (ms per token): prefill of %d tokens, decode of %d tokens\n", n_prompt, n_gen);
    printf("  %-18s %10s %7s %10s %7s\n", "kernel", "prefill", "", "decode", "");
    double sum_p = 0, sum_d = 0;
    const double bp = base.first / n_prompt * 1e3, bd = base.second / n_gen * 1e3;
    for (const char *k : kernels) {
      e.set_repeat(k, times);
      const auto m = measure();
      e.set_repeat("", 0);
      const double dp = (m.first - base.first) / times / n_prompt * 1e3;
      const double dd = (m.second - base.second) / times / n_gen * 1e3;
      sum_p += dp;
      sum_d += dd;
      printf("  %-18s %10.4f %6.1f%% %10.3f %6.1f%%\n", k, dp, 100 * dp / bp, dd, 100 * dd / bd);
      fflush(stdout);
    }
    printf("  %-18s %10.4f %6.1f%% %10.3f %6.1f%%\n", "(unattributed)", bp - sum_p, 100 * (bp - sum_p) / bp, bd - sum_d,
           100 * (bd - sum_d) / bd);
    printf("  %-18s %10.4f %7s %10.3f   = %.1f tok/s prefill, %.1f tok/s decode\n", "total", bp, "", bd,
           n_prompt / base.first, n_gen / base.second);
    return 0;
  }

  rl::fatal("unknown command '%s'", cmd.c_str());
}
