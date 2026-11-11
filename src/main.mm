// redline-engine: the engine's development command line.
//
//   dump    <model.rlm> <tokens.i32> <outdir>          per-layer activations for verification
//   logits  <model.rlm> <tokens.i32> <out.f32> [--chunk N]
//   draft   <model.rlm> <tokens.i32> <out.f32>         MTP head logits for the token after the last one
//   bench   <model.rlm> <tokens.i32> [--prompt N] [--gen N] [--chunk N] [--ctx N] [--mtp 0|1]
//   spec    <model.rlm> <tokens.i32> [--prompt N] [--gen N]   speculative vs plain decoding: same tokens?
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

// Greedy generation of n tokens after whatever has been fed. With `spec`, the
// model's MTP head drafts one token ahead and a pass over the weights checks
// two tokens at once. Counts those passes and the GPU time.
static std::vector<int32_t> generate(Engine &e, int n, bool spec, int *passes, double *gpu) {
  std::vector<int32_t> out;
  int32_t next = e.argmax();
  while ((int)out.size() < n) {
    if (!spec || (int)out.size() + 1 >= n) {
      e.forward(&next, 1);
      *gpu += e.last_gpu_seconds();
      ++*passes;
      out.push_back(next);
      next = e.argmax();
      continue;
    }
    const int32_t draft = e.argmax_of(e.draft_logits(next));
    *gpu += e.last_gpu_seconds();
    e.forward_draft(next, draft);
    *gpu += e.last_gpu_seconds();
    ++*passes;
    out.push_back(next);
    const int32_t tok = e.argmax_of(e.logits(0));
    if (tok == draft) {
      out.push_back(tok);
      next = e.argmax_of(e.logits(1));
    } else {
      e.reject_draft();
      next = tok;
    }
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
    Engine e(argv[2], n_prompt + n_gen + 16, chunk);
    if (!e.has_mtp()) rl::fatal("this model file has no MTP head");
    std::vector<int32_t> out[2];
    for (int mode = 0; mode < 2; mode++) {
      for (int rep = 0; rep < 2; rep++) {  // the first run warms pipelines and pages in the weights
        e.reset();
        e.set_mtp(mode == 1);
        double t0 = now();
        for (int i = 0; i < n_prompt; i += chunk) e.forward(toks.data() + i, std::min(chunk, n_prompt - i));
        const double t_prefill = now() - t0;
        int passes = 0;
        double gpu = 0;
        t0 = now();
        out[mode] = generate(e, n_gen, mode == 1, &passes, &gpu);
        const double dt = now() - t0;
        if (!rep) continue;
        printf("%-12s prefill %7.1f tok/s   decode %d tokens in %d passes, %.3fs = %.1f tok/s", mode ? "speculative:" : "plain:",
               n_prompt / t_prefill, n_gen, passes, dt, n_gen / dt);
        if (mode) printf("   (drafts accepted: %.1f%%)", 100.0 * (n_gen - passes) / std::max(1, passes - 1));
        printf("\n");
      }
    }
    const bool same = out[0] == out[1];
    printf("generated tokens: %s\n", same ? "identical" : "DIFFERENT");
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
    const bool mtp = e.has_mtp() && flag(argc, argv, "--mtp", 1) != 0;
    e.set_mtp(mtp);
    if (e.has_mtp()) printf("MTP head: %s\n", mtp ? "on (speculative decoding)" : "off");

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
      const std::vector<int32_t> out = generate(e, n_gen, mtp, &passes, &gpu);
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
                             "lin_prep_qk", "lin_gates", "gated_delta", "lin_out_norm", "attn_prep", "attn_sdpa",
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
