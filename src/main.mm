// redline command line (development build).
//
//   redline dump   <model.rlm> <tokens.i32> <outdir>          per-layer activations for verification
//   redline logits <model.rlm> <tokens.i32> <out.f32> [--chunk N]
//   redline bench  <model.rlm> <tokens.i32> [--prompt N] [--gen N] [--chunk N] [--ctx N]
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

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: redline dump|logits|bench <model.rlm> <tokens.i32> ...\n");
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

  if (cmd == "bench") {
    const int n_prompt = std::min(flag(argc, argv, "--prompt", 1024), (int)toks.size());
    const int n_gen = flag(argc, argv, "--gen", 128);
    const int chunk = flag(argc, argv, "--chunk", 1024);
    const int ctx = flag(argc, argv, "--ctx", n_prompt + n_gen + 16);
    double t0 = now();
    Engine e(argv[2], ctx, chunk);
    printf("load %.2fs\n", now() - t0);

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

      std::vector<int32_t> out;
      gpu = 0;
      t0 = now();
      int32_t next = e.argmax();
      for (int i = 0; i < n_gen; i++) {
        out.push_back(next);
        e.forward(&next, 1);
        gpu += e.last_gpu_seconds();
        next = e.argmax();
      }
      dt = now() - t0;
      printf("%s decode:  %d tokens in %.3fs = %.1f tok/s (GPU time %.3fs = %.1f tok/s)\n",
             rep ? "warm" : "cold", n_gen, dt, n_gen / dt, gpu, n_gen / gpu);
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
