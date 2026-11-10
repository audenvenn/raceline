// Correctness and speed of the quantized linear kernel on synthetic data.
//
//   correctness: GPU result vs. a double-precision CPU evaluation of the same
//                affine 4-bit model, on a sample of outputs.
//   speed:       decode (M=1), speculative verification (M=8..64) and prefill
//                (M=1024) on the layer shapes of Qwen3.5-4B and Qwen3.8-27B.

#include "../src/metal_rt.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using rl::Macros;
using rl::Metal;

namespace {

constexpr int kGroup = 64;
constexpr int kBlock = 256;

struct LinearParams { int M, K, N, a_stride, c_stride, G; };
struct Tile { int tm, tn, nsg; };

uint64_t g_seed = 0x9e3779b97f4a7c15ULL;
uint64_t rnd() { g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17; return g_seed; }
float uniform(float lo, float hi) { return lo + (hi - lo) * (float)((rnd() >> 11) * (1.0 / 9007199254740992.0)); }

// A quantized layer in the blocked on-GPU layout, plus (optionally) the plain
// row-major values for the CPU reference.
struct Layer {
  int K, N, G, n_blocks;
  id<MTLBuffer> w, s, b;
  std::vector<uint8_t> q;        // [N][K], only kept when keep_ref
  std::vector<_Float16> sc, bi;  // [N][G]
};

Layer make_layer(Metal &mt, int K, int N, bool keep_ref, bool high_nibble_first = false) {
  Layer L;
  L.K = K; L.N = N; L.G = K / kGroup; L.n_blocks = (N + kBlock - 1) / kBlock;
  size_t w_bytes = (size_t)L.n_blocks * K * (kBlock / 2);
  size_t sb_bytes = (size_t)L.n_blocks * L.G * kBlock * sizeof(_Float16);
  L.w = mt.alloc(w_bytes);
  L.s = mt.alloc(sb_bytes);
  L.b = mt.alloc(sb_bytes);
  uint8_t *w = (uint8_t *)L.w.contents;
  _Float16 *s = (_Float16 *)L.s.contents, *b = (_Float16 *)L.b.contents;
  if (!keep_ref) {
    for (size_t i = 0; i < w_bytes; i += 8) { uint64_t x = rnd(); memcpy(w + i, &x, 8); }
    for (size_t i = 0; i < sb_bytes / 2; i++) { s[i] = (_Float16)0.004f; b[i] = (_Float16)-0.03f; }
    return L;
  }
  L.q.resize((size_t)N * K);
  L.sc.resize((size_t)N * L.G);
  L.bi.resize((size_t)N * L.G);
  for (auto &v : L.q) v = (uint8_t)(rnd() & 15);
  for (size_t i = 0; i < L.sc.size(); i++) {
    float sv = uniform(0.001f, 0.012f);
    L.sc[i] = (_Float16)sv;
    L.bi[i] = (_Float16)(-7.5f * sv + uniform(-0.01f, 0.01f));
  }
  for (int n = 0; n < N; n++) {
    int blk = n / kBlock, col = n % kBlock;
    for (int k = 0; k < K; k++) {
      uint8_t v = L.q[(size_t)n * K + k];
      size_t byte = ((size_t)blk * K + k) * (kBlock / 2) + col / 2;
      bool high = (col & 1) != high_nibble_first;
      w[byte] |= high ? (uint8_t)(v << 4) : v;
    }
    for (int g = 0; g < L.G; g++) {
      size_t o = ((size_t)blk * L.G + g) * kBlock + col;
      s[o] = L.sc[(size_t)n * L.G + g];
      b[o] = L.bi[(size_t)n * L.G + g];
    }
  }
  return L;
}

struct Runner {
  Metal &mt;
  id<MTLComputePipelineState> pso_sum, pso_lin;
  Tile t;
  Runner(Metal &m, Tile tile, const char *out_t = "float") : mt(m), t(tile) {
    Macros mac = {{"TM", std::to_string(t.tm)}, {"TN", std::to_string(t.tn)}, {"NSG", std::to_string(t.nsg)},
                  {"GROUP", std::to_string(kGroup)}, {"BLOCK", std::to_string(kBlock)}, {"OUT_T", out_t}};
    pso_sum = mt.pipeline("linear.metal", "group_sums", mac);
    pso_lin = mt.pipeline("linear.metal", "linear_q4a", mac);
  }
  bool fits() const { return 32 * t.nsg <= (int)pso_lin.maxTotalThreadsPerThreadgroup; }

  // Encodes gsum + linear `inner` times (cycling through `layers`); returns GPU seconds per pass.
  double run(const std::vector<Layer> &layers, id<MTLBuffer> a, id<MTLBuffer> gsum, id<MTLBuffer> c,
             int M, int inner, bool with_sums = true) {
    const Layer &L0 = layers[0];
    int m_pad = (M + t.tm - 1) / t.tm * t.tm;
    int n_pad = L0.n_blocks * kBlock;
    LinearParams p{m_pad, L0.K, n_pad, L0.K, n_pad, L0.G};
    id<MTLCommandBuffer> cb = [mt.queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int i = 0; i < inner; i++) {
      const Layer &L = layers[i % layers.size()];
      if (with_sums) {
        [enc setComputePipelineState:pso_sum];
        [enc setBuffer:a offset:0 atIndex:0];
        [enc setBuffer:gsum offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(L.G, m_pad, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }
      [enc setComputePipelineState:pso_lin];
      [enc setBuffer:a offset:0 atIndex:0];
      [enc setBuffer:L.w offset:0 atIndex:1];
      [enc setBuffer:L.s offset:0 atIndex:2];
      [enc setBuffer:L.b offset:0 atIndex:3];
      [enc setBuffer:gsum offset:0 atIndex:4];
      [enc setBuffer:c offset:0 atIndex:5];
      [enc setBytes:&p length:sizeof(p) atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(n_pad / t.tn, m_pad / t.tm, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * t.nsg, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted)
      rl::fatal("command buffer failed: %s", cb.error.localizedDescription.UTF8String);
    return (cb.GPUEndTime - cb.GPUStartTime) / inner;
  }
};

id<MTLBuffer> make_input(Metal &mt, int M, int K, int tm_max = 128) {
  int m_pad = (M + tm_max - 1) / tm_max * tm_max;
  id<MTLBuffer> a = mt.alloc((size_t)m_pad * K * sizeof(_Float16));
  _Float16 *p = (_Float16 *)a.contents;
  for (size_t i = 0; i < (size_t)M * K; i++) p[i] = (_Float16)uniform(-2.0f, 2.0f);
  return a;
}

bool check(Metal &mt, int M, int K, int N, Tile t, bool high_first, bool verbose) {
  Layer L = make_layer(mt, K, N, true, high_first);
  Runner r(mt, t);
  int n_pad = L.n_blocks * kBlock, m_pad = (M + 127) / 128 * 128;
  id<MTLBuffer> a = make_input(mt, M, K);
  id<MTLBuffer> gsum = mt.alloc((size_t)m_pad * L.G * sizeof(float));
  id<MTLBuffer> c = mt.alloc((size_t)m_pad * n_pad * sizeof(float));
  r.run({L}, a, gsum, c, M, 1);

  const _Float16 *av = (const _Float16 *)a.contents;
  const float *cv = (const float *)c.contents;
  double max_err = 0, sum_sq = 0;
  int samples = 0;
  for (int m = 0; m < M; m++) {
    for (int j = 0; j < 96; j++) {
      int n = (j < 4) ? j : (j < 8 ? N - 1 - (j - 4) : (int)(rnd() % N));
      double ref = 0;
      for (int k = 0; k < K; k++) {
        size_t gi = (size_t)n * L.G + k / kGroup;
        double wq = (double)L.sc[gi] * L.q[(size_t)n * K + k] + (double)L.bi[gi];
        ref += (double)av[(size_t)m * K + k] * wq;
      }
      double got = cv[(size_t)m * n_pad + n];
      max_err = fmax(max_err, fabs(got - ref));
      sum_sq += ref * ref;
      samples++;
    }
  }
  double rms = sqrt(sum_sq / samples);
  bool ok = max_err < 2e-3 * rms + 1e-4;
  if (verbose || !ok)
    printf("  M=%-5d K=%-6d N=%-6d tile=%d,%d,%d nibble=%s  max|err|=%.3e  rms(ref)=%.3e  rel=%.2e  %s\n",
           M, K, N, t.tm, t.tn, t.nsg, high_first ? "high-first" : "low-first", max_err, rms, max_err / rms,
           ok ? "OK" : "MISMATCH");
  return ok;
}

void bench(Metal &mt) {
  struct Shape { int K, N; const char *what; double mlx_ms_1, mlx_ms_8, mlx_ms_64, mlx_tflops_1024; };
  // MLX 0.32.4 reference times measured on the same machine (bench/hw/gpu_probe.py).
  const Shape shapes[] = {
      {5120, 17408, "27B gate/up", 0.416, 0.784, 0.936, 11.45},
      {17408, 5120, "27B down", 0.423, 0.877, 1.345, 7.36},
      {2560, 9216, "4B gate/up", 0.107, 0.189, 0.309, 13.20},
  };
  const int Ms[] = {1, 8, 32, 64, 1024};
  const Tile tiles[] = {{16, 64, 2}, {16, 128, 4}, {16, 256, 8}, {32, 64, 2}, {32, 128, 4},
                        {64, 64, 4}, {64, 128, 8}, {128, 64, 8}, {128, 128, 16}};
  for (const Shape &sh : shapes) {
    size_t layer_bytes = (size_t)sh.N * sh.K / 2 + (size_t)sh.N * (sh.K / kGroup) * 4;
    int copies = (int)fmax(1.0, ceil(120e6 / (double)layer_bytes));  // defeat caching of small layers
    std::vector<Layer> layers;
    for (int i = 0; i < copies; i++) layers.push_back(make_layer(mt, sh.K, sh.N, false));
    int n_pad = layers[0].n_blocks * kBlock;
    printf("\n== %s: K=%d N=%d, %.1f MB/layer incl. scales (4.5 bpw), %d copies ==\n", sh.what, sh.K, sh.N,
           layer_bytes / 1e6, copies);
    printf("%6s %-12s %9s %9s %8s   %s\n", "M", "tile", "ms", "W GB/s", "TFLOPS", "vs MLX");
    for (int M : Ms) {
      int m_pad = (M + 127) / 128 * 128;
      id<MTLBuffer> a = make_input(mt, M, sh.K);
      id<MTLBuffer> gsum = mt.alloc((size_t)m_pad * (sh.K / kGroup) * sizeof(float));
      id<MTLBuffer> c = mt.alloc((size_t)m_pad * n_pad * sizeof(float));
      double best = 1e30; Tile bt{};
      for (Tile t : tiles) {
        if (t.tm > 16 && t.tm > 2 * M) continue;
        Runner r(mt, t);
        if (!r.fits()) continue;
        int inner = M >= 512 ? 3 : 2 * copies > 24 ? 2 * copies : 24;
        double dt = 1e30;
        for (int rep = 0; rep < 4; rep++) dt = fmin(dt, r.run(layers, a, gsum, c, M, inner));
        if (dt < best) { best = dt; bt = t; }
      }
      char tb[32]; snprintf(tb, sizeof tb, "%d,%d,%d", bt.tm, bt.tn, bt.nsg);
      double tf = 2.0 * M * sh.K * sh.N / best / 1e12;
      char cmp[64] = "";
      if (M == 1) snprintf(cmp, sizeof cmp, "%.2fx (MLX %.3f ms)", sh.mlx_ms_1 / (best * 1e3), sh.mlx_ms_1);
      if (M == 8) snprintf(cmp, sizeof cmp, "%.2fx (MLX %.3f ms)", sh.mlx_ms_8 / (best * 1e3), sh.mlx_ms_8);
      if (M == 64) snprintf(cmp, sizeof cmp, "%.2fx (MLX %.3f ms)", sh.mlx_ms_64 / (best * 1e3), sh.mlx_ms_64);
      if (M == 1024) snprintf(cmp, sizeof cmp, "%.2fx (MLX %.2f TFLOPS)", tf / sh.mlx_tflops_1024, sh.mlx_tflops_1024);
      printf("%6d %-12s %9.3f %9.1f %8.2f   %s\n", M, tb, best * 1e3, layer_bytes / best / 1e9, tf, cmp);
      fflush(stdout);
    }
  }
}

}  // namespace

// Every tile for one shape and row count: used to pick tiles for the engine.
void sweep(Metal &mt, int K, int N, int M) {
  size_t layer_bytes = (size_t)N * K / 2 + (size_t)N * (K / kGroup) * 4;
  int copies = (int)fmax(1.0, ceil(120e6 / (double)layer_bytes));
  std::vector<Layer> layers;
  for (int i = 0; i < copies; i++) layers.push_back(make_layer(mt, K, N, false));
  int n_pad = layers[0].n_blocks * kBlock, m_pad = (M + 127) / 128 * 128;
  id<MTLBuffer> a = make_input(mt, M, K);
  id<MTLBuffer> gsum = mt.alloc((size_t)m_pad * (K / kGroup) * sizeof(float));
  id<MTLBuffer> c = mt.alloc((size_t)m_pad * n_pad * sizeof(float));
  printf("sweep K=%d N=%d M=%d (%.1f MB/layer, %d copies)\n", K, N, M, layer_bytes / 1e6, copies);
  const Tile tiles[] = {{16, 32, 1}, {16, 64, 2}, {16, 128, 4}, {16, 256, 8}, {32, 32, 1}, {32, 64, 2}, {32, 128, 4},
                        {64, 32, 2}, {64, 64, 4}, {64, 128, 8}, {128, 64, 8}, {128, 128, 16}};
  for (Tile t : tiles) {
    if (t.tm > 16 && t.tm > 2 * M) continue;
    Runner r(mt, t);
    if (!r.fits()) continue;
    int inner = M >= 512 ? 3 : 2 * copies > 24 ? 2 * copies : 24;
    double dt = 1e30;
    for (int rep = 0; rep < 4; rep++) dt = fmin(dt, r.run(layers, a, gsum, c, M, inner));
    printf("  tile %3d,%3d,%2d  %8.3f ms  %7.1f GB/s  %6.2f TFLOPS\n", t.tm, t.tn, t.nsg, dt * 1e3,
           layer_bytes / dt / 1e9, 2.0 * M * K * N / dt / 1e12);
  }
}

int main(int argc, char **argv) {
  bool do_bench = false;
  for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--bench")) do_bench = true;
  if (argc == 5 && !strcmp(argv[1], "--sweep")) {
    @autoreleasepool {
      Metal mt;
      sweep(mt, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    }
    return 0;
  }
  @autoreleasepool {
    Metal mt;
    printf("device: %s\n", mt.device().name.UTF8String);

    // Which nibble of a byte holds the even-indexed element? Settle it empirically.
    bool low_ok = check(mt, 3, 256, 512, {16, 64, 2}, false, false);
    bool high_ok = check(mt, 3, 256, 512, {16, 64, 2}, true, false);
    printf("nibble order: low-first %s, high-first %s\n", low_ok ? "matches" : "no", high_ok ? "matches" : "no");
    if (low_ok == high_ok) {
      check(mt, 3, 256, 512, {16, 64, 2}, false, true);
      check(mt, 3, 256, 512, {16, 64, 2}, true, true);
      rl::fatal("could not determine nibble order");
    }
    bool hf = high_ok;

    printf("correctness (GPU vs double-precision CPU reference):\n");
    bool all = true;
    all &= check(mt, 1, 2560, 9216, {16, 64, 2}, hf, true);
    all &= check(mt, 5, 2560, 9216, {16, 128, 4}, hf, true);
    all &= check(mt, 40, 2560, 9216, {32, 64, 2}, hf, true);
    all &= check(mt, 70, 2560, 9216, {64, 64, 4}, hf, true);
    all &= check(mt, 3, 9216, 2560, {16, 256, 8}, hf, true);
    all &= check(mt, 2, 2560, 1000, {16, 64, 2}, hf, true);   // N not a multiple of the block
    all &= check(mt, 130, 5120, 1024, {128, 64, 8}, hf, true);
    printf("%s\n", all ? "ALL CORRECT" : "FAILURES");
    if (!all) return 1;
    if (do_bench) bench(mt);
  }
  return 0;
}
