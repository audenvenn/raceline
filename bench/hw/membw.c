// CPU-side memory bandwidth probe for Apple Silicon.
//
// Single-stream LLM decode reads every weight once per token, so sustained
// DRAM read bandwidth is the hard ceiling on tokens/s. This measures what the
// CPU cores can pull, per core class, as a reference point for the GPU numbers.
//
// build: clang -O3 -mcpu=native -o membw membw.c -lpthread
// run:   ./membw [buffer_GiB=4]

#include <arm_neon.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static double now_s(void) {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

typedef enum { MODE_READ, MODE_COPY, MODE_FILL } bench_mode;

typedef struct {
  const uint8_t *src;
  uint8_t *dst;
  size_t n;
  bench_mode mode;
  qos_class_t qos;
  uint64_t sink;
} job;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static int g_go = 0;

// Read-only pass: 8 independent accumulators, 128 bytes (one cache line on
// Apple cores) per iteration, so the loop is load-bound rather than add-bound.
static uint64_t read_sum(const uint8_t *p, size_t n) {
  uint64x2_t a0 = vdupq_n_u64(0), a1 = a0, a2 = a0, a3 = a0;
  uint64x2_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    a0 = vaddq_u64(a0, vld1q_u64((const uint64_t *)(p + i)));
    a1 = vaddq_u64(a1, vld1q_u64((const uint64_t *)(p + i + 16)));
    a2 = vaddq_u64(a2, vld1q_u64((const uint64_t *)(p + i + 32)));
    a3 = vaddq_u64(a3, vld1q_u64((const uint64_t *)(p + i + 48)));
    a4 = vaddq_u64(a4, vld1q_u64((const uint64_t *)(p + i + 64)));
    a5 = vaddq_u64(a5, vld1q_u64((const uint64_t *)(p + i + 80)));
    a6 = vaddq_u64(a6, vld1q_u64((const uint64_t *)(p + i + 96)));
    a7 = vaddq_u64(a7, vld1q_u64((const uint64_t *)(p + i + 112)));
  }
  a0 = vaddq_u64(vaddq_u64(a0, a1), vaddq_u64(a2, a3));
  a4 = vaddq_u64(vaddq_u64(a4, a5), vaddq_u64(a6, a7));
  a0 = vaddq_u64(a0, a4);
  return vgetq_lane_u64(a0, 0) + vgetq_lane_u64(a0, 1);
}

static void *worker(void *arg) {
  job *j = (job *)arg;
  pthread_set_qos_class_self_np(j->qos, 0);
  pthread_mutex_lock(&g_mu);
  while (!g_go) pthread_cond_wait(&g_cv, &g_mu);
  pthread_mutex_unlock(&g_mu);
  switch (j->mode) {
    case MODE_READ: j->sink = read_sum(j->src, j->n); break;
    case MODE_COPY: memcpy(j->dst, j->src, j->n); j->sink = j->dst[j->n / 2]; break;
    case MODE_FILL: memset(j->dst, 0x5a, j->n); j->sink = j->dst[j->n / 2]; break;
  }
  return NULL;
}

// Returns best-of-reps bandwidth in GB/s (decimal GB) of bytes *touched*:
// read = n, fill = n written, copy = n read + n written.
static double run(const uint8_t *src, uint8_t *dst, size_t total, int threads,
                  bench_mode mode, qos_class_t qos, int reps) {
  double best = 0;
  size_t per = (total / threads) & ~(size_t)127;
  for (int r = 0; r < reps; r++) {
    pthread_t th[64];
    job jobs[64];
    g_go = 0;
    for (int t = 0; t < threads; t++) {
      jobs[t] = (job){src + (size_t)t * per, dst ? dst + (size_t)t * per : NULL,
                      per, mode, qos, 0};
      pthread_create(&th[t], NULL, worker, &jobs[t]);
    }
    // Let threads park on the condvar before starting the clock.
    struct timespec ts = {0, 20 * 1000 * 1000};
    nanosleep(&ts, NULL);
    double t0 = now_s();
    pthread_mutex_lock(&g_mu);
    g_go = 1;
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
    uint64_t sink = 0;
    for (int t = 0; t < threads; t++) {
      pthread_join(th[t], NULL);
      sink += jobs[t].sink;
    }
    double dt = now_s() - t0;
    double bytes = (double)per * threads * (mode == MODE_COPY ? 2.0 : 1.0);
    double gbps = bytes / dt / 1e9;
    if (gbps > best) best = gbps;
    if (sink == 0x123456789abcULL) fprintf(stderr, "!");
  }
  return best;
}

int main(int argc, char **argv) {
  double gib = argc > 1 ? atof(argv[1]) : 4.0;
  size_t total = (size_t)(gib * (1ull << 30)) & ~(size_t)16383;
  uint8_t *src = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  uint8_t *dst = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (src == MAP_FAILED || dst == MAP_FAILED) { perror("mmap"); return 1; }
  // Fault everything in with non-zero, non-compressible-looking data.
  uint64_t x = 0x9e3779b97f4a7c15ULL;
  for (size_t i = 0; i < total; i += 8) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *(uint64_t *)(src + i) = x;
  }
  memset(dst, 1, total);

  printf("buffer: %.1f GiB x2, reps: best of 5\n", gib);
  printf("%-28s %8s %10s %10s %10s\n", "qos (core class)", "threads", "read GB/s", "fill GB/s", "copy GB/s");
  struct { const char *name; qos_class_t q; int counts[8]; } cfg[] = {
      {"user-interactive (P/Super)", QOS_CLASS_USER_INTERACTIVE, {1, 2, 4, 6, 8, 12, 0}},
      {"background (E only)", QOS_CLASS_BACKGROUND, {1, 2, 4, 6, 0}},
  };
  for (int c = 0; c < 2; c++) {
    for (int k = 0; cfg[c].counts[k]; k++) {
      int t = cfg[c].counts[k];
      double rd = run(src, NULL, total, t, MODE_READ, cfg[c].q, 5);
      double fl = run(src, dst, total, t, MODE_FILL, cfg[c].q, 5);
      double cp = run(src, dst, total, t, MODE_COPY, cfg[c].q, 5);
      printf("%-28s %8d %10.1f %10.1f %10.1f\n", cfg[c].name, t, rd, fl, cp);
      fflush(stdout);
    }
  }
  return 0;
}
