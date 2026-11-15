// Does the GPU slow down when it waits for the CPU? (#47, fixed in #50)
//
// An engine whose experts live on disk stops the GPU after every layer: the
// GPU signals a shared event, the CPU reads the experts the layer chose, and
// signals back; the GPU waits for that value before the next layer. This
// probe runs the same pattern with no model: a command buffer of LAYERS
// "layers" of real work (PER small dispatches each), the GPU handing over to
// the CPU after each layer, and the CPU answering after DELAY microseconds
// from a spinning thread (0: at once). It times the command buffer three ways:
//
//   work only     the same dispatches, no handoffs
//   handoffs      a GPU -> CPU -> GPU round trip after every layer
//   + keep-alive  the same, with a light kernel looping on a second queue
//                 the whole time
//
// and prints the time per layer beyond the work and the CPU's delay. The GPU's
// power management takes the waits for idleness and lowers its performance,
// so the work after each wait runs slower; a light kernel on another queue
// keeps it up. On the M6 Mac mini (two runs): with the CPU answering at once,
// 43-68 us per layer beyond the work, 38-64 with the keep-alive; answering
// after 100-300 us, as a read from the SSD does, 79-102 us beyond the delay,
// 50-67 with the keep-alive. In the engine, where a pass waits 40 times, the
// keep-alive made single-token passes of Qwen3.6-35B-A3B with experts on
// disk 27% faster (44.3 -> 56.4 tok/s; #47, #50). The engine's ExpertCache
// runs such a queue (RACELINE_GPU_KEEPALIVE=0 turns it off).
//
// build: clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation \
//          -o build/hw/gpu_wait bench/hw/gpu_wait.mm
// run:   build/hw/gpu_wait [delay_us ...]       (default: 0 100 300)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <pthread.h>
#include <sys/qos.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static const char *kSource = R"(
#include <metal_stdlib>
using namespace metal;
kernel void work(device float *a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float x = a[i];
  for (int k = 0; k < 64; k++) x = x * 0.999f + 0.001f;
  a[i] = x;
}
kernel void keepalive(device float *a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float x = a[i];
  for (int k = 0; k < 256; k++) x = x * 0.999f + 0.001f;
  a[i] = x;
}
)";

static const int LAYERS = 40, PER = 4, REPS = 11;
using clock_type = std::chrono::steady_clock;

static double seconds_since(clock_type::time_point t) {
  return std::chrono::duration<double>(clock_type::now() - t).count();
}

int main(int argc, char **argv) {
  std::vector<int> delays;
  for (int i = 1; i < argc; i++) delays.push_back(atoi(argv[i]));
  if (delays.empty()) delays = {0, 100, 300};

  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [dev newCommandQueue], side = [dev newCommandQueue];
  NSError *err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSource) options:nil error:&err];
  if (!lib) {
    fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String);
    return 1;
  }
  id<MTLComputePipelineState> work = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"work"] error:nil];
  id<MTLComputePipelineState> keep = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"keepalive"]
                                                                       error:nil];
  id<MTLBuffer> buf = [dev newBufferWithLength:4 << 20 options:MTLResourceStorageModeShared];
  id<MTLBuffer> kbuf = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
  id<MTLSharedEvent> ev = [dev newSharedEvent];
  printf("%s, %d layers of %d dispatches per command buffer, median of %d\n", dev.name.UTF8String, LAYERS, PER, REPS);

  auto dispatch = [&](id<MTLComputeCommandEncoder> enc) {
    [enc setComputePipelineState:work];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(65536, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };

  // One command buffer; returns its wall time in seconds.
  auto run = [&](bool handoffs, int delay_us) {
    uint64_t v0 = ev.signaledValue, v = v0;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int l = 0; l < LAYERS; l++) {
      for (int j = 0; j < PER; j++) dispatch(enc);
      if (!handoffs) continue;
      [enc endEncoding];
      [cb encodeSignalEvent:ev value:++v];   // GPU -> CPU: the layer is done
      [cb encodeWaitForEvent:ev value:++v];  // CPU -> GPU: go on
      enc = [cb computeCommandEncoder];
    }
    [enc endEncoding];
    std::thread cpu([&] {  // answers each signal after delay_us, spinning (no sleep latency)
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      const auto start = clock_type::now();
      for (uint64_t want = v0 + 1; want < v; want += 2) {
        while (ev.signaledValue < want) {
          if (seconds_since(start) > 10) {
            fprintf(stderr, "timeout waiting for the GPU\n");
            ev.signaledValue = v;
            return;
          }
        }
        const auto t = clock_type::now();
        while (seconds_since(t) * 1e6 < delay_us) {
        }
        ev.signaledValue = want + 1;
      }
    });
    const auto t0 = clock_type::now();
    [cb commit];
    [cb waitUntilCompleted];
    const double dt = seconds_since(t0);
    cpu.join();
    return dt;
  };

  // The keep-alive: small dispatches back to back on the side queue while `on`.
  std::atomic<bool> on{false}, quit{false};
  std::thread keeper([&] {
    while (!quit) {
      if (!on) {
        std::this_thread::yield();
        continue;
      }
      @autoreleasepool {
        id<MTLCommandBuffer> cb = [side commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:keep];
        [enc setBuffer:kbuf offset:0 atIndex:0];
        for (int j = 0; j < 8; j++)
          [enc dispatchThreads:MTLSizeMake(1024, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
      }
    }
  });

  auto median = [&](bool handoffs, int delay_us, bool keepalive) {
    on = keepalive;
    std::vector<double> t;
    for (int r = 0; r < REPS + 1; r++) {
      const double dt = run(handoffs, delay_us);
      if (r) t.push_back(dt);  // the first one warms up
    }
    on = false;
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
  };

  const double base = median(false, 0, false);
  printf("work only:                  %7.3f ms per command buffer (%.1f us per layer)\n", base * 1e3,
         base / LAYERS * 1e6);
  for (int d : delays) {
    const double plain = median(true, d, false), kept = median(true, d, true);
    auto extra = [&](double t) { return (t - base) / LAYERS * 1e6 - d; };
    printf("CPU answers after %4d us:   handoffs %7.3f ms (+%6.1f us per layer beyond the delay), "
           "with keep-alive %7.3f ms (+%6.1f us)\n",
           d, plain * 1e3, extra(plain), kept * 1e3, extra(kept));
    fflush(stdout);
  }
  quit = true;
  keeper.join();
  return 0;
}
