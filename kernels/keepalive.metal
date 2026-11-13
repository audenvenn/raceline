// Light work run back to back on a queue of its own while passes with experts
// on disk come in (ExpertCache::keepalive in src/engine.mm): the GPU then
// never sits idle while it waits for the CPU, and keeps its performance.

#include <metal_stdlib>
using namespace metal;

kernel void keepalive(device float *a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float x = a[i];
  for (int k = 0; k < 256; k++) x = x * 0.999f + 0.001f;
  a[i] = x;
}
