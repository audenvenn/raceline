// Thin Metal runtime: device, kernel compilation with caching, buffers.
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <map>
#include <string>

namespace rl {

// Preprocessor macros handed to the Metal compiler; they select tile sizes,
// element types and other per-pipeline specializations.
using Macros = std::map<std::string, std::string>;

class Metal {
 public:
  Metal();

  id<MTLDevice> device() const { return dev_; }
  id<MTLCommandQueue> queue() const { return queue_; }

  // Compiles kernels/<file> (resolving `#include "..."` against the kernel
  // directory) and returns the compute pipeline for `fn`. Results are cached by
  // (file, fn, macros). Aborts with the compiler log on failure.
  id<MTLComputePipelineState> pipeline(const std::string &file, const std::string &fn,
                                       const Macros &macros = {});

  // Shared-storage buffer, zero-filled, length rounded up to a whole page.
  id<MTLBuffer> alloc(size_t bytes);

  // Wraps existing page-aligned memory (e.g. an mmap'd weight file) without copying.
  id<MTLBuffer> wrap(void *ptr, size_t bytes);

  static size_t page_size();
  static size_t round_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

 private:
  std::string load_source(const std::string &file, int depth = 0);

  id<MTLDevice> dev_;
  id<MTLCommandQueue> queue_;
  std::string kernel_dir_;
  std::map<std::string, id<MTLComputePipelineState>> cache_;
};

[[noreturn]] void fatal(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

}  // namespace rl
