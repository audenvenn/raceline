#include "metal_rt.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>

#ifndef RL_KERNEL_DIR
#define RL_KERNEL_DIR "kernels"
#endif

namespace rl {

void fatal(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "redline: fatal: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}

size_t Metal::page_size() { return (size_t)getpagesize(); }

Metal::Metal() {
  dev_ = MTLCreateSystemDefaultDevice();
  if (!dev_) fatal("no Metal device");
  queue_ = [dev_ newCommandQueue];
  const char *env = getenv("RL_KERNEL_DIR");
  kernel_dir_ = env ? env : RL_KERNEL_DIR;
}

std::string Metal::load_source(const std::string &file, int depth) {
  if (depth > 8) fatal("kernel include depth exceeded at %s", file.c_str());
  std::ifstream in(kernel_dir_ + "/" + file);
  if (!in) fatal("cannot read kernel source %s/%s", kernel_dir_.c_str(), file.c_str());
  std::ostringstream out;
  std::string line;
  while (std::getline(in, line)) {
    // Runtime compilation has no include path, so splice local includes by hand.
    size_t pos = line.find("#include \"");
    if (pos != std::string::npos && line.find_first_not_of(" \t") == pos) {
      size_t a = pos + 10, b = line.find('"', a);
      out << load_source(line.substr(a, b - a), depth + 1) << "\n";
    } else {
      out << line << "\n";
    }
  }
  return out.str();
}

id<MTLComputePipelineState> Metal::pipeline(const std::string &file, const std::string &fn,
                                            const Macros &macros) {
  std::string key = file + "|" + fn;
  for (const auto &[k, v] : macros) key += "|" + k + "=" + v;
  auto it = cache_.find(key);
  if (it != cache_.end()) return it->second;

  MTLCompileOptions *opt = [MTLCompileOptions new];
  opt.languageVersion = MTLLanguageVersion4_1;
  NSMutableDictionary<NSString *, NSObject *> *defs = [NSMutableDictionary dictionary];
  for (const auto &[k, v] : macros) defs[@(k.c_str())] = @(v.c_str());
  opt.preprocessorMacros = defs;

  NSError *err = nil;
  id<MTLLibrary> lib = [dev_ newLibraryWithSource:@(load_source(file).c_str()) options:opt error:&err];
  if (!lib) fatal("compiling %s failed:\n%s", file.c_str(), err.localizedDescription.UTF8String);
  id<MTLFunction> func = [lib newFunctionWithName:@(fn.c_str())];
  if (!func) fatal("kernel %s not found in %s", fn.c_str(), file.c_str());
  id<MTLComputePipelineState> pso = [dev_ newComputePipelineStateWithFunction:func error:&err];
  if (!pso) fatal("pipeline for %s failed:\n%s", fn.c_str(), err.localizedDescription.UTF8String);
  cache_[key] = pso;
  return pso;
}

id<MTLBuffer> Metal::alloc(size_t bytes) {
  size_t len = round_up(bytes ? bytes : 1, page_size());
  id<MTLBuffer> buf = [dev_ newBufferWithLength:len options:MTLResourceStorageModeShared];
  if (!buf) fatal("cannot allocate %zu-byte buffer", len);
  return buf;
}

id<MTLBuffer> Metal::wrap(void *ptr, size_t bytes) {
  if ((uintptr_t)ptr % page_size() || bytes % page_size())
    fatal("wrap: memory must be page aligned (ptr %p, %zu bytes)", ptr, bytes);
  id<MTLBuffer> buf = [dev_ newBufferWithBytesNoCopy:ptr
                                              length:bytes
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
  if (!buf) fatal("cannot wrap %zu bytes at %p", bytes, ptr);
  return buf;
}

}  // namespace rl
