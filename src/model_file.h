// Reader for .rlm model files: one read-only mmap, handed to Metal section by
// section without copying.
#pragma once

#include "metal_rt.h"

#include <map>
#include <string>
#include <vector>

namespace rl {

struct Config {
  int hidden = 0, layers = 0, intermediate = 0;
  int heads = 0, kv_heads = 0, head_dim = 0;
  int lin_v_heads = 0, lin_k_heads = 0, lin_k_dim = 0, lin_v_dim = 0, conv_taps = 0;
  int vocab = 0, full_attn_interval = 4, eos = -1;
  float norm_eps = 1e-6f;
  double rope_theta = 1e7;
  float partial_rotary = 0.25f;
  int group = 64, block = 256;

  bool is_linear_layer(int i) const { return (i + 1) % full_attn_interval != 0; }
  int lin_key_dim() const { return lin_k_heads * lin_k_dim; }
  int lin_value_dim() const { return lin_v_heads * lin_v_dim; }
  int lin_conv_dim() const { return 2 * lin_key_dim() + lin_value_dim(); }
  int attn_dim() const { return heads * head_dim; }
  int rope_dims() const { return (int)(head_dim * partial_rotary); }
};

// A quantized matrix (affine 4-bit). `blocked` matrices use the kernel layout
// described in kernels/linear.metal; otherwise rows are stored contiguously.
struct QTensor {
  id<MTLBuffer> buf = nil;
  size_t w = 0, s = 0, b = 0;  // byte offsets into buf
  int k = 0, n = 0, n_pad = 0;
  bool blocked = true;
  std::map<std::string, std::pair<int, int>> parts;  // name -> (first row, rows)

  int part_offset(const std::string &name) const;
};

struct FTensor {
  id<MTLBuffer> buf = nil;
  size_t offset = 0;
  std::vector<int> shape;
  const float *data = nullptr;
};

class ModelFile {
 public:
  ModelFile(Metal &mt, const std::string &path);
  ~ModelFile();
  ModelFile(const ModelFile &) = delete;

  const Config &config() const { return cfg_; }
  const QTensor &q(const std::string &name) const;
  const FTensor &f(const std::string &name) const;
  size_t file_bytes() const { return size_; }

 private:
  Config cfg_;
  void *map_ = nullptr;
  size_t size_ = 0;
  std::map<std::string, QTensor> q_;
  std::map<std::string, FTensor> f_;
};

}  // namespace rl
