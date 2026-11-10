#include "model_file.h"

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rl {

int QTensor::part_offset(const std::string &name) const {
  auto it = parts.find(name);
  if (it == parts.end()) fatal("quantized tensor has no part '%s'", name.c_str());
  return it->second.first;
}

static int geti(NSDictionary *d, NSString *key) {
  id v = d[key];
  if (![v isKindOfClass:[NSNumber class]]) fatal("model header: missing integer '%s'", key.UTF8String);
  return [v intValue];
}

ModelFile::ModelFile(Metal &mt, const std::string &path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) fatal("cannot open %s", path.c_str());
  struct stat st;
  fstat(fd, &st);
  size_ = (size_t)st.st_size;
  map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map_ == MAP_FAILED) fatal("cannot mmap %s", path.c_str());

  const uint8_t *base = (const uint8_t *)map_;
  if (size_ < 16 || memcmp(base, "RLM1", 4) != 0) fatal("%s is not a redline model file", path.c_str());
  uint64_t hdr_len;
  memcpy(&hdr_len, base + 8, 8);
  NSData *json = [NSData dataWithBytesNoCopy:(void *)(base + 16) length:hdr_len freeWhenDone:NO];
  NSError *err = nil;
  NSDictionary *h = [NSJSONSerialization JSONObjectWithData:json options:0 error:&err];
  if (!h) fatal("model header is not valid JSON: %s", err.localizedDescription.UTF8String);

  NSDictionary *c = h[@"config"];
  cfg_.hidden = geti(c, @"hidden_size");
  cfg_.layers = geti(c, @"num_hidden_layers");
  cfg_.intermediate = geti(c, @"intermediate_size");
  cfg_.heads = geti(c, @"num_attention_heads");
  cfg_.kv_heads = geti(c, @"num_key_value_heads");
  cfg_.head_dim = geti(c, @"head_dim");
  cfg_.lin_v_heads = geti(c, @"linear_num_value_heads");
  cfg_.lin_k_heads = geti(c, @"linear_num_key_heads");
  cfg_.lin_k_dim = geti(c, @"linear_key_head_dim");
  cfg_.lin_v_dim = geti(c, @"linear_value_head_dim");
  cfg_.conv_taps = geti(c, @"linear_conv_kernel_dim");
  cfg_.vocab = geti(c, @"vocab_size");
  cfg_.full_attn_interval = geti(c, @"full_attention_interval");
  cfg_.norm_eps = [c[@"rms_norm_eps"] floatValue];
  cfg_.rope_theta = [c[@"rope_theta"] doubleValue];
  cfg_.partial_rotary = [c[@"partial_rotary_factor"] floatValue];
  if ([c[@"eos_token_id"] isKindOfClass:[NSNumber class]]) cfg_.eos = [c[@"eos_token_id"] intValue];
  NSDictionary *qc = h[@"quant"];
  cfg_.group = geti(qc, @"group");
  cfg_.block = geti(qc, @"block");

  // One no-copy Metal buffer per page-aligned section.
  NSArray *sections = h[@"sections"];
  std::vector<id<MTLBuffer>> bufs;
  std::vector<size_t> sec_off;
  for (NSDictionary *s in sections) {
    size_t off = [s[@"offset"] unsignedLongLongValue], n = [s[@"nbytes"] unsignedLongLongValue];
    if (off + n > Metal::round_up(size_, Metal::page_size())) fatal("section '%s' runs past the end of the file", [s[@"name"] UTF8String]);
    bufs.push_back(mt.wrap((void *)(base + off), n));
    sec_off.push_back(off);
  }

  NSDictionary *tensors = h[@"tensors"];
  for (NSString *name in tensors) {
    NSDictionary *t = tensors[name];
    int sec = geti(t, @"section");
    NSString *kind = t[@"kind"];
    if ([kind isEqualToString:@"f32"]) {
      FTensor ft;
      ft.buf = bufs[sec];
      size_t abs = [t[@"offset"] unsignedLongLongValue];
      ft.offset = abs - sec_off[sec];
      ft.data = (const float *)(base + abs);
      for (NSNumber *d in t[@"shape"]) ft.shape.push_back(d.intValue);
      f_[name.UTF8String] = ft;
    } else {
      QTensor qt;
      qt.buf = bufs[sec];
      qt.blocked = [kind isEqualToString:@"q4a"];
      qt.w = [t[@"w"] unsignedLongLongValue] - sec_off[sec];
      qt.s = [t[@"s"] unsignedLongLongValue] - sec_off[sec];
      qt.b = [t[@"b"] unsignedLongLongValue] - sec_off[sec];
      qt.k = geti(t, @"k");
      qt.n = geti(t, @"n");
      qt.n_pad = qt.blocked ? geti(t, @"n_pad") : qt.n;
      NSDictionary *parts = t[@"parts"];
      for (NSString *pn in parts)
        qt.parts[pn.UTF8String] = {geti(parts[pn], @"offset"), geti(parts[pn], @"n")};
      q_[name.UTF8String] = qt;
    }
  }
}

ModelFile::~ModelFile() {
  if (map_ && map_ != MAP_FAILED) munmap(map_, size_);
}

const QTensor &ModelFile::q(const std::string &name) const {
  auto it = q_.find(name);
  if (it == q_.end()) fatal("model has no quantized tensor '%s'", name.c_str());
  return it->second;
}

const FTensor &ModelFile::f(const std::string &name) const {
  auto it = f_.find(name);
  if (it == f_.end()) fatal("model has no float tensor '%s'", name.c_str());
  return it->second;
}

}  // namespace rl
