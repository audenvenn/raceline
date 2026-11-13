#include "engine.h"

#include "../kernels/params.h"

#import <Accelerate/Accelerate.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <unistd.h>
#include <mutex>
#include <pthread.h>
#include <sys/qos.h>
#include <chrono>
#include <thread>

namespace rl {

struct Engine::LinearLayer {
  id<MTLBuffer> conv[2];  // conv state, [3][conv_dim] float; conv[cur] is current
  id<MTLBuffer> S[2];     // DeltaNet state, [Hv][Dv][Dk] float; S[scur] is current
  int cur = 0, scur = 0;
  // A speculative pass writes its new state next to the old one and keeps what
  // went into the recurrence for each row here, so the state after any prefix
  // of its rows can be rebuilt: pre-conv inputs, conv outputs, q|k, gates.
  id<MTLBuffer> sp_in, sp_y, sp_qk, sp_bg;
};

namespace {
struct MtpState {  // the head's bookkeeping, as saved by save_state
  int32_t mtp_pos, n_tok, tok[Engine::kMaxDrafts + 1], draft_next, draft_rows;
};
constexpr int kSpecRows = Engine::kMaxDrafts + 1;  // rows of a speculative pass
constexpr int kMtpRows = kSpecRows;                // hidden states that can wait for their next token
}  // namespace

struct Engine::AttnLayer {
  id<MTLBuffer> K;        // [KVH][blocks of 256 positions][head_dim][256] half
  id<MTLBuffer> V;        // [KVH][positions][head_dim] half
};

namespace {

template <typename P>
void set_params(id<MTLComputeCommandEncoder> enc, const P &p, int index) {
  [enc setBytes:&p length:sizeof(P) atIndex:index];
}

}  // namespace

// ---- Experts kept on disk, the ones in use in memory (RACELINE_EXPERT_CACHE).
//
// A MoE model's experts are most of its weights, and a token uses a few of
// them per layer. With the cache on, the expert sections of the file are not
// handed to Metal (which would have to keep them all resident): the engine
// copies the experts a pass needs into a fixed set of slots and the grouped
// matmuls read them through a per-layer slot table. Which experts a layer
// needs is known only once its router has run on the GPU, so the command
// buffer is split there: the GPU signals a shared event after the routing
// kernel and waits on the next value before the expert matmuls; a CPU thread
// that watches the event loads what is missing (least recently used slots
// first; each layer's shared expert has a slot of its own), updates the table
// and signals. What is computed does not change, only where the weights are read.
//
// A prompt chunk needs nearly every expert of every layer: a cyclic scan, in
// which least-recently-used replacement evicts exactly the experts the next
// layers are about to need, so every chunk would read every expert. Passes of
// more than kMaxDrafts + 1 rows therefore do not count as uses: their hits stay
// where they are in the recency order and what they load goes in at the cold
// end, first to be replaced (LRU insertion policy). A chunk then reads about
// the experts the cache does not hold, cycling them through the coldest slots,
// and the cache keeps what decoding uses. (Replayed on synthetic agent
// sessions with 73% of the experts cached, bench/expert_cache_sim.py: 58% fewer
// reads for prompts than plain LRU, and 6% fewer while decoding.)
// -1: decide from the memory the GPU may use (Engine::Engine).
static double expert_cache_setting() {
  const char *v = getenv("RACELINE_EXPERT_CACHE");
  return v && *v ? atof(v) : -1.0;
}

struct Engine::ExpertCache {
  struct Request {
    int layer, T;
    uint64_t ready, go;
  };
  int layers = 0, E = 0, S = 0, slots = 0;
  int big_rows = 16;                       // passes with more rows do not count as uses
  int topk = 8;                            // experts routed per token
  bool prefetch_on = true;                 // RACELINE_EXPERT_PREFETCH=0: off
  int busy_layer = -1;                     // the layer the GPU computes now, and what it needs
  std::vector<char> busy_need;
  FILE *trace = nullptr;                   // RACELINE_EXPERT_TRACE: what every layer needed
  size_t w_bytes[2] = {0, 0};
  int s_floats[2] = {0, 0};
  std::vector<size_t> src_w[2], src_s[2];   // file offsets of each layer's stacks: codes, row scales
  int fd = -1;
  id<MTLBuffer> w[2], s[2], table, sel;   // sel: the routing kernel's expert per assignment
  id<MTLBuffer> cols;                      // [layer][I] the down stacks' column scales (for SwiGLU)
  std::vector<std::pair<int, int>> owner;  // slot -> (layer, expert)
  // The cached slots in order of use, hottest first: a doubly linked list
  // through prev/next with a sentinel at index `slots` (pinned slots are not in it).
  std::vector<int> prev, next;
  int next_free = 0;
  std::atomic<uint64_t> loads{0}, hits{0};   // read by expert_stats() on another thread
  id<MTLSharedEvent> ev;
  uint64_t value = 0;
  std::thread worker;
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Request> queue;
  bool stop = false;

  // The GPU's power management lowers its performance when the GPU looks idle,
  // and with experts on disk every MoE layer makes the GPU wait for this
  // thread. Measured on an M6 (#47): single-token passes of Qwen3.6-35B-A3B
  // ran at 32.5 tok/s, and at 51.7 with a light kernel running alongside on
  // another queue. So while passes come (one submitted within the last 50 ms),
  // a thread of its own keeps such a kernel running back to back; it sleeps
  // when generation stops. RACELINE_GPU_KEEPALIVE=0 turns it off.
  id<MTLCommandQueue> ka_queue = nil;
  id<MTLComputePipelineState> ka_pipe = nil;
  id<MTLBuffer> ka_buf = nil;
  std::thread ka_thread;
  std::mutex ka_mu;
  std::condition_variable ka_cv;
  std::atomic<int64_t> ka_last{0};   // steady_clock ns of the last request
  bool ka_stop = false;

  static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void keepalive() {
    constexpr int64_t kIdle = 50'000'000;   // ns without a request before it stops
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(ka_mu);
        ka_cv.wait(lock, [&] { return ka_stop || now_ns() - ka_last.load() < kIdle; });
        if (ka_stop) return;
      }
      while (!ka_stop && now_ns() - ka_last.load() < kIdle) {
        @autoreleasepool {
          id<MTLCommandBuffer> cb = [ka_queue commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:ka_pipe];
          [enc setBuffer:ka_buf offset:0 atIndex:0];
          for (int j = 0; j < 8; j++)
            [enc dispatchThreads:MTLSizeMake(1024, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [enc endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
        }
      }
    }
  }

  int32_t *tab(int layer) const { return (int32_t *)table.contents + (size_t)layer * (E + 1); }

  // One expert into one slot. pread() reads each part in one request (a copy
  // out of the mapping would fault it in 16 KB at a time).
  void load(int layer, int e, int slot) {
    for (int k = 0; k < 2; k++) {
      read_all((char *)w[k].contents + slot * w_bytes[k], w_bytes[k], src_w[k][layer] + e * w_bytes[k]);
      read_all((float *)s[k].contents + (size_t)slot * s_floats[k], s_floats[k] * sizeof(float),
               src_s[k][layer] + (size_t)e * s_floats[k] * sizeof(float));
    }
  }

  void read_all(void *dst, size_t n, size_t off) {
    for (size_t got = 0; got < n;) {
      const ssize_t r = pread(fd, (char *)dst + got, n - got, (off_t)(off + got));
      if (r <= 0) fatal("reading experts from the model file failed");
      got += (size_t)r;
    }
  }

  void unlink(int i) {
    next[prev[i]] = next[i];
    prev[next[i]] = prev[i];
  }
  void insert(int i, bool hot) {             // at the hot or the cold end
    const int a = hot ? slots : prev[slots], b = next[a];
    prev[i] = a, next[i] = b, next[a] = i, prev[b] = i;
  }

  // A free slot, or the coldest one that holds nothing this request needs
  // (nor what the GPU computes with right now, while prefetching); -1 if none.
  int take_slot(int layer, const std::vector<char> &need) {
    if (next_free < slots) return next_free++;
    for (int i = prev[slots]; i != slots; i = prev[i]) {
      const auto [l, e] = owner[i];
      if ((l == layer && need[e]) || (l == busy_layer && busy_need[e])) continue;
      unlink(i);
      tab(l)[e] = -1;
      return i;
    }
    return -1;
  }

  // Reads the missing experts of `need` into slots (in parallel: the SSD serves
  // concurrent requests far faster than one after another). A request always
  // finds slots (the cache holds a layer's worth besides the pinned ones); a
  // prefetch takes what is free while the GPU works and stops there.
  void fill(int layer, const std::vector<char> &need, bool hot, bool prefetching = false) {
    int32_t *t = tab(layer);
    std::vector<std::pair<int, int>> todo;   // (expert, slot)
    for (int e = 0; e < E; e++) {
      if (!need[e] || t[e] >= 0) continue;
      const int slot = take_slot(layer, need);
      if (slot < 0) {
        if (prefetching) break;
        fatal("expert cache: no slot to reuse");
      }
      owner[slot] = {layer, e};
      insert(slot, hot);
      todo.push_back({e, slot});
    }
    if (todo.empty()) return;
    dispatch_apply(todo.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t j) { load(layer, todo[j].first, todo[j].second); });
    for (auto [e, slot] : todo) t[e] = slot;
    loads += todo.size();
  }

  void handle(const Request &r) {
    const int32_t *pick = (const int32_t *)sel.contents;
    int32_t *t = tab(r.layer);
    std::vector<char> need(E, 0);
    for (int a = 0; a < r.T * S; a++)
      if (pick[a] < E) need[pick[a]] = 1;
    if (trace) {
      std::vector<uint8_t> bits((E + 7) / 8, 0);
      for (int e = 0; e < E; e++) bits[e / 8] |= need[e] << (e % 8);
      const int32_t head[2] = {r.layer, r.T};
      fwrite(head, sizeof head, 1, trace);
      fwrite(bits.data(), 1, bits.size(), trace);
    }
    const bool big = r.T > big_rows;
    for (int e = 0; e < E; e++)
      if (need[e] && t[e] >= 0) {
        hits++;
        if (!big) unlink(t[e]), insert(t[e], true);
      }
    busy_layer = -1;
    fill(r.layer, need, !big);
    busy_layer = r.layer;                    // from the signal on, the GPU computes with these
    busy_need.swap(need);
  }

  // While the GPU computes a prompt chunk's layer, the next layer's experts
  // that are not cached: a chunk of 4 * E / topk rows or more needs about 98%
  // of them, and reading them now overlaps the reads with the GPU's work.
  void prefetch(const Request &r) {
    if (!prefetch_on || (int64_t)r.T * topk < 4LL * E) return;
    fill(r.layer, std::vector<char>(E, 1), false, true);
  }

  void run() {
    // The GPU waits on this thread once per MoE layer: keep it on a performance core.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for (;;) {
      Request r;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return stop || !queue.empty(); });
        if (stop) return;
        r = queue.front();
        queue.pop_front();
      }
      while (ev.signaledValue < r.ready) {   // the GPU gets there within the pass: spin
        if (stop) return;
      }
      handle(r);
      ev.signaledValue = r.go;
      Request nxt{-1, 0, 0, 0};
      {
        std::lock_guard<std::mutex> lock(mu);
        if (!queue.empty()) nxt = queue.front();
      }
      if (nxt.layer == r.layer + 1 && nxt.T == r.T) prefetch(nxt);
    }
  }

  void submit(const Request &r) {
    {
      std::lock_guard<std::mutex> lock(mu);
      queue.push_back(r);
    }
    cv.notify_one();
    if (ka_pipe) {
      ka_last = now_ns();
      ka_cv.notify_one();
    }
  }

  ~ExpertCache() {
    {
      std::lock_guard<std::mutex> lock(mu);
      stop = true;
    }
    cv.notify_one();
    {
      std::lock_guard<std::mutex> lock(ka_mu);
      ka_stop = true;
    }
    ka_cv.notify_one();
    if (ka_thread.joinable()) ka_thread.join();
    if (worker.joinable()) worker.join();   // before the file it reads from is closed
    if (fd >= 0) close(fd);
    if (trace) fclose(trace);
  }
};

Engine::~Engine() = default;

Engine::ExpertStats Engine::expert_stats() const {
  if (!cache_) return {0, 0, 0, 0, 0};
  const ExpertCache &x = *cache_;
  const size_t slot_bytes = x.w_bytes[0] + x.w_bytes[1] + (x.s_floats[0] + x.s_floats[1]) * sizeof(float);
  return {x.slots, x.layers * (x.E + 1), slot_bytes, x.hits, x.loads};
}

Engine::Engine(const std::string &model_path, int max_ctx, int max_chunk)
    : model_(new ModelFile(mt_, model_path, false)), max_ctx_(max_ctx), max_chunk_(max_chunk) {
  const Config &c = config();
  // Experts in memory, or on disk with a cache: the cache when the whole model,
  // its attention cache and scratch would not fit in 85% of what Metal
  // recommends the GPU use, or when RACELINE_EXPERT_CACHE says so (0: never).
  double cache_gb = 0;
  if (c.is_moe()) {
    const double setting = expert_cache_setting();
    const double budget = 0.85 * (double)mt_.device().recommendedMaxWorkingSetSize;
    const double experts = (double)model_->expert_bytes(), rest = (double)model_->file_bytes() - experts;
    int attn_layers = 1;  // the draft head's
    for (int i = 0; i < c.layers; i++) attn_layers += !c.is_linear_layer(i);
    const double kv = 2.0 * attn_layers * c.kv_heads * c.head_dim * ((max_ctx + 255) / 256 * 256) * sizeof(_Float16);
    const double scratch = 1.5e9;
    if (setting > 0) cache_gb = setting;
    else if (setting < 0 && rest + experts + kv + scratch > budget) {
      cache_gb = (budget - rest - kv - scratch) / 1e9;
      if (cache_gb < 1) fatal("this model needs about %.1f GB of GPU memory outside its experts; this Mac allows %.1f",
                              (rest + kv + scratch) / 1e9, budget / 1e9);
    }
  }
  if (cache_gb <= 0) model_->map_expert_sections(mt_);
  if (c.lin_k_dim != 128 || c.head_dim != 256 || c.rope_dims() != 64 || c.conv_taps != 4 || c.group != 64 ||
      c.block != 256 || c.lin_v_dim % 8 != 0 || c.heads % c.kv_heads != 0)
    fatal("unsupported model geometry (kernels assume Dk=128, head_dim=256, 64 rotary dims, 4 conv taps)");

  has_mtp_ = mtp_ = c.native && model_->has("mtp.fc");
  const int n_layers = c.layers + (has_mtp_ ? 1 : 0);  // the head's decoder layer is layer c.layers

  // A head pass can carry the parked rows of a speculative pass on top of a chunk.
  rows_cap_ = (int)Metal::round_up((size_t)max_chunk_ + (has_mtp_ ? kMtpRows : 0), 128);
  x_stride_ = std::max({c.hidden, c.intermediate, c.lin_value_dim(), c.attn_dim(), has_mtp_ ? 2 * c.hidden : 0});
  proj_stride_ = 0;
  for (int i = 0; i < n_layers; i++) {
    if (!c.is_moe()) proj_stride_ = std::max(proj_stride_, model_->q(lname(i, "mlp.gate_up")).n_pad);
    proj_stride_ = std::max(proj_stride_, model_->q(lname(i, is_linear(i) ? "lin.in_proj" : "attn.qkv")).n_pad);
  }

  const size_t R = (size_t)rows_cap_;
  tok_ = mt_.alloc(R * sizeof(int32_t));
  h_ = mt_.alloc(R * c.hidden * sizeof(float));
  xn_ = mt_.alloc(R * x_stride_ * sizeof(_Float16));
  gsum_ = mt_.alloc(R * (x_stride_ / c.group) * sizeof(float));
  proj_ = mt_.alloc(R * proj_stride_ * sizeof(float));
  y_ = mt_.alloc(R * c.lin_conv_dim() * sizeof(float));
  qk_ = mt_.alloc(R * 2 * c.lin_key_dim() * sizeof(float));
  bg_ = mt_.alloc(R * 2 * c.lin_v_heads * sizeof(float));
  yo_ = mt_.alloc(R * c.lin_value_dim() * sizeof(float));
  const int G = c.heads / c.kv_heads;
  kv_blocks_ = (max_ctx_ + 255) / 256;
  rpos_.assign((size_t)3 * (kv_blocks_ * 256 + kSpecRows), 0);
  q_rows_ = rows_cap_ * G + 32;
  aq_ = mt_.alloc((size_t)c.kv_heads * q_rows_ * c.head_dim * sizeof(_Float16));
  al_ = mt_.alloc((size_t)c.kv_heads * q_rows_ * sizeof(float));
  // Scores for a group of rows: [KVH][rows * G][keys]. A decode step touches a
  // sliver of this; a prefill chunk at long context is cut into groups that fit.
  score_cap_ = std::max((size_t)24 << 20, (size_t)c.kv_heads * 64 * kv_blocks_ * 256);
  as_ = mt_.alloc(score_cap_ * sizeof(float));
  ap_ = mt_.alloc(score_cap_ * sizeof(_Float16));
  rope_ = mt_.alloc(R * 64 * sizeof(float));
  logits_ = mt_.alloc((size_t)16 * model_->q("lm_head").n_pad * sizeof(float));
  if (c.is_moe()) {
    const size_t S = c.experts_per_tok + 1, A = R * S, tm = 16;
    const int router = model_->q(lname(0, "moe.router")).n_pad;
    moe_logits_ = mt_.alloc(R * router * sizeof(float));
    moe_tok_ = mt_.alloc(A * sizeof(int32_t));
    moe_inv_ = mt_.alloc(A * sizeof(int32_t));
    moe_w_ = mt_.alloc(A * sizeof(float));
    moe_sel_ = mt_.alloc(A * sizeof(int32_t));
    moe_items_ = mt_.alloc((A / tm + c.experts + 2) * 4 * sizeof(int32_t));
    moe_args_ = mt_.alloc(6 * sizeof(uint32_t));
    // The grouped matmuls read whole tiles of TM rows: room for one past the last row.
    moe_x_ = mt_.alloc((A + tm) * c.hidden * sizeof(_Float16));
    moe_gu_ = mt_.alloc(A * 2 * c.moe_inter * sizeof(float));
    moe_act_ = mt_.alloc((A + tm) * c.moe_inter * sizeof(_Float16));
    moe_y_ = mt_.alloc(A * c.hidden * sizeof(float));
    moe_identity_ = mt_.alloc((c.experts + 1) * sizeof(int32_t));
    for (int e = 0; e <= c.experts; e++) ((int32_t *)moe_identity_.contents)[e] = e;

    if (cache_gb > 0) {
      cache_ = std::make_unique<ExpertCache>();
      ExpertCache &x = *cache_;
      x.layers = c.layers;
      x.E = c.experts;
      x.S = (int)S;
      x.fd = open(model_->path().c_str(), O_RDONLY);
      if (x.fd < 0) fatal("cannot open %s", model_->path().c_str());
      // For measuring a Mac with less memory on one with more: read experts from
      // the SSD every time, past the system's file cache.
      if (getenv("RACELINE_EXPERT_NOCACHE") && atoi(getenv("RACELINE_EXPERT_NOCACHE"))) fcntl(x.fd, F_NOCACHE, 1);
      for (int kind = 0; kind < 2; kind++) {
        for (int i = 0; i < c.layers; i++) {
          const QTensor &t = model_->q(lname(i, kind ? "moe.down" : "moe.gate_up"));
          const int ne = kind ? c.hidden : 2 * c.moe_inter;
          x.w_bytes[kind] = (size_t)ne * t.k / 2;
          x.s_floats[kind] = ne;
          x.src_w[kind].push_back(model_->file_offset(t.host + t.w));
          x.src_s[kind].push_back(model_->file_offset(t.host + t.s));
        }
      }
      const size_t per_slot = x.w_bytes[0] + x.w_bytes[1] + (x.s_floats[0] + x.s_floats[1]) * sizeof(float);
      // At least the shared experts and one layer's worth; at most every expert.
      x.slots = std::min(std::max((int)(cache_gb * 1e9 / per_slot), c.layers + c.experts), c.layers * (c.experts + 1));
      x.big_rows = kMaxDrafts + 1;
      x.topk = c.experts_per_tok;
      if (const char *v = getenv("RACELINE_EXPERT_PREFETCH"); v && *v && !atoi(v)) x.prefetch_on = false;
      if (const char *path = getenv("RACELINE_EXPERT_TRACE"); path && *path) x.trace = fopen(path, "wb");
      for (int kind = 0; kind < 2; kind++) {
        x.w[kind] = mt_.alloc((size_t)x.slots * x.w_bytes[kind]);
        x.s[kind] = mt_.alloc((size_t)x.slots * x.s_floats[kind] * sizeof(float));
      }
      x.cols = mt_.alloc((size_t)c.layers * c.moe_inter * sizeof(float));
      for (int i = 0; i < c.layers; i++) {
        const QTensor &dn = model_->q(lname(i, "moe.down"));
        memcpy((float *)x.cols.contents + (size_t)i * c.moe_inter, dn.host + dn.c, c.moe_inter * sizeof(float));
      }
      x.table = mt_.alloc((size_t)c.layers * (c.experts + 1) * sizeof(int32_t));
      memset(x.table.contents, 0xff, x.table.length);   // -1: not loaded
      x.owner.assign(x.slots, {-1, -1});
      x.prev.assign(x.slots + 1, x.slots);
      x.next.assign(x.slots + 1, x.slots);
      for (int i = 0; i < c.layers; i++) {             // each layer's shared expert: a pinned slot
        x.load(i, c.experts, i);
        x.tab(i)[c.experts] = i;
        x.owner[i] = {i, c.experts};
      }
      x.next_free = c.layers;
      x.sel = moe_sel_;
      x.ev = [mt_.device() newSharedEvent];
      x.worker = std::thread([&x] { x.run(); });
      if (const char *v = getenv("RACELINE_GPU_KEEPALIVE"); !(v && *v && !atoi(v))) {
        x.ka_queue = [mt_.device() newCommandQueue];
        x.ka_pipe = mt_.pipeline("keepalive.metal", "keepalive");
        x.ka_buf = mt_.alloc(1024 * sizeof(float));
        x.ka_thread = std::thread([&x] { x.keepalive(); });
      }
    }
  }

  ones_ = mt_.alloc((size_t)x_stride_ * sizeof(float));
  std::fill_n((float *)ones_.contents, x_stride_, 1.0f);
  if (c.native) {
    if (c.book != "fp4" && c.book != "int15") fatal("codebook '%s' is not supported by this build", c.book.c_str());
    // Norm weights times the column scales (from column col0 on) of the matrix they feed.
    auto fold = [&](const FTensor &norm, const QTensor &consumer, int col0 = 0) {
      id<MTLBuffer> b = mt_.alloc((size_t)c.hidden * sizeof(float));
      const float *cs = consumer.native ? (const float *)(consumer.host + consumer.c) + col0 : nullptr;
      float *o = (float *)b.contents;
      for (int k = 0; k < c.hidden; k++) o[k] = norm.data[k] * (cs ? cs[k] : 1.0f);
      return b;
    };
    for (int i = 0; i < n_layers; i++) {
      norm_in_.push_back(fold(model_->f(lname(i, "in_norm")),
                              model_->q(lname(i, is_linear(i) ? "lin.in_proj" : "attn.qkv"))));
      norm_post_.push_back(fold(model_->f(lname(i, "post_norm")), model_->q(mlp_in(i))));
    }
    norm_final_ = fold(model_->f("final_norm"), model_->q("lm_head"));
    if (has_mtp_) {
      mtp_norm_e_ = fold(model_->f("mtp.norm_e"), model_->q("mtp.fc"), 0);
      mtp_norm_h_ = fold(model_->f("mtp.norm_h"), model_->q("mtp.fc"), c.hidden);
      mtp_norm_out_ = fold(model_->f("mtp.norm"), model_->q("lm_head"));
      if (c.rot_block) {
        if ((c.rot_block & (c.rot_block - 1)) || c.hidden % c.rot_block || c.hidden > 8192)
          fatal("rotation block %d does not fit the hidden size %d", c.rot_block, c.hidden);
        const float *sg = model_->f("rotation.signs").data, *wf = model_->f("final_norm.unfolded").data;
        const float *wm = model_->f("mtp.norm").data;
        const float norm = 1.0f / sqrtf((float)c.rot_block);
        mtp_rot_w1_ = mt_.alloc((size_t)c.hidden * sizeof(float));
        mtp_rot_g_ = mt_.alloc((size_t)c.hidden * sizeof(float));
        float *w1 = (float *)mtp_rot_w1_.contents, *g = (float *)mtp_rot_g_.contents;
        for (int k = 0; k < c.hidden; k++) {
          w1[k] = wf[k] * sg[k] * norm;
          g[k] = wm[k] / wf[k] * sg[k] * norm;
        }
      }
      mtp_h_ = mt_.alloc((size_t)kMtpRows * c.hidden * sizeof(float));
      mtp_out_ = mt_.alloc((size_t)c.hidden * sizeof(float));
      mtp_tok_ = mt_.alloc(R * sizeof(int32_t));
      mtp_rope_ = mt_.alloc(R * 64 * sizeof(float));
      mtp_logits_ = mt_.alloc((size_t)16 * model_->q("lm_head").n_pad * sizeof(float));
    }
  }

  lin_.resize(n_layers);
  attn_.resize(n_layers);
  for (int i = 0; i < n_layers; i++) {
    if (is_linear(i)) {
      auto L = std::make_shared<LinearLayer>();
      for (auto &b : L->conv) b = mt_.alloc((size_t)3 * c.lin_conv_dim() * sizeof(float));
      L->S[0] = mt_.alloc((size_t)c.lin_v_heads * c.lin_v_dim * c.lin_k_dim * sizeof(float));
      if (has_mtp_) {
        L->S[1] = mt_.alloc(L->S[0].length);
        L->sp_in = mt_.alloc((size_t)kSpecRows * c.lin_conv_dim() * sizeof(float));
        L->sp_y = mt_.alloc((size_t)kSpecRows * c.lin_conv_dim() * sizeof(float));
        L->sp_qk = mt_.alloc((size_t)kSpecRows * 2 * c.lin_key_dim() * sizeof(float));
        L->sp_bg = mt_.alloc((size_t)kSpecRows * 2 * c.lin_v_heads * sizeof(float));
      }
      lin_[i] = L;
    } else {
      auto A = std::make_shared<AttnLayer>();
      size_t bytes = (size_t)c.kv_heads * kv_blocks_ * 256 * c.head_dim * sizeof(_Float16);
      A->K = mt_.alloc(bytes);
      A->V = mt_.alloc(bytes);
      attn_[i] = A;
    }
  }
}

std::string Engine::lname(int layer, const char *suffix) const {
  if (layer == config().layers) return std::string("mtp.") + suffix;
  return "layers." + std::to_string(layer) + "." + suffix;
}

void Engine::reset() {
  pos_ = 0;
  for (auto &L : lin_) {
    if (!L) continue;
    for (auto &b : L->conv) memset(b.contents, 0, b.length);
    memset(L->S[0].contents, 0, L->S[0].length);
    L->cur = L->scur = 0;
  }
  mtp_pos_ = 0;
  draft_next_ = -1;
  draft_rows_ = draft_depth_ = 0;
  pend_tok_.clear();
  spec_rows_ = 0;
}

void Engine::set_mtp(bool on) {
  if (on && !has_mtp_) fatal("set_mtp: this model file has no MTP head (tools/add_mtp.py adds one)");
  if (pos_ != 0) fatal("set_mtp: only at position 0");
  mtp_ = on;
  mtp_pos_ = 0;
  draft_next_ = -1;
  draft_rows_ = draft_depth_ = 0;
  pend_tok_.clear();
}

size_t Engine::state_bytes() const {
  size_t n = has_mtp_ ? sizeof(MtpState) + mtp_h_.length : 0;
  for (const auto &L : lin_)
    if (L) n += L->conv[0].length + L->S[0].length;
  return n;
}

void Engine::save_state(void *dst) const {
  if (spec_rows_) fatal("save_state: drafts are open (keep_drafts first)");
  char *p = (char *)dst;
  for (const auto &L : lin_) {
    if (!L) continue;
    memcpy(p, L->conv[L->cur].contents, L->conv[0].length);
    p += L->conv[0].length;
    memcpy(p, L->S[L->scur].contents, L->S[0].length);
    p += L->S[0].length;
  }
  if (has_mtp_) {
    MtpState st{};
    st.mtp_pos = mtp_pos_;
    st.n_tok = (int32_t)pend_tok_.size();
    for (size_t i = 0; i < pend_tok_.size(); i++) st.tok[i] = pend_tok_[i];
    st.draft_next = draft_next_;
    st.draft_rows = draft_rows_;
    memcpy(p, &st, sizeof st);
    memcpy(p + sizeof st, mtp_h_.contents, mtp_h_.length);
  }
}

void Engine::set_mark(int row) {
  if (row < 0) fatal("set_mark: row %d", row);
  mark_row_ = row;
  if (row == 0 || mark_h_) return;
  const Config &c = config();
  mark_conv_.assign(lin_.size(), nil);
  mark_S_.assign(lin_.size(), nil);
  for (size_t i = 0; i < lin_.size(); i++) {
    if (!lin_[i]) continue;
    mark_conv_[i] = mt_.alloc(lin_[i]->conv[0].length);
    mark_S_[i] = mt_.alloc(lin_[i]->S[0].length);
  }
  mark_h_ = mt_.alloc((size_t)c.hidden * sizeof(float));
}

// The layout of save_state, from the mark's buffers.
void Engine::save_mark(void *dst) const {
  if (mark_pos_ < 0) fatal("save_mark: no mark was kept (set_mark before a forward)");
  char *p = (char *)dst;
  for (size_t i = 0; i < lin_.size(); i++) {
    if (!lin_[i]) continue;
    memcpy(p, mark_conv_[i].contents, lin_[i]->conv[0].length);
    p += lin_[i]->conv[0].length;
    memcpy(p, mark_S_[i].contents, lin_[i]->S[0].length);
    p += lin_[i]->S[0].length;
  }
  if (has_mtp_) {
    MtpState st{};
    st.mtp_pos = mtp_ ? mark_pos_ - 1 : 0;   // the mark's last row waits for its next token,
    st.n_tok = 0;                            // as after a pass that ended at the mark
    st.draft_next = -1;
    st.draft_rows = draft_rows_;
    memcpy(p, &st, sizeof st);
    memset(p + sizeof st, 0, mtp_h_.length);
    if (mtp_) memcpy(p + sizeof st, mark_h_.contents, (size_t)config().hidden * sizeof(float));
  }
}

void Engine::load_state(const void *src, int position) {
  if (position < 0 || position > max_ctx_) fatal("load_state: position %d out of range", position);
  const char *p = (const char *)src;
  for (auto &L : lin_) {
    if (!L) continue;
    memcpy(L->conv[L->cur].contents, p, L->conv[0].length);
    p += L->conv[0].length;
    memcpy(L->S[L->scur].contents, p, L->S[0].length);
    p += L->S[0].length;
  }
  if (has_mtp_) {
    MtpState st;
    memcpy(&st, p, sizeof st);
    memcpy(mtp_h_.contents, p + sizeof st, mtp_h_.length);
    mtp_pos_ = mtp_ ? st.mtp_pos : 0;
    pend_tok_.assign(st.tok, st.tok + st.n_tok);
    draft_next_ = st.draft_next;
    draft_rows_ = st.draft_rows;
    draft_depth_ = 0;
  }
  pos_ = position;
  spec_rows_ = 0;
}

// K is [KVH][blocks][head_dim][256] and V [KVH][positions][head_dim]: per head,
// a run of whole blocks is one contiguous range of each.
size_t Engine::kv_bytes(int from, int to) const {
  if (from < 0 || to < from || to > max_ctx_) fatal("kv_bytes: positions %d..%d out of range", from, to);
  const Config &c = config();
  const size_t blocks = (size_t)((to + 255) / 256 - from / 256);
  size_t layers = 0;
  for (const auto &A : attn_) layers += A != nullptr;
  return layers * 2 * c.kv_heads * blocks * 256 * c.head_dim * sizeof(_Float16) + blocks * 256 * 3 * sizeof(int32_t);
}

void Engine::save_kv(int from, int to, void *dst) const {
  if (spec_rows_) fatal("save_kv: drafts are open (keep_drafts first)");
  kv_bytes(from, to);  // range check
  const Config &c = config();
  const int b0 = from / 256, b1 = (to + 255) / 256;
  const size_t run = (size_t)(b1 - b0) * 256 * c.head_dim * sizeof(_Float16);
  const size_t head = (size_t)kv_blocks_ * 256 * c.head_dim * sizeof(_Float16);
  const size_t skip = (size_t)b0 * 256 * c.head_dim * sizeof(_Float16);
  char *p = (char *)dst;
  for (const auto &A : attn_) {
    if (!A) continue;
    for (int h = 0; h < c.kv_heads; h++) {
      memcpy(p, (const char *)A->K.contents + h * head + skip, run);
      memcpy(p + run, (const char *)A->V.contents + h * head + skip, run);
      p += 2 * run;
    }
  }
  memcpy(p, &rpos_[(size_t)b0 * 256 * 3], (size_t)(b1 - b0) * 256 * 3 * sizeof(int32_t));
}

void Engine::load_kv(int from, int to, const void *src) {
  if (spec_rows_) fatal("load_kv: drafts are open (keep_drafts first)");
  kv_bytes(from, to);
  const Config &c = config();
  const int b0 = from / 256, b1 = (to + 255) / 256;
  const size_t run = (size_t)(b1 - b0) * 256 * c.head_dim * sizeof(_Float16);
  const size_t head = (size_t)kv_blocks_ * 256 * c.head_dim * sizeof(_Float16);
  const size_t skip = (size_t)b0 * 256 * c.head_dim * sizeof(_Float16);
  const char *p = (const char *)src;
  for (auto &A : attn_) {
    if (!A) continue;
    for (int h = 0; h < c.kv_heads; h++) {
      memcpy((char *)A->K.contents + h * head + skip, p, run);
      memcpy((char *)A->V.contents + h * head + skip, p + run, run);
      p += 2 * run;
    }
  }
  memcpy(&rpos_[(size_t)b0 * 256 * 3], p, (size_t)(b1 - b0) * 256 * 3 * sizeof(int32_t));
}

void Engine::keep_drafts(int accepted) {
  const Config &c = config();
  if (!spec_rows_) fatal("keep_drafts: the last call was not forward_draft");
  const int rows = spec_rows_, keep = accepted + 1;
  if (accepted < 0 || keep > rows) fatal("keep_drafts: %d of %d drafts", accepted, rows - 1);
  spec_rows_ = 0;
  gpu_seconds_ = 0;
  if (keep == rows) return;  // every row stands, and the new state is already current

  // Rebuild the recurrent state as it is after `keep` rows. Conv windows: the
  // last three of (old window, kept inputs). DeltaNet: rerun the recurrence on
  // the kept rows from the old state. Attention entries of the dropped rows
  // are simply overwritten by the tokens that come next.
  const size_t C = (size_t)c.lin_conv_dim(), row = C * sizeof(float);
  const int kd = c.lin_key_dim(), vd = c.lin_value_dim();
  @autoreleasepool {
    begin();
    if (!scratch_yo_) scratch_yo_ = mt_.alloc(yo_.length);
    for (auto &L : lin_) {
      if (!L) continue;
      char *now = (char *)L->conv[L->cur].contents;
      const char *old = (const char *)L->conv[1 - L->cur].contents, *in = (const char *)L->sp_in.contents;
      for (int j = 0; j < 3; j++) {
        const int idx = keep - 3 + j;
        memcpy(now + j * row, idx >= 0 ? in + idx * row : old + (3 + idx) * row, row);
      }
      [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "gated_delta")];
      [enc_ setBuffer:L->sp_qk offset:0 atIndex:0];
      [enc_ setBuffer:L->sp_y offset:0 atIndex:1];
      [enc_ setBuffer:L->sp_bg offset:0 atIndex:2];
      [enc_ setBuffer:L->S[1 - L->scur] offset:0 atIndex:3];
      [enc_ setBuffer:scratch_yo_ offset:0 atIndex:4];
      set_params(enc_, DeltaParams{keep, c.lin_k_heads, c.lin_k_dim, c.lin_v_heads, c.lin_v_dim, 2 * kd, (int)C, 2 * kd,
                                   2 * c.lin_v_heads, vd, 0}, 5);
      [enc_ setBuffer:L->S[L->scur] offset:0 atIndex:6];
      [enc_ setBuffer:L->S[L->scur] offset:0 atIndex:7];  // no mark: never written
      run_groups(c.lin_v_dim / 8, c.lin_v_heads);
    }
    flush(false);
  }
  pos_ -= rows - keep;
  pend_tok_.resize(keep - 1);  // the kept rows wait for the head; the last one's next token is unknown
}

const float *Engine::logits(int row) const {
  return (const float *)logits_.contents + (size_t)row * model_->q("lm_head").n_pad;
}

Engine::Tile Engine::tile_for(int rows) const {
  // From tests/test_linear --sweep and end-to-end runs on M6. Tiles taller
  // than 32 rows are ~3x slower in this kernel: the in-register scale step
  // does not scale up with the tile.
  if (rows <= 16) return {16, 64, 2};
  return {32, 64, 2};
}

void Engine::encode_linear(const QTensor &w, int rows, id<MTLBuffer> out, size_t out_offset, int out_stride,
                           bool accum, const char *what) {
  const Tile t = tile_for(rows);
  const int M = (rows + t.tm - 1) / t.tm * t.tm;

  if (w.dense) {
    const Macros mac = {{"TM", std::to_string(t.tm)}, {"TN", std::to_string(t.tn)}, {"NSG", std::to_string(t.nsg)},
                        {"GROUP", "64"}, {"BLOCK", "256"}, {"OUT_T", "float"}, {"ACCUM", accum ? "1" : "0"}};
    [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_f16", mac)];
    [enc_ setBuffer:xn_ offset:0 atIndex:0];
    [enc_ setBuffer:w.buf offset:w.w atIndex:1];
    [enc_ setBuffer:out offset:out_offset atIndex:4];
    set_params(enc_, LinearParams{M, w.k, w.n_pad, x_stride_, out_stride, 0}, 5);
    run_groups(w.n_pad / t.tn, M / t.tm, 32 * t.nsg);
    mark(what);
    return;
  }

  if (w.native) {
    // One hardware matmul over the whole input, row scale applied in registers.
    Macros mac = {{"TM", std::to_string(t.tm)}, {"TN", std::to_string(t.tn)}, {"NSG", std::to_string(t.nsg)},
                  {"GROUP", "64"}, {"BLOCK", "256"}, {"OUT_T", "float"}, {"ACCUM", accum ? "1" : "0"},
                  {"W_T", config().book == "fp4" ? "metal_fp4_e2m1_format" : "int4b_format"}};
    LinearParams p{M, w.k, w.n_pad, x_stride_, out_stride, 0};
    [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_q4n", mac)];
    [enc_ setBuffer:xn_ offset:0 atIndex:0];
    [enc_ setBuffer:w.buf offset:w.w atIndex:1];
    [enc_ setBuffer:w.buf offset:w.s atIndex:2];
    [enc_ setBuffer:gsum_ offset:0 atIndex:3];  // only read by the half-step codebook
    [enc_ setBuffer:out offset:out_offset atIndex:4];
    set_params(enc_, p, 5);
    run_groups(w.n_pad / t.tn, M / t.tm, 32 * t.nsg);
    if (accum && repeating(what)) {  // see below: a repeat must not accumulate twice
      mac["ACCUM"] = "0";
      p.c_stride = proj_stride_;
      [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_q4n", mac)];
      [enc_ setBuffer:proj_ offset:0 atIndex:4];
      set_params(enc_, p, 5);
    }
    mark(what);
    return;
  }

  const Macros mac = {{"TM", std::to_string(t.tm)}, {"TN", std::to_string(t.tn)}, {"NSG", std::to_string(t.nsg)},
                      {"GROUP", "64"}, {"BLOCK", "256"}, {"OUT_T", "float"}, {"ACCUM", accum ? "1" : "0"}};
  const LinearParams p{M, w.k, w.n_pad, x_stride_, out_stride, w.k / 64};

  [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "group_sums", mac)];
  [enc_ setBuffer:xn_ offset:0 atIndex:0];
  [enc_ setBuffer:gsum_ offset:0 atIndex:1];
  set_params(enc_, p, 2);
  run_threads(p.G, M);
  mark("group_sums");

  [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_q4a", mac)];
  [enc_ setBuffer:xn_ offset:0 atIndex:0];
  [enc_ setBuffer:w.buf offset:w.w atIndex:1];
  [enc_ setBuffer:w.buf offset:w.s atIndex:2];
  [enc_ setBuffer:w.buf offset:w.b atIndex:3];
  [enc_ setBuffer:gsum_ offset:0 atIndex:4];
  [enc_ setBuffer:out offset:out_offset atIndex:5];
  set_params(enc_, p, 6);
  run_groups(w.n_pad / t.tn, M / t.tm, 32 * t.nsg);
  if (accum && repeating(what)) {
    // A repeat must not add into the residual stream twice: it reruns into scratch.
    Macros plain = mac;
    plain["ACCUM"] = "0";
    LinearParams p2 = p;
    p2.c_stride = proj_stride_;
    [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_q4a", plain)];
    [enc_ setBuffer:proj_ offset:0 atIndex:5];
    set_params(enc_, p2, 6);
  }
  mark(what);
}

void Engine::encode_norm(int which, int layer, int rows, size_t in_offset) {
  const Config &c = config();
  if (c.native) {
    encode_rms(h_, in_offset, c.hidden, which == 0 ? norm_in_[layer] : which == 1 ? norm_post_[layer] : norm_final_, 0,
               rows, 0);
  } else {
    const FTensor &f = which == 2 ? model_->f("final_norm") : model_->f(lname(layer, which == 0 ? "in_norm" : "post_norm"));
    encode_rms(h_, in_offset, c.hidden, f.buf, f.offset, rows, 0);
  }
}

// Weighted RMS norm of `rows` rows of hidden floats into xn_ (from byte out_offset on).
void Engine::encode_rms(id<MTLBuffer> in, size_t in_offset, int in_stride, id<MTLBuffer> wgt, size_t wgt_offset,
                        int rows, size_t out_offset) {
  const Config &c = config();
  // A decode step pays for per-kernel latency, so the row is spread over 16
  // SIMD-groups; a prefill chunk has enough rows to fill the GPU with one
  // SIMD-group each, and launching 16x the threads only costs time there.
  const int sg = rows <= 16 ? 16 : 1;
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "rms_norm_rows", {{"NORM_SG", std::to_string(sg)}})];
  [enc_ setBuffer:in offset:in_offset atIndex:0];
  [enc_ setBuffer:wgt offset:wgt_offset atIndex:1];
  [enc_ setBuffer:xn_ offset:out_offset atIndex:2];
  set_params(enc_, NormParams{rows, c.hidden, in_stride, x_stride_, c.norm_eps}, 3);
  run_groups(1, rows, 32 * sg);
  mark("rms_norm");
}

void Engine::bind_column_scale(const QTensor &consumer, int index) {
  if (consumer.native) [enc_ setBuffer:consumer.buf offset:consumer.c atIndex:index];
  else [enc_ setBuffer:ones_ offset:0 atIndex:index];
}

std::string Engine::mlp_in(int layer) const { return lname(layer, config().is_moe() ? "moe.gate_up" : "mlp.gate_up"); }

void Engine::encode_mlp(int layer, int T) {
  const Config &c = config();
  if (c.is_moe()) return encode_moe(layer, T);
  encode_norm(1, layer, T, 0);
  encode_linear(model_->q(lname(layer, "mlp.gate_up")), T, proj_, 0, proj_stride_, false, "linear:gate_up");

  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "swiglu")];
  [enc_ setBuffer:proj_ offset:0 atIndex:0];
  [enc_ setBuffer:xn_ offset:0 atIndex:1];
  set_params(enc_, SwigluParams{T, c.intermediate, proj_stride_, x_stride_}, 2);
  bind_column_scale(model_->q(lname(layer, "mlp.down")), 3);
  run_threads(c.intermediate, T);
  mark("swiglu");

  encode_linear(model_->q(lname(layer, "mlp.down")), T, h_, 0, c.hidden, true, "linear:down");
}

// A mixture-of-experts MLP: router -> routing (top k, counting sort by expert,
// work list) -> gather the routed rows -> every expert's gate/up over its rows
// in one indirect dispatch -> SwiGLU -> the same for down -> each token's
// weighted sum added to the residual stream. The shared expert is expert
// number E with its own gate, so it runs through the same kernels.
void Engine::encode_moe(int layer, int T) {
  const Config &c = config();
  // Rows per tile of the grouped matmuls: an expert's weights are read once
  // per tile, and a long prompt sends an expert dozens of rows. A row's result
  // does not depend on the tile it is computed in.
  const int S = c.experts_per_tok + 1, A = T * S, TM = A / c.experts >= 24 ? 32 : 16, TN = 64, NSG = 2;
  const QTensor &router = model_->q(lname(layer, "moe.router"));
  const QTensor &gu = model_->q(lname(layer, "moe.gate_up"));
  const QTensor &dn = model_->q(lname(layer, "moe.down"));

  encode_norm(1, layer, T, 0);  // xn_: the normed rows times the gate/up stack's column scales
  encode_linear(router, T, moe_logits_, 0, router.n_pad, false, "moe:router");

  const Macros route = {{"TOPK", std::to_string(c.experts_per_tok)},
                        {"E_MAX", std::to_string((c.experts + 31) / 32 * 32)}, {"TM", std::to_string(TM)}};
  [enc_ setComputePipelineState:mt_.pipeline("moe.metal", "moe_route", route)];
  [enc_ setBuffer:moe_logits_ offset:0 atIndex:0];
  [enc_ setBuffer:moe_tok_ offset:0 atIndex:1];
  [enc_ setBuffer:moe_inv_ offset:0 atIndex:2];
  [enc_ setBuffer:moe_w_ offset:0 atIndex:3];
  [enc_ setBuffer:moe_items_ offset:0 atIndex:4];
  [enc_ setBuffer:moe_args_ offset:0 atIndex:5];
  set_params(enc_, MoeRouteParams{T, c.experts, router.n_pad, 2 * c.moe_inter / TN, c.hidden / TN, c.norm_topk ? 1 : 0},
             6);
  [enc_ setBuffer:moe_sel_ offset:0 atIndex:7];
  run_groups(1, 1, 1024);
  mark("moe_route");

  const bool offload = cache_ && layer < c.layers;
  if (offload) {  // the CPU loads the experts this layer needs before the GPU goes on (ExpertCache)
    // The wait must stay the very next command after the signal, so that the
    // GPU is always waiting before the CPU answers. Running the shared expert
    // in between (to hide the ~40 us the GPU takes to wake up from a wait)
    // gave wrong results whenever the CPU answered first and had refilled a
    // slot the GPU had read earlier in the same command buffer: a wait that is
    // already satisfied when reached did not make those writes visible to the
    // expert matmuls (tests/check_moe.py, chunks of 1 with the smallest cache).
    ExpertCache &x = *cache_;
    [enc_ endEncoding];
    const uint64_t ready = ++x.value, go = ++x.value;
    [cb_ encodeSignalEvent:x.ev value:ready];
    [cb_ encodeWaitForEvent:x.ev value:go];
    enc_ = [cb_ computeCommandEncoder];
    x.submit({layer, T, ready, go});
  }

  [enc_ setComputePipelineState:mt_.pipeline("moe.metal", "moe_gather", route)];
  [enc_ setBuffer:xn_ offset:0 atIndex:0];
  [enc_ setBuffer:moe_tok_ offset:0 atIndex:1];
  [enc_ setBuffer:moe_x_ offset:0 atIndex:2];
  set_params(enc_, MoeGatherParams{A, c.hidden, x_stride_}, 3);
  run_threads(c.hidden / 8, A);
  mark("moe_gather");

  const Macros mm = {{"TM", std::to_string(TM)}, {"TN", std::to_string(TN)}, {"NSG", std::to_string(NSG)},
                     {"GROUP", "64"}, {"BLOCK", "256"}, {"OUT_T", "float"}, {"ACCUM", "0"},
                     {"W_T", c.book == "fp4" ? "metal_fp4_e2m1_format" : "int4b_format"}};
  [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_q4n_moe", mm)];
  [enc_ setBuffer:moe_x_ offset:0 atIndex:0];
  bind_experts(layer, 0, gu);
  [enc_ setBuffer:moe_items_ offset:0 atIndex:3];
  [enc_ setBuffer:moe_gu_ offset:0 atIndex:4];
  set_params(enc_, MoeLinearParams{c.hidden, 2 * c.moe_inter, c.hidden, 2 * c.moe_inter,
                                   (long)c.moe_inter * c.hidden, 2 * c.moe_inter}, 5);
  run_indirect(moe_args_, 0, 32 * NSG);
  mark("moe:gate_up");

  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "swiglu")];
  [enc_ setBuffer:moe_gu_ offset:0 atIndex:0];
  [enc_ setBuffer:moe_act_ offset:0 atIndex:1];
  set_params(enc_, SwigluParams{A, c.moe_inter, 2 * c.moe_inter, c.moe_inter}, 2);
  if (offload) [enc_ setBuffer:cache_->cols offset:(size_t)layer * c.moe_inter * sizeof(float) atIndex:3];
  else bind_column_scale(dn, 3);
  run_threads(c.moe_inter, A);
  mark("moe_swiglu");

  [enc_ setComputePipelineState:mt_.pipeline("linear.metal", "linear_q4n_moe", mm)];
  [enc_ setBuffer:moe_act_ offset:0 atIndex:0];
  bind_experts(layer, 1, dn);
  [enc_ setBuffer:moe_items_ offset:0 atIndex:3];
  [enc_ setBuffer:moe_y_ offset:0 atIndex:4];
  set_params(enc_, MoeLinearParams{c.moe_inter, c.hidden, c.moe_inter, c.hidden,
                                   (long)c.hidden * c.moe_inter / 2, c.hidden}, 5);
  run_indirect(moe_args_, 3 * sizeof(uint32_t), 32 * NSG);
  mark("moe:down");

  [enc_ setComputePipelineState:mt_.pipeline("moe.metal", "moe_combine", route)];
  [enc_ setBuffer:moe_y_ offset:0 atIndex:0];
  [enc_ setBuffer:moe_inv_ offset:0 atIndex:1];
  [enc_ setBuffer:moe_w_ offset:0 atIndex:2];
  [enc_ setBuffer:h_ offset:0 atIndex:3];
  set_params(enc_, MoeCombineParams{T, c.hidden, c.hidden}, 4);
  run_threads(c.hidden, T);
  mark("moe_combine");
}

void Engine::bind_experts(int layer, int kind, const QTensor &stack) {
  if (cache_ && layer < config().layers) {
    [enc_ setBuffer:cache_->w[kind] offset:0 atIndex:1];
    [enc_ setBuffer:cache_->s[kind] offset:0 atIndex:2];
    [enc_ setBuffer:cache_->table offset:(size_t)layer * (config().experts + 1) * sizeof(int32_t) atIndex:6];
  } else {
    [enc_ setBuffer:stack.buf offset:stack.w atIndex:1];
    [enc_ setBuffer:stack.buf offset:stack.s atIndex:2];
    [enc_ setBuffer:moe_identity_ offset:0 atIndex:6];
  }
}

void Engine::encode_linear_attn(int layer, int T, bool spec) {
  const Config &c = config();
  LinearLayer &L = *lin_[layer];
  const QTensor &in_proj = model_->q(lname(layer, "lin.in_proj"));
  const int C = c.lin_conv_dim(), kd = c.lin_key_dim(), vd = c.lin_value_dim();
  const int Hk = c.lin_k_heads, Hv = c.lin_v_heads, Dk = c.lin_k_dim, Dv = c.lin_v_dim;
  // A speculative pass keeps the recurrence inputs per layer (see LinearLayer).
  id<MTLBuffer> y = spec ? L.sp_y : y_, qk = spec ? L.sp_qk : qk_, bg = spec ? L.sp_bg : bg_;

  encode_norm(0, layer, T, 0);
  encode_linear(in_proj, T, proj_, 0, proj_stride_, false, "linear:lin_in");

  const FTensor &cw = model_->f(lname(layer, "lin.conv"));
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "lin_conv")];
  [enc_ setBuffer:proj_ offset:0 atIndex:0];
  [enc_ setBuffer:L.conv[L.cur] offset:0 atIndex:1];
  [enc_ setBuffer:cw.buf offset:cw.offset atIndex:2];
  [enc_ setBuffer:y offset:0 atIndex:3];
  [enc_ setBuffer:L.conv[1 - L.cur] offset:0 atIndex:4];
  [enc_ setBuffer:(spec ? L.sp_in : L.conv[1 - L.cur]) offset:0 atIndex:6];
  const int at = spec ? 0 : mark_row_;  // the pass's mark (set_mark)
  [enc_ setBuffer:(at ? mark_conv_[layer] : L.conv[1 - L.cur]) offset:0 atIndex:7];
  set_params(enc_, ConvParams{T, C, proj_stride_, C, spec ? 1 : 0, at}, 5);
  run_threads(C, T + 3 + (spec ? T : 0) + (at ? 3 : 0));
  mark("lin_conv");
  L.cur = 1 - L.cur;

  const LinPrepParams pp{T, Hk, Dk, Hv, C, 2 * kd, proj_stride_, in_proj.part_offset("b"),
                         in_proj.part_offset("a"), 2 * Hv};
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "lin_prep_qk")];
  [enc_ setBuffer:y offset:0 atIndex:0];
  [enc_ setBuffer:qk offset:0 atIndex:1];
  set_params(enc_, pp, 2);
  run_groups(2 * Hk, T);
  mark("lin_prep_qk");

  const FTensor &A_log = model_->f(lname(layer, "lin.A_log"));
  const FTensor &dt_bias = model_->f(lname(layer, "lin.dt_bias"));
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "lin_gates")];
  [enc_ setBuffer:proj_ offset:0 atIndex:0];
  [enc_ setBuffer:A_log.buf offset:A_log.offset atIndex:1];
  [enc_ setBuffer:dt_bias.buf offset:dt_bias.offset atIndex:2];
  [enc_ setBuffer:bg offset:0 atIndex:3];
  set_params(enc_, pp, 4);
  run_threads(Hv, T);
  mark("lin_gates");

  // The new state replaces the old one in place, except in a speculative pass,
  // which leaves the old one intact next to it.
  const int out = spec ? 1 - L.scur : L.scur;
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "gated_delta")];
  [enc_ setBuffer:qk offset:0 atIndex:0];
  [enc_ setBuffer:y offset:0 atIndex:1];
  [enc_ setBuffer:bg offset:0 atIndex:2];
  [enc_ setBuffer:L.S[L.scur] offset:0 atIndex:3];
  [enc_ setBuffer:yo_ offset:0 atIndex:4];
  set_params(enc_, DeltaParams{T, Hk, Dk, Hv, Dv, 2 * kd, C, 2 * kd, 2 * Hv, vd, at}, 5);
  [enc_ setBuffer:L.S[out] offset:0 atIndex:6];
  [enc_ setBuffer:(at ? mark_S_[layer] : L.S[out]) offset:0 atIndex:7];
  run_groups(Dv / 8, Hv);
  if (repeating("gated_delta")) {  // a repeat must not advance the state again: it runs on scratch
    if (!scratch_state_) {
      scratch_state_ = mt_.alloc(L.S[0].length);
      scratch_yo_ = mt_.alloc(yo_.length);
    }
    [enc_ setBuffer:scratch_state_ offset:0 atIndex:3];
    [enc_ setBuffer:scratch_yo_ offset:0 atIndex:4];
    [enc_ setBuffer:scratch_state_ offset:0 atIndex:6];
    [enc_ setBuffer:scratch_state_ offset:0 atIndex:7];
  }
  mark("gated_delta");
  L.scur = out;

  const FTensor &nw = model_->f(lname(layer, "lin.norm"));
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "lin_out_norm")];
  [enc_ setBuffer:yo_ offset:0 atIndex:0];
  [enc_ setBuffer:proj_ offset:0 atIndex:1];
  [enc_ setBuffer:nw.buf offset:nw.offset atIndex:2];
  [enc_ setBuffer:xn_ offset:0 atIndex:3];
  set_params(enc_, OutNormParams{T, Hv, Dv, vd, proj_stride_, in_proj.part_offset("z"), x_stride_, c.norm_eps}, 4);
  bind_column_scale(model_->q(lname(layer, "lin.out_proj")), 5);
  run_groups(Hv, T);
  mark("lin_out_norm");

  encode_linear(model_->q(lname(layer, "lin.out_proj")), T, h_, 0, c.hidden, true, "linear:lin_out");
}

void Engine::encode_full_attn(int layer, int T, int pos0, id<MTLBuffer> rope) {
  const Config &c = config();
  AttnLayer &A = *attn_[layer];
  const QTensor &qkv = model_->q(lname(layer, "attn.qkv"));
  const int G = c.heads / c.kv_heads, D = c.head_dim;

  encode_norm(0, layer, T, 0);
  encode_linear(qkv, T, proj_, 0, proj_stride_, false, "linear:attn_qkv");

  AttnParams ap{};
  ap.T = T, ap.H = c.heads, ap.KVH = c.kv_heads, ap.D = D;
  ap.proj_stride = proj_stride_, ap.off_k = qkv.part_offset("k"), ap.off_v = qkv.part_offset("v");
  ap.q_rows = q_rows_, ap.out_stride = x_stride_, ap.pos0 = pos0, ap.Lb = kv_blocks_;
  ap.eps = c.norm_eps, ap.scale = 1.0f / sqrtf((float)D);
  const FTensor &qw = model_->f(lname(layer, "attn.q_norm"));
  const FTensor &kw = model_->f(lname(layer, "attn.k_norm"));
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "attn_prep")];
  [enc_ setBuffer:proj_ offset:0 atIndex:0];
  [enc_ setBuffer:qw.buf offset:qw.offset atIndex:1];
  [enc_ setBuffer:kw.buf offset:kw.offset atIndex:2];
  [enc_ setBuffer:rope offset:0 atIndex:3];
  [enc_ setBuffer:aq_ offset:0 atIndex:4];
  [enc_ setBuffer:A.K offset:0 atIndex:5];
  [enc_ setBuffer:A.V offset:0 atIndex:6];
  set_params(enc_, ap, 7);
  run_groups(c.heads + 2 * c.kv_heads, T);
  mark("attn_prep");

  // Rows are taken in groups whose scores fit the scratch buffers: all of
  // them in one group unless the context is long.
  for (int r0 = 0; r0 < T;) {
    int R = T - r0, kv_rows = 0, n_pad = 0;
    Tile t{};
    for (;; R = (R + 1) / 2) {
      t = tile_for(R * G);
      kv_rows = (R * G + t.tm - 1) / t.tm * t.tm;
      n_pad = (pos0 + r0 + R + 255) / 256 * 256;
      if ((size_t)c.kv_heads * kv_rows * n_pad <= score_cap_ || R == 1) break;
    }
    ap.pos0 = pos0 + r0;
    ap.M = R * G, ap.kv_rows = kv_rows, ap.n_pad = n_pad, ap.n_stride = n_pad;
    const Macros mac = {{"TM", std::to_string(t.tm)}, {"TN", std::to_string(t.tn)}, {"NSG", std::to_string(t.nsg)}};

    [enc_ setComputePipelineState:mt_.pipeline("attention.metal", "attn_scores", mac)];
    [enc_ setBuffer:aq_ offset:(size_t)r0 * G * D * sizeof(_Float16) atIndex:0];
    [enc_ setBuffer:A.K offset:0 atIndex:1];
    [enc_ setBuffer:as_ offset:0 atIndex:2];
    set_params(enc_, ap, 3);
    run_groups(n_pad / t.tn, kv_rows / t.tm, 32 * t.nsg, c.kv_heads);
    mark("attn_scores");

    [enc_ setComputePipelineState:mt_.pipeline("attention.metal", "attn_softmax", mac)];
    [enc_ setBuffer:as_ offset:0 atIndex:0];
    [enc_ setBuffer:ap_ offset:0 atIndex:1];
    [enc_ setBuffer:al_ offset:0 atIndex:2];
    set_params(enc_, ap, 3);
    run_groups(1, R * G, 32, c.kv_heads);
    mark("attn_softmax");

    [enc_ setComputePipelineState:mt_.pipeline("attention.metal", "attn_pv", mac)];
    [enc_ setBuffer:ap_ offset:0 atIndex:0];
    [enc_ setBuffer:A.V offset:0 atIndex:1];
    [enc_ setBuffer:al_ offset:0 atIndex:2];
    [enc_ setBuffer:proj_ offset:(size_t)r0 * proj_stride_ * sizeof(float) atIndex:3];
    [enc_ setBuffer:xn_ offset:(size_t)r0 * x_stride_ * sizeof(_Float16) atIndex:4];
    set_params(enc_, ap, 5);
    bind_column_scale(model_->q(lname(layer, "attn.o_proj")), 6);
    run_groups(D / t.tn, kv_rows / t.tm, 32 * t.nsg, c.kv_heads);
    mark("attn_pv");
    r0 += R;
  }

  encode_linear(model_->q(lname(layer, "attn.o_proj")), T, h_, 0, c.hidden, true, "linear:attn_out");
}

void Engine::run_threads(int w, int h) {
  last_ = {true, MTLSizeMake(w, h, 1), MTLSizeMake(32, 1, 1)};
  redo();
}

void Engine::run_indirect(id<MTLBuffer> args, size_t offset, int threads_per_group) {
  last_ = {false, MTLSizeMake(0, 0, 0), MTLSizeMake(threads_per_group, 1, 1), args, offset};
  redo();
}

// `threads_per_group` threads (a whole number of 32-thread SIMD-groups) per grid cell.
void Engine::run_groups(int w, int h, int threads_per_group, int depth) {
  last_ = {false, MTLSizeMake(w, h, depth), MTLSizeMake(threads_per_group, 1, 1)};
  redo();
}

void Engine::redo() {
  if (last_.indirect)
    [enc_ dispatchThreadgroupsWithIndirectBuffer:last_.indirect
                            indirectBufferOffset:last_.indirect_offset
                           threadsPerThreadgroup:last_.per_group];
  else if (last_.threads) [enc_ dispatchThreads:last_.grid threadsPerThreadgroup:last_.per_group];
  else [enc_ dispatchThreadgroups:last_.grid threadsPerThreadgroup:last_.per_group];
}

void Engine::mark(const char *what) {
  if (repeating(what))
    for (int i = 0; i < repeat_times_; i++) redo();
}

void Engine::begin() {
  cb_ = [mt_.queue() commandBuffer];
  enc_ = [cb_ computeCommandEncoder];
}

void Engine::flush(bool reopen) {
  [enc_ endEncoding];
  [cb_ commit];
  [cb_ waitUntilCompleted];
  if (cb_.status != MTLCommandBufferStatusCompleted)
    fatal("GPU command buffer failed: %s", cb_.error.localizedDescription.UTF8String);
  gpu_seconds_ += cb_.GPUEndTime - cb_.GPUStartTime;
  cb_ = nil;
  enc_ = nil;
  if (reopen) begin();
}

// A text token's rotary position: one past the largest component of the one before.
int32_t Engine::text_position(int p) const {
  if (p == 0) return 0;
  const int32_t *q = &rpos_[(size_t)(p - 1) * 3];
  return std::max(q[0], std::max(q[1], q[2])) + 1;
}

void Engine::set_positions(int p0, int T, const ImageRows *spans, int n_spans) {
  int s = 0;
  for (int t = 0; t < T; t++) {
    int32_t *q = &rpos_[(size_t)(p0 + t) * 3];
    while (s < n_spans && t >= spans[s].first + spans[s].rows) s++;
    if (s < n_spans && t >= spans[s].first) {
      const int k = spans[s].offset + t - spans[s].first;            // the patch's index in its image
      const int32_t start = k == 0 ? text_position(p0 + t) : q[-3];  // the image's time position
      q[0] = start;
      q[1] = start + k / spans[s].width;
      q[2] = start + k % spans[s].width;
    } else {
      q[0] = q[1] = q[2] = text_position(p0 + t);
    }
  }
}

void Engine::rope_position(int p, int32_t out[3]) const {
  if (p < 0 || p >= pos_) fatal("rope_position: position %d is not in the context (%d tokens)", p, pos_);
  memcpy(out, &rpos_[(size_t)p * 3], 3 * sizeof(int32_t));
}

// Rotary angles in double precision: position * theta^(-i/32). Qwen3.5's
// M-RoPE interleaves its sections (11, 11, 10): frequency i turns with the
// height position when i % 3 == 1, with the width when i % 3 == 2 and i < 30,
// and with the time otherwise. For text the three are the same.
void Engine::rope_row(int p, float *out) const {
  const int32_t *q = &rpos_[(size_t)p * 3];
  for (int i = 0; i < 32; i++) {
    const int k = i % 3 == 1 ? 1 : (i % 3 == 2 && i < 30) ? 2 : 0;
    const double ang = (double)q[k] * pow(config().rope_theta, -(double)i / 32.0);
    out[i] = (float)cos(ang);
    out[32 + i] = (float)sin(ang);
  }
}

void Engine::fill_rope(id<MTLBuffer> buf, int pos0, int T) {
  float *rope = (float *)buf.contents;
  for (int t = 0; t < T; t++) rope_row(pos0 + t, rope + t * 64);
}

void Engine::rope_angles(int p, float out[64]) const {
  if (p < 0 || p >= pos_) fatal("rope_angles: position %d is not in the context (%d tokens)", p, pos_);
  rope_row(p, out);
}

// h_[t] = row tok[t] of the token table.
void Engine::encode_embed(id<MTLBuffer> tok, int T) {
  const Config &c = config();
  const QTensor &emb = model_->q("embed");
  if (emb.native) {
    const char *levels = c.book == "fp4" ? "0,0.5,1,1.5,2,3,4,6,-0.0,-0.5,-1,-1.5,-2,-3,-4,-6"
                                         : "0,1,2,3,4,5,6,7,-8,-7,-6,-5,-4,-3,-2,-1";
    [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "embed_rows_native", {{"BOOK_LEVELS", levels}})];
  } else {
    [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "embed_rows")];
  }
  [enc_ setBuffer:tok offset:0 atIndex:0];
  [enc_ setBuffer:emb.buf offset:emb.w atIndex:1];
  [enc_ setBuffer:emb.buf offset:emb.s atIndex:2];
  [enc_ setBuffer:emb.buf offset:(emb.native ? emb.c : emb.b) atIndex:3];
  [enc_ setBuffer:h_ offset:0 atIndex:4];
  set_params(enc_, EmbedParams{T, c.hidden, c.hidden / 64, c.hidden}, 5);
  run_threads(c.hidden / 8, T);
  mark("embed");
}

void Engine::encode_image_rows(int shift, int M) {
  const int H = config().hidden;
  const size_t row = (size_t)H * sizeof(float);
  int src = 0;
  for (const ImageRows &s : spans_) {
    int a = s.first + shift, from = src;
    const int b = std::min(s.first + s.rows + shift, M);
    src += s.rows;
    if (a < 0) {
      from -= a;
      a = 0;
    }
    if (b <= a) continue;
    [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "copy_rows")];
    [enc_ setBuffer:emb_ offset:from * row atIndex:0];
    [enc_ setBuffer:h_ offset:a * row atIndex:1];
    set_params(enc_, CopyParams{b - a, H}, 2);
    run_threads(H, b - a);
    mark("image_rows");
  }
}

void Engine::encode_copy_rows(id<MTLBuffer> dst, int dst_row, int src_row, int rows) {
  const size_t row_bytes = (size_t)config().hidden * sizeof(float);
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "copy_rows")];
  [enc_ setBuffer:h_ offset:src_row * row_bytes atIndex:0];
  [enc_ setBuffer:dst offset:dst_row * row_bytes atIndex:1];
  set_params(enc_, CopyParams{rows, config().hidden}, 2);
  run_threads(config().hidden, rows);
  mark("copy_rows");
}

void Engine::encode_rot(const char *kernel, id<MTLBuffer> src, size_t src_offset, id<MTLBuffer> w1,
                        id<MTLBuffer> w2, size_t w2_offset, int rows, size_t out_offset) {
  const Config &c = config();
  const Macros mac = {{"ROT_D", std::to_string(c.hidden)}, {"ROT_BLOCK", std::to_string(c.rot_block)}};
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", kernel, mac)];
  [enc_ setBuffer:src offset:src_offset atIndex:0];
  [enc_ setBuffer:w1 offset:0 atIndex:1];
  [enc_ setBuffer:w2 offset:w2_offset atIndex:2];
  [enc_ setBuffer:xn_ offset:out_offset atIndex:3];
  set_params(enc_, RotNormParams{rows, c.hidden, c.hidden, x_stride_, c.norm_eps}, 4);
  run_groups(1, rows, 256);
  mark("mtp_rot");
}

void Engine::encode_mtp_hidden(id<MTLBuffer> src, const FTensor &w1, int rows, int first_row) {
  if (rows <= 0) return;
  const Config &c = config();
  [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "rms_norm2_rows")];
  [enc_ setBuffer:src offset:0 atIndex:0];
  [enc_ setBuffer:w1.buf offset:w1.offset atIndex:1];
  [enc_ setBuffer:mtp_norm_h_ offset:0 atIndex:2];
  [enc_ setBuffer:xn_ offset:((size_t)first_row * x_stride_ + c.hidden) * sizeof(_Float16) atIndex:3];
  set_params(enc_, NormParams{rows, c.hidden, c.hidden, x_stride_, c.norm_eps}, 4);
  run_groups(1, rows);
  mark("mtp_norm");
}

void Engine::encode_mtp_layer(int M, int pos0, bool logits, int image_shift) {
  const Config &c = config();
  const int H = c.hidden;
  // From here on h_ is the head's own residual stream.
  encode_embed(mtp_tok_, M);
  encode_image_rows(image_shift, M);  // a head row whose next token is an image patch reads the patch
  encode_rms(h_, 0, H, mtp_norm_e_, 0, M, 0);
  encode_linear(model_->q("mtp.fc"), M, h_, 0, H, false, "linear:mtp_fc");
  encode_full_attn(c.layers, M, pos0, mtp_rope_);
  encode_mlp(c.layers, M);
  if (logits) {
    encode_copy_rows(mtp_out_, 0, M - 1, 1);  // what draft_more continues from
    if (c.rot_block) {  // into the rotated stream the output head expects
      const QTensor &hd = model_->q("lm_head");
      encode_rot("mtp_out_rot", h_, (size_t)(M - 1) * H * sizeof(float), mtp_rot_g_, hd.buf, hd.c, 1, 0);
    } else {
      encode_rms(h_, (size_t)(M - 1) * H * sizeof(float), H, mtp_norm_out_, 0, 1, 0);
    }
    QTensor head = model_->q("lm_head");
    const int stride = head.n_pad;
    head.n_pad = draft_main();
    encode_linear(head, 1, mtp_logits_, 0, stride, false, "linear:mtp_head");
    if (draft_tail() >= 0) {  // the appended rows: the same matrix from a later block on
      QTensor tail = model_->q("lm_head");
      const int r0 = draft_tail();
      tail.w += (size_t)(r0 / 256) * (size_t)tail.k * 128;
      tail.s += (size_t)r0 * sizeof(float);
      tail.n_pad = stride - r0;
      encode_linear(tail, 1, mtp_logits_, (size_t)head.n_pad * sizeof(float), stride, false, "linear:mtp_head");
    }
  }
}

void Engine::set_draft_vocab(int rows) { draft_vocab_ = rows <= 0 ? 0 : (rows + 255) / 256 * 256; }

void Engine::set_draft_tail(int first) { draft_tail_ = first < 0 ? -1 : first / 256 * 256; }

int Engine::draft_main() const {
  const int all = model_->q("lm_head").n_pad;
  return draft_vocab_ > 0 ? std::min(draft_vocab_, all) : all;
}

int Engine::draft_tail() const {
  const QTensor &head = model_->q("lm_head");
  if (draft_tail_ < 0 || draft_tail_ <= draft_main() || draft_tail_ >= head.n_pad || !head.native) return -1;
  return draft_tail_;
}

int Engine::draft_vocab() const {
  const int tail = draft_tail();
  return draft_main() + (tail < 0 ? 0 : model_->q("lm_head").n_pad - tail);
}

void Engine::encode_mtp(int n_carry, int n_chunk, int keep_row, bool logits) {
  const Config &c = config();
  if (c.rot_block) {  // the model's hidden states are rotated: undo that before the head's norms
    const size_t half = (size_t)c.hidden * sizeof(_Float16);
    if (n_carry > 0) encode_rot("mtp_hidden_rot", mtp_h_, 0, mtp_rot_w1_, mtp_norm_h_, 0, n_carry, half);
    if (n_chunk > 0)
      encode_rot("mtp_hidden_rot", h_, 0, mtp_rot_w1_, mtp_norm_h_, 0, n_chunk,
                 (size_t)n_carry * x_stride_ * sizeof(_Float16) + half);
  } else {
    const FTensor &fn = model_->f("final_norm");
    encode_mtp_hidden(mtp_h_, fn, n_carry, 0);
    encode_mtp_hidden(h_, fn, n_chunk, n_carry);
  }
  if (keep_row >= 0) encode_copy_rows(mtp_h_, 0, keep_row, 1);
  if (n_carry + n_chunk > 0) encode_mtp_layer(n_carry + n_chunk, mtp_pos_, logits, n_carry - 1);
}

const float *Engine::forward(const int32_t *tokens, int T) { return run(tokens, T, false); }

const float *Engine::forward_embeds(const int32_t *tokens, int T, const ImageRows *spans, int n_spans,
                                    const float *embeds) {
  if (T < 1 || T > max_chunk_) fatal("forward: %d tokens, chunk limit is %d", T, max_chunk_);
  if (n_spans < 0 || (n_spans > 0 && (!spans || !embeds))) fatal("forward_embeds: no spans or no vectors");
  int end = 0, rows = 0;
  for (int i = 0; i < n_spans; i++) {
    const ImageRows &s = spans[i];
    if (s.rows < 1 || s.width < 1 || s.offset < 0 || s.first < end || s.first + s.rows > T)
      fatal("forward_embeds: span %d (rows %d to %d of %d) is out of order or out of range", i, s.first,
            s.first + s.rows, T);
    if (s.offset > 0 && (s.first != 0 || pos_ == 0))
      fatal("forward_embeds: an image continued from the previous chunk must start the chunk");
    end = s.first + s.rows;
    rows += s.rows;
  }
  if (n_spans > 0) {
    const size_t bytes = (size_t)rows * config().hidden * sizeof(float);
    if (!emb_) emb_ = mt_.alloc((size_t)max_chunk_ * config().hidden * sizeof(float));
    memcpy(emb_.contents, embeds, bytes);
  }
  return run(tokens, T, false, spans, n_spans);
}

const float *Engine::forward_draft(const int32_t *tokens, int n) {
  if (!mtp_) fatal("forward_draft: the model has no MTP head, or it is off");
  if (n < 2 || n > kSpecRows) fatal("forward_draft: %d rows (2 to %d)", n, kSpecRows);
  return run(tokens, n, true);
}

const float *Engine::draft_logits(int32_t next) {
  if (!mtp_) fatal("draft_logits: the model has no MTP head, or it is off");
  if (spec_rows_) fatal("draft_logits: drafts are open (keep_drafts first)");
  const int n = pos_ - mtp_pos_;
  if (n < 1) fatal("draft_logits: no position is waiting for its next token");
  gpu_seconds_ = 0;

  int32_t *succ = (int32_t *)mtp_tok_.contents;
  int j = 0;
  for (int32_t t : pend_tok_) succ[j++] = t;
  succ[j++] = next;
  fill_rope(mtp_rope_, mtp_pos_, n);
  @autoreleasepool {
    begin();
    encode_mtp(n, 0, -1, true);
    flush(false);
  }
  mtp_pos_ += n;
  pend_tok_.clear();
  draft_next_ = next;
  draft_rows_ = n;
  draft_depth_ = 1;
  return (const float *)mtp_logits_.contents;
}

const float *Engine::draft_more(int32_t guess) {
  if (!mtp_ || draft_depth_ < 1) fatal("draft_more: call draft_logits first");
  // The head's previous output stands in for the hidden state of the position
  // the model has not computed yet; its cache entry is overwritten when the
  // real position is processed.
  const int slot = mtp_pos_ + draft_depth_ - 1;
  if (slot >= max_ctx_) fatal("context window of %d tokens is full", max_ctx_);
  gpu_seconds_ = 0;
  *(int32_t *)mtp_tok_.contents = guess;
  set_positions(slot, 1);  // text, past the context: drafts are tokens
  fill_rope(mtp_rope_, slot, 1);
  @autoreleasepool {
    begin();
    encode_mtp_hidden(mtp_out_, model_->f("mtp.norm"), 1, 0);
    encode_mtp_layer(1, slot, true);
    flush(false);
  }
  draft_depth_++;
  return (const float *)mtp_logits_.contents;
}

const float *Engine::run(const int32_t *tokens, int T, bool spec, const ImageRows *spans, int n_spans) {
  const Config &c = config();
  if (T < 1 || T > max_chunk_) fatal("forward: %d tokens, chunk limit is %d", T, max_chunk_);
  if (pos_ + T > max_ctx_) fatal("context window of %d tokens is full", max_ctx_);
  if (spec_rows_) fatal("forward: drafts are open (keep_drafts first)");
  if (mtp_ && pos_ > 0 && pos_ == mtp_pos_ && tokens[0] != draft_next_) {
    // draft_logits ran the head on the last position for a next token that is
    // not the one arriving (a generation was cut short): that position waits again.
    const size_t row = (size_t)c.hidden * sizeof(float);
    if (draft_rows_ > 1) memmove(mtp_h_.contents, (const char *)mtp_h_.contents + (draft_rows_ - 1) * row, row);
    mtp_pos_ -= 1;
  }
  const int n_carry = mtp_ ? pos_ - mtp_pos_ : 0;  // hidden states waiting for their next token
  if (spec && n_carry != 0) fatal("forward_draft: call draft_logits first");
  gpu_seconds_ = 0;

  const int at = spec ? 0 : mark_row_;  // a snapshot inside this pass (set_mark)
  if (at && at >= T) fatal("forward: mark at row %d of a pass of %d rows", at, T);
  mark_row_ = 0;
  mark_pos_ = -1;
  memcpy(tok_.contents, tokens, (size_t)T * sizeof(int32_t));
  spans_.assign(spans, spans + n_spans);
  set_positions(pos_, T, spans, n_spans);
  fill_rope(rope_, pos_, T);
  // Head positions whose next token this call supplies: the waiting ones and
  // all rows of this chunk but the last.
  const int head_rows = mtp_ && !spec ? n_carry + T - 1 : 0;
  if (head_rows > 0) {
    int32_t *succ = (int32_t *)mtp_tok_.contents;
    int j = 0;
    for (int32_t t : pend_tok_) succ[j++] = t;
    for (int t = n_carry > 0 ? 0 : 1; t < T; t++) succ[j++] = tokens[t];
    fill_rope(mtp_rope_, mtp_pos_, head_rows);
  }

  @autoreleasepool {
    begin();
    if (!spans_.empty() && c.rot_block) {  // into the rotated stream: x -> x Q
      if (!emb_rot_) {
        const float *sg = model_->f("rotation.signs").data;
        emb_rot_ = mt_.alloc((size_t)c.hidden * sizeof(float));
        for (int k = 0; k < c.hidden; k++) ((float *)emb_rot_.contents)[k] = sg[k] / sqrtf((float)c.rot_block);
      }
      int rows = 0;
      for (const ImageRows &s : spans_) rows += s.rows;
      const Macros mac = {{"ROT_D", std::to_string(c.hidden)}, {"ROT_BLOCK", std::to_string(c.rot_block)}};
      [enc_ setComputePipelineState:mt_.pipeline("ops.metal", "rotate_rows", mac)];
      [enc_ setBuffer:emb_ offset:0 atIndex:0];
      [enc_ setBuffer:emb_rot_ offset:0 atIndex:1];
      set_params(enc_, CopyParams{rows, c.hidden}, 2);
      run_groups(1, rows, 256);
      mark("rotate_rows");
    }
    encode_embed(tok_, T);
    encode_image_rows(0, T);
    mark_row_ = at;   // read by encode_linear_attn
    for (int i = 0; i < c.layers; i++) {
      if (c.is_linear_layer(i)) encode_linear_attn(i, T, spec);
      else encode_full_attn(i, T, pos_, rope_);
      encode_mlp(i, T);
      if (!dump_dir_.empty()) {
        flush(true);
        char path[1024];
        snprintf(path, sizeof path, "%s/layer_%02d.f32", dump_dir_.c_str(), i);
        FILE *f = fopen(path, "wb");
        if (!f) fatal("cannot write %s", path);
        fwrite(h_.contents, sizeof(float), (size_t)T * c.hidden, f);
        fclose(f);
      }
    }

    // Logits for the last row, or for every row of a speculative pass.
    const int out_rows = spec ? T : 1;
    encode_norm(2, 0, out_rows, (size_t)(T - out_rows) * c.hidden * sizeof(float));
    const QTensor &head = model_->q("lm_head");
    encode_linear(head, out_rows, logits_, 0, head.n_pad, false, "linear:lm_head");
    if (at && mtp_) encode_copy_rows(mark_h_, 0, at - 1, 1);  // the head's waiting row at the mark
    if (mtp_) {
      if (spec) encode_copy_rows(mtp_h_, 0, 0, T);
      else encode_mtp(n_carry, T - 1, T - 1, false);
    }
    flush(false);
  }
  mark_row_ = 0;
  if (at) mark_pos_ = pos_ + at;

  spans_.clear();
  pos_ += T;
  if (mtp_) {
    mtp_pos_ += head_rows;
    pend_tok_.clear();
    if (spec) pend_tok_.assign(tokens + 1, tokens + T);
    draft_next_ = -1;
    draft_depth_ = 0;
  }
  spec_rows_ = spec ? T : 0;
  if (!dump_dir_.empty()) {
    FILE *f = fopen((dump_dir_ + "/logits.f32").c_str(), "wb");
    if (!f) fatal("cannot write logits dump");
    fwrite(logits_.contents, sizeof(float), (size_t)c.vocab, f);
    fclose(f);
  }
  return (const float *)logits_.contents;
}

int Engine::argmax() const { return argmax_of((const float *)logits_.contents); }

int Engine::argmax_of(const float *logits, int n) const {
  float best;
  vDSP_Length idx;
  vDSP_maxvi(logits, 1, &best, &idx, (vDSP_Length)(n > 0 ? std::min(n, config().vocab) : config().vocab));
  return (int)idx;
}

}  // namespace rl
