#!/usr/bin/env python3
"""GPU-side hardware ceilings, measured through MLX (the strongest existing baseline).

Three questions, each of which bounds a different inference regime:

  1. read bandwidth  -> ceiling for single-stream decode (weights read once per token)
  2. dense GEMM rate -> ceiling for prefill (compute-bound)
  3. how much of (1) and (2) MLX's *quantized* kernels actually reach

Run:  .venv/bin/python bench/hw/gpu_probe.py [--quick]
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn


def best_time(fn, reps=7, warmup=2):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------- bandwidth

READ_SRC = {
    # Each thread walks its own contiguous chunk.
    "contiguous": """
        uint tid = thread_position_in_grid.x;
        uint acc = 0;
        uint base = tid * CH;
        for (uint i = 0; i < CH; ++i) { acc += w[base + i]; }
        out[tid] = acc;
    """,
    # The 32 lanes of a SIMD-group read 32 adjacent words (one 128-byte line)
    # per iteration, so each group streams one contiguous region.
    "lane_interleaved": """
        uint tid = thread_position_in_grid.x;
        uint grp = tid / 32;
        uint lane = tid % 32;
        uint acc = 0;
        uint base = grp * 32 * CH + lane;
        for (uint i = 0; i < CH; ++i) { acc += w[base + i * 32]; }
        out[tid] = acc;
    """,
    # Same as lane_interleaved but each lane pulls 4 words per iteration
    # (512 contiguous bytes per SIMD-group per iteration).
    "lane_interleaved_x4": """
        uint tid = thread_position_in_grid.x;
        uint grp = tid / 32;
        uint lane = tid % 32;
        uint acc = 0;
        uint base = grp * 32 * CH + lane * 4;
        for (uint i = 0; i < CH; i += 4) {
          const device uint* p = w + base + i * 32;
          acc += p[0] + p[1] + p[2] + p[3];
        }
        out[tid] = acc;
    """,
}


def bench_bandwidth(gib: float):
    n_words = int(gib * (1 << 30)) // 4
    n_words -= n_words % (32 * 4096)
    w = mx.random.randint(0, 1 << 30, shape=(n_words,), dtype=mx.uint32)
    mx.eval(w)
    print(f"\n[1] GPU read bandwidth  (buffer {n_words * 4 / 2**30:.2f} GiB uint32)")
    print(f"    {'pattern':22s} {'words/thread':>12s} {'GB/s':>8s}")
    t = best_time(lambda: mx.eval(mx.sum(w)))
    print(f"    {'mx.sum (builtin)':22s} {'-':>12s} {n_words * 4 / t / 1e9:8.1f}")
    results = {}
    for name, src in READ_SRC.items():
        k = mx.fast.metal_kernel(name=f"read_{name}", input_names=["w"], output_names=["out"], source=src)
        for ch in (64, 256, 1024, 4096):
            n_threads = n_words // ch

            def run():
                (o,) = k(
                    inputs=[w],
                    template=[("CH", ch)],
                    grid=(n_threads, 1, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(n_threads,)],
                    output_dtypes=[mx.uint32],
                )
                mx.eval(o)

            t = best_time(run)
            gbps = n_words * 4 / t / 1e9
            results[(name, ch)] = gbps
            print(f"    {name:22s} {ch:12d} {gbps:8.1f}")
    best = max(results.items(), key=lambda kv: kv[1])
    print(f"    -> best: {best[0][0]} / {best[0][1]} words: {best[1]:.1f} GB/s")
    del w
    return best[1]


# --------------------------------------------------------------------- GEMM


def bench_gemm(quick: bool):
    print("\n[2] dense GEMM throughput  (C[M,N] = A[M,K] @ B[K,N])")
    print(f"    {'dtype':9s} {'M':>6s} {'K':>6s} {'N':>6s} {'ms':>9s} {'TFLOPS':>8s}")
    shapes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
              (1024, 2560, 9216), (1024, 5120, 17408), (4096, 5120, 17408),
              (64, 5120, 17408), (8, 5120, 17408)]
    if quick:
        shapes = shapes[1:5]
    peak = {}
    for dt in (mx.float32, mx.float16, mx.bfloat16):
        for M, K, N in shapes:
            a = mx.random.normal((M, K)).astype(dt)
            b = mx.random.normal((K, N)).astype(dt)
            mx.eval(a, b)
            t = best_time(lambda: mx.eval(a @ b), reps=5)
            tf = 2.0 * M * K * N / t / 1e12
            name = str(dt).split(".")[-1]
            peak[name] = max(peak.get(name, 0), tf)
            print(f"    {name:9s} {M:6d} {K:6d} {N:6d} {t * 1e3:9.2f} {tf:8.2f}")
    print("    -> peak: " + ", ".join(f"{k} {v:.2f} TFLOPS" for k, v in peak.items()))
    return peak


# ---------------------------------------------------------- quantized paths


def bench_quantized(bw_ceiling: float, quick: bool):
    print("\n[3] MLX quantized linear (affine, group 64): decode = bandwidth view, prefill = compute view")
    print(f"    {'bits':>4s} {'in':>6s} {'out':>6s} {'T':>6s} {'ms/layer':>9s} {'GB/s':>8s} {'%bw':>6s} {'TFLOPS':>8s}")
    dims = [(2560, 9216), (9216, 2560), (5120, 17408), (17408, 5120), (5120, 5120)]
    if quick:
        dims = dims[:1] + dims[2:3]
    for bits in (4, 8):
        for din, dout in dims:
            # Enough distinct layers that the working set far exceeds any cache.
            w_bytes = din * dout * bits / 8 + (din // 64) * dout * 4
            n_layers = max(2, int(600e6 // w_bytes))
            layers = []
            for _ in range(n_layers):
                lin = nn.Linear(din, dout, bias=False)
                q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)
                mx.eval(q.parameters())
                layers.append(q)
                del lin
            for T in (1, 2, 4, 8, 64, 1024):
                if quick and T not in (1, 4, 1024):
                    continue
                x = mx.random.normal((T, din)).astype(mx.float16)
                mx.eval(x)
                use = layers if T <= 8 else layers[:2]

                def run():
                    mx.eval([l(x) for l in use])

                t = best_time(run, reps=5) / len(use)
                gbps = w_bytes / t / 1e9
                tf = 2.0 * T * din * dout / t / 1e12
                print(f"    {bits:4d} {din:6d} {dout:6d} {T:6d} {t * 1e3:9.3f} {gbps:8.1f} "
                      f"{100 * gbps / bw_ceiling:5.0f}% {tf:8.2f}")
            del layers
            mx.clear_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gib", type=float, default=2.0)
    ap.add_argument("--only", choices=["bw", "gemm", "quant"])
    args = ap.parse_args()
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    print(f"device: {info.get('device_name')}  arch: {info.get('architecture')}  mlx {mx.__version__}")
    bw = 150.0
    if args.only in (None, "bw"):
        bw = bench_bandwidth(args.gib)
    if args.only in (None, "gemm"):
        bench_gemm(args.quick)
    if args.only in (None, "quant"):
        bench_quantized(bw, args.quick)


if __name__ == "__main__":
    main()
