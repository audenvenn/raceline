"""`raceline probe`: what this Mac can do for a local LLM, measured in a few seconds.

Two numbers bound single-user inference on Apple Silicon:

  memory bandwidth   Producing a token reads every weight once, so
                     tokens/s <= bandwidth / size of the weights.
  matmul throughput  Reading a prompt is compute bound: every weight takes
                     part in one multiply-add per token, so
                     tokens/s <= FLOPS / (2 * parameters).

The probe measures both on the GPU and prints the resulting ceilings for a
few model sizes. Given a model, it also measures what a backend actually
reaches, as a fraction of those ceilings.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

GB = 1e9

# The 32 lanes of a SIMD-group read 32 adjacent words per iteration, so every
# group streams one contiguous region: the access pattern that reaches the
# bandwidth the memory system has (a contiguous chunk per thread does not).
READ_SRC = """
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
"""


def _best(fn, reps=6, warmup=2) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def machine() -> dict:
    info = dict(chip=_sysctl("machdep.cpu.brand_string") or "Apple Silicon",
                ram_gb=int(_sysctl("hw.memsize") or 0) / 2**30,
                cores=[int(c) for c in (_sysctl(f"hw.perflevel{i}.physicalcpu") for i in range(3)) if c.isdigit()],
                os=subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip())
    try:
        sp = json.loads(subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"], capture_output=True,
                                       text=True, timeout=20).stdout)
        info["gpu_cores"] = int(sp["SPDisplaysDataType"][0].get("sppci_cores", 0))
    except Exception:
        info["gpu_cores"] = 0
    return info


def gpu_bandwidth(gib: float = 1.0) -> float:
    """GB/s the GPU can read from memory."""
    import mlx.core as mx

    n_words = int(gib * 2**30) // 4
    n_words -= n_words % (32 * 4096)
    w = mx.random.randint(0, 1 << 30, shape=(n_words,), dtype=mx.uint32)
    mx.eval(w)
    kernel = mx.fast.metal_kernel(name="raceline_probe_read", input_names=["w"], output_names=["out"], source=READ_SRC)
    best = 0.0
    for ch in (256, 1024, 4096):
        def run():
            (o,) = kernel(inputs=[w], template=[("CH", ch)], grid=(n_words // ch, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[(n_words // ch,)], output_dtypes=[mx.uint32])
            mx.eval(o)
        best = max(best, n_words * 4 / _best(run) / GB)
    return best


def gpu_matmul() -> float:
    """Peak TFLOPS of a dense half-precision matrix product."""
    import mlx.core as mx

    peak = 0.0
    for m, k, n in ((2048, 2048, 2048), (4096, 4096, 4096), (1024, 5120, 17408)):
        a = mx.random.normal((m, k)).astype(mx.float16)
        b = mx.random.normal((k, n)).astype(mx.float16)
        mx.eval(a, b)
        peak = max(peak, 2.0 * m * k * n / _best(lambda: mx.eval(a @ b), reps=4) / 1e12)
    return peak


# (label, parameters in billions, weight bytes at 4 bits per weight plus token table and head)
SIZES = (("4B", 4.2, 2.3 * GB), ("9B", 9.0, 5.0 * GB), ("27B", 27.0, 13.6 * GB), ("70B", 70.0, 35.5 * GB))


def measure_model(model: str, backend_name=None, prompt_tokens: int = 1024, gen_tokens: int = 96) -> dict:
    """Prefill and greedy decode speed of a backend on this machine."""
    from .backends import Sampling, load_backend

    b = load_backend(model, backend_name)
    text = open(__file__, encoding="utf-8").read() * 40
    ids = b.tokenizer.encode(text, add_special_tokens=False)[:prompt_tokens]
    out = {}
    for rep in range(2):  # the first pass pages the weights in and compiles kernels
        b.reset()
        t0 = time.perf_counter()
        b.prefill(ids)
        out["prefill"] = len(ids) / (time.perf_counter() - t0)
        t0 = time.perf_counter()
        n = sum(1 for _ in b.generate(Sampling(temperature=0.0), gen_tokens, ids))
        out["decode"] = n / (time.perf_counter() - t0)
    counters = getattr(b, "counters", None)
    if counters and counters.get("drafts"):
        out["accepted"] = counters["accepted"] / counters["drafts"]
    out["backend"] = b.name
    if model.endswith(".rlm"):
        # What a pass reads: every section but the token table (looked up, not read), the head and the files.
        with open(model, "rb") as f:
            f.seek(8)
            hdr = json.loads(f.read(int.from_bytes(f.read(8), "little")))
        out["bytes"] = sum(sec["nbytes"] for sec in hdr["sections"] if sec["name"] not in ("embed", "mtp", "files"))
    else:
        out["bytes"] = sum(os.path.getsize(os.path.join(model, f)) for f in os.listdir(model) if f.endswith(".safetensors"))
    return out


def run(model=None, backend=None) -> None:
    m = machine()
    cores = "+".join(str(c) for c in m["cores"]) or "?"
    print(f"{m['chip']} · {cores} CPU cores · {m['gpu_cores'] or '?'}-core GPU · {m['ram_gb']:.0f} GB · macOS {m['os']}\n")
    bw = gpu_bandwidth(min(1.0, m["ram_gb"] / 16))
    print(f"  GPU memory bandwidth       {bw:6.1f} GB/s")
    tf = gpu_matmul()
    print(f"  matrix multiply (half)     {tf:6.1f} TFLOPS\n")
    print("  What that allows, per model size at 4 bits per weight:\n")
    print(f"  {'model':8s} {'weights':>9s} {'fits':>6s} {'decode ceiling':>16s} {'prefill ceiling':>17s}")
    for label, params, nbytes in SIZES:
        fits = "yes" if nbytes / 2**30 + 3 < m["ram_gb"] else "no"
        print(f"  {label:8s} {nbytes / GB:7.1f} GB {fits:>6s} {bw * GB / nbytes:10.1f} tok/s {tf * 1e12 / (2 * params * 1e9):11.0f} tok/s")
    print("\n  The decode ceiling is for one token per pass over the weights. A draft that is")
    print("  verified in the same pass (raceline uses the model's own MTP head) goes above it.")
    if not model:
        return
    print(f"\n  Measuring {model} ...")
    r = measure_model(model, backend)
    ceil_d = bw * GB / r["bytes"]
    line = f"  {r['backend']} backend: decode {r['decode']:.1f} tok/s = {100 * r['decode'] / ceil_d:.0f}% of the one-token ceiling"
    if "accepted" in r:
        line += f" ({100 * r['accepted']:.0f}% of drafts accepted)"
    print(line + f"; prefill {r['prefill']:.0f} tok/s")
