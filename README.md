# redline

A from-scratch inference engine for Apple Silicon that tries to run one model
family as close to the hardware's limits as it can, for a single local user.
The end goal is a usable local agent runtime on top of it.

**Status: early.** The engine runs Qwen3.5 / Qwen3.8 correctly and reproducibly,
but end to end it is still *slower* than MLX. The numbers below say exactly
where it stands. `redline` is a working name.

- Target model: **Qwen3.8-27B** (developed against Qwen3.5-4B, same architecture:
  75% Gated DeltaNet layers, 25% softmax attention).
- Target machine: Apple M5 / M6 class GPUs (the per-core GPU neural
  accelerators are required). Developed on an M6 Mac mini, 24 GB, macOS 27.2.

## What is here

| Path | What it is |
|---|---|
| `kernels/linear.metal` | Quantized linear layer that hands packed 4-bit weights straight to the GPU neural accelerators (Metal 4 `matmul2d` with `uint4b` operands), over a weight layout built for streaming. |
| `kernels/ops.metal` | Everything else in a layer: embedding, RMS norm, DeltaNet recurrence, attention, SwiGLU. |
| `src/` | Metal runtime (kernels compile at run time, no Xcode needed), mmap'd model file, forward pass with recurrent state. |
| `tools/convert.py` | MLX 4-bit model → `.rlm`. Same quantized values, different memory layout, page-aligned for zero-copy loading. |
| `tools/fetch_model.py` | Resumable model download from ModelScope or Hugging Face. |
| `tests/` | Kernel check against a double-precision CPU reference; layer-by-layer check against MLX. |
| `bench/` | Hardware probes and MLX / llama.cpp baselines. |

## Correctness

The converted model is numerically the model MLX runs (affine 4-bit, groups of
64), so the engine is checked directly against MLX evaluated in float32:

| Model | Worst layer rel. error | Max logit diff | Same greedy token |
|---|---|---|---|
| Qwen3.5-4B (32 layers) | 1.5e-3 | 0.041 | yes |
| Qwen3.8-27B (64 layers) | 2.1e-3 | 0.024 | yes |

For scale, MLX's default bfloat16 execution sits 1.6e-2 and 2.4e-2 away from
the same float32 reference. Feeding a prompt in chunks of 1, 17, 40 or 96
tokens gives **bit-identical** logits.

## Speed, measured on the M6 (24 GB)

End to end, 1024-token prompt, greedy decode:

| Model | Engine | Load | Prefill | Decode |
|---|---|---|---|---|
| Qwen3.5-4B | MLX 0.32.4 | 1.2 s | 1767 tok/s | 50.8 tok/s |
| | llama.cpp 0.9.0 (Q4_K_M) | – | 1250 tok/s | 39.9 tok/s |
| | **redline**, MLX's weights | 0.03 s | 1588 tok/s | 37.4 tok/s |
| | **redline**, native FP4 weights | 0.03 s | 1683 tok/s | 43.2 tok/s |
| Qwen3.8-27B | MLX 0.32.4 | 4.6 s | 269 tok/s | 9.2 tok/s |
| | **redline**, MLX's weights | 0.05 s | ~205–230 tok/s | 7.2 tok/s |

The 27B figures are noisy: at 4-bit it needs 15–18 GB and the machine swaps.
The native format has not been run on the 27B yet.

### Native weight format

`tools/quantize.py` turns an original checkpoint into weights of the form
`s[n] * c[k] * L[q]`: a 4-bit code the accelerators decode themselves, a scale
per output row and one per input column, nothing per group. It works one layer
at a time (GPTQ on calibration activations), so a checkpoint larger than RAM
converts. Held-out quality against the float model, Qwen3.5-4B, 1152 positions:

| Weights | Bits/weight | KL (nats) | Perplexity | Top-1 agreement |
|---|---|---|---|---|
| MLX affine 4-bit, groups of 64 | 4.5 | 0.072 | 3.465 | 88.5% |
| native FP4, GPTQ | 4.0 | 0.083 | 3.379 | 88.6% |
| native 16-level integer, GPTQ | 4.0 | 0.097 | 3.543 | 88.6% |
| native FP4, round to nearest | 4.0 | 0.103 | 3.478 | 88.9% |
| native 16-level integer, round to nearest | 4.0 | 0.155 | 3.782 | 85.4% |

FP4 with GPTQ is on par with MLX's format at 11% less memory. The engine's
output on that file matches MLX running the same decoded weights
(`tests/verify_native.py`).

The linear kernel in isolation, same quantized weights as MLX, Qwen3.8-27B
gate/up projection (5120 → 17408):

| Rows | redline | MLX | |
|---|---|---|---|
| 1 | 0.372 ms | 0.416 ms | about parity once MLX's in-model overhead is discounted |
| 8 | 0.373 ms | 0.784 ms | 2.1x |
| 64 | 0.746 ms | 0.936 ms | 1.25x |
| 1024 | 16.0 TFLOPS | 11.45 TFLOPS | 1.4x |

So the kernel is ahead where several rows share one pass over the weights, and
the engine around it is not yet: about 450 GPU dispatches per decoded token,
small kernels written for clarity, and a naive attention loop.

## What the hardware allows (same machine)

| Limit | Measured |
|---|---|
| Memory read bandwidth, CPU or GPU | ~150 GB/s (decode ceiling: ~63 tok/s for the 4B, ~10 tok/s for the 27B at 4.5 bits/weight) |
| GPU accelerators, half × 4-bit weights | ~19–20 TFLOPS |
| GPU accelerators, 8-bit activations × 4-bit weights | ~31–36 TFLOPS |
| MLX's quantized matmul in prefill | 7.4–13.2 TFLOPS |

Two findings shape the roadmap:

1. `matmul2d` accepts 4-bit, 2-bit, FP4 and FP8 operands natively. MLX
   dequantizes in software first and does not use them.
2. With per-group scales applied in registers (what an MLX-compatible format
   needs), this kernel tops out near 12–16 TFLOPS and tiles taller than 32 rows
   run ~3x slower. The accelerator's direct path with a scale-free operand is
   the one that reaches 19–31 TFLOPS.

## Roadmap

1. Engine overhead: fewer and wider kernels per token, attention that scales
   with context, tile choice per layer shape.
2. A native weight format without per-group scales inside the matmul, with its
   accuracy measured against the float model before it is trusted.
3. 8-bit activations for prefill, same condition.
4. Multi-token prediction: the checkpoints ship an MTP head that MLX drops;
   verifying 8–32 candidate tokens costs about one pass over the weights here.
5. Agent runtime: tokenizer, chat template, tool calls with constrained
   decoding, state checkpoints for the recurrent layers, an OpenAI-compatible
   local server.

## Build and run

```sh
cmake -S . -B build/cmake -G Ninja && cmake --build build/cmake

tools/fetch_model.py mlx-community/Qwen3.5-4B-MLX-4bit        # into models/
python tools/convert.py models/Qwen3.5-4B-MLX-4bit models/qwen3.5-4b.rlm

build/cmake/test_linear                                        # kernel vs CPU reference
python tests/verify_against_mlx.py models/Qwen3.5-4B-MLX-4bit models/qwen3.5-4b.rlm tokens.i32
build/cmake/redline bench models/qwen3.5-4b.rlm tokens.i32 --prompt 1024 --gen 128
build/cmake/redline profile models/qwen3.5-4b.rlm tokens.i32  # per-kernel GPU time
```

Token files are raw little-endian int32; tokenization is not in the engine yet.
The Python tools need `numpy`; verification and baselines also need `mlx-lm`.

## License

Apache-2.0. Model weights are not part of this repository and keep their own
licenses (the Qwen models used here are Apache-2.0).
