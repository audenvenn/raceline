# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Converted model
files are published separately: in GitHub releases (`models-v1` for 0.1.0,
`models-v2` from 0.2.0 on) and on Hugging Face.

## [Unreleased]

## [0.2.0] - 2026-11-12

### Changed

- Model conversion rotates the residual stream before quantizing (random
  signs times Hadamard blocks, folded into the weights; `--no-rotate` turns
  it off). The model computes the same function and its weights suit the
  format's per-row and per-column scales far better. Held-out KL to the float
  model: Qwen3.5-4B 0.083 to 0.072 nats, now level with MLX's affine 4-bit
  format at 4.5 bits per weight (0.072), with top-1 agreement 88.6% to 91.1%;
  Qwen3.8-27B 0.152 to 0.148, still behind MLX's 0.105 (#15 stays open).
- The draft head of a rotated model undoes the rotation with two
  Walsh-Hadamard kernels; no weights are read for it.
- `raceline pull` reads the `models-v2` release, which holds the rotated
  files. raceline 0.1.0 keeps reading `models-v1`.

### Added

- `pip install` compiles the engine and ships it with its kernels: `pip
  install git+https://github.com/audenvenn/raceline` needs only the Xcode
  command line tools (#9).
- `raceline pull --source huggingface`, from `audenvenn/raceline-<model>`;
  `RACELINE_SOURCE` sets the default.
- `tools/release_model.py --card` (model cards) and `tools/publish_model.py`
  (uploads to the GitHub release and Hugging Face).
- `tools/quantize.py` prints how far each layer's hidden states have drifted
  from the float model's; `RACELINE_ACT_ORDER=1` tries GPTQ act-order (no
  gain measured, off by default).
- `bench/race_demo.py`: two runtimes answering the same request, replayed side
  by side as an animated GIF; `bench/chat_bench.py --json`.

## [0.1.0] - 2026-11-11

First public release.

### Added

- Engine for dense Qwen3.5 / Qwen3.8 text models on Apple GPU neural
  accelerators: native 4-bit weight format (FP4 codes with row and column
  scales, GPTQ), mapped model files, attention as hardware matrix products,
  results that do not depend on how a prompt is chunked.
- Speculative decoding from the model's own multi-token-prediction head: up to
  seven drafts verified per pass, bit-identical to one-token decoding, exact
  under sampling through coupled draws, with state replay for partly accepted
  passes.
- Runtime: session reuse with snapshots of the recurrent state, chat rendering
  and tool-call parsing, an OpenAI-compatible server (`raceline serve`), a
  terminal agent with approval prompts (`raceline chat`), a hardware probe
  (`raceline probe`), model downloads (`raceline pull`), and an MLX backend.
- Tools: `quantize.py` (original checkpoint to `.rlm`, resumable),
  `add_mtp.py`, `convert.py` (from an MLX 4-bit model), `release_model.py`,
  and the studies behind the design (`mtp_study.py`, `quant_study.py`).
- Verification against MLX and CPU references, bit-exactness tests for
  speculative decoding, and benchmarks against mlx-lm.
- Project infrastructure: CI, contribution guide, security policy, issue and
  pull-request templates.

[Unreleased]: https://github.com/audenvenn/raceline/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/audenvenn/raceline/releases/tag/v0.2.0
[0.1.0]: https://github.com/audenvenn/raceline/releases/tag/v0.1.0
