# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Converted model
files are published separately, in the `models-v1` release.

## [Unreleased]

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
