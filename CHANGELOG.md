# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Converted model
files are published separately: in GitHub releases (`models-v1` for 0.1.0,
`models-v2` from 0.2.0 on) and on Hugging Face.

## [Unreleased]

### Added

- `bench/agent_eval.py --set extended`: eight harder agent tasks closer to
  real work (a bug across a small package, a 600-line file, tests that must
  catch three of four broken versions of the code, a module and a script
  written from a description, a signature change across files, a command
  to implement in a CLI). The original eight stay the default (`core`).

- `bench/race_demo.py record-agent` records the terminal agent on one of the
  agent tasks, and `render --speed` replays faster than measured. The README
  now opens with the 27B agent fixing the same bug on mlx-lm and on raceline
  (102.7 s against 30.1 s, replayed at three times the speed).

## [0.3.0] - 2026-11-12

### Added

- Qwen3.5-9B: `raceline pull qwen3.5-9b` (4.7 GB, from the `models-v2`
  release or Hugging Face). It fits a 16 GB Mac; on the agent tasks it solves
  as many as Qwen3.8-27B (7 of 8) in 128 s instead of 317, and decodes at 57
  to 114 tok/s where mlx-lm does 27 to 29. `tools/release_model.py --needs`
  sets the raceline version a model card asks for.
- Drafts copied from the context. When the last three tokens occurred
  earlier in the context and what followed them there agrees with the draft
  head's first guess, that earlier continuation is drafted instead: the lines
  an edit replaces, a file rewritten with one change, quoted code. A pass
  checks up to 15 drafts (up from 7; 16 rows cost 3 to 6% more than one).
  `RACELINE_LOOKUP` sets the most per pass, `0` turns copying off. Output is
  unchanged, as with every draft. On the agent tasks of
  `bench/agent_eval.py`, Qwen3.5-4B decodes at 184 tok/s with them and 151
  without; Qwen3.8-27B at 32.1 and 31.5.
- `bench/agent_eval.py`: eight small coding tasks (`bench/agent_tasks.py`)
  run by the agent with every tool call approved, each judged by a hidden
  check; a fixed seed makes runs that differ only in drafting write the same
  tokens. `tests/test_agent_tasks.py` checks that every task fails as handed
  out and passes with a reference solution.
- `tests/test_server.py`: the OpenAI-compatible server against a scripted
  runtime, run by CI.
- `bench/agent_bench.py --corpus unseen --samples N`: continue code the
  models cannot have seen (frozen copies of this project's own files in
  `bench/corpus/`) instead of the standard library, which they partly
  remember, over several stretches of text. The benchmark scripts warn when
  other processes load the machine.
- Every draft can name the chat template's special tokens (`<tool_call>`,
  `</think>`, `<|im_end|>`, ...): the block of the output matrix that holds
  them is appended to a draft's restricted vocabulary (512 rows). Before, an
  agent's turn ended a draft chain at each of them (1 to 1.5% of its output).

### Changed

- `raceline chat` samples as Qwen recommends for coding: temperature 0.6,
  top-p 0.95, top-k 20 with thinking on (0.7, 0.8, 20 with `--no-think`),
  instead of the model's general-purpose setting (temperature 1.0 for the
  27B, 0.7 with top-p 0.8 for the 4B, which ships none). On the agent tasks
  it solves as many (the 27B 7 of 8 in each of two runs, the 4B 5 of 8 on
  average over three) and the 4B writes 19% fewer tokens, with fewer
  reasoning loops.
  `bench/agent_eval.py` uses the same setting. `raceline serve` keeps the
  model's own defaults; clients send their own.
- The draft vocabulary is chosen from the context's text tokens only, and
  64k ids are used while at most 3% of them lie beyond (before: 1% of all
  tokens, special ones included). Agent conversations mostly got the whole
  248k-row pass for their drafts: 7.4 ms per draft on Qwen3.8-27B, against
  3.9 ms for the 64k one.
- The engine refuses, with an explanation, to start on a Mac before Apple M5
  or on macOS before 27, instead of failing in the Metal compiler; MLX models
  still run there. `RACELINE_ANY_GPU=1` skips the check.
- `raceline pull` says when a model is too large for the Mac's memory.
- The server ends a generation as soon as its client disconnects, reports an
  error that happens mid-stream as a stream event, and accepts the
  `developer` role as system instructions.

### Fixed

- Every shell command the agent ran forked the agent's whole process, which
  with Qwen3.8-27B maps 14 GB of weights and GPU buffers. That cost seconds
  per command and made the prompt processing after it about nine times
  slower: the eight agent tasks of `bench/agent_eval.py` took 752 s, now 317
  s. The forked copy could also die before it ran the command (SIGTRAP in the
  allocator's fork handler on macOS 27, for about 2% of commands). Commands
  are started with `posix_spawn` now, never `fork`; `raceline probe` and the
  hardware check read `sysctl` values directly instead of running `sysctl`.
- A shell command that timed out left whatever it had started running: a
  test stuck in a loop kept a CPU core busy after the agent moved on. Commands
  run in a process group of their own, and the whole group is killed.
- Sampling with `presence_penalty` or `repetition_penalty` crashed in the
  draft step when a recent token lay outside the draft's vocabulary (an
  IndexError on the first draft after `<|im_start|>`).

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

[Unreleased]: https://github.com/audenvenn/raceline/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/audenvenn/raceline/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/audenvenn/raceline/releases/tag/v0.2.0
[0.1.0]: https://github.com/audenvenn/raceline/releases/tag/v0.1.0
