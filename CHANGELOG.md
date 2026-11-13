# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Converted model
files are published separately: in GitHub releases (`models-v1` for 0.1.0,
`models-v2` from 0.2.0 on) and on Hugging Face.

## [Unreleased]

### Fixed

- A model whose weights, attention cache and scratch need more GPU memory
  than Metal recommends for the Mac loaded fine and then failed halfway
  through the first long prompt ("Insufficient Memory"): Qwen3.8-27B with
  `--ctx 65536` on a 24 GB Mac needs 19.3 GB of the 19.1 allowed. The engine
  now refuses it when it opens and names a context that fits with room for
  images (`--ctx 32768` there; 62208 is the limit). `RACELINE_ANY_SIZE=1`
  loads it anyway.

## [0.7.0] - 2026-11-13

### Added

- `tools/act_study.py`: what 8-bit activations would cost, measured on the
  decoded model before any kernel (#6). On Qwen3.5-4B, int8 inputs to the
  matrices that read the normed (rotated) residual stream leave the KL to
  the float model at 0.0599; on every matrix it rises to 0.0677.

### Changed

- The session's snapshots at branch points (the end of the system prompt,
  the start of the reply) are taken inside the prompt's passes
  (`Engine::set_mark`: the DeltaNet recurrence and the conv windows keep
  their state at that row) instead of ending a pass there (#60). On a
  mixture of experts with its experts on disk each extra pass was a sweep of
  SSD reads: a 3,490-token request with both branch points on
  Qwen3.6-35B-A3B on a 24 GB Mac now syncs in 3.3 s instead of 4.5, reading
  a third fewer experts. Dense models are as before. `tests/verify_marks.py`
  checks the state against a pass that ends at the mark, byte for byte.
- A fresh expert cache fills its free slots in the background while no pass
  is waiting, a hundred megabytes at a time and without evicting anything,
  instead of staying empty until prompts bring the experts in. The first
  request to Qwen3.6-35B-A3B on a 24 GB Mac, 3,490 tokens sent five seconds
  after the model opened: 2.6 s instead of 3.7, with half the experts read
  on demand; one sent at once is no slower. Outputs are bit-identical.
  `RACELINE_EXPERT_PRELOAD=0` turns it off.

## [0.6.0] - 2026-11-13

### Added

- Image input (#28). The models read images with their own vision tower,
  which `tools/convert_vision.py` writes to a vision file (float16, 0.67 GB
  for the 4B, about 0.9 GB for the others) that `raceline pull` fetches with
  the model. The terminal agent takes `--image FILE` and `/image FILE`, and its
  `read_file` shows the model image files; the server reads OpenAI's
  `image_url` parts with a data URL and Anthropic's base64 `image` blocks, in
  user messages and tool results. An image is scaled down to a megapixel at
  most (`RACELINE_IMAGE_PIXELS`), a row of context per 32 x 32 pixels.
- The engine takes rows as vectors (`forward_embeds`, `rl_forward_embeds`)
  with Qwen's three-dimensional rotary positions (M-RoPE), which travel with
  the attention entries of a conversation set aside; text is bit-for-bit
  unchanged. The session keys an image's rows by the image's content, so
  they are computed once per conversation and never confused with another
  image of the same size.
- Checks against transformers: `tests/verify_vision.py` (patches identical,
  the tower within 8e-5 in float32) and `tests/verify_image_input.py` (every
  rotary position and angle equal; after an image, KL 0.072 nats and 94.7%
  top-1 agreement on Qwen3.5-4B's answer, against 0.165 and 89.8% without
  one), with `tests/reference_image.py` making the references;
  `tests/check_image_chat.py` for the runtime on a real model.

### Changed

- Pillow is a dependency (images are decoded with it).

### Fixed

- A process that forked after the engine had loaded a model (a program
  embedding the runtime that runs `subprocess`, a notebook's `!command`)
  ran every later pass slower: Qwen3.6-35B-A3B decoded at 6 instead of 69
  tok/s, Qwen3.8-27B at 8 instead of 18 (#53). The model file was mapped
  private, which is copy-on-write; a fork marks it so in the parent too, and
  the GPU driver's wiring of the weights then made the kernel copy them. The
  file is now mapped shared (it is only read); outputs are bit-identical.
  `tests/check_fork.py` checks it on a real model. raceline's own agent and
  probe did not fork after loading.

## [0.5.1] - 2026-11-13

### Fixed

- Mixture-of-experts models with experts on disk ran well below their speed,
  by an amount that changed from session to session (#47). Every MoE layer
  makes the GPU wait while the CPU reads the missing experts, and the GPU's
  power management takes those waits for idleness and lowers the GPU's
  performance. While such passes come in, the engine now keeps a small
  kernel running on a command queue of its own (`RACELINE_GPU_KEEPALIVE=0`
  turns it off). Qwen3.6-35B-A3B on a 24 GB Mac: chat answers at 61 to 72
  tok/s instead of 45 to 52, about 55 tok/s without drafts instead of 32 to
  44; as a 16 GB Mac would run it, 30.1 tok/s instead of 15.6 (39.8 with
  drafts instead of 28.1). Dense models wait for nothing inside a pass and
  are unchanged.

- `bench/agent_eval.py`: the `write-tests` task counted only test
  functions at module level, so tests written as a `unittest.TestCase`
  (as both 35B-A3B models wrote them) failed although they ran and caught
  the broken versions of the code. Methods of a test class count now.
  Measured again: on the extended set Qwen3.6-35B-A3B solves 7 of 8 (0.5.0's
  notes said 6), Qwen3.5-35B-A3B 6 (said 5), Qwen3.5-9B 6.

- `raceline pull` without a model marks a mixture of experts whose experts
  stay partly on the SSD by the rule the engine applies (the memory the GPU
  may use, not the Mac's); Qwen3.6-35B-A3B went unmarked on a 24 GB Mac.

- `tests/check_moe.py` removes its work directory (about 800 MB) after a
  pass instead of leaving it in the temporary directory.

## [0.5.0] - 2026-11-13

### Added

- Qwen3.6-35B-A3B: `raceline pull qwen3.6-35b-a3b` (18.0 GB, from the
  `models-v2` release or Hugging Face). Qwen's newer 35B mixture of experts,
  same architecture as Qwen3.5-35B-A3B and much stronger at agentic coding
  in Qwen's evaluation. It is harder to quantize: KL 0.114 nats to the
  float model (Qwen3.5-35B-A3B: 0.060), where MLX's 4-bit format, measured
  by the converter on the same text, gets 0.121. On the agent tasks it
  solves 7 of 8 and, with the check fixed after the release, 7 of the 8
  harder ones (Qwen3.5-35B-A3B: 7 and 6).

- `raceline serve` keeps conversations set aside (#8). When a request starts
  another conversation (a second chat, or a side request such as a title in
  the middle of a long session), the server keeps the one it leaves: its
  recurrent state and its attention entries from where the two part. When a
  later request continues it, they are put back and nothing of it is
  computed again; before, the whole conversation was. Kept within a twelfth
  of the Mac's memory, at most 2 GB (`RACELINE_PARK_GB`), least recently
  used first out; conversations under 512 tokens past the shared start are
  cheaper to recompute and are not kept. The terminal agent holds one
  conversation and keeps none. Engine: `rl_save_kv` / `rl_load_kv` copy the
  attention entries of a range of positions in blocks of 256.

- `raceline serve` speaks Anthropic's Messages API as well:
  `POST /v1/messages` (streaming or not, tools, thinking) and
  `/v1/messages/count_tokens`, so clients written for it can use the local
  model: checked with the anthropic SDK, whose parser reads every event, and
  with Claude Code, which solved one of the agent tasks through it on
  Qwen3.5-35B-A3B. System messages inside a conversation (which Claude Code
  sends) become user messages there, for both APIs. Images are left out with
  a note, as the model runs without vision input.

- `raceline chat` and `serve` take `--ctx TOKENS`, the engine's context
  window (32768 by default; agents with long system prompts and tool lists
  need more). The attention cache costs 22 KB per token for the 35B-A3B
  models and 68 KB for Qwen3.8-27B.

- `raceline pull` without a model lists the models it can fetch, their
  sizes, and how each fits this Mac's memory.

### Changed

- The terminal agent keeps the reasoning of earlier replies in the prompt
  where the chat template has the option (`preserve_thinking`: Qwen3.6, and
  Qwen3.8, whose template keeps it by default), as Qwen recommends for
  agents. With Qwen3.6's default, a new request re-renders the previous turn
  without its reasoning, so everything from that turn's first reply on (tool
  outputs included) was processed again; now the whole conversation is
  reused. The Qwen3.5 templates have no such option. `raceline serve` takes
  `preserve_thinking` from the request (or its `chat_template_kwargs`).

### Fixed

- A conversation that does not fit the context window gets, from the OpenAI
  API, an HTTP 400 with OpenAI's `context_length_exceeded` code before any
  stream starts, which clients use to shorten the conversation; it was an
  error event inside the stream.

- On a new install, `raceline chat` and `serve` started with transformers'
  notice that PyTorch was not found and models would not be available.
  raceline uses transformers only for tokenizers, which need no PyTorch; the
  notice is silenced for that import.

## [0.4.0] - 2026-11-13

### Added

- Qwen3.5-35B-A3B: `raceline pull qwen3.5-35b-a3b` (18.3 GB, from the
  `models-v2` release or Hugging Face).

- Mixture-of-experts models (Qwen3.5-35B-A3B and the like, #5). The
  converter stacks each layer's experts into two 4-bit tensors in a file
  section of their own; the engine routes on the GPU, sorts the
  (token, expert) pairs by expert and runs every expert over just the rows
  routed to it in one grouped matmul, sized by the GPU (indirect dispatch).
  Prompts fed in chunks and speculative passes stay bit-identical, as with
  the dense models. `tests/check_moe.py` runs every check on a tiny random
  MoE model.

- Qwen3.5-35B-A3B on a 24 GB Mac: 56 to 121 tok/s with 73% of its experts in
  memory and the rest read from the SSD when needed; the agent solves 7 of
  the 8 tasks in 147 s at 61.5 tok/s. mlx-lm, which has to hold all 19.7 GB
  of MLX's build, swaps and decodes at 3.5 tok/s. As a 16 GB Mac would run
  it (5.5 GB of experts cached, misses read from the SSD): 15.5 tok/s, 26.7
  with drafts. KL 0.062 against the float model.

- Experts on disk. When a MoE model does not fit in the memory the GPU may
  use, the engine leaves the experts in the file and keeps the ones in use
  in a cache: after each layer's routing a CPU thread reads the missing
  experts from the SSD (in parallel, with `pread`) while the GPU waits.
  Prompt chunks do not count as uses (a chunk needs every expert, a scan
  that would flush plain LRU) and prefetch the next layer's experts while
  the GPU works. `--expert-cache GB` / `RACELINE_EXPERT_CACHE` set the
  cache (0: everything in memory); `RACELINE_EXPERT_NOCACHE=1` reads past
  the system's file cache, to measure a smaller Mac on a larger one.
  `bench/expert_cache_sim.py` replays routing traces
  (`RACELINE_EXPERT_TRACE`) to compare replacement policies.

- MoE models process prompts 2048 rows at a time (each missing expert is
  read once per pass): 1273 tok/s for a 4096-token prompt against 792.

- `raceline probe` measures SSD reads of an expert's size; `raceline chat`
  and `serve` say when a model's experts stay on disk and how much of them
  is kept in memory.

- The converter (`tools/quantize.py`, `tools/add_mtp.py`) holds several GB
  less per layer, and bounds MLX's pool of freed buffers; converting a MoE
  model still needs most of a 24 GB Mac (#31).

- Draft counts follow what a row of the verifying pass costs, measured
  (a least-squares line through rows and GPU time of recent passes): free
  on a dense model, not in a mixture of experts.

- `bench/agent_eval.py --set extended`: eight harder agent tasks closer to
  real work (a bug across a small package, a 600-line file, tests that must
  catch three of four broken versions of the code, a module and a script
  written from a description, a signature change across files, a command
  to implement in a CLI). The original eight stay the default (`core`).

- `bench/race_demo.py record-agent` records the terminal agent on one of the
  agent tasks, and `render --speed` replays faster than measured. The README
  now opens with the 27B agent fixing the same bug on mlx-lm and on raceline
  (102.7 s against 30.1 s, replayed at three times the speed); it is the
  only image, as the older chat replay looked like a second copy of it.

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

[Unreleased]: https://github.com/audenvenn/raceline/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/audenvenn/raceline/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/audenvenn/raceline/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/audenvenn/raceline/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/audenvenn/raceline/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/audenvenn/raceline/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/audenvenn/raceline/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/audenvenn/raceline/releases/tag/v0.2.0
[0.1.0]: https://github.com/audenvenn/raceline/releases/tag/v0.1.0
