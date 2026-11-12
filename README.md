# raceline

[![CI](https://github.com/audenvenn/raceline/actions/workflows/ci.yml/badge.svg)](https://github.com/audenvenn/raceline/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A local agent runtime for Apple Silicon with its own inference engine, built
to get as much out of one Mac, for one user, as the hardware allows.

On an M6 Mac mini with 24 GB it runs **Qwen3.8-27B at 25 to 40 tokens/s,
where mlx-lm manages 8 to 9** on the same machine, Qwen3.5-9B at 57 to 114
against 27 to 29, and Qwen3.5-4B at 94 to 178 against 46 to 51. Its agent
finishes eight small coding tasks with the 27B in 5 minutes where the same
agent on mlx-lm takes 39, and with the 9B, which fits a 16 GB Mac, in 2. The text it produces is
bit-for-bit what decoding one token at a time produces, at any sampling
temperature.

![The same agent fixing the same bug with Qwen3.8-27B on mlx-lm and on raceline, replayed at three times the measured speed](docs/agent_race.gif)

```
$ raceline chat qwen3.8-27b --yes -p "Run test_inventory.py. If a test fails, find the cause in inventory.py, fix it, and run the tests again until they all pass."

  > list_dir(path=".")
    inventory.py  [+1 lines]
  > find_files(pattern="*inventory*.py")
    inventory.py  [+1 lines]
  > read_file(path="inventory.py")
    1	"""A tiny inventory: items with a unit price and a quantity."""  [+24 lines]
  > read_file(path="test_inventory.py")
    1	from inventory import Inventory  [+22 lines]
  > bash(command="python3 test_inventory.py")
    [exit code 1]  [+7 lines]

The tests revealed two bugs in inventory.py:

1. `total_value()` is summing only the prices, ignoring quantity (it's returning 1.25 instead of 0.5×4 + 0.75×2 = 3.5).
2. `remove()` leaves items with a quantity of 0 behind (it only removes when the quantity goes *below* 0), so `test_remove_last` will fail after the first fix.

Fixing both:

  > edit_file(path="inventory.py", old_string="        return sum(item[\"price\"] for item in self.items.values())", new_string="        return sum(item[\"price\"] * item[\"quantity\"] for item in ...)
    Edited inventory.py.
  > edit_file(path="inventory.py", old_string="        if item[\"quantity\"] < 0:", new_string="        if item[\"quantity\"] <= 0:")
    Edited inventory.py.
  > bash(command="python3 test_inventory.py")
    [exit code 0]  [+1 lines]

All tests pass now. [...]
  [5 steps · context 2757 tok (2732 reused) · 853 out at 27.5 tok/s, 55% of drafts accepted · 38.2s]
```

(An actual run of Qwen3.8-27B on that Mac mini; the model's reasoning between the steps is left out.)

It is a terminal agent (`raceline chat`), an OpenAI-compatible server
(`raceline serve`) and a hardware probe (`raceline probe`) on top of a small
engine written for one model family, Qwen3.5 / Qwen3.8.

## Numbers

M6 Mac mini (12-core GPU, 24 GB), macOS 27.2, mlx-lm 0.32.2 on MLX 0.32.4.
Both runtimes are measured back to back through the same Python interface,
with nothing else running (`bench/results/` has the raw numbers; all rows of
a table come from one session, on 2026-11-12).

**The agent, end to end.** `bench/agent_eval.py` gives the agent eight small
coding tasks (a bug that failing tests point at, a rename across files, a
function to implement from its docstring, a new command-line option, defaults
for a config loader, docstrings without code changes, a rewrite from
`os.path` to `pathlib`, a binary-search bug), approves every tool call, and
lets a hidden check decide each task. Thinking on (effort low), seed 1, each
model's general-purpose sampling (temperature 1.0, top-p 0.95 for the 27B;
0.7, 0.8 for the 4B), except the 9B, measured later with the coding setting
`raceline chat` uses now (0.6, 0.95; it solves as many on the other two). The time is the whole run: prompt processing,
writing and the tools. Drafting never changes what is written, so the two
raceline rows of a model wrote the same tokens; mlx-lm runs other 4-bit
weights, which write other text:

| Model | Runtime | Tasks solved | Tokens written | Decoding | All eight tasks |
|---|---|---|---|---|---|
| Qwen3.8-27B | mlx-lm, MLX 4-bit weights | 7 of 8 | 19,823 | 8.8 tok/s | 2316 s |
| | **raceline** | 7 of 8 | 9,038 | **32.1 tok/s** | **317 s** |
| | raceline, copied drafts off | 7 of 8 | 9,038 | 31.5 tok/s | 322 s |
| Qwen3.5-9B | mlx-lm, MLX 4-bit weights | 5 of 8 | 13,318 | 27.4 tok/s | 500 s |
| | **raceline** | 7 of 8 | 11,513 | **99.9 tok/s** | **128 s** |
| | raceline, copied drafts off | 7 of 8 | 11,513 | 89.9 tok/s | 140 s |
| Qwen3.5-4B | mlx-lm, MLX 4-bit weights | 4 of 8 | 20,174 | 46.8 tok/s | 443 s |
| | **raceline** | 5 of 8 | 16,924 | **183.6 tok/s** | **161 s** |
| | raceline, copied drafts off | 5 of 8 | 16,924 | 150.5 tok/s | 181 s |

`--set extended` runs eight harder tasks for working on the agent (several
files, a 600-line file, tests that must catch broken versions of the code, a
new module from a description); the 9B solves 7 of them in 164 s.

The 27B's whole run is 7.3 times shorter on raceline, partly because these
weights wrote half as many tokens as MLX's on these tasks; decoding alone is
3.6 times faster. Copied drafts made 14 to 56% of the 4B's output and 13 to
44% of the 27B's.

**Qwen3.8-27B.** Continuing source code after a prompt of that length,
greedy, 128 tokens (`bench/agent_bench.py`). The Python standard library is
code the model has seen in training and partly remembers, which suits
drafts; this project's own code (`--corpus unseen`, four stretches) was
written after the model was:

| Text | Context | Runtime | Prompt processing | Decoding | Memory |
|---|---|---|---|---|---|
| standard library | 1,024 tokens | mlx-lm, MLX 4-bit weights | 277 tok/s | 8.7 tok/s | 18.2 GB |
| | | **raceline** | 245 tok/s | **40.2 tok/s** | 3.5 GB + 13.8 GB file mapped |
| | | raceline, drafts off | 247 tok/s | 9.1 tok/s | |
| | 4,096 tokens | mlx-lm, MLX 4-bit weights | 261 tok/s | 8.4 tok/s | 19.6 GB |
| | | **raceline** | 237 tok/s | **29.4 tok/s** | 3.5 GB + 13.8 GB file mapped |
| | | raceline, drafts off | 230 tok/s | 8.9 tok/s | |
| unseen code, 4 stretches | 1,024 tokens | mlx-lm, MLX 4-bit weights | 277 tok/s | 8.7 tok/s | |
| | | **raceline** | 230 tok/s | **29.4 tok/s** | |

Chat answers, thinking off, three requests of up to 300 tokens
(`bench/chat_bench.py`):

| Sampling | mlx-lm | raceline | raceline, drafts off |
|---|---|---|---|
| greedy | 8.6 tok/s | **26.4 tok/s** | 9.2 tok/s |
| temperature 0.7, top-p 0.8, top-k 20 | 8.7 tok/s | **25.4 tok/s** | 9.2 tok/s |
| temperature 1.0, top-p 0.95, top-k 20 | 8.7 tok/s | **25.9 tok/s** | 9.2 tok/s |

**Qwen3.5-9B.** The same benchmarks:

| Text | Context | Runtime | Prompt processing | Decoding | Memory |
|---|---|---|---|---|---|
| standard library | 1,024 tokens | mlx-lm, MLX 4-bit weights | 939 tok/s | 29.0 tok/s | 7.3 GB |
| | | **raceline** | 933 tok/s | **113.6 tok/s** | 2.1 GB + 4.7 GB file mapped |
| | | raceline, drafts off | 968 tok/s | 28.3 tok/s | |
| | 8,192 tokens | mlx-lm, MLX 4-bit weights | 869 tok/s | 26.5 tok/s | 8.6 GB |
| | | **raceline** | 862 tok/s | **66.4 tok/s** | 2.1 GB + 4.7 GB file mapped |
| | | raceline, drafts off | 867 tok/s | 26.4 tok/s | |
| unseen code, 4 stretches | 1,024 tokens | mlx-lm, MLX 4-bit weights | 938 tok/s | 27.9 tok/s | |
| | | **raceline** | 892 tok/s | **71.4 tok/s** | |

| Sampling | mlx-lm | raceline | raceline, drafts off |
|---|---|---|---|
| greedy | 27.8 tok/s | **61.3 tok/s** | 29.0 tok/s |
| temperature 0.7, top-p 0.8, top-k 20 | 27.6 tok/s | **60.8 tok/s** | 28.6 tok/s |
| temperature 1.0, top-p 0.95, top-k 20 | 27.8 tok/s | **56.8 tok/s** | 28.6 tok/s |

**Qwen3.5-4B.** The same benchmarks:

| Text | Context | Runtime | Prompt processing | Decoding | Memory |
|---|---|---|---|---|---|
| standard library | 1,024 tokens | mlx-lm, MLX 4-bit weights | 1664 tok/s | 50.3 tok/s | 4.7 GB |
| | | **raceline** | 1790 tok/s | **178.2 tok/s** | 2.0 GB + 2.5 GB file mapped |
| | | raceline, drafts off | 1858 tok/s | 46.5 tok/s | |
| | 8,192 tokens | mlx-lm, MLX 4-bit weights | 1554 tok/s | 45.9 tok/s | 6.0 GB |
| | | **raceline** | 1544 tok/s | **114.0 tok/s** | 2.0 GB + 2.5 GB file mapped |
| | | raceline, drafts off | 1630 tok/s | 41.2 tok/s | |
| unseen code, 4 stretches | 1,024 tokens | mlx-lm, MLX 4-bit weights | 1661 tok/s | 49.8 tok/s | |
| | | **raceline** | 1778 tok/s | **163.7 tok/s** | |

| Sampling | mlx-lm | raceline | raceline, drafts off |
|---|---|---|---|
| greedy | 50.6 tok/s | **100.2 tok/s** | 47.6 tok/s |
| temperature 0.7, top-p 0.8, top-k 20 | 48.9 tok/s | **98.1 tok/s** | 46.9 tok/s |
| temperature 1.0, top-p 0.95, top-k 20 | 48.9 tok/s | **93.5 tok/s** | 46.6 tok/s |

How to read this. Without drafts the engine decodes at about MLX's speed, and
it processes prompts at about MLX's speed with or without them (up to 12%
slower for the 27B, level for the 9B, up to 8% faster for the 4B): there is
no advantage there. The gain is the drafts, and it depends on how
predictable the text is: 3.4x to 4.6x for source code and about 3x for prose
on the 27B, 2x to 4x on the 9B and the 4B, where a draft costs more relative
to a pass. The files measured are the
published ones. Memory is the peak footprint of the process; the model file
is mapped read-only on top of that and its pages can be dropped and re-read
by the system.

## Where the speed comes from

A Mac generating text is limited by memory bandwidth: every weight is read
once per pass, so 150 GB/s and 2.1 GB of weights allow about 70 passes per
second, 12.9 GB about 12. `raceline probe` measures that ceiling on your
machine. The engine is built around getting more than one token out of a pass.

1. **Drafts from the model's own head, checked in the same pass.** Qwen3.5 /
   Qwen3.8 checkpoints ship a small multi-token-prediction (MTP) head that
   mlx-lm's converter drops. It guesses the token after next and, run again
   on its own output, the ones after that. The engine feeds the next token
   and up to 15 guesses through the model together; every guess the model
   confirms is a token that needed no pass of its own. On held-out text the
   first four guesses hold 79%, 71%, 64% and 59% of the time, each given the
   ones before it (Qwen3.5-4B; the 27B's first guess holds 82%). Where the
   text repeats what is already in the context, the guesses are copied from
   there instead (see below).
2. **A pass costs about the same for 1 or 16 rows.** The linear-layer kernel
   hands packed 4-bit weights straight to the GPU's neural accelerators
   (Metal 4 `matmul2d` with FP4 operands) over a layout made for streaming.
   The time goes into reading weights, so the extra rows of a speculative
   pass are nearly free. Attention is two such matrix products, so the cache
   is read once for all rows as well.
3. **Wrong guesses cost almost nothing.** Three quarters of these models'
   layers are recurrent (Gated DeltaNet) and cannot be rewound. A speculative
   pass leaves the old state next to the new one and keeps, per layer, what
   went into the recurrence; when only some guesses hold, the state is
   rebuilt for exactly those rows by replaying the recurrence, without a pass
   over the weights.
4. **4.0 bits per weight that the hardware decodes itself.** Weights are
   stored as `row scale x column scale x FP4 code`, quantized with GPTQ one
   layer at a time after a rotation that makes them even, with 11% less to
   read per pass than MLX's 4-bit format (4.5 bits). In held-out quality that
   is level with MLX's format for the 4B model and behind it for the 27B; the
   tables are below. The file is mapped, not loaded.
5. **Nothing is recomputed between turns.** The runtime keeps the model where
   the last request left it and snapshots the recurrent state at the points a
   conversation can branch from, so a follow-up request only processes what
   is new (2732 of 2757 context tokens reused in the run above).
6. **Tools do not copy the engine.** A shell command started the usual way
   forks the agent's process first, and with the 27B that process maps 14 GB
   of weights and GPU buffers: the fork takes seconds and the prompt
   processing after it runs several times slower. The agent starts commands
   with `posix_spawn`, in a process group of their own; on the agent tasks
   above that is the difference between 752 and 317 seconds.

MTP speculative decoding is not new on the Mac (llama.cpp and several MLX
based servers have it). What this project adds is an engine in which the
extra rows are close to free, drafts that stay exact under sampling, and the
numbers to check both.

## Quick start

Requirements: an Apple Silicon Mac whose GPU has neural accelerators (M5 / M6
families), macOS 27 or later, the Xcode command line tools and Python 3.10 or
later. It has been run on one machine, the M6 Mac mini above.

```sh
python3 -m venv ~/.raceline && source ~/.raceline/bin/activate
pip install git+https://github.com/audenvenn/raceline    # compiles the engine, half a minute

raceline pull  qwen3.5-4b     # 2.5 GB, converted and ready to run
raceline chat  qwen3.5-4b     # terminal agent in the current directory
raceline serve qwen3.5-4b     # OpenAI-compatible API on localhost:8080
raceline probe qwen3.5-4b     # this Mac's limits, and what the model reaches

raceline pull  qwen3.5-9b     # 4.7 GB; fits a 16 GB Mac, and the best agent per second
raceline pull  qwen3.8-27b    # 13.8 GB: the best; wants a Mac with 24 GB or more
```

Without the Xcode command line tools, install the wheel attached to the
[latest release](https://github.com/audenvenn/raceline/releases/latest)
instead: it has the engine compiled in.

`raceline pull` downloads from this repository's releases (`--source
huggingface` takes the same files from
[Hugging Face](https://huggingface.co/audenvenn)), verifies checksums and
continues an interrupted download.

To work on raceline itself, or to convert a checkpoint yourself:

```sh
git clone https://github.com/audenvenn/raceline && cd raceline
cmake -S . -B build/cmake -G Ninja && cmake --build build/cmake
python3 -m venv .venv && .venv/bin/pip install -e .

.venv/bin/python tools/fetch_model.py Qwen/Qwen3.5-4B                         # 9 GB, into models/
.venv/bin/python tools/quantize.py models/Qwen3.5-4B models/qwen3.5-4b.rlm    # about ten minutes
.venv/bin/python tools/add_mtp.py  models/Qwen3.5-4B models/qwen3.5-4b.rlm    # about a minute
.venv/bin/raceline chat models/qwen3.5-4b.rlm
```

A checkpoint is processed a layer at a time, so it never has to fit in
memory (Qwen3.8-27B is 55 GB and takes about an hour and a half on the M6),
and an interrupted conversion continues with `--resume`.

Any MLX model directory also runs, through the MLX backend and without the
engine's speedups: `raceline chat models/Qwen3.5-4B-MLX-4bit`.

## The agent

`raceline chat` gives the model seven tools. `read_file`, `list_dir`,
`find_files` and `search` run freely; `bash`, `write_file` and `edit_file`
ask first (`--yes` approves everything, answering `a` approves the rest of
the session). The line after each turn shows the size of the context, how
much of it was reused, and the decoding speed. `--no-think` skips the model's
reasoning phase; `--effort` sets the reasoning effort for models whose chat
template has one (Qwen3.8). It samples as Qwen recommends for coding
(temperature 0.6, top-p 0.95, top-k 20 with thinking; `--temperature`
overrides).

`raceline serve` speaks the OpenAI chat-completions protocol, streaming or not,
with tool calls and the model's reasoning as `reasoning_content`:

```sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Name three sorting algorithms."}],
  "enable_thinking": false}'
```

Responses carry `usage.prompt_tokens_details.cached_tokens` and a `timings`
object, so a client can see what a request cost. One conversation is served
at a time; this is a single-user runtime.

## Configuration

Environment variables, all optional:

| Variable | Meaning |
|---|---|
| `RACELINE_MODEL` | Model used when a command names none. |
| `RACELINE_CACHE` | Where pulled models and unpacked tokenizer files live (default `~/.cache/raceline`). |
| `RACELINE_MTP` | `0` turns drafts off: one token per pass. |
| `RACELINE_DRAFTS` | Most drafts per pass from the model's head (default 6, at most 15). |
| `RACELINE_LOOKUP` | Most drafts per pass copied from the context (default 15; `0` turns copying off). |
| `RACELINE_MIN_CONFIDENCE` | Fixed threshold below which drafting stops, instead of the measured one. |
| `RACELINE_DRAFT_VOCAB` | Token-id limit of a draft's vocabulary pass (`0` for the whole vocabulary; chosen from the context by default). |
| `RACELINE_SOURCE` | Where `raceline pull` downloads from by default: `github` or `huggingface`. |
| `RACELINE_RELEASES` | A base URL `raceline pull` downloads from instead. |
| `RACELINE_TOKENIZER` | Tokenizer directory for a model file that does not embed one. |
| `RACELINE_LIB`, `RL_KERNEL_DIR` | The engine library and the kernel sources, when they are not where the build left them. |
| `RACELINE_ANY_GPU` | `1` lets the engine try a Mac it does not support (before Apple M5); please report the result in #2. |

## What is checked

| Check | Result |
|---|---|
| Engine on MLX's own 4-bit weights against MLX in float32, layer by layer (`tests/verify_against_mlx.py`) | worst layer 1.5e-3 relative, same top token; MLX's bfloat16 path is ten times further from that reference |
| Engine on a native file against MLX running the decoded weights (`tests/verify_native.py`) | same top token, logits within 1.6e-2 relative (the reference holds the weights in float16) |
| MTP head against MLX on the same weights (`tests/verify_mtp.py`) | same draft, logits within 5e-3 relative |
| A prompt fed in chunks of 96, 17 or 1 tokens | bit-identical logits |
| Plain decoding against 1 to 15 drafts per pass (`raceline-engine spec`) | identical tokens |
| Passes with wrong, partly right and right drafts (chained by the head, or copied), abandoned drafts, state save and load, against a clean run (`tests/verify_spec.py`) | bit-identical logits and drafts, on both models |
| Greedy and sampled text with and without drafts, through the runtime; whole agent sessions with and without copied drafts (`bench/agent_eval.py`) | identical |
| Linear kernels against a double-precision CPU reference (`build/cmake/test_linear`) | pass |
| Session reuse and snapshots, tool-call parsing, the sampler, copied drafts, resuming a conversion, model downloads (`tests/test_runtime.py`); the server (`tests/test_server.py`); the agent tasks' checks (`tests/test_agent_tasks.py`). No GPU needed, run by CI | pass |

The comparisons with MLX in the first three rows are on Qwen3.5-4B: the float
27B model does not fit in this machine's memory.

Quantization quality against the float model on held-out text, 1152
positions (`tools/quantize.py` reports this for every conversion):

| Qwen3.5-4B | Bits per weight | KL (nats) | Perplexity | Top-1 agreement |
|---|---|---|---|---|
| MLX affine 4-bit, groups of 64 | 4.5 | 0.072 | 3.465 | 88.5% |
| **FP4, GPTQ, rotated (the published file)** | 4.0 | 0.072 | 3.417 | 91.1% |
| FP4, GPTQ, not rotated (raceline 0.1.0) | 4.0 | 0.083 | 3.379 | 88.6% |
| FP4, round to nearest, not rotated | 4.0 | 0.103 | 3.478 | 88.9% |

| Qwen3.5-9B | Bits per weight | KL (nats) | Perplexity | Top-1 agreement |
|---|---|---|---|---|
| MLX affine 4-bit, groups of 64 | 4.5 | 0.093 | 2.846 | 89.8% |
| **FP4, GPTQ, rotated (the published file)** | 4.0 | 0.080 | 2.836 | 91.1% |

| Qwen3.8-27B | Bits per weight | KL (nats) | Perplexity | Top-1 agreement |
|---|---|---|---|---|
| MLX affine 4-bit, groups of 64 | 4.5 | 0.105 | 1.861 | 92.2% |
| **FP4, GPTQ, rotated (the published file)** | 4.0 | 0.148 | 1.979 | 90.6% |
| FP4, GPTQ, not rotated (raceline 0.1.0) | 4.0 | 0.152 | 1.999 | 91.5% |

For the 4B model the rotation brings the 4-bit format level with MLX's at 0.5
bits per weight less, and for the 9B ahead of it. For the 27B it does not: the published file is still
measurably further from the float model than MLX's format, so the speed
tables compare weights of somewhat different quality there. Measured on what
the runtimes compute (`bench/kl_compare.py`), the typical position is close
(median KL 0.0039 against mlx-lm's 0.0032) and the gap is a tail: 5% of the
positions carry two thirds of the KL. The speed-up does
not come from the format (without drafts the two decode at about the same
speed). Where the 27B loses quality is tracked in
[#15](https://github.com/audenvenn/raceline/issues/15): the hidden states
drift little through the first 18 layers and then a bit more with every
DeltaNet layer; more calibration data, GPTQ act-order and keeping the
DeltaNet gates in half precision were measured and do not change that.

## How it works

**Model file (`.rlm`).** One file: a JSON header, then page-aligned sections
that are mapped and handed to Metal without a copy, plus the tokenizer and
chat template. Every large matrix is `W[n][k] = s[n] * c[k] * L[q[n][k]]`
with `L` the sixteen FP4 values the accelerators decode natively. The column
scales are folded into whatever produces the matrix's input and the row
scales are applied to the result while it is still in registers, so the
inner product itself is a single hardware matmul over raw 4-bit codes. Output
rows are stored in blocks of 256 as `[input][256 codes]`, which makes each
threadgroup read one contiguous stream.

**Rotation.** One scale per row and one per column, and none per group, is
what lets the accelerators read raw codes, but it handles uneven weights
badly: on Qwen3.8-27B the hidden states of a model converted that way drifted
from the float model's twice as fast per layer as with MLX's 4-bit format,
which has a scale and an offset for every 64 weights. So before quantizing,
the conversion rotates the residual stream by an orthogonal matrix `Q` (random
signs times Hadamard blocks): every norm's weight is folded into the matrices
that read it, those are multiplied by `Q`, the matrices that write the stream
by `Q^T`, the token table and output head accordingly. The model computes
the same function in other coordinates, and its weights become far more even
(the idea of QuaRot and QuIP). The rotation lives in the weights and costs
nothing at run time, except at the draft head, where a norm sits between the
stream and a weight: there the engine undoes it with a sign flip and a fast
Walsh-Hadamard transform, reading no weights. Relative drift of the hidden
states over the first layers of Qwen3.8-27B (`tools/quantize.py` prints it
for every layer):

| Layer | FP4, no rotation | MLX affine 4-bit (4.5 bits) | FP4, rotated (4.0 bits) |
|---|---|---|---|
| 1 | 0.071 | 0.035 | 0.025 |
| 3 | 0.084 | 0.048 | 0.046 |
| 6 | 0.085 | 0.046 | 0.045 |

**The MTP head.** `tools/mtp_study.py` measured what it is worth before any
engine work. Its top choice is the model's own next-next token 81% of the
time. It needs its attention history (56% without). Only its input projection
suffers from 4-bit storage (62% with that one matrix rounded to 4 bits, 79%
with everything else in 4 bits), so that matrix is kept in half precision.
Feeding the head's output back through its final norm, in place of the hidden
state the model has not computed yet, gives the further drafts. A draft's
pass over the output matrix is limited to the frequent part of the
vocabulary (token ids below 64k cover 98% of English and source code, below
128k 99% of Chinese), chosen from the recent context.

**Exact sampling with drafts.** A draft is only useful if it equals the token
the model then picks. With greedy decoding that is the model's top token; at
a temperature it is a random draw, and a guess that ignores the draw is
mostly wrong. Here every draw is a Gumbel-max whose noise depends only on
(seed, position, token), and the head's guess for a position uses the same
noise as the model's choice there. Each is still an exact sample from its
own distribution; they coincide whenever the two distributions are close.

**How many drafts.** Drafting continues while the chance that the chain so
far holds (the product of the drafts' probabilities under the head) is above
a threshold that follows the measured cost of a draft relative to a token.

**Drafts copied from the context.** An agent writes much of what it has
read: the lines an edit replaces, a file rewritten with one change, code it
quotes in its reasoning. The runtime remembers where every run of three
tokens last occurred in the context. When the newest three occurred before
and the head's first guess agrees with what followed them there, that
earlier continuation is drafted instead of a chain from the head, up to 15
tokens at once: a pass of 16 rows costs 3% more than a pass of one on the
27B (6% on the 4B), and the copy costs nothing to make. The head's agreement
filters out chance matches. Like every draft, a copied one only stands if it
is the token the model picks.

**Hardware limits.** `raceline probe` prints them for your Mac:

```
Apple M6 · 2+4+6 CPU cores · 12-core GPU · 24 GB · macOS 27.2

  GPU memory bandwidth        150.1 GB/s
  matrix multiply (half)       18.4 TFLOPS

  What that allows, per model size at 4 bits per weight:

  model      weights   fits   decode ceiling   prefill ceiling
  4B           2.3 GB    yes       65.3 tok/s        2189 tok/s
  9B           5.0 GB    yes       30.0 tok/s        1021 tok/s
  27B         13.6 GB    yes       11.0 tok/s         340 tok/s
  70B         35.5 GB     no        4.2 tok/s         131 tok/s
```

## Repository

| Path | What it is |
|---|---|
| `raceline/` | The runtime: backends (engine, MLX), session reuse and snapshots, chat rendering and tool-call parsing, OpenAI-compatible server, terminal agent, probe. |
| `src/`, `kernels/` | The engine: Metal runtime, model file reader, forward pass, speculative decoding. Kernels are compiled at run time. C API in `src/capi.h`. |
| `tools/quantize.py`, `tools/add_mtp.py` | Original checkpoint to native `.rlm`; adds the MTP head. `tools/convert.py` repacks an MLX 4-bit model instead (MLX's exact values, no drafts). |
| `tools/release_model.py`, `tools/publish_model.py` | Cut a model file into release parts with a model card; upload them to the GitHub release and Hugging Face. |
| `setup.py` | Builds the engine into the Python package on `pip install`. |
| `tools/mtp_study.py`, `tools/quant_study.py` | The measurements behind the design choices above. |
| `tests/`, `bench/` | Verification against MLX and CPU references; benchmarks and hardware probes. |

## Limits

- One model family: dense Qwen3.5 / Qwen3.8 text models. No MoE variants, no
  vision input.
- The 27B conversion loses more quality than MLX's 4-bit format does (KL
  0.148 against 0.105 nats to the float model on held-out text, #15; a
  power-of-two scale per 32 weights, which the accelerators could apply, did
  not help in simulation). The 4B is level with MLX's format and the 9B ahead.
- Without drafts, decoding is at 70% of the bandwidth ceiling for the 4B
  model and 84% for the 27B: a pass is some 380 to 760 GPU dispatches, many
  of them small kernels.
- Prompt processing is at MLX's speed or slightly below. The accelerators run 8-bit activations
  against 4-bit weights about 1.6x faster than half precision; that path is
  not built and its accuracy is not measured.
- One conversation context. Alternating between different system prompts
  reprocesses the prompt each time.
- Run on a single M6 machine. Engine errors (an unsupported model geometry,
  misuse of the C API) end the process with a message.
- Converted files are published for the models named above; anything else
  has to be converted locally from the original checkpoint.

## Contributing

Issues and pull requests are welcome; [CONTRIBUTING.md](CONTRIBUTING.md) has
the ground rules and the test matrix, [CHANGELOG.md](CHANGELOG.md) what
changed. The most useful thing right now is a
[benchmark from another Mac](https://github.com/audenvenn/raceline/issues/2):
everything above was measured on one machine. Security problems go through
[SECURITY.md](SECURITY.md).

## License

Apache-2.0. Model weights are not part of this repository and keep their own
licenses (the Qwen models used here are Apache-2.0).
