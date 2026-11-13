# raceline

[![CI](https://github.com/audenvenn/raceline/actions/workflows/ci.yml/badge.svg)](https://github.com/audenvenn/raceline/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A local agent runtime for Apple Silicon with its own inference engine, built
to get as much out of one Mac, for one user, as the hardware allows.

On an M6 Mac mini with 24 GB it runs **Qwen3.8-27B at 25 to 40 tokens/s,
where mlx-lm manages 8 to 9** on the same machine, Qwen3.5-9B at 57 to 114
against 27 to 29, and Qwen3.5-4B at 94 to 178 against 46 to 51.
**Qwen3.6-35B-A3B, a mixture of experts too large for the memory the GPU may
use, runs at 61 to 118 tokens/s with its experts read from the SSD as they
are needed; mlx-lm has to hold all of such a model and swaps (3.5 tok/s on
Qwen3.5-35B-A3B, the same size).** Its agent
finishes eight small coding tasks with the 27B in 5 minutes where the same
agent on mlx-lm takes 39, and with the 9B, which fits a 16 GB Mac, in 2. The text it produces is
bit-for-bit what decoding one token at a time produces, at any sampling
temperature. It reads images too, with each model's own vision tower:
screenshots and diagrams, in the agent and through both APIs.

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
`raceline chat` uses now (0.6, 0.95; it solves as many on the other two),
and the two 35B-A3B models, measured with it a day later with the GPU
keep-alive (the date is in the agent's system prompt, so their text differs
from the day before). The time
is the whole run: prompt processing, writing and the tools. Drafting never changes what is written, so the two
raceline rows of a model wrote the same tokens; mlx-lm runs other 4-bit
weights, which write other text:

| Model | Runtime | Tasks solved | Tokens written | Decoding | All eight tasks |
|---|---|---|---|---|---|
| Qwen3.8-27B | mlx-lm, MLX 4-bit weights | 7 of 8 | 19,823 | 8.8 tok/s | 2316 s |
| | **raceline** | 7 of 8 | 9,038 | **32.1 tok/s** | **317 s** |
| | raceline, copied drafts off | 7 of 8 | 9,038 | 31.5 tok/s | 322 s |
| Qwen3.6-35B-A3B | mlx-lm, MLX 4-bit weights | does not fit | | | |
| | **raceline**, experts on disk | 7 of 8 | 8,384 | **70.3 tok/s** | **148 s** |
| Qwen3.5-35B-A3B | mlx-lm, MLX 4-bit weights | does not fit | | | |
| | **raceline**, experts on disk | 7 of 8 | 6,993 | **67.1 tok/s** | **128 s** |
| Qwen3.5-9B | mlx-lm, MLX 4-bit weights | 5 of 8 | 13,318 | 27.4 tok/s | 500 s |
| | **raceline** | 7 of 8 | 11,513 | **99.9 tok/s** | **128 s** |
| | raceline, copied drafts off | 7 of 8 | 11,513 | 89.9 tok/s | 140 s |
| Qwen3.5-4B | mlx-lm, MLX 4-bit weights | 4 of 8 | 20,174 | 46.8 tok/s | 443 s |
| | **raceline** | 5 of 8 | 16,924 | **183.6 tok/s** | **161 s** |
| | raceline, copied drafts off | 5 of 8 | 16,924 | 150.5 tok/s | 181 s |

`--set extended` runs eight harder tasks (several files, a 600-line file,
tests that must catch broken versions of the code, a new module from a
description, a signature change across files). They separate the models
more than the first eight (all four on 2026-11-13, the 35B-A3B models with
the GPU keep-alive described under "How it works"):

| Model on raceline | First eight tasks | Extended eight |
|---|---|---|
| Qwen3.8-27B | 7 of 8 in 317 s | **8 of 8** in 310 s |
| Qwen3.6-35B-A3B, experts on disk | 7 of 8 in 148 s | **7 of 8** in 144 s |
| Qwen3.5-9B | 7 of 8 in 128 s | 6 of 8 in 155 s |
| Qwen3.5-35B-A3B, experts on disk | 7 of 8 in 128 s | 6 of 8 in 163 s |

One run each with seed 1: a task more or less is within the noise. The 27B
is the strongest agent here and the slowest; Qwen3.6-35B-A3B solves nearly
as much in about half the time.

**A conversation's next request.** `--followup` asks one more question after
each task, as a user does in a session. The Qwen3.5 templates leave the
reasoning of earlier replies out of the prompt, so the previous turn is
rendered differently from what the model wrote and is processed again; on
Qwen3.5-35B-A3B the eight follow-ups processed 10,955 of their 20,019 prompt
tokens again (9.6 s). Qwen3.6 and Qwen3.8 can keep that reasoning
(`preserve_thinking`), which `raceline chat` asks for: on Qwen3.6-35B-A3B
the follow-ups processed 160 of 24,579 tokens (1.4 s), 20 per request, the
new message itself; with the template's default it was 12,270 of 21,334.

The 27B's whole run is 7.3 times shorter on raceline, partly because these
weights wrote half as many tokens as MLX's on these tasks; decoding alone is
3.6 times faster. Copied drafts made 14 to 56% of the 4B's output and 13 to
44% of the 27B's.

**Qwen3.6-35B-A3B and Qwen3.5-35B-A3B.** Mixtures of experts with the same
shape: 35B parameters, of which a token uses 3B. raceline's files are 18.0
and 18.3 GB, MLX's 4-bit builds 19.7 GB, and Metal lets the GPU of a 24 GB
Mac use 19.1 GB. raceline keeps the 73% of the experts that fit in memory,
the rest on disk, and reads what a layer needs when it is missing (3 to 8%
of the routed experts; an expert, 1.6 MB, takes 0.4 ms from the SSD).
mlx-lm loads the whole model: after a 1,024-token prompt Qwen3.5-35B-A3B
decoded at 3.5 tok/s with 21.3 GB in use, swapping; its 4,096-token run was
still going after 11 minutes when swap passed 8 GB and a watchdog stopped
it, as it stopped mlx-lm's other runs. The two raceline columns were
measured back to back on 2026-11-13 with the GPU keep-alive described below;
mlx-lm's row on the 12th:

| Benchmark | mlx-lm, MLX 4-bit (3.5) | raceline, Qwen3.6-35B-A3B | raceline, Qwen3.5-35B-A3B |
|---|---|---|---|
| standard library, 1,024 tokens: prompt / decoding | 290 / 3.5 tok/s, swapping | 1171 / **118.0 tok/s** | 1118 / **120.1 tok/s** |
| standard library, 4,096 tokens: prompt / decoding | stopped (swap) | 1542 / **74.5 tok/s** | 1647 / **97.3 tok/s** |
| chat answers, greedy / temperature 0.7 / 1.0 | | **61.4 / 68.7 / 71.5 tok/s** | **60.0 / 64.8 / 63.0 tok/s** |
| memory | 21.3 GB and swap | 14.4 GB | 14.1 GB |

Without drafts both decode at about 55 tok/s (`raceline-engine bench`,
1,024 tokens of context), with them at 118. As a 16 GB Mac would run them,
with 5.5 to 5.8 GB of experts in memory and every miss read from the SSD
past the file cache (`RACELINE_EXPERT_CACHE=<GB> RACELINE_EXPERT_NOCACHE=1`):
30.1 tok/s for Qwen3.6-35B-A3B and 25.6 for 3.5, 39.8 and 32.5 with drafts,
prompts at about 750 to 830 tok/s. Before the keep-alive, the same runs gave
15.6 and 15.5 (28.1 and 26.7 with drafts), and on a 24 GB Mac 32 to 53 tok/s
without drafts depending on the session, 61.4 against 45.5 greedy in chat
(#47). Prompts are processed 2,048 rows at a
time for MoE models: a pass reads each missing expert once.

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

raceline pull                 # the models, and how each fits this Mac
raceline pull  qwen3.5-4b     # 2.5 GB, converted and ready to run
raceline chat  qwen3.5-4b     # terminal agent in the current directory
raceline chat  qwen3.5-4b --image screenshot.png -p "What does this show?"
raceline serve qwen3.5-4b     # OpenAI-compatible API on localhost:8080
raceline probe qwen3.5-4b     # this Mac's limits, and what the model reaches

raceline pull  qwen3.5-9b         # 4.7 GB; fits a 16 GB Mac, and the best agent per second
raceline pull  qwen3.8-27b        # 13.8 GB, dense; wants a Mac with 24 GB or more
raceline pull  qwen3.6-35b-a3b    # 18.0 GB, a mixture of experts: its experts can stay on the SSD
raceline pull  qwen3.5-35b-a3b    # 18.3 GB, its predecessor
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

The models read images. `raceline pull` fetches a model's vision file with
it; then `--image FILE` sends an image with the first message, `/image FILE`
with the next one, and `read_file` shows the model an image file (PNG, JPEG,
GIF, WebP, BMP, TIFF) instead of its bytes.

`raceline serve` speaks the OpenAI chat-completions protocol, streaming or not,
with tool calls and the model's reasoning as `reasoning_content`:

```sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Name three sorting algorithms."}],
  "enable_thinking": false}'
```

It also speaks Anthropic's Messages API (`POST /v1/messages`, streaming,
tools, thinking blocks), so clients written for that API can use the local
model, for example the anthropic SDK with `base_url="http://localhost:8080"`.
Claude Code pointed at it (`ANTHROPIC_BASE_URL`) solved the `inventory` agent
task on Qwen3.5-35B-A3B in 52 s, with every request reusing the previous one.
[docs/serve.md](docs/serve.md) has examples for both APIs and for Claude Code.
Images in either API go to the model's vision tower
([docs/serve.md](docs/serve.md#images)). The context window is 32768 tokens unless `--ctx` sets another (agents with long system
prompts and tool lists need more; the models take up to 262144).

Responses carry `usage.prompt_tokens_details.cached_tokens` and a `timings`
object, so a client can see what a request cost. Requests are served one at
a time; this is a single-user runtime. A conversation the client leaves for
another one (a second chat, or a side request such as a title) is set aside
with its recurrent state and attention entries, and continued later without
being processed again (within a twelfth of the Mac's memory, at most 2 GB;
`RACELINE_PARK_GB` sets it).

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
| `RACELINE_EXPERT_CACHE` | Mixture-of-experts models: GB of experts kept in memory, the rest read from disk when needed (`0`: all in memory; by default the engine decides from the memory the GPU may use). |
| `RACELINE_EXPERT_PRELOAD` | `0` stops the background filling of a fresh expert cache (on by default; see "Experts on disk"). |
| `RACELINE_GPU_KEEPALIVE` | `0` stops the light kernel that keeps the GPU busy while a mixture of experts waits for experts read from disk (on by default; see "Experts on disk"). |
| `RACELINE_IMAGE_PIXELS` | Pixels an image is scaled down to at most (default 1048576, a megapixel: 1024 rows of context; the models take up to 16.7 million). |
| `RACELINE_VISION` | The vision file to use, when it is not the one next to the model file. |
| `RACELINE_PARK_GB` | `raceline serve`: memory for conversations set aside while the client is in another one (default: a twelfth of the Mac's memory, at most 2; `0` keeps none). |
| `RACELINE_ANY_SIZE` | `1` loads a model whose weights, attention cache and scratch need more GPU memory than Metal recommends for this Mac (refused by default: such a model fails halfway through a long prompt). |
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
| Mixture of experts, on a tiny random MoE model (`tests/check_moe.py`):
- conversion;
- the engine against MLX on the decoded weights;
- chunks of 96, 17 and 1 tokens with the experts in memory, on disk with the smallest cache, and on disk 70% cached;
- speculative passes. | same top token; every run bit-identical |
| Qwen3.5-35B-A3B's first 4 layers against MLX on the decoded weights (`tests/verify_moe.py`), experts in memory and on disk | logits within 1.5e-3 relative, same top token; bit-identical with the experts on disk |
| Image processor and vision tower against transformers' (`tests/verify_vision.py`, references from `tests/reference_image.py`) | patches identical; the tower's output within 8e-5 relative in float32, cosine 0.99998 in float16 |
| Image input end to end on Qwen3.5-4B against transformers (`tests/verify_image_input.py`) | every rotary position equal, rotary angles within 1.9e-6; on the answer after an image KL 0.072 nats and 94.7% top-1 agreement, against 0.165 and 89.8% for a prompt without one; 1.65 nats with the image's rows zeroed |
| Images through the chat runtime (`tests/check_image_chat.py`) | read right; a follow-up reuses the image's rows; another image of the same size is not taken for it; set aside and continued without computing its rows again |
| Snapshots taken inside a prompt pass against a pass that ends there (`tests/verify_marks.py`), on Qwen3.5-4B and Qwen3.6-35B-A3B | the same state byte for byte, the same logits, the same drafted tokens after restoring |
| Session reuse and snapshots, tool-call parsing, the sampler, copied drafts, resuming a conversion, model downloads, image parts and their rows (`tests/test_runtime.py`); the server (`tests/test_server.py`); the agent tasks' checks (`tests/test_agent_tasks.py`). No GPU needed, run by CI | pass |

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

| Qwen3.6-35B-A3B | Bits per weight | KL (nats) | Perplexity | Top-1 agreement |
|---|---|---|---|---|
| MLX affine 4-bit, groups of 64 (the converter's `--book mlx-affine4`) | 4.5 | 0.121 | 2.109 | 91.1% |
| **FP4, GPTQ (down projections of the experts rounded to nearest), rotated (the published file)** | 4.0 | **0.114** | 2.113 | 90.8% |

| Qwen3.5-35B-A3B | Bits per weight | KL (nats) | Perplexity | Top-1 agreement |
|---|---|---|---|---|
| MLX affine 4-bit, groups of 64 | 4.5 | not measured | | |
| **FP4, GPTQ (down projections of the experts rounded to nearest), rotated (the published file)** | 4.0 | 0.060 | 2.209 | 92.1% |

Qwen3.6-35B-A3B loses about twice as much to 4 bits as Qwen3.5-35B-A3B,
whichever the format: the converter's own measurement of MLX's format (round
to nearest in groups of 64, on the same held-out text; mlx-lm itself swaps
on this Mac) lands at the same place. The drift of the hidden states grows
in its last ten layers.

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
DeltaNet gates in half precision were measured and do not change that, nor
would a low-rank correction of the weights' error (`tools/lowrank_study.py`:
rank 32 removes 1 to 2.5% of a matrix's output error, rank 128 at most 6%;
after the rotation the error has no low-rank structure to catch).

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

**Mixture of experts.** In Qwen3.5-35B-A3B every layer's MLP is 256
experts, of which a router picks 8 per token, plus a shared expert that every
token uses. The converter stacks a layer's experts into two 4-bit tensors
(gate/up and down, the shared expert last, so it runs through the same code)
in a file section of their own. In the engine a routing kernel picks each
token's experts, sorts the (token, expert) pairs by expert and writes a work
list on the GPU; one grouped matmul then runs every expert over just the rows
routed to it, dispatched with the sizes the GPU computed, and a last kernel
adds each token's nine results in a fixed order. Nothing a token gets depends
on the other tokens in the pass, so prompts fed in chunks and speculative
passes stay bit-identical, as with the dense models.

**Experts on disk.** The experts are most of a MoE model's weights, and a
token touches few of them. When a model does not fit in the memory the GPU
may use, the engine leaves the experts in the file and keeps the ones in use
in a cache: after each layer's routing the GPU waits while a CPU thread reads
the missing experts from the SSD into the least recently used slots (all of
a layer's misses at once) and updates the slot table the grouped matmuls read
through. What is computed does not change, only where the weights come from.
A prompt chunk needs nearly every expert of every layer, a cyclic scan in which
least-recently-used replacement evicts just what the next layers need; prompt
chunks therefore do not count as uses, and what they load goes in at the cold
end of the cache. Replayed on synthetic agent sessions
(`bench/expert_cache_sim.py`) with 73% of the experts cached, that reads 58%
fewer experts for prompts than plain LRU and 6% fewer while decoding.
`--expert-cache GB` sets the cache's size by hand.

Those waits cost more than their own length. The GPU's power management
takes a GPU that waits on the CPU for an idle one and lowers its
performance, so everything after the wait runs slower: on the M6, a light
kernel running alongside on another command queue made single-token passes
of Qwen3.6-35B-A3B 59% faster, and that of the tiny test model 2.6 times
(#47). While passes with experts on disk come in, the engine therefore keeps
such a kernel running back to back on a queue of its own and stops it 50 ms
after the last one (`RACELINE_GPU_KEEPALIVE=0` turns it off). Dense models
do not wait on the CPU inside a pass and gain nothing from it.

A fresh cache is empty, and the first prompt would read nearly every expert
from the SSD. While no pass is waiting, the CPU thread therefore fills the
free slots in the background, a hundred megabytes at a time and never
evicting anything; a request that comes in waits for one such step at most.
On the 24 GB Mac a 3,490-token first request to Qwen3.6-35B-A3B, sent five
seconds after the model opened, takes 2.6 s instead of 3.7
(`RACELINE_EXPERT_PRELOAD=0` turns the preload off).

**Images.** The Qwen3.5 / 3.6 / 3.8 checkpoints carry a vision tower, which
the `.rlm` conversion leaves out: `tools/convert_vision.py` writes it to a
vision file of its own that `raceline pull` fetches with the model. The tower
is a ViT over 16-pixel patches (24 layers for the 4B, 27 for the others) and
runs in MLX (`raceline/vision.py`), in float16: in 8 bits its output was a
hundred times further from transformers'. On the M6 it takes 0.6 s for a
1280 x 800 screenshot with the 4B's tower and 1.1 s with the others'. Its
output, one vector per 2 x 2 patches, enters the engine in place of the
image's `<|image_pad|>` rows (`forward_embeds`), rotated like the model's
residual stream, with Qwen's three-dimensional rotary positions: time, row
and column, interleaved over the 32 rotary frequencies. Text has all three
equal, so text comes out bit-for-bit as before. In the session an image's
rows are keyed by the image's content, so a conversation that goes on reuses
them and another image of the same size is never taken for it; their
positions travel with a conversation set aside. Qwen3.6-35B-A3B on the 24 GB
Mac, experts on disk: a 1280 x 800 screenshot and a question (1021 rows) are
read in 3.4 s, tower included, and text decodes at 74 tok/s before and after
it, without swapping: the tower's 0.9 GB fit beside the model.

**Hardware limits.** `raceline probe` prints them for your Mac:

```
Apple M6 · 2+4+6 CPU cores · 12-core GPU · 24 GB · macOS 27.2

  GPU memory bandwidth        149.8 GB/s
  matrix multiply (half)       19.4 TFLOPS
  SSD, 1.6 MB random reads      4.2 GB/s one at a time, 7.4 GB/s eight at a time

  What that allows, per model size at 4 bits per weight:

  model      weights   fits   decode ceiling   prefill ceiling
  4B           2.3 GB    yes       65.1 tok/s        2304 tok/s
  9B           5.0 GB    yes       30.0 tok/s        1075 tok/s
  27B         13.6 GB    yes       11.0 tok/s         358 tok/s
  35B-A3B     18.6 GB    SSD       99.9 tok/s        3225 tok/s
  70B         35.5 GB     no        4.2 tok/s         138 tok/s
```

A mixture of experts reads only the experts its tokens are routed to, so its
decode ceiling counts the 1.5 GB a token of Qwen3.5-35B-A3B reads, not the
whole model. "SSD": the model does not fit in the memory the GPU may use, and
the experts that are not in memory are read from the SSD when a layer needs
them, one expert (1.6 MB) in 0.4 ms on this Mac.

## Repository

| Path | What it is |
|---|---|
| `raceline/` | The runtime: backends (engine, MLX), session reuse and snapshots, chat rendering and tool-call parsing, OpenAI-compatible server, terminal agent, probe. |
| `src/`, `kernels/` | The engine: Metal runtime, model file reader, forward pass, speculative decoding. Kernels are compiled at run time. C API in `src/capi.h`. |
| `tools/quantize.py`, `tools/add_mtp.py` | Original checkpoint to native `.rlm`; adds the MTP head. `tools/convert.py` repacks an MLX 4-bit model instead (MLX's exact values, no drafts). |
| `tools/convert_vision.py` | The checkpoint's vision tower to a vision file (float16) for image input. |
| `tools/release_model.py`, `tools/publish_model.py` | Cut a model file into release parts with a model card; upload them to the GitHub release and Hugging Face. |
| `setup.py` | Builds the engine into the Python package on `pip install`. |
| `tools/mtp_study.py`, `tools/quant_study.py`, `tools/act_study.py`, `tools/lowrank_study.py` | The measurements behind the design choices above: what 8-bit activations would cost (#6), what a low-rank correction of the weights would gain (#15). |
| `tests/`, `bench/` | Verification against MLX and CPU references; benchmarks and hardware probes. |

## Limits

- One model family: Qwen3.5 / 3.6 / 3.8, dense and mixture of experts.
  Images, but not video.
- Images go through the vision tower in MLX rather than the engine: 0.6 to
  1.1 s per screenshot, and the tower stays in memory (0.67 GB for the 4B,
  about 0.9 GB for the others) once an image has come; with a mixture of
  experts whose experts are on disk it is read in again for each request
  that has one (0.1 to 0.3 s) and its memory handed back after.
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
- One request at a time. Conversations a client switches between are set
  aside within a memory budget; beyond it, the least recently used one is
  processed again when it comes back.
- Run on a single M6 machine. Engine errors (an unsupported model geometry,
  misuse of the C API) end the process with a message.
- Converted files are published for the models named above; anything else
  has to be converted locally from the original checkpoint.

## Related

[h3metal](https://github.com/audenvenn/h3metal): image generation on the
matrix units of M5/M6 Macs with its own Metal engine. It runs FLUX.2 [klein]
4B/9B, Z-Image-Turbo and MiniMax-H3-Image with LoRA adapters, 2.1-2.3x
faster than mflux.

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
