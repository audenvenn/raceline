# Using `raceline serve` from other clients

`raceline serve` runs one model for one user and speaks two protocols on the
same port: OpenAI's chat completions (`/v1/chat/completions`) and Anthropic's
Messages API (`/v1/messages`). Anything that lets you set a base URL for
either can use the local model.

```sh
raceline serve qwen3.5-35b-a3b --ctx 65536      # http://127.0.0.1:8080
```

## Context window

The engine's context window is 32768 tokens unless `--ctx` sets another.
Agents with long system prompts and many tools need more. A request that
does not fit is refused; Anthropic's API answers "prompt is too long: N
tokens > M maximum", which clients take as a cue to compact the
conversation. The models take up to 262144 tokens. The attention cache costs
22 KB per token for Qwen3.5-35B-A3B (1.4 GB at 65536) and 68 KB for
Qwen3.8-27B.

## OpenAI's API

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
reply = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Name three sorting algorithms."}],
    extra_body={"enable_thinking": False},
)
print(reply.choices[0].message.content)
```

Tool calls, streaming and `stream_options.include_usage` work as in OpenAI's
API. The model's reasoning arrives as `reasoning_content`. Extensions, all
optional: `top_k`, `repetition_penalty`, `enable_thinking` (also inside
`chat_template_kwargs`), and `reasoning_effort` for the models whose template
has one (Qwen3.8).

## Anthropic's API

```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="local")
message = client.messages.create(
    model="local", max_tokens=1024,
    messages=[{"role": "user", "content": "Name three sorting algorithms."}],
)
print(message.content[0].text)
```

Streaming, tools, `stop_sequences` and `count_tokens` work as in Anthropic's
API. `thinking: {"type": "enabled"}` turns the model's reasoning on and
returns it as thinking blocks; without it the model answers directly.
Server-side tools (web search and the like) are left out, images are
replaced by a note (no vision input yet, #28), and forcing a tool with
`tool_choice` is treated as `auto`. `usage.input_tokens` counts what was
processed and `cache_read_input_tokens` what the session reused.

### Claude Code

Claude Code takes the server's address from `ANTHROPIC_BASE_URL`:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=local \
  claude --model qwen3.5-35b-a3b --tools "Bash,Edit,Read,Write,Glob,Grep"
```

The model name is only a label (the server runs the model it was started
with), and the API key is not checked. Fewer tools mean a shorter prompt
(five thousand tokens with these six). On Qwen3.5-35B-A3B it solved the
`inventory` task of `bench/agent_tasks.py` in 52 s, every request reusing all
of the previous one. A local model is not Claude: expect it to need more
guidance on larger tasks.

## What is reused between requests

Requests are served one at a time. The server keeps the model where the last
request left it, so a client that resends a growing conversation only pays
for what is new.

A client that switches to another conversation and back (a second chat, or a
side request such as a title in the middle of a long session) finds the
first one set aside with its recurrent state and attention entries, and
nothing of it is processed again. On Qwen3.5-35B-A3B with 8,000 tokens of
history, the first token after a side request comes in 0.33 s instead of
6.47 s (`bench/switch_bench.py`). `RACELINE_PARK_GB` sets the memory for
this (by default a twelfth of the Mac's memory, at most 2 GB).
