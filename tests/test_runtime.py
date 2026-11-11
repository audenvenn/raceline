#!/usr/bin/env python3
"""Tests of the agent runtime that need no model and no GPU.

  .venv/bin/python tests/test_runtime.py        (or: pytest tests/test_runtime.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from raceline.backends import Backend, Sampling, Snapshot  # noqa: E402
from raceline.chat import normalize_messages, parse_tool_call  # noqa: E402
from raceline.session import Session  # noqa: E402


class FakeBackend(Backend):
    """A 'model' whose next token is a hash of its whole context, with a
    recurrent state that cannot be rewound: only snapshots bring it back."""

    def __init__(self):
        self.reset()
        self.fed = 0

    def reset(self):
        self.ctx, self.state = [], 0

    @property
    def position(self):
        return len(self.ctx)

    def _feed(self, t):
        self.ctx.append(t)
        self.state = (self.state * 1000003 + t + 1) % (1 << 61)
        self.fed += 1

    def prefill(self, tokens, progress=None):
        for t in tokens:
            self._feed(t)

    def generate(self, sampling, max_tokens, history):
        for _ in range(max_tokens):
            t = self.state % 97
            self._feed(t)
            yield t

    def snapshot(self):
        return Snapshot(self.position, self.state, None, 8)

    def restore(self, snap):
        assert snap.position <= self.position
        del self.ctx[snap.position:]
        self.state = snap.state


def reference_state(tokens):
    b = FakeBackend()
    b.prefill(tokens)
    return b.state


def test_session_extends_without_refeeding():
    b = FakeBackend()
    s = Session(b)
    assert s.sync([1, 2, 3, 4]) == dict(prompt_tokens=4, cached_tokens=0)
    out = list(s.generate(Sampling(), 5))
    prompt = [1, 2, 3, 4] + out + [9, 9]
    b.fed = 0
    assert s.sync(prompt)["cached_tokens"] == 9
    assert b.fed == 2 and b.ctx == prompt and b.state == reference_state(prompt)


def test_session_resumes_from_the_nearest_snapshot():
    b = FakeBackend()
    s = Session(b)
    system, user = list(range(10, 30)), list(range(40, 50))
    s.sync(system + user, branch_points=[len(system)], pin=len(system))
    list(s.generate(Sampling(), 8))
    other = system + [7, 7, 7]                       # same system prompt, different conversation
    b.fed = 0
    use = s.sync(other, branch_points=[len(system)], pin=len(system))
    assert use["cached_tokens"] == len(system) and b.fed == 3
    assert b.ctx == other and b.state == reference_state(other)
    b.fed = 0                                        # diverging before any snapshot: start over
    assert s.sync([5] + other)["cached_tokens"] == 0 and b.fed == len(other) + 1


def test_session_stops_on_stop_token():
    b = FakeBackend()
    s = Session(b)
    s.sync([3, 1, 4])
    first = b.state % 97
    out = list(s.generate(Sampling(), 50, stop_ids={first}))
    assert out == [first] and s.tokens == [3, 1, 4, first] and b.ctx == s.tokens


def test_snapshot_budget_keeps_the_pinned_one():
    b = FakeBackend()
    s = Session(b, max_snapshots=3)
    prompt = list(range(200))
    s.sync(prompt, branch_points=[10, 50, 90, 130, 170], pin=10)
    positions = [x.position for x in s._snaps]
    assert len(positions) == 3 and 10 in positions and positions[-1] == 200


def test_tool_call_parsing_uses_the_schema_for_types():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "n": {"type": "integer"}, "flags": {"type": "array"}}}}}]
    text = ("\n<function=f>\n<parameter=path>\n123\n</parameter>\n<parameter=n>\n42\n</parameter>\n"
            "<parameter=flags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=note>\nline one\nline two\n</parameter>\n</function>\n")
    assert parse_tool_call(text, tools) == [{"name": "f", "arguments": {
        "path": "123", "n": 42, "flags": ["a", "b"], "note": "line one\nline two"}}]
    # A call cut off before its closing tag still yields what was complete.
    assert parse_tool_call("<function=f>\n<parameter=n>\n7\n</parameter>\n<parameter=path>\n/tm", tools) == [
        {"name": "f", "arguments": {"n": 7}}]


def test_messages_are_normalized_for_the_template():
    msgs = normalize_messages([
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "f", "arguments": "{\"n\": 1}"}}]}])
    assert msgs[0]["content"] == "ab" and msgs[1]["content"] == ""
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"n": 1}


def test_sampler_matches_its_distribution_and_couples():
    try:
        from raceline.engine import _library, _sample
        _library()
    except Exception as e:  # the engine library is not built
        print(f"  (skipped: {e})")
        return
    rng = np.random.default_rng(0)
    z = (rng.normal(size=4000) * 2).astype(np.float32)
    s = Sampling(temperature=0.8, top_p=0.9, top_k=20)
    top = np.argsort(-z, kind="stable")[:20]
    p = np.exp((z[top] - z[top[0]]).astype(np.float64) / 0.8)
    p /= p.sum()
    keep = int(np.searchsorted(np.cumsum(p), 0.9)) + 1
    p = p[:keep] / p[:keep].sum()
    draws = np.array([_sample(z, s, [], 11, pos) for pos in range(20000)])
    assert set(draws) <= set(top[:keep].tolist())
    assert max(abs((draws == t).mean() - q) for t, q in zip(top[:keep], p)) < 0.012
    # A slightly different distribution asked about the same position mostly agrees;
    # independent draws from these two distributions would agree far less often.
    z2 = z + (rng.normal(size=z.size) * 0.15).astype(np.float32)
    same = np.mean([_sample(z, s, [], 11, pos) == _sample(z2, s, [], 11, pos) for pos in range(4000)])
    apart = np.mean([_sample(z, s, [], 11, pos) == _sample(z2, s, [], 12, pos) for pos in range(4000)])
    assert same > 0.8 and same > apart + 0.3, (same, apart)
    assert _sample(z, Sampling(temperature=0.0), [], 0, 0) == int(np.argmax(z))


def test_layout_replay_matches_the_writer():
    """Resuming an interrupted conversion relies on recomputing where the writer put everything."""
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    from convert import Layout, Writer
    rng = np.random.default_rng(0)
    w, plan = Writer(os.path.join(tempfile.mkdtemp(), "t.rlm")), Layout()
    for o in (w, plan):
        o.begin("embed")
        if o is w:
            o.add_native_rows("embed", rng.integers(0, 16, (1000, 64), dtype=np.uint8), np.ones(1000, np.float32),
                              np.ones(64, np.float32), "fp4")
        else:
            o.add_native_rows_layout("embed", 1000, 64, "fp4")
        o.end()
        for i in range(3):
            o.begin(f"layer.{i}")
            o.add_float(f"layers.{i}.in_norm", np.ones(64, np.float32))
            o.add_float(f"layers.{i}.lin.conv", np.ones((96, 4), np.float32))
            parts = [("q", 300), ("k", 77), ("v", 123)]
            if o is w:
                o.add_native(f"layers.{i}.attn.qkv", parts, rng.integers(0, 16, (500, 64), dtype=np.uint8),
                             np.ones(500, np.float32), np.ones(64, np.float32), "fp4")
            else:
                o.add_native_layout(f"layers.{i}.attn.qkv", parts, 64, "fp4")
            o.end()
    assert w.tensors == plan.tensors and w.sections == plan.sections and w.f.tell() == plan.pos
    w.finish({})


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} passed")
