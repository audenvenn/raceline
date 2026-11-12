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


class KVFakeBackend(FakeBackend):
    """FakeBackend whose context doubles as its attention entries: they can be
    copied out and back in blocks, and every token it writes depends on them as
    well as on the recurrent state, so a conversation brought back wrongly
    writes other tokens than one recomputed from scratch."""

    kv_block = 4

    def generate(self, sampling, max_tokens, history):
        for _ in range(max_tokens):
            t = (self.state + sum((i + 1) * c for i, c in enumerate(self.ctx))) % 97
            self._feed(t)
            yield t

    def save_kv(self, start, end):
        lo = start // self.kv_block * self.kv_block
        return np.array(self.ctx[lo:end], dtype=np.int16)

    def load_kv(self, start, end, data):
        lo = start // self.kv_block * self.kv_block
        assert len(data) == end - lo and len(self.ctx) >= lo
        self.ctx[lo:] = [int(x) for x in data]

    def restore(self, snap):
        assert snap.position <= len(self.ctx)    # its entries must be in place
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


def test_system_messages_fit_the_template():
    msgs = normalize_messages([{"role": "system", "content": "A."}, {"role": "developer", "content": "B."},
                               {"role": "user", "content": "hi"}, {"role": "system", "content": "Reminder."}])
    assert [(m["role"], m["content"]) for m in msgs] == [("system", "A.\n\nB."), ("user", "hi"),
                                                         ("user", "Reminder.")]


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
    # A draft's restricted vocabulary (ids 0..99, then 1000 on) draws exactly what the
    # whole vocabulary would where the ids left out have no mass, penalties included,
    # also for recent tokens it does not cover.
    full = (rng.normal(size=1256) * 2).astype(np.float32)
    full[100:1000] = -1e30
    part = np.concatenate([full[:100], full[1000:]])
    for s3 in (Sampling(temperature=0.9, top_p=0.95, top_k=20, presence_penalty=1.5), Sampling(temperature=0.0)):
        for pos in range(300):
            recent = [7, 500, 1100, int(np.argmax(full))]
            want = _sample(full, s3, recent, 5, pos, with_probability=True)
            got = _sample(part, s3, recent, 5, pos, with_probability=True, tail=(100, 1000))
            assert got[0] == want[0] and abs(got[1] - want[1]) < 1e-9, (pos, got, want)
    assert _sample(z[:2000], Sampling(presence_penalty=1.0), [248045, 3], 0, 0) < 2000   # ids past the end: no crash


def test_lookup_proposes_what_followed_the_last_occurrence():
    from raceline.engine import _Lookup

    #        0  1  2  3  4  5  6  7  8  9 10 11
    ctx = [5, 1, 2, 3, 4, 9, 1, 2, 3, 7, 8, 1]
    look = _Lookup(ctx)
    # (1, 2) then 3 last occurred at 6..8, followed by 7, 8, 1.
    look.push(2)
    assert look.propose(3, 8) == [7, 8, 1, 2]
    assert look.propose(3, 2) == [7, 8]
    assert look.propose(6, 8) == []                  # (1, 2, 6) never occurred
    # Tokens pushed during generation are found too, and the newest occurrence wins.
    for t in (3, 6, 6, 1, 2):
        look.push(t)
    assert look.propose(3, 3) == [6, 6, 1]
    assert _Lookup([1]).propose(2, 4) == [] and look.propose(3, 0) == []


def test_pass_cost_measures_what_a_row_adds():
    from raceline.engine import _PassCost, _copy_length

    rng = np.random.default_rng(0)
    moe, dense = _PassCost(), _PassCost()
    for i in range(200):
        rows = 1 + i % 8
        moe.add(rows, 0.020 + 0.004 * (rows - 1) + rng.normal(0, 2e-4))
        dense.add(rows, 0.020 + rng.normal(0, 2e-4))
    assert abs(moe.per_row() - 0.004) < 3e-4, moe.per_row()
    assert dense.per_row() < 2e-4, dense.per_row()
    same = _PassCost()                     # every pass the same length: no slope to measure
    for _ in range(50):
        same.add(4, 0.03)
    assert same.per_row() == 0.0 and _PassCost().per_row() == 0.0
    # A copy is cut where its rows stop paying: held ** n against the cost of a row.
    assert _copy_length(0.9, 0.0, 15) == 15            # free rows: the whole copy
    assert _copy_length(0.7, 0.2, 15) == 4             # 0.7 ** 4 = 0.24 > 0.2 > 0.7 ** 5
    assert _copy_length(0.5, 0.9, 15) == 1             # the first, which the head agreed with
    assert _copy_length(0.99, 0.01, 6) == 6


def test_command_timeout_kills_what_it_started():
    """A shell command that times out must not leave its children running (a
    test stuck in a loop kept burning a CPU core for hours before this)."""
    import tempfile
    import time

    from raceline.agent import run_command

    d = tempfile.mkdtemp()
    pid_file = os.path.join(d, "pid")
    child = f"import os, time; open({pid_file!r}, 'w').write(str(os.getpid())); time.sleep(600)"
    t0 = time.time()
    assert run_command(f"{sys.executable} -c \"{child}\"; true", d, 1.5) is None   # the shell stays the parent
    assert time.time() - t0 < 10
    pid = int(open(pid_file).read())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError(f"process {pid} outlived the timeout")
    r = run_command("echo hi; exit 3", d, 10)
    assert r.returncode == 3 and r.stdout == "hi\n"


def test_unsupported_macs_are_told_why():
    import platform

    import raceline.engine as eng
    import raceline.sysinfo as si

    saved = (si.sysctl_str, platform.mac_ver, platform.system, platform.machine, os.environ.pop("RACELINE_ANY_GPU", None))
    try:
        platform.system, platform.machine = (lambda: "Darwin"), (lambda: "arm64")
        platform.mac_ver = lambda: ("27.1", ("", "", ""), "arm64")
        si.sysctl_str = lambda name: "Apple M3 Max"
        assert "M5 or later" in eng.unsupported_reason() and "M3 Max" in eng.unsupported_reason()
        si.sysctl_str = lambda name: "Apple M6 Pro"
        assert eng.unsupported_reason() is None
        platform.mac_ver = lambda: ("26.4", ("", "", ""), "arm64")
        assert "macOS 27" in eng.unsupported_reason()
        os.environ["RACELINE_ANY_GPU"] = "1"
        assert eng.unsupported_reason() is None
    finally:
        si.sysctl_str, platform.mac_ver, platform.system, platform.machine = saved[:4]
        os.environ.pop("RACELINE_ANY_GPU", None)
        if saved[4] is not None:
            os.environ["RACELINE_ANY_GPU"] = saved[4]


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


def test_pull_lists_models_with_how_they_fit():
    import contextlib
    import io
    import raceline.pull as pull

    assert set(pull.CATALOG) == set(pull.MODELS)
    saved = pull._memory_bytes
    pull._memory_bytes = lambda: 16 << 30
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pull.list_models()
    finally:
        pull._memory_bytes = saved
    lines = {line.split()[0]: line for line in out.getvalue().splitlines() if line.startswith("  ")}
    assert "too large" in lines["qwen3.8-27b"] and "[" not in lines["qwen3.5-9b"]
    assert "experts partly on the SSD" in lines["qwen3.5-35b-a3b"]   # 2 GB must be resident, 18 GB in all


def test_pull_assembles_parts_and_resumes():
    """A model cut into release parts comes back identical, also when the download was interrupted."""
    import json
    import tempfile
    import raceline.pull as pull
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    from convert import Writer
    from release_model import cut
    tmp = tempfile.mkdtemp()
    model = os.path.join(tmp, "tiny.rlm")
    w = Writer(model)
    w.begin("data")
    w.add_float("x", np.random.default_rng(1).normal(size=700_000).astype(np.float32))
    w.end()
    w.finish(dict(format="raceline", source="test"))
    manifest = cut(model, os.path.join(tmp, "release"), part_size=1_500_000)
    assert len(manifest["parts"]) == 3 and sum(p["size"] for p in manifest["parts"]) == os.path.getsize(model)

    saved, pull.RELEASES = pull.RELEASES, "file://" + os.path.join(tmp, "release")
    try:
        out = pull.pull("tiny.rlm", os.path.join(tmp, "got.rlm"))
        assert open(out, "rb").read() == open(model, "rb").read()
        # An interrupted download: the first part is in place and recorded, the rest is missing.
        dest = os.path.join(tmp, "again.rlm")
        with open(dest + ".partial", "wb") as f:
            f.write(open(model, "rb").read()[:manifest["parts"][0]["size"]])
        with open(dest + ".partial.json", "w") as f:
            json.dump({"done": [0]}, f)
        os.remove(os.path.join(tmp, "release", manifest["parts"][0]["name"]))    # it must not be fetched again
        assert open(pull.pull("tiny.rlm", dest), "rb").read() == open(model, "rb").read()
        assert not os.path.exists(dest + ".partial") and not os.path.exists(dest + ".partial.json")
    finally:
        pull.RELEASES = saved


def _run(session, prompt, n=5, pin=None):
    usage = session.sync(prompt, () if pin is None else (pin,), pin)   # as Runtime.render marks the system prompt
    return usage, list(session.generate(Sampling(), n))


def _fresh(prompt, n=5):
    return _run(Session(KVFakeBackend()), prompt, n)[1]


def test_a_conversation_set_aside_comes_back_without_recomputing():
    system = list(range(1, 11))                       # shared by both conversations; pinned
    a = system + [50 + i % 7 for i in range(30)]       # a long conversation
    b = system + [20, 21, 22]                          # a side request
    s = Session(KVFakeBackend(), min_park_tokens=12)   # the side request, 8 tokens past the system prompt, is not kept
    _, out_a = _run(s, a, pin=len(system))
    usage, out_b = _run(s, b, pin=len(system))
    assert s.parked == 1 and usage["cached_tokens"] == len(system)
    assert out_b == _fresh(b)
    a2 = a + out_a + [60, 61, 62]                      # the long conversation continues
    fed = s.backend.fed
    usage, out_a2 = _run(s, a2, pin=len(system))
    assert usage["cached_tokens"] == len(a) + len(out_a)   # nothing of it recomputed
    assert s.backend.fed - fed == 3 + len(out_a2)          # only the new tokens, and what it wrote
    assert out_a2 == _fresh(a2)                             # and it writes what a fresh run writes
    usage, out_b2 = _run(s, b + out_b + [23], pin=len(system))   # the short one was not worth keeping
    assert usage["cached_tokens"] == len(system) and out_b2 == _fresh(b + out_b + [23])


def test_a_parked_chat_comes_back_from_where_its_last_reply_began():
    """As a chat client does it: the last reply comes back rendered differently from how it was
    written (without its reasoning), so the conversation resumes where that reply began."""
    system = list(range(1, 11))
    history = system + [30 + i % 9 for i in range(40)]
    gen_start, prompt_end = len(history), len(history) + 3          # a generation prompt of 3 tokens
    first = history + [90, 91, 92]
    s = Session(KVFakeBackend(), min_park_tokens=8)
    s.sync(first, (len(system), gen_start), len(system))
    reply = list(s.generate(Sampling(), 6))
    s.sync([77] * 20, (), None)                                     # a side request with its own system prompt
    s.generate(Sampling(), 3)
    assert s.parked == 1
    again = history + [90, 95] + [5, 6, 7, 8]                       # the reply rendered otherwise, then more
    usage = s.sync(again, (len(system), len(again) - 2), len(system))
    assert usage["cached_tokens"] == gen_start, usage               # not 0: from where the reply began
    out = list(s.generate(Sampling(), 5))
    ref = Session(KVFakeBackend())
    ref.sync(again)
    assert out == list(ref.generate(Sampling(), 5))
    del reply, prompt_end


def test_a_parked_conversation_needs_its_shared_start_in_the_context():
    a = list(range(1, 41))
    s = Session(KVFakeBackend(), min_park_tokens=8)
    _run(s, a)
    _run(s, [99] * 30)                                  # parts from `a` at position 0: `a` is parked from 0
    _run(s, [98] * 30)                                  # `[99]*30` parked too; `a` still needs nothing shared
    usage, out = _run(s, a + [7])
    assert usage["cached_tokens"] == len(a) + 0 and out == _fresh(a + [7])
    s2 = Session(KVFakeBackend(), min_park_tokens=8)
    _run(s2, a)
    _run(s2, a[:20] + [99] * 20)                        # `a` parked from block 20
    _run(s2, [98] * 40)                                 # the context no longer holds a[:20]
    fed = s2.backend.fed
    usage, out = _run(s2, a + [7])                      # a[:20] is computed again, the rest comes back
    assert usage["cached_tokens"] == len(a) - 20 and out == _fresh(a + [7])
    assert s2.backend.fed - fed == 20 + 1 + len(out)


def test_parked_conversations_stay_within_their_budget():
    s = Session(KVFakeBackend(), min_park_tokens=8, max_parked_bytes=200)
    for k in range(6):
        _run(s, [k + 1] * 40)
    assert 1 <= s.parked and sum(p.nbytes for p in s._parked) <= 200
    usage, out = _run(s, [6] * 40 + [5])                # the latest is current, so it simply extends
    assert usage["cached_tokens"] == 40 and out == _fresh([6] * 40 + [5])


def test_transformers_notice_is_silenced_for_the_import_only():
    from raceline.backends import without_torch_notice

    key = "TRANSFORMERS_NO_ADVISORY_WARNINGS"
    saved = os.environ.get(key)
    try:
        for before in (None, "0"):
            if before is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = before
            with without_torch_notice():
                assert os.environ[key] == "1"
            assert os.environ.get(key) == before  # the agent's commands must not inherit it
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} passed")
