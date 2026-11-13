"""One model context, reused from request to request.

Agent clients resend the whole conversation on every turn. Re-reading it costs
seconds to minutes of prompt processing on a large local model, so the session
keeps the model where the last request left it and only feeds what is new.

When the new prompt is not an extension of the old one (the client rewrote
history, or an assistant turn renders differently from what was generated),
attention layers could simply be truncated, but the recurrent layers of
Qwen3.5 / Qwen3.8 cannot be rewound. The session therefore snapshots the
recurrent state at the few places a later prompt is likely to branch from, and
resumes from the nearest one.

A client may also go back and forth between conversations: a second chat, or
a side request (a title, a commit message) in the middle of a long session.
With one context each switch would cost the whole other conversation again.
Where the backend can copy out attention entries (`Backend.kv_block`), the
session sets the conversation it is leaving aside instead: its recurrent
state, and its attention entries from where it parts from the new prompt.
When a later prompt continues it, those are put back and nothing is
recomputed. Parked conversations are kept within `max_parked_bytes`, the
least recently used dropped first.

A prompt is a list of keys: token ids, and for the rows of an image negative
numbers that name the image and the row (raceline.images.image_keys). Two
images of the same size are then never mistaken for each other. The rows'
vectors come from the `images` callback, only for rows that are computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Sequence

from .backends import Backend, ImageRows, Sampling, Snapshot
from .images import key_row


def _common(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class _Parked:
    """A conversation set aside. Its attention entries are saved from kv_from
    on (a block boundary); the positions before it are the ones it shared with
    the context that replaced it, and have to hold the same tokens again when
    it comes back (they are computed again if another conversation overwrote
    them)."""
    tokens: list
    snaps: list         # its snapshots past kv_from, ascending; the last one at its end
    kv_from: int
    kv: Any
    nbytes: int
    branches: set       # which of those positions are branch points


class Session:
    def __init__(self, backend: Backend, max_snapshots: int = 6, max_snapshot_bytes: int = 512 << 20,
                 max_parked_bytes: int = 2 << 30, min_park_tokens: int = 512):
        self.backend = backend
        self.tokens: list = []          # exactly the tokens in the backend's context
        self._snaps: list = []          # ascending position
        self._branches: set = set()     # positions of snapshots taken at a caller's branch point
        self._pinned: Optional[int] = None
        self._parked: list = []         # least recently used first
        self._images = None             # the image rows of the prompt being synced (sync)
        self.max_snapshots = max_snapshots
        self.max_snapshot_bytes = max_snapshot_bytes
        self.max_parked_bytes = max_parked_bytes
        self.min_park_tokens = min_park_tokens   # less than this is cheaper to recompute than to keep

    def reset(self) -> None:
        self.backend.reset()
        self.tokens, self._snaps, self._pinned, self._parked = [], [], None, []
        self._branches = set()

    def sync(self, prompt: Sequence[int], branch_points: Sequence[int] = (), pin: Optional[int] = None,
             progress: Optional[Callable[[int, int], None]] = None,
             images: Optional[Callable[[int], tuple]] = None) -> dict:
        """Brings the model to exactly `prompt` and returns what it cost.

        `branch_points` are positions inside the prompt worth a snapshot; `pin`
        is the one to keep for as long as it stays valid (the end of the system
        prompt: every conversation with these tools starts from it). `images`
        maps an image's digest to (its rows' vectors, its width).
        """
        b = self.backend
        self._images = images
        common = _common(self.tokens, prompt)
        if common == len(self.tokens):
            here, snap = common, None
        else:
            snap = max((s for s in self._snaps if s.position <= common), key=lambda s: s.position, default=None)
            here = snap.position if snap is not None else 0

        # A parked conversation that takes the prompt further than `here`, net of the tokens of its
        # shared start that have to be computed again first (if another conversation overwrote them).
        back, back_at, best = None, 0, here
        if b.kv_block:
            for p in self._parked:
                at = self._resume_point(p, prompt)
                net = at - max(0, p.kv_from - _common(self.tokens, p.tokens))
                if at and net > best:
                    back, back_at, best = p, at, net

        redone = 0     # tokens before `start` that had to be computed again all the same
        if back is None and common == len(self.tokens):
            start = common
        else:
            # Everything of the current conversation past this point is about to be overwritten.
            lost = common if back is None else min(common, _common(self.tokens, back.tokens))
            if b.kv_block and self.max_parked_bytes and len(self.tokens) - lost >= self.min_park_tokens:
                self._park(lost)
            if back is not None:
                shared = _common(self.tokens, back.tokens)    # snapshots up to here hold for both
                self._parked.remove(back)
                if shared < back.kv_from:     # its start was overwritten: compute that part again
                    first = max((s for s in self._snaps if s.position <= shared), key=lambda s: s.position,
                                default=None)
                    if first is not None:
                        b.restore(first)
                    else:
                        b.reset()
                    done = first.position if first is not None else 0
                    self._feed(back.tokens[done:back.kv_from], progress)
                    redone = back.kv_from - done
                    self._snaps = [s for s in self._snaps if s.position <= done]
                    self.tokens, shared = list(back.tokens[:back.kv_from]), back.kv_from
                b.load_kv(back.kv_from, len(back.tokens), back.kv)
                resume = max((s for s in back.snaps if s.position <= back_at), key=lambda s: s.position)
                b.restore(resume)
                start = resume.position
                mine = [s for s in self._snaps if s.position <= min(shared, start)]
                self._snaps = mine + [s for s in back.snaps if s.position <= start and
                                      all(m.position != s.position for m in mine)]
                self._snaps.sort(key=lambda s: s.position)
                self._branches = ({p for p in self._branches if p <= min(shared, start)} |
                                  {p for p in back.branches if p <= start})
                self.tokens = back.tokens[:start]
            elif snap is not None:
                b.restore(snap)
                start = snap.position
                self._snaps = [s for s in self._snaps if s.position <= start]
                self.tokens = self.tokens[:start]
            else:
                b.reset()
                start = 0
                self._snaps, self.tokens = [], []
            if self._pinned is not None and self._pinned > start:
                self._pinned = None

        if pin is not None and start <= pin < len(prompt):
            self._pinned = pin
        branches = {p for p in branch_points if start < p < len(prompt)}
        if getattr(b, "marks_in_passes", False):
            # Snapshots at the branch points from inside the passes: no pass has to end there.
            for snap in self._feed(prompt[start:], progress, marks=[p - start for p in sorted(branches)]):
                self._remember(snap)
            self._remember(b.snapshot())
        else:
            cuts = sorted(branches | {len(prompt)})
            position = start
            for cut in cuts:
                if cut > position:
                    self._feed(prompt[position:cut], progress)
                    position = cut
                self._remember(b.snapshot())
        self._branches = {p for p in self._branches if p <= start} | branches
        self.tokens = list(prompt)
        return dict(prompt_tokens=len(prompt), cached_tokens=start - redone)

    def _feed(self, keys: Sequence[int], progress=None, marks=()):
        """Feeds keys to the backend: tokens, and the image rows among them as vectors; `marks`
        (offsets into keys) go to a backend that snapshots inside its passes, -> its snapshots."""
        b = self.backend
        tokens, rows, i = list(keys), [], 0
        while i < len(keys):
            if keys[i] >= 0:
                i += 1
                continue
            digest, row = key_row(keys[i])
            j = i + 1
            while j < len(keys) and keys[j] == keys[j - 1] - 1 and key_row(keys[j])[0] == digest:
                j += 1
            if b.image_token is None or self._images is None:
                raise ValueError("this model takes no images")
            vectors, width = self._images(digest)
            rows.append(ImageRows(i, vectors[row:row + j - i], width, row))
            tokens[i:j] = [b.image_token] * (j - i)
            i = j
        extra = {"images": rows} if rows else {}
        if marks:
            extra["marks"] = marks
        return b.prefill(tokens, progress, **extra) or []

    def generate(self, sampling: Sampling, max_tokens: int, stop_ids=()) -> Iterator[int]:
        """Yields sampled tokens, ending after one in `stop_ids` (which is yielded too)."""
        history = self.tokens
        if history and min(history) < 0:        # an image's rows count as its token there
            history = [t if t >= 0 else self.backend.image_token for t in history]
        for token in self.backend.generate(sampling, max_tokens, history):
            self.tokens.append(token)
            yield token
            if token in stop_ids:
                break

    @property
    def parked(self) -> int:
        """Conversations set aside, for status lines and tests."""
        return len(self._parked)

    @staticmethod
    def _resume_point(p: _Parked, prompt: Sequence[int]) -> int:
        """How far into `prompt` the parked conversation `p` would take the model (0: not at all)."""
        n = _common(p.tokens, prompt)
        return max((s.position for s in p.snaps if s.position <= n), default=0)

    def _park(self, lost: int) -> None:
        """Sets the current conversation aside before its positions from `lost` on are overwritten."""
        b = self.backend
        kv_from = lost // b.kv_block * b.kv_block
        end = b.snapshot()          # the model is at the end of the conversation
        # Its end, to continue it; the last branch point (the start of the last reply, which a chat client
        # sends back rendered without its reasoning); and the end of the last prompt, for a client that
        # asks again. A recurrent state is 50 to 150 MB, so no more.
        before = [s for s in self._snaps if kv_from < s.position < end.position]
        keep = {id(s) for s in [s for s in before if s.position in self._branches][-1:] + before[-1:]}
        snaps = [s for s in before if id(s) in keep] + [end]
        kv = b.save_kv(kv_from, end.position)
        nbytes = kv.nbytes + sum(s.nbytes for s in snaps)
        if nbytes > self.max_parked_bytes:
            return
        self._parked.append(_Parked(list(self.tokens), snaps, kv_from, kv, nbytes,
                                    {s.position for s in snaps} & self._branches))
        while sum(p.nbytes for p in self._parked) > self.max_parked_bytes:
            self._parked.pop(0)

    def _remember(self, snap: Snapshot) -> None:
        self._snaps = [s for s in self._snaps if s.position != snap.position] + [snap]
        self._snaps.sort(key=lambda s: s.position)
        while len(self._snaps) > 1 and (len(self._snaps) > self.max_snapshots or
                                         sum(s.nbytes for s in self._snaps) > self.max_snapshot_bytes):
            # Drop the oldest position that is not the pinned system-prompt snapshot.
            victim = next((s for s in self._snaps if s.position != self._pinned), None)
            if victim is None or victim is self._snaps[-1]:
                break
            self._snaps.remove(victim)
