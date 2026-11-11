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
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional, Sequence

from .backends import Backend, Sampling, Snapshot


class Session:
    def __init__(self, backend: Backend, max_snapshots: int = 6, max_snapshot_bytes: int = 1 << 30):
        self.backend = backend
        self.tokens: list = []          # exactly the tokens in the backend's context
        self._snaps: list = []          # ascending position
        self._pinned: Optional[int] = None
        self.max_snapshots = max_snapshots
        self.max_snapshot_bytes = max_snapshot_bytes

    def reset(self) -> None:
        self.backend.reset()
        self.tokens, self._snaps, self._pinned = [], [], None

    def sync(self, prompt: Sequence[int], branch_points: Sequence[int] = (), pin: Optional[int] = None,
             progress: Optional[Callable[[int, int], None]] = None) -> dict:
        """Brings the model to exactly `prompt` and returns what it cost.

        `branch_points` are positions inside the prompt worth a snapshot; `pin`
        is the one to keep for as long as it stays valid (the end of the system
        prompt: every conversation with these tools starts from it).
        """
        b = self.backend
        common = 0
        for a, t in zip(self.tokens, prompt):
            if a != t:
                break
            common += 1

        if common == len(self.tokens):
            start = common
        else:
            snap = max((s for s in self._snaps if s.position <= common), key=lambda s: s.position, default=None)
            if snap is not None:
                b.restore(snap)
                start = snap.position
            else:
                b.reset()
                start = 0
            self._snaps = [s for s in self._snaps if s.position <= start]
            if self._pinned is not None and self._pinned > start:
                self._pinned = None
            self.tokens = self.tokens[:start]

        if pin is not None and start <= pin < len(prompt):
            self._pinned = pin
        cuts = sorted({p for p in branch_points if start < p < len(prompt)} | {len(prompt)})
        position = start
        for cut in cuts:
            if cut > position:
                b.prefill(list(prompt[position:cut]), progress)
                position = cut
            self._remember(b.snapshot())
        self.tokens = list(prompt)
        return dict(prompt_tokens=len(prompt), cached_tokens=start)

    def generate(self, sampling: Sampling, max_tokens: int, stop_ids=()) -> Iterator[int]:
        """Yields sampled tokens, ending after one in `stop_ids` (which is yielded too)."""
        for token in self.backend.generate(sampling, max_tokens, self.tokens):
            self.tokens.append(token)
            yield token
            if token in stop_ids:
                break

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
