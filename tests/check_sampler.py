"""The sampler draws from the distribution it declares, over the whole vocabulary.

Synthetic logits with a heavy tail (a vocabulary of 4096, rounded to steps of 1/8 so that ties occur, as with
4-bit weights), drawn at 60,000 positions per setting, against the exact distribution computed by sorting the
whole vocabulary: a chi-square test with tokens of equal value merged (which tied token a top-p cut keeps is
implementation-defined) and rare ones pooled. Also: with a top-k, every draw is the same as the sampler before
the fix; without one, a draw differs only where the tail wins. Needs libraceline.dylib (RACELINE_LIB).

    RACELINE_LIB=build/cmake/libraceline.dylib python tests/check_sampler.py
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from raceline.backends import Sampling  # noqa: E402
from raceline.engine import _sample  # noqa: E402

V, N = 4096, 60_000


def logits_with_a_heavy_tail(seed=0):
    rng = np.random.default_rng(seed)
    rank = np.arange(1, V + 1)
    z = -1.05 * np.log(rank) * 1.6 + rng.normal(0, 0.3, V)
    z = np.round(z * 8) / 8                                   # ties, as 4-bit logits have
    return rng.permutation(z).astype(np.float32)              # token ids are not ranks


def exact(logits, s, recent):
    """The declared distribution over the whole vocabulary (float64)."""
    z = logits.astype(np.float64).copy()
    if recent and (s.repetition_penalty != 1.0 or s.presence_penalty):
        for t in np.unique(np.asarray(recent[-64:])):
            if s.repetition_penalty != 1.0:
                z[t] = z[t] / s.repetition_penalty if z[t] > 0 else z[t] * s.repetition_penalty
            z[t] -= s.presence_penalty
    order = np.argsort(-z, kind="stable")
    if s.top_k > 0:
        order = order[:s.top_k]
    lp = (z[order] - z[order[0]]) / s.temperature
    p = np.exp(lp)
    p /= p.sum()
    keep = len(order)
    if 0 < s.top_p < 1:
        keep = min(keep, int(np.searchsorted(np.cumsum(p), s.top_p)) + 1)
    if s.min_p > 0:
        keep = min(keep, int(np.count_nonzero(p >= s.min_p * p[0])))
    out = np.zeros(V)
    out[order[:keep]] = p[:keep] / p[:keep].sum()
    return out, z


def chi2_p(counts, probs, values):
    """Chi-square p-value with tokens of equal (penalized) value merged and expected counts under 5 pooled."""
    groups = {}
    for t in np.flatnonzero((probs > 0) | (counts > 0)):
        g = groups.setdefault(float(values[t]), [0, 0.0])
        g[0] += counts[t]
        g[1] += probs[t] * N
    obs, exp_ = [], []
    pool_o = pool_e = 0.0
    for o, e in groups.values():
        if e < 5:
            pool_o, pool_e = pool_o + o, pool_e + e
        else:
            obs.append(o)
            exp_.append(e)
    if pool_e > 0 or pool_o > 0:
        obs.append(pool_o)
        exp_.append(max(pool_e, 1e-9))
    obs, exp_ = np.array(obs), np.array(exp_)
    if np.any(obs[exp_ < 1e-6] > 0):
        return 0.0, float("inf")                              # draws where none may land
    stat = float(((obs - exp_) ** 2 / exp_).sum())
    dof = len(obs) - 1
    # Wilson-Hilferty: chi-square tail without scipy
    zscore = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(zscore / math.sqrt(2)), stat


def old_sampler(logits, s, recent, seed, pos):
    """The draw without the whole-vocabulary step: the code path before the fix. A tail of (V, V) maps entries
    to the same token ids and skips the step, which is for the whole vocabulary only."""
    return _sample(logits, s, recent, seed, pos, tail=(V, V))


def main():
    logits = logits_with_a_heavy_tail()
    recent = list(np.argsort(-logits)[[0, 3, 7, 1500, 3000]])   # penalized tokens in the head and in the tail
    settings = [
        ("T 1, no top-k, no top-p (the case that was wrong)", Sampling(1.0, 1.0, 0, 0.0), []),
        ("T 1, no top-k, top-p 0.95", Sampling(1.0, 0.95, 0, 0.0), []),
        ("T 1, no top-k, top-p 0.999 (reaches into the tail)", Sampling(1.0, 0.999, 0, 0.0), []),
        ("T 1, no top-k, min-p 1e-6 (tail tokens qualify: the whole vocabulary is ranked)", Sampling(1.0, 1.0, 0, 1e-6), []),
        ("T 1, no top-k, min-p 0.01", Sampling(1.0, 1.0, 0, 0.01), []),
        ("T 0.8, no top-k, penalties on head and tail tokens", Sampling(0.8, 1.0, 0, 0.0, 1.3, 0.5), recent),
        ("T 0.6, top-k 20, top-p 0.95 (the card's)", Sampling(0.6, 0.95, 20, 0.0), []),
    ]
    failed = 0
    for name, s, rec in settings:
        probs, values = exact(logits, s, rec)
        draws = np.array([_sample(logits, s, rec, 7, pos) for pos in range(N)])
        counts = np.bincount(draws, minlength=V)
        pval, stat = chi2_p(counts, probs, values)
        rank = np.empty(V, dtype=np.int64)
        rank[np.argsort(-values, kind="stable")] = np.arange(V)
        beyond_obs = int((rank[draws] >= 1024).sum())
        beyond_exp = float(probs[rank >= 1024].sum() * N)
        line = f"{name}: chi-square p {pval:.3f}; draws past rank 1024: {beyond_obs} (expected {beyond_exp:.1f})"
        if True:   # against the code path before the fix
            before = np.array([old_sampler(logits, s, rec, 7, pos) for pos in range(N)])
            differ = np.flatnonzero(before != draws)
            if s.top_k > 0:
                line += f"; differs from 0.8.0 in {len(differ)} draws"
                failed += len(differ) > 0
            else:
                # where the head won, the draw is the same token; only top-p/min-p cuts made against the whole
                # mass, or a tail token, may differ
                head_changes = int((rank[draws[differ]] < 1024).sum())
                line += f"; differs from 0.8.0 in {len(differ)} draws ({head_changes} of them inside the head)"
        ok = pval > 0.001
        failed += not ok
        print(("ok   " if ok else "FAIL ") + line, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
