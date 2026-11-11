# Contributing to raceline

Thanks for looking. This is a small project with a narrow goal: get as much
out of one Apple Silicon Mac, for one user, as the hardware allows, and say
exactly how much that is. Contributions that serve that goal are welcome.

## Ground rules

- **English only.** Code, comments, documentation, commit messages, issues and
  pull requests are in English. CI checks the repository for this.
- **Measured claims.** A statement about speed, memory or quality comes with
  the machine, the commands and the numbers. Speed is measured back to back
  against the baseline, with nothing else using the GPU; a number taken while
  another job runs is not a measurement.
- **Exactness is not negotiable.** Speculative decoding must produce what
  one-token-at-a-time decoding produces, bit for bit. Changes to the engine
  keep `tests/verify_spec.py` and `raceline-engine spec` passing.
- **One change per pull request**, and one change per measurement.

## Setting up

```sh
cmake -S . -B build/cmake -G Ninja && cmake --build build/cmake
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/raceline pull qwen3.5-4b        # a model to test with (2.5 GB)
```

The Metal kernels in `kernels/` are compiled at run time. After touching
`kernels/` or `src/`, rebuild **every** target (`cmake --build build/cmake`):
a stale binary and new kernel sources disagree silently and give wrong results.

## Tests

| What | Needs | Command |
|---|---|---|
| Runtime logic: sessions, snapshots, tool-call parsing, sampler, conversion resume | nothing | `python tests/test_runtime.py` |
| Linear kernels against a CPU reference | GPU | `build/cmake/test_linear` |
| Speculative decoding is bit-exact | GPU, a model with an MTP head | `python tests/verify_spec.py <model.rlm> <tokens.i32>` |
| Plain against 1 to 7 drafts per pass | GPU, model | `build/cmake/raceline-engine spec <model.rlm> <tokens.i32> --drafts 1 2 3 4 5 6 7` |
| Engine against MLX on the same weights | GPU, model, the original checkpoint | `python tests/verify_native.py`, `python tests/verify_mtp.py` |
| Speed | idle GPU, model, an MLX conversion as the baseline | `bench/agent_bench.py`, `bench/chat_bench.py` |

Token files are raw little-endian int32. `tools/make_tokens.py <model.rlm>
README.md tokens.i32 --count 96` makes one from any text; 96 tokens are enough
for the verification scripts, the benchmarks of the engine's own command line
want a few thousand.

CI runs the first row, byte-compiles the Python sources and checks the
English-only rule. It cannot run the GPU tests, so say in the pull request
which of them you ran and on what machine.

## Workflow

1. For anything larger than a fix, open an issue first and describe what you
   want to change and why.
2. Branch from `main`. `main` is protected: changes arrive through pull
   requests, CI has to pass, and pull requests are squash-merged.
3. Write commit and pull-request titles in the imperative ("Add ...", "Fix
   ..."), 72 characters or fewer. In the body, say what changed and what was
   measured or verified.
4. Add an entry under "Unreleased" in `CHANGELOG.md` if users will notice.

## Releases

- Versions follow [Semantic Versioning](https://semver.org). The version lives
  in `pyproject.toml` and `raceline/__init__.py`.
- A release is a tag `vX.Y.Z` on `main` with the matching `CHANGELOG.md`
  section as its notes.
- Converted models live in the `models-v1` release, separate from code
  versions. `tools/release_model.py` cuts a `.rlm` file into parts of under
  2 GB and writes the manifest that `raceline pull` reads.

## Reporting numbers from your machine

The project has been developed on a single Mac. If you run it on another
one, please open a "Benchmark from another Mac" issue with the output of
`raceline probe <model>`; that is a contribution in its own right.
