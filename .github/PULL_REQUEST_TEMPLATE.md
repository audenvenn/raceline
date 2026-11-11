## What this changes

<!-- One or two sentences. Link the issue it addresses, if there is one. -->

## How it was checked

<!-- Tick what you ran and paste the relevant output. Speed claims need the machine, the commands, and a back-to-back baseline on an idle GPU. -->

- [ ] `python tests/test_runtime.py`
- [ ] `build/cmake/test_linear` (after touching `kernels/` or `src/`)
- [ ] `tests/verify_native.py`, `tests/verify_mtp.py`, `tests/verify_spec.py` on a model (after touching the engine)
- [ ] `build/cmake/raceline-engine spec <model> <tokens>` reports identical tokens
- [ ] Benchmarks, if the change is about speed

## Checklist

- [ ] Code, comments and documentation are in English
- [ ] `CHANGELOG.md` has an entry under "Unreleased" if users will notice the change
