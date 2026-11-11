# Security policy

## Reporting a vulnerability

Please report security problems privately through
[GitHub's private vulnerability reporting](https://github.com/audenvenn/raceline/security/advisories/new),
not in a public issue. You can expect an acknowledgement within a week.
Fixes are released for the latest version only.

## What is in scope

- The terminal agent running a shell command or writing a file without the
  approval it is documented to ask for.
- The server accepting connections or performing actions beyond what is
  documented.
- Memory safety problems when reading a `.rlm` model file or a release
  manifest.
- `raceline pull` accepting a file that does not match its published
  checksums.

## What users should know

- `raceline serve` has no authentication. It listens on `127.0.0.1` by
  default; do not bind it to a public interface.
- `raceline chat --yes`, or answering `a` at an approval prompt, lets the
  model run shell commands and change files without asking. Use it only in
  directories and on tasks where that is acceptable.
- A model's output is not trusted input. The agent shows every tool call
  before or as it runs; read what you approve.
- Model files are data, but they are mapped and parsed by native code: use
  files from sources you trust.
