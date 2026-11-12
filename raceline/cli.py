"""Command line: raceline chat | serve | probe | pull."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .backends import Sampling, load_backend


def _defaults(model: str, backend) -> Sampling:
    """Sampling defaults: the model's generation_config.json where it has one."""
    s = Sampling()
    for d in (getattr(backend, "files_dir", None), model, os.path.dirname(model)):
        path = os.path.join(d, "generation_config.json") if d else ""
        if os.path.isfile(path):
            cfg = json.load(open(path))
            for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
                if cfg.get(key) is not None:
                    setattr(s, key, cfg[key])
            break
    return s


def _runtime(args):
    from .chat import Runtime

    from .pull import resolve

    args.model = resolve(args.model)
    backend = load_backend(args.model, args.backend)
    sampling = _defaults(args.model, backend)
    if args.temperature is not None:
        sampling.temperature = args.temperature
    return Runtime(backend, sampling)


def main(argv=None):
    try:
        return _main(argv)
    except RuntimeError as e:      # a problem with the setup, not a bug: no traceback
        if os.environ.get("RACELINE_DEBUG"):
            raise
        print(f"raceline: {e}", file=sys.stderr)
        sys.exit(1)


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="raceline", description="A local agent runtime for Apple Silicon.")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("model", nargs="?",
                       help="a .rlm file, the name of a pulled model (qwen3.5-4b), or an MLX model directory "
                            "(default: $RACELINE_MODEL, or the pulled model if there is only one)")
        p.add_argument("--backend", choices=["mlx", "raceline"], help="default: by model type")
        p.add_argument("--temperature", type=float)

    p = sub.add_parser("serve", help="OpenAI-compatible API on localhost")
    common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)

    p = sub.add_parser("chat", help="terminal agent with file and shell tools")
    common(p)
    p.add_argument("-p", "--prompt", help="run one request and exit")
    p.add_argument("--cwd", default=".", help="working directory for the tools")
    p.add_argument("--yes", action="store_true", help="approve shell commands and file writes without asking")
    p.add_argument("--no-think", action="store_true", help="skip the model's reasoning phase")
    p.add_argument("--effort", choices=["low", "medium", "xhigh"], default="low",
                   help="reasoning effort, for models whose template takes one (default: low, for short steps)")
    p.add_argument("--max-tokens", type=int, default=4096)

    p = sub.add_parser("probe", help="measure this Mac's limits for local inference (and a model against them)")
    p.add_argument("model", nargs="?", help="optional: a model to measure against the limits")
    p.add_argument("--backend", choices=["mlx", "raceline"])

    p = sub.add_parser("pull", help="download a converted model: " + ", ".join(sorted(__import__("raceline.pull", fromlist=["MODELS"]).MODELS)))
    p.add_argument("model")
    p.add_argument("--dest", help="file or directory to save to (default: the raceline cache)")
    p.add_argument("--source", choices=["github", "huggingface"],
                   help="where to download from (default: github, or $RACELINE_SOURCE)")

    args = ap.parse_args(argv)
    if args.command == "pull":
        from .pull import pull
        pull(args.model, args.dest, args.source)
        return
    if args.command == "probe":
        from .probe import run
        from .pull import resolve
        return run(resolve(args.model) if args.model else None, args.backend)
    if not args.model:
        from .pull import default_model
        args.model = default_model()
    if not args.model:
        ap.error("no model given: name one, set RACELINE_MODEL, or fetch one with `raceline pull qwen3.5-4b`")
    rt = _runtime(args)

    if args.command == "serve":
        from .server import serve
        serve(rt, getattr(rt.backend, "model_id", os.path.basename(args.model)), args.host, args.port)
    else:
        from .agent import Agent, repl
        agent = Agent(rt, args.cwd, auto_approve=args.yes, thinking=not args.no_think, max_tokens=args.max_tokens,
                      interactive=args.prompt is None, color=sys.stdout.isatty(), effort=args.effort)
        if args.prompt is None:
            repl(agent)
        else:
            totals = agent.turn(args.prompt)
            print(agent.status(totals), file=sys.stderr)


if __name__ == "__main__":
    main()
