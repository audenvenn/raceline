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


def agent_sampling(model: str, backend, thinking: bool) -> Sampling:
    """Sampling for the agent: what Qwen recommends for coding, in thinking mode
    temperature 0.6, top-p 0.95, top-k 20 (a model's generation_config.json
    holds its general-purpose setting instead), without thinking 0.7, 0.8, 20.
    On the tasks of bench/agent_eval.py this solves as many and writes fewer
    tokens: fewer reasoning loops."""
    s = _defaults(model, backend)
    s.temperature, s.top_p, s.top_k = (0.6, 0.95, 20) if thinking else (0.7, 0.8, 20)
    return s


def _runtime(args, agent: bool = False):
    from .chat import Runtime

    from .pull import resolve

    args.model = resolve(args.model)
    if getattr(args, "expert_cache", None) is not None:
        os.environ["RACELINE_EXPERT_CACHE"] = str(args.expert_cache)
    backend = load_backend(args.model, args.backend, getattr(args, "ctx", None))
    cache = getattr(backend, "expert_cache_size", lambda: None)()
    if cache:
        print(f"raceline: {100 * cache[1]:.0f}% of the model's experts are kept in memory ({cache[0]:.1f} GB); the others "
              "are read from the SSD when a layer needs them (--expert-cache GB changes that)", file=sys.stderr)
    sampling = agent_sampling(args.model, backend, not args.no_think) if agent else _defaults(args.model, backend)
    if args.temperature is not None:
        sampling.temperature = args.temperature
    # The terminal agent holds one conversation; keeping others aside is for the server's clients.
    return Runtime(backend, sampling, park_bytes=0 if agent else None)


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
        p.add_argument("--expert-cache", type=float, metavar="GB",
                       help="mixture-of-experts models: keep the experts on disk and the ones in use in a cache "
                            "of this size (0: all in memory; default: decided from this Mac's memory; "
                            "RACELINE_EXPERT_CACHE)")
        p.add_argument("--ctx", type=int, metavar="TOKENS",
                       help="context window of the engine (default 32768; the models take up to 262144, at "
                            "22 KB of attention cache per token for the 35B-A3B models and 68 KB for Qwen3.8-27B)")

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
    p.add_argument("--image", action="append", default=[], metavar="FILE",
                   help="an image that goes with the first message (repeatable; models with a vision file)")

    p = sub.add_parser("probe", help="measure this Mac's limits for local inference (and a model against them)")
    p.add_argument("model", nargs="?", help="optional: a model to measure against the limits")
    p.add_argument("--backend", choices=["mlx", "raceline"])

    models = ", ".join(__import__("raceline.pull", fromlist=["MODELS"]).MODELS)
    p = sub.add_parser("pull", help="download a converted model: " + models)
    p.add_argument("model", nargs="?", help=f"one of {models} (none: list them, with how each fits this Mac)")
    p.add_argument("--dest", help="file or directory to save to (default: the raceline cache)")
    p.add_argument("--source", choices=["github", "huggingface"],
                   help="where to download from (default: github, or $RACELINE_SOURCE)")

    args = ap.parse_args(argv)
    if args.command == "pull":
        from .pull import list_models, pull
        if args.model is None:
            list_models()
        else:
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
    rt = _runtime(args, agent=args.command == "chat")

    if args.command == "serve":
        from .server import serve
        serve(rt, getattr(rt.backend, "model_id", os.path.basename(args.model)), args.host, args.port)
    else:
        from .agent import Agent, repl
        agent = Agent(rt, args.cwd, auto_approve=args.yes, thinking=not args.no_think, max_tokens=args.max_tokens,
                      interactive=args.prompt is None, color=sys.stdout.isatty(), effort=args.effort)
        for path in args.image:
            if getattr(rt, "vision", None) is None:
                raise RuntimeError("this model takes no images: its vision file is missing "
                                   "(`raceline pull` fetches it with a converted model)")
            if not os.path.isfile(path):
                raise RuntimeError(f"no such image: {path}")
        if args.prompt is None:
            repl(agent, args.image)
        else:
            totals = agent.turn(args.prompt, args.image)
            print(agent.status(totals), file=sys.stderr)


if __name__ == "__main__":
    main()
