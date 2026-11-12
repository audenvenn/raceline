#!/usr/bin/env python3
"""The agent tasks of bench/agent_tasks.py are fair: each task's hidden check
fails on the files as handed out and passes on a reference solution.

  python3 tests/test_agent_tasks.py        (no GPU, no model)
"""

import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bench"))
from agent_eval import check  # noqa: E402
from agent_tasks import TASKS  # noqa: E402


def _slugify(title, max_length=60):
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if len(slug) <= max_length:
        return slug
    cut = slug.rfind("-", 0, max_length + 1)
    return slug[:cut] if cut > 0 else slug[:max_length]


SOLUTIONS = {
    "inventory": {"inventory.py": lambda s: s.replace('item["quantity"] < 0', 'item["quantity"] <= 0').replace(
        'item["price"] for', 'item["price"] * item["quantity"] for')},
    "rename": {f: (lambda s: s.replace("calc_area", "area")) for f in ("geometry.py", "report.py")},
    "slugify": {"text_utils.py": lambda s: s.replace("    raise NotImplementedError\n", "\n".join(
        "    " + line if line else line for line in [
            "import re", "slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')",
            "if len(slug) <= max_length:", "    return slug", "cut = slug.rfind('-', 0, max_length + 1)",
            "return slug[:cut] if cut > 0 else slug[:max_length]", ""]))},
    "cli-flag": {"wordcount.py": lambda s: s.replace(
        '    ap.add_argument("--top", type=int, default=5, help="how many words to print")\n',
        '    ap.add_argument("--top", type=int, default=5, help="how many words to print")\n'
        '    ap.add_argument("--min-length", type=int, default=1)\n').replace(
        "        counts = count_words(f.read())\n",
        "        counts = count_words(f.read())\n"
        "    counts = collections.Counter({w: n for w, n in counts.items() if len(w) >= args.min_length})\n")},
    "config-defaults": {"config.py": lambda s: s.replace(
        '    return Config(name=data["name"], timeout=data["timeout"], retries=data["retries"])',
        '    if "name" not in data:\n        raise ValueError("settings: name is missing")\n'
        '    return Config(name=data["name"], timeout=data.get("timeout", 30), retries=data.get("retries", 3))')},
    "docstrings": {"mathlib.py": lambda s: re.sub(
        r"(def (clamp|mean|median|lerp|is_prime)\(.*\):\n)", r'\1    """Does what its name says."""\n', s)},
    "pathlib": {"files.py": lambda s: '''"""File helpers for the build scripts."""

from pathlib import Path


def find_by_extension(folder, extension):
    return sorted(str(p) for p in Path(folder).iterdir() if p.is_file() and p.suffix == extension)


def replace_extension(path, extension):
    return str(Path(path).with_suffix(extension))


def stem(path):
    return Path(path).stem


def ensure_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def total_size(folder):
    return sum(p.stat().st_size for p in Path(folder).iterdir() if p.is_file())
'''},
    "bisect": {"search.py": lambda s: s.replace("lo, hi = 0, len(items) - 1", "lo, hi = 0, len(items)")},
    "cart": {"shop/cart.py": lambda s: s.replace(
        "sum(apply_discount(price * qty, self.discount_percent) for _, price, qty in self.lines)",
        "sum(price * qty for _, price, qty in self.lines)")},
    "write-tests": {"test_durations.py": lambda s: '''from durations import parse_duration


def test_units():
    assert parse_duration("2h") == 7200 and parse_duration("3m") == 180 and parse_duration("45s") == 45


def test_combined():
    assert parse_duration("1h30m") == 5400 and parse_duration("2h5s") == 7205 and parse_duration("1h1m1s") == 3661


def expect_error(text):
    try:
        parse_duration(text)
    except ValueError:
        return
    raise AssertionError(f"no ValueError for {text!r}")


def test_invalid():
    for text in ("", "5x", "1m2h", "1h1h", "h"):
        expect_error(text)


def test_spaces():
    assert parse_duration(" 10m ") == 600


if __name__ == "__main__":
    test_units()
    test_combined()
    test_invalid()
    test_spaces()
    print("all tests passed")
'''},
    "lru": {"lru.py": lambda s: '''from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity, self.data = capacity, OrderedDict()

    def get(self, key, default=None):
        if key not in self.data:
            return default
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)

    def __len__(self):
        return len(self.data)
'''},
    "csv-report": {"report.py": lambda s: '''import csv
from collections import defaultdict
from decimal import Decimal

totals = defaultdict(Decimal)
with open("sales.csv", newline="") as f:
    for row in csv.DictReader(f):
        totals[row["region"]] += Decimal(row["amount"])
for region, total in sorted(totals.items(), key=lambda kv: (-round(kv[1], 2), kv[0])):
    print(f"{region}: {total:.2f}")
'''},
    "email": {"validator.py": lambda s: s.replace(r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*$",
                                                  r"^[a-zA-Z0-9._+-]+@[a-zA-Z0-9]+(-[a-zA-Z0-9]+)*(\.[a-zA-Z0-9-]+)+$")},
    "long-file": {"stats.py": lambda s: s.replace(
        "    return s[int(p / 100 * len(s))]",
        "    import math\n    return s[0] if p == 0 else s[math.ceil(p / 100 * len(s)) - 1]")},
    "signature": {
        "notify.py": lambda s: s.replace('''def send(user, text):
    """The message as it goes out."""
    return f"To {user}: {text}"''', '''def send(user, text, channel="email"):
    """The message as it goes out."""
    if channel == "sms":
        return ("SMS: " + f"To {user}: {text}")[:160]
    return f"To {user}: {text}"'''),
        "billing.py": lambda s: s.replace('''possible to avoid a late fee, or contact us if you believe this is an error.")''',
                                          '''possible to avoid a late fee, or contact us if you believe this is an error.",
                channel="sms")''')},
    "todo-cli": {"todo.py": lambda s: s.replace('''        print("not implemented yet", file=sys.stderr)
        return 2''', '''        if not 1 <= args.n <= len(items):
            print(f"no item {args.n}", file=sys.stderr)
            return 1
        items[args.n - 1]["done"] = True
        save(items)''')},
}


def _materialize(task, edits=None):
    work = tempfile.mkdtemp()
    files = dict(task["files"])
    for name in edits or {}:
        files.setdefault(name, None)             # a file the solution creates
    for name, text in files.items():
        if edits and name in edits:
            new = edits[name](text)
            assert new != text, f"{task['name']}: the reference edit of {name} changed nothing"
            text = new
        if text is None:
            continue
        os.makedirs(os.path.dirname(os.path.join(work, name)), exist_ok=True)
        with open(os.path.join(work, name), "w", encoding="utf-8") as f:
            f.write(text)
    return work


def test_every_task_has_a_reference_solution():
    assert sorted(SOLUTIONS) == sorted(TASKS)


def test_checks_fail_as_handed_out_and_pass_when_solved():
    assert _slugify("one two three", 9) == "one-two"
    for name, task in TASKS.items():
        work = _materialize(task)
        solved, _ = check(task, work)
        shutil.rmtree(work)
        assert not solved, f"{name}: solved before the agent did anything"
        work = _materialize(task, SOLUTIONS[name])
        solved, why = check(task, work)
        shutil.rmtree(work)
        assert solved, f"{name}: the reference solution fails its check: {why}"


def test_changing_a_kept_file_fails_the_task():
    task = TASKS["inventory"]
    work = _materialize(task, {"test_inventory.py": lambda s: s.replace("== 3.5", "== 2.5"),
                               **SOLUTIONS["inventory"]})
    solved, why = check(task, work)
    shutil.rmtree(work)
    assert not solved and "test_inventory.py" in why


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} passed")
