"""Small coding tasks for bench/agent_eval.py.

Each task is a few files, the request a user would type, and a check the
agent never sees: a Python snippet run in the task's directory after the
agent is done (exit code 0 means solved; `ORIGINAL_FILES` holds the files as
they were). Files listed in `keep` must come out unchanged, so a task cannot
be solved by editing its tests.

Test scripts print nothing that changes from run to run (no timings), so
with a fixed seed every run of a task sees the same tool output and writes
the same tokens: runs that differ only in how drafts are made can be
compared token for token.
"""

TASKS = {}


def task(name, prompt, files, check, keep=()):
    TASKS[name] = dict(name=name, prompt=prompt, files=files, check=check, keep=tuple(keep))


task(
    "inventory",
    "Run test_inventory.py. If a test fails, find the cause in inventory.py, fix it, and run the tests again "
    "until they all pass.",
    {
        "inventory.py": '''"""A tiny inventory: items with a unit price and a quantity."""


class Inventory:
    def __init__(self):
        self.items = {}

    def add(self, name, price, quantity=1):
        if name in self.items:
            self.items[name]["quantity"] += quantity
        else:
            self.items[name] = {"price": price, "quantity": quantity}

    def remove(self, name, quantity=1):
        item = self.items[name]
        item["quantity"] -= quantity
        if item["quantity"] < 0:
            del self.items[name]

    def total_value(self):
        return sum(item["price"] for item in self.items.values())

    def count(self):
        return sum(item["quantity"] for item in self.items.values())
''',
        "test_inventory.py": '''from inventory import Inventory


def test_total_value():
    inv = Inventory()
    inv.add("apple", 0.5, 4)
    inv.add("pear", 0.75, 2)
    assert inv.total_value() == 3.5, inv.total_value()


def test_remove_last():
    inv = Inventory()
    inv.add("apple", 0.5, 2)
    inv.remove("apple", 2)
    assert "apple" not in inv.items
    assert inv.count() == 0


if __name__ == "__main__":
    test_total_value()
    test_remove_last()
    print("all tests passed")
''',
    },
    """
import subprocess, sys
r = subprocess.run([sys.executable, "test_inventory.py"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
from inventory import Inventory
inv = Inventory()
inv.add("a", 2.0, 3)
inv.remove("a", 1)
assert inv.total_value() == 4.0 and inv.count() == 2
""",
    keep=["test_inventory.py"],
)

task(
    "rename",
    "Rename the function calc_area to area everywhere in this project, call sites included. "
    "`python3 main.py` must print exactly what it prints now.",
    {
        "geometry.py": '''"""Areas and perimeters of simple shapes."""

import math


def calc_area(shape, *size):
    """Area of a 'square', 'rectangle' or 'circle'."""
    if shape == "square":
        return size[0] ** 2
    if shape == "rectangle":
        return size[0] * size[1]
    if shape == "circle":
        return math.pi * size[0] ** 2
    raise ValueError(f"unknown shape: {shape}")


def perimeter(shape, *size):
    """Perimeter of a 'square', 'rectangle' or 'circle'."""
    if shape == "square":
        return 4 * size[0]
    if shape == "rectangle":
        return 2 * (size[0] + size[1])
    if shape == "circle":
        return 2 * math.pi * size[0]
    raise ValueError(f"unknown shape: {shape}")
''',
        "report.py": '''from geometry import calc_area, perimeter


def describe(shape, *size):
    a = calc_area(shape, *size)
    p = perimeter(shape, *size)
    return f"{shape} {size}: area {a:.2f}, perimeter {p:.2f}"


def largest(shapes):
    """The (shape, size) with the largest area."""
    return max(shapes, key=lambda s: calc_area(s[0], *s[1]))
''',
        "main.py": '''from report import describe, largest

SHAPES = [("square", (3,)), ("rectangle", (2, 5)), ("circle", (1.5,))]

for shape, size in SHAPES:
    print(describe(shape, *size))
print("largest:", largest(SHAPES)[0])
''',
    },
    """
import pathlib, subprocess, sys
for p in pathlib.Path(".").glob("*.py"):
    assert "calc_area" not in p.read_text(), p
r = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
assert r.stdout == (
    "square (3,): area 9.00, perimeter 12.00\\n"
    "rectangle (2, 5): area 10.00, perimeter 14.00\\n"
    "circle (1.5,): area 7.07, perimeter 9.42\\n"
    "largest: rectangle\\n"), r.stdout
from geometry import area
assert area("square", 2) == 4
""",
)

task(
    "slugify",
    "Implement slugify in text_utils.py as its docstring describes.",
    {
        "text_utils.py": '''"""Helpers for turning titles into URLs."""


def slugify(title: str, max_length: int = 60) -> str:
    """Turn a title into a URL slug.

    - lowercase
    - every run of characters other than a-z and 0-9 becomes a single hyphen
    - no hyphen at the start or the end
    - at most max_length characters, cut at a hyphen when the slug is too
      long (never in the middle of a word, unless the first word alone is
      longer than max_length, in which case it is cut at max_length)

    >>> slugify("Hello, World!")
    'hello-world'
    >>> slugify("  Python 3.14 -- what's new?  ")
    'python-3-14-what-s-new'
    >>> slugify("one two three", max_length=9)
    'one-two'
    """
    raise NotImplementedError
''',
    },
    """
from text_utils import slugify
cases = [
    (("Hello, World!",), "hello-world"),
    (("  Python 3.14 -- what's new?  ",), "python-3-14-what-s-new"),
    (("one two three",), "one-two-three"),
    (("one two three", 9), "one-two"),
    (("one two three", 7), "one-two"),
    (("one two three", 6), "one"),
    (("supercalifragilistic", 5), "super"),
    (("---",), ""),
    (("CamelCase_and_snake",), "camelcase-and-snake"),
    (("a" * 70,), "a" * 60),
]
for args, want in cases:
    got = slugify(*args)
    assert got == want, (args, got, want)
""",
)

task(
    "cli-flag",
    "Add a --min-length option to wordcount.py: words shorter than that many characters are not counted "
    "(default 1, which counts every word). Everything else must work as before.",
    {
        "wordcount.py": '''"""Print the most common words of a text file."""

import argparse
import collections
import re


def count_words(text):
    return collections.Counter(re.findall(r"[a-z']+", text.lower()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=5, help="how many words to print")
    args = ap.parse_args()
    with open(args.path, encoding="utf-8") as f:
        counts = count_words(f.read())
    for word, n in counts.most_common(args.top):
        print(f"{n:5d} {word}")


if __name__ == "__main__":
    main()
''',
        "sample.txt": "The cat and the dog. The cat sat on a mat, and a dog sat on the cat's mat.\n"
                      "A big dog is a happy dog; the small cat is a happy cat.\n",
    },
    """
import subprocess, sys
def run(*args):
    r = subprocess.run([sys.executable, "wordcount.py", "sample.txt", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout
assert run() == "    5 the\\n    5 a\\n    4 cat\\n    4 dog\\n    2 and\\n", run()
assert run("--top", "3", "--min-length", "4") == "    2 happy\\n    1 cat's\\n    1 small\\n", run("--top", "3", "--min-length", "4")
assert run("--min-length", "1", "--top", "2") == "    5 the\\n    5 a\\n"
""",
    keep=["sample.txt"],
)

task(
    "config-defaults",
    "load_config in config.py crashes when settings.json leaves out optional keys. Make timeout default to 30 "
    "and retries default to 3. name stays required: if it is missing, raise a ValueError that says so.",
    {
        "config.py": '''"""Settings for the sync job, read from a JSON file."""

import json
from dataclasses import dataclass


@dataclass
class Config:
    name: str
    timeout: int
    retries: int


def load_config(path):
    with open(path) as f:
        data = json.load(f)
    return Config(name=data["name"], timeout=data["timeout"], retries=data["retries"])
''',
        "settings.json": '{\n  "name": "nightly"\n}\n',
    },
    """
import json, os, tempfile
from config import Config, load_config
assert load_config("settings.json") == Config("nightly", 30, 3)
d = tempfile.mkdtemp()
def cfg(obj):
    p = os.path.join(d, "s.json")
    with open(p, "w") as f:
        json.dump(obj, f)
    return load_config(p)
assert cfg({"name": "a", "timeout": 5}) == Config("a", 5, 3)
assert cfg({"name": "b", "retries": 0}) == Config("b", 30, 0)
assert cfg({"name": "c", "timeout": 1, "retries": 2}) == Config("c", 1, 2)
try:
    cfg({"timeout": 1})
except ValueError as e:
    assert "name" in str(e), e
else:
    raise AssertionError("no ValueError for a missing name")
""",
    keep=["settings.json"],
)

task(
    "docstrings",
    "Add a one-line docstring to every function in mathlib.py that has none. Do not change any code.",
    {
        "mathlib.py": '''"""Small numeric helpers."""


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("mean of an empty sequence")
    return sum(values) / len(values)


def median(values):
    s = sorted(values)
    if not s:
        raise ValueError("median of an empty sequence")
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def gcd(a, b):
    """Greatest common divisor (Euclid)."""
    while b:
        a, b = b, a % b
    return abs(a)


def lerp(a, b, t):
    return a + (b - a) * t


def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True
''',
    },
    """
import ast
src = open("mathlib.py").read()
tree = ast.parse(src)
funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
assert sorted(f.name for f in funcs) == ["clamp", "gcd", "is_prime", "lerp", "mean", "median"], [f.name for f in funcs]
for f in funcs:
    doc = ast.get_docstring(f)
    assert doc and doc.strip(), f.name
assert ast.get_docstring([f for f in funcs if f.name == "gcd"][0]) == "Greatest common divisor (Euclid)."
def strip(t):
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.Module)) and n.body and isinstance(n.body[0], ast.Expr) \\
                and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:]
    return ast.dump(t)
assert strip(tree) == strip(ast.parse(ORIGINAL_FILES["mathlib.py"])), "the code changed"
""",
)

task(
    "pathlib",
    "Rewrite files.py to use pathlib instead of os.path and os.listdir. Keep every function's name, arguments "
    "and return type (paths are still returned as strings).",
    {
        "files.py": '''"""File helpers for the build scripts."""

import os


def find_by_extension(folder, extension):
    """Sorted paths of the files in folder (not its subfolders) with the given extension, e.g. '.txt'."""
    out = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1] == extension:
            out.append(path)
    return sorted(out)


def replace_extension(path, extension):
    """The same path with another extension: replace_extension('a/b.txt', '.md') == 'a/b.md'."""
    return os.path.splitext(path)[0] + extension


def stem(path):
    """File name without folder and extension: stem('a/b.tar.gz') == 'b.tar'."""
    return os.path.splitext(os.path.basename(path))[0]


def ensure_folder(path):
    """Create the folder (and its parents) if it does not exist; return the path."""
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def total_size(folder):
    """Total size in bytes of the files directly in folder."""
    total = 0
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    return total
''',
    },
    """
import os, tempfile
src = open("files.py").read()
assert "os.path" not in src and "os.listdir" not in src, "still uses os.path / os.listdir"
assert "pathlib" in src
import files
d = tempfile.mkdtemp()
for name, data in [("a.txt", "x"), ("b.txt", "yy"), ("c.md", "zzz")]:
    with open(os.path.join(d, name), "w") as f:
        f.write(data)
os.mkdir(os.path.join(d, "sub.txt"))
got = files.find_by_extension(d, ".txt")
assert got == [os.path.join(d, "a.txt"), os.path.join(d, "b.txt")], got
assert all(isinstance(p, str) for p in got)
assert files.replace_extension("a/b.txt", ".md") == "a/b.md"
assert files.stem("a/b.tar.gz") == "b.tar"
new = os.path.join(d, "x", "y")
assert files.ensure_folder(new) == new and os.path.isdir(new)
assert files.ensure_folder(new) == new
assert files.total_size(d) == 6
""",
)

task(
    "bisect",
    "test_search.py fails. Find the bug in search.py and fix it.",
    {
        "search.py": '''"""Binary search over a sorted list."""


def insert_position(items, value):
    """Index at which value would be inserted to keep items sorted (after any equal items)."""
    lo, hi = 0, len(items) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def contains(items, value):
    i = insert_position(items, value)
    return i > 0 and items[i - 1] == value
''',
        "test_search.py": '''from search import contains, insert_position


def test_insert_position():
    items = [1, 3, 3, 5, 8]
    assert insert_position(items, 0) == 0
    assert insert_position(items, 3) == 3
    assert insert_position(items, 4) == 3
    assert insert_position(items, 8) == 5
    assert insert_position(items, 9) == 5
    assert insert_position([], 1) == 0


def test_contains():
    items = [2, 4, 6]
    assert contains(items, 6)
    assert not contains(items, 7)
    assert not contains([], 1)


if __name__ == "__main__":
    test_insert_position()
    test_contains()
    print("all tests passed")
''',
    },
    """
import bisect, random, subprocess, sys
r = subprocess.run([sys.executable, "test_search.py"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
from search import contains, insert_position
rng = random.Random(0)
for _ in range(300):
    items = sorted(rng.randrange(20) for _ in range(rng.randrange(12)))
    v = rng.randrange(-2, 22)
    assert insert_position(items, v) == bisect.bisect_right(items, v), (items, v)
    assert contains(items, v) == (v in items)
""",
    keep=["test_search.py"],
)
