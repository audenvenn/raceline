"""Small coding tasks for bench/agent_eval.py.

Each task is a few files, the request a user would type, and a check the
agent never sees: a Python snippet run in the task's directory after the
agent is done (exit code 0 means solved; `ORIGINAL_FILES` holds the files as
they were). Files listed in `keep` must come out unchanged, so a task cannot
be solved by editing its tests.

The `core` set is small, single-file work; the `extended` set is closer to
a working day: several files, a long file, writing tests that must catch a
bug, a new module from a description.

Test scripts print nothing that changes from run to run (no timings), so
with a fixed seed every run of a task sees the same tool output and writes
the same tokens: runs that differ only in how drafts are made can be
compared token for token.
"""

TASKS = {}


def task(name, prompt, files, check, keep=(), group="core"):
    TASKS[name] = dict(name=name, prompt=prompt, files=files, check=check, keep=tuple(keep), group=group)


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


# ---------------------------------------------------------------- extended set

task(
    "cart",
    "test_cart.py fails. Find out why and fix the code, not the tests.",
    {
        "shop/__init__.py": "",
        "shop/pricing.py": '''"""Prices, discounts and tax for the shop."""

TAX_RATE = 0.08


def apply_discount(amount, percent):
    """amount reduced by percent (0 to 100)."""
    return amount * (1 - percent / 100)


def with_tax(amount):
    return round(amount * (1 + TAX_RATE), 2)
''',
        "shop/cart.py": '''"""A shopping cart."""

from .pricing import apply_discount, with_tax


class Cart:
    def __init__(self, discount_percent=0):
        self.lines = []
        self.discount_percent = discount_percent

    def add(self, name, unit_price, quantity=1):
        self.lines.append((name, unit_price, quantity))

    def subtotal(self):
        return sum(apply_discount(price * qty, self.discount_percent) for _, price, qty in self.lines)

    def total(self):
        """The subtotal less the discount, plus tax."""
        return with_tax(apply_discount(self.subtotal(), self.discount_percent))
''',
        "test_cart.py": '''from shop.cart import Cart


def test_subtotal_is_before_discount():
    c = Cart(discount_percent=10)
    c.add("pen", 2.50, 4)
    c.add("pad", 5.00)
    assert c.subtotal() == 15.0, c.subtotal()


def test_total_applies_discount_once_and_tax():
    c = Cart(discount_percent=10)
    c.add("pen", 2.50, 4)
    c.add("pad", 5.00)
    assert c.total() == 14.58, c.total()   # 15.00 less 10% is 13.50, plus 8% tax


def test_no_discount():
    c = Cart()
    c.add("book", 20.0, 2)
    assert c.total() == 43.2, c.total()


if __name__ == "__main__":
    test_subtotal_is_before_discount()
    test_total_applies_discount_once_and_tax()
    test_no_discount()
    print("all tests passed")
''',
    },
    """
import subprocess, sys
r = subprocess.run([sys.executable, "test_cart.py"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
from shop.cart import Cart
from shop.pricing import apply_discount, with_tax
c = Cart(25)
c.add("x", 100.0)
assert c.subtotal() == 100.0 and c.total() == 81.0, (c.subtotal(), c.total())
assert apply_discount(200, 50) == 100 and with_tax(100) == 108.0
""",
    keep=["test_cart.py"],
    group="extended",
)

task(
    "write-tests",
    "Write tests for parse_duration in durations.py, in a new file test_durations.py: at least four test "
    "functions covering hours, minutes, seconds, combinations and invalid input. It must run with "
    "`python3 test_durations.py` (pytest is not installed). Run it.",
    {
        "durations.py": '''"""Parse durations such as '1h30m', '45s' or '2h' into seconds."""

import re

_PART = re.compile(r"(\\d+)([hms])")


def parse_duration(text):
    """Seconds in a duration written as hours, minutes and seconds, in that
    order, each at most once: '1h30m' -> 5400, '45s' -> 45, '2h5s' -> 7205.
    Raises ValueError for anything else ('', '5x', '1m2h', '1h1h')."""
    text = text.strip()
    if not text or not re.fullmatch(r"(\\d+h)?(\\d+m)?(\\d+s)?", text):
        raise ValueError(f"not a duration: {text!r}")
    unit = {"h": 3600, "m": 60, "s": 1}
    return sum(int(n) * unit[u] for n, u in _PART.findall(text))
''',
    },
    """
import ast, subprocess, sys
src = open("test_durations.py").read()
# Test functions at module level or methods of a unittest.TestCase: both are what the task asks for.
tests = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]
assert len(tests) >= 4, f"{len(tests)} test functions"
def run():
    return subprocess.run([sys.executable, "test_durations.py"], capture_output=True, text=True, timeout=60)
r = run()
assert r.returncode == 0, "the tests fail on the correct code: " + r.stderr[-300:]
good = ORIGINAL_FILES["durations.py"]
mutants = [
    good.replace('"m": 60', '"m": 6'),                                         # minutes wrong
    good.replace('(\\\\d+h)?(\\\\d+m)?(\\\\d+s)?', '((\\\\d+[hms])*)'),                       # order and repeats accepted
    good.replace("if not text or not", "if not"),                               # '' accepted
    good.replace('"h": 3600', '"h": 360'),                                      # hours wrong
]
caught = 0
try:
    for m in mutants:
        assert m != good
        open("durations.py", "w").write(m)
        caught += run().returncode != 0
finally:
    open("durations.py", "w").write(good)
assert caught >= 3, f"the tests catch {caught} of {len(mutants)} broken versions"
""",
    keep=["durations.py"],
    group="extended",
)

task(
    "lru",
    "main.py uses an LRUCache that does not exist yet. Write lru.py with a class LRUCache(capacity): "
    "get(key, default=None) returns the value and makes the key the most recently used; put(key, value) "
    "inserts or updates it and, when the cache holds more than capacity entries, removes the least recently "
    "used one; len(cache) is the number of entries. `python3 main.py` must print `1 None 3`.",
    {
        "main.py": '''from lru import LRUCache

cache = LRUCache(2)
cache.put("a", 1)
cache.put("b", 2)
cache.get("a")
cache.put("c", 3)          # evicts "b", the least recently used
print(cache.get("a"), cache.get("b"), cache.get("c"))
''',
    },
    """
import collections, random, subprocess, sys
r = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
assert r.returncode == 0 and r.stdout == "1 None 3\\n", (r.stdout, r.stderr[-300:])
from lru import LRUCache
c = LRUCache(1)
c.put("a", 1); c.put("b", 2)
assert c.get("a") is None and c.get("b") == 2 and len(c) == 1
c = LRUCache(2)
c.put("a", 1); c.put("b", 2); c.put("a", 10)       # an update makes "a" recent
c.put("c", 3)
assert c.get("b", "gone") == "gone" and c.get("a") == 10 and len(c) == 2
rng = random.Random(7)
for cap in (1, 2, 3, 5):
    c, ref = LRUCache(cap), collections.OrderedDict()
    for _ in range(400):
        k = rng.randrange(8)
        if rng.random() < 0.5:
            v = rng.randrange(100)
            c.put(k, v); ref[k] = v; ref.move_to_end(k)
            if len(ref) > cap:
                ref.popitem(last=False)
        else:
            want = ref.get(k)
            if k in ref:
                ref.move_to_end(k)
            assert c.get(k) == want, (cap, k)
        assert len(c) == len(ref)
""",
    group="extended",
)

task(
    "csv-report",
    "Write report.py: it reads sales.csv and prints one line per region, `region: total`, with the total of "
    "the amount column to two decimals, the region with the largest total first (equal totals by name). "
    "Only the standard library.",
    {
        "sales.csv": "date,region,product,amount\n2026-01-03,north,widget,120.50\n2026-01-03,south,widget,80.00\n"
                     "2026-01-04,north,gadget,42.25\n2026-01-05,east,widget,99.99\n2026-01-05,south,gadget,15.00\n"
                     "2026-01-06,north,widget,7.25\n2026-01-07,east,gadget,0.01\n",
    },
    """
import subprocess, sys
def run():
    r = subprocess.run([sys.executable, "report.py"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-300:]
    return r.stdout
assert run() == "north: 170.00\\neast: 100.00\\nsouth: 95.00\\n", run()
try:
    open("sales.csv", "w").write("date,region,product,amount\\n2026-02-01,west,a,10\\n2026-02-01,alpha,b,5.25\\n"
                                 "2026-02-02,west,c,0.25\\n2026-02-03,beta,d,10.25\\n2026-02-04,alpha,e,5\\n"
                                 "2026-02-05,gamma,f,3.75\\n")
    assert run() == "alpha: 10.25\\nbeta: 10.25\\nwest: 10.25\\ngamma: 3.75\\n", run()
finally:
    open("sales.csv", "w").write(ORIGINAL_FILES["sales.csv"])
""",
    keep=["sales.csv"],
    group="extended",
)

task(
    "email",
    "test_validator.py fails. Fix is_email in validator.py; do not change the tests.",
    {
        "validator.py": '''"""Checks for user input."""

import re

EMAIL = re.compile(r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9-]+(\\.[a-zA-Z0-9-]+)*$")


def is_email(text):
    return bool(EMAIL.match(text))
''',
        "test_validator.py": '''from validator import is_email

GOOD = ["ann@example.com", "a.b-c_d@mail.example.org", "first+tag@example.co.uk", "x@y.io"]
BAD = ["ann", "ann@", "@example.com", "ann@example", "ann@@example.com", "ann @example.com",
       "ann@example..com", "ann@.example.com"]


def test_good():
    for e in GOOD:
        assert is_email(e), e


def test_bad():
    for e in BAD:
        assert not is_email(e), e


if __name__ == "__main__":
    test_good()
    test_bad()
    print("all tests passed")
''',
    },
    """
import subprocess, sys
r = subprocess.run([sys.executable, "test_validator.py"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-300:]
from validator import is_email
for e in ["a+b@c.de", "q@x.y.z", "A.B@EXAMPLE.COM"]:
    assert is_email(e), e
for e in ["a@b", "a b@c.de", "a@b.", "a@-", "", "a@b..c"]:
    assert not is_email(e), e
""",
    keep=["test_validator.py"],
    group="extended",
)


def _long_stats_module():
    lines = ['"""Small statistics helpers used across the reporting code."""', "", "import math", "", ""]
    for n in range(1, 41):
        lines += [f"def scale_{n}(xs):", f'    """Every value times {n}."""', f"    return [x * {n} for x in xs]", "", ""]
    lines += ['def mean(xs):', '    """Arithmetic mean; ValueError for no values."""', "    if not xs:",
              '        raise ValueError("mean of nothing")', "    return sum(xs) / len(xs)", "", ""]
    for n in range(41, 81):
        lines += [f"def shift_{n}(xs):", f'    """Every value plus {n}."""', f"    return [x + {n} for x in xs]", "", ""]
    lines += ["def percentile(xs, p):",
              '    """Nearest-rank percentile: the smallest value with at least p percent of the',
              "    values at or below it, for 0 < p <= 100; p = 0 gives the smallest value.",
              '    ValueError for no values or p outside 0..100."""',
              "    if not xs or not 0 <= p <= 100:",
              '        raise ValueError("bad percentile request")',
              "    s = sorted(xs)",
              "    return s[int(p / 100 * len(s))]", "", ""]
    for n in range(81, 121):
        lines += [f"def clip_{n}(xs):", f'    """Every value capped at {n}."""', f"    return [min(x, {n}) for x in xs]", "", ""]
    return "\n".join(lines).rstrip() + "\n"


task(
    "long-file",
    "test_stats.py fails. Find the bug in stats.py and fix it.",
    {
        "stats.py": _long_stats_module(),
        "test_stats.py": '''from stats import mean, percentile, scale_3


def test_mean():
    assert mean([1, 2, 3, 6]) == 3


def test_percentile():
    data = [15, 20, 35, 40, 50]
    assert percentile(data, 0) == 15
    assert percentile(data, 30) == 20
    assert percentile(data, 40) == 20
    assert percentile(data, 50) == 35
    assert percentile(data, 100) == 50


def test_scale():
    assert scale_3([1, 2]) == [3, 6]


if __name__ == "__main__":
    test_mean()
    test_percentile()
    test_scale()
    print("all tests passed")
''',
    },
    """
import math, random, subprocess, sys
r = subprocess.run([sys.executable, "test_stats.py"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-300:]
import stats
rng = random.Random(3)
for _ in range(300):
    xs = [rng.randrange(-50, 50) for _ in range(rng.randrange(1, 12))]
    p = rng.choice([0, 100, rng.randrange(1, 100), rng.uniform(0.1, 99.9)])
    s = sorted(xs)
    want = s[0] if p == 0 else s[math.ceil(p / 100 * len(s)) - 1]
    assert stats.percentile(xs, p) == want, (xs, p, stats.percentile(xs, p), want)
assert stats.clip_100([150, 5]) == [100, 5] and stats.shift_41([1]) == [42]
""",
    keep=["test_stats.py"],
    group="extended",
)

task(
    "signature",
    "Add a `channel` keyword argument to notify.send. With \"email\", the default, it works as now. With "
    "\"sms\", the string send returns starts with \"SMS: \" (before everything else) and is cut to at most 160 "
    "characters. Then make the overdue reminder in billing.py go out by SMS. Nothing else may change.",
    {
        "notify.py": '''"""Messages to users."""


def send(user, text):
    """The message as it goes out."""
    return f"To {user}: {text}"
''',
        "billing.py": '''from notify import send


def invoice(user, amount):
    return send(user, f"Your invoice for {amount:.2f} EUR is ready.")


def overdue(user, amount, days):
    return send(user, f"Your payment of {amount:.2f} EUR is {days} days overdue. Please pay as soon as "
                      "possible to avoid a late fee, or contact us if you believe this is an error.")
''',
        "signup.py": '''from notify import send


def welcome(user):
    return send(user, "Welcome aboard!")
''',
    },
    """
import inspect
import notify, billing, signup
assert notify.send("ann", "hi") == "To ann: hi"
assert notify.send("ann", "hi", channel="email") == "To ann: hi"
sms = notify.send("ann", "x" * 300, channel="sms")
assert sms.startswith("SMS: ") and len(sms) == 160, (sms[:30], len(sms))
assert notify.send("ann", "short", channel="sms").startswith("SMS: ") and "short" in notify.send("ann", "short", channel="sms")
assert inspect.signature(notify.send).parameters["channel"].default == "email"
assert billing.invoice("bo", 12.5) == "To bo: Your invoice for 12.50 EUR is ready."
o = billing.overdue("bo", 12.5, 9)
assert o.startswith("SMS: ") and len(o) <= 160 and "9 days overdue" in o, o
assert signup.welcome("cy") == "To cy: Welcome aboard!"
""",
    group="extended",
)

task(
    "todo-cli",
    "Implement the `done` command of todo.py: `python3 todo.py done N` marks item N (as numbered by `list`) "
    "as done, and `list` then shows it with [x]. A number that does not exist prints an error to stderr and "
    "exits with status 1.",
    {
        "todo.py": '''"""A to-do list on the command line, kept in todo.json."""

import argparse
import json
import os
import sys

PATH = "todo.json"


def load():
    if not os.path.exists(PATH):
        return []
    with open(PATH) as f:
        return json.load(f)


def save(items):
    with open(PATH, "w") as f:
        json.dump(items, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("add")
    p.add_argument("text")
    p = sub.add_parser("done")
    p.add_argument("n", type=int)
    args = ap.parse_args(argv)
    items = load()
    if args.cmd == "add":
        items.append({"text": args.text, "done": False})
        save(items)
    elif args.cmd == "list":
        for i, item in enumerate(items, 1):
            print(f"{i}. [{'x' if item['done'] else ' '}] {item['text']}")
    elif args.cmd == "done":
        print("not implemented yet", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
''',
    },
    """
import os, subprocess, sys
if os.path.exists("todo.json"):
    os.remove("todo.json")
def run(*a):
    return subprocess.run([sys.executable, "todo.py", *a], capture_output=True, text=True, timeout=60)
for t in ("buy milk", "call bob", "write report"):
    assert run("add", t).returncode == 0
r = run("done", "2")
assert r.returncode == 0, r.stderr
assert run("list").stdout == "1. [ ] buy milk\\n2. [x] call bob\\n3. [ ] write report\\n", run("list").stdout
r = run("done", "7")
assert r.returncode == 1 and r.stderr.strip(), (r.returncode, r.stderr)
r = run("done", "0")
assert r.returncode == 1, r.returncode
assert run("list").stdout.count("[x]") == 1
os.remove("todo.json")
""",
    group="extended",
)
