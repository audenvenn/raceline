"""System facts read directly, without starting a process.

A process with GPU and tokenizer threads should not fork (see
raceline.agent.run_command), and running the `sysctl` command would be a fork.
"""

from __future__ import annotations

import ctypes
import ctypes.util


def _raw(name: str) -> bytes:
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
    except OSError:
        return b""
    size = ctypes.c_size_t(0)
    if libc.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0) != 0 or size.value == 0:
        return b""
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, 0) != 0:
        return b""
    return buf.raw[:size.value]


def sysctl_str(name: str) -> str:
    """A text sysctl, such as machdep.cpu.brand_string ("" if there is none)."""
    return _raw(name).rstrip(b"\0").decode(errors="replace")


def sysctl_int(name: str) -> int:
    """An integer sysctl, such as hw.memsize (0 if there is none)."""
    raw = _raw(name)
    return int.from_bytes(raw, "little") if len(raw) in (4, 8) else 0
