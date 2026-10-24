"""Run model-written Python in a subprocess with time and memory limits.

This is process isolation, not a security boundary. It is fine for math code on
a cluster node; do not point it at untrusted users.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover
    resource = None


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool

    def to_observation(self, max_chars: int = 2000) -> str:
        """Text the model sees as the tool response."""
        if self.timed_out:
            text = "Error: execution timed out."
        elif self.returncode != 0:
            err = self.stderr.strip().splitlines()
            # Keep the tail of the traceback: the actual error is on the last line.
            text = "Error:\n" + "\n".join(err[-8:])
        else:
            text = self.stdout.rstrip() or "(no output; use print() to see values)"
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text


def _limits(mem_bytes: int, cpu_seconds: int):
    def apply():
        if resource is None:
            return
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        os.setsid()  # own process group so a timeout kills children too

    return apply


def run_python(
    code: str,
    timeout_s: float = 10.0,
    mem_bytes: int = 2 * 1024**3,
) -> ExecResult:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_limits(mem_bytes, int(timeout_s) + 1),
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            return ExecResult(out, "", -1, True)
    return ExecResult(proc.stdout, proc.stderr, proc.returncode, False)
