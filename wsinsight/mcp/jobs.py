"""Background-job manager for long-running WSInsight subcommands.

Each job runs as a subprocess invocation of ``python -m wsinsight ...``,
which gives us:

* clean cancellation via SIGINT (the existing :mod:`wsinsight.cancel`
  two-press handler is reused exactly as written);
* per-job GPU isolation via ``CUDA_VISIBLE_DEVICES``;
* no thread-safety concerns with TensorFlow / PyTorch / Click globals.

Jobs are scheduled onto a fixed pool of workers; the default pool size
equals the number of GPUs visible to the host (or 1 on CPU-only
machines), and can be overridden with ``wsinsight-mcp --max-concurrent N``.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

_LOG_RING_SIZE = 4000  # lines kept per job
_LOG_LINE_TRUNCATE = 2000  # chars per line


def _detect_default_max_concurrent() -> int:
    """Default concurrency = number of GPUs (or 1 if no GPUs)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [tok for tok in visible.split(",") if tok.strip() != ""]
        return max(1, len(ids))
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except Exception:
        pass
    return 1


def _visible_gpu_ids() -> list[str] | None:
    """Return CUDA_VISIBLE_DEVICES as a list, or None if unset."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None
    return [tok.strip() for tok in visible.split(",") if tok.strip() != ""]


@dataclass
class JobState:
    """Snapshot-friendly state of one background job."""

    id: str
    command: str
    argv: list[str]
    status: str = "pending"  # pending | running | done | failed | cancelled
    pid: int | None = None
    gpu_id: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    error: str | None = None
    log_lines: deque = field(
        default_factory=lambda: deque(maxlen=_LOG_RING_SIZE)
    )
    total_lines: int = 0  # monotonically increasing line counter
    cancel_requested_at: float | None = None
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _reader: threading.Thread | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable snapshot (no logs, no proc handle)."""
        duration = None
        if self.started_at is not None:
            end = self.finished_at if self.finished_at is not None else time.time()
            duration = round(end - self.started_at, 3)
        return {
            "id": self.id,
            "command": self.command,
            "argv": list(self.argv),
            "status": self.status,
            "pid": self.pid,
            "gpu_id": self.gpu_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": duration,
            "returncode": self.returncode,
            "error": self.error,
            "total_log_lines": self.total_lines,
            "cancel_requested": self.cancel_requested_at is not None,
        }


class JobManager:
    """Process-pool-style scheduler for ``wsinsight`` subprocess jobs."""

    def __init__(
        self,
        max_concurrent: int | None = None,
        experimental: bool = False,
    ) -> None:
        self.max_concurrent = (
            max_concurrent if max_concurrent and max_concurrent > 0
            else _detect_default_max_concurrent()
        )
        self.experimental = experimental
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        # GPU-pinning queue. Empty list ⇒ CPU-only host (no pinning done).
        self._gpu_pool: list[str] | None = _visible_gpu_ids()
        self._gpu_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(value=self.max_concurrent)

    # ----- public API ------------------------------------------------------

    def submit(self, command: str, argv_tail: list[str]) -> str:
        """Schedule a new job. Returns the job_id immediately."""
        job_id = uuid.uuid4().hex[:12]
        argv = [sys.executable, "-m", "wsinsight"] + argv_tail
        state = JobState(id=job_id, command=command, argv=argv)
        with self._lock:
            self._jobs[job_id] = state
        threading.Thread(
            target=self._run_job, args=(state,), daemon=True, name=f"wsinsight-job-{job_id}"
        ).start()
        return job_id

    def status(self, job_id: str) -> dict | None:
        """Return a snapshot dict, or None if no such job."""
        with self._lock:
            j = self._jobs.get(job_id)
        return j.to_dict() if j else None

    def logs(self, job_id: str, since_line: int = 0, max_lines: int = 500) -> dict | None:
        """Return ``{lines, next_line, total}`` for the job, or None."""
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            total = j.total_lines
            buffered = list(j.log_lines)
        # log_lines holds at most _LOG_RING_SIZE lines; the i-th buffered line
        # corresponds to absolute line index (total - len(buffered) + i).
        first_buffered_idx = total - len(buffered)
        if since_line < first_buffered_idx:
            since_line = first_buffered_idx
        start = since_line - first_buffered_idx
        chunk = buffered[start : start + max_lines]
        next_line = since_line + len(chunk)
        return {
            "job_id": job_id,
            "lines": chunk,
            "next_line": next_line,
            "total": total,
            "truncated": first_buffered_idx > 0,
        }

    def cancel(self, job_id: str) -> dict | None:
        """Send SIGINT to the job; second call escalates to SIGTERM."""
        with self._lock:
            j = self._jobs.get(job_id)
        if j is None:
            return None
        if j.status not in ("pending", "running"):
            return j.to_dict()
        proc = j._proc
        now = time.time()
        if proc is None:
            j.status = "cancelled"
            j.cancel_requested_at = now
            j.finished_at = now
            return j.to_dict()
        try:
            if j.cancel_requested_at is None:
                # First cancel: graceful (SIGINT triggers wsinsight.cancel).
                proc.send_signal(signal.SIGINT)
                j.cancel_requested_at = now
            else:
                # Second cancel: hard.
                proc.terminate()
        except ProcessLookupError:
            pass
        return j.to_dict()

    def list(self) -> list[dict]:
        """Return snapshot dicts for all known jobs."""
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    # ----- internal --------------------------------------------------------

    def _acquire_gpu(self) -> str | None:
        if not self._gpu_pool:
            return None
        with self._gpu_lock:
            return self._gpu_pool.pop(0) if self._gpu_pool else None

    def _release_gpu(self, gpu_id: str | None) -> None:
        if gpu_id is None or self._gpu_pool is None:
            return
        with self._gpu_lock:
            self._gpu_pool.append(gpu_id)

    def _run_job(self, state: JobState) -> None:
        # Throttle concurrency.
        self._slots.acquire()
        gpu_id: str | None = None
        try:
            gpu_id = self._acquire_gpu()
            state.gpu_id = gpu_id
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            if self.experimental:
                env["WSINSIGHT_EXPERIMENTAL"] = "1"
            # Force unbuffered child stdout for real-time log streaming.
            env.setdefault("PYTHONUNBUFFERED", "1")
            state.started_at = time.time()
            state.status = "running"
            try:
                proc = subprocess.Popen(  # noqa: S603
                    state.argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError as exc:
                state.status = "failed"
                state.error = f"subprocess launch failed: {exc}"
                state.finished_at = time.time()
                return
            state._proc = proc
            state.pid = proc.pid
            state._reader = threading.Thread(
                target=self._pump_logs,
                args=(state, proc),
                daemon=True,
                name=f"wsinsight-job-{state.id}-log",
            )
            state._reader.start()
            rc = proc.wait()
            # Drain any remaining buffered log lines.
            if state._reader is not None:
                state._reader.join(timeout=2.0)
            state.returncode = rc
            state.finished_at = time.time()
            if state.cancel_requested_at is not None and rc != 0:
                state.status = "cancelled"
            elif rc == 0:
                state.status = "done"
            else:
                state.status = "failed"
                if state.error is None:
                    state.error = f"process exited with code {rc}"
        finally:
            self._release_gpu(gpu_id)
            self._slots.release()

    @staticmethod
    def _pump_logs(state: JobState, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if len(line) > _LOG_LINE_TRUNCATE:
                    line = line[:_LOG_LINE_TRUNCATE] + "…[truncated]"
                state.log_lines.append(line)
                state.total_lines += 1
        except Exception as exc:  # pragma: no cover
            state.log_lines.append(f"[mcp] log reader crashed: {exc!r}")
            state.total_lines += 1
        finally:
            with contextlib.suppress(Exception):
                proc.stdout.close()


__all__ = ["JobManager", "JobState"]
