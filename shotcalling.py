"""Shotcalling: bounded-concurrency batch throughput runner.

Fires many independent jobs at a node pool with bounded concurrency
(max_in_flight), per-job timeout, and partial-failure tolerance: one
slow or failing job never stalls the rest.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

__all__ = ["Shotcaller", "ShotcallingError"]

Job = Union[Callable[[], Any], Dict[str, Any]]


class ShotcallingError(Exception):
    """Raised for invalid configuration or job specs."""


class Shotcaller:
    """Run a batch of independent jobs with bounded concurrency.

    Args:
        jobs: iterable of callables, or dicts ``{"id": ..., "fn": callable}``.
            Jobs that are neither raise :class:`ShotcallingError`.
        max_in_flight: maximum jobs executing concurrently (>= 1).
        timeout_s: per-job wall-clock timeout in seconds (>= 0). A job
            exceeding it is reported as ``timeout``; its thread is not
            killed but its result is abandoned.

    ``run()`` returns results in deterministic input order, one entry per
    job: ``{"id", "status", "result"|"error", "duration_s"}`` where status
    is ``ok``, ``timeout``, or ``failed``. A failing job never affects
    the others.
    """

    def __init__(
        self,
        jobs: Iterable[Job],
        max_in_flight: int = 8,
        timeout_s: float = 30.0,
    ) -> None:
        if max_in_flight < 1:
            raise ShotcallingError("max_in_flight must be >= 1")
        if timeout_s < 0:
            raise ShotcallingError("timeout_s must be >= 0")

        self._specs: List[Dict[str, Any]] = []
        for i, job in enumerate(jobs):
            if callable(job):
                self._specs.append({"id": f"job-{i}", "fn": job})
            elif isinstance(job, dict) and callable(job.get("fn")):
                self._specs.append({"id": job.get("id", f"job-{i}"), "fn": job["fn"]})
            else:
                raise ShotcallingError(f"job at index {i} is not callable or a valid dict")

        self._max_in_flight = max_in_flight
        self._timeout_s = timeout_s

    @property
    def job_count(self) -> int:
        return len(self._specs)

    def run(self) -> Dict[str, Any]:
        """Execute all jobs; return ``{"results": [...], "summary": {...}}``."""
        results: List[Dict[str, Any]] = [None] * len(self._specs)  # type: ignore[list-item]
        lock = threading.Lock()

        def execute(index: int, fn: Callable[[], Any]) -> None:
            import time

            start = time.monotonic()
            entry: Dict[str, Any] = {"id": self._specs[index]["id"]}
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(fn)
                try:
                    value = future.result(timeout=self._timeout_s)
                    entry["status"] = "ok"
                    entry["result"] = value
                except FutureTimeout:
                    future.cancel()
                    entry["status"] = "timeout"
                    entry["error"] = f"timed out after {self._timeout_s}s"
            except Exception as exc:  # job failure is data, not a crash
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                entry["duration_s"] = round(time.monotonic() - start, 6)
                executor.shutdown(wait=False, cancel_futures=True)
                with lock:
                    results[index] = entry

        with ThreadPoolExecutor(max_workers=self._max_in_flight) as pool:
            for idx, spec in enumerate(self._specs):
                pool.submit(execute, idx, spec["fn"])

        summary = {"ok": 0, "timeout": 0, "failed": 0, "total": len(results)}
        for r in results:
            summary[r["status"]] += 1

        return {"results": results, "summary": summary}
