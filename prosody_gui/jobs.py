from __future__ import annotations

import itertools
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


JobFn = Callable[["Job"], Any]


@dataclass
class Job:
    id: str
    action: str
    status: str = "queued"
    result: Any = None
    error: str | None = None
    events: "queue.Queue[str | None]" = field(default_factory=queue.Queue)
    logs: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.logs.append(message)
        self.events.put(message)


class JobManager:
    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._job: Job | None = None
        self._lock = threading.Lock()

    def start(self, action: str, fn: JobFn) -> Job:
        job = Job(id=str(next(self._counter)), action=action)
        with self._lock:
            if self._job and self._job.status in {"queued", "running"}:
                raise RuntimeError("Wait for the current operation to finish.")
            self._job = job

        def run() -> None:
            job.status = "running"
            job.log(f"Started {action}.")
            try:
                job.result = fn(job)
                job.status = "done"
                job.log(f"Finished {action}.")
            except Exception as exc:  # noqa: BLE001 - surfaced to the browser log
                job.status = "failed"
                job.error = str(exc)
                job.log(f"Failed: {exc}")
                job.log(traceback.format_exc())
            finally:
                job.events.put(None)

        threading.Thread(target=run, daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            if self._job and self._job.id == job_id:
                return self._job
            return None

    def busy(self) -> bool:
        with self._lock:
            return bool(self._job and self._job.status in {"queued", "running"})
