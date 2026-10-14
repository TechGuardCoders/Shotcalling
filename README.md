# Shotcalling

Bulk batch throughput runner: fire many independent jobs at a node pool with
bounded concurrency, per-job timeout, and partial-failure tolerance. One slow
or failing job never stalls the rest.

## Results

`Shotcaller(jobs, max_in_flight=8, timeout_s=30).run()` returns results in
deterministic input order with per-job status (`ok` / `timeout` / `failed`)
plus summary counts:

```python
from shotcalling import Shotcaller

def probe(node):
    return f"{node} healthy"

c = Shotcaller(
    [{"id": n, "fn": lambda n=n: probe(n)} for n in ("a", "b", "c")],
    max_in_flight=8,
    timeout_s=5,
)
out = c.run()
print(out["summary"])   # {'ok': 3, 'timeout': 0, 'failed': 0, 'total': 3}
print(out["results"][0])  # {'id': 'a', 'status': 'ok', 'result': 'a healthy', 'duration_s': ...}
```

Jobs are bare callables or `{"id", "fn"}` dicts. A job that raises is
reported as `failed` with its exception; a job that exceeds `timeout_s` is
reported as `timeout` — neither affects the other jobs.

## Architecture

```
jobs ──► Shotcaller ──► ThreadPoolExecutor(max_in_flight)
                            │  one wrapper task per job
                            │  each wrapper runs fn in its own 1-worker
                            │  executor with future.result(timeout_s)
                            ▼
                 results[i] (input order, lock-guarded writes)
                            ▼
                 {"results": [...], "summary": {ok, timeout, failed, total}}
```

- **Bounded concurrency**: a single `ThreadPoolExecutor` caps in-flight jobs
  at `max_in_flight`.
- **Per-job timeout**: each job runs under its own future with
  `result(timeout_s)`; expired jobs are abandoned, never block the batch.
- **Partial-failure tolerance**: exceptions are caught per job and recorded
  as data; the batch always completes.
- **Deterministic ordering**: results are written to fixed input indices.

## Why

Fan-out work (health checks, batch inference, per-node maintenance) fails in
two annoying ways: one hung node stalls the whole batch, and one exception
kills the run. Shotcalling makes both impossible while keeping throughput
bounded so you don't flatten the pool.

Pure Python stdlib (`threading` / `concurrent.futures`). No network required.

## Development

```bash
uv venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install pytest
pytest -q
```
