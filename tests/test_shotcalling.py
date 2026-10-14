import threading
import time

import pytest

from shotcalling import Shotcaller, ShotcallingError


def test_all_ok_basic():
    c = Shotcaller([lambda: 1, lambda: 2, lambda: 3])
    out = c.run()
    assert [r["result"] for r in out["results"]] == [1, 2, 3]
    assert out["summary"] == {"ok": 3, "timeout": 0, "failed": 0, "total": 3}


def test_dict_jobs_with_ids():
    c = Shotcaller([{"id": "a", "fn": lambda: "x"}, {"id": "b", "fn": lambda: "y"}])
    out = c.run()
    assert [r["id"] for r in out["results"]] == ["a", "b"]


def test_deterministic_ordering_despite_variable_duration():
    def slow():
        time.sleep(0.05)
        return "slow"

    def fast():
        return "fast"

    c = Shotcaller([slow, fast, slow, fast], max_in_flight=4)
    out = c.run()
    assert [r["result"] for r in out["results"]] == ["slow", "fast", "slow", "fast"]


def test_failure_isolation():
    def boom():
        raise ValueError("nope")

    c = Shotcaller([lambda: 1, boom, lambda: 2])
    out = c.run()
    assert out["results"][0]["status"] == "ok"
    assert out["results"][1]["status"] == "failed"
    assert "ValueError" in out["results"][1]["error"]
    assert out["results"][2]["status"] == "ok"
    assert out["summary"]["failed"] == 1
    assert out["summary"]["ok"] == 2


def test_timeout_marks_job():
    def hang():
        time.sleep(5)

    c = Shotcaller([hang, lambda: "fine"], timeout_s=0.1, max_in_flight=2)
    start = time.monotonic()
    out = c.run()
    assert time.monotonic() - start < 2  # did not wait for the hang
    assert out["results"][0]["status"] == "timeout"
    assert out["results"][1]["status"] == "ok"


def test_slow_job_does_not_stall_others():
    def hang():
        time.sleep(5)

    c = Shotcaller([lambda: "a", hang, lambda: "b"], timeout_s=0.1, max_in_flight=3)
    out = c.run()
    assert out["summary"]["ok"] == 2
    assert out["summary"]["timeout"] == 1


def test_bounded_concurrency_respected():
    peak = 0
    current = 0
    lock = threading.Lock()

    def probe():
        nonlocal peak, current
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.02)
        with lock:
            current -= 1

    jobs = [probe for _ in range(20)]
    Shotcaller(jobs, max_in_flight=4).run()
    assert peak <= 4


def test_invalid_config_raises():
    with pytest.raises(ShotcallingError):
        Shotcaller([], max_in_flight=0)
    with pytest.raises(ShotcallingError):
        Shotcaller([], timeout_s=-1)


def test_invalid_job_raises():
    with pytest.raises(ShotcallingError):
        Shotcaller(["not-callable"])


def test_empty_batch():
    out = Shotcaller([]).run()
    assert out["results"] == []
    assert out["summary"]["total"] == 0
