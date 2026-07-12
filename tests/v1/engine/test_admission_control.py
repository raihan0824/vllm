# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the AdmissionController (429 backpressure logic).

Covers both gates: --max-waiting-requests (queue bound) and
--admission-max-kv-usage (KV pressure). try_reserve() returns None when
admitted, or a rejection reason string ("queue_full" / "kv_pressure").
"""

from vllm.v1.engine.async_llm import AdmissionController
from vllm.v1.metrics.stats import SchedulerStats


def make_stats(
    running: int, waiting: int, deferred: int = 0, kv_usage: float = 0.0
) -> SchedulerStats:
    return SchedulerStats(
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        num_skipped_waiting_reqs=deferred,
        kv_cache_usage=kv_usage,
    )


def make_controller(
    max_num_seqs: int,
    max_waiting_requests: int | None = None,
    max_kv_usage: float | None = None,
) -> AdmissionController:
    return AdmissionController(
        max_num_seqs=max_num_seqs,
        max_waiting_requests=max_waiting_requests,
        max_kv_usage=max_kv_usage,
    )


def drain(controller: AdmissionController) -> int:
    """Reserve until rejected; return how many were admitted."""
    admitted = 0
    while controller.try_reserve() is None:
        admitted += 1
        assert admitted < 10_000, "runaway: controller never rejects"
    return admitted


# ---------------------------------------------------------------------------
# Queue-bound gate (--max-waiting-requests)
# ---------------------------------------------------------------------------


def test_cold_start_assumes_max_num_seqs():
    c = make_controller(max_num_seqs=128, max_waiting_requests=4)
    # No stats yet: capacity = max_num_seqs.
    assert c.capacity_estimate() == 128
    assert drain(c) == 132
    assert c.try_reserve() == "queue_full"


def test_kv_bound_engine_shrinks_capacity():
    """The dead-zone case: engine saturates at 86 running < max_num_seqs=128.

    With a backlog reported, capacity must be the actual running count, so
    the cap is 86 + 4 = 90 rather than 128 + 4 = 132.
    """
    c = make_controller(max_num_seqs=128, max_waiting_requests=4)
    c.update_stats(0, make_stats(running=86, waiting=2, deferred=2))
    assert c.capacity_estimate() == 86
    assert drain(c) == 90


def test_empty_queue_restores_max_num_seqs():
    c = make_controller(max_num_seqs=128, max_waiting_requests=4)
    c.update_stats(0, make_stats(running=86, waiting=3))
    assert c.capacity_estimate() == 86
    c.update_stats(0, make_stats(running=40, waiting=0))
    assert c.capacity_estimate() == 128


def test_running_above_max_num_seqs_is_respected():
    # If the engine somehow runs more than max_num_seqs (e.g. config drift),
    # never estimate capacity below what is observably running.
    c = make_controller(max_num_seqs=8, max_waiting_requests=2)
    c.update_stats(0, make_stats(running=12, waiting=0))
    assert c.capacity_estimate() == 12


def test_release_frees_slots():
    c = make_controller(max_num_seqs=2, max_waiting_requests=1)
    assert drain(c) == 3
    assert c.try_reserve() == "queue_full"
    c.release()
    assert c.try_reserve() is None
    assert c.try_reserve() == "queue_full"


def test_release_never_underflows():
    c = make_controller(max_num_seqs=2, max_waiting_requests=1)
    c.release()
    assert c.reserved == 0
    assert drain(c) == 3


def test_deferred_requests_count_as_backlog():
    c = make_controller(max_num_seqs=128, max_waiting_requests=4)
    # Only deferred (skipped-waiting) requests: still a backlog.
    c.update_stats(0, make_stats(running=100, waiting=0, deferred=2))
    assert c.capacity_estimate() == 100


def test_multi_engine_dp_sums_stats():
    c = make_controller(max_num_seqs=64, max_waiting_requests=4)
    # Two DP engines, both backlogged: capacity = sum of running.
    c.update_stats(0, make_stats(running=50, waiting=1))
    c.update_stats(1, make_stats(running=40, waiting=2))
    assert c.capacity_estimate() == 90
    # Both drained: capacity = max_num_seqs per engine.
    c.update_stats(0, make_stats(running=10, waiting=0))
    c.update_stats(1, make_stats(running=5, waiting=0))
    assert c.capacity_estimate() == 128


def test_burst_cannot_overshoot():
    """A synchronous burst admits exactly the cap, never more."""
    c = make_controller(max_num_seqs=2, max_waiting_requests=1)
    results = [c.try_reserve() for _ in range(100)]
    assert results.count(None) == 3
    assert results.count("queue_full") == 97


# ---------------------------------------------------------------------------
# KV-pressure gate (--admission-max-kv-usage)
# ---------------------------------------------------------------------------


def test_kv_gate_rejects_at_threshold():
    c = make_controller(max_num_seqs=128, max_kv_usage=0.90)
    c.update_stats(0, make_stats(running=50, waiting=0, kv_usage=0.95))
    assert c.try_reserve() == "kv_pressure"
    # Nothing was reserved by a rejection.
    assert c.reserved == 0


def test_kv_gate_admits_below_threshold():
    c = make_controller(max_num_seqs=128, max_kv_usage=0.90)
    c.update_stats(0, make_stats(running=50, waiting=0, kv_usage=0.89))
    assert c.try_reserve() is None
    assert c.reserved == 1


def test_kv_gate_boundary_is_inclusive():
    c = make_controller(max_num_seqs=128, max_kv_usage=0.90)
    c.update_stats(0, make_stats(running=50, waiting=0, kv_usage=0.90))
    assert c.try_reserve() == "kv_pressure"


def test_kv_gate_fails_open_on_cold_start():
    # No stats yet: KV usage unknown, must admit.
    c = make_controller(max_num_seqs=128, max_kv_usage=0.90)
    assert c.try_reserve() is None


def test_kv_gate_only_mode_has_no_queue_bound():
    c = make_controller(max_num_seqs=2, max_kv_usage=0.90)
    c.update_stats(0, make_stats(running=2, waiting=50, kv_usage=0.50))
    # Queue gate disabled: any number of requests admitted below threshold.
    results = [c.try_reserve() for _ in range(500)]
    assert results.count(None) == 500


def test_kv_gate_uses_min_across_dp_engines():
    # DP router steers to the least-loaded engine, so gate on the minimum.
    c = make_controller(max_num_seqs=64, max_kv_usage=0.90)
    c.update_stats(0, make_stats(running=60, waiting=0, kv_usage=0.97))
    c.update_stats(1, make_stats(running=20, waiting=0, kv_usage=0.40))
    assert c.try_reserve() is None
    c.update_stats(1, make_stats(running=60, waiting=0, kv_usage=0.93))
    assert c.try_reserve() == "kv_pressure"


def test_combined_gates_kv_checked_first():
    c = make_controller(max_num_seqs=2, max_waiting_requests=1, max_kv_usage=0.90)
    # Saturated queue AND high KV: kv_pressure wins (leading indicator).
    c.update_stats(0, make_stats(running=2, waiting=5, kv_usage=0.95))
    assert c.try_reserve() == "kv_pressure"
    # KV drops: queue gate takes over. capacity = running (2, backlogged),
    # so with no prior reservations the cap is capacity(2) + bound(1) = 3.
    c.update_stats(0, make_stats(running=2, waiting=5, kv_usage=0.50))
    assert drain(c) == 3
    assert c.try_reserve() == "queue_full"


def test_combined_gates_recover():
    c = make_controller(max_num_seqs=4, max_waiting_requests=2, max_kv_usage=0.90)
    c.update_stats(0, make_stats(running=4, waiting=0, kv_usage=0.95))
    assert c.try_reserve() == "kv_pressure"
    c.update_stats(0, make_stats(running=4, waiting=0, kv_usage=0.70))
    assert c.try_reserve() is None
