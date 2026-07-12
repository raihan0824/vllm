# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the --max-waiting-requests AdmissionController logic."""

from vllm.v1.engine.async_llm import AdmissionController
from vllm.v1.metrics.stats import SchedulerStats


def make_stats(running: int, waiting: int, deferred: int = 0) -> SchedulerStats:
    return SchedulerStats(
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        num_skipped_waiting_reqs=deferred,
    )


def drain(controller: AdmissionController) -> int:
    """Reserve until rejected; return how many were admitted."""
    admitted = 0
    while controller.try_reserve():
        admitted += 1
        assert admitted < 10_000, "runaway: controller never rejects"
    return admitted


def test_cold_start_assumes_max_num_seqs():
    c = AdmissionController(max_num_seqs=128, max_waiting_requests=4)
    # No stats yet: capacity = max_num_seqs.
    assert c.capacity_estimate() == 128
    assert drain(c) == 132


def test_kv_bound_engine_shrinks_capacity():
    """The dead-zone case: engine saturates at 86 running < max_num_seqs=128.

    With a backlog reported, capacity must be the actual running count, so
    the cap is 86 + 4 = 90 rather than 128 + 4 = 132.
    """
    c = AdmissionController(max_num_seqs=128, max_waiting_requests=4)
    c.update_stats(0, make_stats(running=86, waiting=2, deferred=2))
    assert c.capacity_estimate() == 86
    assert drain(c) == 90


def test_empty_queue_restores_max_num_seqs():
    c = AdmissionController(max_num_seqs=128, max_waiting_requests=4)
    c.update_stats(0, make_stats(running=86, waiting=3))
    assert c.capacity_estimate() == 86
    c.update_stats(0, make_stats(running=40, waiting=0))
    assert c.capacity_estimate() == 128


def test_running_above_max_num_seqs_is_respected():
    # If the engine somehow runs more than max_num_seqs (e.g. config drift),
    # never estimate capacity below what is observably running.
    c = AdmissionController(max_num_seqs=8, max_waiting_requests=2)
    c.update_stats(0, make_stats(running=12, waiting=0))
    assert c.capacity_estimate() == 12


def test_release_frees_slots():
    c = AdmissionController(max_num_seqs=2, max_waiting_requests=1)
    assert drain(c) == 3
    assert not c.try_reserve()
    c.release()
    assert c.try_reserve()
    assert not c.try_reserve()


def test_release_never_underflows():
    c = AdmissionController(max_num_seqs=2, max_waiting_requests=1)
    c.release()
    assert c.reserved == 0
    assert drain(c) == 3


def test_deferred_requests_count_as_backlog():
    c = AdmissionController(max_num_seqs=128, max_waiting_requests=4)
    # Only deferred (skipped-waiting) requests: still a backlog.
    c.update_stats(0, make_stats(running=100, waiting=0, deferred=2))
    assert c.capacity_estimate() == 100


def test_multi_engine_dp_sums_stats():
    c = AdmissionController(max_num_seqs=64, max_waiting_requests=4)
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
    c = AdmissionController(max_num_seqs=2, max_waiting_requests=1)
    results = [c.try_reserve() for _ in range(100)]
    assert results.count(True) == 3
    assert results.count(False) == 97
