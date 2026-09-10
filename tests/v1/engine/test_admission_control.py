# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the AdmissionController (429 backpressure logic).

Covers both gates: --max-waiting-requests (queue bound) and
--admission-max-kv-usage (KV pressure). try_reserve() returns None when
admitted, or a rejection reason string ("queue_full" / "kv_pressure").
"""

import asyncio
import contextlib

import pytest

from vllm.entrypoints.serve.utils.api_utils import (
    ADMISSION_SLOT_SCOPE_KEY,
    AdmissionSlot,
    AdmissionSlotMiddleware,
)
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
    max_prompt_tokens: int | None = None,
) -> AdmissionController:
    return AdmissionController(
        max_num_seqs=max_num_seqs,
        max_waiting_requests=max_waiting_requests,
        max_kv_usage=max_kv_usage,
        max_prompt_tokens=max_prompt_tokens,
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


# ---------------------------------------------------------------------------
# Long-prompt gate (--admission-max-prompt-tokens)
# ---------------------------------------------------------------------------


def fill(controller: AdmissionController, n: int) -> None:
    for _ in range(n):
        assert controller.try_reserve() is None


def test_long_prompt_rejected_when_busy():
    c = make_controller(max_num_seqs=8, max_prompt_tokens=25000)
    fill(c, 4)  # reserved=4, capacity=8 -> busy (>= half)
    assert c.is_busy()
    assert c.should_reject_long_prompt(25001)
    assert not c.should_reject_long_prompt(25000)  # boundary: <= threshold OK


def test_long_prompt_served_when_idle():
    c = make_controller(max_num_seqs=8, max_prompt_tokens=25000)
    fill(c, 3)  # reserved=3, capacity=8 -> not busy
    assert not c.is_busy()
    assert not c.should_reject_long_prompt(100_000)


def test_long_prompt_gate_disabled_by_default():
    c = make_controller(max_num_seqs=8, max_waiting_requests=2)
    fill(c, 8)
    assert not c.should_reject_long_prompt(1_000_000)


def test_long_prompt_busy_tracks_capacity_estimate():
    # Engine backlogged at 4 running (capacity=4): busy from reserved>=2.
    c = make_controller(max_num_seqs=64, max_prompt_tokens=25000)
    c.update_stats(0, make_stats(running=4, waiting=3))
    fill(c, 2)
    assert c.is_busy()
    assert c.should_reject_long_prompt(30_000)


def test_long_prompt_gate_recovers_after_release():
    c = make_controller(max_num_seqs=8, max_prompt_tokens=25000)
    fill(c, 4)
    assert c.should_reject_long_prompt(30_000)
    c.release()  # reserved=3 -> not busy
    assert not c.should_reject_long_prompt(30_000)


def test_long_prompt_gate_alone_enables_controller_reserve_flow():
    # Only the prompt gate configured: reserve/release must still work
    # (no queue bound -> never queue_full), and busy state derives from it.
    c = make_controller(max_num_seqs=4, max_prompt_tokens=25000)
    results = [c.try_reserve() for _ in range(10)]
    assert results.count(None) == 10  # no queue gate -> all admitted
    assert c.is_busy()
    assert c.should_reject_long_prompt(25001)


# ---------------------------------------------------------------------------
# Slot lifetime (AdmissionSlot / AdmissionSlotMiddleware)
#
# A slot outlives its route handler, so it cannot be released by the handler.
# Releasing it from the response's background task is not enough either:
# Starlette skips background tasks when a send() fails, so a client aborting
# mid-stream would leak the slot. Leaks are unrecoverable -- `reserved` only
# ratchets up until every request is rejected while the engine sits idle.
# ---------------------------------------------------------------------------


async def _noop_receive() -> dict:
    return {"type": "http.request"}


async def _noop_send(message) -> None:
    pass


def _reserving_app(controller: AdmissionController, body):
    """Downstream ASGI app that reserves a slot, then runs `body`."""

    async def app(scope, receive, send) -> None:
        assert controller.try_reserve() is None
        scope[ADMISSION_SLOT_SCOPE_KEY] = AdmissionSlot(controller.release)
        await body(scope, receive, send)

    return app


def _call(app) -> None:
    asyncio.run(app({"type": "http"}, _noop_receive, _noop_send))


def test_slot_releases_exactly_once():
    c = make_controller(max_num_seqs=8, max_waiting_requests=2)
    assert c.try_reserve() is None
    slot = AdmissionSlot(c.release)

    slot.release()
    slot.release()

    assert c.reserved == 0


def test_middleware_releases_slot_after_response():
    c = make_controller(max_num_seqs=8, max_waiting_requests=2)

    async def respond(scope, receive, send) -> None:
        assert c.reserved == 1

    _call(AdmissionSlotMiddleware(_reserving_app(c, respond)))
    assert c.reserved == 0


def test_middleware_releases_slot_when_client_aborts_stream():
    """The leak that bricked a production replica: uvicorn raises when writing
    to a socket the client already closed, so the response's background task
    never runs."""
    c = make_controller(max_num_seqs=8, max_waiting_requests=2)

    async def abort(scope, receive, send) -> None:
        raise OSError("client disconnected")

    with pytest.raises(OSError):
        _call(AdmissionSlotMiddleware(_reserving_app(c, abort)))

    assert c.reserved == 0


def test_middleware_releases_slot_on_cancellation():
    c = make_controller(max_num_seqs=8, max_waiting_requests=2)

    async def cancel(scope, receive, send) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        _call(AdmissionSlotMiddleware(_reserving_app(c, cancel)))

    assert c.reserved == 0


def test_aborted_streams_do_not_ratchet_capacity():
    """Many aborted streams in a row must not exhaust admission capacity."""
    c = make_controller(max_num_seqs=4, max_waiting_requests=2)

    async def abort(scope, receive, send) -> None:
        raise OSError("client disconnected")

    app = AdmissionSlotMiddleware(_reserving_app(c, abort))
    for _ in range(100):
        with pytest.raises(OSError):
            _call(app)

    assert c.reserved == 0
    assert c.try_reserve() is None


def test_middleware_ignores_requests_without_a_slot():
    c = make_controller(max_num_seqs=8, max_waiting_requests=2)
    seen = []

    async def app(scope, receive, send) -> None:
        seen.append(scope["type"])

    _call(AdmissionSlotMiddleware(app))
    asyncio.run(
        AdmissionSlotMiddleware(app)({"type": "lifespan"}, _noop_receive, _noop_send)
    )

    assert seen == ["http", "lifespan"]
    assert c.reserved == 0


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_slot_released_when_client_aborts_a_real_streaming_response(spec_version):
    """End-to-end through FastAPI/Starlette, the shape that bricked a replica.

    Starlette picks a different StreamingResponse code path per ASGI spec
    version, and neither reaches the response's background tasks once send()
    fails, so both are exercised here.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    from vllm.entrypoints.serve.utils.api_utils import load_aware_call

    controller = make_controller(max_num_seqs=1000, max_waiting_requests=1000)

    class FakeEngineClient:
        def admission_control_enabled(self) -> bool:
            return True

        def try_reserve_request_slot(self) -> str | None:
            return controller.try_reserve()

        def release_request_slot(self) -> None:
            controller.release()

        def record_request_rejected(self, reason: str) -> None:
            pass

    app = FastAPI()
    app.state.engine_client = FakeEngineClient()
    app.state.enable_server_load_tracking = False

    @app.post("/v1/chat/completions")
    @load_aware_call
    async def handler(raw_request: Request):
        async def stream():
            for i in range(10):
                yield f"data: {i}\n\n".encode()

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def send_until_socket_dies(message) -> None:
        # uvicorn raises when writing to a socket the peer already closed.
        if message["type"] == "http.response.body":
            raise OSError(104, "Connection reset by peer")

    def receive_body_then_hang():
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"{}", "more_body": False}
            # The client never sends http.disconnect; its socket just dies.
            await asyncio.Event().wait()

        return receive

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"test"), (b"content-type", b"application/json")],
        "client": ("10.0.0.1", 1234),
        "server": ("test", 80),
        "extensions": {},
    }

    async def abort_one() -> None:
        with contextlib.suppress(BaseException):
            await AdmissionSlotMiddleware(app)(
                dict(scope), receive_body_then_hang(), send_until_socket_dies
            )

    async def abort_many() -> None:
        for _ in range(20):
            await abort_one()

    asyncio.run(abort_many())

    assert controller.reserved == 0
