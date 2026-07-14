# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for --max-waiting-requests (queue-depth backpressure).

When the engine's waiting queue is full, new requests are rejected before any
tokenization/prefill work with an OpenAI-shaped HTTP 429 (+ Retry-After) instead
of queueing unboundedly. Mirrors SGLang's test_request_queue_validation.py, but
asserts 429 (TGI semantics) rather than SGLang's 503.

These are GPU integration tests: they start a real server via
``RemoteOpenAIServer`` and require the test model to load.
"""

import asyncio
import json

import httpx
import pytest

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"

MAX_NUM_SEQS = 2
MAX_WAITING_REQUESTS = 1
# The frontend admits at most capacity_estimate + MAX_WAITING_REQUESTS
# in-flight requests. The capacity estimate adapts to the engine's actual
# sustained running count; for this tiny model KV cache is plentiful, so the
# engine saturates exactly at MAX_NUM_SEQS and the admission cap is stable at:
CAPACITY = MAX_NUM_SEQS + MAX_WAITING_REQUESTS

# Long enough (with ignore_eos) that an accepted request stays in-flight for a
# few seconds, so a concurrent burst reliably observes a full queue.
SLOW_MAX_TOKENS = 1900

BASE_ARGS = [
    "--dtype",
    "bfloat16",
    "--enforce-eager",
    "--max-model-len",
    "2048",
    "--gpu-memory-utilization",
    "0.3",
    "--max-num-seqs",
    str(MAX_NUM_SEQS),
]

CHAT_URL = "/v1/chat/completions"


@pytest.fixture(scope="module")
def bounded_server():
    args = BASE_ARGS + ["--max-waiting-requests", str(MAX_WAITING_REQUESTS)]
    with RemoteOpenAIServer(MODEL_NAME, args) as server:
        yield server


@pytest.fixture(scope="module")
def unbounded_server():
    with RemoteOpenAIServer(MODEL_NAME, BASE_ARGS) as server:
        yield server


def chat_payload(max_tokens: int, stream: bool = False) -> dict:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Tell me a very long story."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        # vLLM extension: keep generating up to max_tokens so the request stays
        # in-flight and occupies a slot for the duration of the test.
        "ignore_eos": True,
    }


def make_client(server: RemoteOpenAIServer, n_conns: int) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=server.url_root,
        timeout=httpx.Timeout(120.0),
        limits=httpx.Limits(max_connections=n_conns, max_keepalive_connections=n_conns),
    )


def assert_overloaded_body(response: httpx.Response) -> None:
    """Validate the OpenAI-shaped 429 error body + Retry-After header."""
    assert response.headers.get("Retry-After") == "2"
    body = response.json()
    assert set(body.keys()) == {"error"}
    err = body["error"]
    assert err["type"] == "rate_limit_error"
    assert err["code"] == 429
    assert isinstance(err["message"], str) and err["message"]


def parse_counter(metrics_text: str, name: str, **labels) -> float:
    """Sum samples of a Prometheus counter matching all given label=value."""
    total = 0.0
    prefix = f"{name}{{"
    for line in metrics_text.splitlines():
        if not line.startswith(prefix):
            continue
        label_blob, _, value = line.rpartition("}")
        label_str = label_blob[len(name) + 1 :]
        if all(f'{k}="{v}"' in label_str for k, v in labels.items()):
            total += float(value)
    return total


async def fetch_rejected_total(client: httpx.AsyncClient) -> float:
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    return parse_counter(
        resp.text, "vllm:num_requests_rejected_total", reason="queue_full"
    )


@pytest.mark.asyncio
async def test_serial_requests_all_succeed(bounded_server):
    """Test 1: serial requests never trip the bound -> all 200."""
    async with make_client(bounded_server, n_conns=4) as client:
        for _ in range(6):
            resp = await client.post(CHAT_URL, json=chat_payload(max_tokens=8))
            assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_burst_rejects_excess_with_429(bounded_server):
    """Test 2: N >> capacity concurrent slow requests.

    At most `capacity` get 200; the rest get a well-formed 429; nothing hangs.
    """
    n = 24
    payload = chat_payload(max_tokens=SLOW_MAX_TOKENS)
    async with make_client(bounded_server, n_conns=n + 4) as client:
        results = await asyncio.gather(
            *(client.post(CHAT_URL, json=payload) for _ in range(n)),
            return_exceptions=True,
        )

    # No request hung or errored at the transport layer.
    assert all(isinstance(r, httpx.Response) for r in results), results
    codes = [r.status_code for r in results]
    accepted = [r for r in results if r.status_code == 200]
    rejected = [r for r in results if r.status_code == 429]

    assert len(accepted) + len(rejected) == n, codes
    # Hard upper bound: the frontend must never admit more than capacity.
    assert len(accepted) <= CAPACITY, codes
    # With N >> capacity slow requests, excess must be shed.
    assert len(rejected) >= n - CAPACITY, codes
    for resp in rejected:
        assert_overloaded_body(resp)


@pytest.mark.asyncio
async def test_streaming_rejected_before_any_sse_bytes(bounded_server):
    """Test 3: a rejected streaming request gets 429 before any SSE bytes."""
    async with make_client(bounded_server, n_conns=CAPACITY + 8) as client:
        # Saturate all slots with slow in-flight requests.
        occupiers = [
            asyncio.create_task(
                client.post(CHAT_URL, json=chat_payload(max_tokens=SLOW_MAX_TOKENS))
            )
            for _ in range(CAPACITY + 4)
        ]
        try:
            # Give the occupiers time to be admitted (reserve their slots).
            await asyncio.sleep(1.5)

            async with client.stream(
                "POST", CHAT_URL, json=chat_payload(max_tokens=64, stream=True)
            ) as resp:
                assert resp.status_code == 429
                assert resp.headers.get("Retry-After") == "2"
                body = await resp.aread()
                # The body is a single JSON error, NOT an SSE `data:` frame.
                assert not body.lstrip().startswith(b"data:")
                err = json.loads(body)["error"]
                assert err["type"] == "rate_limit_error"
                assert err["code"] == 429
        finally:
            for task in occupiers:
                task.cancel()
            await asyncio.gather(*occupiers, return_exceptions=True)


@pytest.mark.asyncio
async def test_rejected_requests_do_zero_prefill(bounded_server):
    """Test 4: rejected requests never reach the engine.

    Rejection happens before add_request(), so each 429 increments
    vllm:num_requests_rejected_total and does zero prefill work. We assert the
    counter delta equals exactly the number of observed 429s.
    """
    n = 24
    payload = chat_payload(max_tokens=SLOW_MAX_TOKENS)
    async with make_client(bounded_server, n_conns=n + 4) as client:
        before = await fetch_rejected_total(client)
        results = await asyncio.gather(
            *(client.post(CHAT_URL, json=payload) for _ in range(n)),
            return_exceptions=True,
        )
        after = await fetch_rejected_total(client)

    num_429 = sum(
        1 for r in results if isinstance(r, httpx.Response) and r.status_code == 429
    )
    assert num_429 > 0
    assert after - before == num_429


@pytest.fixture(scope="module")
def long_prompt_gated_server():
    """Server with only the long-prompt gate (--admission-max-prompt-tokens)."""
    args = BASE_ARGS + ["--admission-max-prompt-tokens", "64"]
    with RemoteOpenAIServer(MODEL_NAME, args) as server:
        yield server


def long_prompt_payload(n_words: int, max_tokens: int, stream: bool = False) -> dict:
    payload = chat_payload(max_tokens=max_tokens, stream=stream)
    payload["messages"] = [{"role": "user", "content": "word " * n_words}]
    return payload


@pytest.mark.asyncio
async def test_long_prompt_served_when_idle(long_prompt_gated_server):
    """A prompt over the threshold is served normally on an idle engine."""
    async with make_client(long_prompt_gated_server, n_conns=2) as client:
        resp = await client.post(
            CHAT_URL, json=long_prompt_payload(n_words=200, max_tokens=8)
        )
        assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_long_prompt_rejected_when_busy(long_prompt_gated_server):
    """Over-threshold prompts get 429 while the engine is busy; short ones pass."""
    async with make_client(long_prompt_gated_server, n_conns=8) as client:
        # Saturate the engine (MAX_NUM_SEQS slots) with slow short requests.
        occupiers = [
            asyncio.create_task(
                client.post(CHAT_URL, json=chat_payload(max_tokens=SLOW_MAX_TOKENS))
            )
            for _ in range(MAX_NUM_SEQS)
        ]
        try:
            await asyncio.sleep(1.5)

            # Long prompt while busy -> 429 rate_limit_error, no SSE bytes.
            resp = await client.post(
                CHAT_URL, json=long_prompt_payload(n_words=200, max_tokens=8)
            )
            assert resp.status_code == 429, resp.text
            err = resp.json()["error"]
            assert err["type"] == "rate_limit_error"
            assert err["code"] == 429

            # Short prompt while busy -> still admitted (no queue gate here).
            resp = await client.post(CHAT_URL, json=chat_payload(max_tokens=8))
            assert resp.status_code == 200, resp.text
        finally:
            for task in occupiers:
                task.cancel()
            await asyncio.gather(*occupiers, return_exceptions=True)


@pytest.mark.asyncio
async def test_default_unbounded_is_unchanged(unbounded_server):
    """Test 5: regression. Without the flag, a burst produces zero 429s."""
    n = 24
    payload = chat_payload(max_tokens=16)
    async with make_client(unbounded_server, n_conns=n + 4) as client:
        results = await asyncio.gather(
            *(client.post(CHAT_URL, json=payload) for _ in range(n)),
            return_exceptions=True,
        )

    assert all(isinstance(r, httpx.Response) for r in results), results
    codes = [r.status_code for r in results]
    assert all(code == 200 for code in codes), codes
    # The rejection metric must not even be emitted when the flag is unset.
    async with make_client(unbounded_server, n_conns=2) as client:
        metrics = (await client.get("/metrics")).text
    assert "vllm:num_requests_rejected_total" not in metrics
