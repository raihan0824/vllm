# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a burst of concurrent chat requests that use sampling penalties.

Reproduces the 2026-10-27 Gemma-4-26B crash: with spec decode every request
contributes one rejection-sampler row per draft token, and the dense
(rows x vocab) penalty tables OOMed the engine. Requests are sent in waves of
increasing concurrency; after each wave the server's /health is checked.

Example:
    python penalty_burst_replay.py --base-url http://host:8000 \
        --model google/gemma-4-26B-A4B-it --waves 32,64,128,256
"""

import argparse
import asyncio
import json
import random
import time

import aiohttp

WORDS = [
    "river",
    "mountain",
    "signal",
    "harbor",
    "lantern",
    "copper",
    "orchard",
    "meadow",
    "quartz",
    "falcon",
    "ember",
    "canyon",
    "velvet",
    "glacier",
    "thunder",
    "willow",
    "prism",
    "cobalt",
    "summit",
    "hollow",
]


def make_prompt(rng: random.Random, approx_words: int) -> str:
    body = " ".join(
        rng.choice(WORDS) + str(rng.randint(0, 999)) for _ in range(approx_words)
    )
    return (
        "Here are some notes:\n"
        + body
        + "\n\nWrite a very long, detailed essay that continues these notes."
    )


async def one_request(session, args, rng, idx, use_penalties):
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": make_prompt(rng, rng.randint(*args.prompt_words)),
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.8,
        "stream": True,
        # Keep every request decoding until max_tokens.
        "ignore_eos": True,
    }
    if use_penalties:
        payload.update(
            repetition_penalty=args.repetition_penalty,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
        )
    start = time.perf_counter()
    try:
        async with session.post(
            f"{args.base_url}/v1/chat/completions", json=payload
        ) as resp:
            if resp.status != 200:
                return resp.status, 0, time.perf_counter() - start, await resp.text()
            chunks = 0
            async for line in resp.content:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    data = json.loads(line[6:])
                    if "error" in data:
                        return 500, chunks, time.perf_counter() - start, str(data)
                    chunks += 1
            return 200, chunks, time.perf_counter() - start, ""
    except Exception as e:  # connection reset when the engine dies
        return -1, 0, time.perf_counter() - start, repr(e)


async def health(session, base_url) -> int:
    try:
        async with session.get(
            f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)
        ) as r:
            return r.status
    except Exception:
        return -1


async def main(args):
    rng = random.Random(args.seed)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    conn = aiohttp.TCPConnector(limit=0)
    summary = []
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        # A crash only counts if the server was healthy before the first wave.
        if await health(session, args.base_url) != 200:
            print("RESULT: NOT READY (server unhealthy before the first wave)")
            return 3
        for wave in args.waves:
            n_pen = int(round(wave * args.penalty_fraction))
            flags = [True] * n_pen + [False] * (wave - n_pen)
            rng.shuffle(flags)
            t0 = time.perf_counter()
            results = await asyncio.gather(
                *(one_request(session, args, rng, i, f) for i, f in enumerate(flags))
            )
            elapsed = time.perf_counter() - t0
            codes = {}
            for code, _, _, _ in results:
                codes[code] = codes.get(code, 0) + 1
            chunks = sum(r[1] for r in results)
            errors = [r[3] for r in results if r[0] != 200][:3]
            h = await health(session, args.base_url)
            line = {
                "wave": wave,
                "penalty_reqs": n_pen,
                "codes": codes,
                "stream_chunks": chunks,
                "elapsed_s": round(elapsed, 1),
                "health_after": h,
                "sample_errors": [e[:300] for e in errors],
            }
            summary.append(line)
            print(json.dumps(line), flush=True)
            if h != 200:
                print(
                    f"RESULT: CRASHED (server unhealthy after wave {wave})",
                    flush=True,
                )
                return 1
            if any(c != 200 for c in codes):
                print(f"RESULT: ERRORS in wave {wave}", flush=True)
                return 2
            await asyncio.sleep(args.pause)
    print("RESULT: SURVIVED all waves", flush=True)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--waves",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[32, 64, 128, 256],
    )
    p.add_argument("--penalty-fraction", type=float, default=1.0)
    p.add_argument(
        "--prompt-words",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=(200, 1500),
    )
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--frequency-penalty", type=float, default=0.3)
    p.add_argument("--presence-penalty", type=float, default=0.3)
    p.add_argument("--pause", type=float, default=5.0)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=0)
    raise SystemExit(asyncio.run(main(p.parse_args())))
