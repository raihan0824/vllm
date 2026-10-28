# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed-traffic smoke under load: plain, streamed, tool-call and json_schema
chat requests from concurrent workers, plus a poison schema (empty "enum")
every ~30 s. Passes if no request fails server-side (5xx / dropped), /health
stays 200, and every poison request gets a 400.
"""

import argparse
import asyncio
import json
import random
import time

import aiohttp

POISON = {"type": "object", "properties": {"c": {"type": "string", "enum": []}}}
VALID = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "temp_c": {"type": "integer"}},
    "required": ["city", "temp_c"],
}
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def body(kind: str, model: str, rng: random.Random) -> dict:
    b = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"Weather in city #{rng.randint(1, 999)}?"}
        ],
        "max_tokens": rng.choice((32, 128, 256)),
        "temperature": 0.7,
    }
    if kind == "stream":
        b["stream"] = True
    elif kind == "tools":
        b["tools"] = [TOOL]
    elif kind == "json":
        b["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "w", "schema": VALID},
        }
    elif kind == "poison":
        b["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "p", "schema": POISON},
        }
    return b


async def send(session, url, b):
    try:
        async with session.post(url, json=b) as r:
            text = await r.text()
            if r.status == 200 and b.get("stream") and '"error"' in text:
                return 599, text[:200]
            return r.status, text[:200]
    except Exception as e:
        return -1, repr(e)[:200]


async def main(args):
    url = f"{args.base_url}/v1/chat/completions"
    stop = time.time() + args.seconds
    counts: dict[str, dict[int, int]] = {}
    bad: list[str] = []
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def worker(w):
            rng = random.Random(w)
            while time.time() < stop:
                kind = rng.choice(("plain", "stream", "tools", "json"))
                status, text = await send(session, url, body(kind, args.model, rng))
                counts.setdefault(kind, {}).setdefault(status, 0)
                counts[kind][status] += 1
                if status >= 500 or status < 0:
                    bad.append(f"{kind} {status} {text}")

        async def poisoner():
            rng = random.Random(0)
            while time.time() < stop:
                status, text = await send(session, url, body("poison", args.model, rng))
                counts.setdefault("poison", {}).setdefault(status, 0)
                counts["poison"][status] += 1
                if status != 400:
                    bad.append(f"poison {status} {text}")
                await asyncio.sleep(30)

        async def health():
            while time.time() < stop:
                try:
                    async with session.get(f"{args.base_url}/health") as r:
                        if r.status != 200:
                            bad.append(f"health {r.status}")
                except Exception as e:
                    bad.append(f"health {e!r}"[:200])
                await asyncio.sleep(10)

        await asyncio.gather(
            *(worker(w) for w in range(args.concurrency)), poisoner(), health()
        )
    print(json.dumps(counts, sort_keys=True), flush=True)
    for line in bad[:10]:
        print("BAD", line, flush=True)
    # 429 is the front door's backpressure, not a failure.
    print("RESULT: " + ("PASS" if not bad else f"FAILED ({len(bad)} bad)"), flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument("--concurrency", type=int, default=24)
    raise SystemExit(asyncio.run(main(p.parse_args())))
