# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay of the 2026-11-09 01:43 WIB Gemma-4-26B illegal memory access: a
chunked prefill of a request with 51+ images, next to a one-image request and
text decodes, each new image count forcing a new Triton specialization.

Sends one request per image count in the ladder, with background traffic, and
checks /health after each. Stops at the first dead server.
"""

import argparse
import asyncio
import base64
import io
import random
import time

import aiohttp
from PIL import Image


def make_image(rng: random.Random, width: int, height: int) -> str:
    # Noise blocks so every image has its own hash (no MM cache hits).
    tile = Image.effect_noise((width // 8 + 1, height // 8 + 1), rng.uniform(40, 90))
    tint = Image.new("RGB", tile.size, tuple(rng.randrange(256) for _ in range(3)))
    img = Image.blend(tint, tile.convert("RGB"), 0.6).resize((width, height))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


SIZES = ((640, 480), (1024, 768), (480, 640), (800, 800), (1280, 720), (333, 777))


def image_body(model: str, n: int, rng: random.Random, max_tokens: int) -> dict:
    content: list[dict] = []
    for i in range(n):
        width, height = rng.choice(SIZES)
        content.append(
            {"type": "image_url", "image_url": {"url": make_image(rng, width, height)}}
        )
        content.append({"type": "text", "text": f"Screenshot {i}."})
    content.append({"type": "text", "text": "Describe the last screenshot."})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }


async def post(session, url: str, body: dict) -> tuple[int, str]:
    try:
        async with session.post(url, json=body) as r:
            return r.status, (await r.text())[:300]
    except Exception as e:
        return -1, repr(e)[:300]


async def healthy(session, base: str) -> bool:
    try:
        async with session.get(f"{base}/health") as r:
            return r.status == 200
    except Exception:
        return False


async def main(args) -> int:
    url = f"{args.base_url}/v1/chat/completions"
    rng = random.Random(args.seed)
    stop = asyncio.Event()
    background = {"ok": 0, "bad": 0}
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:

        async def text_worker(w: int):
            local = random.Random(args.seed + w)
            while not stop.is_set():
                body = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Count from {local.randint(1, 999)} upward.",
                        }
                    ],
                    "max_tokens": local.choice((64, 128, 256)),
                    "temperature": 0,
                }
                status, _ = await post(session, url, body)
                background["ok" if status == 200 else "bad"] += 1
                if status != 200:
                    await asyncio.sleep(1)

        async def one_image_worker():
            local = random.Random(args.seed + 100)
            while not stop.is_set():
                status, _ = await post(
                    session, url, image_body(args.model, 1, local, 16)
                )
                background["ok" if status == 200 else "bad"] += 1
                if status != 200:
                    await asyncio.sleep(1)

        tasks = [asyncio.create_task(text_worker(w)) for w in range(args.text_workers)]
        tasks.append(asyncio.create_task(one_image_worker()))
        code = 0
        try:
            for n in args.ladder:
                for rep in range(args.repeats):
                    body = await asyncio.to_thread(
                        image_body, args.model, n, rng, args.max_tokens
                    )
                    t0 = time.time()
                    status, text = await post(session, url, body)
                    secs = time.time() - t0
                    up = await healthy(session, args.base_url)
                    print(
                        f"images={n:4d} rep={rep} -> {status} in {secs:7.1f}s "
                        f"health={up} background={background} {text[:160]!r}",
                        flush=True,
                    )
                    if not up:
                        print(f"RESULT: SERVER DOWN at images={n}", flush=True)
                        code = 1
                        return code
                    if status != 200:
                        code = 2
            print("RESULT: survived ladder" if code == 0 else "RESULT: non-200 seen",
                  flush=True)
            return code
        finally:
            stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--ladder", type=int, nargs="+", default=[1, 8, 16, 32, 51, 64, 100]
    )
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--text-workers", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--seed", type=int, default=7)
    raise SystemExit(asyncio.run(main(p.parse_args())))
