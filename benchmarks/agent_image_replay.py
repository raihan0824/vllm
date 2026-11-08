# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Second replay for the 2026-11-09 Gemma-4-26B illegal memory access.

shapes: odd image shapes, alone and as 51 copies with distinct noise.
agent:  growing multi-turn sessions that resend every earlier image, for prefix
        and MM cache hits, chunked prefill past 15k tokens, and block churn,
        next to one-image requests and long decodes.
"""

import argparse
import asyncio
import base64
import io
import random
import time

import aiohttp
from PIL import Image

ODD_SHAPES = (
    (1, 1), (2, 2), (3, 5), (16, 16), (17, 33), (64, 63), (1, 4000), (4000, 1),
    (10, 8000), (8000, 10), (48, 4096), (4096, 48), (5000, 5000), (8000, 6000),
    (1920, 1080), (1080, 1920), (1366, 768), (2559, 1439),
)  # fmt: skip


def data_url(rng: random.Random, width: int, height: int, fmt: str = "JPEG") -> str:
    tile = Image.effect_noise(
        (max(1, width // 8), max(1, height // 8)), rng.uniform(30, 90)
    )
    tint = Image.new("RGB", tile.size, tuple(rng.randrange(256) for _ in range(3)))
    img = Image.blend(tint, tile.convert("RGB"), 0.6).resize((width, height))
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=70) if fmt == "JPEG" else img.save(buf, fmt)
    mime = fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


async def post(session, url, body):
    try:
        async with session.post(url, json=body) as r:
            return r.status, (await r.text())
    except Exception as e:
        return -1, repr(e)


async def healthy(session, base):
    try:
        async with session.get(f"{base}/health") as r:
            return r.status == 200
    except Exception:
        return False


async def run_shapes(session, args, rng) -> bool:
    url = f"{args.base_url}/v1/chat/completions"
    for w, h in ODD_SHAPES:
        for n in (1, 51):
            urls = await asyncio.to_thread(
                lambda n=n: [data_url(rng, w, h) for _ in range(n)]
            )
            content = [part(u) for u in urls] + [{"type": "text", "text": "Describe."}]
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 16,
                "temperature": 0,
            }
            t0 = time.time()
            status, text = await post(session, url, body)
            up = await healthy(session, args.base_url)
            print(
                f"shape {w}x{h} n={n} -> {status} in {time.time() - t0:6.1f}s "
                f"health={up} {text[:110] if status != 200 else ''!r}",
                flush=True,
            )
            if not up:
                print(f"RESULT: SERVER DOWN at shape {w}x{h} n={n}", flush=True)
                return False
    return True


async def run_agent(session, args, rng, deadline, s: int, state: dict) -> None:
    url = f"{args.base_url}/v1/chat/completions"
    local = random.Random(args.seed + s)
    while time.time() < deadline and not state["down"]:
        # Unique system text, of a random length, so seq_len lands on every block
        # offset and the sessions do not share a prefix with each other.
        system = f"Session {s}-{local.random()}. " + "Follow the steps. " * local.randint(
            1, 300
        )
        msgs = [{"role": "system", "content": system}]
        shots = [
            await asyncio.to_thread(
                data_url, local, *local.choice(((1366, 768), (1280, 800), (800, 600)))
            )
            for _ in range(4)
        ]
        images = 0
        while images < args.max_images and time.time() < deadline:
            k = local.randint(1, 8)
            content = []
            for _ in range(k):
                # A few screenshots repeat: MM cache hit inside a new turn.
                if local.random() < 0.15:
                    u = local.choice(shots)
                else:
                    u = await asyncio.to_thread(
                        data_url,
                        local,
                        *local.choice(((1366, 768), (1280, 800), (800, 600), (640, 960))),
                    )
                    shots.append(u)
                content.append(part(u))
                images += 1
            content.append({"type": "text", "text": f"Step {images}: what changed?"})
            msgs.append({"role": "user", "content": content})
            body = {
                "model": args.model,
                "messages": msgs,
                "max_tokens": local.choice((16, 64, 128)),
                "temperature": 0,
            }
            t0 = time.time()
            status, text = await post(session, url, body)
            state["requests"] += 1
            state["max_images"] = max(state["max_images"], images)
            if status != 200:
                state["bad"] += 1
                state["last_bad"] = f"s{s} images={images} {status} {text[:160]}"
                if status != 429:
                    await asyncio.sleep(1)
                break
            state["slowest"] = max(state["slowest"], time.time() - t0)
            msgs.append({"role": "assistant", "content": "Noted."})


async def background(session, args, deadline, state):
    url = f"{args.base_url}/v1/chat/completions"
    local = random.Random(args.seed + 500)

    async def text():
        while time.time() < deadline and not state["down"]:
            body = {
                "model": args.model,
                "messages": [
                    {"role": "user", "content": f"Count from {local.randint(1, 999)}."}
                ],
                "max_tokens": local.choice((64, 256)),
                "temperature": 0,
            }
            status, _ = await post(session, url, body)
            if status != 200:
                await asyncio.sleep(0.2)

    async def one_image():
        while time.time() < deadline and not state["down"]:
            u = await asyncio.to_thread(data_url, local, 800, 600)
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": [part(u)]}],
                "max_tokens": 16,
                "temperature": 0,
            }
            status, _ = await post(session, url, body)
            if status != 200:
                await asyncio.sleep(0.2)

    await asyncio.gather(*(text() for _ in range(2)), one_image())


async def main(args) -> int:
    rng = random.Random(args.seed)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        if args.mode == "shapes":
            ok = await run_shapes(session, args, rng)
            print("RESULT: survived shapes" if ok else "RESULT: down", flush=True)
            return 0 if ok else 1
        deadline = time.time() + args.seconds
        state = dict(
            requests=0, bad=0, last_bad="", max_images=0, slowest=0.0, down=False
        )

        async def watch():
            while time.time() < deadline:
                await asyncio.sleep(30)
                up = await healthy(session, args.base_url)
                print(f"{time.strftime('%T')} health={up} {state}", flush=True)
                if not up:
                    state["down"] = True
                    return

        await asyncio.gather(
            watch(),
            background(session, args, deadline, state),
            *(
                run_agent(session, args, rng, deadline, s, state)
                for s in range(args.sessions)
            ),
        )
        print(
            "RESULT: SERVER DOWN" if state["down"] else "RESULT: survived agent soak",
            state,
            flush=True,
        )
        return 1 if state["down"] else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("shapes", "agent"), required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seconds", type=int, default=1200)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--max-images", type=int, default=70)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--seed", type=int, default=11)
    raise SystemExit(asyncio.run(main(p.parse_args())))
