# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end repro of the 2026-10-28 gpt-oss-120b crash: a JSON schema that
xgrammar cannot compile (empty "enum") killed the engine.

1. A valid json_schema request, so the engine latches the xgrammar backend.
2. Poison schemas via response_format, via a forced tool call, and streaming.
3. Health and a normal request afterwards: the server must still serve.

With the fix every poison request is a 400 and the server stays up; on the
old image the first poison request kills the engine.
"""

import argparse
import json
import time
import urllib.error
import urllib.request

POISON = {
    "empty enum (incident)": {
        "type": "object",
        "properties": {"color": {"type": "string", "enum": []}},
        "required": ["color"],
    },
    "missing $ref": {"type": "object", "properties": {"a": {"$ref": "#/$defs/x"}}},
    "minItems > maxItems": {"type": "array", "minItems": 3, "maxItems": 1},
    "minimum > maximum": {"type": "integer", "minimum": 5, "maximum": 1},
    "unknown type": {"type": "nonsense"},
    # Unsupported keyword routes this to guidance during validation; the engine
    # still compiles with xgrammar, so this exercises the in-engine guard.
    "uniqueItems + empty enum": {
        "type": "array",
        "uniqueItems": True,
        "items": {"enum": []},
    },
}
VALID = {
    "type": "object",
    "properties": {"color": {"type": "string", "enum": ["red", "blue"]}},
    "required": ["color"],
}


def post(base, body, timeout):
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            if body.get("stream"):
                errors = [
                    json.loads(line[6:])
                    for line in raw.splitlines()
                    if line.startswith("data: {") and '"error"' in line
                ]
                if errors:
                    err = errors[0].get("error", errors[0])
                    return (
                        err.get("code", 200),
                        err.get("message", ""),
                        time.time() - t0,
                    )
                return r.status, "stream ok", time.time() - t0
            content = json.loads(raw)["choices"][0]["message"]
            return r.status, json.dumps(content)[:120], time.time() - t0
    except urllib.error.HTTPError as e:
        data = json.loads(e.read() or b"{}")
        return e.code, data.get("error", {}).get("message", ""), time.time() - t0
    except Exception as e:
        return -1, repr(e)[:160], time.time() - t0


def health(base):
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=10) as r:
            return r.status
    except Exception:
        return -1


def chat(model, schema=None, tool_schema=None, stream=False):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Pick a color. Answer in JSON."}],
        "max_tokens": 64,
        "temperature": 0,
        "stream": stream,
    }
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        }
    if tool_schema is not None:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "set_color",
                    "description": "Set the color.",
                    "parameters": tool_schema,
                },
            }
        ]
        body["tool_choice"] = "required"
    return body


def main(args):
    m, base, ok = args.model, args.base_url, True
    steps = [("valid json_schema (latches xgrammar)", chat(m, VALID), 200)]
    for name, schema in POISON.items():
        steps.append((f"poison response_format: {name}", chat(m, schema), 400))
    steps.append(
        (
            "poison forced tool call: empty enum",
            chat(m, tool_schema=POISON["empty enum (incident)"]),
            400,
        )
    )
    steps.append(
        (
            "poison streaming: empty enum",
            chat(m, POISON["empty enum (incident)"], stream=True),
            400,
        )
    )
    steps.append(("normal request after poison", chat(m), 200))
    steps.append(("valid json_schema after poison", chat(m, VALID), 200))
    for name, body, want in steps:
        status, text, secs = post(base, body, args.timeout)
        h = health(base)
        good = status == want and h == 200 and ".cc:" not in text
        ok &= good
        print(
            f"{'PASS' if good else 'FAIL'} {name:45s} -> {status} in {secs:5.1f}s "
            f"health={h} {text[:150]!r}",
            flush=True,
        )
        if h != 200:
            print("RESULT: SERVER DOWN after this request", flush=True)
            return 1
    print("RESULT: " + ("ALL PASS" if ok else "FAILED"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--timeout", type=float, default=300)
    raise SystemExit(main(p.parse_args()))
