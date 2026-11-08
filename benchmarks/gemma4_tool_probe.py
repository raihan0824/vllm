# Probe Gemma-4 tool calls per schema shape: is the call conformant, cut, or empty?
import json, sys, urllib.request, urllib.error
import jsonschema
base, model = sys.argv[1], sys.argv[2]
CALL = {"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}
CALLS = {"type": "object", "properties": {"calls": {"type": "array", "items": CALL}}, "required": ["calls"]}
CITY = {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city", "days"]}
SHAPES = {
  "plain": (CITY, ["city", "days"]),
  "root $ref": ({"$ref": "#/$defs/W", "$defs": {"W": CITY}}, ["city", "days"]),
  "root allOf": ({"allOf": [CITY]}, ["city", "days"]),
  "root anyOf": ({"anyOf": [CITY, {"type": "object", "properties": {"zip": {"type": "string"}}, "required": ["zip"]}]}, None),
  "root oneOf": ({"oneOf": [CITY, {"type": "object", "properties": {"zip": {"type": "string"}}, "required": ["zip"]}]}, None),
  "typed+allOf": ({"type": "object", "allOf": [CITY]}, ["city", "days"]),
  "no root type": ({"properties": CITY["properties"], "required": ["city", "days"]}, ["city", "days"]),
  "optional anyOf": ({"type": "object", "properties": {"city": {"type": "string"}, "days": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}, "required": ["city"]}, ["city"]),
  "nested $ref": ({"type": "object", "properties": {"q": {"$ref": "#/$defs/W"}}, "required": ["q"], "$defs": {"W": CITY}}, ["q"]),
  "wrapper calls": (CALLS, ["calls"]),
}
PROMPTS = ["What is the weather in Jakarta for the next 3 days? Use the tool.", "Forecast for Paris, 5 days. Call the function.", "Get the 2 day forecast for Tokyo using the tool."]
def go(params, prompt, choice, max_tokens, stream=False):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens, "tool_choice": choice,
            "tools": [{"type": "function", "function": {"name": "get_forecast", "description": "Weather forecast.", "parameters": params}}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {"err": e.read()[:200].decode(errors="replace")}
for name, (params, req) in SHAPES.items():
    for choice in ("auto", "required"):
        res = {"valid": 0, "invalid": 0, "empty": 0, "no_call": 0, "http_err": 0}
        sample = ""
        for p in PROMPTS:
            s, o = go(params, p, choice, 200)
            if s != 200:
                res["http_err"] += 1; sample = sample or str(o)[:150]; continue
            m = o["choices"][0]["message"]; tc = m.get("tool_calls") or []
            if not tc:
                res["no_call"] += 1; sample = sample or repr((m.get("content") or "")[:110]); continue
            try:
                a = json.loads(tc[0]["function"]["arguments"] or "{}")
            except Exception:
                a = None
            if not a: res["empty"] += 1
            elif jsonschema.Draft202012Validator(params).is_valid(a): res["valid"] += 1
            else: res["invalid"] += 1
            sample = sample or tc[0]["function"]["arguments"][:110]
        print(f"{name:14s} {choice:9s} {res} e.g. {sample}", flush=True)
print("== cut call: max_tokens too small")
for mt in (5, 10, 15, 20, 30):
    s, o = go(CITY, PROMPTS[0], "required", mt)
    if s == 200:
        c = o["choices"][0]; m = c["message"]
        print(f"max_tokens={mt:3d} finish={c['finish_reason']} tool_calls={json.dumps(m.get('tool_calls'))[:130] if m.get('tool_calls') else None} content={(m.get('content') or '')[:90]!r}", flush=True)
    else:
        print(mt, s, o)
