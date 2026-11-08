"""Build-time check for the gemma4-toolfold overlay (no GPU at build time)."""

import inspect
import json
from unittest.mock import MagicMock

import jinja2

from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.model_executor.layers.utils import get_penalties_rows_per_chunk
from vllm.multimodal.media.connector import _wrap_media_fetch_error
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser
from vllm.v1.structured_output.request import GRAMMAR_COMPILE_ERROR_PREFIX

CITY = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


def render(parameters):
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, **_: json.dumps(value)
    with open("/vllm-workspace/examples/tool_chat_template_gemma4.jinja") as f:
        template = env.from_string(f.read())
    tool = {
        "type": "function",
        "function": {"name": "t", "description": "d", "parameters": parameters},
    }
    tools = Gemma4EngineToolParser.tools_for_chat_template([tool])
    return template.render(
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        add_generation_prompt=True,
        bos_token="<bos>",
    )


# Every root shape must show the parameters and close its braces.
for schema in (
    {"$ref": "#/$defs/W", "$defs": {"W": CITY}},
    {"allOf": [CITY]},
    {"anyOf": [CITY, CITY]},
    {"oneOf": [CITY, CITY]},
    {"properties": CITY["properties"], "required": ["city"]},
):
    declaration = render(schema).split("<|tool>")[1].split("<tool|>")[0]
    assert "city:" in declaration, schema
    assert declaration.count("{") == declaration.count("}"), schema
assert render(CITY) == render({"allOf": [CITY]})

# The request's plain schema must pass through as the same object.
plain = {"type": "function", "function": {"name": "t", "parameters": CITY}}
assert Gemma4EngineToolParser.tools_for_chat_template([plain])[0] is plain

# The crash dump must take the scheduler's requests.
assert "requests" in inspect.signature(dump_engine_exception).parameters
dump_engine_exception(MagicMock(), MagicMock(), None, {})

# The base image's own patches must still be there.
assert get_penalties_rows_per_chunk(262144) == 46
assert _wrap_media_fetch_error
assert GRAMMAR_COMPILE_ERROR_PREFIX
print("gemma4-toolfold patch verified")
