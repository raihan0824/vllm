# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4 tool declarations for schemas the chat template cannot render as written.

A root ``$ref`` / ``allOf`` / ``anyOf`` / ``oneOf`` has no ``properties`` of its own:
the template rendered ``parameters:{}`` with an unbalanced brace, and every call on
the prod image came back as ``{}``.
"""

import copy
import json
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import jinja2
import pytest
import torch

from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser
from vllm.tool_parsers.gemma4_utils import (
    normalize_tool_parameters,
    normalize_tools_for_template,
)
from vllm.tool_parsers.json_schema_fold import fold_schema_combinators

TEMPLATE = (
    Path(__file__).parents[2] / "examples" / "tool_chat_template_gemma4.jinja"
).read_text()

S = {"type": "string"}
I = {"type": "integer"}  # noqa: E741
CITY = {
    "type": "object",
    "properties": {"city": S, "days": I},
    "required": ["city", "days"],
}
ZIP = {"type": "object", "properties": {"zip": S}, "required": ["zip"]}
CALL = {
    "type": "object",
    "properties": {"name": S, "arguments": {"type": "object"}},
    "required": ["name", "arguments"],
}
CALLS = {
    "type": "object",
    "properties": {"calls": {"type": "array", "items": CALL}},
    "required": ["calls"],
}


def render(parameters: dict) -> str:
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, **_: json.dumps(value)
    tools = normalize_tools_for_template(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_forecast",
                    "description": "d",
                    "parameters": parameters,
                },
            }
        ]
    )
    out = env.from_string(TEMPLATE).render(
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        add_generation_prompt=True,
        bos_token="<bos>",
    )
    return out[out.index("<|tool>") + len("<|tool>") : out.index("<tool|>")]


def balanced(declaration: str) -> bool:
    return declaration.count("{") == declaration.count("}")


class TestShapesFromProd:
    """The six shapes of the 2026-11-09 probe on the prod image."""

    def test_plain_object_is_untouched(self):
        assert normalize_tool_parameters(CITY) is CITY
        assert render(CITY) == (
            'declaration:get_forecast{description:<|"|>d<|"|>,parameters:{'
            'properties:{city:{type:<|"|>STRING<|"|>},days:{type:<|"|>INTEGER<|"|>}},'
            'required:[<|"|>city<|"|>,<|"|>days<|"|>],type:<|"|>OBJECT<|"|>}}'
        )

    def test_wrapper_object_is_untouched(self):
        assert normalize_tool_parameters(CALLS) is CALLS

    @pytest.mark.parametrize(
        "schema",
        [
            {"$ref": "#/$defs/W", "$defs": {"W": CITY}},
            {"$ref": "#/definitions/W", "definitions": {"W": CITY}},
            {"allOf": [CITY]},
            {"allOf": [{"$ref": "#/$defs/W"}], "$defs": {"W": CITY}},
            {"type": "object", "allOf": [CITY]},
        ],
    )
    def test_root_ref_and_all_of_render_like_the_plain_object(self, schema):
        assert render(schema) == render(CITY)

    @pytest.mark.parametrize("key", ["anyOf", "oneOf"])
    def test_root_union_shows_every_parameter(self, key):
        declaration = render({key: [CITY, ZIP]})
        assert balanced(declaration)
        for name in ("city:", "days:", "zip:"):
            assert name in declaration
        # No key is required by both branches.
        assert "required:" not in declaration

    def test_root_union_keeps_keys_every_branch_requires(self):
        both = {
            "type": "object",
            "properties": {"id": S, "city": S},
            "required": ["id", "city"],
        }
        other = {
            "type": "object",
            "properties": {"id": S, "zip": S},
            "required": ["id", "zip"],
        }
        out = normalize_tool_parameters({"anyOf": [both, other]})
        assert out["required"] == ["id"]

    def test_nested_ref_is_inlined(self):
        declaration = render(
            {
                "type": "object",
                "properties": {"q": {"$ref": "#/$defs/W"}},
                "required": ["q"],
                "$defs": {"W": CITY},
            }
        )
        assert balanced(declaration)
        assert "city:" in declaration
        assert "q:{properties:{" in declaration
        assert "$ref" not in declaration
        assert "$defs" not in declaration


class TestNormalize:
    def test_request_schema_is_not_mutated(self):
        schema = {"$ref": "#/$defs/W", "$defs": {"W": copy.deepcopy(CITY)}}
        before = copy.deepcopy(schema)
        tools = [{"type": "function", "function": {"name": "t", "parameters": schema}}]
        out = normalize_tools_for_template(tools)
        assert schema == before
        assert tools[0]["function"]["parameters"] is schema
        assert out[0]["function"]["parameters"] is not schema

    def test_tool_without_a_rewrite_is_the_same_object(self):
        tool = {"type": "function", "function": {"name": "t", "parameters": CITY}}
        assert normalize_tools_for_template([tool])[0] is tool

    def test_no_parameters_key(self):
        tool = {"type": "function", "function": {"name": "t"}}
        assert normalize_tools_for_template([tool])[0] is tool

    def test_parser_hook(self):
        schema = {"allOf": [CITY]}
        tool = {"type": "function", "function": {"name": "t", "parameters": schema}}
        out = Gemma4EngineToolParser.tools_for_chat_template([tool])
        assert out[0]["function"]["parameters"]["required"] == ["city", "days"]

    def test_optional_property_becomes_nullable(self):
        out = normalize_tool_parameters(
            {
                "type": "object",
                "properties": {"x": {"anyOf": [S, {"type": "null"}]}},
            }
        )
        assert out["properties"]["x"] == {"type": "string", "nullable": True}

    def test_description_next_to_ref_wins(self):
        out = normalize_tool_parameters(
            {
                "type": "object",
                "properties": {"q": {"$ref": "#/$defs/W", "description": "mine"}},
                "$defs": {"W": {**CITY, "description": "theirs"}},
            }
        )
        assert out["properties"]["q"]["description"] == "mine"
        assert out["properties"]["q"]["required"] == ["city", "days"]

    def test_recursive_ref_terminates(self):
        schema = {
            "$ref": "#/$defs/N",
            "$defs": {
                "N": {
                    "type": "object",
                    "properties": {"v": I, "next": {"$ref": "#/$defs/N"}},
                    "required": ["v"],
                }
            },
        }
        assert balanced(render(schema))
        assert "v:" in render(schema)

    def test_expansion_past_the_limit_keeps_the_client_schema(self):
        defs: dict = {"L0": {"type": "object", "properties": {"x": S}}}
        for i in range(1, 30):
            ref = {"$ref": f"#/$defs/L{i - 1}"}
            defs[f"L{i}"] = {
                "type": "object",
                "properties": {"a": ref, "b": ref, "c": ref},
            }
        schema = {"$ref": "#/$defs/L29", "$defs": defs}
        assert normalize_tool_parameters(schema) is schema

    @pytest.mark.parametrize(
        "schema",
        [
            {"$ref": "https://example.com/s.json"},
            {"$ref": "#/$defs/missing"},
            {"allOf": "x"},
            {"anyOf": [1, 2]},
            {"oneOf": []},
            {"allOf": [{"required": "a"}, {"required": ["b"]}]},
        ],
    )
    def test_shapes_it_cannot_resolve_stay_balanced(self, schema):
        normalize_tool_parameters(schema)
        assert balanced(render(schema))

    def test_malformed_properties_do_not_raise(self):
        schema = {"anyOf": [{"type": "object"}], "properties": "p"}
        assert normalize_tool_parameters(schema)["properties"] == "p"


class TestTemplateBrace:
    def test_root_without_type_is_closed(self):
        declaration = render({"properties": {"a": S}, "required": ["a"]})
        assert balanced(declaration)
        assert declaration.endswith('type:<|"|>OBJECT<|"|>}}')

    def test_no_parameters(self):
        assert balanced(render({}))


class TestFold:
    """Ported from the SGLang fold tests that do not need a grammar compiler."""

    def fold(self, schema: dict) -> dict:
        return json.loads(fold_schema_combinators(json.dumps(schema))[0])

    def test_merge_rules(self):
        schema = {
            "allOf": [
                {
                    "type": ["string", "null"],
                    "minLength": 2,
                    "maxLength": 9,
                    "enum": ["ab", "abc", None],
                },
                {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 5,
                    "enum": ["abc", "abcd"],
                },
            ]
        }
        assert self.fold(schema) == {
            "type": "string",
            "minLength": 3,
            "maxLength": 5,
            "enum": ["abc"],
        }

    def test_condition_on_optional_key_is_left_alone(self):
        schema = {
            "type": "object",
            "properties": {"k": {"enum": ["a", "b"]}, "v": S},
            "if": {"properties": {"k": {"const": "a"}}},
            "then": {"required": ["v"]},
        }
        text = json.dumps(schema)
        assert fold_schema_combinators(text) == (text, 0)

    def test_number_and_integer(self):
        schema = {"allOf": [{"type": "number"}, {"type": "integer"}]}
        assert self.fold(schema) == {"type": "integer"}

    def test_properties_in_both_branches_merge(self):
        schema = {
            "allOf": [
                {"properties": {"a": {"type": "string"}}},
                {"properties": {"a": {"maxLength": 4}}, "required": ["a"]},
            ]
        }
        assert self.fold(schema) == {
            "properties": {"a": {"type": "string", "maxLength": 4}},
            "required": ["a"],
        }

    @pytest.mark.parametrize(
        "schema",
        [
            {"allOf": [{"type": "string"}, {"type": "integer"}]},
            {"allOf": [{"$ref": "https://example.com/s.json"}, {"type": "object"}]},
            {"allOf": [{"$ref": "#/$defs/missing"}, {"type": "object"}]},
        ],
    )
    def test_conflict_and_unsupported_keep_the_node(self, schema):
        text = json.dumps(schema)
        assert fold_schema_combinators(text) == (text, 0)

    @pytest.mark.parametrize(
        "schema",
        [
            {"allOf": [{"required": "a"}, {"required": ["b"]}]},
            {"allOf": [{"properties": []}, {"properties": {"a": S}}]},
            {"allOf": [{"enum": 3}, {"enum": [3]}]},
            {"allOf": [1, {"type": "object"}]},
            {"allOf": "x"},
            {"anyOf": [{"type": "object"}], "properties": "p"},
            {"allOf": [{"minimum": "1"}, {"minimum": 2}]},
        ],
    )
    def test_malformed_schemas_never_raise(self, schema):
        json.loads(fold_schema_combinators(json.dumps(schema))[0])

    @pytest.mark.parametrize(
        "text", ['{"type": "object"}', json.dumps(CITY), "not json", "[1, 2]"]
    )
    def test_plain_schemas_untouched(self, text):
        assert fold_schema_combinators(text) == (text, 0)

    def test_if_then_on_required_enum_splits_by_value(self):
        schema = {
            "type": "object",
            "properties": {"k": {"enum": ["a", "b"]}, "v": S},
            "required": ["k"],
            "if": {"properties": {"k": {"const": "a"}}},
            "then": {"required": ["v"]},
        }
        branches = self.fold(schema)["anyOf"]
        assert len(branches) == 2
        assert sorted(b["required"] for b in branches) == [["k"], ["k", "v"]]


class TestCrashDumpSummary:
    def feature(self, offset: int, shape: tuple | None) -> MultiModalFeatureSpec:
        data = None
        if shape is not None:
            elem = MagicMock()
            elem.data = torch.zeros(shape)
            data = {"pixel_values": elem}
        return MultiModalFeatureSpec(
            data=data,
            modality="image",
            identifier=f"h{offset}",
            mm_position=PlaceholderRange(offset=offset, length=258),
        )

    def test_cached_request_size_and_shapes(self, monkeypatch):
        lines: list[str] = []
        monkeypatch.setattr(
            "vllm.logging_utils.dump_input.logger.error",
            lambda msg, *args: lines.append(msg % args if args else msg),
        )
        monkeypatch.setattr(
            "vllm.logging_utils.dump_input._dump_engine_exception",
            lambda *args: None,
        )
        request = MagicMock()
        request.num_prompt_tokens = 18000
        request.num_computed_tokens = 15699
        request.mm_features = [
            self.feature(i * 300, (2520, 768) if i == 46 else None) for i in range(52)
        ]
        output = MagicMock()
        output.num_scheduled_tokens = {"a": 1725, "gone": 5}
        output.scheduled_encoder_inputs = {"a": [46, 47]}
        dump_engine_exception(MagicMock(), output, None, {"a": request})

        assert len(lines) == 1
        line = lines[0]
        assert "req_id=a prompt_tokens=18000 computed_tokens=15699" in line
        assert "scheduled_tokens=1725" in line
        assert "mm_items={'image': 52} mm_ranges=52" in line
        assert "46: {'pixel_values': (2520, 768)}" in line
        # Sent as cached: no tensor to measure.
        assert "47: None" in line

    def test_summary_failure_is_silent(self, monkeypatch):
        monkeypatch.setattr(
            "vllm.logging_utils.dump_input._dump_engine_exception",
            lambda *args: None,
        )
        output = MagicMock()
        output.num_scheduled_tokens = {"a": 1}
        output.scheduled_encoder_inputs = {}
        request = MagicMock()
        request.mm_features = None
        dump_engine_exception(MagicMock(), output, None, {"a": request})

    def test_no_requests_keeps_old_behaviour(self, monkeypatch):
        calls = Counter()
        monkeypatch.setattr(
            "vllm.logging_utils.dump_input._dump_engine_exception",
            lambda *args: calls.update(["dump"]),
        )
        dump_engine_exception(MagicMock(), MagicMock(), None)
        assert calls["dump"] == 1
