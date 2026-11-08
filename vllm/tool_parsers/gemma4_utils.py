# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.

"""Gemma4 tool call parsing utilities for offline inference.

Standalone functions that parse decoded model text to extract tool calls
from Gemma4 models. These are pure-Python utilities with zero heavy
dependencies — they work on raw decoded strings from any inference
backend (vLLM, HuggingFace, TGI, etc.).

For the OpenAI-compatible API server tool parser (streaming +
non-streaming), see ``vllm.tool_parsers.gemma4_tool_parser``.
For thinking/reasoning output parsing, see
``vllm.reasoning.gemma4_utils``.

Usage with vLLM offline inference::

    from vllm import LLM, SamplingParams
    from vllm.tool_parsers.gemma4_utils import (
        parse_tool_calls,
        has_tool_response_tag,
    )

    llm = LLM(model="google/gemma-4-it")
    outputs = llm.generate(prompt, SamplingParams(...))
    text = tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=False)

    # Extract tool calls
    tool_calls = parse_tool_calls(text)
    for tc in tool_calls:
        print(f"{tc['name']}({tc['arguments']})")

Ported from ``transformers.models.gemma4.utils_gemma4`` so that vLLM users
do not need a transformers dependency for output parsing.
"""

import json
from typing import Any

import regex as re

from vllm.tool_parsers.json_schema_fold import fold_schema_combinators

# Tool call delimiter tokens as they appear in decoded text.
# The Gemma 4 chat template renders only a node's own properties / required / type.
_SCHEMA_COMBINATOR_MARKERS = ('"allOf"', '"anyOf"', '"oneOf"', '"$ref"', '"if"')
_SCHEMA_DROPPED_KEYS = frozenset(
    ("$ref", "$defs", "definitions", "allOf", "anyOf", "oneOf", "if", "then", "else")
)
_SCHEMA_MAX_DEPTH = 32
_SCHEMA_MAX_NODES = 20000


class _SchemaTooLarge(Exception):
    """Inlining would expand past the limit: keep the client's schema."""


class _PromptSchema:
    """Rewrite a folded JSON schema into what the Gemma 4 template can render.

    Only for the prompt: the request keeps the client's schema. Local ``$ref`` are
    inlined; ``anyOf`` / ``oneOf`` of objects become one object with the union of the
    properties and the keys every branch requires; a null branch becomes ``nullable``.
    """

    def __init__(self, root: dict[str, Any]) -> None:
        self.root = root
        self.nodes = 0

    def resolve(self, ref: Any) -> dict[str, Any] | None:
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return None
        target: Any = self.root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return None
            target = target[part]
        return target if isinstance(target, dict) else None

    def union(self, branches: list[dict[str, Any]]) -> dict[str, Any]:
        non_null = [b for b in branches if b.get("type") != "null"]
        if not non_null:
            return {"type": "null"}
        nullable = len(non_null) < len(branches)
        if len(non_null) == 1:
            out = dict(non_null[0])
        elif all(isinstance(b.get("properties"), dict) for b in non_null):
            properties: dict[str, Any] = {}
            for branch in non_null:
                for name, sub in branch["properties"].items():
                    properties.setdefault(name, sub)
            required = [
                key
                for key in non_null[0].get("required") or []
                if all(key in (b.get("required") or []) for b in non_null)
            ]
            out = {"type": "object", "properties": properties}
            if required:
                out["required"] = required
        else:
            # The template has room for one type: the first branch stands for all.
            out = dict(non_null[0])
        if nullable:
            out["nullable"] = True
        return out

    def node(self, schema: Any, depth: int, seen: tuple[str, ...] = ()) -> Any:
        if not isinstance(schema, dict):
            return schema
        self.nodes += 1
        if self.nodes > _SCHEMA_MAX_NODES:
            raise _SchemaTooLarge
        out = {k: v for k, v in schema.items() if k not in _SCHEMA_DROPPED_KEYS}
        if depth > _SCHEMA_MAX_DEPTH:
            return out

        ref = schema.get("$ref")
        if ref is not None:
            target = self.resolve(ref) if ref not in seen else None
            if target is not None:
                inlined = self.node(target, depth + 1, (*seen, ref))
                out = {**inlined, **out}

        for branch in schema.get("allOf") or []:
            merged = self.node(branch, depth + 1, seen)
            if not isinstance(merged, dict):
                continue
            own_properties = out.get("properties")
            own_required = out.get("required")
            out = {**merged, **out}
            if isinstance(own_properties, dict):
                out["properties"] = {**merged.get("properties", {}), **own_properties}
            if isinstance(own_required, list):
                extra = merged.get("required") or []
                out["required"] = own_required + [
                    key for key in extra if key not in own_required
                ]

        for key in ("anyOf", "oneOf"):
            branches = schema.get(key)
            if isinstance(branches, list) and branches:
                nodes = [self.node(b, depth + 1, seen) for b in branches]
                out = {
                    **self.union([b for b in nodes if isinstance(b, dict)]),
                    **out,
                }
                break

        if isinstance(out.get("properties"), dict):
            out["properties"] = {
                name: self.node(sub, depth + 1, seen)
                for name, sub in out["properties"].items()
            }
        if isinstance(out.get("items"), dict):
            out["items"] = self.node(out["items"], depth + 1, seen)
        return out


def normalize_tool_parameters(parameters: Any) -> Any:
    """Rewrite a tool's parameters for the Gemma 4 chat template.

    A schema with no combinators or ``$ref`` is returned as is, so its prompt does not
    change. Any failure also returns the client's schema.
    """
    if not isinstance(parameters, dict):
        return parameters
    try:
        text = json.dumps(parameters)
        if not any(marker in text for marker in _SCHEMA_COMBINATOR_MARKERS):
            return parameters
        folded = json.loads(fold_schema_combinators(text)[0])
        out = _PromptSchema(folded).node(folded, 0)
    except (TypeError, ValueError, RecursionError, _SchemaTooLarge):
        return parameters
    if not isinstance(out, dict) or not out:
        return parameters
    if isinstance(out.get("properties"), dict):
        out.setdefault("type", "object")
    return out


def normalize_tools_for_template(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return tools whose parameters the Gemma 4 chat template can render."""
    out = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and "parameters" in function:
            parameters = normalize_tool_parameters(function["parameters"])
            if parameters is not function["parameters"]:
                tool = {**tool, "function": {**function, "parameters": parameters}}
        out.append(tool)
    return out


# Standard format: <|tool_call>call:name{args}<tool_call|>
_TOOL_CALL_START_TAG = "<|tool_call>"
_TOOL_CALL_END_TAG = "<tool_call|>"
_TOOL_RESPONSE_START_TAG = "<|tool_response>"

# Gemma4 escape token as it appears in decoded text.
_ESCAPE_TOKEN = '<|"|>'


def _parse_tool_arguments(args_str: str) -> dict[str, str]:
    """Parse tool call arguments from the Gemma4 compact format.

    Delegates to the native ``<|"|>``-aware parser from
    ``vllm.parser.gemma4``, which handles internal quotes, nested
    objects, arrays, and all Gemma4 value types correctly.

    Args:
        args_str: Raw argument string from inside ``call:name{...}``.

    Returns:
        Dictionary of argument name → string value.
    """
    if not args_str or not args_str.strip():
        return {}

    from vllm.parser.gemma4 import _parse_gemma4_args

    parsed = _parse_gemma4_args(args_str)
    return {k: str(v) if not isinstance(v, str) else v for k, v in parsed.items()}


def parse_tool_calls(text: str, *, strict: bool = False) -> list[dict]:
    """Parse tool calls from decoded Gemma4 model output.

    Uses a tiered parsing strategy to handle known output variations in
    Gemma4 models, which may emit
    non-standard tool call formats.

    Parsing tiers:
        1. **Standard**: ``<|tool_call>call:name{args}<tool_call|>``
           (special token IDs 48/49 in decoded text)
        2. **Fallback** (when ``strict=False``): bare ``call:name{args}``
           patterns, including ``<call>name{args}`` (fragmented tokens from
           multimodal inputs)

    Args:
        text: Decoded model output text (from ``tokenizer.decode(...,
            skip_special_tokens=False)``).
        strict: If ``True``, only match the standard ``<|tool_call>`` format.
            If ``False`` (default), also try fallback patterns for
            known Gemma4 output variations.

    Returns:
        A list of dicts, each with keys:
            - ``"name"``: The tool function name (e.g. ``"get_weather"``).
            - ``"arguments"``: A dict of argument name → value.

    Example::

        >>> from vllm.tool_parsers.gemma4_utils import parse_tool_calls
        >>> output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        >>> tool_calls = parse_tool_calls(output)
        >>> for tc in tool_calls:
        ...     print(f"Call: {tc['name']}({tc['arguments']})")
    """
    results = []

    # Tier 1: Standard format with special tokens.
    # <|tool_call>call:name{args}<tool_call|>
    # Note: Some Gemma4 models emit <turn|> instead of <tool_call|>.
    standard_pattern = r"<\|tool_call\>call:(\w+)\{(.*?)\}(?:<tool_call\|>|<turn\|>)"
    for match in re.finditer(standard_pattern, text, re.DOTALL):
        name, args_str = match.group(1), match.group(2)
        results.append(
            {
                "name": name,
                "arguments": _parse_tool_arguments(args_str),
            }
        )

    if results or strict:
        return results

    # Tier 2: Fallback for known Gemma4 output variations.
    # Matches: <call>name{args}, call:name{args}, or bare call:name{args}<eos>
    fallback_pattern = r"(?:<call>|(?:^|\s)call:)(\w+)\{(.*?)\}"
    for match in re.finditer(fallback_pattern, text, re.DOTALL):
        name, args_str = match.group(1), match.group(2)
        results.append(
            {
                "name": name,
                "arguments": _parse_tool_arguments(args_str),
            }
        )

    return results


def has_tool_response_tag(text: str) -> bool:
    """Check if model output properly ends with a tool response tag.

    Some Gemma4 models sometimes emit ``<eos>`` instead of
    ``<|tool_response>`` after a tool call. This helper detects
    whether the model used the proper termination, so callers can
    decide whether to inject ``<|tool_response>`` into the next prompt.

    Args:
        text: Decoded model output text.

    Returns:
        ``True`` if the output ends with ``<|tool_response>``
        (proper behavior), ``False`` otherwise.

    Example::

        >>> from vllm.tool_parsers.gemma4_utils import has_tool_response_tag
        >>> if not has_tool_response_tag(model_output):
        ...     # Model used <eos> instead — inject <|tool_response> manually
        ...     next_prompt = "<|tool_response>" + tool_result
    """
    stripped = text.rstrip()
    return stripped.endswith(_TOOL_RESPONSE_START_TAG)
