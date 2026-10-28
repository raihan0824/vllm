# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A JSON schema the grammar backend cannot compile must fail only its own
request with a 400, never the engine (gpt-oss-120b crash, 2026-10-28)."""

import asyncio
from concurrent.futures import Future
from http import HTTPStatus
from unittest.mock import Mock

import pytest

from tests.v1.core.utils import EOS_TOKEN_ID, create_requests, create_scheduler
from vllm.config import StructuredOutputsConfig
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output.backend_xgrammar import XgrammarJSONSchemaError
from vllm.v1.structured_output.request import (
    GRAMMAR_COMPILE_ERROR_PREFIX,
    grammar_compile_error_message,
)

# Schemas xgrammar's converter rejects. Before the fix, "auto" validation fell
# back to the guidance backend for these, and the engine (which compiles with
# the backend it started with, xgrammar) crashed on them.
POISON_SCHEMAS = {
    "empty enum (incident)": {
        "type": "object",
        "properties": {"color": {"type": "string", "enum": []}},
        "required": ["color"],
    },
    "missing $ref": {"type": "object", "properties": {"a": {"$ref": "#/$defs/x"}}},
    "minItems > maxItems": {"type": "array", "minItems": 3, "maxItems": 1},
    "minimum > maximum": {"type": "integer", "minimum": 5, "maximum": 1},
    "unknown type": {"type": "nonsense"},
}


def _poison_request(request_id: str, schema: dict) -> Request:
    sampling_params = SamplingParams(
        max_tokens=16, structured_outputs=StructuredOutputsParams(json=schema)
    )
    sampling_params.structured_outputs._backend = "guidance"
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=[0, 1],
        sampling_params=sampling_params,
        pooling_params=None,
    )


def _finish_errors(scheduler):
    scheduler_output = scheduler.schedule()
    assert not scheduler_output.num_scheduled_tokens
    return scheduler.update_from_output(
        scheduler_output, ModelRunnerOutput(req_ids=[], req_id_to_index={})
    )


@pytest.mark.parametrize("async_grammar", [True, False])
def test_compile_error_finishes_only_that_request(async_grammar):
    scheduler = create_scheduler()
    manager = scheduler.structured_output_manager
    manager.backend = Mock()
    manager.backend.compile_grammar.side_effect = RuntimeError(
        "[15:00:03] /project/cpp/json_schema_converter.cc:3363: "
        "enum array must not be empty"
    )
    manager._use_async_grammar_compilation = async_grammar

    request = _poison_request("poison", POISON_SCHEMAS["empty enum (incident)"])
    manager.grammar_init(request)
    grammar = request.structured_output_request._grammar
    assert isinstance(grammar, Future)
    assert isinstance(grammar.exception(timeout=5), RuntimeError)

    scheduler.add_request(request)
    outputs = _finish_errors(scheduler)  # Must not raise.

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    output = outputs[0].outputs[0]
    assert output.request_id == "poison"
    assert output.finish_reason == FinishReason.ERROR
    assert output.stop_reason == (
        GRAMMAR_COMPILE_ERROR_PREFIX
        + "Invalid structured output: enum array must not be empty"
    )

    # The engine keeps serving.
    healthy = create_requests(num_requests=1, req_ids=["healthy"])[0]
    scheduler.add_request(healthy)
    assert [r.req_id for r in scheduler.schedule().scheduled_new_reqs] == ["healthy"]


@pytest.mark.parametrize("schema", POISON_SCHEMAS.values(), ids=POISON_SCHEMAS.keys())
def test_real_xgrammar_compile_error_does_not_raise(schema):
    # The engine's backend is xgrammar (started by a valid request); the poison
    # request resolved to another backend during validation, as in the crash.
    scheduler = create_scheduler()
    manager = scheduler.structured_output_manager
    valid = _poison_request("valid", {"type": "object"})
    valid.sampling_params.structured_outputs._backend = "xgrammar"
    manager.grammar_init(valid)
    assert type(manager.backend).__name__ == "XgrammarBackend"

    request = _poison_request("poison", schema)
    manager.grammar_init(request)
    request.structured_output_request._grammar.exception(timeout=30)
    scheduler.add_request(request)
    outputs = _finish_errors(scheduler)

    output = outputs[0].outputs[0]
    assert output.finish_reason == FinishReason.ERROR
    assert output.stop_reason.startswith(GRAMMAR_COMPILE_ERROR_PREFIX)
    assert ".cc:" not in output.stop_reason  # No native source path.


def test_aborted_before_finish_gets_no_second_output():
    scheduler = create_scheduler()
    request = _poison_request("poison", POISON_SCHEMAS["empty enum (incident)"])
    request.structured_output_request.grammar = RuntimeError("boom")
    scheduler.add_request(request)
    scheduler.schedule()  # Records the compile error.
    scheduler.finish_requests("poison", RequestStatus.FINISHED_ABORTED)
    outputs = scheduler.update_from_output(
        scheduler.schedule(), ModelRunnerOutput(req_ids=[], req_id_to_index={})
    )
    assert not any(
        o.request_id == "poison" for eco in outputs.values() for o in eco.outputs
    )


def test_output_processor_turns_marker_into_validation_error():
    message = grammar_compile_error_message(
        RuntimeError("enum array must not be empty")
    )
    collector = RequestOutputCollector(output_kind=Mock(), request_id="poison")
    req_state = Mock()
    req_state.routed_experts_chunks = []
    req_state.queue = collector
    req_state.is_prefilling = False
    req_state.streaming_input = False
    req_state.detokenizer.update.return_value = None
    req_state.request_id = "poison"

    from vllm.v1.engine.output_processor import OutputProcessor

    processor = OutputProcessor.__new__(OutputProcessor)
    processor.request_states = {"poison": req_state}
    processor.tracing_enabled = False
    processor._update_stats_from_output = Mock()
    processor._update_stats_from_finished = Mock()
    processor._finish_request = Mock()

    processor.process_outputs(
        [
            EngineCoreOutput(
                request_id="poison",
                new_token_ids=[],
                finish_reason=FinishReason.ERROR,
                stop_reason=GRAMMAR_COMPILE_ERROR_PREFIX + message,
            )
        ]
    )
    req_state.make_request_output.assert_not_called()
    processor._finish_request.assert_called_once_with(req_state)
    with pytest.raises(VLLMValidationError) as exc_info:
        asyncio.run(collector.get())
    assert "enum array must not be empty" in str(exc_info.value)
    assert create_error_response(exc_info.value).error.code == HTTPStatus.BAD_REQUEST


def _verify(schema, backend="auto"):
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    params._validate_structured_outputs(
        Mock(is_diffusion=False),
        StructuredOutputsConfig(backend=backend),
        tokenizer=Mock(),
    )
    return params


@pytest.mark.parametrize("schema", POISON_SCHEMAS.values(), ids=POISON_SCHEMAS.keys())
def test_auto_validation_rejects_unconvertible_schema(schema):
    with pytest.raises(XgrammarJSONSchemaError) as exc_info:
        _verify(schema)
    message = str(exc_info.value)
    assert message.startswith("Invalid JSON schema for structured output: ")
    assert ".cc:" not in message
    err = create_error_response(exc_info.value)
    assert err.error.code == HTTPStatus.BAD_REQUEST


def test_auto_validation_keeps_fallback_for_unsupported_features(monkeypatch):
    # Valid schema using a keyword xgrammar lacks: still routed to guidance.
    import vllm.sampling_params as sp

    monkeypatch.setattr(sp, "_get_llg_tokenizer", lambda tok: Mock(), raising=False)
    monkeypatch.setattr(
        "vllm.v1.structured_output.backend_guidance.validate_guidance_grammar",
        lambda *a, **k: None,
    )
    params = _verify(
        {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
    )
    assert params.structured_outputs._backend == "guidance"


def test_auto_validation_accepts_valid_schema():
    params = _verify(
        {"type": "object", "properties": {"c": {"type": "string", "enum": ["red"]}}}
    )
    assert params.structured_outputs._backend == "xgrammar"
