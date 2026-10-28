# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import functools
import json
import re
from concurrent.futures import Future
from concurrent.futures._base import TimeoutError
from typing import TYPE_CHECKING, Any, cast

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputKey,
    StructuredOutputOptions,
)

if TYPE_CHECKING:
    from vllm.reasoning import ReasoningParser


# stop_reason prefix with which the engine reports a request whose grammar
# failed to compile; the frontend turns it into a validation error (HTTP 400).
GRAMMAR_COMPILE_ERROR_PREFIX = "vllm.invalid_structured_output:"

# xgrammar errors start with "[hh:mm:ss] /path/to/file.cc:123: ".
_NATIVE_ERROR_LOCATION = re.compile(r"^\[[\d:]+\]\s*\S+:\d+:\s*")


def grammar_compile_error_detail(exc: Exception) -> str:
    """The first line of a grammar compilation error, without the native
    source location.

    Args:
        exc: The exception raised while compiling the grammar.

    Returns:
        The cleaned error text, or the exception type name if it is empty.
    """
    lines = str(exc).strip().splitlines()
    detail = _NATIVE_ERROR_LOCATION.sub("", lines[0]) if lines else ""
    return detail or type(exc).__name__


def grammar_compile_error_message(exc: Exception) -> str:
    """Client-facing message for a grammar that failed to compile."""
    return f"Invalid structured output: {grammar_compile_error_detail(exc)}"


@dataclasses.dataclass
class StructuredOutputRequest:
    params: StructuredOutputsParams
    # A failed grammar compilation is stored as its exception, so the scheduler
    # can fail only this request instead of the engine.
    _grammar: (
        Future[StructuredOutputGrammar] | StructuredOutputGrammar | Exception | None
    ) = None
    reasoning_ended: bool | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None
    # Cached per request; do not share reasoning parsers across requests because
    # their behavior can depend on reasoning_parser_kwargs.
    reasoner: "ReasoningParser | None" = None

    @staticmethod
    def from_sampling_params(
        sampling_params: SamplingParams | None,
    ) -> "StructuredOutputRequest | None":
        if sampling_params is None:
            return None
        params = sampling_params.structured_outputs
        if not params or params.all_constraints_none():
            return None
        return StructuredOutputRequest(params=params)

    def _check_grammar_completion(self) -> bool:
        # NOTE: We have to lazy import to gate circular imports
        from vllm.v1.request import RequestStatus

        if isinstance(self._grammar, Future):
            try:
                # We will check whether the future is ready within 100 us
                self._grammar = self._grammar.result(timeout=0.0001)
                self.status = RequestStatus.WAITING
            except TimeoutError:
                return False
            except Exception as e:
                self._grammar = e
        return True

    @property
    def is_grammar_ready(self) -> bool:
        return self._check_grammar_completion()

    @property
    def grammar(self) -> StructuredOutputGrammar | Exception | None:
        """The compiled grammar, the exception if compilation failed, or None
        while compilation is still running."""
        if not self._check_grammar_completion():
            return None
        return cast(StructuredOutputGrammar | Exception | None, self._grammar)

    @grammar.setter
    def grammar(
        self, grammar: StructuredOutputGrammar | Future[StructuredOutputGrammar]
    ) -> None:
        self._grammar = grammar

    @functools.cached_property
    def structured_output_key(self) -> StructuredOutputKey:
        return get_structured_output_key(self.params)


def get_structured_output_key(params: StructuredOutputsParams) -> StructuredOutputKey:
    if params.json is not None:
        if not isinstance(params.json, str):
            json_str = json.dumps(params.json)
        else:
            json_str = params.json
        return StructuredOutputOptions.JSON, json_str
    if params.json_object:
        return StructuredOutputOptions.JSON_OBJECT, ""
    if params.regex is not None:
        return StructuredOutputOptions.REGEX, params.regex
    if params.choice is not None:
        if not isinstance(params.choice, str):
            json_str = json.dumps(params.choice)
        else:
            json_str = params.choice
        return StructuredOutputOptions.CHOICE, json_str
    if params.grammar is not None:
        return StructuredOutputOptions.GRAMMAR, params.grammar
    if params.structural_tag is not None:
        return StructuredOutputOptions.STRUCTURAL_TAG, params.structural_tag
    raise ValueError("No valid structured output parameter found")
