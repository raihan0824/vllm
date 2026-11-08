# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import Gemma4ParserToolAdapter
from vllm.tool_parsers.gemma4_utils import normalize_tools_for_template


class Gemma4EngineToolParser(Gemma4ParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False

    @staticmethod
    def tools_for_chat_template(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rewrite tool schemas the Gemma 4 chat template cannot render.

        A root ``$ref`` / ``allOf`` / ``anyOf`` / ``oneOf`` has no ``properties`` of its
        own, so the template rendered no parameters and the model called the tool with
        ``{}``. There is no tool grammar on this path, so the prompt is all it gets.
        """
        return normalize_tools_for_template(tools)

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Skip structured-output JSON for required/named tool choice.

        Gemma4 emits its native ``<|tool_call>call:...`` syntax, which the
        parser extracts directly. The base ``ToolParser.adjust_request`` would
        set ``structured_outputs`` for required/named and force JSON via guided
        decoding, conflicting with that native syntax (it leaks as content and
        crashes EngineCore under speculative decoding). Skip it so the model
        emits its native format (mirrors the GLM4 parser).
        """
        if request.tools:
            tc = request.tool_choice
            if tc == "required" or isinstance(
                tc, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
            ):
                request.skip_special_tokens = False
                return request
        return super().adjust_request(request)
