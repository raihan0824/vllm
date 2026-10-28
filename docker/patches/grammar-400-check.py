"""Build-time check for the grammar-400 overlay (no GPU at build time)."""

import sys
from unittest.mock import Mock

import vllm.v1.core.sched.scheduler  # noqa: F401
import vllm.v1.engine.output_processor  # noqa: F401
from vllm.config import StructuredOutputsConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_xgrammar import XgrammarJSONSchemaError
from vllm.v1.structured_output.request import GRAMMAR_COMPILE_ERROR_PREFIX

# The incident schema must be rejected at validation with a clean message.
params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json={"type": "object", "properties": {"c": {"type": "string", "enum": []}}}
    )
)
try:
    params._validate_structured_outputs(
        Mock(is_diffusion=False), StructuredOutputsConfig(backend="auto"), Mock()
    )
    sys.exit("empty enum was accepted")
except XgrammarJSONSchemaError as e:
    expected = "Invalid JSON schema for structured output: enum array must not be empty"
    assert str(e) == expected, e
assert GRAMMAR_COMPILE_ERROR_PREFIX

# The base image's own patches must still be there.
assert hasattr(EngineArgs(), "max_waiting_requests"), "mwr flag missing"
if sys.argv[1:] == ["gemma"]:
    from vllm.model_executor.layers.utils import get_penalties_rows_per_chunk
    from vllm.multimodal.media.connector import _wrap_media_fetch_error

    assert get_penalties_rows_per_chunk(262144) == 46
    assert _wrap_media_fetch_error
print("grammar-400 patch verified:", sys.argv[1:] or ["gpt-oss"])
