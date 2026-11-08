# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import enum
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.stats import SchedulerStats
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


def prepare_object_to_dump(obj) -> str:
    if isinstance(obj, str):
        return f"'{obj}'"  # Double quotes
    elif isinstance(obj, dict):
        dict_str = ", ".join(
            {f"{str(k)}: {prepare_object_to_dump(v)}" for k, v in obj.items()}
        )
        return f"{{{dict_str}}}"
    elif isinstance(obj, list):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, set):
        return f"[{', '.join([prepare_object_to_dump(v) for v in list(obj)])}]"
        # return [prepare_object_to_dump(v) for v in list(obj)]
    elif isinstance(obj, tuple):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, enum.Enum):
        return repr(obj)
    elif isinstance(obj, torch.Tensor):
        # We only print the 'draft' of the tensor to not expose sensitive data
        # and to get some metadata in case of CUDA runtime crashed
        return f"Tensor(shape={obj.shape}, device={obj.device},dtype={obj.dtype})"
    elif hasattr(obj, "anon_repr"):
        return obj.anon_repr()
    elif hasattr(obj, "__dict__"):
        items = obj.__dict__.items()
        dict_str = ", ".join(
            [f"{str(k)}={prepare_object_to_dump(v)}" for k, v in items]
        )
        return f"{type(obj).__name__}({dict_str})"
    else:
        # Hacky way to make sure we can serialize the object in JSON format
        try:
            return json.dumps(obj)
        except (TypeError, OverflowError):
            return repr(obj)


def dump_engine_exception(
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
    requests: Mapping[str, Any] | None = None,
):
    # NOTE: ensure we can log extra info without risking raises
    # unexpected errors during logging
    with contextlib.suppress(Exception):
        _dump_engine_exception(config, scheduler_output, scheduler_stats)
    with contextlib.suppress(Exception):
        _dump_request_summary(scheduler_output, requests)


def _mm_item_shapes(feature: Any) -> dict[str, Any] | None:
    """Tensor shapes of one multimodal item; None if the item was sent as cached."""
    if feature.data is None:
        return None
    shapes: dict[str, Any] = {}
    for key, elem in feature.data.items():
        value = getattr(elem, "data", elem)
        shapes[key] = (
            tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        )
    return shapes


def _dump_request_summary(
    scheduler_output: SchedulerOutput,
    requests: Mapping[str, Any] | None,
) -> None:
    """Log each scheduled request's size, which the output omits for cached requests.

    ``mm_ranges`` is the request's share of the attention kernel's MAX_MM_RANGES.
    """
    if not requests:
        return
    encoder_inputs = scheduler_output.scheduled_encoder_inputs
    for req_id, num_scheduled in scheduler_output.num_scheduled_tokens.items():
        request = requests.get(req_id)
        if request is None:
            continue
        features = request.mm_features
        items = Counter(feature.modality for feature in features)
        ranges = sum(
            len(feature.mm_position.extract_embeds_range()) for feature in features
        )
        shapes = {
            idx: _mm_item_shapes(features[idx])
            for idx in encoder_inputs.get(req_id, [])
        }
        logger.error(
            "Dumping scheduled request: req_id=%s prompt_tokens=%d computed_tokens=%d "
            "scheduled_tokens=%d mm_items=%s mm_ranges=%d encoder_inputs=%s",
            req_id,
            request.num_prompt_tokens,
            request.num_computed_tokens,
            num_scheduled,
            dict(items),
            ranges,
            shapes,
        )


def _dump_engine_exception(
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
):
    logger.error(
        "Dumping input data for V1 LLM engine (v%s) with config: %s, ",
        VLLM_VERSION,
        config,
    )
    try:
        dump_obj = prepare_object_to_dump(scheduler_output)
        logger.error("Dumping scheduler output for model execution: %s", dump_obj)
        if scheduler_stats:
            logger.error("Dumping scheduler stats: %s", scheduler_stats)
    except Exception:
        logger.exception("Error preparing object to dump")
