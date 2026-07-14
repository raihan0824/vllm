# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.config import ModelConfig, VllmConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.inputs import EngineInput, PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor

if TYPE_CHECKING:
    from vllm.v1.engine import PauseMode


@dataclass
class StreamingInput:
    """Input data for a streaming generation request.

    This is used with generate() to support multi-turn streaming sessions
    where inputs are provided via an async generator.
    """

    prompt: EngineInput
    sampling_params: SamplingParams | None = None


class EngineClient(ABC):
    """Protocol class for Clients to Engine"""

    vllm_config: VllmConfig
    model_config: ModelConfig
    renderer: BaseRenderer
    input_processor: InputProcessor

    @property
    @abstractmethod
    def is_running(self) -> bool: ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool: ...

    @property
    @abstractmethod
    def errored(self) -> bool: ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException: ...

    @abstractmethod
    def generate(
        self,
        prompt: EngineCoreRequest
        | PromptType
        | EngineInput
        | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    @abstractmethod
    def encode(
        self,
        prompt: PromptType | EngineInput,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        ...

    @abstractmethod
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request,
                        or an iterable of such ids.
        """
        ...

    @abstractmethod
    async def notify_kv_transfer_request_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> None:
        """Notify the engine that a KV-transfer request was rejected before
        engine admission, so connector-side cleanup can run (e.g. free
        prefill blocks pinned on the P node).
        """
        ...

    @abstractmethod
    async def is_tracing_enabled(self) -> bool: ...

    @abstractmethod
    async def do_log_stats(self) -> None: ...

    @abstractmethod
    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    @abstractmethod
    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def stop_profile(self) -> None:
        """Stop profiling the engine"""
        ...

    @abstractmethod
    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""
        ...

    @abstractmethod
    async def reset_encoder_cache(self) -> None:
        """Reset the encoder cache"""
        ...

    @abstractmethod
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache and optionally any configured connector cache"""
        ...

    @abstractmethod
    async def sleep(self, level: int = 1, mode: "PauseMode" = "abort") -> None:
        """Sleep the engine"""
        ...

    @abstractmethod
    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the engine"""
        ...

    @abstractmethod
    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        ...

    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        ...

    @abstractmethod
    async def pause_generation(
        self,
        *,
        mode: "PauseMode" = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause new generation/encoding requests.

        Args:
            mode: How to handle in-flight requests:
                - ``"abort"``: Abort all in-flight requests immediately
                  and return partial results with "abort" reason (default).
                - ``"wait"``: Wait for in-flight requests to complete.
                - ``"keep"``: Freeze requests in queue; they resume on
                  :meth:`resume_generation`.
            wait_for_inflight_requests: DEPRECATED. Use ``mode="wait"`` instead.
            clear_cache: DEPRECATED. Whether to clear KV and prefix caches
                after draining.
        """
        ...

    @abstractmethod
    async def resume_generation(self) -> None:
        """Resume accepting generation/encoding requests."""
        ...

    @abstractmethod
    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        ...

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown the engine with optional timeout."""
        ...

    def admission_control_enabled(self) -> bool:
        """Whether backpressure/admission control is active for this engine.

        When ``False`` (the default), frontends skip all reservation work and
        behavior is identical to an unbounded queue. Enabled by setting
        ``--max-waiting-requests`` and/or ``--admission-max-kv-usage``.
        """
        return False

    def try_reserve_request_slot(self) -> str | None:
        """Synchronously admit a request under backpressure (admission control).

        Called by the frontend before any tokenization or prefill work. If
        the engine can accept the request, a slot is reserved (an in-flight
        counter is incremented) and ``None`` is returned; the caller MUST
        later call :meth:`release_request_slot` exactly once when the request
        finishes. If the engine is overloaded, a short machine-readable
        rejection reason (e.g. ``queue_full``, ``kv_pressure``) is returned
        and no slot is reserved.

        The reservation is synchronous (no ``await`` before the counter is
        bumped) so that a simultaneous burst of requests sees each other's
        reservations and cannot collectively overshoot the bound. The default
        implementation always admits, preserving the historical unbounded-queue
        behavior.
        """
        return None

    def release_request_slot(self) -> None:  # noqa: B027
        """Release a slot previously reserved by :meth:`try_reserve_request_slot`."""
        pass

    def try_admit_prompt(self, num_prompt_tokens: int) -> str | None:
        """Length-based admission check (admission control), post-tokenization.

        Called by the serving layer once the prompt token count is known,
        before any prefill work. Returns ``None`` to proceed, or a short
        rejection reason (e.g. ``long_prompt``) to reject with HTTP 429.
        Unlike :meth:`try_reserve_request_slot`, this neither reserves nor
        releases anything. The default implementation always admits.
        """
        return None

    def record_request_rejected(self, reason: str) -> None:  # noqa: B027
        """Record that an incoming request was rejected before admission.

        Args:
            reason: Short machine-readable reason label (e.g. ``queue_full``)
                used for the ``vllm:num_requests_rejected_total`` metric.
        """
        pass

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ) -> None:
        """Scale the engine"""
        raise NotImplementedError

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """Perform a collective RPC call to the given path."""
        raise NotImplementedError

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get supported tasks"""
        raise NotImplementedError

    async def init_weight_transfer_engine(
        self, init_request: WeightTransferInitRequest
    ) -> None:
        """Initialize weight transfer for RL training."""
        raise NotImplementedError

    async def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        """Start a new weight update."""
        raise NotImplementedError

    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:
        """Batched weight update for RL training."""
        raise NotImplementedError

    async def finish_weight_update(self) -> None:
        """Finish the current weight update."""
        raise NotImplementedError
