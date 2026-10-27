# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.utils import (
    apply_penalties,
    get_penalties_rows_per_chunk,
)
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
    prompt_row_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Applies presence, frequency and repetition penalties to the logits.

    If prompt_row_indices is given, logits row i uses the prompt in row
    prompt_row_indices[i] of prompt_token_ids (e.g. one row per draft token
    with spec decode); otherwise row i uses row i.

    The prompt and output token tensors are built per chunk of rows, so the
    memory used on top of the logits is bounded regardless of the batch size.
    """
    num_rows, vocab_size = logits.shape
    max_output_len = max((len(ids) for ids in output_token_ids), default=0)
    # int64 prompt and output token ids are allocated per row of a chunk.
    token_bytes_per_row = (prompt_token_ids.shape[1] + max_output_len) * 8
    chunk_size = get_penalties_rows_per_chunk(vocab_size, token_bytes_per_row)
    # Decided for the whole tensor, not per chunk (see apply_penalties).
    use_repetition_penalties_kernel = logits.is_cuda and logits.is_contiguous()

    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        if prompt_row_indices is None:
            prompt_tokens_t = prompt_token_ids[start:end]
        else:
            prompt_tokens_t = prompt_token_ids[prompt_row_indices[start:end]]
        output_tokens_t = _convert_to_tensors(
            output_token_ids[start:end], vocab_size, logits.device
        )

        # In the async scheduling case, rows that won't have penalties applied
        # may contain -1 placeholder token ids. We must replace these with
        # valid token ids so that the scatter done in apply_penalties is valid.
        # NOTE(nick): The penalties implementation is currently quite
        # inefficient and will be reworked anyhow.
        output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

        apply_penalties(
            logits[start:end],
            prompt_tokens_t,
            output_tokens_t,
            presence_penalties[start:end],
            frequency_penalties[start:end],
            repetition_penalties[start:end],
            use_repetition_penalties_kernel=use_repetition_penalties_kernel,
        )
    return logits


def _convert_to_tensors(
    output_token_ids: list[list[int]], vocab_size: int, device: torch.device
) -> torch.Tensor:
    """
    Convert the different list data structures to tensors.
    """
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    )
    return output_tokens_tensor.to(device, non_blocking=True)
