# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The chunked penalties path must match the original dense implementation
bit for bit, and its extra memory must not grow with the number of rows."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.utils as layer_utils
import vllm.v1.sample.ops.penalties as penalties_ops
from vllm._custom_ops import apply_repetition_penalties
from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _no_pin_memory_without_cuda(monkeypatch):
    # Pinning needs a visible GPU, even when a driver is installed.
    if not torch.cuda.is_available():
        monkeypatch.setattr(penalties_ops, "is_pin_memory_available", lambda: False)


def _stock_apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """The penalties implementation before chunking (dense int64 tables)."""
    num_seqs, vocab_size = logits.shape
    output_tokens_t = make_tensor_with_pad(
        output_token_ids, pad=vocab_size, device=logits.device, dtype=torch.int64
    )
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_token_ids, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_t, vocab_size, num_seqs
    )
    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def _penalties(n: int, gen: torch.Generator, device: str):
    def pick(choices: list[float]) -> torch.Tensor:
        idx = torch.randint(0, len(choices), (n,), generator=gen)
        return torch.tensor(choices, dtype=torch.float32)[idx].to(device)

    # Include the neutral values so some rows have no effective penalty.
    presence = pick([0.0, 0.0, 0.5, -0.3, 1.7])
    frequency = pick([0.0, 0.0, 0.25, -1.1, 2.0])
    repetition = pick([1.0, 1.0, 1.05, 1.3, 0.8])
    return presence, frequency, repetition


def _make_batch(
    num_reqs: int,
    vocab_size: int,
    max_prompt_len: int,
    max_output_len: int,
    gen: torch.Generator,
    device: str,
):
    # Small token ranges so tokens repeat and counts exceed 1.
    hot = min(vocab_size, 64)
    prompts = []
    for i in range(num_reqs):
        n = int(torch.randint(1, max_prompt_len + 1, (1,), generator=gen))
        ids = torch.randint(0, vocab_size, (n,), generator=gen)
        ids[: n // 2] = torch.randint(0, hot, (n // 2,), generator=gen)
        if i == 0:
            ids[0] = vocab_size - 1
        prompts.append(ids.tolist())
    prompt_token_ids = make_tensor_with_pad(
        prompts, pad=vocab_size, device=device, dtype=torch.int64
    )
    outputs = []
    for i in range(num_reqs):
        n = 0 if i % 5 == 1 else int(torch.randint(0, max_output_len + 1, (1,)))
        ids = torch.randint(0, hot, (n,), generator=gen).tolist()
        # Async scheduling placeholders.
        if i % 4 == 2 and ids:
            ids[-1] = -1
        outputs.append(ids)
    return prompt_token_ids, outputs


def _make_logits(num_rows: int, vocab_size: int, gen: torch.Generator, device: str):
    logits = torch.randn(num_rows, vocab_size, generator=gen) * 5
    logits[:, :3] = torch.tensor([0.0, -0.0, float("-inf")])
    return logits.to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk_bytes", [1, 20_000, 1 << 30])
@pytest.mark.parametrize("vocab_size", [1000, 262144])
def test_sampler_penalties_match_stock(device, chunk_bytes, vocab_size, monkeypatch):
    monkeypatch.setattr(layer_utils, "PENALTIES_CHUNK_BYTES", chunk_bytes)
    gen = torch.Generator().manual_seed(vocab_size + chunk_bytes)
    num_reqs = 13
    prompt_token_ids, outputs = _make_batch(num_reqs, vocab_size, 40, 30, gen, device)
    presence, frequency, repetition = _penalties(num_reqs, gen, device)
    logits = _make_logits(num_reqs, vocab_size, gen, device)

    expected = _stock_apply_all_penalties(
        logits.clone(), prompt_token_ids, presence, frequency, repetition, outputs
    )
    metadata = SimpleNamespace(
        no_penalties=False,
        prompt_token_ids=prompt_token_ids,
        presence_penalties=presence,
        frequency_penalties=frequency,
        repetition_penalties=repetition,
    )
    actual = Sampler.apply_penalties(logits.clone(), metadata, outputs)
    _assert_bitwise_equal(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk_bytes", [1, 20_000, 1 << 30])
@pytest.mark.parametrize("vocab_size", [1000, 262144])
def test_rejection_sampler_penalties_match_stock(
    device, chunk_bytes, vocab_size, monkeypatch
):
    monkeypatch.setattr(layer_utils, "PENALTIES_CHUNK_BYTES", chunk_bytes)
    gen = torch.Generator().manual_seed(7 * vocab_size + chunk_bytes)
    num_reqs = 9
    prompt_token_ids, outputs = _make_batch(num_reqs, vocab_size, 40, 30, gen, device)
    # Up to k=4 drafts per request, one request without drafts.
    spec_token_ids = []
    for i in range(num_reqs):
        k = 0 if i == 3 else int(torch.randint(1, 5, (1,), generator=gen))
        spec_token_ids.append(torch.randint(0, 64, (k,), generator=gen).tolist())
    num_draft_tokens = [len(s) for s in spec_token_ids]
    rows_outputs = RejectionSampler._combine_outputs_with_spec_tokens(
        outputs, spec_token_ids
    )
    repeat_indices = torch.arange(num_reqs).repeat_interleave(
        torch.tensor(num_draft_tokens)
    )
    repeat_indices = repeat_indices.to(device)
    presence, frequency, repetition = _penalties(num_reqs, gen, device)
    logits = _make_logits(sum(num_draft_tokens), vocab_size, gen, device)

    expected = _stock_apply_all_penalties(
        logits.clone(),
        prompt_token_ids[repeat_indices],
        presence[repeat_indices],
        frequency[repeat_indices],
        repetition[repeat_indices],
        rows_outputs,
    )
    metadata = SimpleNamespace(
        no_penalties=False,
        prompt_token_ids=prompt_token_ids,
        presence_penalties=presence,
        frequency_penalties=frequency,
        repetition_penalties=repetition,
    )
    actual = RejectionSampler.apply_penalties(
        logits.clone(), metadata, None, repeat_indices, rows_outputs
    )
    _assert_bitwise_equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("chunk_bytes", [1, 1 << 30])
def test_non_contiguous_logits_match_stock(chunk_bytes, monkeypatch):
    # Stock uses the torch fallback for non-contiguous logits; single-row
    # chunks of such a tensor are contiguous but must keep using it.
    monkeypatch.setattr(layer_utils, "PENALTIES_CHUNK_BYTES", chunk_bytes)
    gen = torch.Generator().manual_seed(3)
    num_reqs, vocab_size = 6, 1000
    prompt_token_ids, outputs = _make_batch(num_reqs, vocab_size, 20, 20, gen, "cuda")
    presence, frequency, repetition = _penalties(num_reqs, gen, "cuda")
    base = _make_logits(num_reqs, vocab_size + 8, gen, "cuda")

    expected_base = base.clone()
    expected = _stock_apply_all_penalties(
        expected_base[:, :vocab_size],
        prompt_token_ids,
        presence,
        frequency,
        repetition,
        outputs,
    )
    actual_base = base.clone()
    actual_logits = actual_base[:, :vocab_size]
    assert not actual_logits.is_contiguous()
    apply_all_penalties(
        actual_logits, prompt_token_ids, presence, frequency, repetition, outputs
    )
    _assert_bitwise_equal(actual_base, expected_base)
    _assert_bitwise_equal(actual_logits.contiguous(), expected.contiguous())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_penalties_peak_memory_is_bounded():
    # Prod shape that crashed: 256 requests x 4 draft tokens, 262k vocab.
    num_reqs, k, vocab_size, prompt_len = 256, 4, 262144, 4096
    num_rows = num_reqs * k
    device = "cuda"
    free, _ = torch.cuda.mem_get_info()
    if free < 4 * 1024**3:
        pytest.skip("needs 4 GiB of free GPU memory")

    gen = torch.Generator(device=device).manual_seed(0)
    prompt_token_ids = torch.randint(
        0, vocab_size, (num_reqs, prompt_len), device=device, generator=gen
    )
    outputs = [list(range(i % 512)) for i in range(num_rows)]
    repeat_indices = torch.arange(num_reqs, device=device).repeat_interleave(k)
    ones = torch.ones(num_rows, device=device)
    logits = torch.randn(num_rows, vocab_size, device=device, generator=gen)

    torch.accelerator.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    apply_all_penalties(
        logits,
        prompt_token_ids,
        ones * 0.1,
        ones * 0.1,
        ones * 1.1,
        outputs,
        prompt_row_indices=repeat_indices,
    )
    torch.accelerator.synchronize()
    extra = torch.cuda.max_memory_allocated() - before

    # The dense implementation needs > 3 GiB here (2 GiB per int64 table).
    assert extra <= layer_utils.PENALTIES_CHUNK_BYTES * 1.05, extra
