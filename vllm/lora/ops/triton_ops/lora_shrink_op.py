# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

import torch

from vllm import envs
from vllm.lora.ops.triton_ops.kernel_utils import do_shrink_kernel, mm_k
from vllm.lora.ops.triton_ops.utils import (
    _get_lora_a_ptr,
    get_lora_op_configs,
    supports_pdl,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# Deterministic split-K for the batch-invariant shrink path. 0 keeps the
# existing single-pass (S1) kernel; 8 enables the fixed two-pass kernel
# below. envs.py rejects any other value, so this is read once at import
# time and does not vary per call.
_TWO_PASS_SPLIT_K = envs.VLLM_LORA_DETERMINISTIC_SPLIT_K
if _TWO_PASS_SPLIT_K:
    if not envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError(
            "VLLM_LORA_DETERMINISTIC_SPLIT_K=8 requires VLLM_BATCH_INVARIANT=1; "
            "the deterministic two-pass reduction is only meaningful under "
            "batch-invariant execution."
        )
    if envs.VLLM_LORA_ENABLE_DUAL_STREAM:
        raise RuntimeError(
            "VLLM_LORA_DETERMINISTIC_SPLIT_K=8 cannot be combined with "
            "VLLM_LORA_ENABLE_DUAL_STREAM=1 in this release; disable one of "
            "the two before starting the process."
        )


def _get_two_pass_partials(
    output: torch.Tensor, num_slices: int, split_k: int, m: int, n: int
) -> torch.Tensor:
    # A plain per-call allocation, deliberately not a reused/pre-sized
    # workspace buffer. Earlier revisions tried to make a shared, named
    # scratch pool safe across every LoRA layer type, every stream (the
    # persistent compute stream and the independent MoE shared-experts
    # overlap stream), every CUDA-graph capture stream (decoder and, for
    # multimodal models, the encoder), both the V1 and V2 runners, and
    # engine restart within one process -- and each attempt to patch the
    # sharing surfaced a new corner where a captured graph's scratch
    # pointer could still move or a stream could still need to grow past a
    # lock. torch.empty() sidesteps all of that: CUDA graph capture routes
    # allocations made during capture into that graph's own private memory
    # pool (see https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs),
    # so a captured graph's partials pointer is stable for that graph's
    # lifetime without any cross-call bookkeeping, and concurrent streams
    # each get their own independent allocation, never aliasing. The
    # tradeoff is a small per-call allocation instead of a reused buffer;
    # PyTorch's caching allocator absorbs repeated same-size allocations
    # cheaply outside of capture.
    return torch.empty(
        (num_slices, split_k, m, n), dtype=torch.float32, device=output.device
    )


@triton.jit
def _lora_shrink_partial_kernel(
    input_ptr,
    lora_ptr,
    partial_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    partial_d0_stride,
    partial_d1_stride,
    partial_d2_stride,
    partial_d3_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SLICE_NUM: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K
    pid_m_n = pid_sk_m_n // SPLIT_K
    num_pid_in_group = GROUP_SIZE_M * cta_n_num
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(cta_m_num - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)
    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return

    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (
        token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    )
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
    else:
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(input_ptr.dtype.element_ty)
        )

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptr = (
        input_ptr
        + ram[:, None].to(tl.int64) * input_d0_stride
        + offset_k[None, :] * input_d1_stride
    )
    b_ptr = (
        cur_lora_ptr
        + lora_d0_stride * lora_id
        + rbn[None, :] * lora_d1_stride
        + offset_k[:, None] * lora_d2_stride
    )
    accumulator = mm_k(
        a_ptr,
        b_ptr,
        input_d1_stride,
        lora_d2_stride,
        offset_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        False,
        cur_lora_ptr.dtype.element_ty,
        False,
        base_k=pid_sk * BLOCK_K,
    )

    offset_cm = tl.arange(0, BLOCK_M)
    partial_out = (
        partial_ptr
        + slice_id * partial_d0_stride
        + pid_sk * partial_d1_stride
        + ram[:, None] * partial_d2_stride
        + offset_n[None, :] * partial_d3_stride
    )
    mask = (offset_cm[:, None] < cta_m_len) & (offset_n[None, :] < N)
    tl.store(partial_out, accumulator, mask=mask)


@triton.jit
def _lora_shrink_reduce_kernel(
    partial_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scaling,
    partial_d0_stride,
    partial_d1_stride,
    partial_d2_stride,
    partial_d3_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid_m_n = tl.program_id(axis=0)
    num_pid_in_group = GROUP_SIZE_M * cta_n_num
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(cta_m_num - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)
    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return

    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (
        token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    )
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_cm = tl.arange(0, BLOCK_M)
    mask = (offset_cm[:, None] < cta_m_len) & (offset_n[None, :] < N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split_id in range(SPLIT_K):
        partial_in = (
            partial_ptr
            + slice_id * partial_d0_stride
            + split_id * partial_d1_stride
            + ram[:, None] * partial_d2_stride
            + offset_n[None, :] * partial_d3_stride
        )
        accumulator += tl.load(partial_in, mask=mask, other=0.0)

    output = (
        out_ptr
        + slice_id * output_d0_stride
        + ram[:, None] * output_d1_stride
        + offset_n[None, :] * output_d2_stride
    )
    tl.store(output, accumulator * scaling, mask=mask)


@triton.jit
def _lora_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    M,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scaling,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K

    pid_m_n = pid_sk_m_n // SPLIT_K
    num_pid_in_group = GROUP_SIZE_M * cta_n_num
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(cta_m_num - first_pid_m, GROUP_SIZE_M)

    # Column-major ordering within groups for better cache reuse
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        # Early exit for the no-lora case.
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # Early exit CTA.
        return

    # num rows this CTA should process.
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    # Identify all rows that this CTA should process.
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (
        token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    )
    # Load all relevant row indices.
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_shrink_kernel(
        pid_n,
        pid_sk,
        slice_id,
        lora_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        cta_m_len,
        ram,  # array identifying the rows of Input ptr to operate on
        # input strides
        input_d0_stride,
        input_d1_stride,
        # lora strides
        lora_d0_stride,
        lora_d1_stride,
        lora_d2_stride,
        # output strides
        output_d0_stride,
        output_d1_stride,
        output_d2_stride,
        scaling,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        SLICE_NUM,
        USE_GDC,
    )


@torch.inference_mode()
def _lora_shrink(
    inputs: torch.Tensor,  #  shape [num_tokens, hidden_size]
    lora_a_weights: list[torch.Tensor],  # shape [num_loras, lora_rank, hidden_size]
    output_tensor: torch.Tensor,  # shape [num_slices, num_tokens, lora_rank]
    token_lora_mapping: torch.Tensor,  # shape [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # shape [num_tokens]
    num_tokens_per_lora: torch.Tensor,  # shape [max-loras + 1]
    lora_token_start_loc: torch.Tensor,  # shape [max-loras + 2]
    lora_ids: torch.Tensor,  # shape [max-loras + 1]
    no_lora_flag_cpu: torch.Tensor,  # shape [1]
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    scaling: float,
) -> None:
    """Args:
    inputs (torch.Tensor): Input tensor
    lora_a_weights (list[torch.Tensor]): LoRA weights
    output_tensor (torch.Tensor): output tensor
    token_lora_mapping (torch.Tensor): A tensor mapping each input token
        to the lora-id related to that token. A value of -1 indicates that
        LoRA doesn't apply to that token.
    token_indices_sorted_by_lora_ids (torch.Tensor): Row/Token indices from
        the A matrix grouped by LoRA IDs.
    num_tokens_per_lora (torch.Tensor): num_tokens_per_lora[i] is the number
        of tokens that are to be processed by LoRA ID lora_ids[i]
    lora_token_start_loc (torch.Tensor): A cumulative sum of
        num_tokens_per_lora. lora_token_start_loc[0] is always 0 so that
        lora_token_start_loc[i], along with num_tokens_per_lora[i]
        identifies the region in token_indices_sorted_by_lora_ids that
        LoRA lora_ids[i] should process.
    lora_ids (torch.Tensor): LoRA ids to process.
    no_lora_flag_cpu (torch.Tensor): A CPU tensor of size 1, that indicates
        if there are any requests that require LoRA.
    num_active_loras (torch.Tensor): A CPU tensor of size 1, containing the
        number of active LoRAs. Stored as a tensor (not int) so
        torch.compile treats it as dynamic rather than a constant.
    scaling (float): Scaling factor.

    """
    assert no_lora_flag_cpu.numel() == 1
    if inputs.size(0) == 0:
        # An empty first call has no work to do.
        return
    partials: torch.Tensor | None = None
    if _TWO_PASS_SPLIT_K:
        partials = _get_two_pass_partials(
            output_tensor,
            output_tensor.size(0),
            _TWO_PASS_SPLIT_K,
            inputs.size(0),
            output_tensor.size(-1),
        )
    if no_lora_flag_cpu.item():
        # None of the inputs require LoRA.
        return

    assert inputs.dtype == lora_a_weights[0].dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    for weight in lora_a_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]

    assert inputs.size(1) == lora_a_weights[0].size(-1)
    inputs = inputs.contiguous()
    assert output_tensor.is_contiguous()

    # metadata sanity check
    M = inputs.size(0)
    assert token_lora_mapping.size(0) == M
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

    output_tensor.zero_()

    (lora_ptr_tensor, lora_strides_d0, lora_strides_d1, lora_strides_d2) = (
        _get_lora_a_ptr(lora_a_weights, inputs.device)
    )
    N, K = lora_a_weights[0].shape[-2:]  # K=hidden_size,N=rank
    NUM_SLICES = len(lora_a_weights)
    MAX_LORAS = lora_ids.size(0)

    # Triton kernel configs
    kernel_config = get_lora_op_configs(
        "shrink",
        max_loras=MAX_LORAS,
        batch=M,
        hidden_size=K,
        rank=N,
        num_slices=NUM_SLICES,
    )
    BLOCK_M = kernel_config["block_m"]
    BLOCK_N = kernel_config["block_n"]
    BLOCK_K = kernel_config["block_k"]
    SPLIT_K = kernel_config["split_k"]
    NUM_WARPS = kernel_config["num_warps"]
    NUM_STAGES = kernel_config["num_stages"]
    NUM_CTAS = kernel_config["num_ctas"]
    GROUP_SIZE_M = kernel_config.get("group_size_m", 8)

    # Deterministic split-K path (Section 2 of the LoRA split-K plan). Each
    # split writes to a unique FP32 scratch tile, then a second kernel
    # reduces splits in a fixed order. Batch invariance requires every M to
    # preserve the same logical K partition and fixed reduction order; do
    # not dispatch to S1 based on M, and do not restore the old M<=128
    # S8/S1 gate.
    if _TWO_PASS_SPLIT_K:
        BLOCK_K = 256
        SPLIT_K = _TWO_PASS_SPLIT_K
    EVEN_K = K % (BLOCK_K * SPLIT_K) == 0  # type: ignore

    # TODO (varun): This grid formulation maximizes parallelization at the
    # cost of wasteful thread block launch when only few of the input tokens
    # require LoRA. This might not be the best in all cases.
    grid = (
        SPLIT_K * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        NUM_SLICES,
        num_active_loras.item(),
    )

    if _TWO_PASS_SPLIT_K:
        assert partials is not None
        _lora_shrink_partial_kernel[grid](
            inputs,
            lora_ptr_tensor,
            partials,
            M,
            N,
            K,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            inputs.stride(0),
            inputs.stride(1),
            lora_strides_d0,
            lora_strides_d1,
            lora_strides_d2,
            partials.stride(0),
            partials.stride(1),
            partials.stride(2),
            partials.stride(3),
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            EVEN_K,
            SPLIT_K,
            GROUP_SIZE_M,
            NUM_SLICES,
            num_warps=NUM_WARPS,
            num_ctas=NUM_CTAS,
            num_stages=NUM_STAGES,
        )
        reduce_grid = (
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            NUM_SLICES,
            num_active_loras.item(),
        )
        _lora_shrink_reduce_kernel[reduce_grid](
            partials,
            output_tensor,
            M,
            N,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            scaling,
            partials.stride(0),
            partials.stride(1),
            partials.stride(2),
            partials.stride(3),
            output_tensor.stride(0),
            output_tensor.stride(1),
            output_tensor.stride(2),
            BLOCK_M,
            BLOCK_N,
            SPLIT_K,
            GROUP_SIZE_M,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        return

    # PDL only works when dual-stream is being used.
    use_gdc = supports_pdl(inputs.device) and envs.VLLM_LORA_ENABLE_DUAL_STREAM
    _lora_shrink_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor.stride(2),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        GROUP_SIZE_M,
        NUM_SLICES,
        use_gdc,
        num_warps=NUM_WARPS,
        num_ctas=NUM_CTAS,
        num_stages=NUM_STAGES,
        launch_pdl=use_gdc,
    )

    return


try:
    direct_register_custom_op(
        op_name="lora_shrink",
        op_func=_lora_shrink,
        mutates_args=["output_tensor"],
    )
    lora_shrink = torch.ops.vllm.lora_shrink

except AttributeError:
    lora_shrink = _lora_shrink
