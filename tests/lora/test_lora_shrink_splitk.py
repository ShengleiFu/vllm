# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the LoRA deterministic split-K shrink kernel
(VLLM_LORA_DETERMINISTIC_SPLIT_K=8 under VLLM_BATCH_INVARIANT=1).

These tests run the split-K path in *subprocesses* because the kernel
module reads VLLM_LORA_DETERMINISTIC_SPLIT_K / VLLM_BATCH_INVARIANT once
at import time; setting the env inside the pytest process after vllm has
already been imported would not change the dispatch.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="split-K shrink kernel requires CUDA"
)


@pytest.fixture(autouse=True)
def cleanup_fixture():
    """Override conftest's cleanup_fixture — every test here runs its GPU
    work in a subprocess, so there is no in-process dist/model state to
    tear down, and the conftest teardown otherwise requires `ray`.
    """
    yield


@pytest.fixture(autouse=True)
def dynamo_reset():
    """Override conftest's dynamo_reset — not needed for subprocess tests."""
    yield


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(script: str, env_overrides: dict[str, str], timeout: int = 180) -> dict:
    """Run `script` in a fresh subprocess with env set before interpreter start.

    The script must print a single JSON line to stdout as its result.
    """
    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"subprocess produced no output; stderr:\n{proc.stderr}"
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"could not parse subprocess JSON output: {lines[-1]!r}\n"
            f"stderr:\n{proc.stderr}"
        ) from e


_S8_ENV = {"VLLM_BATCH_INVARIANT": "1", "VLLM_LORA_DETERMINISTIC_SPLIT_K": "8"}


# ---------------------------------------------------------------------------
# Numerical correctness against an independent CPU FP64 reference.
# ---------------------------------------------------------------------------

_NUMERIC_SCRIPT = textwrap.dedent(
    """
    import json
    import torch
    from vllm.lora.ops.triton_ops import lora_shrink
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.manual_seed(0)
    device = "cuda:0"
    init_workspace_manager(torch.device(device))

    K = {K}
    RANK = {RANK}
    NSLICES = {NSLICES}
    DTYPE = getattr(torch, "{DTYPE}")
    SCALING = {SCALING}
    NUM_LORAS = 2
    BATCH = 5
    SEQ_LENS = [3, 1, 7, 2, 4]
    assert sum(SEQ_LENS) and len(SEQ_LENS) == BATCH
    total_tokens = sum(SEQ_LENS)

    inputs = torch.randn((total_tokens, K), dtype=DTYPE, device=device)
    lora_weights = [
        torch.randn((NUM_LORAS, RANK, K), dtype=DTYPE, device=device)
        for _ in range(NSLICES)
    ]

    token_lora_mapping = torch.repeat_interleave(
        torch.randint(0, NUM_LORAS, (BATCH,), device=device),
        torch.tensor(SEQ_LENS, device=device),
    ).to(torch.int32)

    lora_meta = LoRAKernelMeta.make(
        max_loras=NUM_LORAS, max_num_tokens=total_tokens, device=device
    )
    lora_meta.prepare_tensors(token_lora_mapping)

    out = torch.zeros((NSLICES, total_tokens, RANK), dtype=torch.float32, device=device)
    _LORA_A_PTR_DICT.clear()
    lora_shrink(
        inputs,
        lora_weights,
        out,
        *lora_meta.meta_args(token_nums=total_tokens, specialize_active_lora=False),
        SCALING,
    )
    torch.accelerator.synchronize()

    # Independent CPU FP64 reference: per-token, per-slice matmul loop.
    inputs_cpu = inputs.double().cpu()
    mapping_cpu = token_lora_mapping.cpu()
    ref = torch.zeros((NSLICES, total_tokens, RANK), dtype=torch.float64)
    for s in range(NSLICES):
        w_cpu = lora_weights[s].double().cpu()
        for lid in range(NUM_LORAS):
            mask = mapping_cpu == lid
            if not mask.any():
                continue
            ref[s, mask] = SCALING * (inputs_cpu[mask] @ w_cpu[lid].T)

    out_cpu = out.double().cpu()
    finite = bool(torch.isfinite(out_cpu).all().item())
    diff = (out_cpu - ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rtol = atol = 0.005
    close = bool(torch.allclose(out_cpu, ref, rtol=rtol, atol=atol))
    print(json.dumps({{
        "finite": finite,
        "max_err": max_err,
        "mean_err": mean_err,
        "close": close,
    }}))
    """
)


@pytest.mark.parametrize(
    "K,RANK,NSLICES,DTYPE,SCALING",
    [
        (2560, 8, 3, "bfloat16", 0.73),
        (9728, 16, 1, "float16", 1.0),
        (4097, 64, 1, "bfloat16", 0.5),
        (1024, 8, 1, "float16", 1.0),  # K < S*BK=2048: empty K-partition
    ],
    ids=["K2560_r8_3slices", "K9728_r16", "K4097_r64_tail", "K1024_r8_empty_partition"],
)
def test_lora_shrink_splitk_numerical_correctness(K, RANK, NSLICES, DTYPE, SCALING):
    script = _NUMERIC_SCRIPT.format(
        K=K, RANK=RANK, NSLICES=NSLICES, DTYPE=DTYPE, SCALING=SCALING
    )
    result = _run(script, _S8_ENV)
    assert result["finite"], f"non-finite output: {result}"
    print(
        f"[{DTYPE} K={K} rank={RANK} nslices={NSLICES}] "
        f"max_err={result['max_err']:.6f} mean_err={result['mean_err']:.6f}"
    )
    assert result["close"], f"numerical mismatch vs CPU FP64 reference: {result}"


# ---------------------------------------------------------------------------
# Self batch-invariance: the target row's shrink+expand output must not
# depend on what else is in the batch (composition, position, other
# adapters), and must be byte-identical across repeated launches.
# ---------------------------------------------------------------------------

_SELF_BI_SCRIPT = textwrap.dedent(
    """
    import json
    import torch
    from vllm.lora.ops.triton_ops import lora_shrink, lora_expand
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT
    from vllm.v1.worker.workspace import init_workspace_manager

    device = "cuda:0"
    init_workspace_manager(torch.device(device))

    # HIDDEN (=K) is chosen so every one of the 8 fixed splits (BLOCK_K=256)
    # gets at least one nonzero K-block: 2560 / 256 = 10 blocks, so all 8
    # splits contribute and two contribute twice. With HIDDEN=256 (a single
    # BK256 block) only split 0 would ever be nonzero, never exercising the
    # multi-partial reduction this kernel exists for.
    HIDDEN = 2560
    RANK = 16
    NUM_ADAPTERS = 3
    NSLICES = 1
    SCALING = 0.7
    DTYPE = torch.bfloat16
    REPEATS = 20

    g = torch.Generator(device=device).manual_seed(1234)

    lora_a = [
        torch.randn(
            (NUM_ADAPTERS, RANK, HIDDEN), dtype=DTYPE, device=device, generator=g
        )
    ]
    lora_b = [
        torch.randn(
            (NUM_ADAPTERS, HIDDEN, RANK), dtype=DTYPE, device=device, generator=g
        )
    ]

    TARGET_ADAPTER = 0
    target_input = torch.randn((1, HIDDEN), dtype=DTYPE, device=device, generator=g)
    target_residual = torch.randn((1, HIDDEN), dtype=DTYPE, device=device, generator=g)

    def run_once(M, target_pos, companion_adapter_ids):
        assert len(companion_adapter_ids) == M - 1
        inputs = torch.randn((M, HIDDEN), dtype=DTYPE, device=device, generator=g)
        inputs[target_pos] = target_input[0]
        mapping = torch.empty((M,), dtype=torch.int32, device=device)
        ci = 0
        for i in range(M):
            if i == target_pos:
                mapping[i] = TARGET_ADAPTER
            else:
                mapping[i] = companion_adapter_ids[ci]
                ci += 1
        residual = torch.randn((M, HIDDEN), dtype=DTYPE, device=device, generator=g)
        residual[target_pos] = target_residual[0]

        lora_meta = LoRAKernelMeta.make(
            max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
        )
        lora_meta.prepare_tensors(mapping)

        shrink_out = torch.zeros((NSLICES, M, RANK), dtype=torch.float32, device=device)
        _LORA_A_PTR_DICT.clear()
        lora_shrink(
            inputs, lora_a, shrink_out,
            *lora_meta.meta_args(token_nums=M, specialize_active_lora=False),
            SCALING,
        )
        # Snapshot the raw FP32 shrink output before it is rounded through
        # BF16 expand+add: rounding can hide a real FP32 reduction diff.
        shrink_target = shrink_out[:, target_pos].clone()

        expand_out = residual.clone()
        _LORA_B_PTR_DICT.clear()
        lora_expand(
            shrink_out, lora_b, expand_out,
            *lora_meta.meta_args(token_nums=M, specialize_active_lora=False),
            offset_start=0,
            add_inputs=True,
        )
        torch.accelerator.synchronize()
        return expand_out[target_pos].clone(), shrink_target

    configs = {
        "M1": (1, 0, []),
        "M127_first_same_adapter": (127, 0, [TARGET_ADAPTER] * 126),
        "M127_last_diff_adapters": (127, 126, [1, 2] * 63),
        "M128_first": (128, 0, [1] * 127),
        "M129_last": (129, 128, [2] * 128),
        "M31_mid": (31, 15, [0, 1, 2] * 10),
        "M32_mid": (32, 16, [0, 1, 2] * 10 + [1]),
        "M33_mid": (33, 16, [0, 1, 2] * 10 + [1, 2]),
        "M64_mixed": (64, 40, ([0, 1, 2] * 21)[:63]),
    }

    expand_ref, shrink_ref = run_once(*configs["M1"])

    results = {}
    for name, (M, pos, companions) in configs.items():
        outputs = [run_once(M, pos, companions) for _ in range(REPEATS)]
        expand_outs = [o[0] for o in outputs]
        shrink_outs = [o[1] for o in outputs]
        results[name] = {
            "expand_self_identical_across_repeats": all(
                torch.equal(expand_outs[0], o) for o in expand_outs[1:]
            ),
            "expand_matches_M1_reference": torch.equal(expand_outs[0], expand_ref),
            "shrink_self_identical_across_repeats": all(
                torch.equal(shrink_outs[0], o) for o in shrink_outs[1:]
            ),
            "shrink_matches_M1_reference": torch.equal(shrink_outs[0], shrink_ref),
        }

    # Two-adapter, three-slice point, plus its M=1 analog using the same
    # target adapter/weights/target row so it is a real reference, not a
    # repeat-only check.
    NSLICES3 = 3
    lora_a3 = [
        torch.randn(
            (NUM_ADAPTERS, RANK, HIDDEN), dtype=DTYPE, device=device, generator=g
        )
        for _ in range(NSLICES3)
    ]
    lora_b3 = [
        torch.randn(
            (NUM_ADAPTERS, HIDDEN, RANK), dtype=DTYPE, device=device, generator=g
        )
        for _ in range(NSLICES3)
    ]
    target_residual3 = torch.randn(
        (1, HIDDEN * NSLICES3), dtype=DTYPE, device=device, generator=g
    )

    def run_once_3slices(M, target_pos, mapping_list):
        mapping = torch.tensor(mapping_list, dtype=torch.int32, device=device)
        inputs = torch.randn((M, HIDDEN), dtype=DTYPE, device=device, generator=g)
        inputs[target_pos] = target_input[0]
        residual = torch.randn(
            (M, HIDDEN * NSLICES3), dtype=DTYPE, device=device, generator=g
        )
        residual[target_pos] = target_residual3[0]

        lora_meta3 = LoRAKernelMeta.make(
            max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
        )
        lora_meta3.prepare_tensors(mapping)

        shrink_out3 = torch.zeros(
            (NSLICES3, M, RANK), dtype=torch.float32, device=device
        )
        _LORA_A_PTR_DICT.clear()
        lora_shrink(
            inputs, lora_a3, shrink_out3,
            *lora_meta3.meta_args(token_nums=M, specialize_active_lora=False),
            SCALING,
        )
        shrink_target3 = shrink_out3[:, target_pos].clone()

        expand_out3 = residual.clone()
        _LORA_B_PTR_DICT.clear()
        lora_expand(
            shrink_out3, lora_b3, expand_out3,
            *lora_meta3.meta_args(token_nums=M, specialize_active_lora=False),
            offset_start=0,
            add_inputs=True,
        )
        torch.accelerator.synchronize()
        return expand_out3[target_pos].clone(), shrink_target3

    target_pos3_m1 = 0
    expand_ref3, shrink_ref3 = run_once_3slices(1, target_pos3_m1, [TARGET_ADAPTER])

    M3 = 50
    target_pos3 = 25
    mapping3 = [0 if i == target_pos3 else (1 if i % 2 == 0 else 0) for i in range(M3)]
    outputs3 = [run_once_3slices(M3, target_pos3, mapping3) for _ in range(REPEATS)]
    expand_outs3 = [o[0] for o in outputs3]
    shrink_outs3 = [o[1] for o in outputs3]
    results["dual_adapter_3slices"] = {
        "expand_self_identical_across_repeats": all(
            torch.equal(expand_outs3[0], o) for o in expand_outs3[1:]
        ),
        "expand_matches_M1_reference": torch.equal(expand_outs3[0], expand_ref3),
        "shrink_self_identical_across_repeats": all(
            torch.equal(shrink_outs3[0], o) for o in shrink_outs3[1:]
        ),
        "shrink_matches_M1_reference": torch.equal(shrink_outs3[0], shrink_ref3),
    }

    print(json.dumps(results))
    """
)


def test_lora_shrink_splitk_self_batch_invariance():
    result = _run(_SELF_BI_SCRIPT, _S8_ENV, timeout=300)
    failures = []
    for name, r in result.items():
        for key, ok in r.items():
            if not ok:
                failures.append(f"{name}: {key} failed")
    print(json.dumps(result, indent=2))
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Empty input and no-LoRA regression.
# ---------------------------------------------------------------------------

_EMPTY_INPUT_SCRIPT = textwrap.dedent(
    """
    import json
    import torch
    from vllm.lora.ops.triton_ops import lora_shrink
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
    )

    device = "cuda:0"
    init_workspace_manager(torch.device(device))

    HIDDEN = 128
    RANK = 8
    NUM_ADAPTERS = 2
    NSLICES = 1
    SCALING = 1.0
    DTYPE = torch.bfloat16

    lora_a = [torch.randn((NUM_ADAPTERS, RANK, HIDDEN), dtype=DTYPE, device=device)]

    results = {}

    # 1. First call with M=0 must not create a named workspace owner.
    inputs0 = torch.empty((0, HIDDEN), dtype=DTYPE, device=device)
    out0 = torch.zeros((NSLICES, 0, RANK), dtype=torch.float32, device=device)
    lora_meta0 = LoRAKernelMeta.make(
        max_loras=NUM_ADAPTERS, max_num_tokens=0, device=device
    )
    lora_meta0.prepare_tensors(torch.empty((0,), dtype=torch.int32, device=device))
    _LORA_A_PTR_DICT.clear()
    lora_shrink(
        inputs0, lora_a, out0,
        *lora_meta0.meta_args(token_nums=0, specialize_active_lora=False),
        SCALING,
    )
    named_owners_after_empty_call = list(
        current_workspace_manager()._named_workspaces.keys()
    )
    results["m0_no_named_owner_created"] = (named_owners_after_empty_call == [])

    # 2. All no-LoRA (M>0 but every row unassigned): workspace IS reserved
    # (per the runner-owned-scratch contract), output stays untouched/zero.
    M = 16
    inputs_noop = torch.randn((M, HIDDEN), dtype=DTYPE, device=device)
    out_noop = torch.full((NSLICES, M, RANK), -1.0, dtype=torch.float32, device=device)
    mapping_noop = torch.full((M,), -1, dtype=torch.int32, device=device)
    lora_meta_noop = LoRAKernelMeta.make(
        max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
    )
    lora_meta_noop.prepare_tensors(mapping_noop)
    _LORA_A_PTR_DICT.clear()
    lora_shrink(
        inputs_noop, lora_a, out_noop,
        *lora_meta_noop.meta_args(token_nums=M, specialize_active_lora=False),
        SCALING,
    )
    torch.accelerator.synchronize()
    # Contract: an all-no-LoRA call returns before touching output_tensor.
    results["all_no_lora_output_untouched"] = bool(
        torch.equal(
            out_noop,
            torch.full((NSLICES, M, RANK), -1.0, dtype=torch.float32, device=device),
        )
    )

    # 3. Mixed valid/-1 rows: valid rows get real values, -1 rows are
    # zeroed (output_tensor.zero_() runs before dispatch for the mixed case).
    mapping_mixed = torch.tensor(
        [0, -1, 1, -1, 0, 1, -1, 0] * 2, dtype=torch.int32, device=device
    )
    inputs_mixed = torch.randn((M, HIDDEN), dtype=DTYPE, device=device)
    out_mixed = torch.full((NSLICES, M, RANK), -1.0, dtype=torch.float32, device=device)
    lora_meta_mixed = LoRAKernelMeta.make(
        max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
    )
    lora_meta_mixed.prepare_tensors(mapping_mixed)
    _LORA_A_PTR_DICT.clear()
    lora_shrink(
        inputs_mixed, lora_a, out_mixed,
        *lora_meta_mixed.meta_args(token_nums=M, specialize_active_lora=False),
        SCALING,
    )
    torch.accelerator.synchronize()
    invalid_rows = (mapping_mixed == -1).cpu()
    out_mixed_cpu = out_mixed[0].cpu()
    results["mixed_invalid_rows_are_zero"] = bool(
        torch.all(out_mixed_cpu[invalid_rows] == 0.0).item()
    )
    results["mixed_valid_rows_nonzero"] = bool(
        torch.any(out_mixed_cpu[~invalid_rows] != 0.0).item()
    )

    # 4. Metadata swap between calls must not leak stale scratch. Checking
    # only that newly-invalid rows read zero is not enough: the wrapper
    # unconditionally calls output_tensor.zero_() before dispatch, so that
    # much is true even if the reducer never actually rewrites those rows.
    # Poison the raw named-workspace scratch before the second call (so a
    # missed write shows up as garbage, not an already-zero value), verify
    # the newly-*valid* rows against an independent CPU FP64 reference, and
    # round-trip back to the original mapping to confirm the buffer is
    # fully rewritten, not merely masked.
    mapping_a = torch.tensor(
        [0, 0, -1, -1, 0, 0, -1, -1] * 2, dtype=torch.int32, device=device
    )
    mapping_b = torch.tensor(
        [-1, -1, 1, 1, -1, -1, 1, 1] * 2, dtype=torch.int32, device=device
    )
    shared_out = torch.zeros((NSLICES, M, RANK), dtype=torch.float32, device=device)
    lora_meta_swap = LoRAKernelMeta.make(
        max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
    )

    lora_meta_swap.prepare_tensors(mapping_a)
    _LORA_A_PTR_DICT.clear()
    lora_shrink(
        inputs_mixed, lora_a, shared_out,
        *lora_meta_swap.meta_args(token_nums=M, specialize_active_lora=False),
        SCALING,
    )
    torch.accelerator.synchronize()
    first_call_snapshot = shared_out.clone()

    # Poison the two-pass named scratch's raw bytes so any row the
    # reducer fails to (re)write shows up as garbage instead of a
    # coincidentally-zero value.
    manager = current_workspace_manager()
    with torch.inference_mode():
        # The scratch buffer was allocated inside lora_shrink's own
        # @torch.inference_mode() context, so it is an inference tensor;
        # mutating it in place from ordinary (non-inference) code raises
        # RuntimeError. Poisoning it needs the same mode.
        for slots in manager._named_workspaces.values():
            for slot in slots:
                if slot is not None:
                    slot.fill_(0xFF)

    lora_meta_swap.prepare_tensors(mapping_b)
    _LORA_A_PTR_DICT.clear()
    lora_shrink(
        inputs_mixed, lora_a, shared_out,
        *lora_meta_swap.meta_args(token_nums=M, specialize_active_lora=False),
        SCALING,
    )
    torch.accelerator.synchronize()

    became_invalid = ((mapping_a != -1) & (mapping_b == -1)).cpu()
    became_valid = ((mapping_a == -1) & (mapping_b != -1)).cpu()
    second_call_cpu = shared_out[0].double().cpu()
    # Rows valid under mapping_a but invalid under mapping_b must now read 0,
    # not the stale nonzero value from the first call.
    results["metadata_swap_no_stale_scratch"] = bool(
        torch.all(second_call_cpu[became_invalid] == 0.0).item()
    )
    results["metadata_swap_output_finite"] = bool(
        torch.isfinite(second_call_cpu).all().item()
    )

    # Independent CPU FP64 reference for the rows that only became valid
    # under mapping_b -- this is the part the pre-fix test never checked.
    inputs_mixed_cpu = inputs_mixed.double().cpu()
    mapping_b_cpu = mapping_b.cpu()
    w_cpu = lora_a[0].double().cpu()
    ref_b = torch.zeros((M, RANK), dtype=torch.float64)
    for lid in range(NUM_ADAPTERS):
        mask = mapping_b_cpu == lid
        if mask.any():
            ref_b[mask] = SCALING * (inputs_mixed_cpu[mask] @ w_cpu[lid].T)
    results["metadata_swap_newly_valid_rows_correct"] = bool(
        torch.allclose(
            second_call_cpu[became_valid], ref_b[became_valid], rtol=0.005, atol=0.005
        )
    )

    # Round-trip: swap back to mapping_a and confirm the exact original
    # result reproduces, proving the buffer was legitimately recomputed
    # both times rather than the poisoning happening to be masked once.
    lora_meta_swap.prepare_tensors(mapping_a)
    _LORA_A_PTR_DICT.clear()
    lora_shrink(
        inputs_mixed, lora_a, shared_out,
        *lora_meta_swap.meta_args(token_nums=M, specialize_active_lora=False),
        SCALING,
    )
    torch.accelerator.synchronize()
    results["metadata_swap_roundtrip_matches_first_call"] = bool(
        torch.equal(shared_out, first_call_snapshot)
    )

    print(json.dumps(results))
    """
)


def test_lora_shrink_splitk_empty_input_and_no_lora():
    result = _run(_EMPTY_INPUT_SCRIPT, _S8_ENV)
    print(json.dumps(result, indent=2))
    failures = [name for name, ok in result.items() if not ok]
    assert not failures, f"failed checks: {failures}; full result: {result}"


# ---------------------------------------------------------------------------
# Workspace owner-key regression: the two-pass scratch's named-workspace
# owner must be keyed by the current CUDA stream, not just (ubatch, lane).
# The MoE shared-experts overlap stream (gated by
# VLLM_DISABLE_SHARED_EXPERTS_STREAM, independent of and on by default
# regardless of VLLM_LORA_ENABLE_DUAL_STREAM) can run a LoRA-adapted
# shared-experts layer concurrently with this stream inside the same
# (ubatch, lane) slot. Two distinct streams must therefore get distinct
# scratch, even though this means a stream used only transiently (e.g.
# memory/graph-memory profiling) keeps its slot allocated rather than being
# reclaimed once real capture starts on a different stream — an accepted,
# bounded memory cost documented in _get_two_pass_partials.
# ---------------------------------------------------------------------------

_STREAM_ISOLATION_SCRIPT = textwrap.dedent(
    """
    import json
    import torch
    from vllm.lora.ops.triton_ops import lora_shrink
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
    )

    device = "cuda:0"
    init_workspace_manager(torch.device(device))

    HIDDEN = 256
    RANK = 8
    NUM_ADAPTERS = 2
    NSLICES = 1
    SCALING = 1.0
    DTYPE = torch.bfloat16
    M = 8

    lora_a = [torch.randn((NUM_ADAPTERS, RANK, HIDDEN), dtype=DTYPE, device=device)]

    def run_on_stream(stream):
        inputs = torch.randn((M, HIDDEN), dtype=DTYPE, device=device)
        mapping = torch.randint(
            0, NUM_ADAPTERS, (M,), dtype=torch.int32, device=device
        )
        lora_meta = LoRAKernelMeta.make(
            max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
        )
        lora_meta.prepare_tensors(mapping)
        out = torch.zeros((NSLICES, M, RANK), dtype=torch.float32, device=device)
        with torch.cuda.stream(stream):
            stream.wait_stream(torch.cuda.current_stream())
            _LORA_A_PTR_DICT.clear()
            lora_shrink(
                inputs, lora_a, out,
                *lora_meta.meta_args(token_nums=M, specialize_active_lora=False),
                SCALING,
            )
        stream.synchronize()

    # Sequential streams (e.g. memory/graph-memory profiling followed by real
    # CUDA graph capture), both acting on ubatch 0 / lane 0.
    profiling_stream = torch.cuda.Stream(device=device)
    run_on_stream(profiling_stream)

    manager = current_workspace_manager()
    owner_keys_after_profiling = list(manager._named_workspaces.keys())
    ptr_after_profiling = (
        manager._named_workspaces[owner_keys_after_profiling[0]][0].data_ptr()
    )

    capture_stream = torch.cuda.Stream(device=device)
    run_on_stream(capture_stream)

    owner_keys_after_capture = list(manager._named_workspaces.keys())
    capture_key = next(
        k for k in owner_keys_after_capture if k != owner_keys_after_profiling[0]
    )
    ptr_after_capture = manager._named_workspaces[capture_key][0].data_ptr()

    print(json.dumps({
        "owner_keys_after_profiling": [repr(k) for k in owner_keys_after_profiling],
        "owner_keys_after_capture": [repr(k) for k in owner_keys_after_capture],
        "distinct_owner_per_stream": len(owner_keys_after_capture) == 2,
        "distinct_buffer_per_stream": ptr_after_profiling != ptr_after_capture,
    }))
    """
)


def test_lora_shrink_splitk_workspace_owner_isolated_per_stream():
    """Each distinct CUDA stream must get its own named-workspace scratch
    slot. A prior implementation collapsed the owner key to (ubatch, lane)
    only, which reused one slot across streams and let two concurrent
    lora_shrink calls on different streams (e.g. the MoE shared-experts
    overlap stream and the main stream) corrupt each other's partials; see
    test_lora_shrink_splitk_shared_experts_stream_isolation for the
    concurrent-corruption regression this owner key prevents.
    """
    result = _run(_STREAM_ISOLATION_SCRIPT, _S8_ENV)
    print(json.dumps(result, indent=2))
    assert result["distinct_owner_per_stream"], result
    assert result["distinct_buffer_per_stream"], result


# ---------------------------------------------------------------------------
# Concurrent-stream correctness regression (the P1 gap): the MoE
# shared-experts overlap stream (gated by VLLM_DISABLE_SHARED_EXPERTS_STREAM,
# independent of VLLM_LORA_ENABLE_DUAL_STREAM and on by default) can run a
# LoRA-adapted shared-experts layer concurrently with the main stream inside
# the *same* (ubatch, lane) slot -- unlike test_..._ordinary_concurrent_streams
# below, which isolates concurrent streams onto separate lanes and therefore
# does not exercise this same-lane hazard.
# ---------------------------------------------------------------------------

_SHARED_EXPERTS_STREAM_SCRIPT = textwrap.dedent(
    """
    import json
    import torch
    from vllm.lora.ops.triton_ops import lora_shrink
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT
    from vllm.v1.worker.workspace import init_workspace_manager

    device = "cuda:0"
    init_workspace_manager(torch.device(device))

    HIDDEN = 2560
    RANK = 16
    NUM_ADAPTERS = 2
    NSLICES = 1
    SCALING = 0.9
    DTYPE = torch.bfloat16
    M = 32
    ROUNDS = 50

    lora_a = [torch.randn((NUM_ADAPTERS, RANK, HIDDEN), dtype=DTYPE, device=device)]

    def run_serial(inputs, mapping):
        meta = LoRAKernelMeta.make(
            max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
        )
        meta.prepare_tensors(mapping)
        out = torch.zeros((NSLICES, M, RANK), dtype=torch.float32, device=device)
        _LORA_A_PTR_DICT.clear()
        lora_shrink(
            inputs, lora_a, out,
            *meta.meta_args(token_nums=M, specialize_active_lora=False),
            SCALING,
        )
        torch.accelerator.synchronize()
        return out.clone()

    # Mirrors FusedMoE SharedExperts.maybe_forward_async: base work stays on
    # the main/default stream while a second CUDA stream runs concurrently,
    # both within the *same* (ubatch, lane) slot -- no use_workspace_lane
    # separation, since production code has no such separation between the
    # main stream and the shared-experts overlap stream.
    aux_stream = torch.cuda.Stream(device=device)

    mismatches = []
    for round_idx in range(ROUNDS):
        g = torch.Generator(device=device).manual_seed(2000 + round_idx)
        inputs_main = torch.randn((M, HIDDEN), dtype=DTYPE, device=device, generator=g)
        mapping_main = torch.randint(
            0, NUM_ADAPTERS, (M,), dtype=torch.int32, device=device, generator=g
        )
        inputs_aux = torch.randn((M, HIDDEN), dtype=DTYPE, device=device, generator=g)
        mapping_aux = torch.randint(
            0, NUM_ADAPTERS, (M,), dtype=torch.int32, device=device, generator=g
        )

        ref_main = run_serial(inputs_main, mapping_main)
        ref_aux = run_serial(inputs_aux, mapping_aux)

        meta_main = LoRAKernelMeta.make(
            max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
        )
        meta_main.prepare_tensors(mapping_main)
        meta_aux = LoRAKernelMeta.make(
            max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
        )
        meta_aux.prepare_tensors(mapping_aux)
        out_main = torch.zeros((NSLICES, M, RANK), dtype=torch.float32, device=device)
        out_aux = torch.zeros((NSLICES, M, RANK), dtype=torch.float32, device=device)

        main_stream = torch.cuda.current_stream()
        input_ready = torch.cuda.Event()
        input_ready.record(main_stream)

        _LORA_A_PTR_DICT.clear()
        lora_shrink(
            inputs_main, lora_a, out_main,
            *meta_main.meta_args(token_nums=M, specialize_active_lora=False),
            SCALING,
        )
        with torch.cuda.stream(aux_stream):
            input_ready.wait(aux_stream)
            _LORA_A_PTR_DICT.clear()
            lora_shrink(
                inputs_aux, lora_a, out_aux,
                *meta_aux.meta_args(token_nums=M, specialize_active_lora=False),
                SCALING,
            )
            output_ready = torch.cuda.Event()
            output_ready.record(aux_stream)
        output_ready.wait(main_stream)
        torch.accelerator.synchronize()

        if not torch.equal(out_main, ref_main):
            mismatches.append(f"round {round_idx} main-stream mismatch")
        if not torch.equal(out_aux, ref_aux):
            mismatches.append(f"round {round_idx} aux-stream mismatch")

    print(json.dumps({
        "mismatches": mismatches,
        "all_matched": len(mismatches) == 0,
    }))
    """
)


def test_lora_shrink_splitk_shared_experts_stream_isolation():
    """Regression for the P1 review finding: without the current stream in
    the named-workspace owner key, a lora_shrink call on the MoE
    shared-experts overlap stream and one on the main stream -- both in the
    same (ubatch, lane) slot, exactly as SharedExperts.maybe_forward_async
    schedules it regardless of VLLM_LORA_ENABLE_DUAL_STREAM -- alias the
    same scratch and corrupt each other's partials.
    """
    result = _run(_SHARED_EXPERTS_STREAM_SCRIPT, _S8_ENV, timeout=300)
    print(json.dumps({k: v for k, v in result.items() if k != "mismatches"}, indent=2))
    assert result["all_matched"], result


# ---------------------------------------------------------------------------
# Init-time guards: illegal env combinations must raise at import, not
# silently fall back to a different algorithm.
# ---------------------------------------------------------------------------

_IMPORT_PROBE = "import vllm.lora.ops.triton_ops.lora_shrink_op"


def test_lora_shrink_splitk_requires_batch_invariant():
    env = os.environ.copy()
    env["VLLM_LORA_DETERMINISTIC_SPLIT_K"] = "8"
    env["VLLM_BATCH_INVARIANT"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    assert proc.returncode != 0, (
        "expected import to fail without VLLM_BATCH_INVARIANT=1"
    )
    assert "VLLM_BATCH_INVARIANT" in proc.stderr, proc.stderr


def test_lora_shrink_splitk_rejects_dual_stream_combo():
    env = os.environ.copy()
    env["VLLM_LORA_DETERMINISTIC_SPLIT_K"] = "8"
    env["VLLM_BATCH_INVARIANT"] = "1"
    env["VLLM_LORA_ENABLE_DUAL_STREAM"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    assert proc.returncode != 0, (
        "expected import to fail when combined with dual-stream"
    )
    assert "VLLM_LORA_ENABLE_DUAL_STREAM" in proc.stderr, proc.stderr


def test_lora_shrink_splitk_invalid_value_rejected():
    env = os.environ.copy()
    env["VLLM_LORA_DETERMINISTIC_SPLIT_K"] = "4"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm.envs as envs; envs.VLLM_LORA_DETERMINISTIC_SPLIT_K",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    assert proc.returncode != 0, (
        "expected VLLM_LORA_DETERMINISTIC_SPLIT_K=4 to be rejected"
    )


# ---------------------------------------------------------------------------
# Launch-config check: the plan requires BM=32/BN=16/warps=4/stages=2/ctas=1
# for the S8 path. get_lora_op_configs skips tuned-JSON overrides whenever
# is_batch_invariant is set (see utils.load_lora_op_config), so this should
# hold unconditionally across shapes — verify it explicitly rather than
# assuming it.
# ---------------------------------------------------------------------------

_LAUNCH_CONFIG_SCRIPT = textwrap.dedent(
    """
    import json
    from vllm.lora.ops.triton_ops.utils import get_lora_op_configs

    shapes = [
        dict(max_loras=2, batch=5, hidden_size=2560, rank=8, num_slices=3),
        dict(max_loras=2, batch=5, hidden_size=9728, rank=16, num_slices=1),
        dict(max_loras=2, batch=5, hidden_size=4097, rank=64, num_slices=1),
        dict(max_loras=2, batch=5, hidden_size=1024, rank=8, num_slices=1),
        dict(max_loras=3, batch=129, hidden_size=256, rank=16, num_slices=1),
    ]
    results = []
    for shape in shapes:
        cfg = get_lora_op_configs("shrink", **shape)
        results.append({
            "shape": shape,
            "block_m": cfg["block_m"],
            "block_n": cfg["block_n"],
            "num_warps": cfg["num_warps"],
            "num_stages": cfg["num_stages"],
            "num_ctas": cfg["num_ctas"],
        })
    print(json.dumps(results))
    """
)


def test_lora_shrink_splitk_launch_config_matches_plan():
    result = _run(_LAUNCH_CONFIG_SCRIPT, {"VLLM_BATCH_INVARIANT": "1"})
    print(json.dumps(result, indent=2))
    mismatches = []
    for entry in result:
        expected = {
            "block_m": 32,
            "block_n": 16,
            "num_warps": 4,
            "num_stages": 2,
            "num_ctas": 1,
        }
        actual = {k: entry[k] for k in expected}
        if actual != expected:
            mismatches.append((entry["shape"], actual, expected))
    assert not mismatches, f"launch config diverged from plan for shapes: {mismatches}"


# The above only queries get_lora_op_configs() as a standalone helper call.
# get_lora_op_configs's block_k/split_k are overridden by _lora_shrink itself
# for the S8 path (BLOCK_K forced to 256, SPLIT_K forced to 8) before the
# Triton kernels are launched, so recording the config dict does not observe
# the effective S8/BK256 launch. Instead, intercept the actual
# partial/reduce kernel launches and read back their real constexpr args by
# name (via JITFunction.arg_names, the same pattern used elsewhere in vllm,
# e.g. vllm/triton_utils/force_first_config.py).
_EFFECTIVE_LAUNCH_SCRIPT = textwrap.dedent(
    """
    import json
    import torch
    import vllm.lora.ops.triton_ops.lora_shrink_op as lora_shrink_op
    from vllm.lora.ops.triton_ops import lora_shrink
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT
    from vllm.v1.worker.workspace import init_workspace_manager

    device = "cuda:0"
    init_workspace_manager(torch.device(device))

    K, RANK, NSLICES, NUM_LORAS, M = 2560, 8, 3, 2, 5
    DTYPE = torch.bfloat16
    SCALING = 0.73

    class _RecordingKernel:
        def __init__(self, real_kernel, sink):
            self._real = real_kernel
            self._sink = sink

        def __getitem__(self, grid):
            launcher = self._real[grid]

            def wrapped(*args, **kwargs):
                bound = dict(zip(self._real.arg_names, args))
                bound.update(kwargs)
                self._sink.append(bound)
                return launcher(*args, **kwargs)

            return wrapped

    partial_calls = []
    reduce_calls = []
    lora_shrink_op._lora_shrink_partial_kernel = _RecordingKernel(
        lora_shrink_op._lora_shrink_partial_kernel, partial_calls
    )
    lora_shrink_op._lora_shrink_reduce_kernel = _RecordingKernel(
        lora_shrink_op._lora_shrink_reduce_kernel, reduce_calls
    )

    inputs = torch.randn((M, K), dtype=DTYPE, device=device)
    lora_weights = [
        torch.randn((NUM_LORAS, RANK, K), dtype=DTYPE, device=device)
        for _ in range(NSLICES)
    ]
    mapping = torch.randint(0, NUM_LORAS, (M,), dtype=torch.int32, device=device)
    lora_meta = LoRAKernelMeta.make(
        max_loras=NUM_LORAS, max_num_tokens=M, device=device
    )
    lora_meta.prepare_tensors(mapping)
    out = torch.zeros((NSLICES, M, RANK), dtype=torch.float32, device=device)
    _LORA_A_PTR_DICT.clear()
    lora_shrink(
        inputs, lora_weights, out,
        *lora_meta.meta_args(token_nums=M, specialize_active_lora=False),
        SCALING,
    )
    torch.accelerator.synchronize()

    assert len(partial_calls) == 1, (
        f"expected exactly one partial-kernel launch, got {len(partial_calls)}"
    )
    assert len(reduce_calls) == 1, (
        f"expected exactly one reduce-kernel launch, got {len(reduce_calls)}"
    )
    partial_args = partial_calls[0]
    reduce_args = reduce_calls[0]
    print(json.dumps({
        "partial_block_m": partial_args["BLOCK_M"],
        "partial_block_n": partial_args["BLOCK_N"],
        "partial_block_k": partial_args["BLOCK_K"],
        "partial_split_k": partial_args["SPLIT_K"],
        "partial_num_warps": partial_args["num_warps"],
        "partial_num_stages": partial_args["num_stages"],
        "partial_num_ctas": partial_args["num_ctas"],
        "reduce_block_m": reduce_args["BLOCK_M"],
        "reduce_block_n": reduce_args["BLOCK_N"],
        "reduce_split_k": reduce_args["SPLIT_K"],
        "reduce_num_warps": reduce_args["num_warps"],
        "reduce_num_stages": reduce_args["num_stages"],
    }))
    """
)


def test_lora_shrink_splitk_effective_launch_config():
    result = _run(_EFFECTIVE_LAUNCH_SCRIPT, _S8_ENV)
    print(json.dumps(result, indent=2))
    expected = {
        "partial_block_m": 32,
        "partial_block_n": 16,
        "partial_block_k": 256,
        "partial_split_k": 8,
        "partial_num_warps": 4,
        "partial_num_stages": 2,
        "partial_num_ctas": 1,
        "reduce_block_m": 32,
        "reduce_block_n": 16,
        "reduce_split_k": 8,
        "reduce_num_warps": 4,
        "reduce_num_stages": 2,
    }
    assert result == expected, (
        f"actual kernel-launch config diverged from plan: {result} != {expected}"
    )


# ---------------------------------------------------------------------------
# Default-path regression: with the new switch off, the existing BI/non-BI
# shrink path must be unaffected.
# ---------------------------------------------------------------------------

_DEFAULT_PATH_SCRIPT = textwrap.dedent(
    """
    import json
    import torch
    from vllm.lora.ops.triton_ops import lora_shrink
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT
    import vllm.envs as envs

    device = "cuda:0"
    torch.manual_seed(0)

    K, RANK, NSLICES, NUM_LORAS, M = 2049, 32, 1, 4, 16
    DTYPE = torch.bfloat16
    SCALING = 0.5

    inputs = torch.randn((M, K), dtype=DTYPE, device=device)
    lora_weights = [torch.randn((NUM_LORAS, RANK, K), dtype=DTYPE, device=device)]
    mapping = torch.randint(0, NUM_LORAS, (M,), dtype=torch.int32, device=device)

    lora_meta = LoRAKernelMeta.make(
        max_loras=NUM_LORAS, max_num_tokens=M, device=device
    )
    lora_meta.prepare_tensors(mapping)

    out = torch.zeros((NSLICES, M, RANK), dtype=torch.float32, device=device)
    _LORA_A_PTR_DICT.clear()
    lora_shrink(
        inputs, lora_weights, out,
        *lora_meta.meta_args(token_nums=M, specialize_active_lora=False),
        SCALING,
    )
    torch.accelerator.synchronize()

    inputs_cpu = inputs.double().cpu()
    mapping_cpu = mapping.cpu()
    w_cpu = lora_weights[0].double().cpu()
    ref = torch.zeros((M, RANK), dtype=torch.float64)
    for lid in range(NUM_LORAS):
        mask = mapping_cpu == lid
        if mask.any():
            ref[mask] = SCALING * (inputs_cpu[mask] @ w_cpu[lid].T)

    close = bool(torch.allclose(out[0].double().cpu(), ref, rtol=0.005, atol=0.005))
    print(json.dumps({
        "split_k_env": envs.VLLM_LORA_DETERMINISTIC_SPLIT_K,
        "batch_invariant_env": envs.VLLM_BATCH_INVARIANT,
        "close": close,
    }))
    """
)


@pytest.mark.parametrize("batch_invariant", ["1", "0"], ids=["BI_on_S1", "BI_off"])
def test_lora_shrink_default_path_unaffected(batch_invariant):
    env = {
        "VLLM_BATCH_INVARIANT": batch_invariant,
        "VLLM_LORA_DETERMINISTIC_SPLIT_K": "0",
    }
    result = _run(_DEFAULT_PATH_SCRIPT, env)
    print(json.dumps(result, indent=2))
    assert result["split_k_env"] == 0
    assert result["close"], f"default shrink path regressed: {result}"


# ---------------------------------------------------------------------------
# CUDA Graph capture/replay: the padded-capacity scratch address must stay
# stable across replays, replaying with different (legal) inputs/metadata
# must match a fresh eager run of that same input, and restoring the
# original inputs must reproduce the original result exactly.
# ---------------------------------------------------------------------------

_GRAPH_SCRIPT = textwrap.dedent(
    """
    import json
    import torch
    from vllm.lora.ops.triton_ops import lora_shrink
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        lock_workspace,
    )

    device = "cuda:0"
    init_workspace_manager(torch.device(device))

    HIDDEN = 2560
    RANK = 16
    NUM_ADAPTERS = 3
    NSLICES = 1
    SCALING = 0.6
    DTYPE = torch.bfloat16
    M_CAP = 96

    def make_mapping(seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randint(
            0, NUM_ADAPTERS, (M_CAP,), dtype=torch.int32, device=device, generator=g
        )

    lora_a = [torch.randn((NUM_ADAPTERS, RANK, HIDDEN), dtype=DTYPE, device=device)]

    def eager_reference(inputs, mapping):
        out = torch.zeros(
            (NSLICES, M_CAP, RANK), dtype=torch.float32, device=device
        )
        meta = LoRAKernelMeta.make(
            max_loras=NUM_ADAPTERS, max_num_tokens=M_CAP, device=device
        )
        meta.prepare_tensors(mapping)
        _LORA_A_PTR_DICT.clear()
        lora_shrink(
            inputs, lora_a, out,
            *meta.meta_args(token_nums=M_CAP, specialize_active_lora=False),
            SCALING,
        )
        torch.accelerator.synchronize()
        return out.clone()

    original_inputs = torch.randn((M_CAP, HIDDEN), dtype=DTYPE, device=device)
    original_mapping = make_mapping(1)
    original_reference = eager_reference(original_inputs, original_mapping)

    static_inputs = torch.zeros((M_CAP, HIDDEN), dtype=DTYPE, device=device)
    static_out = torch.zeros(
        (NSLICES, M_CAP, RANK), dtype=torch.float32, device=device
    )
    static_inputs.copy_(original_inputs)
    lora_meta = LoRAKernelMeta.make(
        max_loras=NUM_ADAPTERS, max_num_tokens=M_CAP, device=device
    )
    lora_meta.prepare_tensors(original_mapping)

    # Warmup on a side stream (required before capture), then lock so the
    # scratch address cannot move once the graph references it. Real capture
    # (gpu_model_runner.capture_model) warms up and captures every shape on
    # the *same* graph_capture() stream before locking the workspace
    # afterward; mirror that here by explicitly capturing on warmup_stream
    # too, since the named-workspace owner key now includes the stream.
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            _LORA_A_PTR_DICT.clear()
            lora_shrink(
                static_inputs, lora_a, static_out,
                *lora_meta.meta_args(token_nums=M_CAP, specialize_active_lora=False),
                SCALING,
            )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.accelerator.synchronize()

    manager = current_workspace_manager()
    owner_key = ("lora_shrink_two_pass", warmup_stream)
    ptr_before_lock = manager._named_workspaces[owner_key][0].data_ptr()
    lock_workspace()

    graph = torch.cuda.CUDAGraph()
    _LORA_A_PTR_DICT.clear()
    with torch.cuda.graph(graph, stream=warmup_stream):
        lora_shrink(
            static_inputs, lora_a, static_out,
            *lora_meta.meta_args(token_nums=M_CAP, specialize_active_lora=False),
            SCALING,
        )

    ptr_after_capture = manager._named_workspaces[owner_key][0].data_ptr()
    results = {"ptr_stable_through_capture": ptr_before_lock == ptr_after_capture}

    # Replay 0: unchanged inputs/metadata -- must match the original eager
    # reference exactly.
    graph.replay()
    torch.accelerator.synchronize()
    results["replay0_matches_original_eager"] = bool(
        torch.equal(static_out, original_reference)
    )

    # Replay 1: swap in different, still-legal inputs and metadata (no
    # recapture) and compare against a fresh eager run of that same input.
    new_inputs = torch.randn((M_CAP, HIDDEN), dtype=DTYPE, device=device)
    new_mapping = make_mapping(2)
    new_reference = eager_reference(new_inputs, new_mapping)
    static_inputs.copy_(new_inputs)
    lora_meta.prepare_tensors(new_mapping)
    graph.replay()
    torch.accelerator.synchronize()
    results["replay1_matches_fresh_eager"] = bool(
        torch.equal(static_out, new_reference)
    )
    ptr_after_replay1 = manager._named_workspaces[owner_key][0].data_ptr()
    results["ptr_stable_through_replay1"] = ptr_before_lock == ptr_after_replay1

    # Replay 2: restore the original inputs/metadata and confirm the graph
    # reproduces the original reference again (no accumulated state leak).
    static_inputs.copy_(original_inputs)
    lora_meta.prepare_tensors(original_mapping)
    graph.replay()
    torch.accelerator.synchronize()
    results["replay2_restored_matches_original"] = bool(
        torch.equal(static_out, original_reference)
    )
    ptr_after_replay2 = manager._named_workspaces[owner_key][0].data_ptr()
    results["ptr_stable_through_replay2"] = ptr_before_lock == ptr_after_replay2

    print(json.dumps(results))
    """
)


def test_lora_shrink_splitk_cuda_graph_capture_replay():
    result = _run(_GRAPH_SCRIPT, _S8_ENV, timeout=300)
    print(json.dumps(result, indent=2))
    failures = [name for name, ok in result.items() if not ok]
    assert not failures, f"failed checks: {failures}; full result: {result}"


# ---------------------------------------------------------------------------
# Ordinary concurrent CUDA streams (not the LoRA dual-stream/PDL feature,
# which split-K=8 explicitly rejects at init): two plain streams on two
# distinct workspace lanes must not alias each other's scratch and must each
# match their own serial reference.
# ---------------------------------------------------------------------------

_CONCURRENT_STREAMS_SCRIPT = textwrap.dedent(
    """
    import json
    import torch
    from vllm.lora.ops.triton_ops import lora_shrink
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        use_workspace_lane,
    )

    device = "cuda:0"
    init_workspace_manager(torch.device(device), num_ubatches=1, num_lanes=2)

    HIDDEN = 2560
    RANK = 16
    NUM_ADAPTERS = 2
    NSLICES = 1
    SCALING = 0.9
    DTYPE = torch.bfloat16
    M = 32
    ROUNDS = 100

    lora_a = [torch.randn((NUM_ADAPTERS, RANK, HIDDEN), dtype=DTYPE, device=device)]

    def run_serial(inputs, mapping):
        meta = LoRAKernelMeta.make(
            max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
        )
        meta.prepare_tensors(mapping)
        out = torch.zeros((NSLICES, M, RANK), dtype=torch.float32, device=device)
        _LORA_A_PTR_DICT.clear()
        lora_shrink(
            inputs, lora_a, out,
            *meta.meta_args(token_nums=M, specialize_active_lora=False),
            SCALING,
        )
        torch.accelerator.synchronize()
        return out.clone()

    stream0 = torch.cuda.Stream(device=device)
    stream1 = torch.cuda.Stream(device=device)

    manager = current_workspace_manager()
    lane_ptrs = {}
    mismatches = []
    for round_idx in range(ROUNDS):
        g = torch.Generator(device=device).manual_seed(1000 + round_idx)
        inputs0 = torch.randn((M, HIDDEN), dtype=DTYPE, device=device, generator=g)
        mapping0 = torch.randint(
            0, NUM_ADAPTERS, (M,), dtype=torch.int32, device=device, generator=g
        )
        inputs1 = torch.randn((M, HIDDEN), dtype=DTYPE, device=device, generator=g)
        mapping1 = torch.randint(
            0, NUM_ADAPTERS, (M,), dtype=torch.int32, device=device, generator=g
        )

        ref0 = run_serial(inputs0, mapping0)
        ref1 = run_serial(inputs1, mapping1)

        meta0 = LoRAKernelMeta.make(
            max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
        )
        meta0.prepare_tensors(mapping0)
        meta1 = LoRAKernelMeta.make(
            max_loras=NUM_ADAPTERS, max_num_tokens=M, device=device
        )
        meta1.prepare_tensors(mapping1)
        out0 = torch.zeros((NSLICES, M, RANK), dtype=torch.float32, device=device)
        out1 = torch.zeros((NSLICES, M, RANK), dtype=torch.float32, device=device)

        main_stream = torch.cuda.current_stream()
        with use_workspace_lane(0), torch.cuda.stream(stream0):
            # inputs0/mapping0/meta0 were built on the main stream above;
            # without this wait, stream0 could launch before those writes
            # are visible to it.
            stream0.wait_stream(main_stream)
            _LORA_A_PTR_DICT.clear()
            lora_shrink(
                inputs0, lora_a, out0,
                *meta0.meta_args(token_nums=M, specialize_active_lora=False),
                SCALING,
            )
            lane_ptrs[0] = (
                manager._named_workspaces[("lora_shrink_two_pass", stream0)][0]
                .data_ptr()
            )
        with use_workspace_lane(1), torch.cuda.stream(stream1):
            stream1.wait_stream(main_stream)
            _LORA_A_PTR_DICT.clear()
            lora_shrink(
                inputs1, lora_a, out1,
                *meta1.meta_args(token_nums=M, specialize_active_lora=False),
                SCALING,
            )
            lane_ptrs[1] = (
                manager._named_workspaces[("lora_shrink_two_pass", stream1)][1]
                .data_ptr()
            )

        stream0.synchronize()
        stream1.synchronize()

        if not torch.equal(out0, ref0):
            mismatches.append(f"round {round_idx} lane0 mismatch")
        if not torch.equal(out1, ref1):
            mismatches.append(f"round {round_idx} lane1 mismatch")

    print(json.dumps({
        "lanes_non_aliasing": lane_ptrs[0] != lane_ptrs[1],
        # One owner entry per distinct stream: run_serial's reference calls
        # use the main/current stream, and stream0/stream1 each get their
        # own -- three total, not one per round (100 rounds reuse the same
        # three streams throughout).
        "owner_slot_count_bounded": len(manager._named_workspaces) == 3,
        "mismatches": mismatches,
        "all_matched": len(mismatches) == 0,
    }))
    """
)


def test_lora_shrink_splitk_ordinary_concurrent_streams():
    result = _run(_CONCURRENT_STREAMS_SCRIPT, _S8_ENV, timeout=300)
    print(json.dumps({k: v for k, v in result.items() if k != "mismatches"}, indent=2))
    assert result["lanes_non_aliasing"], result
    assert result["owner_slot_count_bounded"], result
    assert result["all_matched"], result["mismatches"]
