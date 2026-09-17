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

    HIDDEN = 256
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

        expand_out = residual.clone()
        _LORA_B_PTR_DICT.clear()
        lora_expand(
            shrink_out, lora_b, expand_out,
            *lora_meta.meta_args(token_nums=M, specialize_active_lora=False),
            offset_start=0,
            add_inputs=True,
        )
        torch.accelerator.synchronize()
        return expand_out[target_pos].clone()

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

    reference = run_once(*configs["M1"])

    results = {}
    for name, (M, pos, companions) in configs.items():
        outputs = [run_once(M, pos, companions) for _ in range(REPEATS)]
        self_identical = all(torch.equal(outputs[0], o) for o in outputs[1:])
        matches_reference = torch.equal(outputs[0], reference)
        results[name] = {
            "self_identical_across_repeats": self_identical,
            "matches_M1_reference": matches_reference,
        }

    # Two-adapter, three-slice point.
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
    M3 = 50
    target_pos3 = 25
    mapping3 = torch.tensor(
        [0 if i == target_pos3 else (1 if i % 2 == 0 else 0) for i in range(M3)],
        dtype=torch.int32,
        device=device,
    )
    inputs3 = torch.randn((M3, HIDDEN), dtype=DTYPE, device=device, generator=g)
    inputs3[target_pos3] = target_input[0]
    residual3 = torch.randn(
        (M3, HIDDEN * NSLICES3), dtype=DTYPE, device=device, generator=g
    )

    lora_meta3 = LoRAKernelMeta.make(
        max_loras=NUM_ADAPTERS, max_num_tokens=M3, device=device
    )
    lora_meta3.prepare_tensors(mapping3)

    def run_once_3slices():
        shrink_out3 = torch.zeros(
            (NSLICES3, M3, RANK), dtype=torch.float32, device=device
        )
        _LORA_A_PTR_DICT.clear()
        lora_shrink(
            inputs3, lora_a3, shrink_out3,
            *lora_meta3.meta_args(token_nums=M3, specialize_active_lora=False),
            SCALING,
        )
        expand_out3 = residual3.clone()
        _LORA_B_PTR_DICT.clear()
        lora_expand(
            shrink_out3, lora_b3, expand_out3,
            *lora_meta3.meta_args(token_nums=M3, specialize_active_lora=False),
            offset_start=0,
            add_inputs=True,
        )
        torch.accelerator.synchronize()
        return expand_out3[target_pos3].clone()

    outputs3 = [run_once_3slices() for _ in range(REPEATS)]
    results["dual_adapter_3slices"] = {
        "self_identical_across_repeats": all(
            torch.equal(outputs3[0], o) for o in outputs3[1:]
        ),
        "matches_M1_reference": None,
    }

    print(json.dumps(results))
    """
)


def test_lora_shrink_splitk_self_batch_invariance():
    result = _run(_SELF_BI_SCRIPT, _S8_ENV, timeout=300)
    failures = []
    for name, r in result.items():
        if not r["self_identical_across_repeats"]:
            failures.append(f"{name}: NOT self-identical across 20 repeats")
        if r["matches_M1_reference"] is False:
            failures.append(f"{name}: target row output differs from M=1 reference")
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

    # 4. Metadata swap between calls must not leak stale scratch: reuse the
    # same output/workspace buffers across two calls with disjoint valid
    # rows and confirm the second call's invalid rows are zero, not stale
    # values from the first call's valid rows at the same position.
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

    lora_meta_swap.prepare_tensors(mapping_b)
    _LORA_A_PTR_DICT.clear()
    lora_shrink(
        inputs_mixed, lora_a, shared_out,
        *lora_meta_swap.meta_args(token_nums=M, specialize_active_lora=False),
        SCALING,
    )
    torch.accelerator.synchronize()
    # Rows valid under mapping_a but invalid under mapping_b must now read 0,
    # not the stale nonzero value from the first call.
    became_invalid = ((mapping_a != -1) & (mapping_b == -1)).cpu()
    second_call_cpu = shared_out[0].cpu()
    results["metadata_swap_no_stale_scratch"] = bool(
        torch.all(second_call_cpu[became_invalid] == 0.0).item()
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
