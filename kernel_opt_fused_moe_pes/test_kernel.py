#!/usr/bin/env python3
"""Evaluator entry for the MI308X FlyDSL FP8 PTPC fused_moe PES workspace."""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile


M_VALUES = [1, 16, 32, 64, 128, 256, 512]
SHAPE = {
    "name": "task16",
    "E": 512,
    "TOPK": 10,
    "model_dim": 4096,
    "inter_dim": 256,
}
PROFILE_STEPS = [
    "routing",
    "quant",
    "stage1",
    "stage2",
    "fused_1stage",
    "finalize",
    "overhead",
    "other",
]
ATREX_V2_BASELINE_US = {
    1: {"routing": 17.7, "quant": 5.1, "stage1": 11.6, "stage2": 7.9, "overhead": 1.5, "other": 0.0, "kernel_sum": 43.9, "e2e_avg": 544.9, "e2e_min": 528.9},
    16: {"routing": 12.2, "quant": 8.3, "stage1": 91.4, "stage2": 54.6, "overhead": 2.4, "other": 0.0, "kernel_sum": 169.0, "e2e_avg": 601.0, "e2e_min": 575.2},
    32: {"routing": 12.0, "quant": 8.7, "stage1": 120.9, "stage2": 85.3, "overhead": 4.3, "other": 0.0, "kernel_sum": 231.2, "e2e_avg": 587.6, "e2e_min": 563.8},
    64: {"routing": 13.1, "quant": 8.4, "stage1": 180.5, "stage2": 128.6, "overhead": 4.6, "other": 0.0, "kernel_sum": 335.2, "e2e_avg": 663.4, "e2e_min": 620.2},
    128: {"routing": 13.9, "quant": 9.3, "stage1": 244.4, "stage2": 170.6, "overhead": 5.2, "other": 0.0, "kernel_sum": 443.4, "e2e_avg": 735.1, "e2e_min": 714.8},
    256: {"routing": 15.8, "quant": 14.4, "stage1": 275.1, "stage2": 185.5, "overhead": 5.7, "other": 0.0, "kernel_sum": 496.4, "e2e_avg": 796.2, "e2e_min": 782.6},
    512: {"routing": 20.1, "quant": 23.5, "stage1": 334.2, "stage2": 200.0, "overhead": 5.9, "other": 0.0, "kernel_sum": 583.8, "e2e_avg": 938.0, "e2e_min": 925.4},
}


def _require_aiter_base() -> Path:
    value = os.environ.get("AITER_BASE")
    if not value:
        raise RuntimeError(
            "AITER_BASE must point to an aiter checkout. "
            "Example: export AITER_BASE=/path/to/aiter"
        )
    return Path(value).expanduser()


def _load_candidate(path: Path):
    workspace = str(path.parent.resolve())
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    spec = importlib.util.spec_from_file_location("candidate_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load candidate module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_kernel"] = module
    spec.loader.exec_module(module)
    return module


def _load_aiter():
    aiter_base = _require_aiter_base()
    if (aiter_base / "aiter" / "__init__.py").exists():
        aiter_base_str = str(aiter_base.resolve())
        if aiter_base_str not in sys.path:
            sys.path.insert(0, aiter_base_str)

    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe as aiter_fused_moe
    from aiter.fused_moe import fused_topk, torch_moe_stage1, torch_moe_stage2
    from aiter.ops.quant import get_torch_quant
    from aiter.ops.shuffle import shuffle_weight
    from aiter.test_common import checkAllclose

    return {
        "ActivationType": ActivationType,
        "QuantType": QuantType,
        "dtypes": dtypes,
        "aiter_fused_moe": aiter_fused_moe,
        "fused_topk": fused_topk,
        "torch_moe_stage1": torch_moe_stage1,
        "torch_moe_stage2": torch_moe_stage2,
        "get_torch_quant": get_torch_quant,
        "shuffle_weight": shuffle_weight,
        "checkAllclose": checkAllclose,
    }


def flush_cache(size_mb=128, device="cuda:0", dtype=torch.int32, rounds=2):
    n = (size_mb * 1024 * 1024) // torch.tensor([], dtype=dtype).element_size()
    buf = torch.empty(n, device=device, dtype=dtype)
    for _ in range(rounds):
        buf.add_(1)
    torch.cuda.synchronize()
    return buf


def profile_cuda_kernels_ordered(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        acc_events=True,
    ) as prof:
        for _ in range(iters):
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()

    cuda_events = []
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.device_time > 0:
            cuda_events.append((evt.name, evt.device_time))

    if not cuda_events:
        return []

    kernels_per_iter = len(cuda_events) // iters
    if kernels_per_iter == 0:
        return []

    return [
        cuda_events[i * kernels_per_iter : (i + 1) * kernels_per_iter]
        for i in range(iters)
    ]


def profile_e2e_cuda_events(fn, warmup=5, iters=20, setup_fn=None):
    for _ in range(warmup):
        if setup_fn:
            setup_fn()
        fn()
    torch.cuda.synchronize()

    times_us = []
    for _ in range(iters):
        if setup_fn:
            setup_fn()
            torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        times_us.append(start_evt.elapsed_time(end_evt) * 1000.0)
    return times_us


def _is_flush_cache_kernel(name):
    lower = name.lower()
    return (
        "cudafunctoronself_add<int>" in lower
        or ("vectorized_elementwise_kernel" in lower and "add<int>" in lower)
    )


def _is_overhead_kernel(lower):
    return (
        "memcpy" in lower
        or "fillfunctor" in lower
        or "aten::fill" in lower
        or "fill_kernel" in lower
    )


def classify_kernels_fp8_ptpc(kernel_list):
    steps = {name: 0.0 for name in PROFILE_STEPS}
    generic_gemm_index = 0
    for name, dt in kernel_list:
        lower = name.lower()
        if _is_flush_cache_kernel(name):
            continue
        if _is_overhead_kernel(lower):
            steps["overhead"] += dt
        elif (
            "moesortingkernel" in lower
            or "moe_sorting" in lower
            or "moe sorting" in lower
            or "topk" in lower
            or "routing" in lower
            or "sort" in lower
        ):
            steps["routing"] += dt
        elif (
            "dynamic_per_token_scaled_quant" in lower
            or "smoothquant" in lower
            or "quant" in lower
            or "cast" in lower
        ):
            steps["quant"] += dt
        elif (
            "1stage" in lower
            or "1_stage" in lower
            or "moe_ck1stage" in lower
            or "ck_moe_stage1_stage2" in lower
            or ("fmoe_" in lower and "pertokenfp8" in lower and "stage1" not in lower)
        ):
            steps["fused_1stage"] += dt
        elif (
            "moe_gemm1" in lower
            or "fmoe_stage1" in lower
            or "ck_moe_stage1" in lower
            or "stage1" in lower
            or "gemm1" in lower
        ):
            steps["stage1"] += dt
        elif (
            "moe_gemm2" in lower
            or "moe_ck2stages_gemm2" in lower
            or "ck_moe_stage2" in lower
            or "mulroutedweight" in lower
            or "stage2" in lower
            or "gemm2" in lower
        ):
            steps["stage2"] += dt
        elif "kernel_moe_gemm" in lower or "gridwisemoegemm" in lower:
            if steps["stage1"] > 0 or steps["fused_1stage"] > 0:
                steps["stage2"] += dt
            elif generic_gemm_index == 0:
                steps["stage1"] += dt
            else:
                steps["stage2"] += dt
            generic_gemm_index += 1
        elif "reduce" in lower or "final" in lower:
            steps["finalize"] += dt
        else:
            steps["other"] += dt
    return steps


def _average_steps(per_iter):
    if not per_iter:
        return {name: 0.0 for name in PROFILE_STEPS}
    steps_all = [classify_kernels_fp8_ptpc(kl) for kl in per_iter]
    return {
        step: sum(row[step] for row in steps_all) / len(steps_all)
        for step in PROFILE_STEPS
    }


def _make_fp8_ptpc_env(aiter, M, device="cuda:0"):
    dtype = torch.bfloat16
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    hidden_states = torch.empty(
        (M, SHAPE["model_dim"]), dtype=dtype, device=device
    ).uniform_(-1, 1)
    w1 = torch.empty(
        (SHAPE["E"], SHAPE["inter_dim"] * 2, SHAPE["model_dim"]),
        dtype=dtype,
        device=device,
    ).uniform_(-1, 1)
    w2 = torch.empty(
        (SHAPE["E"], SHAPE["model_dim"], SHAPE["inter_dim"]),
        dtype=dtype,
        device=device,
    ).uniform_(-1, 1)
    score = torch.empty((M, SHAPE["E"]), dtype=dtype, device=device).uniform_(-1, 1)

    topk_weights, topk_ids = aiter["fused_topk"](
        hidden_states, score, SHAPE["TOPK"], True
    )
    torch_quant = aiter["get_torch_quant"](aiter["QuantType"].per_Token)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=aiter["dtypes"].fp8)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=aiter["dtypes"].fp8)

    return {
        "hidden_states": hidden_states,
        "w1_qt": w1_qt,
        "w2_qt": w2_qt,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "M": M,
        "dtype": dtype,
        "device": device,
    }


def _make_candidate_runner(aiter, candidate_module, env):
    w1_shuffled = aiter["shuffle_weight"](env["w1_qt"], layout=(16, 16), use_int4=False)
    w2_shuffled = aiter["shuffle_weight"](env["w2_qt"], layout=(16, 16), use_int4=False)

    def run():
        return candidate_module.fused_moe_flydsl_fp8_ptpc(
            env["hidden_states"],
            w1_shuffled,
            w2_shuffled,
            env["topk_weights"],
            env["topk_ids"],
            w1_scale=env["w1_scale"],
            w2_scale=env["w2_scale"],
        )

    return run


def _fused_moe_ref(aiter, env):
    torch_quant = aiter["get_torch_quant"](aiter["QuantType"].per_Token)
    a1_qt, a1_scale = torch_quant(env["hidden_states"], quant_dtype=aiter["dtypes"].fp8)
    out1_ref = aiter["torch_moe_stage1"](
        a1_qt,
        env["w1_qt"],
        env["w2_qt"],
        env["topk_weights"],
        env["topk_ids"],
        dtype=env["dtype"],
        activation=aiter["ActivationType"].Silu,
        quant_type=aiter["QuantType"].per_Token,
        a1_scale=a1_scale,
        w1_scale=env["w1_scale"],
        doweight=False,
    )
    a2_qt, a2_scale = torch_quant(out1_ref, quant_dtype=aiter["dtypes"].fp8)
    a2_qt = a2_qt.view(env["M"], SHAPE["TOPK"], -1)
    return aiter["torch_moe_stage2"](
        a2_qt,
        env["w1_qt"],
        env["w2_qt"],
        env["topk_weights"],
        env["topk_ids"],
        dtype=env["dtype"],
        quant_type=aiter["QuantType"].per_Token,
        w2_scale=env["w2_scale"],
        a2_scale=a2_scale,
        doweight=True,
    )


def _profile_runner(fn, device, warmup=10, iters=20):
    flush_fn = lambda: flush_cache(128, device=device)

    def profiled_fn():
        flush_cache(128, device=device)
        return fn()

    per_iter = profile_cuda_kernels_ordered(profiled_fn, warmup=warmup, iters=iters)
    steps = _average_steps(per_iter)
    e2e_times = profile_e2e_cuda_events(
        fn, warmup=warmup, iters=iters, setup_fn=flush_fn
    )
    return {
        "steps": steps,
        "kernel_sum": sum(steps.values()),
        "e2e_avg": sum(e2e_times) / len(e2e_times),
        "e2e_min": min(e2e_times),
        "per_iter": per_iter,
    }


def run_correctness(candidate_path: Path, m_values):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm torch.cuda device is required")

    aiter = _load_aiter()
    candidate = _load_candidate(candidate_path)
    max_err = 0.0
    for M in m_values:
        env = _make_fp8_ptpc_env(aiter, M)
        out = _make_candidate_runner(aiter, candidate, env)()
        ref = _fused_moe_ref(aiter, env)
        torch.cuda.synchronize()
        assert not torch.isnan(out).any(), f"M={M}: candidate output contains NaN"
        err = aiter["checkAllclose"](out, ref, rtol=1e-02, atol=1e-02)
        max_err = max(max_err, float(err))
        assert err <= 0.22, f"M={M}: checkAllclose error ratio {err:.2%}"
    print(f"PASS correctness max_error_ratio={max_err:.6f}")


def run_profile(candidate_path: Path, m_values):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm torch.cuda device is required")

    aiter = _load_aiter()
    candidate = _load_candidate(candidate_path)
    for M in m_values:
        env = _make_fp8_ptpc_env(aiter, M)
        fn = _make_candidate_runner(aiter, candidate, env)
        out = fn()
        torch.cuda.synchronize()
        assert not torch.isnan(out).any(), f"M={M}: candidate output contains NaN"
        profile = _profile_runner(fn, env["device"])
        baseline = ATREX_V2_BASELINE_US[M]
        print(
            f"M={M} kernel_sum_us={profile['kernel_sum']:.1f} "
            f"e2e_avg_us={profile['e2e_avg']:.1f} "
            f"e2e_min_us={profile['e2e_min']:.1f} "
            f"baseline_e2e_avg_us={baseline['e2e_avg']:.1f}"
        )
        assert profile["per_iter"], f"M={M}: profile produced no CUDA kernels"
        assert profile["steps"]["stage1"] > 0, f"M={M}: missing stage1 kernels"
        assert profile["steps"]["stage2"] > 0, f"M={M}: missing stage2 kernels"


def _parse_m_values(raw: str):
    if raw == "all":
        return M_VALUES
    return [int(x) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="kernel.py", help="Candidate kernel path")
    parser.add_argument("--mode", choices=("correctness", "profile"), default="correctness")
    parser.add_argument("--m-values", default="all", help="'all' or comma-separated token counts")
    args = parser.parse_args()

    candidate_path = Path(args.kernel).resolve()
    m_values = _parse_m_values(args.m_values)
    if args.mode == "correctness":
        run_correctness(candidate_path, m_values)
    else:
        run_profile(candidate_path, m_values)


if __name__ == "__main__":
    raise SystemExit(main())
