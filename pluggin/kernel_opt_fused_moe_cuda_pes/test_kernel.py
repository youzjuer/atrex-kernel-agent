#!/usr/bin/env python3
"""Correctness and latency evaluator for FlashInfer-aligned FP4 MoE candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["VLLM_FUSE_QKNORM_ROPE_AND_KVCACHE_WRITE"] = "0"
os.environ["VLLM_FUSE_QKNORM_AND_ROPE"] = "0"

import torch

from reference import ARG_NAMES, make_inputs, run_reference


_VLLM_CONFIG = None
_VLLM_WORKSPACE_READY = False


def _load_candidate(path: Path):
    workspace = Path(__file__).resolve().parent
    candidate_root = path.parent if (path.parent / "src").exists() else workspace
    os.environ["FUSED_MOE_CUDA_ROOT"] = str(candidate_root)
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("candidate_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_kernel"] = module
    spec.loader.exec_module(module)
    return module


def _bench(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _first(result):
    if isinstance(result, (list, tuple)):
        return result[0]
    return result


def _case_kwargs(args, tokens: int):
    return {
        "preset": args.preset,
        "tokens": tokens,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "local_num_experts": args.local_num_experts,
        "local_expert_offset": args.local_expert_offset,
        "seed": args.seed,
    }


def _shape_from_inputs(inputs) -> dict[str, int]:
    hidden_states = inputs[2]
    hidden_scale = inputs[3]
    hidden_size = hidden_states.shape[1]
    if hidden_states.dtype == torch.uint8:
        hidden_size *= 2
    return {
        "hidden_size": hidden_size,
        "hidden_states_shape": tuple(hidden_states.shape),
        "hidden_states_dtype": str(hidden_states.dtype),
        "hidden_states_scale_shape": None if hidden_scale is None else tuple(hidden_scale.shape),
        "hidden_states_scale_dtype": None if hidden_scale is None else str(hidden_scale.dtype),
        "gemm1_weights_shape": tuple(inputs[4].shape),
        "gemm1_weights_scale_shape": tuple(inputs[5].shape),
        "gemm2_weights_shape": tuple(inputs[10].shape),
        "gemm2_weights_scale_shape": tuple(inputs[11].shape),
        "intermediate_size": inputs[20],
        "num_experts": inputs[16],
        "top_k": inputs[17],
        "local_expert_offset": inputs[21],
        "local_num_experts": inputs[22],
        "routing_method_type": inputs[24],
    }


def _error_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 2000 else text[:2000] + "...<truncated>"


def _disable_core_dumps() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def _shape_seed_offset(shape_id: str) -> int:
    try:
        return int(shape_id)
    except ValueError:
        return sum(ord(ch) for ch in shape_id)


def _is_catalog_preset(preset: str) -> bool:
    from workload_shapes import is_catalog_preset

    return is_catalog_preset(preset)


def _init_vllm_flashinfer_runtime():
    """Initialize the vLLM context expected by FlashInfer's TRT-LLM MoE wrapper."""
    global _VLLM_CONFIG, _VLLM_WORKSPACE_READY
    if not _VLLM_WORKSPACE_READY:
        from vllm.v1.worker.workspace import init_workspace_manager

        init_workspace_manager(torch.device("cuda"))
        _VLLM_WORKSPACE_READY = True
    if _VLLM_CONFIG is None:
        from vllm.config import VllmConfig

        _VLLM_CONFIG = VllmConfig()
    return _VLLM_CONFIG


def _make_real_nvfp4_flashinfer_inputs(args, tokens: int) -> tuple:
    """Build the true G11-style packed-NVFP4 FlashInfer input contract."""
    from flashinfer import fp4_quantize
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )
    from workload_shapes import get_shape

    shape = get_shape(args.preset)
    if shape["dtype"] != "nvfp4":
        raise NotImplementedError(
            f"real FlashInfer input builder currently supports nvfp4 only; "
            f"preset {args.preset!r} has dtype={shape['dtype']!r}"
        )
    hidden_size = int(shape["hidden_size"])
    intermediate_size = int(shape["intermediate_size"])
    num_experts = int(shape["num_experts"])
    local_num_experts = int(shape["local_num_experts"])
    top_k = int(shape["top_k"])
    if args.hidden_size is not None and args.hidden_size != hidden_size:
        raise ValueError("hidden-size override would no longer match the real catalog shape")
    if args.intermediate_size is not None and args.intermediate_size != intermediate_size:
        raise ValueError("intermediate-size override would no longer match the real catalog shape")
    if args.local_num_experts is not None and args.local_num_experts != local_num_experts:
        raise ValueError("local-num-experts override would no longer match the real catalog shape")
    if args.local_expert_offset != 0:
        raise ValueError("local-expert-offset override would no longer match the tp=1 catalog shape")
    if local_num_experts != num_experts:
        raise ValueError("real tp=1 FlashInfer target expects all experts to be local")

    _init_vllm_flashinfer_runtime()
    device = "cuda"
    sf_vec_size = 16
    torch.manual_seed(int(args.seed) + _shape_seed_offset(str(shape["group"])))
    one = torch.tensor(1.0, device=device)

    x = torch.randn(tokens, hidden_size, device=device, dtype=torch.bfloat16) / 10
    w13 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    routing_logits = torch.rand(tokens, num_experts, dtype=torch.bfloat16, device=device)

    hidden_states, hidden_states_scale = fp4_quantize(
        x,
        one,
        sf_vec_size=sf_vec_size,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    w13q, w13s = fp4_quantize(
        w13.reshape(num_experts * 2 * intermediate_size, hidden_size),
        one,
        sf_vec_size=sf_vec_size,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    w13q = w13q.reshape(num_experts, 2 * intermediate_size, hidden_size // 2)
    w13s = w13s.reshape(num_experts, 2 * intermediate_size, hidden_size // sf_vec_size)
    w2q, w2s = fp4_quantize(
        w2.reshape(num_experts * hidden_size, intermediate_size),
        one,
        sf_vec_size=sf_vec_size,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    w2q = w2q.reshape(num_experts, hidden_size, intermediate_size // 2)
    w2s = w2s.reshape(num_experts, hidden_size, intermediate_size // sf_vec_size)
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale = (
        prepare_static_weights_for_trtllm_fp4_moe(
            w13q,
            w2q,
            w13s,
            w2s,
            hidden_size,
            intermediate_size,
            num_experts,
        )
    )
    one_per_expert = torch.ones((num_experts,), device=device, dtype=torch.float32)

    return (
        routing_logits,
        None,
        hidden_states,
        hidden_states_scale.view(torch.float8_e4m3fn),
        gemm1_weights,
        gemm1_weights_scale,
        None,
        None,
        None,
        None,
        gemm2_weights,
        gemm2_weights_scale,
        None,
        one_per_expert,
        one_per_expert,
        one_per_expert,
        num_experts,
        top_k,
        0,
        0,
        intermediate_size,
        0,
        local_num_experts,
        None,
        1,
        True,
        None,
        3,
        None,
        max(tokens, 8192),
        True,
        None,
    )


def _make_inputs_for_case(args, tokens: int) -> tuple:
    if _is_catalog_preset(args.preset):
        return _make_real_nvfp4_flashinfer_inputs(args, tokens)
    return make_inputs(**_case_kwargs(args, tokens))


def _uses_real_packed_nvfp4(inputs: tuple) -> bool:
    return inputs[2].dtype == torch.uint8 and inputs[3] is not None


def _call_flashinfer(fn, inputs: tuple):
    if _uses_real_packed_nvfp4(inputs):
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(_init_vllm_flashinfer_runtime()):
            return fn(*inputs)
    return fn(*inputs)


def _flashinfer_reference(inputs: tuple):
    import flashinfer

    fn = getattr(flashinfer, "trtllm_fp4_block_scale_moe", None)
    if fn is None:
        raise RuntimeError("flashinfer.trtllm_fp4_block_scale_moe is not available")
    return _call_flashinfer(fn, inputs)


def _bench_flashinfer_in_worker(token_values, warmup: int, rep: int, args) -> dict:
    try:
        import flashinfer
    except Exception as exc:  # pragma: no cover - depends on host install
        return {"status": "UNAVAILABLE", "error": _error_text(exc), "rows": []}

    fn = getattr(flashinfer, "trtllm_fp4_block_scale_moe", None)
    if fn is None:
        return {
            "status": "UNAVAILABLE",
            "error": "flashinfer.trtllm_fp4_block_scale_moe is not available",
            "rows": [],
        }

    rows = []
    max_abs = None
    max_rel = None
    latencies = []
    for tokens in token_values:
        inputs = _make_inputs_for_case(args, tokens)
        row = {
            "tokens": tokens,
            "preset": args.preset,
            "shape": _shape_from_inputs(inputs),
            "input_contract": (
                "packed_nvfp4_flashinfer" if _uses_real_packed_nvfp4(inputs) else "local_oracle"
            ),
        }
        try:
            if args.check_flashinfer_correctness and not _uses_real_packed_nvfp4(inputs):
                ref = _first(run_reference(*inputs))
                out = _first(_call_flashinfer(fn, inputs))
                torch.cuda.synchronize()
                diff = (out.float() - ref.float()).abs()
                abs_err = float(diff.max().item())
                rel_err = float((diff / ref.float().abs().clamp_min(1e-3)).max().item())
                max_abs = abs_err if max_abs is None else max(max_abs, abs_err)
                max_rel = rel_err if max_rel is None else max(max_rel, rel_err)
                row["max_abs"] = abs_err
                row["max_rel"] = rel_err
                if abs_err > args.atol and rel_err > args.rtol:
                    row["status"] = "FAIL"
                else:
                    row["status"] = "PASS"
            else:
                row["status"] = "PASS"
            if row["status"] == "PASS" and args.mode == "profile":
                row["flashinfer_us"] = _bench(lambda: _call_flashinfer(fn, inputs), warmup, rep)
                latencies.append(row["flashinfer_us"])
        except Exception as exc:
            row["status"] = "ERROR"
            row["error"] = _error_text(exc)
            rows.append(row)
            break
        rows.append(row)

    status = "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "ERROR"
    if any(row.get("status") == "FAIL" for row in rows):
        status = "FAIL"
    return {
        "status": status,
        "version": getattr(flashinfer, "__version__", "unknown"),
        "operator": "flashinfer.trtllm_fp4_block_scale_moe",
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_latency_us": sum(latencies) / len(latencies) if latencies else None,
        "rows": rows,
    }


def _run_flashinfer_worker(token_values, warmup: int, rep: int, args) -> dict:
    with tempfile.NamedTemporaryFile(prefix="flashinfer_moe_", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--flashinfer-worker",
        "--mode",
        args.mode,
        "--preset",
        args.preset,
        "--tokens",
        ",".join(str(t) for t in token_values),
        "--seed",
        str(args.seed),
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--local-expert-offset",
        str(args.local_expert_offset),
        "--json-out",
        str(tmp_path),
    ]
    if args.check_flashinfer_correctness:
        cmd += ["--check-flashinfer-correctness"]
    if args.hidden_size is not None:
        cmd += ["--hidden-size", str(args.hidden_size)]
    if args.intermediate_size is not None:
        cmd += ["--intermediate-size", str(args.intermediate_size)]
    if args.local_num_experts is not None:
        cmd += ["--local-num-experts", str(args.local_num_experts)]

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.flashinfer_timeout_s,
            check=False,
            preexec_fn=_disable_core_dumps if os.name == "posix" else None,
        )
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            return json.loads(tmp_path.read_text())
        return {
            "status": "ERROR",
            "error": (
                f"worker exited with code {proc.returncode}; "
                f"stdout={proc.stdout[-1000:]!r}; stderr={proc.stderr[-1000:]!r}"
            ),
            "rows": [],
        }
    except subprocess.TimeoutExpired as exc:
        return {"status": "ERROR", "error": f"worker timeout after {exc.timeout}s", "rows": []}
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _attach_flashinfer_gate(result: dict, flashinfer_result: dict, target_speedup: float) -> None:
    result["flashinfer"] = flashinfer_result
    flash_status = flashinfer_result.get("status")
    if flash_status != "PASS":
        result["target_status"] = "BLOCKED"
        result["target_reason"] = flashinfer_result.get("error") or f"flashinfer_status={flash_status}"
        return

    by_tokens = {row.get("tokens"): row for row in flashinfer_result.get("rows") or []}
    speedups = []
    for row in result["rows"]:
        flash_row = by_tokens.get(row.get("tokens"))
        flash_us = (flash_row or {}).get("flashinfer_us")
        if flash_us is None:
            continue
        row["flashinfer_us"] = flash_us
        candidate_us = row.get("candidate_us")
        if candidate_us:
            row["speedup_vs_flashinfer"] = flash_us / candidate_us
            row["beats_flashinfer"] = row["speedup_vs_flashinfer"] > target_speedup
            speedups.append(row["speedup_vs_flashinfer"])

    if not speedups:
        candidate_errors = [
            row.get("error")
            for row in result["rows"]
            if row.get("status") == "ERROR" and row.get("error")
        ]
        if candidate_errors:
            result["target_status"] = "MISS"
            result["target_reason"] = (
                "Candidate did not produce comparable latency rows: " + candidate_errors[0]
            )
        else:
            result["target_status"] = "BLOCKED"
            result["target_reason"] = "No comparable candidate/FlashInfer latency rows"
        return
    result["target_speedup_vs_flashinfer"] = sum(speedups) / len(speedups)
    result["target_status"] = (
        "MET"
        if result["status"] == "PASS" and result["target_speedup_vs_flashinfer"] > target_speedup
        else "MISS"
    )


def evaluate(candidate_path: Path, token_values, warmup: int, rep: int, args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cand = _load_candidate(candidate_path)

    rows = []
    max_abs = 0.0
    max_rel = 0.0
    used_real_flashinfer_oracle = False
    for tokens in token_values:
        inputs = _make_inputs_for_case(args, tokens)
        real_inputs = _uses_real_packed_nvfp4(inputs)
        row = {
            "tokens": tokens,
            "preset": args.preset,
            "shape": _shape_from_inputs(inputs),
            "input_contract": "packed_nvfp4_flashinfer" if real_inputs else "local_oracle",
        }
        try:
            if real_inputs:
                used_real_flashinfer_oracle = True
                reference_first = (
                    os.environ.get("ATREX_FLASHINFER_REFERENCE_FIRST", "0").strip().lower()
                    in {"1", "true", "yes", "on"}
                )
                if reference_first:
                    ref = _first(_flashinfer_reference(inputs))
                    out = _first(cand.run(*inputs))
                else:
                    out = _first(cand.run(*inputs))
                    torch.cuda.synchronize()
                    ref = _first(_flashinfer_reference(inputs))
            else:
                ref = _first(run_reference(*inputs))
                out = _first(cand.run(*inputs))
            torch.cuda.synchronize()
            diff = (out.float() - ref.float()).abs()
            abs_err = float(diff.max().item())
            rel_err = float((diff / ref.float().abs().clamp_min(1e-3)).max().item())
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            row["max_abs"] = abs_err
            row["max_rel"] = rel_err
        except Exception as exc:
            row["status"] = "ERROR"
            row["error"] = _error_text(exc)
            rows.append(row)
            continue

        if abs_err > args.atol and rel_err > args.rtol:
            row["status"] = "FAIL"
            rows.append(row)
            continue

        row["status"] = "PASS"
        if args.mode == "profile":
            row["candidate_us"] = _bench(lambda: cand.run(*inputs), warmup, rep)
            if args.bench_torch_oracle:
                row["torch_oracle_us"] = _bench(
                    lambda: run_reference(*inputs), max(1, warmup // 2), max(3, rep // 3)
                )
                row["speedup_vs_torch_oracle"] = (
                    row["torch_oracle_us"] / row["candidate_us"]
                    if row["candidate_us"] > 0
                    else 0.0
                )
        rows.append(row)

    if all(row["status"] == "PASS" for row in rows):
        status = "PASS"
    elif any(row["status"] == "ERROR" for row in rows):
        status = "ERROR"
    else:
        status = "FAIL"
    latency_rows = [row["candidate_us"] for row in rows if row.get("candidate_us") is not None]
    result = {
        "status": status,
        "operator": "flashinfer.trtllm_fp4_block_scale_moe",
        "correctness_oracle": (
            "flashinfer.trtllm_fp4_block_scale_moe (real packed NVFP4 contract)"
            if used_real_flashinfer_oracle
            else "reference.run_reference (PyTorch oracle)"
        ),
        "performance_baseline": (
            "flashinfer.trtllm_fp4_block_scale_moe" if args.compare_flashinfer else None
        ),
        "arg_names": ARG_NAMES,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_latency_us": sum(latency_rows) / len(latency_rows) if latency_rows else None,
        "rows": rows,
    }
    try:
        del inputs
    except UnboundLocalError:
        pass
    torch.cuda.empty_cache()
    if args.mode == "profile" and args.compare_flashinfer:
        flashinfer_result = _run_flashinfer_worker(token_values, warmup, rep, args)
        _attach_flashinfer_gate(result, flashinfer_result, args.flashinfer_target_speedup)
        if args.require_flashinfer and result.get("target_status") == "BLOCKED":
            result["status"] = "BLOCKED"
    return result


def _parse_tokens(raw: str):
    return [int(item) for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="kernel.py")
    parser.add_argument("--mode", choices=("correctness", "profile"), default="correctness")
    parser.add_argument(
        "--preset",
        choices=("smoke", "qwen_micro", "qwen_tp2", "g11", "g8"),
        default="smoke",
    )
    # Default None: catalog presets (g11/g8) fall back to the catalog token
    # buckets; other presets fall back to a small smoke default.
    parser.add_argument("--tokens", default=None)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--local-num-experts", type=int)
    parser.add_argument("--local-expert-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=9)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--compare-flashinfer", action="store_true")
    parser.add_argument("--require-flashinfer", action="store_true")
    parser.add_argument("--flashinfer-worker", action="store_true")
    parser.add_argument(
        "--check-flashinfer-correctness",
        action="store_true",
        help="Also compare FlashInfer output against the PyTorch oracle inside the worker.",
    )
    parser.add_argument(
        "--bench-torch-oracle",
        action="store_true",
        help="Also time the slow PyTorch correctness oracle; never used as the performance baseline.",
    )
    parser.add_argument("--flashinfer-target-speedup", type=float, default=1.0)
    parser.add_argument("--flashinfer-timeout-s", type=float, default=120.0)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    if args.tokens is not None:
        token_values = _parse_tokens(args.tokens)
    else:
        from workload_shapes import get_shape, is_catalog_preset

        if is_catalog_preset(args.preset):
            token_values = list(get_shape(args.preset)["tokens"])
        else:
            token_values = [2, 4]
    if args.flashinfer_worker:
        result = _bench_flashinfer_in_worker(token_values, args.warmup, args.rep, args)
    else:
        result = evaluate(Path(args.kernel).resolve(), token_values, args.warmup, args.rep, args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    if result["status"] == "PASS":
        return 0
    if result["status"] == "BLOCKED":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
