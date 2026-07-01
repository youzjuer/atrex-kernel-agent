#!/usr/bin/env python3
"""Generate and compile a task08 BMM variant for the current routed Mpad.

The task08 SM103 FP4 BMM source is large and external to this repository.  This
probe keeps the source external, patches a temporary copy to use a requested
``kG11RowsPerExpert``, and compiles it as a PyTorch extension.  The result is
evidence for whether the hand-written BMM can move past the fixed-M235 contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


DEFAULT_TASK08_ROOT = Path(
    "/home/youchunbo/code/sumu/omoExplore/proj/proj_019_moe_workload_op_opt"
    "/assets/task_08_cuda_fp4_bmm_sm103"
)
HEADER_NAME = "task08_cuda_fp4_bmm_sm103_v83.cuh"
EXT_SOURCE_NAME = "task08_cuda_fp4_bmm_ext.cu"
V310_SYMBOL = (
    "up_gate_fp4_sm103_umma_tma_u8_cta2_v310_cta_rank_specialized_tid0_consumer_from_v309"
)


def _device_arch_list() -> str:
    if not torch.cuda.is_available():
        return "10.3a"
    major, minor = torch.cuda.get_device_capability()
    suffix = "a" if major >= 10 else ""
    return f"{major}.{minor}{suffix}"


def _replace_once(text: str, old: str, new: str) -> tuple[str, int]:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence of {old!r}, found {count}")
    return text.replace(old, new), count


def _patch_header(text: str, mpad: int) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    text, counts["comment_mpad"] = _replace_once(
        text,
        "A [E=512, Mpad=235, K/2=2048]",
        f"A [E=512, Mpad={mpad}, K/2=2048]",
    )
    text, counts["flattened_rows_comment"] = _replace_once(
        text,
        "Flattened A/output rows are E * Mpad = 120320.",
        f"Flattened A/output rows are E * Mpad = {512 * mpad}.",
    )
    text, counts["rows_const"] = _replace_once(
        text,
        "constexpr int kG11Rows = 512 * 235;",
        f"constexpr int kG11Rows = 512 * {mpad};",
    )
    text, counts["rows_per_expert_const"] = _replace_once(
        text,
        "constexpr int kG11RowsPerExpert = 235;",
        f"constexpr int kG11RowsPerExpert = {mpad};",
    )
    text, counts["fixed_mpad_static_assert"] = _replace_once(
        text,
        'static_assert(kG11RowsPerExpert == 235, "fixed-M235 path assumes task08 MPAD=235");',
        (
            f'static_assert(kG11RowsPerExpert == {mpad}, '
            f'"generated task08 MPAD={mpad} probe");'
        ),
    )
    return text, counts


def _prepare_tree(task08_root: Path, mpad: int) -> tuple[Path, dict[str, int]]:
    src_header = task08_root / "include" / HEADER_NAME
    src_ext = task08_root / "cuda_bmm" / EXT_SOURCE_NAME
    if not src_header.exists():
        raise FileNotFoundError(src_header)
    if not src_ext.exists():
        raise FileNotFoundError(src_ext)

    work_root = Path(tempfile.mkdtemp(prefix=f"task08_mpad{mpad}_"))
    include_dir = work_root / "include"
    cuda_dir = work_root / "cuda_bmm"
    include_dir.mkdir(parents=True)
    cuda_dir.mkdir(parents=True)

    for cuh in (task08_root / "include").glob("*.cuh"):
        shutil.copy2(cuh, include_dir / cuh.name)
    patched_header, counts = _patch_header(src_header.read_text(), mpad)
    (include_dir / HEADER_NAME).write_text(patched_header)
    shutil.copy2(src_ext, cuda_dir / EXT_SOURCE_NAME)
    return work_root, counts


def _compile_extension(work_root: Path, mpad: int, verbose: bool):
    arch_list = _device_arch_list()
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    arch_suffix = arch_list.replace(".", "_").replace("+", "p")
    ext_name = f"task08_mpad{mpad}_v310_probe_sm{arch_suffix}"
    build_dir = Path(tempfile.gettempdir()) / ext_name
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=ext_name,
        sources=[str(work_root / "cuda_bmm" / EXT_SOURCE_NAME)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-gencode=arch=compute_103a,code=sm_103a",
            "-DNDEBUG",
        ],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
        verbose=verbose,
    )


def run_probe(args: argparse.Namespace) -> dict:
    task08_root = Path(args.task08_root).resolve()
    started = time.perf_counter()
    work_root = None
    try:
        work_root, patch_counts = _prepare_tree(task08_root, args.mpad)
        module = _compile_extension(work_root, args.mpad, args.verbose_build)
        has_v310 = hasattr(module, V310_SYMBOL)
        status = "PASS" if has_v310 else "FAIL"
        error = None if has_v310 else f"compiled module missing symbol {V310_SYMBOL}"
    except Exception as exc:  # noqa: BLE001
        patch_counts = {}
        has_v310 = False
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    return {
        "status": status,
        "operator": "generated task08 SM103 FP4 BMM compile probe",
        "task08_root": str(task08_root),
        "mpad": int(args.mpad),
        "work_root": str(work_root) if work_root is not None else None,
        "patch_counts": patch_counts,
        "compiled_symbol": V310_SYMBOL,
        "has_compiled_symbol": has_v310,
        "elapsed_s": elapsed,
        "error": error,
        "next_step": (
            "Wire this generated Mpad into a runtime smoke using current hidden_bmm and prepared GEMM1 weights."
            if status == "PASS"
            else "Inspect the compile error before attempting runtime integration."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task08-root", default=str(DEFAULT_TASK08_ROOT))
    parser.add_argument("--mpad", type=int, default=238)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    result = run_probe(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
