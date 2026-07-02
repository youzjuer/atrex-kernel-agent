#!/usr/bin/env python3
"""Generate and smoke task08 v310 as the G11 GEMM2 BMM shape."""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

import kernel
from probe_prepared_gemm1_order import _decode_prepared_rows
from probe_task08_mpad_compile import (
    DEFAULT_TASK08_ROOT,
    EXT_SOURCE_NAME,
    HEADER_NAME,
    V310_SYMBOL,
    _compile_extension,
    _device_arch_list,
    _patch_header,
    _replace_once,
)
from probe_task08_runtime_smoke import _make_case, _valid_row_sample
from reference import dequantize_fp4_tensor
from workload_shapes import get_shape


DEFAULT_GEMM2_SYMBOL = V310_SYMBOL
ATREX_GEMM2_TILEGRID_SPLITCOL_SYMBOL = "atrex_gemm2_tilegrid_splitcolepi"
ATREX_GEMM2_TILEGRID_SPLITCOL_POSTSYNC_SYMBOL = (
    "atrex_gemm2_tilegrid_splitcolepi_postsync"
)
ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_SYMBOL = (
    "atrex_gemm2_tilegrid_splitcolepi_warpshuffle"
)
ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_SYMBOL = (
    "atrex_gemm2_tilegrid_splitcolepi_warpshuffle_postldsync"
)
ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_POSTSYNC_SYMBOL = (
    "atrex_gemm2_tilegrid_splitcolepi_warpshuffle_postldsync_postsync"
)
ATREX_GEMM2_TILEGRID_SPLITCOL_LDX32_WARPSHUFFLE_POSTLDSYNC_SYMBOL = (
    "atrex_gemm2_tilegrid_splitcolldx32_warpshuffle_postldsync"
)
DIAG_GEMM2_SYMBOLS = frozenset(
    {
        ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_SYMBOL,
        ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_SYMBOL,
        ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_POSTSYNC_SYMBOL,
        ATREX_GEMM2_TILEGRID_SPLITCOL_LDX32_WARPSHUFFLE_POSTLDSYNC_SYMBOL,
    }
)


def _bench(fn, warmup: int, rep: int) -> float:
    if rep <= 0:
        return float("nan")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return float(statistics.median(times))


def _trace(args: argparse.Namespace, message: str) -> None:
    if args.trace:
        print(f"[trace] {message}", flush=True)


def _copy_if_changed(src: Path, dst: Path) -> None:
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        return
    shutil.copy2(src, dst)


def _write_text_if_changed(dst: Path, text: str) -> None:
    if dst.exists() and dst.read_text() == text:
        return
    dst.write_text(text)


def _stable_work_root(kind: str, mpad: int) -> Path:
    return Path(tempfile.gettempdir()) / f"atrex_task08_{kind}_mpad{mpad}_src"


def _cpp_bool(value: bool) -> str:
    return "true" if value else "false"


def _gemm2_full_impl_wrapper(
    *,
    symbol: str,
    description: str,
    worker_env: str,
    ldx32: bool = False,
    stream_store: bool = True,
    post_load_sync: bool = False,
    x32_post_load_sync: bool = False,
    skip_post_epilogue_sync: bool = True,
) -> str:
    params = [
        ("false", "kPrefetchAllTma"),
        ("false", "kPrefetchAbTmaOnly"),
        ("false", "kPrefetchSfTmaOnly"),
        ("false", "kCircularAbDesc"),
        ("false", "kSfPairCadence"),
        ("false", "kSkipSplitScaleTmemFence"),
        ("true", "kBScaleArriveReady"),
        ("false", "kRank0SelfReadyArrive"),
        ("false", "kSkipEpilogue"),
        (_cpp_bool(ldx32), "kLdX32Epilogue"),
        (_cpp_bool(stream_store), "kStreamStoreEpilogue"),
        ("false", "kPeerSkipBScaleTma"),
        ("false", "kAfillMma"),
        ("true", "kSplitColumnEpilogue"),
        ("false", "kSkipInvalidRowsEpilogue"),
        ("false", "kSplitAbSfBarrierImpl"),
        ("false", "kDualLoadArriveImpl"),
        ("false", "kSingleTmaProducerImpl"),
        ("4", "kPipeStagesImpl"),
        ("false", "kQuadColumnEpilogue"),
        ("kThreads", "kBlockThreadsImpl"),
        ("true", "kDrainFinalSlotOnlyImpl"),
        ("true", "kSkipPostTileClusterSyncImpl"),
        ("false", "kPostTileLeaderGateImpl"),
        ("false", "kPostTileConsumerGateOnlyImpl"),
        ("false", "kDualPollReadyBScaleImpl"),
        ("false", "kBScaleWaitBeforeReadyImpl"),
        ("false", "kPostTileMmaConsumerGateOnlyImpl"),
        ("false", "kDelayPostTileMmaGateImpl"),
        ("false", "kDelayPostTileGateBeforeReadyImpl"),
        ("false", "kDelayPostTileGateBeforeScaleImpl"),
        (_cpp_bool(skip_post_epilogue_sync), "kSkipPostEpilogueSyncImpl"),
        ("false", "kOverlapProducerEpilogueImpl"),
        ("false", "kSmemCoalescedEpilogueImpl"),
        ("false", "kTmaStoreRank0EpilogueImpl"),
        ("false", "kTmaStoreRank0DeferredEpilogueImpl"),
        ("false", "kSkipPreEpilogueSyncWhenNoEpilogueImpl"),
        ("false", "kPreEpilogueMmaDoneGateImpl"),
        ("false", "kEpilogueNoStoreDiagImpl"),
        ("false", "kEpilogueZeroStoreDiagImpl"),
        ("false", "kEpilogueCoalescedZeroStoreDiagImpl"),
        ("false", "kBufferedOverlapEpilogueImpl"),
        ("false", "kSplitSmemCoalescedEpilogueImpl"),
        ("false", "kTmaStoreExpertEpilogueImpl"),
        ("false", "kTmaStoreExpertDeferredEpilogueImpl"),
        ("false", "kTmaStoreExpertSubtileDbEpilogueImpl"),
        ("false", "kTmaStoreExpertSubtileDbN128EpilogueImpl"),
        ("false", "kPreEpilogueWarpLeaderGateImpl"),
        ("false", "kTmaStoreExpertSubtileDbDelayIssueImpl"),
        ("false", "kTmaStoreExpertSubtileDbNamedBarrierImpl"),
        ("false", "kTmaStoreExpertSubtileDbN32EpilogueImpl"),
        ("false", "kPreEpilogueMmaDoneEpiOnlyGateImpl"),
        ("false", "kPostEpilogueNamedBarrierImpl"),
        ("false", "kWarpStagedCoalescedEpilogueImpl"),
        ("false", "kWarpStagedCoalescedEpilogue2GroupImpl"),
        ("true", "kWarpShuffleCoalescedEpilogueImpl"),
        (_cpp_bool(x32_post_load_sync), "kWarpShuffleX32PostLoadSyncImpl"),
        (_cpp_bool(post_load_sync), "kWarpShufflePostLoadSyncImpl"),
        ("false", "kWarpShufflePairPostLoadSyncImpl"),
        ("false", "kWarpShufflePairCoalescedStoreImpl"),
        ("false", "kWarpShufflePairCoalescedPostStoreSyncImpl"),
        ("false", "kWarpShufflePairCoalescedFinalSyncImpl"),
        ("-1", "kWarpShuffleStoreCachePolicyImpl"),
        ("false", "kWarpShuffleX32Pair16StoreImpl"),
        ("false", "kWarpShuffleX32Pair16PhaseColumnPairsImpl"),
        ("false", "kPreEpilogueAfterThreadSyncImpl"),
        ("false", "kWarpShufflePostStoreSyncImpl"),
        ("false", "kWarpShuffleFinalPostStoreSyncImpl"),
        ("false", "kPostTileEpilogueLastWarpGateImpl"),
        ("false", "kPostTileEpilogueWarpLeaderGateImpl"),
        ("false", "kPostTileEpilogueWarpLeaderNamedGateImpl"),
        ("false", "kPostTileEpilogueColumnGroupNamedGateImpl"),
        ("1", "kDefaultWorkerClusterDivisor"),
        ("false", "kPostTileRank0OnlyGateImpl"),
        ("false", "kUpperHalfNoOverlapEpilogueDiagImpl"),
        ("true", "kTileGridLaunchImpl"),
        ("false", "kEpiPipeTmaImmediateEpilogueImpl"),
        ("false", "kFixedM235AffineExpertOffsetsImpl"),
        ("false", "kWarpUniformTmaProducerElectImpl"),
        ("false", "kWarpUniformBScaleReadyElectImpl"),
        ("false", "kCtaRankSpecializedTid0ConsumerImpl"),
    ]
    param_lines = []
    for idx, (value, comment) in enumerate(params):
        suffix = "," if idx + 1 < len(params) else ">"
        param_lines.append(f"      {value}{suffix} // {comment}")
    params_text = "\n".join(param_lines)
    return f"""
torch::Tensor
{symbol}(
    torch::Tensor a_fp4,
    torch::Tensor a_scale_swizzled_u8,
    torch::Tensor w13_fp4,
    torch::Tensor w13_scale_swizzled_u8,
    torch::Tensor expert_offsets) {{
  return up_gate_fp4_sm103_umma_tma_u8_cta2_v142_tmaprefetch_impl<
{params_text}
      (
      a_fp4,
      a_scale_swizzled_u8,
      w13_fp4,
      w13_scale_swizzled_u8,
      expert_offsets,
      "{worker_env}",
      "{description}");
}}

"""


def _prepare_gemm1_tree(task08_root: Path, mpad: int) -> tuple[Path, dict[str, int]]:
    src_header = task08_root / "include" / HEADER_NAME
    src_ext = task08_root / "cuda_bmm" / EXT_SOURCE_NAME
    if not src_header.exists():
        raise FileNotFoundError(src_header)
    if not src_ext.exists():
        raise FileNotFoundError(src_ext)

    work_root = _stable_work_root("gemm1", mpad)
    include_dir = work_root / "include"
    cuda_dir = work_root / "cuda_bmm"
    include_dir.mkdir(parents=True, exist_ok=True)
    cuda_dir.mkdir(parents=True, exist_ok=True)

    for cuh in (task08_root / "include").glob("*.cuh"):
        if cuh.name == HEADER_NAME:
            continue
        _copy_if_changed(cuh, include_dir / cuh.name)
    patched_header, counts = _patch_header(src_header.read_text(), mpad)
    _write_text_if_changed(include_dir / HEADER_NAME, patched_header)
    _copy_if_changed(src_ext, cuda_dir / EXT_SOURCE_NAME)
    return work_root, counts


def _patch_gemm2_header(
    text: str,
    mpad: int,
    *,
    include_diag_symbols: bool = False,
) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    text, counts["comment_shape"] = _replace_once(
        text,
        "A [E=512, Mpad=235, K/2=2048] x B [E=512, N=2048, K/2=2048]",
        f"A [E=512, Mpad={mpad}, K/2=512] x B [E=512, N=4096, K/2=512]",
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
    text, counts["kbytes_const"] = _replace_once(
        text,
        "constexpr int kG11KBytes = 2048;",
        "constexpr int kG11KBytes = 512;",
    )
    text, counts["kscale_const"] = _replace_once(
        text,
        "constexpr int kG11KScale = 256;",
        "constexpr int kG11KScale = 64;",
    )
    text, counts["output_cols_const"] = _replace_once(
        text,
        "constexpr int kG11OutputCols = 2048;",
        "constexpr int kG11OutputCols = 4096;",
    )
    text, counts["fixed_mpad_static_assert"] = _replace_once(
        text,
        'static_assert(kG11RowsPerExpert == 235, "fixed-M235 path assumes task08 MPAD=235");',
        (
            f'static_assert(kG11RowsPerExpert == {mpad}, '
            f'"generated task08 GEMM2 MPAD={mpad} probe");'
        ),
    )
    custom_wrapper = f"""
torch::Tensor
{ATREX_GEMM2_TILEGRID_SPLITCOL_SYMBOL}(
    torch::Tensor a_fp4,
    torch::Tensor a_scale_swizzled_u8,
    torch::Tensor w13_fp4,
    torch::Tensor w13_scale_swizzled_u8,
    torch::Tensor expert_offsets) {{
  return up_gate_fp4_sm103_umma_tma_u8_cta2_v142_tmaprefetch_tilegrid_compat_impl<
      false, // kPrefetchAllTma
      false, // kPrefetchAbTmaOnly
      false, // kPrefetchSfTmaOnly
      false, // kCircularAbDesc
      false, // kSfPairCadence
      false, // kSkipSplitScaleTmemFence
      true,  // kBScaleArriveReady
      false, // kRank0SelfReadyArrive
      false, // kSkipEpilogue
      false, // kLdX32Epilogue
      false, // kStreamStoreEpilogue
      false, // kPeerSkipBScaleTma
      false, // kAfillMma
      true,  // kSplitColumnEpilogue
      false, // kSkipInvalidRowsEpilogue
      false, // kSplitAbSfBarrierImpl
      false, // kDualLoadArriveImpl
      false, // kSingleTmaProducerImpl
      4,
      false, // kQuadColumnEpilogue
      kThreads,
      true,  // kDrainFinalSlotOnlyImpl
      true,  // kSkipPostTileClusterSyncImpl
      false, // kPostTileLeaderGateImpl
      false, // kPostTileConsumerGateOnlyImpl
      false, // kDualPollReadyBScaleImpl
      false, // kBScaleWaitBeforeReadyImpl
      false, // kPostTileMmaConsumerGateOnlyImpl
      false, // kDelayPostTileMmaGateImpl
      false, // kDelayPostTileGateBeforeReadyImpl
      false, // kDelayPostTileGateBeforeScaleImpl
      true,  // kSkipPostEpilogueSyncImpl
      false, // kOverlapProducerEpilogueImpl
      false, // kSmemCoalescedEpilogueImpl
      false, // kTmaStoreRank0EpilogueImpl
      false, // kTmaStoreRank0DeferredEpilogueImpl
      false, // kSkipPreEpilogueSyncWhenNoEpilogueImpl
      false, // kPreEpilogueMmaDoneGateImpl
      false, // kEpilogueNoStoreDiagImpl
      false, // kEpilogueZeroStoreDiagImpl
      false, // kEpilogueCoalescedZeroStoreDiagImpl
      true,  // kTileGridLaunchImpl
      false>( // kEpiPipeTmaImmediateEpilogueImpl
      a_fp4,
      a_scale_swizzled_u8,
      w13_fp4,
      w13_scale_swizzled_u8,
      expert_offsets,
      "ATREX_GEMM2_TILEGRID_WORKER_CLUSTERS",
      "atrex GEMM2 tile-grid split-column epilogue probe");
}}

torch::Tensor
{ATREX_GEMM2_TILEGRID_SPLITCOL_POSTSYNC_SYMBOL}(
    torch::Tensor a_fp4,
    torch::Tensor a_scale_swizzled_u8,
    torch::Tensor w13_fp4,
    torch::Tensor w13_scale_swizzled_u8,
    torch::Tensor expert_offsets) {{
  return up_gate_fp4_sm103_umma_tma_u8_cta2_v142_tmaprefetch_tilegrid_compat_impl<
      false, // kPrefetchAllTma
      false, // kPrefetchAbTmaOnly
      false, // kPrefetchSfTmaOnly
      false, // kCircularAbDesc
      false, // kSfPairCadence
      false, // kSkipSplitScaleTmemFence
      true,  // kBScaleArriveReady
      false, // kRank0SelfReadyArrive
      false, // kSkipEpilogue
      false, // kLdX32Epilogue
      false, // kStreamStoreEpilogue
      false, // kPeerSkipBScaleTma
      false, // kAfillMma
      true,  // kSplitColumnEpilogue
      false, // kSkipInvalidRowsEpilogue
      false, // kSplitAbSfBarrierImpl
      false, // kDualLoadArriveImpl
      false, // kSingleTmaProducerImpl
      4,
      false, // kQuadColumnEpilogue
      kThreads,
      true,  // kDrainFinalSlotOnlyImpl
      true,  // kSkipPostTileClusterSyncImpl
      false, // kPostTileLeaderGateImpl
      false, // kPostTileConsumerGateOnlyImpl
      false, // kDualPollReadyBScaleImpl
      false, // kBScaleWaitBeforeReadyImpl
      false, // kPostTileMmaConsumerGateOnlyImpl
      false, // kDelayPostTileMmaGateImpl
      false, // kDelayPostTileGateBeforeReadyImpl
      false, // kDelayPostTileGateBeforeScaleImpl
      false, // kSkipPostEpilogueSyncImpl
      false, // kOverlapProducerEpilogueImpl
      false, // kSmemCoalescedEpilogueImpl
      false, // kTmaStoreRank0EpilogueImpl
      false, // kTmaStoreRank0DeferredEpilogueImpl
      false, // kSkipPreEpilogueSyncWhenNoEpilogueImpl
      false, // kPreEpilogueMmaDoneGateImpl
      false, // kEpilogueNoStoreDiagImpl
      false, // kEpilogueZeroStoreDiagImpl
      false, // kEpilogueCoalescedZeroStoreDiagImpl
      true,  // kTileGridLaunchImpl
      false>( // kEpiPipeTmaImmediateEpilogueImpl
      a_fp4,
      a_scale_swizzled_u8,
      w13_fp4,
      w13_scale_swizzled_u8,
      expert_offsets,
      "ATREX_GEMM2_TILEGRID_POSTSYNC_WORKER_CLUSTERS",
      "atrex GEMM2 tile-grid split-column post-sync epilogue probe");
}}

"""
    if include_diag_symbols:
        custom_wrapper += _gemm2_full_impl_wrapper(
            symbol=ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_SYMBOL,
            description=(
                "atrex GEMM2 diagnostic tile-grid split-column warp-shuffle streaming "
                "epilogue probe"
            ),
            worker_env="ATREX_GEMM2_TILEGRID_WARPSHUFFLE_WORKER_CLUSTERS",
            ldx32=False,
            stream_store=True,
            post_load_sync=False,
            x32_post_load_sync=False,
            skip_post_epilogue_sync=True,
        )
        custom_wrapper += _gemm2_full_impl_wrapper(
            symbol=ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_SYMBOL,
            description=(
                "atrex GEMM2 diagnostic tile-grid split-column warp-shuffle streaming "
                "epilogue post-load-sync probe"
            ),
            worker_env="ATREX_GEMM2_TILEGRID_WARPSHUFFLE_POSTLDSYNC_WORKER_CLUSTERS",
            ldx32=False,
            stream_store=True,
            post_load_sync=True,
            x32_post_load_sync=False,
            skip_post_epilogue_sync=True,
        )
        custom_wrapper += _gemm2_full_impl_wrapper(
            symbol=ATREX_GEMM2_TILEGRID_SPLITCOL_LDX32_WARPSHUFFLE_POSTLDSYNC_SYMBOL,
            description=(
                "atrex GEMM2 diagnostic tile-grid split-column ldx32 warp-shuffle "
                "streaming epilogue post-load-sync probe"
            ),
            worker_env="ATREX_GEMM2_TILEGRID_LDX32_WARPSHUFFLE_POSTLDSYNC_WORKER_CLUSTERS",
            ldx32=True,
            stream_store=True,
            post_load_sync=False,
            x32_post_load_sync=True,
            skip_post_epilogue_sync=True,
        )
        custom_wrapper += _gemm2_full_impl_wrapper(
            symbol=ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_POSTSYNC_SYMBOL,
            description=(
                "atrex GEMM2 diagnostic tile-grid split-column warp-shuffle streaming "
                "epilogue with post-load sync and post-epilogue sync probe"
            ),
            worker_env="ATREX_GEMM2_TILEGRID_WARPSHUFFLE_POSTLDSYNC_POSTSYNC_WORKER_CLUSTERS",
            ldx32=False,
            stream_store=True,
            post_load_sync=True,
            x32_post_load_sync=False,
            skip_post_epilogue_sync=False,
        )
    text, counts["atrex_tilegrid_splitcol_wrapper"] = _replace_once(
        text,
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {",
        custom_wrapper + "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {",
    )
    bind = f'''  m.def(
      "{ATREX_GEMM2_TILEGRID_SPLITCOL_SYMBOL}",
      &{ATREX_GEMM2_TILEGRID_SPLITCOL_SYMBOL},
      "atrex generated task08 GEMM2 tile-grid split-column epilogue probe");
  m.def(
      "{ATREX_GEMM2_TILEGRID_SPLITCOL_POSTSYNC_SYMBOL}",
      &{ATREX_GEMM2_TILEGRID_SPLITCOL_POSTSYNC_SYMBOL},
      "atrex generated task08 GEMM2 tile-grid split-column post-sync epilogue probe");
'''
    if include_diag_symbols:
        bind += f'''  m.def(
      "{ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_SYMBOL}",
      &{ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_SYMBOL},
      "atrex generated task08 GEMM2 diagnostic tile-grid split-column warp-shuffle epilogue probe");
  m.def(
      "{ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_SYMBOL}",
      &{ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_SYMBOL},
      "atrex generated task08 GEMM2 diagnostic tile-grid split-column warp-shuffle post-load-sync epilogue probe");
  m.def(
      "{ATREX_GEMM2_TILEGRID_SPLITCOL_LDX32_WARPSHUFFLE_POSTLDSYNC_SYMBOL}",
      &{ATREX_GEMM2_TILEGRID_SPLITCOL_LDX32_WARPSHUFFLE_POSTLDSYNC_SYMBOL},
      "atrex generated task08 GEMM2 diagnostic tile-grid split-column ldx32 warp-shuffle post-load-sync epilogue probe");
  m.def(
      "{ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_POSTSYNC_SYMBOL}",
      &{ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_POSTSYNC_SYMBOL},
      "atrex generated task08 GEMM2 diagnostic tile-grid split-column warp-shuffle post-load-sync post-sync epilogue probe");
'''
    text, counts["atrex_tilegrid_splitcol_bind"] = _replace_once(
        text,
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n",
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n" + bind,
    )
    return text, counts


def _prepare_gemm2_tree(
    task08_root: Path,
    mpad: int,
    *,
    include_diag_symbols: bool = False,
) -> tuple[Path, dict[str, int]]:
    src_header = task08_root / "include" / HEADER_NAME
    src_ext = task08_root / "cuda_bmm" / EXT_SOURCE_NAME
    if not src_header.exists():
        raise FileNotFoundError(src_header)
    if not src_ext.exists():
        raise FileNotFoundError(src_ext)

    work_root = _stable_work_root("gemm2_diag" if include_diag_symbols else "gemm2", mpad)
    include_dir = work_root / "include"
    cuda_dir = work_root / "cuda_bmm"
    include_dir.mkdir(parents=True, exist_ok=True)
    cuda_dir.mkdir(parents=True, exist_ok=True)

    for cuh in (task08_root / "include").glob("*.cuh"):
        if cuh.name == HEADER_NAME:
            continue
        _copy_if_changed(cuh, include_dir / cuh.name)
    patched_header, counts = _patch_gemm2_header(
        src_header.read_text(),
        mpad,
        include_diag_symbols=include_diag_symbols,
    )
    _write_text_if_changed(include_dir / HEADER_NAME, patched_header)
    _copy_if_changed(src_ext, cuda_dir / EXT_SOURCE_NAME)
    return work_root, counts


def _compile_gemm2_extension(
    work_root: Path,
    mpad: int,
    verbose: bool,
    *,
    extension_tag: str = "",
):
    arch_list = _device_arch_list()
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    arch_suffix = arch_list.replace(".", "_").replace("+", "p")
    tag = f"_{extension_tag}" if extension_tag else ""
    ext_name = f"task08_gemm2{tag}_mpad{mpad}_v310_probe_sm{arch_suffix}"
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


def _sample_cols(cols: int, limit: int, device: torch.device) -> torch.Tensor:
    if limit <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    if limit >= cols:
        return torch.arange(cols, device=device, dtype=torch.long)
    if limit == 1:
        return torch.zeros((1,), device=device, dtype=torch.long)
    return torch.linspace(0, cols - 1, steps=limit, device=device).to(torch.long)


@torch.no_grad()
def _compare_sample(
    *,
    gemm2_out: torch.Tensor,
    mid_q: torch.Tensor,
    mid_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_scale: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    intermediate_size: int,
) -> dict[str, float | int | bool]:
    mid = dequantize_fp4_tensor(
        mid_q.reshape(-1, intermediate_size // 2).index_select(0, rows),
        mid_scale.reshape(-1, intermediate_size // 16).index_select(0, rows),
    )
    experts = rows // mid_q.shape[1]
    actual = gemm2_out.index_select(0, rows).index_select(1, cols).float()
    expected_cols = []
    for idx, col in enumerate(cols.detach().cpu().tolist()):
        expert_rows: list[torch.Tensor] = []
        for row_idx, expert in enumerate(experts.detach().cpu().tolist()):
            decoded = _decode_prepared_rows(
                w2_fp4[int(expert)],
                w2_scale[int(expert)],
                torch.tensor([int(col)], device=w2_fp4.device, dtype=torch.long),
            )[0]
            expert_rows.append((mid[row_idx] * decoded).sum())
        expected_cols.append(torch.stack(expert_rows))
    expected = torch.stack(expected_cols, dim=1).to(torch.bfloat16).float()
    diff = (actual - expected).abs()
    rel = diff / expected.abs().clamp_min(1e-3)
    return {
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "rows": int(rows.numel()),
        "cols": int(cols.numel()),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
    }


def run_probe(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    started = time.perf_counter()
    shape = get_shape(args.preset)
    if shape["dtype"] != "nvfp4":
        raise NotImplementedError("task08 GEMM2 probe targets packed NVFP4")
    task08_root = Path(args.task08_root).resolve()
    gemm2_symbol = args.gemm2_symbol or DEFAULT_GEMM2_SYMBOL
    include_diag_symbols = gemm2_symbol in DIAG_GEMM2_SYMBOLS

    _trace(args, "compile GEMM1 extension")
    gemm1_root, gemm1_patch_counts = _prepare_gemm1_tree(task08_root, args.mpad)
    gemm1_module = _compile_extension(gemm1_root, args.mpad, args.verbose_build)
    _trace(args, "compile GEMM2 extension")
    gemm2_root, gemm2_patch_counts = _prepare_gemm2_tree(
        task08_root,
        args.mpad,
        include_diag_symbols=include_diag_symbols,
    )
    gemm2_module = _compile_gemm2_extension(
        gemm2_root,
        args.mpad,
        args.verbose_build,
        extension_tag="diag" if include_diag_symbols else "",
    )
    gemm1_fn = getattr(gemm1_module, V310_SYMBOL)
    if not hasattr(gemm2_module, gemm2_symbol):
        raise RuntimeError(f"compiled GEMM2 module missing symbol {gemm2_symbol}")
    gemm2_fn = getattr(gemm2_module, gemm2_symbol)

    _trace(args, "build real G11 case")
    case = _make_case(args)
    _trace(args, "run GEMM1 BMM")
    gemm1_out = gemm1_fn(
        case["a_fp4"],
        case["a_scale_swizzled_u8"],
        case["w13_fp4"],
        case["w13_scale_swizzled_u8"],
        case["expert_offsets"],
    )
    torch.cuda.synchronize()
    intermediate_size = int(shape["intermediate_size"])
    hidden_size = int(shape["hidden_size"])
    _trace(args, "run SwiGLU requant")
    mid_q, mid_scale, mid_scale_swizzled = kernel.swiglu_requant_from_bmm(
        gemm1_out,
        case["expert_counts"],
        padded_rows=int(case["padded_rows"]),
        intermediate_size=intermediate_size,
    )
    torch.cuda.synchronize()
    mid_q_flat = mid_q.reshape(int(case["local_num_experts"]) * int(case["padded_rows"]), -1)
    _trace(args, "run GEMM2 BMM")
    gemm2_out = gemm2_fn(
        mid_q_flat.contiguous(),
        mid_scale_swizzled.view(torch.uint8).reshape(int(case["local_num_experts"]), -1),
        case["w2_fp4"],
        case["w2_scale_swizzled_u8"],
        case["expert_offsets"],
    )
    torch.cuda.synchronize()
    _trace(args, "GEMM2 BMM returned")

    shape_ok = tuple(gemm2_out.shape) == (
        int(case["local_num_experts"]) * int(case["padded_rows"]),
        hidden_size,
    )
    rows = (
        torch.empty((0,), device=gemm2_out.device, dtype=torch.long)
        if args.check_rows <= 0
        else _valid_row_sample(
            case["expert_counts"],
            padded_rows=int(case["padded_rows"]),
            limit=args.check_rows,
            device=gemm2_out.device,
        )
    )
    cols = _sample_cols(hidden_size, args.check_cols, gemm2_out.device)
    finite_sample = True
    if rows.numel() > 0 and cols.numel() > 0:
        finite_sample = bool(torch.isfinite(gemm2_out.index_select(0, rows).index_select(1, cols)).all())
    compare = None
    if not args.skip_compare:
        _trace(args, "compare GEMM2 sample")
        compare = _compare_sample(
            gemm2_out=gemm2_out,
            mid_q=mid_q,
            mid_scale=mid_scale,
            w2_fp4=case["w2_fp4"],
            w2_scale=case["w2_scale_swizzled_u8"].view(torch.float8_e4m3fn).reshape(
                int(case["local_num_experts"]), hidden_size, intermediate_size // 16
            ),
            rows=rows,
            cols=cols,
            intermediate_size=intermediate_size,
        )
    _trace(args, "benchmark GEMM2 BMM" if not args.skip_bench else "skip GEMM2 benchmark")
    gemm2_us = (
        float("nan")
        if args.skip_bench
        else _bench(
            lambda: gemm2_fn(
                mid_q_flat,
                mid_scale_swizzled.view(torch.uint8).reshape(int(case["local_num_experts"]), -1),
                case["w2_fp4"],
                case["w2_scale_swizzled_u8"],
                case["expert_offsets"],
            ),
            args.warmup,
            args.rep,
        )
    )
    status = (
        "PASS"
        if shape_ok
        and finite_sample
        and (
            args.skip_compare
            or (
                compare is not None
                and compare["finite"]
                and compare["max_abs"] <= args.max_abs
                and compare["max_rel"] <= args.max_rel
            )
        )
        else "FAIL"
    )
    return {
        "status": status,
        "operator": "generated task08 SM103 FP4 GEMM2 BMM runtime probe",
        "preset": args.preset,
        "tokens": int(case["tokens"]),
        "task08_root": str(task08_root),
        "work_roots": {
            "gemm1": str(gemm1_root),
            "gemm2": str(gemm2_root),
        },
        "compiled_symbol": {
            "gemm1": V310_SYMBOL,
            "gemm2": gemm2_symbol,
        },
        "patch_counts": {
            "gemm1": gemm1_patch_counts,
            "gemm2": gemm2_patch_counts,
        },
        "shape": {
            "padded_rows": int(case["padded_rows"]),
            "mid_q_flat": tuple(mid_q_flat.shape),
            "mid_scale_swizzled": tuple(mid_scale_swizzled.shape),
            "w2_fp4": tuple(case["w2_fp4"].shape),
            "w2_scale_swizzled_u8": tuple(case["w2_scale_swizzled_u8"].shape),
            "gemm2_out": tuple(gemm2_out.shape),
        },
        "checks": {
            "out_shape": "PASS" if shape_ok else "FAIL",
            "finite_sample": "PASS" if finite_sample else "FAIL",
            "sample_compare": compare,
            "sample_compare_skipped": bool(args.skip_compare),
            "thresholds": {
                "max_abs": args.max_abs,
                "max_rel": args.max_rel,
            },
        },
        "latency_us": {
            "task08_generated_gemm2_bmm_only": gemm2_us,
        },
        "elapsed_s": time.perf_counter() - started,
        "next_step": "Add final top-k weighted scatter/reorder from physical GEMM2 columns to token-major output.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task08-root", default=str(DEFAULT_TASK08_ROOT))
    parser.add_argument("--preset", default="g11")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--mpad", type=int, default=238)
    parser.add_argument("--tile-tokens-dim", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=3)
    parser.add_argument("--check-rows", type=int, default=16)
    parser.add_argument("--check-cols", type=int, default=32)
    parser.add_argument("--max-abs", type=float, default=0.75)
    parser.add_argument("--max-rel", type=float, default=8.0)
    parser.add_argument("--gemm2-symbol", default=DEFAULT_GEMM2_SYMBOL)
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    try:
        result = run_probe(args)
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "FAIL",
            "operator": "generated task08 SM103 FP4 GEMM2 BMM runtime probe",
            "preset": args.preset,
            "mpad": int(args.mpad),
            "error": f"{type(exc).__name__}: {exc}",
        }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
