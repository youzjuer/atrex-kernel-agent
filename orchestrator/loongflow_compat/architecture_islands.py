"""Architecture-aware PCA routing and fixed-interval island exchange for PES."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger("atrex.pes_architecture")

ANALYSIS_VERSION = 2
PCA_COMPONENTS = 3

FEATURE_NAMES = (
    "warp_specialization",
    "cluster_launch",
    "tma",
    "cluster_tma_broadcast",
    "wgmma",
    "cp_async",
    "cooperative_grid_sync",
    "persistent_kernel",
    "cuda_graph",
    "shared_memory_histogram",
    "hierarchical_histogram",
    "bank_replicated_histogram",
    "per_cta_offsets",
    "radix_sort",
    "cub_radix_sort",
    "block_radix_sort",
    "bitwise_radix_sort",
    "expert_parallel_scan",
    "stable_rank",
    "radix_passes",
    "radix_bits",
    "atomic_ops",
    "warp_collectives",
    "cute_dsl",
    "vectorized_io",
    "barrier_pipeline",
    "shared_memory_uses",
    "kernel_definitions",
    "launch_sites",
    "code_chars",
    "code_lines",
)

_LOG_SCALED_FEATURES = {
    "atomic_ops",
    "warp_collectives",
    "vectorized_io",
    "barrier_pipeline",
    "shared_memory_uses",
    "kernel_definitions",
    "launch_sites",
    "code_chars",
    "code_lines",
}

_SEMANTIC_ANCHORS = {
    "warp_specialization": 0,
    "cluster_tma_broadcast": 1,
    "tma_wgmma": 2,
    "persistent_cooperative": 3,
    "cute_dsl": 4,
    "async_pipeline": 5,
    "cub_radix_sort": 6,
    "bitwise_radix_sort": 6,
    "expert_parallel_scan": 7,
    "hierarchical_histogram": 7,
    "graph_histogram": 7,
}
MIN_ARCHITECTURE_ISLANDS = max(_SEMANTIC_ANCHORS.values()) + 1

ISLAND_PROFILES = (
    "warp_specialization",
    "cluster_tma_broadcast",
    "tma_wgmma",
    "persistent_cooperative",
    "cute_dsl",
    "async_pipeline",
    "radix_family",
    "hierarchical_stable_bucket",
)

_TAG_FEATURES = tuple(
    name
    for name in FEATURE_NAMES
    if name
    not in {
        "radix_passes",
        "radix_bits",
        "atomic_ops",
        "warp_collectives",
        "vectorized_io",
        "barrier_pipeline",
        "shared_memory_uses",
        "kernel_definitions",
        "launch_sites",
        "code_chars",
        "code_lines",
    }
)


def _without_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    source = re.sub(r"//[^\n]*", " ", source)
    return re.sub(
        r"(?m)^\s*#(?!\s*(?:include|define|if|ifdef|ifndef|elif|else|endif|pragma)\b).*$",
        " ",
        source,
    )


def _count(code: str, pattern: str) -> int:
    return len(re.findall(pattern, code, flags=re.IGNORECASE | re.MULTILINE))


def _present(code: str, pattern: str) -> int:
    return int(bool(re.search(pattern, code, flags=re.IGNORECASE | re.MULTILINE)))


def _integer_constant(code: str, names: str) -> int:
    match = re.search(
        rf"\b(?:{names})\b\s*(?:=|:)\s*(\d+)",
        code,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return int(match.group(1)) if match else 0


def extract_architecture_features(source: str) -> dict[str, int]:
    """Extract named CUDA/CuTe architecture signals before PCA projection."""
    source = source if isinstance(source, str) else str(source or "")
    code = _without_comments(source)

    cluster_launch = _present(
        code,
        r"__cluster_dims__|cudaLaunchAttributeClusterDimension|clusterDim|"
        r"this_cluster\s*\(|cluster_group|cluster\.sync",
    )
    tma = _present(
        code,
        r"\bTMA\b|\btma_[A-Za-z0-9_]*|CUtensorMap|cuTensorMapEncode|"
        r"cp\.async\.bulk\.tensor|make_tiled_tma_atom|SM9\d_TMA",
    )
    multicast = _present(
        code,
        r"multicast|broadcast|cta[_ ]?mask|cluster[_ ]?mask|mcast",
    )
    cp_async = _present(
        code,
        r"cp\.async(?!\.bulk\.tensor)|__pipeline_memcpy_async|" r"cuda::memcpy_async",
    )
    warp_branches = _count(
        code,
        r"(?:warp[_ ]?(?:id|idx)|warpId)\s*(?:==|!=|<=|>=|<|>)",
    )
    explicit_warp_roles = _present(
        code,
        r"warp[_ ]?speciali[sz]|producer[_ ]?warp|consumer[_ ]?warp|"
        r"warp[_ ]?role|elect_one_sync",
    )
    producer_consumer = _present(code, r"\bproducer\b") and _present(
        code, r"\bconsumer\b"
    )
    warp_specialization = int(
        explicit_warp_roles or (warp_branches >= 2 and producer_consumer)
    )

    cooperative = _present(
        code,
        r"this_grid\s*\(|grid\.sync\s*\(|cudaLaunchCooperativeKernel|"
        r"cooperative_groups::grid_group",
    )
    persistent = _present(
        code,
        r"\bpersistent\b|while\s*\([^)]*(?:atomic|work|task)|"
        r"for\s*\([^;]*;[^;]*;[^)]*\+=\s*(?:gridDim|cute\.size)",
    )
    graph = _present(
        code,
        r"cudaGraph|cudaStreamBeginCapture|cudaGraphExec|graph_cache|graphCache",
    )
    shared_uses = _count(code, r"__shared__|extern\s+__shared__|SharedStorage")
    histogram = _present(code, r"\bhist(?:ogram)?\w*|\bbins?\b")
    per_cta_offsets = _present(
        code,
        r"per[_ ]?cta|cta[_ ]?(?:base|offset)|block[_ ]?offset|"
        r"tile[_ ]?(?:base|offset|count)",
    )
    bank_replicated_histogram = int(
        bool(histogram)
        and bool(
            _present(
                code,
                r"local_cnts_rep|hist(?:ogram)?[_ ]?(?:rep|bank)|"
                r"(?:NUM|k)[_ ]?(?:HIST[_ ]?)?BANKS|warp[_ ]?bank",
            )
        )
    )
    hierarchical_histogram = int(
        bool(histogram)
        and bool(
            per_cta_offsets
            or _present(
                code,
                r"partial[_ ]?(?:hist|count)|global[_ ]?(?:hist|count)|"
                r"prefix[_ ]?(?:hist|count)|merge[_ ]?(?:hist|count)",
            )
        )
    )
    cub_radix_sort = _present(code, r"cub::DeviceRadixSort|DeviceRadixSort")
    block_radix_sort = _present(code, r"cub::BlockRadixSort|BlockRadixSort")
    explicit_radix = _present(
        code,
        r"\bradix[_ ]?(?:sort|pass|bits?|digit)|\bRadixSort\b|"
        r"begin[_ ]?bit|end[_ ]?bit",
    )
    bitwise_radix_sort = int(
        bool(explicit_radix)
        and not bool(cub_radix_sort or block_radix_sort)
        and bool(
            _present(
                code,
                r"bitwise[_ ]?radix|radix[_ ]?(?:pass|bits?|digit)|"
                r"digit[_ ]?bits?|\bbit[_ ]?pass",
            )
        )
    )
    radix_sort = int(bool(cub_radix_sort or block_radix_sort or explicit_radix))
    radix_bits = _integer_constant(
        code, r"RADIX_BITS|kRadixBits|radix_bits|DIGIT_BITS|kDigitBits"
    )
    if radix_sort and radix_bits == 0 and (cub_radix_sort or block_radix_sort):
        radix_bits = 8
    radix_passes = _integer_constant(
        code, r"RADIX_PASSES|kRadixPasses|radix_passes|NUM_PASSES|kNumPasses"
    )
    if bitwise_radix_sort and radix_passes == 0 and radix_bits > 0:
        radix_passes = max(1, math.ceil(8 / radix_bits))
    explicit_expert_parallel = _present(
        code,
        r"expert[_ ]?parallel|expert[_ ]?(?:owned|scan)",
    )
    expert_owned_scan = all(
        (
            _present(
                code,
                r"(?:const\s+)?int\s+expert\s*=\s*blockIdx\.x",
            ),
            _present(
                code,
                r"topk\s*\[[^]]+\]\s*==\s*expert|" r"expert\s*==\s*topk\s*\[[^]]+\]",
            ),
            _present(code, r"sorted_token_indices\s*\["),
        )
    )
    expert_parallel_scan = int(bool(explicit_expert_parallel or expert_owned_scan))
    stable_rank = _present(
        code,
        r"stable[_ ]?(?:rank|scan|scatter)|rank[_ ]?in[_ ]?(?:warp|block)|"
        r"local[_ ]?rank|exclusive[_ ]?(?:scan|prefix)",
    )
    atomic_ops = _count(
        code,
        r"\batomic(?:Add|Sub|Exch|Min|Max|Inc|Dec|CAS|And|Or|Xor)\s*\(",
    )
    warp_collectives = _count(
        code,
        r"__(?:shfl|match|ballot|syncwarp|activemask)\w*\s*\(|"
        r"coalesced_group|tiled_partition",
    )
    cute_dsl = _present(
        code,
        r"(?:from|import)\s+cutlass(?:\.cute)?|@cute\.(?:kernel|jit)|"
        r"cute\.compile|cute\.Tensor",
    )

    return {
        "warp_specialization": warp_specialization,
        "cluster_launch": cluster_launch,
        "tma": tma,
        "cluster_tma_broadcast": int(cluster_launch and tma and multicast),
        "wgmma": _present(code, r"\bwgmma\b|warp[_ ]?group[_ ]?mma|warpgroup"),
        "cp_async": cp_async,
        "cooperative_grid_sync": cooperative,
        "persistent_kernel": int(persistent),
        "cuda_graph": graph,
        "shared_memory_histogram": int(bool(shared_uses) and bool(histogram)),
        "hierarchical_histogram": hierarchical_histogram,
        "bank_replicated_histogram": bank_replicated_histogram,
        "per_cta_offsets": per_cta_offsets,
        "radix_sort": radix_sort,
        "cub_radix_sort": cub_radix_sort,
        "block_radix_sort": block_radix_sort,
        "bitwise_radix_sort": bitwise_radix_sort,
        "expert_parallel_scan": expert_parallel_scan,
        "stable_rank": stable_rank,
        "radix_passes": radix_passes,
        "radix_bits": radix_bits,
        "atomic_ops": atomic_ops,
        "warp_collectives": warp_collectives,
        "cute_dsl": cute_dsl,
        "vectorized_io": _count(
            code,
            r"\b(?:int2|int4|uint2|uint4|float2|float4|half2|__half2)\b|"
            r"reinterpret_cast\s*<[^>]*(?:2|4)\s*\*",
        ),
        "barrier_pipeline": _count(
            code,
            r"mbarrier|pipeline_(?:producer|consumer)|cuda::pipeline|"
            r"producer_commit|consumer_wait",
        ),
        "shared_memory_uses": shared_uses,
        "kernel_definitions": _count(code, r"\b__global__\b|@cute\.kernel"),
        "launch_sites": _count(
            code, r"<<<|cudaLaunchKernel|cudaLaunchCooperativeKernel"
        ),
        "code_chars": len(source),
        "code_lines": len(source.splitlines()),
    }


def architecture_label(features: dict[str, int]) -> str:
    if features.get("cluster_tma_broadcast"):
        return "cluster_tma_broadcast"
    if features.get("cub_radix_sort") or features.get("block_radix_sort"):
        return "cub_radix_sort"
    if features.get("bitwise_radix_sort"):
        return "bitwise_radix_sort"
    if features.get("expert_parallel_scan"):
        return "expert_parallel_scan"
    if features.get("hierarchical_histogram") or features.get(
        "bank_replicated_histogram"
    ):
        return "hierarchical_histogram"
    if features.get("warp_specialization"):
        return "warp_specialization"
    if features.get("tma") or features.get("wgmma"):
        return "tma_wgmma"
    if features.get("persistent_kernel") or features.get("cooperative_grid_sync"):
        return "persistent_cooperative"
    if features.get("cute_dsl"):
        return "cute_dsl"
    if features.get("cp_async") or features.get("barrier_pipeline"):
        return "async_pipeline"
    if (
        features.get("cuda_graph")
        or features.get("shared_memory_histogram")
        or features.get("atomic_ops")
    ):
        return "graph_histogram"
    return "baseline"


def architecture_tags(features: dict[str, int]) -> list[str]:
    return [name for name in _TAG_FEATURES if int(features.get(name, 0) or 0) > 0]


def _feature_vector(features: dict[str, int]) -> np.ndarray:
    values = []
    for name in FEATURE_NAMES:
        value = max(0.0, float(features.get(name, 0) or 0))
        values.append(math.log1p(value) if name in _LOG_SCALED_FEATURES else value)
    return np.asarray(values, dtype=np.float64)


def fit_architecture_pca(feature_rows: Iterable[dict[str, int]]) -> dict[str, Any]:
    """Fit deterministic standardized PCA with NumPy SVD, without sklearn."""
    rows = list(feature_rows)
    width = len(FEATURE_NAMES)
    matrix = (
        np.vstack([_feature_vector(row) for row in rows])
        if rows
        else np.empty((0, width), dtype=np.float64)
    )
    mean = matrix.mean(axis=0) if len(matrix) else np.zeros(width)
    scale = matrix.std(axis=0) if len(matrix) else np.ones(width)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (matrix - mean) / scale if len(matrix) else matrix

    components = np.zeros((PCA_COMPONENTS, width), dtype=np.float64)
    singular_values = np.zeros(PCA_COMPONENTS, dtype=np.float64)
    if len(matrix):
        _, singular, right = np.linalg.svd(standardized, full_matrices=False)
        count = min(PCA_COMPONENTS, len(right))
        components[:count] = right[:count]
        singular_values[:count] = singular[:count]
        for index in range(count):
            pivot = int(np.argmax(np.abs(components[index])))
            if components[index, pivot] < 0:
                components[index] *= -1

    coordinates = standardized @ components.T if len(matrix) else np.empty((0, 3))
    power = singular_values**2
    explained = power / power.sum() if power.sum() > 0 else np.zeros_like(power)
    return {
        "version": ANALYSIS_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "components": components.tolist(),
        "explained_variance_ratio": explained.tolist(),
        "coordinates": coordinates.tolist(),
    }


def transform_architecture_pca(
    features: dict[str, int], model: dict[str, Any]
) -> np.ndarray:
    vector = _feature_vector(features)
    mean = np.asarray(model.get("mean", np.zeros(len(FEATURE_NAMES))), dtype=float)
    scale = np.asarray(model.get("scale", np.ones(len(FEATURE_NAMES))), dtype=float)
    components = np.asarray(model.get("components", []), dtype=float)
    if mean.shape != vector.shape or scale.shape != vector.shape:
        return np.zeros(PCA_COMPONENTS, dtype=float)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    if components.shape != (PCA_COMPONENTS, len(FEATURE_NAMES)):
        return np.zeros(PCA_COMPONENTS, dtype=float)
    result = ((vector - mean) / scale) @ components.T
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def cluster_pca_coordinates(
    coordinates: Iterable[Iterable[float]], max_clusters: int
) -> tuple[list[int], list[list[float]]]:
    """Deterministic farthest-point k-means with canonical cluster numbering."""
    points = np.asarray(list(coordinates), dtype=np.float64)
    if points.size == 0:
        return [], []
    if points.ndim == 1:
        points = points.reshape(1, -1)

    unique = np.unique(np.round(points, 12), axis=0)
    count = max(1, min(int(max_clusters), len(unique)))
    first = int(np.argmax(np.linalg.norm(unique, axis=1)))
    centroids = [unique[first]]
    while len(centroids) < count:
        distances = np.min(
            np.stack([np.sum((unique - center) ** 2, axis=1) for center in centroids]),
            axis=0,
        )
        candidate = int(np.argmax(distances))
        if distances[candidate] <= 1e-18:
            break
        centroids.append(unique[candidate])
    centroid_array = np.vstack(centroids)

    labels = np.zeros(len(points), dtype=int)
    for _ in range(32):
        distances = np.stack(
            [np.sum((points - center) ** 2, axis=1) for center in centroid_array],
            axis=1,
        )
        updated_labels = np.argmin(distances, axis=1)
        updated = centroid_array.copy()
        for index in range(len(centroid_array)):
            members = points[updated_labels == index]
            if len(members):
                updated[index] = members.mean(axis=0)
        if np.array_equal(updated_labels, labels) and np.allclose(
            updated, centroid_array
        ):
            labels = updated_labels
            centroid_array = updated
            break
        labels = updated_labels
        centroid_array = updated

    order = sorted(
        range(len(centroid_array)),
        key=lambda index: tuple(np.round(centroid_array[index], 12).tolist()),
    )
    remap = {old: new for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in labels], dtype=int)
    centroid_array = centroid_array[order]
    return labels.tolist(), centroid_array.tolist()


def validate_architecture_island_count(num_islands: int) -> int:
    num_islands = int(num_islands)
    if num_islands < MIN_ARCHITECTURE_ISLANDS:
        raise ValueError(
            "architecture-aware routing requires at least "
            f"{MIN_ARCHITECTURE_ISLANDS} islands for distinct semantic anchors; "
            f"received {num_islands}"
        )
    return num_islands


def architecture_island_id(label: str, pca_cluster: int, num_islands: int) -> int:
    num_islands = validate_architecture_island_count(num_islands)
    if label in _SEMANTIC_ANCHORS:
        return _SEMANTIC_ANCHORS[label]
    general = list(range(MIN_ARCHITECTURE_ISLANDS, num_islands))
    if not general:
        general = list(range(min(6, num_islands), num_islands))
    if not general:
        return int(pca_cluster) % num_islands
    return general[int(pca_cluster) % len(general)]


def island_profile(island_id: int) -> str:
    if 0 <= int(island_id) < len(ISLAND_PROFILES):
        return ISLAND_PROFILES[int(island_id)]
    return f"pca_general_{int(island_id)}"


def _lock_context(memory: object):
    lock = getattr(memory, "_lock", None)
    return lock if lock is not None else nullcontext()


def _metadata(solution: object) -> dict[str, Any]:
    metadata = getattr(solution, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        solution.metadata = metadata
    return metadata


def _score(solution: object) -> float:
    try:
        return float(getattr(solution, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _analysis_for(
    features: dict[str, int],
    model: dict[str, Any],
    cluster: int,
    num_islands: int,
) -> dict[str, Any]:
    label = architecture_label(features)
    coordinates = transform_architecture_pca(features, model)
    island_id = architecture_island_id(label, cluster, num_islands)
    return {
        "features": features,
        "tags": architecture_tags(features),
        "label": label,
        "coordinates": [round(float(value), 8) for value in coordinates],
        "cluster": int(cluster),
        "island_id": int(island_id),
    }


def _apply_analysis(
    solution: object,
    analysis: dict[str, Any],
    model: dict[str, Any],
    refit_iteration: int,
) -> None:
    metadata = _metadata(solution)
    island_id = int(analysis["island_id"])
    metadata.update(
        {
            "architecture_analysis_version": ANALYSIS_VERSION,
            "architecture_features": analysis["features"],
            "architecture_tags": analysis["tags"],
            "architecture_label": analysis["label"],
            "pca_coordinates": analysis["coordinates"],
            "pca_cluster": int(analysis["cluster"]),
            "pca_explained_variance_ratio": [
                round(float(value), 8)
                for value in model.get("explained_variance_ratio", [])
            ],
            "architecture_island_id": island_id,
            "architecture_home_island_id": island_id,
            "architecture_island_profile": island_profile(island_id),
            "architecture_refit_iteration": int(refit_iteration),
        }
    )
    solution.island_id = island_id


def fit_architecture_population(
    solutions: Iterable[object], num_islands: int, refit_iteration: int = 0
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    ordered = sorted(
        list(solutions),
        key=lambda item: (
            str(getattr(item, "solution_id", "")),
            int(getattr(item, "iteration", 0) or 0),
        ),
    )
    features = [
        extract_architecture_features(getattr(solution, "solution", ""))
        for solution in ordered
    ]
    model = fit_architecture_pca(features)
    labels, centroids = cluster_pca_coordinates(
        model.get("coordinates", []), max(1, int(num_islands))
    )
    model["cluster_centroids"] = centroids
    semantic_labels = [architecture_label(row) for row in features]
    general_indexes = [
        index
        for index, label in enumerate(semantic_labels)
        if label not in _SEMANTIC_ANCHORS
    ]
    general_cluster_count = max(
        1,
        (
            int(num_islands) - MIN_ARCHITECTURE_ISLANDS
            if int(num_islands) > MIN_ARCHITECTURE_ISLANDS
            else int(num_islands) - 6
        ),
    )
    general_labels, general_centroids = cluster_pca_coordinates(
        [model["coordinates"][index] for index in general_indexes],
        general_cluster_count,
    )
    model["general_cluster_centroids"] = general_centroids
    for offset, index in enumerate(general_indexes):
        labels[index] = general_labels[offset] if offset < len(general_labels) else 0
    model["refit_iteration"] = int(refit_iteration)

    analyses: dict[str, dict[str, Any]] = {}
    for index, solution in enumerate(ordered):
        solution_id = str(getattr(solution, "solution_id", "") or index)
        cluster = labels[index] if index < len(labels) else 0
        analyses[solution_id] = _analysis_for(
            features[index], model, cluster, num_islands
        )
    return model, analyses


def ensure_architecture_islands(memory: object, num_islands: int) -> int:
    """Resize old checkpoints and all island indexes to the configured topology."""
    target = validate_architecture_island_count(num_islands)
    with _lock_context(memory):
        old_islands = list(getattr(memory, "islands", []) or [])
        islands = [
            set(old_islands[index]) if index < len(old_islands) else set()
            for index in range(target)
        ]
        for overflow in old_islands[target:]:
            islands[0].update(overflow)

        old_maps = list(getattr(memory, "island_feature_maps", []) or [])
        feature_maps = [
            dict(old_maps[index]) if index < len(old_maps) else {}
            for index in range(target)
        ]
        old_bests = list(getattr(memory, "island_best_solution", []) or [])
        island_bests = [
            old_bests[index] if index < len(old_bests) else None
            for index in range(target)
        ]

        locks = getattr(memory, "_island_locks", None)
        if not isinstance(locks, dict):
            locks = {}
        locks = {index: locks.get(index, threading.RLock()) for index in range(target)}

        memory.num_islands = target
        memory.islands = islands
        memory.island_feature_maps = feature_maps
        memory.island_best_solution = island_bests
        memory.island_capacity = [len(island) for island in islands]
        memory._island_locks = locks
        population_size = max(1, int(getattr(memory, "population_size", 100) or 100))
        memory.solutions_per_island = max(1, population_size // target)
        memory.current_island = int(getattr(memory, "current_island", 0) or 0) % target
    return target


def _map_elite_key(memory: object, solution: object) -> str | None:
    raw = _metadata(solution).get("MAP_Elite_feature")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, dict) or not raw:
        return None
    converter = getattr(memory, "_feature_coords_to_key", None)
    if callable(converter):
        try:
            return str(converter(raw))
        except Exception:
            pass
    return "-".join(str(value) for value in raw.values())


def _rebuild_rankings(memory: object) -> None:
    populations = getattr(memory, "populations", {})
    memory.island_capacity = [len(island) for island in memory.islands]
    memory.island_best_solution = []
    for island in memory.islands:
        candidates = [
            populations[solution_id]
            for solution_id in island
            if solution_id in populations
        ]
        best = max(candidates, key=_score) if candidates else None
        memory.island_best_solution.append(
            getattr(best, "solution_id", None) if best is not None else None
        )
    if populations:
        best = max(populations.values(), key=_score)
        memory.best_solution_id = getattr(best, "solution_id", "")
    else:
        memory.best_solution_id = ""


def rebuild_architecture_islands(
    memory: object,
    num_islands: int,
    refit_iteration: int | None = None,
) -> dict[str, int]:
    """Refit PCA and atomically reconstruct all selectable island indexes."""
    with _lock_context(memory):
        target = ensure_architecture_islands(memory, num_islands)
        populations = getattr(memory, "populations", None)
        if not isinstance(populations, dict):
            return {"solutions": 0, "islands": target}

        refit_iteration = int(
            getattr(memory, "last_iteration", 0)
            if refit_iteration is None
            else refit_iteration
        )
        home = [
            solution
            for solution in populations.values()
            if not _metadata(solution).get("migrated", False)
        ]
        fit_set = home or list(populations.values())
        model, analyses = fit_architecture_population(fit_set, target, refit_iteration)
        memory._atrex_architecture_pca_model = model
        memory._atrex_pca_refit_iteration = refit_iteration

        memory.islands = [set() for _ in range(target)]
        memory.island_feature_maps = [{} for _ in range(target)]
        for solution_id, solution in sorted(populations.items()):
            metadata = _metadata(solution)
            if metadata.get("migrated", False):
                island_id = int(metadata.get("migration_target_island", 0)) % target
                solution.island_id = island_id
                metadata["architecture_island_id"] = island_id
                metadata["architecture_island_profile"] = island_profile(island_id)
            else:
                analysis = analyses.get(str(solution_id))
                if analysis is None:
                    features = extract_architecture_features(
                        getattr(solution, "solution", "")
                    )
                    analysis = _analysis_for(features, model, 0, target)
                _apply_analysis(solution, analysis, model, refit_iteration)
                island_id = int(solution.island_id)
            memory.islands[island_id].add(solution_id)

            feature_key = _map_elite_key(memory, solution)
            if feature_key is not None:
                feature_map = memory.island_feature_maps[island_id]
                existing_id = feature_map.get(feature_key)
                existing = populations.get(existing_id) if existing_id else None
                if existing is None or _score(solution) > _score(existing):
                    feature_map[feature_key] = solution_id

        elites = getattr(memory, "elites", None)
        if isinstance(elites, set):
            elites.intersection_update(populations)
        _rebuild_rankings(memory)
        logger.info(
            "Rebuilt %d architecture-classified solutions across %d islands",
            len(populations),
            target,
        )
        return {"solutions": len(populations), "islands": target}


def classify_and_route_solution(
    memory: object, solution: object, num_islands: int
) -> dict[str, Any]:
    """Project one Summary child into the frozen PCA model and select its island."""
    with _lock_context(memory):
        target = ensure_architecture_islands(memory, num_islands)
        model = getattr(memory, "_atrex_architecture_pca_model", None)
        populations = getattr(memory, "populations", {})
        if not isinstance(model, dict) or model.get("feature_names") != list(
            FEATURE_NAMES
        ):
            if isinstance(populations, dict) and populations:
                rebuild_architecture_islands(memory, target)
                model = getattr(memory, "_atrex_architecture_pca_model", None)
            else:
                features = extract_architecture_features(
                    getattr(solution, "solution", "")
                )
                model = fit_architecture_pca([features])
                model["cluster_centroids"] = [[0.0] * PCA_COMPONENTS]
                model["refit_iteration"] = int(getattr(solution, "iteration", 0) or 0)
                memory._atrex_architecture_pca_model = model

        features = extract_architecture_features(getattr(solution, "solution", ""))
        coordinates = transform_architecture_pca(features, model)
        label = architecture_label(features)
        centroid_key = (
            "cluster_centroids"
            if label in _SEMANTIC_ANCHORS
            else "general_cluster_centroids"
        )
        centroids = np.asarray(model.get(centroid_key, []), dtype=float)
        if centroids.ndim == 2 and len(centroids):
            cluster = int(np.argmin(np.sum((centroids - coordinates) ** 2, axis=1)))
        else:
            cluster = 0
        analysis = _analysis_for(features, model, cluster, target)
        _apply_analysis(
            solution,
            analysis,
            model,
            int(model.get("refit_iteration", 0) or 0),
        )
        return analysis


def _source_hash(solution: object) -> str:
    source = getattr(solution, "solution", "")
    text = source if isinstance(source, str) else str(source or "")
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def expire_migration_copies(memory: object, boundary: int) -> int:
    """Remove the previous exchange wave from selectable indexes, retaining history."""
    populations = getattr(memory, "populations", None)
    if not isinstance(populations, dict):
        return 0
    expired = {
        solution_id
        for solution_id, solution in populations.items()
        if _metadata(solution).get("migrated", False)
        and int(_metadata(solution).get("migration_iteration", -1)) < int(boundary)
    }
    if not expired:
        return 0

    for solution_id in expired:
        solution = populations.pop(solution_id, None)
        if solution is not None:
            _metadata(solution)["migration_expired_at"] = int(boundary)
    for island in getattr(memory, "islands", []):
        island.difference_update(expired)
    for feature_map in getattr(memory, "island_feature_maps", []):
        for key, solution_id in list(feature_map.items()):
            if solution_id in expired:
                feature_map.pop(key, None)
    elites = getattr(memory, "elites", None)
    if isinstance(elites, set):
        elites.difference_update(expired)
    return len(expired)


def perform_island_exchange(memory: object, boundary: int) -> int:
    """Copy each island's top fraction to both neighboring architecture islands."""
    populations = getattr(memory, "populations", None)
    solutions = getattr(memory, "solutions", None)
    islands = getattr(memory, "islands", None)
    if (
        not isinstance(populations, dict)
        or not isinstance(solutions, dict)
        or not isinstance(islands, list)
        or len(islands) < 2
    ):
        return 0

    snapshots: list[list[object]] = []
    for island in islands:
        candidates = [
            populations[solution_id]
            for solution_id in island
            if solution_id in populations
            and not _metadata(populations[solution_id]).get("migrated", False)
        ]
        snapshots.append(sorted(candidates, key=_score, reverse=True))

    migration_rate = max(0.0, float(getattr(memory, "migration_rate", 0.2) or 0.2))
    migrated = 0
    for origin, candidates in enumerate(snapshots):
        if not candidates:
            continue
        count = max(1, int(len(candidates) * migration_rate))
        targets = sorted({(origin - 1) % len(islands), (origin + 1) % len(islands)})
        for source_solution in candidates[:count]:
            source_id = str(getattr(source_solution, "solution_id", ""))
            source_hash = _source_hash(source_solution)
            for target in targets:
                if target == origin:
                    continue
                if any(
                    _source_hash(populations[solution_id]) == source_hash
                    for solution_id in islands[target]
                    if solution_id in populations
                ):
                    continue

                digest = hashlib.sha1(
                    f"{source_id}:{origin}:{target}:{boundary}".encode("utf-8")
                ).hexdigest()[:12]
                migration_id = f"mig{digest}"
                if migration_id in solutions:
                    continue
                migrant = copy.deepcopy(source_solution)
                migrant.solution_id = migration_id
                migrant.parent_id = source_id
                migrant.generation = (
                    int(getattr(source_solution, "generation", 0) or 0) + 1
                )
                migrant.island_id = target
                migrant.iteration = int(boundary)
                metadata = _metadata(migrant)
                metadata.pop("duplicate_of", None)
                metadata.update(
                    {
                        "migrated": True,
                        "migration_source_id": source_id,
                        "migration_origin_island": origin,
                        "migration_target_island": target,
                        "migration_iteration": int(boundary),
                        "architecture_home_island_id": int(
                            _metadata(source_solution).get(
                                "architecture_home_island_id", origin
                            )
                        ),
                        "architecture_island_id": target,
                        "architecture_island_profile": island_profile(target),
                    }
                )
                populations[migration_id] = migrant
                solutions[migration_id] = migrant
                islands[target].add(migration_id)
                migrated += 1

    _rebuild_rankings(memory)
    return migrated


def maybe_exchange_islands(
    memory: object,
    num_islands: int,
    migration_interval: int,
) -> dict[str, int]:
    """Run exactly one exchange when global iterations cross a fixed boundary."""
    interval = max(1, int(migration_interval))
    current = max(0, int(getattr(memory, "last_iteration", 0) or 0))
    boundary = (current // interval) * interval
    last = max(0, int(getattr(memory, "_atrex_last_migration_iteration", 0) or 0))
    if boundary <= 0 or boundary <= last:
        return {"boundary": boundary, "expired": 0, "migrated": 0}

    with _lock_context(memory):
        last = max(0, int(getattr(memory, "_atrex_last_migration_iteration", 0) or 0))
        if boundary <= last:
            return {"boundary": boundary, "expired": 0, "migrated": 0}
        expired = expire_migration_copies(memory, boundary)
        rebuild_architecture_islands(memory, num_islands, boundary)
        migrated = perform_island_exchange(memory, boundary)
        memory._atrex_last_migration_iteration = boundary
        # Reuse the upstream checkpoint field, but store global iteration rather
        # than the old max-island-capacity trigger.
        memory.last_migration_generation = boundary
        logger.info(
            "Architecture island exchange at iteration %d: expired=%d migrated=%d",
            boundary,
            expired,
            migrated,
        )
        return {"boundary": boundary, "expired": expired, "migrated": migrated}


def initialize_architecture_memory(
    memory: object, num_islands: int, migration_interval: int
) -> None:
    with _lock_context(memory):
        target = ensure_architecture_islands(memory, num_islands)
        memory.migration_interval = max(1, int(migration_interval))
        if not hasattr(memory, "_atrex_last_migration_iteration"):
            memory._atrex_last_migration_iteration = 0
        populations = getattr(memory, "populations", {})
        if populations and not isinstance(
            getattr(memory, "_atrex_architecture_pca_model", None), dict
        ):
            rebuild_architecture_islands(memory, target)


def architecture_checkpoint_payload(memory: object) -> dict[str, Any]:
    model = copy.deepcopy(getattr(memory, "_atrex_architecture_pca_model", {}) or {})
    model.pop("coordinates", None)
    return {
        "version": ANALYSIS_VERSION,
        "num_islands": int(getattr(memory, "num_islands", 0) or 0),
        "island_profiles": [
            island_profile(index)
            for index in range(int(getattr(memory, "num_islands", 0) or 0))
        ],
        "migration_interval": int(getattr(memory, "migration_interval", 20) or 20),
        "last_migration_iteration": int(
            getattr(memory, "_atrex_last_migration_iteration", 0) or 0
        ),
        "pca_refit_iteration": int(
            getattr(memory, "_atrex_pca_refit_iteration", 0) or 0
        ),
        "pca_model": model,
    }


def write_architecture_checkpoint(
    memory: object, checkpoint_root: str, tag: str
) -> bool:
    metadata_path = (
        Path(checkpoint_root) / "checkpoints" / f"checkpoint-{tag}" / "metadata.json"
    )
    if not metadata_path.is_file():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["atrex_architecture"] = architecture_checkpoint_payload(memory)
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(metadata_path)
        return True
    except Exception as exc:
        logger.warning("Could not persist architecture checkpoint metadata: %s", exc)
        return False


def _assignments_are_valid(
    memory: object, populations: dict[str, object], target: int
) -> bool:
    assigned = set().union(*memory.islands) if memory.islands else set()
    if assigned != set(populations):
        return False
    for solution_id, solution in populations.items():
        try:
            island_id = int(getattr(solution, "island_id", -1))
        except (TypeError, ValueError):
            return False
        if not 0 <= island_id < target or solution_id not in memory.islands[island_id]:
            return False
        if _metadata(solution).get("architecture_analysis_version") != ANALYSIS_VERSION:
            return False
    return True


def restore_architecture_checkpoint(
    memory: object,
    checkpoint_path: str,
    num_islands: int,
    migration_interval: int,
) -> dict[str, int]:
    state: dict[str, Any] = {}
    metadata_path = Path(checkpoint_path) / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        candidate = payload.get("atrex_architecture", {})
        if isinstance(candidate, dict):
            state = candidate
    except Exception:
        state = {}

    with _lock_context(memory):
        target = ensure_architecture_islands(memory, num_islands)
        interval = max(1, int(migration_interval))
        memory.migration_interval = interval
        current = max(0, int(getattr(memory, "last_iteration", 0) or 0))
        state_is_current = state.get("version") == ANALYSIS_VERSION
        if state_is_current:
            last = max(0, int(state.get("last_migration_iteration", 0) or 0))
        else:
            # A legacy one-island checkpoint had no architecture exchanges.
            # Start at the preceding boundary so it does not exchange mid-period.
            last = (current // interval) * interval
        memory._atrex_last_migration_iteration = last
        memory.last_migration_generation = last
        populations = getattr(memory, "populations", {})
        saved_model = state.get("pca_model")
        assignments_are_valid = isinstance(
            populations, dict
        ) and _assignments_are_valid(memory, populations, target)
        saved_model_is_valid = (
            isinstance(saved_model, dict)
            and saved_model.get("feature_names") == list(FEATURE_NAMES)
            and int(state.get("num_islands", 0) or 0) == target
        )
        if state_is_current and saved_model_is_valid and assignments_are_valid:
            memory._atrex_architecture_pca_model = saved_model
            memory._atrex_pca_refit_iteration = int(
                state.get("pca_refit_iteration", last) or last
            )
            _rebuild_rankings(memory)
            result = {"solutions": len(populations), "islands": target}
        else:
            result = rebuild_architecture_islands(memory, target, current)
        result["last_migration_iteration"] = last
        return result
