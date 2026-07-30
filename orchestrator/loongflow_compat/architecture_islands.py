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

from orchestrator.loongflow_compat.protocols import EvolutionMemory, EvolutionSolution

logger = logging.getLogger("atrex.pes_architecture")

ANALYSIS_VERSION = 3
STAGNATION_CHECKPOINT_VERSION = 1
PCA_COMPONENTS = 3

FEATURE_NAMES = (
    "warp_specialization",
    "cluster_launch",
    "cluster_dsmem",
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
    "cluster_dsmem": 1,
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
    cluster_dsmem = int(
        bool(cluster_launch)
        and bool(
            _present(
                code,
                r"map_shared_rank|__cluster_map_shared_rank|distributed[_ ]?shared|"
                r"cluster[_ ]?(?:shared|dsmem)|\bDSMEM\b",
            )
        )
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
        "cluster_dsmem": cluster_dsmem,
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
    if features.get("cluster_dsmem"):
        return "cluster_dsmem"
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


def _lock_context(memory: EvolutionMemory):
    lock = getattr(memory, "_lock", None)
    return lock if lock is not None else nullcontext()


def _metadata(solution: EvolutionSolution) -> dict[str, Any]:
    metadata = getattr(solution, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        solution.metadata = metadata
    return metadata


def _score(solution: EvolutionSolution) -> float:
    try:
        return float(getattr(solution, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_migration_copy(solution: EvolutionSolution) -> bool:
    return bool(_metadata(solution).get("migrated", False))


def architecture_home_island(solution: EvolutionSolution) -> int | None:
    """Return the semantic home island; migration copies never count as home."""
    if is_migration_copy(solution):
        return None
    metadata = _metadata(solution)
    raw = metadata.get(
        "architecture_home_island_id",
        metadata.get("architecture_island_id", getattr(solution, "island_id", None)),
    )
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def architecture_home_occupancy(memory: EvolutionMemory) -> list[int]:
    """Count selectable, non-migrated members whose semantic home is each island."""
    populations = getattr(memory, "populations", {})
    num_islands = max(0, int(getattr(memory, "num_islands", 0) or 0))
    occupancy = [0] * num_islands
    if not isinstance(populations, dict):
        return occupancy
    for solution in populations.values():
        home = architecture_home_island(solution)
        if home is not None and 0 <= home < num_islands:
            occupancy[home] += 1
    return occupancy


def _remove_selectable_solutions(
    memory: EvolutionMemory, solution_ids: set[str], *, reason: str
) -> None:
    populations = getattr(memory, "populations", {})
    if not isinstance(populations, dict) or not solution_ids:
        return
    for solution_id in solution_ids:
        solution = populations.pop(solution_id, None)
        if solution is not None:
            metadata = _metadata(solution)
            metadata["population_eviction_reason"] = reason
            metadata["population_evicted_at_iteration"] = int(
                getattr(memory, "last_iteration", 0) or 0
            )
    for island in getattr(memory, "islands", []):
        island.difference_update(solution_ids)
    for feature_map in getattr(memory, "island_feature_maps", []):
        for key, solution_id in list(feature_map.items()):
            if solution_id in solution_ids:
                feature_map.pop(key, None)
    elites = getattr(memory, "elites", None)
    if isinstance(elites, set):
        elites.difference_update(solution_ids)


def enforce_architecture_population_limit(
    memory: EvolutionMemory,
    *,
    exclude_solution_id: str | None = None,
    minimum_home_per_island: int = 6,
    maximum_migrant_fraction: float = 0.2,
) -> dict[str, Any]:
    """Bound population size without allowing strong families to erase weak islands."""
    populations = getattr(memory, "populations", None)
    if not isinstance(populations, dict):
        raise RuntimeError("architecture retention requires memory.populations:dict")
    population_size = max(1, int(getattr(memory, "population_size", 100) or 100))
    num_islands = max(1, int(getattr(memory, "num_islands", 1) or 1))
    minimum_home_per_island = max(0, int(minimum_home_per_island))
    minimum_home_per_island = min(
        minimum_home_per_island, population_size // num_islands
    )
    maximum_migrant_fraction = min(1.0, max(0.0, float(maximum_migrant_fraction)))
    migrant_budget = int(math.floor(population_size * maximum_migrant_fraction))
    overflow = max(0, len(populations) - population_size)
    migrant_count = sum(
        is_migration_copy(solution) for solution in populations.values()
    )
    if maximum_migrant_fraction >= 1.0:
        excess_migrants = 0
    else:
        # Removing a migrant shrinks both numerator and denominator. Solve for
        # the smallest removal count that bounds the resulting population share.
        excess_migrants = max(
            0,
            int(
                math.ceil(
                    (migrant_count - maximum_migrant_fraction * len(populations))
                    / (1.0 - maximum_migrant_fraction)
                )
            ),
        )
    removal_target = max(overflow, excess_migrants)
    if removal_target == 0:
        _rebuild_rankings(memory)
        return {
            "removed": 0,
            "migrants_removed": 0,
            "migrant_budget": migrant_budget,
            "home_occupancy": architecture_home_occupancy(memory),
        }

    protected = {
        str(solution_id)
        for solution_id in (
            getattr(memory, "best_solution_id", None),
            exclude_solution_id,
        )
        if solution_id
    }
    home_groups: list[list[EvolutionSolution]] = [[] for _ in range(num_islands)]
    migrants: list[EvolutionSolution] = []
    for solution in populations.values():
        if is_migration_copy(solution):
            migrants.append(solution)
            continue
        home = architecture_home_island(solution)
        if home is not None and 0 <= home < num_islands:
            home_groups[home].append(solution)

    def rank_key(solution: EvolutionSolution) -> tuple[float, int, str]:
        return (
            _score(solution),
            -int(getattr(solution, "iteration", 0) or 0),
            str(getattr(solution, "solution_id", "")),
        )

    for group in home_groups:
        for solution in sorted(group, key=rank_key, reverse=True)[
            :minimum_home_per_island
        ]:
            protected.add(str(solution.solution_id))

    removal: list[EvolutionSolution] = []
    selected: set[str] = set()

    def take(
        candidates: Iterable[EvolutionSolution],
        count: int,
        protected_ids: set[str] | None = None,
    ) -> None:
        protected_ids = protected if protected_ids is None else protected_ids
        for solution in sorted(candidates, key=rank_key):
            solution_id = str(getattr(solution, "solution_id", ""))
            if (
                len(removal) >= removal_target
                or count <= 0
                or not solution_id
                or solution_id in protected_ids
                or solution_id in selected
            ):
                continue
            removal.append(solution)
            selected.add(solution_id)
            count -= 1

    take(migrants, excess_migrants)
    take(
        (
            solution
            for solution in populations.values()
            if str(getattr(solution, "solution_id", "")) not in selected
        ),
        removal_target - len(removal),
    )
    if len(removal) < removal_target:
        # A pathological configuration may protect more entries than the capacity.
        # Preserve the global best/current child, but relax island quotas deterministically.
        hard_protected = {
            str(solution_id)
            for solution_id in (
                getattr(memory, "best_solution_id", None),
                exclude_solution_id,
            )
            if solution_id
        }
        take(
            (
                solution
                for solution in populations.values()
                if str(getattr(solution, "solution_id", "")) not in hard_protected
            ),
            removal_target - len(removal),
            hard_protected,
        )

    removed_ids = {str(solution.solution_id) for solution in removal[:removal_target]}
    migrants_removed = sum(
        is_migration_copy(solution) for solution in removal[:removal_target]
    )
    _remove_selectable_solutions(
        memory,
        removed_ids,
        reason="architecture_retention_population_limit",
    )
    _rebuild_rankings(memory)
    logger.info(
        "Architecture retention removed %d solutions (%d migrants); home=%s "
        "migrant_budget=%d population=%d/%d",
        len(removed_ids),
        migrants_removed,
        architecture_home_occupancy(memory),
        migrant_budget,
        len(populations),
        population_size,
    )
    return {
        "removed": len(removed_ids),
        "migrants_removed": migrants_removed,
        "migrant_budget": migrant_budget,
        "home_occupancy": architecture_home_occupancy(memory),
    }


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
    solution: EvolutionSolution,
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


def ensure_architecture_islands(memory: EvolutionMemory, num_islands: int) -> int:
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


def _map_elite_key(memory: EvolutionMemory, solution: EvolutionSolution) -> str | None:
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


def _rebuild_rankings(memory: EvolutionMemory) -> None:
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
    memory: EvolutionMemory,
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
    memory: EvolutionMemory, solution: EvolutionSolution, num_islands: int
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

        if not isinstance(model, dict):
            raise RuntimeError("architecture PCA model was not initialized")
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


def _source_hash(solution: EvolutionSolution) -> str:
    source = getattr(solution, "solution", "")
    text = source if isinstance(source, str) else str(source or "")
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def expire_migration_copies(memory: EvolutionMemory, boundary: int) -> int:
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


def perform_island_exchange(memory: EvolutionMemory, boundary: int) -> int:
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

    snapshots: list[list[EvolutionSolution]] = []
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
    memory: EvolutionMemory,
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
    memory: EvolutionMemory, num_islands: int, migration_interval: int
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


def _current_stagnation_best_marker(
    memory: EvolutionMemory,
) -> tuple[int, float] | None:
    populations = getattr(memory, "populations", {})
    if not isinstance(populations, dict) or not populations:
        return None
    candidates = list(populations.values())
    best_score = max(_score(solution) for solution in candidates)
    best_iteration = min(
        int(getattr(solution, "iteration", 0) or 0)
        for solution in candidates
        if abs(_score(solution) - best_score) <= 1e-12
    )
    return best_iteration, round(best_score, 12)


def _stagnation_checkpoint_payload(memory: EvolutionMemory) -> dict[str, Any]:
    marker = getattr(memory, "_atrex_stagnation_best_marker", None)
    if not (
        isinstance(marker, (tuple, list))
        and len(marker) == 2
        and isinstance(marker[0], (int, float))
        and isinstance(marker[1], (int, float))
    ):
        marker = None
    pending = getattr(memory, "_atrex_pending_stagnation_escape", None)
    bootstrap_attempts = getattr(memory, "_atrex_bootstrap_attempts", {})
    if not isinstance(bootstrap_attempts, dict):
        bootstrap_attempts = {}
    pending_bootstrap = getattr(memory, "_atrex_pending_architecture_bootstrap", None)
    return {
        "version": STAGNATION_CHECKPOINT_VERSION,
        "best_marker": list(marker) if marker is not None else None,
        "seed_bucket": getattr(memory, "_atrex_stagnation_seed_bucket", None),
        "retry_bucket": getattr(memory, "_atrex_stagnation_retry_bucket", None),
        "seed_retries": max(
            0, int(getattr(memory, "_atrex_stagnation_seed_retries", 0) or 0)
        ),
        "last_seed_id": str(
            getattr(memory, "_atrex_last_stagnation_seed_id", "") or ""
        ),
        "pending_escape": copy.deepcopy(pending) if isinstance(pending, dict) else None,
        "bootstrap_attempts": {
            str(key): max(0, int(value or 0))
            for key, value in bootstrap_attempts.items()
        },
        "pending_bootstrap": (
            copy.deepcopy(pending_bootstrap)
            if isinstance(pending_bootstrap, dict)
            else None
        ),
    }


def _latest_completed_stagnation_escape(
    memory: EvolutionMemory,
) -> dict[str, Any] | None:
    solutions = getattr(memory, "solutions", {})
    if not isinstance(solutions, dict):
        return None
    records: list[dict[str, Any]] = []
    for solution in solutions.values():
        metadata = _metadata(solution)
        for key in ("stagnation_escape_satisfied", "stagnation_escape_violation"):
            record = metadata.get(key)
            if isinstance(record, dict) and "bucket" in record:
                candidate = copy.deepcopy(record)
                candidate["checkpoint_outcome"] = (
                    "satisfied" if key.endswith("satisfied") else "violation"
                )
                records.append(candidate)
    if not records:
        return None
    return max(
        records,
        key=lambda record: (
            int(record.get("expected_iteration", -1) or -1),
            int(record.get("bucket", -1) or -1),
        ),
    )


def restore_stagnation_checkpoint_state(
    memory: EvolutionMemory, state: object
) -> dict[str, Any]:
    """Restore escape scheduling state, with a best-effort legacy reconstruction."""
    payload = state if isinstance(state, dict) else {}
    is_current = payload.get("version") == STAGNATION_CHECKPOINT_VERSION
    latest = None if is_current else _latest_completed_stagnation_escape(memory)

    marker = payload.get("best_marker") if is_current else None
    if isinstance(marker, list) and len(marker) == 2:
        try:
            best_marker: tuple[int, float] | None = (
                int(marker[0]),
                round(float(marker[1]), 12),
            )
        except (TypeError, ValueError):
            best_marker = None
    else:
        best_marker = None
    if not is_current:
        best_marker = _current_stagnation_best_marker(memory)
        if latest is not None and best_marker is not None:
            latest_iteration = int(latest.get("expected_iteration", -1) or -1)
            if latest_iteration <= best_marker[0]:
                latest = None

    def optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    if is_current:
        seed_bucket = optional_int(payload.get("seed_bucket"))
        retry_bucket = optional_int(payload.get("retry_bucket"))
        retries = max(0, int(payload.get("seed_retries", 0) or 0))
        last_seed_id = str(payload.get("last_seed_id") or "")
        pending = payload.get("pending_escape")
        pending = copy.deepcopy(pending) if isinstance(pending, dict) else None
        raw_bootstrap_attempts = payload.get("bootstrap_attempts")
        bootstrap_attempts = (
            {
                str(key): max(0, int(value or 0))
                for key, value in raw_bootstrap_attempts.items()
            }
            if isinstance(raw_bootstrap_attempts, dict)
            else {}
        )
        pending_bootstrap = payload.get("pending_bootstrap")
        pending_bootstrap = (
            copy.deepcopy(pending_bootstrap)
            if isinstance(pending_bootstrap, dict)
            else None
        )
    else:
        retry_bucket = optional_int(latest.get("bucket")) if latest else None
        retries = max(0, int(latest.get("retry", 0) or 0)) if latest else 0
        last_seed_id = str(latest.get("seed_id") or "") if latest else ""
        max_attempts = max(1, int(latest.get("max_attempts", 2) or 2)) if latest else 2
        seed_bucket = retry_bucket
        if (
            latest
            and latest.get("checkpoint_outcome") == "violation"
            and retries < max_attempts
        ):
            seed_bucket = None
        pending = None
        bootstrap_attempts = {}
        pending_bootstrap = None

    memory._atrex_stagnation_best_marker = best_marker
    memory._atrex_stagnation_seed_bucket = seed_bucket
    memory._atrex_stagnation_retry_bucket = retry_bucket
    memory._atrex_stagnation_seed_retries = retries
    memory._atrex_last_stagnation_seed_id = last_seed_id
    memory._atrex_pending_stagnation_escape = pending
    memory._atrex_bootstrap_attempts = bootstrap_attempts
    memory._atrex_pending_architecture_bootstrap = pending_bootstrap
    return {
        "restored": is_current,
        "legacy_reconstructed": not is_current and latest is not None,
        "seed_bucket": seed_bucket,
        "retry_bucket": retry_bucket,
        "seed_retries": retries,
        "pending_escape": pending is not None,
        "bootstrap_attempts": len(bootstrap_attempts),
        "pending_bootstrap": pending_bootstrap is not None,
    }


def restore_stagnation_checkpoint(
    memory: EvolutionMemory, checkpoint_path: str
) -> dict[str, Any]:
    """Restore stagnation state without requiring architecture-island routing."""
    state: object = None
    metadata_path = Path(checkpoint_path) / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        architecture = payload.get("atrex_architecture")
        if isinstance(architecture, dict):
            state = architecture.get("stagnation_escape")
    except (OSError, TypeError, ValueError):
        state = None
    status = restore_stagnation_checkpoint_state(memory, state)
    memory._atrex_stagnation_checkpoint_status = status
    return status


def architecture_checkpoint_payload(memory: EvolutionMemory) -> dict[str, Any]:
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
        "stagnation_escape": _stagnation_checkpoint_payload(memory),
    }


def write_architecture_checkpoint(
    memory: EvolutionMemory, checkpoint_root: str, tag: str
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
    memory: EvolutionMemory, populations: dict[str, Any], target: int
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
    memory: EvolutionMemory,
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
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(
            "Ignoring unavailable architecture checkpoint state %s: %s",
            metadata_path,
            exc,
        )
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
            if not isinstance(saved_model, dict):
                raise RuntimeError("validated architecture PCA model is not a mapping")
            memory._atrex_architecture_pca_model = saved_model
            memory._atrex_pca_refit_iteration = int(
                state.get("pca_refit_iteration", last) or last
            )
            _rebuild_rankings(memory)
            result = {"solutions": len(populations), "islands": target}
        else:
            result = rebuild_architecture_islands(memory, target, current)
        stagnation = restore_stagnation_checkpoint_state(
            memory, state.get("stagnation_escape") if state_is_current else None
        )
        memory._atrex_stagnation_checkpoint_status = stagnation
        result["last_migration_iteration"] = last
        return result
