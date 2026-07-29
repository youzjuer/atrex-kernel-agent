"""Validated checkpoint repairs for LoongFlow compatibility adapters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot satisfy the compatibility contract."""


@dataclass(frozen=True)
class PopulationRestoreReport:
    metadata_path: str
    loaded_population_count: int
    selectable_population_count: int
    removed_lineage_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_checkpoint_islands(metadata_path: Path) -> list[list[str]]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint metadata is missing: {metadata_path}"
        ) from exc
    except OSError as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint metadata cannot be read: {metadata_path}: {exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint metadata is not valid JSON: {metadata_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError(
            f"checkpoint metadata root must be an object: {metadata_path}"
        )
    islands = payload.get("islands")
    if not isinstance(islands, list) or not all(
        isinstance(island, list) for island in islands
    ):
        raise CheckpointCompatibilityError(
            f"checkpoint metadata has no valid islands list: {metadata_path}"
        )
    return [
        [str(solution_id) for solution_id in island if solution_id]
        for island in islands
    ]


def restore_checkpoint_population_indexes(
    memory: object,
    checkpoint_path: str | Path,
    canonical_solution: Callable[[list[object]], object],
) -> PopulationRestoreReport:
    """Remove lineage-only records that upstream loaded into selectable populations."""
    metadata_path = Path(checkpoint_path) / "metadata.json"
    saved_islands = _load_checkpoint_islands(metadata_path)
    selectable_ids = {solution_id for island in saved_islands for solution_id in island}

    populations = getattr(memory, "populations", None)
    solutions = getattr(memory, "solutions", None)
    islands = getattr(memory, "islands", None)
    lock = getattr(memory, "_lock", None)
    invalid = []
    if not isinstance(populations, dict):
        invalid.append("populations:dict")
    if not isinstance(solutions, dict):
        invalid.append("solutions:dict")
    if not isinstance(islands, list):
        invalid.append("islands:list")
    elif not all(isinstance(island, set) for island in islands):
        invalid.append("islands[*]:set")
    if lock is None or not hasattr(lock, "__enter__"):
        invalid.append("_lock:context-manager")
    if invalid:
        raise CheckpointCompatibilityError(
            "LoongFlow checkpoint memory contract mismatch: " + ", ".join(invalid)
        )

    missing = selectable_ids - set(solutions)
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise CheckpointCompatibilityError(
            f"checkpoint islands reference {len(missing)} unloaded solutions: {sample}"
        )

    with lock:
        loaded_ids = set(populations)
        populations.clear()
        populations.update(
            {solution_id: solutions[solution_id] for solution_id in selectable_ids}
        )
        for island in islands:
            island.intersection_update(selectable_ids)

        elites = getattr(memory, "elites", None)
        if not isinstance(elites, set):
            raise CheckpointCompatibilityError(
                "LoongFlow checkpoint memory contract mismatch: elites:set"
            )
        elites.intersection_update(selectable_ids)

        feature_maps = getattr(memory, "island_feature_maps", None)
        if feature_maps is not None:
            if not isinstance(feature_maps, list) or not all(
                isinstance(feature_map, dict) for feature_map in feature_maps
            ):
                raise CheckpointCompatibilityError(
                    "LoongFlow checkpoint memory contract mismatch: "
                    "island_feature_maps:list[dict]"
                )
            for feature_map in feature_maps:
                for key, solution_id in list(feature_map.items()):
                    if solution_id not in selectable_ids:
                        feature_map.pop(key, None)

        if getattr(memory, "best_solution_id", None) not in selectable_ids:
            memory.best_solution_id = (
                canonical_solution(list(populations.values())).solution_id
                if populations
                else None
            )
        memory.island_best_solution = [
            (
                canonical_solution(
                    [populations[solution_id] for solution_id in island]
                ).solution_id
                if island
                else None
            )
            for island in islands
        ]
        if hasattr(memory, "island_capacity"):
            memory.island_capacity = [len(island) for island in islands]

    return PopulationRestoreReport(
        metadata_path=str(metadata_path),
        loaded_population_count=len(loaded_ids),
        selectable_population_count=len(selectable_ids),
        removed_lineage_count=len(loaded_ids - selectable_ids),
    )
