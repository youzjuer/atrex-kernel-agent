"""Structural types for the private LoongFlow boundary used by compatibility code."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol


class EvolutionSolution(Protocol):
    solution_id: str
    solution: str
    score: float
    sample_weight: float
    metadata: dict[str, Any]
    island_id: int
    iteration: int
    parent_id: str | None
    generation: int
    timestamp: Any
    evaluation: Any
    summary: str

    def to_dict(self) -> dict[str, Any]: ...


class EvolutionMemory(Protocol):
    populations: dict[str, EvolutionSolution]
    solutions: dict[str, EvolutionSolution]
    elites: set[str]
    islands: list[set[str]]
    island_feature_maps: list[dict[str, str]]
    island_best_solution: list[str | None]
    island_capacity: list[int]
    solutions_per_island: int
    num_islands: int
    current_island: int
    best_solution_id: str | None
    last_iteration: int
    migration_interval: int
    migration_rate: float
    last_migration_generation: int
    population_size: int
    boltzmann_temperature: float
    sampling_weight_power: float
    use_sampling_weight: bool
    _lock: AbstractContextManager[Any]
    _island_locks: dict[int, AbstractContextManager[Any]]
    _atrex_architecture_pca_model: dict[str, Any] | None
    _atrex_pca_refit_iteration: int
    _atrex_last_migration_iteration: int
    _atrex_stagnation_best_marker: Any
    _atrex_stagnation_seed_bucket: int | None
    _atrex_stagnation_retry_bucket: int | None
    _atrex_stagnation_seed_retries: int
    _atrex_last_stagnation_seed_id: str
    _atrex_pending_stagnation_escape: dict[str, Any] | None
    _atrex_bootstrap_attempts: dict[str, int]
    _atrex_pending_architecture_bootstrap: dict[str, Any] | None
    _atrex_stagnation_checkpoint_status: dict[str, Any]

    def _feature_coords_to_key(self, coordinates: Any) -> str: ...

    def _prepare_solution(self, solution: Any) -> None: ...

    async def _check_migration(self) -> None: ...

    def list_solutions(self, *args: Any, **kwargs: Any) -> Any: ...

    def sample(self, *args: Any, **kwargs: Any) -> Any: ...


class EvolutionDatabase(Protocol):
    config: Any
    _memory: EvolutionMemory

    def sample_solution(self, island_id: int | None = None) -> dict[str, Any]: ...

    async def add_solution(self, solution: Any) -> str: ...

    def load_checkpoint(self, checkpoint_path: str) -> None: ...

    async def save_checkpoint(self, checkpoint_path: str, tag: str) -> Any: ...
