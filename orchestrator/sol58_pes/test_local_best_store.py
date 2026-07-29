from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from orchestrator.sol58_pes.local_best_store import LocalBestContext, LocalBestStore


def _context(root: Path) -> LocalBestContext:
    profile = {"id": "profile-a", "name": "official"}
    return LocalBestContext(
        root=root,
        target_latency_ms=0.01,
        repeat_count=3,
        cache_schema_version=3,
        cuda_language="cuda_cpp",
        cute_language="cute_dsl",
        evaluation_stack_version="v1.1",
        gpu_type="B200",
        measurement_profile=lambda: dict(profile),
        local_evaluation_contract=lambda language, current: {
            "language": language,
            "profile": current,
        },
        parse_traces=lambda *args, **kwargs: {},
        language_from_source_path=lambda path: (
            "cute_dsl" if path.endswith(".py") else "cuda_cpp"
        ),
        detect_source_language=lambda source: (
            "cute_dsl" if "import torch" in source else "cuda_cpp"
        ),
        source_dependency_violation=lambda source: None,
        python_dependency_violation=lambda source: None,
    )


def _record(store: LocalBestStore, source: str, latency: float) -> dict[str, object]:
    return {
        "kernel_sha256": store.source_hash(source),
        "source_language": "cuda_cpp",
        "latency_ms_median": latency,
        "repeat_latencies_ms": [latency] * 3,
        "measurement_profile_id": "profile-a",
    }


class TestLocalBestStore(unittest.TestCase):
    def test_compare_and_swap_rejects_slower_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalBestStore(_context(Path(tmp)))
            self.assertTrue(store.save(_record(store, "fast", 0.8), "fast"))
            self.assertFalse(store.save(_record(store, "slow", 0.9), "slow"))
            persisted = json.loads(store.path.read_text(encoding="utf-8"))
            source = store.kernel_path("cuda_cpp").read_text(encoding="utf-8")

        self.assertEqual(persisted["kernel_sha256"], store.source_hash("fast"))
        self.assertEqual(source, "fast")

    def test_file_lock_keeps_fastest_concurrent_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalBestStore(_context(Path(tmp)))
            candidates = [
                (f"source-{index}", 1.0 - index / 100.0) for index in range(8)
            ]

            def save(candidate: tuple[str, float]) -> bool:
                source, latency = candidate
                return store.save(_record(store, source, latency), source)

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(save, candidates))
            persisted = json.loads(store.path.read_text(encoding="utf-8"))

        self.assertAlmostEqual(persisted["latency_ms_median"], 0.93)
        self.assertEqual(persisted["kernel_sha256"], store.source_hash("source-7"))

    def test_recovery_never_replaces_profile_best_with_slower_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalBestStore(_context(Path(tmp)))
            fast = _record(store, "fast", 0.8)
            fast["updated_at"] = "2026-01-01T00:00:00+00:00"
            slow = _record(store, "slow", 0.9)
            slow["updated_at"] = "2026-01-02T00:00:00+00:00"
            store.record_recovery(fast)
            store.record_recovery(slow)
            recovery = json.loads(store.recovery_path.read_text(encoding="utf-8"))

        self.assertEqual(
            recovery["profiles"]["profile-a"]["kernel_sha256"],
            store.source_hash("fast"),
        )

    def test_authoritative_fitness_is_profile_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalBestStore(_context(Path(tmp)))
            store.record_authoritative_fitness(
                "hash",
                source_language="cuda_cpp",
                official_score=0.9,
                official_latency_ms=0.006,
                local_latency_ms=0.007,
                submission_id=42,
            )
            registry = json.loads(
                store.authoritative_fitness_path.read_text(encoding="utf-8")
            )

        row = registry["sources"]["hash"]
        self.assertEqual(row["measurement_profile_id"], "profile-a")
        self.assertEqual(row["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
