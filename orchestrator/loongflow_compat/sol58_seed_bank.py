"""Validated architecture seeds used to escape a stagnant SOL58 population."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArchitectureSeed:
    seed_id: str
    family: str
    language: str
    purpose: str
    source: str
    source_path: str
    source_sha256: str


def default_seed_manifest() -> Path:
    return Path(__file__).resolve().parents[1] / "sol58_pes" / "seed_bank.json"


def configured_seed_manifest() -> Path:
    configured = os.environ.get("ATREX_PES_SEED_MANIFEST", "").strip()
    return Path(configured).expanduser() if configured else default_seed_manifest()


def load_seed_bank(
    manifest_path: str | Path | None = None,
    *,
    language: str = "cuda_cpp",
) -> list[ArchitectureSeed]:
    """Load a deterministic, self-contained seed bank and reject bad manifests."""
    path = (
        Path(manifest_path) if manifest_path is not None else configured_seed_manifest()
    )
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("seeds"), list):
        raise ValueError(f"unsupported SOL58 seed manifest: {path}")

    selected: list[ArchitectureSeed] = []
    seen: set[str] = set()
    for raw in payload["seeds"]:
        if not isinstance(raw, dict):
            raise ValueError(f"invalid seed entry in {path}")
        seed_id = str(raw.get("id") or "").strip()
        family = str(raw.get("family") or "").strip()
        seed_language = str(raw.get("language") or "").strip()
        relative_path = str(raw.get("path") or "").strip()
        purpose = str(raw.get("purpose") or "").strip()
        if not seed_id or seed_id in seen:
            raise ValueError(f"missing or duplicate seed id {seed_id!r} in {path}")
        seen.add(seed_id)
        if not family or not seed_language or not relative_path or not purpose:
            raise ValueError(f"incomplete seed entry {seed_id!r} in {path}")
        if seed_language != language:
            continue

        source_path = (path.parent / relative_path).resolve()
        try:
            source_path.relative_to(path.parent)
        except ValueError as exc:
            raise ValueError(
                f"seed {seed_id!r} escapes manifest directory: {source_path}"
            ) from exc
        source = source_path.read_text(encoding="utf-8").strip()
        if not source:
            raise ValueError(f"seed {seed_id!r} is empty: {source_path}")
        if "PYBIND11_MODULE" not in source or "void run(" not in source:
            raise ValueError(f"CUDA seed {seed_id!r} lacks the SOL58 entry point")
        selected.append(
            ArchitectureSeed(
                seed_id=seed_id,
                family=family,
                language=seed_language,
                purpose=purpose,
                source=source,
                source_path=str(source_path),
                source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )
        )

    if not selected:
        raise ValueError(f"no {language!r} architecture seeds in {path}")
    return selected


def seed_bank_fingerprint(seeds: list[ArchitectureSeed]) -> str:
    material = "\n".join(
        f"{seed.seed_id}:{seed.family}:{seed.source_sha256}" for seed in seeds
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
