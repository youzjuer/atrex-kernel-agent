#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evolutionary database manager for the atrex full-agent (PES) optimization loop.

This tool is the state hub / single source of truth for the evolutionary
Plan-Execute-Summary loop. It carries what ``memory_manager.py`` (a linear
``memory/v<N>.json`` sequence) cannot: a multi-island MAP-Elites population, an
elite archive, parent lineage (``parent_id``), and diversity-adaptive Boltzmann
parent selection.

Algorithms and default parameters mirror the upstream LoongFlow approach
(``agentsdk/memory/evolution/{in_memory.py, boltzmann.py, base_memory.py}``),
reimplemented with the Python standard library only (no numpy) so the tool runs
on any host, GPU box or not. See ``docs/evolution-db-design.md`` for the full
spec.

What this tool does NOT do: compile/run/profile kernels, call an LLM, or manage
git. Score and correctness are supplied from the outside via ``add``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Defaults (= upstream LoongFlow defaults; see reference/evolve_config.schema.json)
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG: Dict[str, Any] = {
    "population_size": 100,
    "num_islands": 3,
    "elite_archive_size": 50,
    "migration_interval": 10,
    "migration_rate": 0.2,
    "feature_dimensions": ["complexity", "diversity", "score"],
    "feature_bins": None,
    "diversity_reference_size": 20,
    "selection": {
        "strategy": "boltzmann",
        "initial_temperature": 1.0,
        "min_temperature": 0.5,
        "max_temperature": 2.0,
        "exploration_rate": 0.2,
        "use_sampling_weight": True,
        "sampling_weight_power": 1.0,
    },
    "n_candidates": 3,
    "budget": {"max_generations": 30},
    "convergence": {"no_improve_patience": 5, "min_rel_improve": 0.01},
    "score_metric": "speedup_vs_baseline",
}

VALUES_CAP = 1000  # max retained feature values for min-max stats


# --------------------------------------------------------------------------- #
# Solution record (fields aligned to upstream Solution dataclass)
# --------------------------------------------------------------------------- #


@dataclass
class Solution:
    solution_id: str = ""
    parent_id: Optional[str] = None
    generate_plan: str = ""
    island_id: int = 0
    iteration: int = 0
    generation: int = 0
    timestamp: float = 0.0
    sample_cnt: int = 0
    sample_weight: float = 0.0
    score: float = 0.0
    evaluation: str = ""
    summary: str = ""
    solution: str = ""  # relative path to the code snapshot
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Solution":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


# --------------------------------------------------------------------------- #
# Diversity helpers (upstream base_memory.py)
# --------------------------------------------------------------------------- #


def fast_code_distance(c1: str, c2: str) -> float:
    """Cheap code distance for the MAP-Elites ``diversity`` feature.

    Mirrors upstream ``_fast_code_diversity``:
    0.1*|len diff| + 10*|line diff| + 0.5*|chars in larger not in smaller|.
    """
    if c1 == c2:
        return 0.0
    len1, len2 = len(c1), len(c2)
    line_diff = abs(c1.count("\n") - c2.count("\n")) * 10
    length_diff = abs(len1 - len2) * 0.1
    smaller, larger = (c1, c2) if len1 < len2 else (c2, c1)
    small_set = set(smaller)
    char_diff = sum(1 for ch in set(larger) if ch not in small_set) * 0.5
    return length_diff + line_diff + char_diff


def population_diversity(codes: List[str], sample_size: int = 50) -> float:
    """Normalized population diversity in [0,1] for adaptive temperature.

    Mirrors upstream ``_calculate_diversity``: mean over sampled pairs of
    0.4*len_diff + 0.3*line_diff + 0.3*charset_symmetric_diff (each normalized).
    """
    n = len(codes)
    if n <= 1:
        return 0.0
    if n > sample_size:
        codes = random.sample(codes, sample_size)
    scores: List[float] = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            s1, s2 = codes[i] or "", codes[j] or ""
            len_diff = abs(len(s1) - len(s2)) / max(1, max(len(s1), len(s2)))
            l1, l2 = s1.count("\n"), s2.count("\n")
            line_diff = abs(l1 - l2) / max(1, max(l1, l2))
            set1, set2 = set(s1), set(s2)
            char_diff = len(set1 ^ set2) / max(1, max(len(set1), len(set2)))
            scores.append(0.4 * len_diff + 0.3 * line_diff + 0.3 * char_diff)
    return sum(scores) / len(scores) if scores else 0.0


def diverse_reference_set(codes: List[str], ref_size: int) -> List[str]:
    """Greedy most-diverse subset (upstream ``_update_diversity_reference_set``)."""
    if len(codes) <= ref_size:
        return list(codes)
    # seed with the two most distant
    selected: List[str] = []
    best_pair = (0, 1)
    best_d = -1.0
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            d = fast_code_distance(codes[i], codes[j])
            if d > best_d:
                best_d, best_pair = d, (i, j)
    selected = [codes[best_pair[0]], codes[best_pair[1]]]
    remaining = [c for k, c in enumerate(codes) if k not in best_pair]
    while len(selected) < ref_size and remaining:
        nxt = max(remaining, key=lambda c: min(fast_code_distance(c, s) for s in selected))
        selected.append(nxt)
        remaining.remove(nxt)
    return selected


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class DBError(Exception):
    """Raised for state errors (exit code 3)."""


class NotFound(Exception):
    """Raised when a referenced entity is missing (exit code 4)."""


# --------------------------------------------------------------------------- #
# Evolutionary database
# --------------------------------------------------------------------------- #


class EvolutionDB:
    def __init__(self, workspace: str):
        self.ws = Path(workspace)
        self.db = self.ws / "database"
        self.sol_dir = self.db / "solutions"
        self.ckpt_dir = self.db / "checkpoints"
        self.config_path = self.db / "config.json"
        self.state_path = self.db / "state.json"
        self.config: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}

    # ----- persistence ----------------------------------------------------- #

    def _empty_state(self) -> Dict[str, Any]:
        n = self.config["num_islands"]
        return {
            "solutions": {},
            "islands": [[] for _ in range(n)],
            "island_feature_maps": [{} for _ in range(n)],
            "island_best": [None] * n,
            "elites": [],
            "feature_stats": {},
            "current_island": 0,
            "current_island_counter": 0,
            "last_migration_generation": 0,
            "best_solution_id": None,
            "best_score": None,
            "baseline": {"solution_id": None, "latency_us": None},
            "history": [],
            "convergence": {"no_improve_streak": 0, "stopped": False, "stop_reason": None},
        }

    def load(self) -> None:
        if not self.config_path.exists():
            raise DBError(f"database not initialized at {self.db} (run `init` first)")
        self.config = json.loads(self.config_path.read_text())
        self.state = json.loads(self.state_path.read_text())

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))

    def _write_config(self) -> None:
        self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False))

    # ----- config helpers -------------------------------------------------- #

    def feature_bins(self) -> int:
        fb = self.config.get("feature_bins")
        if fb:
            return int(fb)
        dims = len(self.config["feature_dimensions"])
        return int(pow(self.config["elite_archive_size"], 1.0 / dims) + 0.99)

    # ----- solution io ----------------------------------------------------- #

    def _sol(self, sid: str) -> Dict[str, Any]:
        s = self.state["solutions"].get(sid)
        if s is None:
            raise NotFound(f"solution {sid} not found")
        return s

    def _code_text(self, sol: Dict[str, Any]) -> str:
        p = self.ws / sol["solution"]
        try:
            return p.read_text()
        except OSError:
            return ""

    def _persist_solution(self, sol: Dict[str, Any]) -> None:
        d = self.sol_dir / sol["solution_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "solution.json").write_text(json.dumps(sol, indent=2, ensure_ascii=False))

    # ----- feature stats / binning (upstream base_memory.py) --------------- #

    def _update_stat(self, name: str, value: float) -> None:
        fs = self.state["feature_stats"]
        if name not in fs:
            fs[name] = {"min": value, "max": value, "values": []}
        st = fs[name]
        st["min"] = min(st["min"], value)
        st["max"] = max(st["max"], value)
        st["values"].append(value)
        if len(st["values"]) > VALUES_CAP:
            st["values"] = st["values"][-VALUES_CAP:]

    def _scale(self, name: str, value: float) -> float:
        st = self.state["feature_stats"].get(name)
        if not st or st["max"] == st["min"]:
            return 0.5
        return min(1.0, max(0.0, (value - st["min"]) / (st["max"] - st["min"])))

    def _bin(self, name: str, value: float) -> int:
        self._update_stat(name, value)
        scaled = self._scale(name, value)
        nb = self.feature_bins()
        return max(0, min(nb - 1, int(scaled * nb)))

    def _map_elites_key(self, sol: Dict[str, Any]) -> str:
        code = self._code_text(sol)
        coords: List[int] = []
        for dim in self.config["feature_dimensions"]:
            if dim == "complexity":
                v = float(len(code))
            elif dim == "diversity":
                others = [
                    self._code_text(s)
                    for sid, s in self.state["solutions"].items()
                    if sid != sol["solution_id"]
                ]
                if len(others) < 1:
                    coords.append(0)
                    continue
                ref = diverse_reference_set(others, self.config["diversity_reference_size"])
                ds = [fast_code_distance(code, r) for r in ref if r != code]
                v = sum(ds) / len(ds) if ds else 0.0
            elif dim == "score":
                v = float(sol["score"])
            else:
                raise DBError(f"unsupported feature dimension: {dim}")
            coords.append(self._bin(dim, v))
        return "-".join(str(c) for c in coords)

    # ----- population mechanics (upstream in_memory.py) -------------------- #

    def _assign_island(self, explicit: Optional[int]) -> int:
        n = self.config["num_islands"]
        if explicit is not None:
            if not 0 <= explicit < n:
                raise DBError(f"island {explicit} out of range [0,{n})")
            return explicit
        isl = self.state["current_island"]
        per = max(1, self.config["population_size"] // n)
        self.state["current_island_counter"] += 1
        if self.state["current_island_counter"] >= per:
            self.state["current_island_counter"] = 0
            self.state["current_island"] = (isl + 1) % n
        return isl

    def _score_of(self, sid: str) -> float:
        return float(self.state["solutions"][sid]["score"] or 0.0)

    def _update_island(self, sol: Dict[str, Any]) -> None:
        isl = sol["island_id"]
        key = sol["metadata"]["MAP_Elite_feature"]
        fmap = self.state["island_feature_maps"][isl]
        members = self.state["islands"][isl]
        occupant = fmap.get(key)
        if occupant is None:
            fmap[key] = sol["solution_id"]
            if sol["solution_id"] not in members:
                members.append(sol["solution_id"])
        elif self._score_of(sol["solution_id"]) > self._score_of(occupant):
            fmap[key] = sol["solution_id"]
            if occupant in members:
                members.remove(occupant)
            if sol["solution_id"] not in members:
                members.append(sol["solution_id"])
        # else: new solution does not occupy a cell (stays in populations pool)

    def _update_elites(self, sol: Dict[str, Any]) -> None:
        elites = self.state["elites"]
        if sol["solution_id"] not in elites:
            elites.append(sol["solution_id"])
        elites.sort(key=self._score_of, reverse=True)
        cap = self.config["elite_archive_size"]
        if len(elites) > cap:
            del elites[cap:]

    def _update_island_best(self, isl: int) -> None:
        members = self.state["islands"][isl]
        if members:
            self.state["island_best"][isl] = max(members, key=self._score_of)

    def _check_migration(self, generation: int) -> None:
        n = self.config["num_islands"]
        if n <= 1 or generation <= 0:
            return
        interval = self.config["migration_interval"]
        if generation - self.state["last_migration_generation"] < interval:
            return
        rate = self.config["migration_rate"]
        snapshot = [list(m) for m in self.state["islands"]]
        for i in range(n):
            members = snapshot[i]
            if not members:
                continue
            k = max(1, int(math.ceil(rate * len(members))))
            migrants = sorted(members, key=self._score_of, reverse=True)[:k]
            tgt = (i + 1) % n
            for mid in migrants:
                sol = self.state["solutions"][mid]
                key = sol["metadata"]["MAP_Elite_feature"]
                fmap = self.state["island_feature_maps"][tgt]
                tmembers = self.state["islands"][tgt]
                occ = fmap.get(key)
                if occ is None or self._score_of(mid) > self._score_of(occ):
                    fmap[key] = mid
                    if occ in tmembers:
                        tmembers.remove(occ)
                    if mid not in tmembers:
                        tmembers.append(mid)
            self._update_island_best(tgt)
        self.state["last_migration_generation"] = generation

    def _prune(self) -> None:
        cap = self.config["population_size"]
        sols = self.state["solutions"]
        while len(sols) > cap:
            worst = min(sols, key=self._score_of)
            del sols[worst]
            for members in self.state["islands"]:
                if worst in members:
                    members.remove(worst)
            for fmap in self.state["island_feature_maps"]:
                for k in [k for k, v in fmap.items() if v == worst]:
                    del fmap[k]
            if worst in self.state["elites"]:
                self.state["elites"].remove(worst)

    def _update_best(self) -> None:
        sols = self.state["solutions"]
        if not sols:
            self.state["best_solution_id"] = None
            self.state["best_score"] = None
            return
        bid = max(sols, key=self._score_of)
        self.state["best_solution_id"] = bid
        self.state["best_score"] = self._score_of(bid)

    # ----- adaptive-temperature Boltzmann selection (upstream boltzmann.py) - #

    def _adaptive_temperature(self) -> float:
        sel = self.config["selection"]
        base = sel["initial_temperature"]
        codes = [self._code_text(s) for s in self.state["solutions"].values()]
        div = population_diversity(codes)
        new_temp = base * (1 + (2 * div - 1))
        new_temp = max(sel["min_temperature"], min(sel["max_temperature"], new_temp))
        return 0.8 * new_temp + 0.2 * base

    def _select_one(self, temperature: float) -> Optional[str]:
        sel = self.config["selection"]
        sols = self.state["solutions"]
        if not sols:
            return None
        ids = list(sols)
        if random.random() < sel["exploration_rate"]:
            return random.choice(ids)
        elites = [e for e in self.state["elites"] if e in sols]
        non_elites = [i for i in ids if i not in set(elites)]
        cand: List[str] = []
        cand.extend(random.sample(elites, min(3, len(elites))))
        cand.extend(random.sample(non_elites, min(2, len(non_elites))))
        if len(cand) < 5:
            pool = elites if len(elites) > len(non_elites) else non_elites
            extra = [x for x in pool if x not in cand]
            random.shuffle(extra)
            cand.extend(extra[: 5 - len(cand)])
        if not cand:
            cand = ids[:]
        scores = [self._score_of(c) for c in cand]
        max_s = max(scores)
        probs = [math.exp((s - max_s) / temperature) for s in scores]
        if sel["use_sampling_weight"]:
            power = sel["sampling_weight_power"]
            weights = [
                (sols[c]["sample_weight"] or 1.0) ** power for c in cand
            ]
            probs = [p * w for p, w in zip(probs, weights)]
        total = sum(probs)
        if total <= 0 or any(math.isnan(p) for p in probs):
            return max(cand, key=self._score_of)
        r = random.random() * total
        acc = 0.0
        for c, p in zip(cand, probs):
            acc += p
            if r <= acc:
                return c
        return cand[-1]

    # ----- public commands ------------------------------------------------- #

    def init(self, config_file: Optional[str]) -> Dict[str, Any]:
        self.db.mkdir(parents=True, exist_ok=True)
        self.sol_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(DEFAULT_CONFIG)
        if config_file:
            cfg.update(json.loads(Path(config_file).read_text()))
        self.config = cfg
        self._write_config()
        self.state = self._empty_state()
        self.save()
        return {"database": str(self.db), "config": str(self.config_path)}

    def _next_solution_id(self, code: str, parent: Optional[str], generation: int) -> str:
        raw = f"{code}|{parent}|{generation}|{time.time()}|{random.random()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:8]

    def _admit(self, sol: Dict[str, Any]) -> None:
        sid = sol["solution_id"]
        self.state["solutions"][sid] = sol
        sol["metadata"]["MAP_Elite_feature"] = self._map_elites_key(sol)
        self._update_island(sol)
        self._update_elites(sol)
        self._update_island_best(sol["island_id"])
        self._check_migration(sol["generation"])
        self._prune()
        self._update_best()
        self._persist_solution(sol)

    def import_seed(self, from_mem: str, code: str) -> Dict[str, Any]:
        code_path = self.ws / code
        if not code_path.exists():
            raise NotFound(f"seed code not found: {code_path}")
        latency = None
        mem_path = self.ws / from_mem
        if mem_path.exists():
            try:
                mem = json.loads(mem_path.read_text())
                latency = (mem.get("performance") or {}).get("latency_us")
            except (OSError, json.JSONDecodeError):
                latency = None
        code_text = code_path.read_text()
        sid = self._next_solution_id(code_text, None, 0)
        dest = self.sol_dir / sid
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(code_path, dest / "kernel.py")
        sol = Solution(
            solution_id=sid,
            parent_id=None,
            generate_plan="baseline seed",
            island_id=0,
            iteration=0,
            generation=0,
            timestamp=time.time(),
            score=1.0,  # speedup vs itself
            evaluation="seed (baseline)",
            solution=str((dest / "kernel.py").relative_to(self.ws)),
            metadata={
                "lang": "",
                "code_sha256": hashlib.sha256(code_text.encode()).hexdigest(),
                "correctness": {"rel_err": None, "status": "PASS"},
                "performance": {"latency_us": latency},
                "optimization": {"action_category": "baseline", "action_description": "seed"},
            },
        ).to_dict()
        self.state["baseline"] = {"solution_id": sid, "latency_us": latency}
        self._admit(sol)
        self.save()
        self.checkpoint(0)
        return {"solution_id": sid, "baseline_latency_us": latency}

    def add(self, args: argparse.Namespace) -> Dict[str, Any]:
        code_path = Path(args.code)
        if not code_path.is_absolute():
            code_path = self.ws / args.code
        if not code_path.exists():
            raise NotFound(f"code not found: {code_path}")
        code_text = code_path.read_text()
        sid = self._next_solution_id(code_text, args.parent, args.generation)
        dest = self.sol_dir / sid
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(code_path, dest / "kernel.py")

        correctness = args.correctness
        score = args.score
        if score is None:
            base_lat = self.state["baseline"]["latency_us"]
            if correctness == "PASS" and base_lat and args.latency_us:
                score = base_lat / args.latency_us
            else:
                score = 0.0
        if correctness != "PASS":
            score = 0.0

        island = self._assign_island(args.island)
        parent = None if (args.parent in (None, "null", "None")) else args.parent
        sol = Solution(
            solution_id=sid,
            parent_id=parent,
            generate_plan=args.generate_plan or "",
            island_id=island,
            iteration=args.generation,
            generation=args.generation,
            timestamp=time.time(),
            score=float(score),
            evaluation=args.evaluation or "",
            solution=str((dest / "kernel.py").relative_to(self.ws)),
            metadata={
                "lang": args.lang or "",
                "code_sha256": hashlib.sha256(code_text.encode()).hexdigest(),
                "correctness": {"rel_err": args.rel_err, "status": correctness},
                "performance": {
                    "latency_us": args.latency_us,
                    "tflops": args.tflops,
                    "bandwidth_gbps": args.bandwidth_gbps,
                },
                "optimization": {
                    "action_category": args.action_category or "",
                    "action_description": "",
                },
                "iteration_ref": args.iteration_ref or "",
            },
        ).to_dict()
        if args.evidence_file:
            try:
                sol["metadata"]["profile_evidence"] = json.loads(
                    Path(args.evidence_file).read_text()
                )
            except (OSError, json.JSONDecodeError):
                pass
        self._admit(sol)
        self.save()
        occupies = sol["metadata"]["MAP_Elite_feature"] in {
            k for k in self.state["island_feature_maps"][island]
            if self.state["island_feature_maps"][island][k] == sid
        }
        return {
            "solution_id": sid,
            "island_id": island,
            "score": score,
            "map_elite_feature": sol["metadata"]["MAP_Elite_feature"],
            "occupies_cell": occupies,
            "is_elite": sid in self.state["elites"],
        }

    def select_parents(self, n: int) -> List[Dict[str, Any]]:
        if not self.state["solutions"]:
            raise DBError("population is empty; import a seed first")
        temp = self._adaptive_temperature()
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            pid = self._select_one(temp)
            if pid is None:
                break
            self.state["solutions"][pid]["sample_cnt"] += 1
            s = self.state["solutions"][pid]
            out.append(
                {
                    "solution_id": pid,
                    "score": s["score"],
                    "island_id": s["island_id"],
                    "generation": s["generation"],
                    "code": s["solution"],
                    "generate_plan": s["generate_plan"],
                    "action_category": (s["metadata"].get("optimization") or {}).get(
                        "action_category", ""
                    ),
                    "summary": s["summary"],
                }
            )
        self.save()
        return out

    def checkpoint(self, generation: int) -> Dict[str, Any]:
        self._update_best()
        best_id = self.state["best_solution_id"]
        best_score = self.state["best_score"]

        # convergence vs previous checkpoint best
        conv = self.state["convergence"]
        prev_best = self.state["history"][-1]["best_score"] if self.state["history"] else None
        if prev_best is not None and best_score is not None:
            denom = abs(prev_best) if prev_best else 1.0
            rel = (best_score - prev_best) / denom
            if rel < self.config["convergence"]["min_rel_improve"]:
                conv["no_improve_streak"] += 1
            else:
                conv["no_improve_streak"] = 0
        self.state["history"].append(
            {"generation": generation, "best_score": best_score, "best_solution_id": best_id}
        )
        patience = self.config["convergence"]["no_improve_patience"]
        max_gen = self.config["budget"]["max_generations"]
        if conv["no_improve_streak"] >= patience:
            conv["stopped"], conv["stop_reason"] = True, "no_improve"
        elif generation >= max_gen:
            conv["stopped"], conv["stop_reason"] = True, "budget_exhausted"

        cdir = self.ckpt_dir / f"iter-{generation}"
        (cdir / "solutions").mkdir(parents=True, exist_ok=True)
        meta = self._build_metadata(generation)
        (cdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        if best_id:
            (cdir / "best_solution.json").write_text(
                json.dumps(self.state["solutions"][best_id], indent=2, ensure_ascii=False)
            )
        for sid, sol in self.state["solutions"].items():
            (cdir / "solutions" / f"{sid}.json").write_text(
                json.dumps(sol, indent=2, ensure_ascii=False)
            )
        self.save()
        return {
            "checkpoint": str(cdir),
            "best_solution_id": best_id,
            "best_score": best_score,
            "stopped": conv["stopped"],
            "stop_reason": conv["stop_reason"],
            "no_improve_streak": conv["no_improve_streak"],
        }

    def _build_metadata(self, generation: int) -> Dict[str, Any]:
        islands_state = []
        for i in range(self.config["num_islands"]):
            islands_state.append(
                {
                    "island_id": i,
                    "members": list(self.state["islands"][i]),
                    "best_solution_id": self.state["island_best"][i],
                    "feature_map": dict(self.state["island_feature_maps"][i]),
                }
            )
        fs = {
            k: {"min": v["min"], "max": v["max"], "values": v["values"][-100:]}
            for k, v in self.state["feature_stats"].items()
        }
        return {
            "generation": generation,
            "timestamp": time.time(),
            "baseline": self.state["baseline"],
            "best_solution_id": self.state["best_solution_id"],
            "best_score": self.state["best_score"],
            "population_size_current": len(self.state["solutions"]),
            "elites": list(self.state["elites"]),
            "islands_state": islands_state,
            "feature_stats": fs,
            "migration": {"last_migration_generation": self.state["last_migration_generation"]},
            "history": list(self.state["history"]),
            "convergence": dict(self.state["convergence"]),
        }

    def best(self, island: Optional[int]) -> Dict[str, Any]:
        if island is not None:
            bid = self.state["island_best"][island]
            if not bid:
                raise NotFound(f"island {island} has no solutions")
            return self.state["solutions"][bid]
        if not self.state["best_solution_id"]:
            raise NotFound("no solutions in database")
        return self.state["solutions"][self.state["best_solution_id"]]

    def lineage(self, sid: str) -> List[Dict[str, Any]]:
        chain: List[Dict[str, Any]] = []
        cur: Optional[str] = sid
        seen = set()
        while cur and cur not in seen:
            seen.add(cur)
            s = self._sol(cur)
            chain.append(
                {
                    "solution_id": cur,
                    "parent_id": s["parent_id"],
                    "generation": s["generation"],
                    "score": s["score"],
                    "action_category": (s["metadata"].get("optimization") or {}).get(
                        "action_category", ""
                    ),
                }
            )
            cur = s["parent_id"]
        return chain

    def summary(self) -> Dict[str, Any]:
        return {
            "generations": len(self.state["history"]),
            "population_size": len(self.state["solutions"]),
            "num_islands": self.config["num_islands"],
            "island_sizes": [len(m) for m in self.state["islands"]],
            "elites": len(self.state["elites"]),
            "best_solution_id": self.state["best_solution_id"],
            "best_score": self.state["best_score"],
            "baseline": self.state["baseline"],
            "convergence": self.state["convergence"],
            "history": self.state["history"],
        }

    def list_solutions(self, generation: Optional[int], island: Optional[int]) -> List[Dict[str, Any]]:
        out = []
        for sid, s in self.state["solutions"].items():
            if generation is not None and s["generation"] != generation:
                continue
            if island is not None and s["island_id"] != island:
                continue
            out.append(
                {
                    "solution_id": sid,
                    "generation": s["generation"],
                    "island_id": s["island_id"],
                    "score": s["score"],
                    "correctness": (s["metadata"].get("correctness") or {}).get("status"),
                    "parent_id": s["parent_id"],
                }
            )
        out.sort(key=lambda x: (x["generation"], -(x["score"] or 0)))
        return out

    def config_op(self, get: Optional[str], sets: Optional[List[str]]) -> Dict[str, Any]:
        if sets:
            for kv in sets:
                if "=" not in kv:
                    raise DBError(f"--set expects key=value, got {kv}")
                key, val = kv.split("=", 1)
                try:
                    parsed = json.loads(val)
                except json.JSONDecodeError:
                    parsed = val
                node = self.config
                parts = key.split(".")
                for p in parts[:-1]:
                    node = node.setdefault(p, {})
                node[parts[-1]] = parsed
            self._write_config()
        if get:
            node: Any = self.config
            for p in get.split("."):
                node = node[p]
            return {get: node}
        return self.config


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _emit(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    elif isinstance(obj, list):
        for row in obj:
            print(json.dumps(row, ensure_ascii=False) if isinstance(row, dict) else row)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}")
    else:
        print(obj)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evolutionary database manager for the atrex PES loop.")
    p.add_argument("--seed", type=int, default=None, help="Seed RNG for reproducibility.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--workspace", required=True)
        sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("init", help="Create database/ and config.json.")
    common(sp)
    sp.add_argument("--config", default=None)

    sp = sub.add_parser("import-seed", help="Register baseline (v0) as the seed solution.")
    common(sp)
    sp.add_argument("--from", dest="from_mem", default="memory/v0.json")
    sp.add_argument("--code", default="kernel.py")

    sp = sub.add_parser("add", help="Register an evaluated candidate.")
    common(sp)
    sp.add_argument("--generation", type=int, required=True)
    sp.add_argument("--parent", default=None)
    sp.add_argument("--code", required=True)
    sp.add_argument("--lang", default=None)
    sp.add_argument("--score", type=float, default=None)
    sp.add_argument("--correctness", choices=["PASS", "FAIL", "TIMEOUT_FAIL"], required=True)
    sp.add_argument("--rel-err", dest="rel_err", type=float, default=None)
    sp.add_argument("--latency-us", dest="latency_us", type=float, default=None)
    sp.add_argument("--tflops", type=float, default=None)
    sp.add_argument("--bandwidth-gbps", dest="bandwidth_gbps", type=float, default=None)
    sp.add_argument("--island", type=int, default=None)
    sp.add_argument("--generate-plan", dest="generate_plan", default=None)
    sp.add_argument("--action-category", dest="action_category", default=None)
    sp.add_argument("--evaluation", default=None)
    sp.add_argument("--evidence-file", dest="evidence_file", default=None)
    sp.add_argument("--iteration-ref", dest="iteration_ref", default=None)

    sp = sub.add_parser("select-parents", help="Sample N parents (adaptive Boltzmann).")
    common(sp)
    sp.add_argument("--n", type=int, default=None)

    sp = sub.add_parser("checkpoint", help="Finalize a generation.")
    common(sp)
    sp.add_argument("--generation", type=int, required=True)

    sp = sub.add_parser("best", help="Print the current best solution.")
    common(sp)
    sp.add_argument("--island", type=int, default=None)

    sp = sub.add_parser("lineage", help="Walk a solution's parent chain.")
    common(sp)
    sp.add_argument("--solution", required=True)

    sp = sub.add_parser("summary", help="Print progress summary.")
    common(sp)

    sp = sub.add_parser("list", help="List solutions.")
    common(sp)
    sp.add_argument("--generation", type=int, default=None)
    sp.add_argument("--island", type=int, default=None)

    sp = sub.add_parser("config", help="Get/set evolution config.")
    common(sp)
    sp.add_argument("--get", default=None)
    sp.add_argument("--set", dest="set_", action="append", default=None)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)
    db = EvolutionDB(args.workspace)
    try:
        if args.command == "init":
            _emit(db.init(args.config), args.json)
            return 0
        db.load()
        if args.command == "import-seed":
            _emit(db.import_seed(args.from_mem, args.code), args.json)
        elif args.command == "add":
            _emit(db.add(args), args.json)
        elif args.command == "select-parents":
            n = args.n if args.n is not None else db.config["n_candidates"]
            _emit(db.select_parents(n), args.json)
        elif args.command == "checkpoint":
            _emit(db.checkpoint(args.generation), args.json)
        elif args.command == "best":
            _emit(db.best(args.island), args.json)
        elif args.command == "lineage":
            _emit(db.lineage(args.solution), args.json)
        elif args.command == "summary":
            _emit(db.summary(), args.json)
        elif args.command == "list":
            _emit(db.list_solutions(args.generation, args.island), args.json)
        elif args.command == "config":
            _emit(db.config_op(args.get, args.set_), args.json)
        return 0
    except DBError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except NotFound as e:
        print(f"error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
