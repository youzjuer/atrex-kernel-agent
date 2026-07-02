#!/usr/bin/env python3
"""tp=1 MoE performance shapes, consumed from the proj_019 workload catalog.

Shapes are the single source of truth in
``proj_019_moe_workload_op_opt/assets/workload_shape_catalog.py`` (task03). This
module only *reads* that catalog — it never copies or hardcodes shape numbers
(per task03 acceptance: "consume via catalog, do not spawn a one-off script").

Locate the catalog by setting ``PROJ019_ROOT`` to the proj_019 directory, or
place ``proj_019_moe_workload_op_opt`` near this repo (searched automatically).

Performance target shapes (both prefill, tp=1, single rank / all experts local):

- ``g11`` — Qwen3.5-Plus prefill, tokens=9500, nvfp4 (active P0)
- ``g8``  — Qwen3.5-Plus prefill, tokens 7680..8576 (P2 backlog)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

# Catalog deployment_ids selected as the tp=1 performance shapes.
TP1_PERF_GROUPS: Tuple[str, ...] = ("G11", "G8")


def _candidate_roots() -> Iterable[Path]:
    env = os.environ.get("PROJ019_ROOT")
    if env:
        yield Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        yield parent / "proj_019_moe_workload_op_opt"
        yield parent / "moe" / "proj_019_moe_workload_op_opt"


def _load_catalog_module():
    tried = []
    for root in _candidate_roots():
        cat = root / "assets" / "workload_shape_catalog.py"
        tried.append(str(cat))
        if cat.exists():
            spec = importlib.util.spec_from_file_location(
                "proj019_workload_shape_catalog", cat
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            # Register before exec so dataclass type resolution can see the module.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module, root
    raise FileNotFoundError(
        "proj_019 workload_shape_catalog.py not found. Set PROJ019_ROOT to the "
        "proj_019_moe_workload_op_opt directory. Searched:\n  " + "\n  ".join(tried[:8])
    )


def _group_to_shape(group_id: str, grp: Any, root: Path) -> Dict[str, Any]:
    if grp.tp_size != 1:
        raise ValueError(f"{group_id} is tp_size={grp.tp_size}, expected tp=1")
    return {
        "name": f"{grp.model}_{grp.role}_{group_id}_tp1",
        "group": group_id,
        "hidden_size": int(grp.hidden),
        "intermediate_size": int(grp.intermediate_hidden),
        "num_experts": int(grp.num_experts),
        "local_num_experts": int(grp.num_local_experts),
        "top_k": int(grp.top_k),
        "tokens": [int(t) for t in grp.tokens],
        "tp_size": int(grp.tp_size),
        "ep_size": int(grp.ep_size),
        "dtype": str(grp.dtype),
        "priority": str(grp.priority),
        "focus_status": str(grp.focus_status),
        "source": str(root),
    }


def load_tp1_shapes(groups: Iterable[str] = TP1_PERF_GROUPS) -> Dict[str, Dict[str, Any]]:
    """Return ``{preset_name: shape_dict}`` for the tp=1 performance groups."""
    catalog, root = _load_catalog_module()
    shapes: Dict[str, Dict[str, Any]] = {}
    for group_id in groups:
        grp = catalog.get_workload_group(group_id)
        shapes[group_id.lower()] = _group_to_shape(group_id, grp, root)
    return shapes


def get_shape(preset: str) -> Dict[str, Any]:
    shapes = load_tp1_shapes()
    key = preset.lower()
    if key not in shapes:
        raise KeyError(f"unknown tp=1 preset {preset!r}; available: {sorted(shapes)}")
    return shapes[key]


def list_presets() -> Tuple[str, ...]:
    return tuple(sorted(load_tp1_shapes()))


def is_catalog_preset(preset: str) -> bool:
    return preset.lower() in {g.lower() for g in TP1_PERF_GROUPS}


if __name__ == "__main__":
    import json

    print(json.dumps(load_tp1_shapes(), indent=2, ensure_ascii=False))
