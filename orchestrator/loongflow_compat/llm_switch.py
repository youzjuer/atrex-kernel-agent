#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility LLM switch helper for LoongFlow math agents.

Some local mlsys26-flashinfer-contest checkouts import this module from the
planner/executor/summary agents, but do not ship the file. Keep the behavior
conservative: use the configured model as-is so PES semantics are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMSwitchConfig:
    enabled: bool = False
    providers: list[dict[str, Any]] = field(default_factory=list)
    score_thresholds: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_any(cls, value: Any) -> "LLMSwitchConfig":
        if isinstance(value, cls):
            return value
        if not value:
            return cls()
        if isinstance(value, dict):
            return cls(
                enabled=bool(value.get("enabled", False)),
                providers=list(value.get("providers") or []),
                score_thresholds=list(value.get("score_thresholds") or []),
            )
        return cls()


def _provider_name(llm_config: Any) -> str:
    explicit = getattr(llm_config, "model_provider", None)
    if explicit:
        return str(explicit)

    model = str(getattr(llm_config, "model", "") or "")
    if "/" in model:
        return model.split("/", 1)[0] or "default"
    return "default"


def _previous_speedup(context: Any) -> float:
    for attr in ("previous_speedup", "best_speedup", "init_score"):
        value = getattr(context, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def choose_llm_config(
    context: Any,
    llm_config: Any,
    switch_config: Optional[LLMSwitchConfig] = None,
) -> tuple[Any, str, float]:
    _ = switch_config
    return llm_config, _provider_name(llm_config), _previous_speedup(context)
