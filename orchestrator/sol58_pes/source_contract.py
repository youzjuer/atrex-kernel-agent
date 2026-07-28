"""Language scheduling and standalone-source validation for SOL58 candidates."""

from __future__ import annotations

import ast
import json
import math
import re
import sys
from pathlib import Path


SOURCE_LANGUAGE_CUDA = "cuda_cpp"
SOURCE_LANGUAGE_CUTE = "cute_dsl"
SOURCE_LANGUAGE_AUTO = "auto"
SUPPORTED_CODE_LANGUAGES = {
    SOURCE_LANGUAGE_CUDA,
    SOURCE_LANGUAGE_CUTE,
    SOURCE_LANGUAGE_AUTO,
}

_INCLUDE_DIRECTIVE_RE = re.compile(r"^\s*#\s*include\b(.*)$")
_SOURCE_INCLUDE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu"}
_ALLOWED_CUTE_IMPORT_ROOTS = {"torch", "cutlass", "cuda"}
_STDLIB_IMPORT_ROOTS = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


def normalize_code_language(value: str | None, *, default: str) -> str:
    language = (value or default or SOURCE_LANGUAGE_CUDA).strip().lower()
    aliases = {
        "cuda": SOURCE_LANGUAGE_CUDA,
        "cu": SOURCE_LANGUAGE_CUDA,
        "cutedsl": SOURCE_LANGUAGE_CUTE,
        "cute": SOURCE_LANGUAGE_CUTE,
    }
    language = aliases.get(language, language)
    if language not in SUPPORTED_CODE_LANGUAGES:
        raise ValueError(
            f"unsupported SOL58 code language {language!r}; "
            "expected cuda_cpp, cute_dsl, or auto"
        )
    return language


def program_iteration(program_path: str) -> int | None:
    normalized = str(program_path).replace("\\", "/")
    matches = re.findall(r"(?:^|/)(\d+)/executor(?:/|$)", normalized)
    return int(matches[-1]) if matches else None


def required_source_language(
    program_path: str,
    code_language: str | None,
    cutedsl_rate: float | None,
    schedule_period: int | None,
    *,
    default_code_language: str,
    default_cutedsl_rate: float,
    default_schedule_period: int,
) -> str | None:
    mode = normalize_code_language(code_language, default=default_code_language)
    if mode != SOURCE_LANGUAGE_AUTO:
        return mode

    rate = default_cutedsl_rate if cutedsl_rate is None else float(cutedsl_rate)
    period = (
        default_schedule_period if schedule_period is None else int(schedule_period)
    )
    if not 0.0 <= rate <= 1.0:
        raise ValueError("SOL58_CUTEDSL_GENERATION_RATE must be between 0 and 1")
    if period <= 0:
        raise ValueError("SOL58_CUTEDSL_SCHEDULE_PERIOD must be positive")

    iteration = program_iteration(program_path)
    if iteration is None or rate <= 0.0:
        return None
    required_slots = min(period, math.ceil(rate * period))
    slot = (iteration - 1) % period
    return SOURCE_LANGUAGE_CUTE if slot < required_slots else None


def language_from_source_path(path: str) -> str | None:
    suffix = Path(path).suffix.lower()
    if suffix in {".cu", ".cpp", ".cc", ".cxx"}:
        return SOURCE_LANGUAGE_CUDA
    if suffix == ".py":
        return SOURCE_LANGUAGE_CUTE
    return None


def detect_source_language(kernel_source: str) -> str | None:
    if "PYBIND11_MODULE" in kernel_source and "#include" in kernel_source:
        return SOURCE_LANGUAGE_CUDA
    cute_import = re.search(
        r"(?m)^\s*(?:import\s+cutlass\.cute(?:\s+as\s+\w+)?|"
        r"from\s+cutlass(?:\.cute)?\s+import\s+)",
        kernel_source,
    )
    if cute_import and re.search(r"(?m)^\s*(?:async\s+)?def\s+run\s*\(", kernel_source):
        return SOURCE_LANGUAGE_CUTE
    return None


def extract_kernel_source(
    raw: str,
    code_language: str | None,
    *,
    default_code_language: str,
) -> str:
    text = raw.strip()
    mode = normalize_code_language(code_language, default=default_code_language)

    if text.startswith("{"):
        try:
            data = json.loads(text)
            candidates: list[tuple[str, str, str | None]] = []
            for source in data.get("sources", []):
                path = str(source.get("path", ""))
                content = source.get("content")
                if content:
                    candidates.append(
                        (path, str(content).strip(), language_from_source_path(path))
                    )
            if mode != SOURCE_LANGUAGE_AUTO:
                for _, content, path_language in candidates:
                    if path_language == mode or detect_source_language(content) == mode:
                        return content
            else:
                for _, content, _ in candidates:
                    if detect_source_language(content) is not None:
                        return content
                for _, content, path_language in candidates:
                    if path_language is not None:
                        return content
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            pass

    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    lines = text.splitlines()
    if lines and lines[0].strip().lower() in {
        "cuda",
        "cu",
        "cpp",
        "c++",
        "cuda_cpp",
        "kernel.cu",
        "python",
        "py",
        "cute",
        "cutedsl",
        "cute_dsl",
        "kernel.py",
    }:
        text = "\n".join(lines[1:]).strip()
    return text


def source_dependency_violation(kernel_source: str) -> str | None:
    """Return why a purported single-file kernel depends on another source file."""
    for line_number, line in enumerate(kernel_source.splitlines(), start=1):
        match = _INCLUDE_DIRECTIVE_RE.match(line)
        if not match:
            continue

        operand = match.group(1).strip()
        if operand.startswith('"'):
            return (
                f"line {line_number}: quoted include {operand!r} is not allowed; "
                "the submitted kernel must be self-contained"
            )
        if not operand.startswith("<") or ">" not in operand:
            return f"line {line_number}: dynamic include {operand!r} is not allowed"

        include_path = operand[1 : operand.index(">")].strip()
        normalized = include_path.replace("\\", "/")
        path_parts = normalized.split("/")
        suffix = Path(normalized).suffix.lower()
        if (
            not normalized
            or normalized.startswith("/")
            or ".." in path_parts
            or re.match(r"^[A-Za-z]:/", normalized)
        ):
            return f"line {line_number}: non-portable include path {include_path!r}"
        if suffix in _SOURCE_INCLUDE_SUFFIXES:
            return (
                f"line {line_number}: including source file {include_path!r} is not "
                "allowed; emit one complete kernel.cu"
            )
    return None


def python_dependency_violation(kernel_source: str) -> str | None:
    """Reject syntax errors, relative imports, and unavailable local Python modules."""
    try:
        tree = ast.parse(kernel_source)
    except SyntaxError as exc:
        return f"Python syntax error at line {exc.lineno}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return f"line {node.lineno}: relative imports are not allowed"
            modules = [node.module or ""]
        else:
            continue

        for module in modules:
            root = module.split(".", 1)[0]
            if (
                root not in _ALLOWED_CUTE_IMPORT_ROOTS
                and root not in _STDLIB_IMPORT_ROOTS
            ):
                return (
                    f"line {node.lineno}: import {module!r} is not an allowed "
                    "standalone CuTe DSL runtime dependency"
                )
    return None


def cute_dsl_validation_error(kernel_source: str) -> str | None:
    dependency_violation = python_dependency_violation(kernel_source)
    if dependency_violation:
        return dependency_violation
    if not re.search(
        r"(?m)^\s*(?:import\s+cutlass\.cute\s+as\s+cute|"
        r"from\s+cutlass\s+import\s+cute)",
        kernel_source,
    ):
        return (
            "missing `import cutlass.cute as cute` "
            "(or equivalent `from cutlass import cute`)"
        )
    if not re.search(r"@cute\.(?:kernel|jit)\b", kernel_source):
        return "missing a CuTe DSL `@cute.kernel` or `@cute.jit` definition"
    if "cute.compile" not in kernel_source:
        return "missing `cute.compile`"

    tree = ast.parse(kernel_source)
    run_function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ),
        None,
    )
    if run_function is None:
        return "missing destination-passing `run` function"
    positional = [*run_function.args.posonlyargs, *run_function.args.args]
    argument_names = [argument.arg for argument in positional]
    required = ["topk_idx", "sorted_token_indices", "expert_offsets"]
    if (
        argument_names != required
        or run_function.args.vararg is not None
        or run_function.args.kwonlyargs
        or run_function.args.defaults
    ):
        return (
            "run signature must be exactly `run(topk_idx, sorted_token_indices, "
            "expert_offsets)` with only optional **kwargs beyond those arguments"
        )
    return None


def candidate_validation_error(
    kernel_source: str,
    code_language: str | None,
    *,
    default_code_language: str,
) -> tuple[str | None, str | None]:
    mode = normalize_code_language(code_language, default=default_code_language)
    detected = detect_source_language(kernel_source)
    if detected is None:
        return (
            None,
            "candidate is neither a complete CUDA C++ source with PYBIND11_MODULE "
            "nor a complete CuTe DSL Python source with run()",
        )
    if mode != SOURCE_LANGUAGE_AUTO and detected != mode:
        return (
            detected,
            f"candidate language {detected!r} is not allowed in {mode!r} mode",
        )
    if detected == SOURCE_LANGUAGE_CUDA:
        if "#include" not in kernel_source or "PYBIND11_MODULE" not in kernel_source:
            return detected, "CUDA C++ source must contain includes and PYBIND11_MODULE"
        dependency_violation = source_dependency_violation(kernel_source)
        if dependency_violation:
            return detected, dependency_violation
        return detected, None
    return detected, cute_dsl_validation_error(kernel_source)
