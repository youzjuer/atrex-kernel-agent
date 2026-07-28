"""SOL-ExecBench submission serialization and bounded HTTP polling."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from orchestrator.sol58_pes.evaluator_state import tail


@dataclass(frozen=True)
class OfficialApiConfig:
    base_url: str
    request_timeout: float
    poll_interval: float
    pending_result_grace: float
    async_refresh_delay: float
    poll_timeout: float
    kernel_id: int
    gpu_type: str
    evaluation_stack_version: str
    terminal_statuses: frozenset[str]


def official_compile_options(solution: dict[str, Any]) -> dict[str, Any]:
    spec = dict(solution.get("spec") or {})
    compile_options = dict(spec.get("compile_options") or {})
    cuda_cflags = list(compile_options.get("cuda_cflags") or [])
    compile_options["cuda_cflags"] = [
        flag
        for flag in cuda_cflags
        if "-gencode" not in str(flag) and not str(flag).startswith("-arch")
    ]
    compile_options.setdefault("ld_flags", ["-lcuda"])
    return compile_options


def build_official_submission(
    workspace: Path,
    *,
    gpu_type: str,
    cute_language: str,
    author: str | None = None,
) -> dict[str, Any]:
    solution = json.loads((workspace / "solution.json").read_text(encoding="utf-8"))
    solution["name"] = f"sol58_pes_official_{uuid.uuid4().hex[:8]}"
    solution["author"] = author or os.environ.get(
        "SOL58_OFFICIAL_AUTHOR", "atrex-loongflow-pes"
    )
    languages = set((solution.get("spec") or {}).get("languages") or [])
    language_label = "CuTe DSL" if cute_language in languages else "CUDA C++"
    solution["description"] = (
        f"LoongFlow PES-generated {language_label} candidate for SOL kernel 58; "
        "official v1.1 fitness calibration."
    )

    spec = dict(solution.get("spec") or {})
    spec["target_hardware"] = [gpu_type]
    if cute_language in languages:
        spec.pop("compile_options", None)
        spec.pop("binding", None)
    else:
        spec["compile_options"] = official_compile_options(solution)
    solution["spec"] = spec

    for source in solution.get("sources", []):
        path = workspace / str(source.get("path", ""))
        if not path.is_file():
            raise FileNotFoundError(f"official submission source not found: {path}")
        source["content"] = path.read_text(encoding="utf-8")
    if not solution.get("sources"):
        raise ValueError("official submission has no sources")
    return solution


def multipart_form(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    content: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    boundary = f"----atrex-sol58-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            content,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def http_json(
    method: str,
    path: str,
    token: str,
    *,
    base_url: str,
    maximum_timeout: float,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    request_timeout: float | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url}{path}", data=body, method=method)
    request.add_header("Accept", "application/json")
    request.add_header("Authorization", f"Bearer {token}")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    timeout = (
        maximum_timeout
        if request_timeout is None
        else max(0.1, min(maximum_timeout, request_timeout))
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
            return json.loads(text) if text.strip() else {}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            payload = {"detail": tail(text, 1000)}
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        raise RuntimeError(
            f"official API {method} {path} failed HTTP {exc.code}: {detail}"
        ) from exc


def parse_timestamp(value: Any) -> float | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (OverflowError, TypeError, ValueError):
        return None


def format_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def has_official_score(data: dict[str, Any]) -> bool:
    return data.get("sol_score") is not None and data.get("latency_ms") is not None


def official_status(data: dict[str, Any]) -> str:
    return str(data.get("status") or "").upper()


def is_stale_pending_result(
    data: dict[str, Any], *, pending_result_grace: float, now: float
) -> bool:
    if official_status(data) != "PENDING_RESULT" or has_official_score(data):
        return False
    available_at = parse_timestamp(data.get("result_available_at"))
    if available_at is None:
        available_at = parse_timestamp(data.get("finished_at"))
    return available_at is not None and now - available_at >= pending_result_grace


def is_deferred_pending_result(
    data: dict[str, Any], *, deadline: float, poll_interval: float, now: float
) -> bool:
    if official_status(data) != "PENDING_RESULT" or has_official_score(data):
        return False
    available_at = parse_timestamp(data.get("result_available_at"))
    return bool(
        available_at is not None
        and available_at > deadline
        and available_at - now > poll_interval
    )


def normalize_submission_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    return dict(data or {})


def get_official_submission(
    submission_id: Any,
    token: str,
    *,
    config: OfficialApiConfig,
    request: Callable[..., dict[str, Any]],
    request_timeout: float | None = None,
) -> dict[str, Any]:
    deadline = (
        None
        if request_timeout is None
        else time.monotonic() + max(0.1, request_timeout)
    )

    def get(path: str) -> dict[str, Any]:
        if deadline is None:
            return request("GET", path, token)
        remaining = max(0.1, deadline - time.monotonic())
        return request("GET", path, token, request_timeout=remaining)

    primary = normalize_submission_data(get(f"/api/submissions/{submission_id}"))
    available_at = parse_timestamp(primary.get("result_available_at"))
    status = official_status(primary)
    if (
        has_official_score(primary)
        or status in {"PENDING", "QUEUED", "RUNNING", "EVALUATING"}
        or (
            status == "PENDING_RESULT"
            and available_at is not None
            and available_at - time.time() > config.poll_interval
        )
    ):
        return primary

    try:
        params = (
            f"kernel_id={config.kernel_id}&gpu_type={config.gpu_type}&offset=0&limit=50"
        )
        listed = get(f"/api/submissions?{params}")
        rows = (listed.get("data") or {}).get("submissions") or []
        for row in rows:
            if str(row.get("id")) == str(submission_id):
                merged = dict(primary)
                merged.update(
                    {key: value for key, value in row.items() if value is not None}
                )
                return merged
    except Exception as exc:
        primary["list_refresh_error"] = str(exc)
    return primary


def write_official_result(workspace: Path, result: dict[str, Any]) -> None:
    (workspace / "official_submission_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def poll_official_submission(
    submission_id: Any,
    token: str,
    workspace: Path,
    *,
    config: OfficialApiConfig,
    get_submission: Callable[..., dict[str, Any]],
    upload: dict[str, Any] | None,
    poll_timeout: float | None,
    sleep: Callable[[float], None] = time.sleep,
    wall_time: Callable[[], float] = time.time,
) -> dict[str, Any]:
    timeout = config.poll_timeout if poll_timeout is None else poll_timeout
    deadline = wall_time() + max(0.0, timeout)
    last: dict[str, Any] = {
        "id": submission_id,
        "status": "PENDING",
        "upload": upload or {},
        "cache_hit": False,
        "evaluation_stack_version": config.evaluation_stack_version,
    }
    while True:
        now = wall_time()
        try:
            data = get_submission(submission_id, token)
            last = dict(data)
            last.setdefault("id", submission_id)
            last["upload"] = upload or last.get("upload") or {}
            last["cache_hit"] = False
            last["evaluation_stack_version"] = data.get(
                "evaluation_stack_version", config.evaluation_stack_version
            )
        except Exception as exc:
            last["poll_error"] = str(exc)

        if is_deferred_pending_result(
            last,
            deadline=deadline,
            poll_interval=config.poll_interval,
            now=now,
        ):
            last["upstream_status"] = official_status(last)
            last["status"] = "DEFERRED_RESULT"
            last["next_refresh_at"] = last.get("result_available_at")
            last["error_log"] = (
                "official result is not available until "
                f"{last.get('result_available_at')}; deferring final score refresh"
            )
        elif is_stale_pending_result(
            last,
            pending_result_grace=config.pending_result_grace,
            now=now,
        ):
            last["upstream_status"] = official_status(last)
            last["status"] = "STALE_PENDING_RESULT"
            last["next_refresh_at"] = format_timestamp(now + config.async_refresh_delay)
            last["error_log"] = (
                "official result stayed PENDING_RESULT after result availability for "
                f">{config.pending_result_grace:.0f}s"
            )

        write_official_result(workspace, last)
        status = official_status(last)
        if status in config.terminal_statuses or has_official_score(last):
            break
        remaining = deadline - wall_time()
        if remaining <= 0:
            last["upstream_status"] = official_status(last)
            last["status"] = "TIMEOUT"
            last["next_refresh_at"] = format_timestamp(
                wall_time() + config.async_refresh_delay
            )
            last["error_log"] = f"official polling timed out after {timeout:.0f}s"
            write_official_result(workspace, last)
            break
        sleep(min(config.poll_interval, remaining))
    return last
