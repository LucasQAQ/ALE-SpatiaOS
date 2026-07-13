#!/usr/bin/env python3
"""Thin Data Layer adapter for ALE's native experiment runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


REQUEST_SCHEMA = "agentic-sandbox-evaluator-request-v0"
RESULT_SCHEMA = "agentic-evaluator-result-v1"
SAFE_PROFILE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
SAFE_TASK = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]{0,255}$")
SAFE_MODEL = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/+-]{0,255}$")


class ContractError(ValueError):
    """Raised when a Data Layer request violates the ALE adapter contract."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _safe(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ContractError(f"{label} is not a safe identifier")
    if ".." in value.split("/"):
        raise ContractError(f"{label} must not traverse directories")
    return value


def _tasks(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 256:
        raise ContractError("tasks must contain 1-256 ALE task paths")
    tasks = [_safe(item, SAFE_TASK, f"tasks[{index}]") for index, item in enumerate(value)]
    if len(set(tasks)) != len(tasks):
        raise ContractError("tasks must not contain duplicates")
    return tasks


def load_request(path: Path) -> dict[str, Any]:
    try:
        request = _object(json.loads(path.read_text(encoding="utf-8")), "request")
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"could not read evaluator request: {exc}") from exc
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ContractError(f"schema_version must be {REQUEST_SCHEMA}")
    if request.get("benchmark") != "ale":
        raise ContractError("benchmark must be ale")
    _tasks(request.get("tasks"))
    _safe(request.get("method"), SAFE_PROFILE, "method")
    _safe(request.get("model"), SAFE_MODEL, "model")
    _safe(request.get("environment_profile", "environment"), SAFE_PROFILE, "environment_profile")
    n_cases = request.get("n_cases")
    if not isinstance(n_cases, int) or isinstance(n_cases, bool) or not 1 <= n_cases <= 10_000:
        raise ContractError("n_cases must be an integer from 1 through 10000")
    case_indices = request.get("case_indices")
    if case_indices not in (None, []):
        raise ContractError("ALE uses native variants; case_indices is not supported by this adapter")
    return request


def native_output_root(request_path: Path) -> Path:
    expected = request_path.resolve().parent.parent / "native" / "ale"
    configured = os.environ.get("AGENTIC_EVAL_NATIVE_ROOT")
    root = Path(configured).resolve() if configured else expected
    if root != expected:
        raise ContractError("AGENTIC_EVAL_NATIVE_ROOT must be the current Data Layer run's native/ale directory")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _trusted_profile(repo_root: Path, category: str, profile: str) -> Path:
    path = repo_root / "configs" / category / f"{profile}.yaml"
    expected_root = (repo_root / "configs" / category).resolve()
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(expected_root)
    except (OSError, ValueError) as exc:
        raise ContractError(f"unknown trusted {category} profile: {profile}") from exc
    if not resolved.is_file():
        raise ContractError(f"trusted {category} profile is not a file: {profile}")
    return resolved


def _trusted_task(repo_root: Path, task: str) -> None:
    task_root = (repo_root / "tasks").resolve()
    candidate = task_root / task
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(task_root)
    except (OSError, ValueError) as exc:
        raise ContractError(f"unknown trusted ALE task: {task}") from exc
    if not resolved.is_dir():
        raise ContractError(f"ALE task is not a directory: {task}")


def materialize_experiment(repo_root: Path, native_root: Path, request: dict[str, Any]) -> Path:
    method = str(request["method"])
    model = str(request["model"])
    environment_profile = str(request.get("environment_profile") or "environment")
    source_agent = _trusted_profile(repo_root, "agents", method)
    environment = _trusted_profile(repo_root, "environments", environment_profile)
    tasks = _tasks(request["tasks"])
    for task in tasks:
        _trusted_task(repo_root, task)

    raw_agent = yaml.safe_load(source_agent.read_text(encoding="utf-8"))
    agent = _object(raw_agent, f"agent profile {method}")
    agent["model"] = model
    input_root = native_root / "adapter-inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    generated_agent = input_root / "agent.yaml"
    generated_agent.write_text(yaml.safe_dump(agent, sort_keys=False), encoding="utf-8")

    variants = list(range(int(request["n_cases"])))
    experiment = {
        "name": "data_layer_ale_eval",
        "agent": str(generated_agent),
        "environment": str(environment),
        "tasks": [{"path": task, "variants": variants} for task in tasks],
        "output": {"root": str(native_root / "native-runs")},
    }
    experiment_path = input_root / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
    return experiment_path


def build_native_command(repo_root: Path, experiment_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ale_run",
        "run",
        str(experiment_path),
    ]


def _git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", commit) else None


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _artifacts(native_root: Path, run_dir: Path) -> list[dict[str, str]]:
    core_roles = {
        "events.jsonl": "native_events",
        "run.json": "native_run_metadata",
        "trajectory.json": "native_trajectory",
        "eval_result.json": "native_eval_result",
    }
    artifacts: list[dict[str, str]] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink() or not path.is_file() or not _inside(native_root, path):
            continue
        if path.name in core_roles:
            role = core_roles[path.name]
        elif "output" in path.relative_to(run_dir).parts:
            role = "generated_output"
        elif "origin_log" in path.relative_to(run_dir).parts:
            role = "native_agent_log"
        else:
            continue
        artifacts.append({"role": role, "path": path.resolve().relative_to(native_root.resolve()).as_posix()})
        if len(artifacts) >= 512:
            break
    return artifacts


def _status(run_meta: dict[str, Any], eval_result: dict[str, Any]) -> str:
    status = str(run_meta.get("status") or "")
    if status == "completed":
        return "pass"
    if status in {"failed", "timeout"}:
        return "fail"
    if status in {"cancelled", "not_executed"}:
        return "infra_error"
    if str(eval_result.get("eval_status") or "") in {"unsupported", "not_supported"}:
        return "unsupported"
    return "invalid"


def cells_from_native_runs(native_root: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    runs_root = native_root / "native-runs"
    if not runs_root.is_dir():
        return cells
    for run_path in sorted(runs_root.rglob("run.json")):
        if run_path.is_symlink() or not _inside(native_root, run_path):
            continue
        run_dir = run_path.parent
        eval_path = run_dir / "eval_result.json"
        try:
            run_meta = _object(json.loads(run_path.read_text(encoding="utf-8")), "ALE run.json")
            eval_result = _object(json.loads(eval_path.read_text(encoding="utf-8")), "ALE eval_result.json")
        except (OSError, json.JSONDecodeError, ContractError):
            continue
        task = _object(run_meta.get("task"), "ALE run.json task")
        task_path = str(task.get("path") or "").removeprefix("tasks/")
        variant = task.get("variant_index")
        if not task_path or not isinstance(variant, int):
            continue
        score = eval_result.get("score")
        metrics = {"score": score} if isinstance(score, (int, float)) and not isinstance(score, bool) else {}
        cells.append({
            "case_id": f"{task_path}#v{variant}",
            "task": task_path,
            "attempt_id": "attempt_001",
            "status": _status(run_meta, eval_result),
            "metrics": metrics,
            "metadata": {
                "native_run_id": run_meta.get("run_id"),
                "native_status": run_meta.get("status"),
                "eval_status": eval_result.get("eval_status"),
                "termination": run_meta.get("termination"),
                "timings": run_meta.get("timings"),
                "usage": run_meta.get("usage"),
                "agent": run_meta.get("agent"),
                "native_run_dir": run_dir.relative_to(native_root).as_posix(),
            },
            "artifacts": _artifacts(native_root, run_dir),
        })
    return cells


def write_manifest(
    native_root: Path,
    repo_root: Path,
    request: dict[str, Any],
    native_return_code: int,
) -> dict[str, Any]:
    cells = cells_from_native_runs(native_root)
    expected = len(request["tasks"]) * int(request["n_cases"])
    if not cells:
        status = "failed"
    elif len(cells) != expected or native_return_code != 0:
        status = "partial"
    else:
        status = "complete"
    manifest = {
        "schema_version": RESULT_SCHEMA,
        "benchmark": "ale",
        "evaluator": {
            "name": "ale",
            "version": _git_commit(repo_root),
            "native_runner": "uv run python -m ale_run run",
            "adapter": ".agentic/evaluator_adapter.py",
            "adapter_contract": "ale-native-evaluator-adapter-v1",
            "environment_profile": request.get("environment_profile") or "environment",
            "native_return_code": native_return_code,
            "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest(),
        },
        "status": status,
        "cells": cells,
    }
    temporary = native_root / f"results.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(native_root / "results.json")
    return manifest


def run(request_path: Path, repo_root: Path) -> int:
    request = load_request(request_path)
    native_root = native_output_root(request_path)
    experiment_path = materialize_experiment(repo_root, native_root, request)
    command = build_native_command(repo_root, experiment_path)
    logs = native_root / "adapter-logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (
        (logs / "stdout.txt").open("w", encoding="utf-8") as stdout,
        (logs / "stderr.txt").open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(command, cwd=repo_root, stdout=stdout, stderr=stderr, text=True, check=False)
    manifest = write_manifest(native_root, repo_root, request, completed.returncode)
    return 0 if manifest["status"] != "failed" else 1


def contract_check(repo_root: Path) -> int:
    required = [repo_root / "ale_run" / "__main__.py", repo_root / "ale_run" / "orchestration" / "run_writer.py"]
    missing = [str(path.relative_to(repo_root)) for path in required if not path.is_file()]
    if missing:
        print(json.dumps({"ok": False, "missing": missing}))
        return 1
    with tempfile.TemporaryDirectory(prefix="ale-agentic-contract-") as directory:
        native_root = Path(directory)
        request = {
            "tasks": ["demo/hello"],
            "method": "dummy",
            "model": "none",
            "n_cases": 1,
            "environment_profile": "docker",
        }
        experiment = materialize_experiment(repo_root, native_root, request)
        from ale_run.orchestration.config_loader import load_experiment

        spec = load_experiment(experiment)
        native = subprocess.run(
            [*build_native_command(repo_root, experiment), "--dry-run"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        ok = (
            len(spec.agents) == 1
            and len(spec.tasks) == 1
            and spec.tasks[0].path == "demo/hello"
            and native.returncode == 0
            and "demo/hello" in native.stdout
        )
    print(json.dumps({
        "ok": ok,
        "native_runner": "python -m ale_run run",
        "dry_contract": "demo/hello",
        "native_return_code": native.returncode,
    }))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ALE Data Layer evaluator adapter")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--contract-check", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        if args.contract_check:
            return contract_check(repo_root)
        if args.request is None:
            parser.error("--request is required unless --contract-check is used")
        return run(args.request.resolve(), repo_root)
    except (ContractError, OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"adapter error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
