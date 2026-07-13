import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ale_agentic_adapter", ROOT / ".agentic" / "evaluator_adapter.py"
)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def request():
    return {
        "schema_version": adapter.REQUEST_SCHEMA,
        "benchmark": "ale",
        "tasks": ["demo/hello"],
        "method": "dummy",
        "model": "none",
        "environment_profile": "docker",
        "n_cases": 1,
        "case_indices": None,
    }


def test_materialized_experiment_preserves_native_defaults(tmp_path):
    experiment_path = adapter.materialize_experiment(ROOT, tmp_path, request())
    experiment = adapter.yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    assert set(experiment) == {"name", "agent", "environment", "tasks", "output"}
    assert experiment["tasks"] == [{"path": "demo/hello", "variants": [0]}]
    assert "concurrency" not in experiment
    assert "cleanup_mode" not in experiment
    assert "prompt_suffix" not in experiment
    assert "wall_time_s" not in experiment
    generated_agent = adapter.yaml.safe_load(Path(experiment["agent"]).read_text(encoding="utf-8"))
    assert generated_agent["model"] == "none"


def test_request_rejects_arbitrary_paths_and_case_indices(tmp_path):
    path = tmp_path / "request.json"
    payload = request()
    payload["method"] = "../secret"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(adapter.ContractError):
        adapter.load_request(path)
    payload = request()
    payload["case_indices"] = [1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(adapter.ContractError, match="case_indices"):
        adapter.load_request(path)


def test_native_fixture_is_indexed_without_recomputing_score(tmp_path):
    native_root = tmp_path
    run_dir = native_root / "native-runs" / "dummy" / "none" / "demo__hello" / "v0" / "20260713_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "output").mkdir()
    run = {
        "run_id": "dummy__none__demo__hello__v0__20260713_000000",
        "task": {"path": "tasks/demo/hello", "variant_index": 0},
        "status": "completed",
        "score": 0.75,
        "agent": {"id": "dummy", "model": "none"},
        "termination": {"reason": "completed"},
        "timings": {"duration_s": 1.0},
        "usage": None,
    }
    evaluation = {"eval_status": "completed", "score": 0.75, "eval_duration_s": 0.2, "error": None}
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (run_dir / "eval_result.json").write_text(json.dumps(evaluation), encoding="utf-8")
    (run_dir / "trajectory.json").write_text("{}", encoding="utf-8")
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "output" / "answer.txt").write_text("ok", encoding="utf-8")
    outside = native_root.parent / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    os.symlink(outside, run_dir / "output" / "escaped.txt")

    cells = adapter.cells_from_native_runs(native_root)
    assert len(cells) == 1
    assert cells[0]["case_id"] == "demo/hello#v0"
    assert cells[0]["metrics"] == {"score": 0.75}
    assert cells[0]["status"] == "pass"
    assert all(not item["path"].endswith("escaped.txt") for item in cells[0]["artifacts"])


def test_contract_check_loads_native_experiment():
    assert adapter.contract_check(ROOT) == 0
