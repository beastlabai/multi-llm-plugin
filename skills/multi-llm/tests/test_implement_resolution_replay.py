"""The implement worker retains decisions made before its run was claimed."""

import json

import implement_orchestrator
from tests.test_write_executor_orchestrators import repo, _seed_generate_tasks  # noqa: F401
from utils import detach, status_file, unattended
from utils.output_handler import get_phase_dir, get_output_dir


def test_task_suggestion_auto_skip_survives_launcher_reentry(repo, monkeypatch):
    _seed_generate_tasks(repo)
    review = get_output_dir(repo.plan) / "review-tasks"
    review.mkdir(parents=True)
    (review / "grouped.json").write_text(json.dumps([
        {"suggestions": [{"title": "Add coverage", "importance": "HIGH"}]}
    ]))
    monkeypatch.setattr(detach, "_ACTIVE", None)
    monkeypatch.setattr(implement_orchestrator, "run_implement_write_phase", lambda *a, **kw: 0)
    monkeypatch.setattr(implement_orchestrator, "maybe_detach_implement_phase", lambda *a, **kw: 0)
    argv = ["--plan-file", str(repo.plan), "--yes", "--exec-subagents"]
    assert implement_orchestrator.main(argv) == 0

    phase = str(get_phase_dir(repo.plan, "implement"))

    def claim_worker(*args, **kwargs):
        handle = detach.RunHandle(phase, run_id="reentry", mode="implement",
                                  plan_file=str(repo.plan), detached=True)
        status_file.claim_run(phase, {"run_id": "reentry", "state": "running"})
        monkeypatch.setattr(detach, "_ACTIVE", handle)
        return None

    monkeypatch.setattr(implement_orchestrator, "maybe_detach_implement_phase", claim_worker)
    assert implement_orchestrator.main(argv) == 0
    resolutions = status_file.read_run(phase).get("resolutions", [])
    assert [entry["site"] for entry in resolutions] == [unattended.row(7).site_id]
    assert "skipped" in resolutions[0]["decision"]
