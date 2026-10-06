#!/usr/bin/env python3
"""``validation_context`` reaches the host (plan §2.4, task 2.5's surfacing half).

A detached run's caller never reads the child's ``MULTI_LLM_RESULT`` or
stderr, so the published object has to cross three steps: ``.status.json``
(the single writer, ``validation.publish_validation_context``), the view
``detach.classify_run`` builds, and ``run_status.main``'s re-record. These
tests hand-write the terminal ``.status.json`` the writer leaves behind and
assert the terminal ``status`` payload **and** its rendered text, plus the
``report.md`` header line and the pure helpers in ``utils/validation_context``.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro  # noqa: E402
from utils import cli_common, detach, run_status, status_file  # noqa: E402
from utils import validation_context as vc  # noqa: E402
from utils.review_orchestrator_base import aggregate_results  # noqa: E402

TASKS_SENTENCE = "Validation saw 8 of 20 sections of the tasks file"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "a plan.md"
    plan.write_text("# Plan\n\n## Tasks\n- [ ] 1. Do the thing\n", encoding="utf-8")
    return plan


def _phase_dir(plan: Path, phase: str = "review-plan") -> Path:
    directory = plan.parent / plan.stem.replace(" ", "_") / phase
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _plan_path_source(snapshot: str = "/abs/review-plan/validation-snapshot.md") -> dict:
    return {
        "label": "plan", "mode": "path", "chars": 60000, "bytes": 108225,
        "path": snapshot, "sha256": "a" * 64, "dropped_sections": ["## Appendix"],
    }


def _tasks_truncated_source() -> dict:
    return {
        "label": "tasks", "mode": "truncated", "chars": 60000, "bytes": 140000,
        "path": None, "sha256": "b" * 64, "dropped_sections": ["Task 9"],
        "sections_total": 20, "sections_shown": 8,
        "reason": "snapshot unwritable",
    }


def _obj(phase: str, sources: list, insufficient: int = 0) -> dict:
    return {
        "phase": phase, "limit": 60000,
        "mode": vc.worst_mode(s["mode"] for s in sources),
        "insufficient": insufficient, "sources": sources,
    }


def _finished_run(phase_dir: Path, phase: str, run_id: str, obj) -> None:
    body = {"phase": phase, "state": "complete"}
    if obj is not None:
        body["validation_context"] = obj
    status_file.write_status(str(phase_dir), body)
    status_file.claim_run(str(phase_dir), {
        "run_id": run_id, "state": "done", "exit_code": 0, "detached": True,
    })


def _poll(plan: Path, capsys, *extra):
    """A terminal ``status`` poll: (exit, result payload, rendered text)."""
    ctx = cli_common.activate_result(cli_common.ResultContext(mode="status"))
    try:
        code = run_status.main(["--plan-file", str(plan), *extra])
        payload = ctx.payload(code)
    finally:
        cli_common.deactivate_result()
    return code, payload, capsys.readouterr().out


def _seed_unapplied_review_tasks(plan: Path, review_tasks: Path) -> None:
    """A finished review-tasks with one unapplied group → a workflow trailer."""
    from utils.state_manager import StateManager

    grouped = {
        "format_version": 2,
        "groups": [{
            "category": "addition", "theme": "g1",
            "suggestions": [{"title": "s1", "desc": "d", "importance": "LOW",
                              "type": "addition"}],
        }],
    }
    (review_tasks / "grouped.json").write_text(json.dumps(grouped), encoding="utf-8")
    validation = {"format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]}
    (review_tasks / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
    state = StateManager(plan)
    state.mark_phase_completed("review-tasks")
    state.save()


# ---------------------------------------------------------------------------
# The detach view carries the key
# ---------------------------------------------------------------------------

class TestTheViewCarriesTheKey:
    def test_classify_run_copies_validation_context(self, tmp_path):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        obj = _obj("review-plan", [_plan_path_source()])
        _finished_run(phase, "review-plan", "run-vc-view", obj)
        view = detach.classify_run(str(phase))
        assert view["validation_context"] == obj

    def test_classify_run_reads_none_when_absent(self, tmp_path):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _finished_run(phase, "review-plan", "run-vc-none", None)
        assert detach.classify_run(str(phase))["validation_context"] is None


# ---------------------------------------------------------------------------
# The terminal poll: payload + text
# ---------------------------------------------------------------------------

class TestTheTerminalPoll:
    def test_a_truncated_source_sets_the_object_and_the_notice(self, tmp_path, capsys):
        """(a)"""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-tasks")
        obj = _obj("review-tasks", [_tasks_truncated_source()])
        _finished_run(phase, "review-tasks", "run-vc-a", obj)

        code, payload, out = _poll(plan, capsys, "--view", "run")

        assert code == 0
        assert payload["validation_context"] == obj
        # `--view run` on a fresh plan also carries the workflow trailer
        # ("suggested next: ..."); the truncation sentence leads it.
        assert payload["notice"].split("; ")[0] == TASKS_SENTENCE
        assert "validation context (review-tasks): tasks: TRUNCATED, 8 of 20 sections" in out

    def test_the_truncation_notice_comes_first_then_the_workflow_trailer(
        self, tmp_path, capsys
    ):
        """(b) one notice, truncation sentence first, "; ", trailer."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-tasks")
        _seed_unapplied_review_tasks(plan, phase)
        obj = _obj("review-tasks", [_tasks_truncated_source()])
        _finished_run(phase, "review-tasks", "run-vc-b", obj)

        code, payload, out = _poll(
            plan, capsys, "--view", "run", "--phase", "review-tasks"
        )

        assert code == 0
        notice = payload["notice"]
        assert notice.startswith(TASKS_SENTENCE + "; ")
        trailer = notice[len(TASKS_SENTENCE) + 2:]
        assert "review-tasks left 1 valid task suggestion group unapplied" in trailer
        assert "suggested next:" in trailer
        assert "workflow notice: review-tasks left 1 valid" in out
        assert payload["next"] is None

    def test_the_workflow_trailer_alone_is_unchanged(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-tasks")
        _seed_unapplied_review_tasks(plan, phase)
        _finished_run(phase, "review-tasks", "run-vc-b2", None)

        _, payload, _ = _poll(plan, capsys, "--view", "run", "--phase", "review-tasks")

        assert "Validation saw" not in payload["notice"]
        assert payload["notice"].startswith("notice: ") or "suggested next:" in payload["notice"]
        assert "validation_context" not in payload

    def test_the_truncation_notice_is_set_off_view_run_too(self, tmp_path, capsys):
        """Any terminal poll, not only `--view run`."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-tasks")
        obj = _obj("review-tasks", [_tasks_truncated_source()])
        _finished_run(phase, "review-tasks", "run-vc-all", obj)

        _, payload, out = _poll(plan, capsys, "--view", "all")

        # `--view all` carries no workflow trailer: the sentence alone, unchanged.
        assert payload["notice"] == TASKS_SENTENCE
        assert payload["validation_context"] == obj
        assert "validation context (review-tasks):" in out

    def test_a_path_source_is_excerpt_plus_path_and_sets_no_notice(
        self, tmp_path, capsys
    ):
        """(c)"""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        snapshot = str(phase / "validation-snapshot.md")
        obj = _obj("review-plan", [_plan_path_source(snapshot)])
        _finished_run(phase, "review-plan", "run-vc-c", obj)

        _, payload, out = _poll(plan, capsys, "--view", "run")

        assert payload["validation_context"] == obj
        assert "Validation saw" not in (payload.get("notice") or "")
        assert (
            f"plan: 60000-char excerpt inline + full snapshot by path ({snapshot})"
            in out
        )
        assert "inline in full" not in out

    def test_no_object_means_no_key_and_no_notice(self, tmp_path, capsys):
        """(d) a rerun whose context fits inline publishes nothing."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _finished_run(phase, "review-plan", "run-vc-d", None)
        status_file.update_status(str(phase), {"validation_context": None})

        _, payload, out = _poll(plan, capsys, "--view", "run")

        assert "validation_context" not in payload
        assert "Validation saw" not in (payload.get("notice") or "")
        assert "validation context" not in out

    def test_review_tasks_plan_by_path_and_tasks_truncated(self, tmp_path, capsys):
        """(e) two source entries, top-level truncated, text and notice agree."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-tasks")
        obj = vc.build_validation_context(
            phase="review-tasks", limit=60000,
            sources=[_tasks_truncated_source(), _plan_path_source()],
        )
        assert [s["label"] for s in obj["sources"]] == ["plan", "tasks"]
        assert obj["mode"] == "truncated"
        _finished_run(phase, "review-tasks", "run-vc-e", obj)

        _, payload, out = _poll(plan, capsys, "--view", "run")

        assert payload["validation_context"] == obj
        assert len(payload["validation_context"]["sources"]) == 2
        assert payload["notice"].split("; ")[0] == TASKS_SENTENCE
        line = next(l.strip() for l in out.splitlines() if "validation context (" in l)
        assert line == vc.validation_context_text(obj)
        assert "plan: 60000-char excerpt inline + full snapshot by path" in line
        assert "tasks: TRUNCATED, 8 of 20 sections" in line

    def test_several_views_record_the_worst_and_render_every_phase(
        self, tmp_path, capsys
    ):
        plan = _plan(tmp_path)
        review_plan = _phase_dir(plan, "review-plan")
        review_tasks = _phase_dir(plan, "review-tasks")
        path_obj = _obj("review-plan", [_plan_path_source()])
        trunc_obj = _obj("review-tasks", [_tasks_truncated_source()])
        _finished_run(review_plan, "review-plan", "run-vc-p", path_obj)
        _finished_run(review_tasks, "review-tasks", "run-vc-t", trunc_obj)

        _, payload, out = _poll(plan, capsys, "--view", "run")

        assert payload["validation_context"] == trunc_obj
        assert payload["notice"].split("; ")[0] == TASKS_SENTENCE
        assert "validation context (review-plan):" in out
        assert "validation context (review-tasks):" in out

    def test_a_running_poll_carries_no_truncation_notice(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-tasks")
        status_file.write_status(str(phase), {
            "phase": "review-tasks", "state": "models_running",
            "validation_context": _obj("review-tasks", [_tasks_truncated_source()]),
        })
        import os
        from utils import process_control
        status_file.claim_run(str(phase), {
            "run_id": "run-vc-live", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
            "state": "running", "detached": True,
        })

        code, payload, _ = _poll(plan, capsys, "--view", "run")

        assert code == 2
        assert "notice" not in payload


# ---------------------------------------------------------------------------
# report.md header
# ---------------------------------------------------------------------------

def _aggregate(tmp_path: Path, obj) -> str:
    phase_dir = tmp_path / "test-plan" / "review-tasks"
    phase_dir.mkdir(parents=True, exist_ok=True)
    groups = [{
        "theme": "G1", "category": "addition", "models": ["m1"], "priority_score": 5,
        "suggestions": [{"title": "S1", "desc": "D1", "importance": "HIGH",
                          "type": "addition", "reference": "R1", "source_model": "m1"}],
    }]
    (phase_dir / "grouped.json").write_text(json.dumps(groups), encoding="utf-8")
    status = {"phase": "review-tasks", "state": "complete"}
    if obj is not None:
        status["validation_context"] = obj
    status_file.write_status(str(phase_dir), status)
    report_path = aggregate_results(
        prefix="test-plan", out_dir=str(tmp_path / "test-plan"),
        phase_dir=str(phase_dir), phase_name="review-tasks",
        models=["m1"], failed_models={},
    )
    return Path(report_path).read_text(encoding="utf-8")


def _strip_generated(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.startswith("**Generated:**"))


class TestReportHeader:
    def test_review_report_header_names_both_sources_and_the_truncation(self, tmp_path):
        obj = vc.build_validation_context(
            phase="review-tasks", limit=60000,
            sources=[_plan_path_source(), _tasks_truncated_source()],
        )
        report = _aggregate(tmp_path, obj)
        head = report.split("> **Note:**")[0]
        assert "**Validation context:** plan: 60000-char excerpt inline + full snapshot by path" in head
        assert "tasks: TRUNCATED, 8 of 20 sections" in head
        assert f"**Validation context warning:** {TASKS_SENTENCE}" in head
        assert "validation context (review-tasks):" not in head

    def test_a_path_only_object_has_no_warning_line(self, tmp_path):
        report = _aggregate(tmp_path, _obj("review-tasks", [_plan_path_source()]))
        assert "**Validation context:** plan: 60000-char excerpt" in report
        assert "**Validation context warning:**" not in report

    def test_no_object_leaves_the_report_unchanged(self, tmp_path):
        without = _aggregate(tmp_path / "a", None)
        assert "Validation context" not in without
        # A checkpoint that never had the key and one whose key was removed
        # read the same.
        again = _aggregate(tmp_path / "b", None)
        assert _strip_generated(without) == _strip_generated(again)

    def test_code_review_report_header(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()
        results = {"m1": (True, [], None)}

        baseline = cro.generate_review_report(
            plan_path, results, changed_files=["a.py"], phase_dir=None,
        )
        status_file.write_status(str(phase_dir), {"phase": "code-review"})
        absent = cro.generate_review_report(
            plan_path, results, changed_files=["a.py"], phase_dir=phase_dir,
        )
        assert _strip_generated(absent) == _strip_generated(baseline)

        obj = _obj("code-review", [{
            "label": "reviewed-code", "mode": "truncated", "chars": 60000,
            "bytes": 90000, "path": None, "sha256": None, "dropped_sections": [],
            "dropped_files": ["b.py"],
        }])
        status_file.update_status(str(phase_dir), {"validation_context": obj})
        present = cro.generate_review_report(
            plan_path, results, changed_files=["a.py"], phase_dir=phase_dir,
        )
        assert "**Validation context:** reviewed-code: TRUNCATED" in present
        assert (
            "**Validation context warning:** Validation saw 60000 of 90000 chars "
            "of the reviewed code (dropped: b.py)" in present
        )


# ---------------------------------------------------------------------------
# utils/validation_context unit tests
# ---------------------------------------------------------------------------

class TestValidationContextHelpers:
    def test_all_inline_publishes_nothing(self):
        assert vc.build_validation_context(
            phase="review-plan", limit=60000,
            sources=[{"label": "plan", "mode": "inline", "chars": 10}],
        ) is None
        assert vc.build_validation_context(phase="x", limit=1, sources=[]) is None

    def test_aggregate_takes_the_worst_mode_and_smallest_chars(self):
        entries = vc.aggregate_sources([
            {"label": "plan", "mode": "path", "chars_shown": 60000,
             "abs_path": "/snap.md"},
            {"label": "plan", "mode": "truncated", "chars_shown": 59000,
             "reason": "runner cannot read files"},
            {"label": "plan", "mode": "inline", "chars_shown": 50000},
        ])
        assert len(entries) == 1
        assert entries[0]["mode"] == "truncated"
        assert entries[0]["chars"] == 50000
        assert entries[0]["reason"] == "runner cannot read files"

    def test_sources_are_reported_in_label_order(self):
        obj = vc.build_validation_context(
            phase="review-code", limit=60000, insufficient=2,
            sources=[
                {"label": "diff", "mode": "path", "chars": 1},
                {"label": "reviewed-code", "mode": "inline", "chars": 1},
                {"label": "plan", "mode": "path", "chars": 1},
            ],
        )
        assert [s["label"] for s in obj["sources"]] == ["plan", "reviewed-code", "diff"]
        assert obj["mode"] == "path"
        assert obj["insufficient"] == 2

    def test_notice_only_for_truncated(self):
        assert vc.validation_context_notice(_obj("p", [_plan_path_source()])) is None
        assert vc.validation_context_notice(None) is None
        assert vc.validation_context_notice(
            _obj("p", [_tasks_truncated_source()])
        ) == TASKS_SENTENCE

    def test_text_renders_path_as_excerpt_plus_path_and_insufficient(self):
        text = vc.validation_context_text(
            _obj("review-plan", [_plan_path_source("/s.md")], insufficient=1)
        )
        assert text.startswith("validation context (review-plan): ")
        assert "plan: 60000-char excerpt inline + full snapshot by path (/s.md)" in text
        assert "1 verdict(s) answered insufficient-context" in text
        assert vc.validation_context_text(None) is None

    def test_pick_worst_prefers_truncated_then_first_on_tie(self):
        a = _obj("a", [_plan_path_source()])
        b = _obj("b", [_plan_path_source()])
        c = _obj("c", [_tasks_truncated_source()])
        assert vc.pick_worst([None, a, b]) is a
        assert vc.pick_worst([a, c, b]) is c
        assert vc.pick_worst([None]) is None

    def test_report_header_lines(self):
        assert vc.report_header_lines(None) == []
        lines = vc.report_header_lines(_obj("review-tasks", [_tasks_truncated_source()]))
        assert lines[0].startswith("**Validation context:** tasks: TRUNCATED")
        assert lines[1] == f"**Validation context warning:** {TASKS_SENTENCE}"
