#!/usr/bin/env python3
"""``status`` enumerates phase dirs by allow-list, not by "holds a .status.json"
(plan §7.4, tasks 7.4a-7.4d).

A backup folder that copied a ``.status.json`` showed up as a second, duplicate
run. ``output_handler.PHASE_DIR_NAMES`` is now the single source of truth for
which children are phase dirs; the rest are reported as ignored, never silently
dropped, and two views carrying one run id are one run.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import modes as mode_table  # noqa: E402
from utils import cli_common, run_status, status_file  # noqa: E402
from utils.output_handler import (  # noqa: E402
    PHASE_DIR_NAMES,
    PHASE_DIRECTORIES,
    slugify_question,
)

BACKUP = "validation-truncated-backup-20260924"


def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "a plan.md"
    plan.write_text("# Plan\n\n## Tasks\n- [ ] 1. Do the thing\n", encoding="utf-8")
    return plan


def _plan_dir(plan: Path) -> Path:
    return plan.parent / plan.stem.replace(" ", "_")


def _done_run(directory: Path, run_id: str, phase: str, **extra) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    status_file.write_status(str(directory), {
        "phase": phase, "state": "complete", **extra,
        "run": {
            "run_id": run_id, "state": "done", "exit_code": 0, "error": None,
            "detached": True, "pid": 1, "mode": phase,
            "heartbeat_at": status_file.now_iso(),
            "finished_at": status_file.now_iso(),
        },
    })


def _status(plan, *extra, capsys):
    cli_common.activate_result(cli_common.ResultContext(mode="status"))
    try:
        code = run_status.main(
            ["--plan-file", str(plan), "--view", "run", "--all", *extra]
        )
    finally:
        cli_common.deactivate_result()
    return code, capsys.readouterr().out


class TestTheAllowListIsCompleteAndDerived:
    def test_every_launchable_phase_has_its_directory_in_the_allow_list(self):
        """7.4d: a new phase cannot become invisible to ``status``."""
        phases = mode_table.phase_names()
        for required in ("full", "generate-tasks", "ask"):
            assert required in phases
        for phase in phases:
            directory = PHASE_DIRECTORIES.get(phase, phase)
            assert directory in PHASE_DIR_NAMES, (phase, directory)

    def test_the_directory_word_not_the_mode_word(self):
        assert "apply-fixes" in PHASE_DIR_NAMES
        assert "apply-code-fixes" not in PHASE_DIR_NAMES
        # The legacy key no mode owns.
        assert "tasks" in PHASE_DIR_NAMES
        assert len(set(PHASE_DIR_NAMES)) == len(PHASE_DIR_NAMES)

    def test_the_phase_selector_reads_the_same_tuple(self, capsys):
        for word in PHASE_DIR_NAMES:
            assert run_status._resolve_phase(word)


class TestABackupFolderIsNotARun:
    def _layout(self, tmp_path):
        plan = _plan(tmp_path)
        root = _plan_dir(plan)
        _done_run(root / "review-plan", "run-rp-00000001", "review-plan")
        _done_run(root / "generate-tasks", "run-gt-00000001", "generate-tasks")
        _done_run(root / "full", "run-full-0000001", "full")
        _done_run(root / "apply-fixes", "run-af-00000001", "apply-code-fixes")
        question = "Is the rollback strategy sufficient?"
        _done_run(
            root / "ask" / slugify_question(question), "run-ask-0000001", "ask",
            question=question,
        )
        # The backup copied review-plan's checkpoint verbatim.
        backup = root / BACKUP
        backup.mkdir(parents=True)
        (backup / status_file.STATUS_FILENAME).write_text(
            (root / "review-plan" / status_file.STATUS_FILENAME).read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        # A child with no checkpoint is not reported at all.
        (root / "scratch-notes").mkdir()
        return plan, root

    def test_five_views_and_one_ignored_line(self, tmp_path, capsys):
        """7.4c."""
        plan, root = self._layout(tmp_path)
        found = run_status.discover_phase_dirs(plan)
        names = sorted(
            f"ask/{p.name}" if p.parent.name == "ask" else p.name for p in found
        )
        assert len(names) == 5
        assert BACKUP not in names
        assert {"review-plan", "generate-tasks", "full", "apply-fixes"} <= set(names)

        code, out = _status(plan, capsys=capsys)
        assert code == 0
        assert out.count(f"ignored (not a phase dir): {BACKUP}") == 1
        assert "scratch-notes" not in out
        # The ignored line comes after the phase views.
        assert out.index("ignored (not a phase dir)") > out.index("review-plan")

    def test_json_carries_ignored_dirs(self, tmp_path, capsys):
        plan, _root = self._layout(tmp_path)
        code, out = _status(plan, "--json", capsys=capsys)
        body = json.loads(out)
        assert code == 0
        assert len(body["runs"]) == 5
        assert body["ignored_dirs"] == [BACKUP]
        assert not any(BACKUP in run["phase_dir"] for run in body["runs"])

    def test_a_plan_with_nothing_skipped_keeps_its_payload(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _done_run(_plan_dir(plan) / "review-plan", "run-rp-00000002", "review-plan")
        code, out = _status(plan, "--json", capsys=capsys)
        body = json.loads(out)
        assert code == 0
        assert set(body) == {"runs"}
        code, out = _status(plan, capsys=capsys)
        assert "ignored" not in out


class TestOneRunIdIsOneRun:
    def test_a_copied_run_record_is_one_run_named_as_a_copy(self, tmp_path, capsys):
        """7.4b: two allow-listed dirs carrying one run id."""
        plan = _plan(tmp_path)
        root = _plan_dir(plan)
        _done_run(root / "review-plan", "run-dup-0000001", "review-plan")
        # A legacy-named dir holding a copy of review-plan's record.
        (root / "tasks").mkdir()
        (root / "tasks" / status_file.STATUS_FILENAME).write_text(
            (root / "review-plan" / status_file.STATUS_FILENAME).read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        code, out = _status(plan, "--json", capsys=capsys)
        body = json.loads(out)
        assert code == 0
        assert [Path(r["phase_dir"]).name for r in body["runs"]] == ["review-plan"]
        assert body["copied_runs"] == [{
            "dir": "tasks", "run_id": "run-dup-0000001", "copy_of": "review-plan",
        }]
        code, out = _status(plan, capsys=capsys)
        assert "ignored (copy of run run-dup- in review-plan): tasks" in out

    def test_views_without_a_run_id_are_never_merged(self):
        views = [
            {"phase_dir": "/p/review-plan", "phase": "review-plan", "run": {}},
            {"phase_dir": "/p/implement", "phase": "implement", "run": {}},
        ]
        kept, copies = run_status._dedupe_run_ids(views)
        assert kept == views and copies == []
