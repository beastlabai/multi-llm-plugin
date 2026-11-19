"""Each write leg hands the suppression scan the batch's own changes.

The scan (``utils.suppression_scan``) is pure; ``utils.batch_suppressions``
reads its inputs from each leg's record of one batch, never from a
working-tree diff: on the executor leg the batch's pre-batch snapshot (the
declared files and the undeclared writes the sweep adopted), on the native
leg the scan base ``util write-prompt`` records as it binds a code batch. So
the user's own uncommitted edits, and an earlier batch's, are never listed as
this batch's. The hits reach the batch report row, ``summary.md`` (both the
apply report and the implementation summary), the result notice and, on the
native leg, ``util batch-done``'s stdout. Report only: nothing here changes a
batch's outcome or the exit code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.support import scoped_checks_capture as capture  # noqa: E402
from tests.support.main_runs import _run  # noqa: E402
from tests.support.write_prompt_world import _materialize  # noqa: E402
from tests.test_apply_report import _report, _Result  # noqa: E402
from tests.test_write_executor import FakeState, env  # noqa: E402,F401 - the env fixture
from utils import batch_suppressions as bs  # noqa: E402
from utils import check_records, status_text, write_prompt  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.subagent_tasks import (  # noqa: E402
    PRIOR_CHANGES_TOKEN,
    build_inputs,
    make_write_task,
    write_output_name,
)
from utils.suppression_scan import CHECK_THESE_HEADING  # noqa: E402
from utils.task_outcome import TaskOutcome  # noqa: E402
from utils.write_safety import TaskSnapshot  # noqa: E402


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repo(tmp_path, files):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _where(hits):
    return sorted((hit["path"], hit["line"], hit["marker"]) for hit in hits)


# ---------------------------------------------------------------------------
# The executor leg: the pre-batch snapshot
# ---------------------------------------------------------------------------

class TestExecutorScan:

    def _snapshot(self, tmp_path, root, files):
        return TaskSnapshot.take(
            phase_dir=str(tmp_path / "phase"), task_id="apply-fix-batch-0",
            files=files, root=str(root),
        )

    def test_the_snapshot_is_the_before_side(self, tmp_path):
        root = _repo(tmp_path, {"src/a.py": "import os  # noqa: F401\nx = 1\n"})
        # The user's own uncommitted edit, before the batch: never listed.
        _write(root, "src/a.py", "import os  # noqa: F401\nimport re  # noqa: F401\nx = 1\n")
        snapshot = self._snapshot(tmp_path, root, ["src/a.py"])
        _write(root, "src/a.py",
               "import os  # noqa: F401\nimport re  # noqa: F401\nx = 1  # type: ignore\n")
        hits, unscanned = bs.executor_scan(snapshot.directory, ["src/a.py"])
        assert _where(hits) == [("src/a.py", 3, "# type: ignore")]
        assert unscanned == []

    def test_an_adopted_undeclared_file_is_scanned_and_a_created_one_is_all_new(self, tmp_path):
        root = _repo(tmp_path, {"src/a.py": "a = 1\n"})
        snapshot = self._snapshot(tmp_path, root, ["src/a.py"])
        snapshot.adopt("tests/test_a.py", existed=False)
        _write(root, "tests/test_a.py",
               "import pytest\n\n@pytest.mark.skip\ndef test_a():\n    pass\n")
        hits, _ = bs.executor_scan(snapshot.directory, ["src/a.py", "tests/test_a.py"])
        assert [(hit["path"], hit["kind"]) for hit in hits] == [
            ("tests/test_a.py", "skip-marker:pytest.mark.skip")]

    def test_a_changed_payload_is_not_trusted(self, tmp_path):
        root = _repo(tmp_path, {"src/a.py": "a = 1\n"})
        snapshot = self._snapshot(tmp_path, root, ["src/a.py"])
        payload = next((Path(snapshot.directory) / "files").iterdir())
        payload.write_text("a = 1  # noqa\n", encoding="utf-8")
        _write(root, "src/a.py", "a = 2  # noqa\n")
        hits, unscanned = bs.executor_scan(snapshot.directory, ["src/a.py"])
        assert hits == []
        assert [entry["path"] for entry in unscanned] == ["src/a.py"]

    def test_files_with_no_rollback_point_are_not_scanned(self, tmp_path):
        root = _repo(tmp_path, {"src/a.py": "a = 1\n"})
        snapshot = self._snapshot(tmp_path, root, ["src/a.py"])
        hits, unscanned = bs.executor_scan(
            snapshot.directory, ["src/a.py", "src/b.py", "notes.md"], ["big/gen.py"])
        assert hits == []
        assert [entry["path"] for entry in unscanned] == ["src/b.py", "big/gen.py"]

    def test_odd_config_values_cost_no_other_file_its_hits(self, tmp_path):
        """A TOML date and a ``select`` entry that is a table once made the
        whole batch's scan raise; each is now read like any value."""
        root = _repo(tmp_path, {
            "src/a.py": "a = 1\n",
            "pyproject.toml": "[tool.ruff.lint]\nselect = [{x = 1}, 'E']\n",
        })
        snapshot = self._snapshot(tmp_path, root, ["src/a.py", "pyproject.toml"])
        _write(root, "src/a.py", "a = 1  # noqa: E501\n")
        _write(root, "pyproject.toml",
               "[tool.ruff]\nwhen = 2024-01-01\n[tool.ruff.lint]\nselect = ['E']\n")
        hits, unscanned = bs.executor_scan(snapshot.directory, ["src/a.py", "pyproject.toml"])
        assert sorted((hit["path"], hit["kind"]) for hit in hits) == [
            ("pyproject.toml", "cfg-changed"), ("pyproject.toml", "cfg-select-removed"),
            ("src/a.py", "comment-marker:noqa")]
        assert unscanned == []

    def test_a_file_the_scan_fails_on_is_listed_as_not_scanned(self, tmp_path, monkeypatch):
        from utils import suppression_scan

        root = _repo(tmp_path, {"src/a.py": "a = 1\n", "src/b.py": "b = 1\n"})
        snapshot = self._snapshot(tmp_path, root, ["src/a.py", "src/b.py"])
        _write(root, "src/a.py", "a = 1  # noqa: E501\n")
        _write(root, "src/b.py", "b = 1  # noqa: E501\n")
        real = suppression_scan._comment_markers

        def odd(path, src):
            if path == "src/b.py":
                raise ValueError("odd")
            return real(path, src)

        monkeypatch.setattr(suppression_scan, "_comment_markers", odd)
        hits, unscanned = bs.executor_scan(snapshot.directory, ["src/a.py", "src/b.py"])
        assert _where(hits) == [("src/a.py", 1, "# noqa: E501")]
        assert unscanned == [{"path": "src/b.py",
                              "why": "the scan failed on it (ValueError: odd)"}]

    def test_a_link_left_in_the_tree_is_not_read_through(self, tmp_path):
        root = _repo(tmp_path, {"src/a.py": "a = 1\n"})
        outside = tmp_path / "secret.py"
        outside.write_text("TOKEN = 1  # noqa\n", encoding="utf-8")
        snapshot = self._snapshot(tmp_path, root, ["src/a.py"])
        (root / "src" / "a.py").unlink()
        try:
            os.symlink(outside, root / "src" / "a.py")
        except (OSError, NotImplementedError):  # pragma: no cover - no symlinks
            pytest.skip("symlinks unavailable")
        hits, unscanned = bs.executor_scan(snapshot.directory, ["src/a.py"])
        assert hits == []
        assert unscanned and unscanned[0]["path"] == "src/a.py"


# ---------------------------------------------------------------------------
# The native leg: the scan base
# ---------------------------------------------------------------------------

def _tree(root):
    return {path: list(stat) for path, stat in
            (write_prompt.worktree_file_stats(str(root)) or {}).items()}


class TestNativeScan:

    def test_only_the_batch_s_own_additions_are_listed(self, tmp_path):
        root = _repo(tmp_path, {
            "src/a.py": "a = 1\n", "src/b.py": "b = 1\n", "src/c.py": "c = 1\n",
            "tests/test_c.py": "def test_c():\n    pass\n",
        })
        # The user's uncommitted edit, before the bind.
        _write(root, "src/b.py", "b = 1  # noqa: E501\n")
        base = bs.native_base(str(root), ["src/a.py"], _tree(root))
        assert set(base["before"]) == {"src/a.py", "src/b.py"}
        assert json.loads(json.dumps(base)) == base
        # The batch: its declared file, an undeclared file that matched the
        # commit, the user's file again, and a file it created.
        _write(root, "src/a.py", "a = 1  # type: ignore\n")
        _write(root, "tests/test_c.py",
               "import pytest\n\n@pytest.mark.xfail\ndef test_c():\n    pass\n")
        _write(root, "src/b.py", "b = 1  # noqa: E501\nb2 = 2\n")
        _write(root, "src/new.py", "import os  # noqa: F401\n")
        hits, unscanned = bs.native_scan(base, str(root), _tree(root))
        assert _where(hits) == [
            ("src/a.py", 1, "# type: ignore"),
            ("src/new.py", 1, "# noqa: F401"),
            ("tests/test_c.py", 3, "pytest.mark.xfail"),
        ]
        assert unscanned == []

    def test_without_a_tree_only_the_declared_files_are_scanned(self, tmp_path):
        root = _repo(tmp_path, {"src/a.py": "a = 1\n", "src/c.py": "c = 1\n"})
        base = bs.native_base(str(root), ["src/a.py"], None)
        assert base["tree"] is None
        _write(root, "src/a.py", "a = 1  # noqa\n")
        _write(root, "src/c.py", "c = 1  # noqa\n")
        hits, _ = bs.native_scan(base, str(root), _tree(root))
        assert [hit["path"] for hit in hits] == ["src/a.py"]

    def test_another_repository_s_base_lists_nothing(self, tmp_path):
        root = _repo(tmp_path, {"src/a.py": "a = 1\n"})
        base = bs.native_base(str(root), ["src/a.py"], _tree(root))
        _write(root, "src/a.py", "a = 1  # noqa\n")
        assert bs.native_scan(dict(base, repo_root="/elsewhere"), str(root), _tree(root)) == ([], [])


# ---------------------------------------------------------------------------
# The native leg end to end: util write-prompt, then util batch-done
# ---------------------------------------------------------------------------

PROMPT = "# Fix\n\n" + PRIOR_CHANGES_TOKEN + "\n\n1. Fix it.\n"


@pytest.fixture
def native_world(tmp_path):
    root = _repo(tmp_path, {"src/a.py": "def f(a, b, c, d, e):\n    return a\n",
                            "plans/feature.md": "# Feature\n"})
    plan = root / "plans" / "feature.md"
    out = root / "plans" / "feature"
    out.mkdir()
    StateManager(plan).save()
    batches = [{"file_key": "src/a.py", "prompt": PROMPT, "batch_type": "general",
                "fixes": [{"group_id": "f-1", "title": "Priority flag", "file": "src/a.py"}]}]
    phase_dir, _tasks = _materialize(plan, out, "apply_code_fixes", batches, "apply-fixes")
    return root, phase_dir


class TestNativeLegEndToEnd:

    def test_batch_done_prints_what_the_batch_added(self, native_world, capsys, monkeypatch):
        root, phase_dir = native_world
        monkeypatch.chdir(root)
        code, _out, err, _result = _run([
            "util", "write-prompt", "--phase-dir", str(phase_dir),
            "--task-id", "apply-fix-batch-0",
        ], capsys)
        assert code == 0, err
        assert write_prompt.scan_base_path(phase_dir, "apply-fix-batch-0").is_file()
        _write(root, "src/a.py", "def f(a, b, c, d, e, g):  # noqa: PLR0913\n    return a\n")
        code, stdout, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-fix-batch-0", "--status", "completed",
            "--own-check", "passed",
        ], capsys)
        assert code == 0, err
        assert (f"{bs.SUPPRESSION_MARKER} apply-fix-batch-0: src/a.py:1 # noqa: PLR0913 "
                f"— no reason given: def f(a, b, c, d, e, g):  # noqa: PLR0913") in stdout
        assert f'under "{CHECK_THESE_HEADING}"' in stdout
        # Recorded again (the `[OWN_CHECK_NOT_RECORDED]` call), it lists them again.
        code, again, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-fix-batch-0", "--status", "completed",
            "--own-check", "passed",
        ], capsys)
        assert code == 0, err
        assert bs.SUPPRESSION_MARKER in again

    def test_a_scan_that_raises_names_the_declared_files(self, native_world, capsys,
                                                         monkeypatch):
        root, phase_dir = native_world
        monkeypatch.chdir(root)
        code, _out, err, _result = _run([
            "util", "write-prompt", "--phase-dir", str(phase_dir),
            "--task-id", "apply-fix-batch-0",
        ], capsys)
        assert code == 0, err

        def boom(*_args, **_kwargs):
            raise TypeError("odd value")

        monkeypatch.setattr(bs, "native_scan", boom)
        _write(root, "src/a.py", "x = 1  # noqa\n")
        code, stdout, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-fix-batch-0", "--status", "completed",
            "--own-check", "passed",
        ], capsys)
        assert code == 0, err
        assert ("apply-fix-batch-0: src/a.py changed but was not scanned for suppressions "
                "(the suppression scan did not run (TypeError)).") in stdout

    def test_a_clean_failure_and_an_unbound_batch_print_nothing(self, native_world, capsys,
                                                                monkeypatch):
        root, phase_dir = native_world
        monkeypatch.chdir(root)
        _write(root, "src/a.py", "x = 1  # noqa\n")
        # No `util write-prompt` bound it: no base, nothing listed.
        code, stdout, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-fix-batch-0", "--status", "completed",
            "--own-check", "passed",
        ], capsys)
        assert code == 0, err
        assert bs.SUPPRESSION_MARKER not in stdout
        _run(["util", "write-prompt", "--phase-dir", str(phase_dir),
              "--task-id", "apply-fix-batch-0"], capsys)
        _write(root, "src/a.py", "x = 1  # noqa\ny = 2  # noqa\n")
        code, stdout, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-fix-batch-0", "--status", "failed", "--reason", "gave up",
        ], capsys)
        assert code == 0, err
        assert bs.SUPPRESSION_MARKER not in stdout


class TestNativeSiblingTasks:

    def test_a_task_running_beside_it_keeps_its_own_hits(self, tmp_path, capsys, monkeypatch):
        """The native host may run a batch's tasks side by side: each task's
        record lists its own file, never the sibling's."""
        root = _repo(tmp_path, {"src/one.py": "a = 1\n", "src/two.py": "b = 1\n",
                                "plans/feature.md": "# Feature\n"})
        plan = root / "plans" / "feature.md"
        out = root / "plans" / "feature"
        out.mkdir()
        StateManager(plan).save()
        batches = [{"batch_index": 0, "can_parallelize": True, "tasks": [
            {"id": task_id, "title": task_id, "prompt": PROMPT,
             "files_to_modify": [f"src/{name}.py"], "depends_on": [],
             "executor_role": "agent"}
            for task_id, name in (("T001", "one"), ("T002", "two"))
        ]}]
        phase_dir, _tasks = _materialize(plan, out, "implement", batches, "implement")
        monkeypatch.chdir(root)
        for task_id in ("T001", "T002"):
            code, _out, err, _result = _run([
                "util", "write-prompt", "--phase-dir", str(phase_dir), "--task-id", task_id,
            ], capsys)
            assert code == 0, err
        _write(root, "src/one.py", "a = 1  # type: ignore\n")
        _write(root, "src/two.py", "b = 1  # noqa: E501\n")
        for task_id, mine, theirs in (("T001", "src/one.py", "src/two.py"),
                                      ("T002", "src/two.py", "src/one.py")):
            code, stdout, err, _result = _run([
                "util", "batch-done", "--phase-dir", str(phase_dir), "--task-id", task_id,
                "--status", "completed", "--own-check", "passed",
            ], capsys)
            assert code == 0, err
            listed = [line for line in stdout.splitlines()
                      if line.startswith(bs.SUPPRESSION_MARKER)]
            assert len(listed) == 1 and f"{task_id}: {mine}:1" in listed[0], stdout
            assert theirs not in "\n".join(listed)


# ---------------------------------------------------------------------------
# The executor leg end to end: the scheduler, the report and the notice
# ---------------------------------------------------------------------------

class TestExecutorLegEndToEnd:

    def test_the_outcome_report_summary_and_notice_list_it(self, env):
        lines = []
        state = FakeState(env.tmp_path / "state.json")
        summary = env.phase_dir / "summary.md"
        respond = env.native_edits(
            [{"path": "src/a.py", "mode": "patch", "content": "b = 2  # noqa: E501\n"}],
            {"status": "applied", "ids": ["f0"], "files_touched": ["src/a.py"],
             "summary": "fixed", "check": capture.BATCH_CHECK})
        result = env.scheduler([capture.fix_task(env, 0)], kind="apply_code_fixes",
                               state=state, respond=respond, summary_file=str(summary),
                               log=lines.append).run()
        assert result.exit_code == 0, result.messages
        outcome = result.outcomes[0]
        assert _where(outcome.suppressions) == [("src/a.py", 2, "# noqa: E501")]
        assert any(line.startswith(f"{bs.SUPPRESSION_MARKER} apply-fix-batch-0: src/a.py:2")
                   for line in lines), lines
        report = json.loads((env.phase_dir / "write_batch_report.json").read_text())
        assert report["tasks"][0]["suppressions"][0]["reason"] is None
        text = summary.read_text(encoding="utf-8")
        assert CHECK_THESE_HEADING in text and "**no reason given**" in text
        notice = check_records.write_result_notice(result)
        assert "check these: the batches added 1 suppression(s)" in notice
        # A detached caller reads the notice off the mirror instead.
        mirror = json.loads((env.phase_dir / ".status.json").read_text())
        detached = status_text._write_result_notice([mirror])
        assert "check these: the batches added 1 suppression(s)" in detached

    def test_a_resume_still_lists_an_earlier_run_s_hits(self, env):
        state = FakeState(env.tmp_path / "state.json")
        summary = env.phase_dir / "summary.md"
        respond = env.native_edits(
            [{"path": "src/a.py", "mode": "patch", "content": "b = 2  # noqa: E501\n"}],
            {"status": "applied", "ids": ["f0"], "files_touched": ["src/a.py"],
             "summary": "fixed", "check": capture.BATCH_CHECK})
        first = env.scheduler([capture.fix_task(env, 0)], kind="apply_code_fixes",
                              state=state, respond=respond, summary_file=str(summary)).run()
        assert first.exit_code == 0, first.messages
        again = env.scheduler([capture.fix_task(env, 0)], kind="apply_code_fixes",
                              state=state, respond=respond, summary_file=str(summary),
                              resume=True).run()
        assert again.exit_code == 0, again.messages
        assert respond.dispatched == ["apply-fix-batch-0"]
        text = summary.read_text(encoding="utf-8")
        assert CHECK_THESE_HEADING in text
        assert "— apply-fix-batch-0 (earlier run) — " in text
        assert "check these" in check_records.write_result_notice(again)

    def test_a_batch_that_added_none_changes_no_report(self, env):
        state = FakeState(env.tmp_path / "state.json")
        summary = env.phase_dir / "summary.md"
        respond = env.native_edits(
            [{"path": "src/a.py", "mode": "patch", "content": "b = 2\n"}],
            {"status": "applied", "ids": ["f0"], "files_touched": ["src/a.py"],
             "summary": "fixed", "check": capture.BATCH_CHECK})
        result = env.scheduler([capture.fix_task(env, 0)], kind="apply_code_fixes",
                               state=state, respond=respond,
                               summary_file=str(summary)).run()
        assert result.exit_code == 0, result.messages
        row = json.loads((env.phase_dir / "write_batch_report.json").read_text())["tasks"][0]
        assert "suppressions" not in row and "suppressions_not_scanned" not in row
        assert CHECK_THESE_HEADING not in summary.read_text(encoding="utf-8")

    def test_a_plan_editing_phase_is_never_scanned(self, env):
        state = FakeState(env.tmp_path / "state.json")
        respond = env.native_edits(
            [{"path": str(env.plan), "mode": "patch", "content": "x  # noqa\n"}],
            {"status": "applied", "ids": ["s0"], "files_touched": ["plans/p.md"],
             "summary": "edited"})
        result = env.scheduler([env.apply_task(0)], state=state, respond=respond).run()
        assert result.outcomes[0].suppressions == []

    def _stop_a_partial_batch(self, env, state):
        """A two-fix batch that adds a ``# noqa`` and reports only ``f1``
        applied: a partial application, which stops the run."""
        task = make_write_task(
            kind="apply_code_fixes", task_id="apply-fix-batch-0",
            prompt=f"Apply fixes to {env.root / 'src/a.py'}\n\n{PRIOR_CHANGES_TOKEN}\n",
            inputs=build_inputs(plan_file=str(env.plan)), batch_index=0,
            prior_changes_id="src/a.py#00000000", phase_dir=str(env.phase_dir),
            declared_files=[str(env.root / "src/a.py")], expected_ids=["f0", "f1"],
            output_name=write_output_name("apply_code_fixes", 0),
        )
        respond = env.native_edits(
            [{"path": "src/a.py", "mode": "patch", "content": "b = 2  # noqa: E501\n"}],
            {"status": "applied", "ids": ["f1"], "files_touched": ["src/a.py"],
             "summary": "fixed f1; f0 left alone", "check": capture.BATCH_CHECK})
        stopped = env.scheduler([task], kind="apply_code_fixes", state=state,
                                respond=respond).run()
        assert stopped.exit_code == 3, stopped.messages
        assert stopped.outcomes[0].unapplied_ids == ["f0"]
        row = json.loads((env.phase_dir / "write_batch_report.json").read_text())["tasks"][0]
        assert _where(row["suppressions"]) == [("src/a.py", 2, "# noqa: E501")]
        return task, respond, row

    def test_a_batch_kept_at_your_request_carries_what_its_stop_found(self, env, monkeypatch):
        """``_keep_at_request`` never scans again: the kept row, which replaces
        the stopped run's, carries the hits the scan found when it stopped, so
        nothing edited after the stop is listed as the batch's."""
        state = FakeState(env.tmp_path / "state.json")
        summary = env.phase_dir / "summary.md"
        task, respond, row = self._stop_a_partial_batch(env, state)
        scanned = []
        monkeypatch.setattr(bs, "executor_scan",
                            lambda *args, **kwargs: scanned.append(args) or ([], []))
        lines = []
        kept = env.scheduler([task], kind="apply_code_fixes", state=state, respond=respond,
                             resume=True, partial_skips=["apply-fix-batch-0"],
                             summary_file=str(summary), log=lines.append).run()
        assert kept.exit_code == 0, kept.messages
        assert respond.dispatched == ["apply-fix-batch-0"], "kept, not dispatched again"
        assert scanned == [], "a kept batch is not scanned again"
        outcome = kept.outcomes[0]
        assert outcome.suppressions == row["suppressions"]
        assert any(line.startswith(f"{bs.SUPPRESSION_MARKER} apply-fix-batch-0: src/a.py:2")
                   for line in lines), lines
        rows = kept.report_rows()
        assert len(rows) == 1 and not rows[0].get("from_earlier_run")
        assert rows[0]["suppressions"] == row["suppressions"]
        text = summary.read_text(encoding="utf-8")
        assert CHECK_THESE_HEADING in text and "— apply-fix-batch-0 — " in text

    def test_a_kept_batch_with_no_stopped_row_on_record_names_its_files(self, env):
        state = FakeState(env.tmp_path / "state.json")
        task, respond, _row = self._stop_a_partial_batch(env, state)
        (env.phase_dir / "write_batch_report.json").unlink()
        kept = env.scheduler([task], kind="apply_code_fixes", state=state, respond=respond,
                             resume=True, partial_skips=["apply-fix-batch-0"]).run()
        assert kept.exit_code == 0, kept.messages
        outcome = kept.outcomes[0]
        assert outcome.suppressions == []
        assert outcome.suppressions_not_scanned == [
            {"path": "src/a.py", "why": bs.KEPT_NOT_ON_RECORD}]

    @pytest.mark.parametrize("damage", [
        {"suppressions": 5},
        {"suppressions": [{"path": "src/a.py", "line": "x", "marker": "# noqa"}]},
        {"suppressions_not_scanned": ["src/a.py"]},
    ])
    def test_a_damaged_stopped_row_never_fails_the_keep(self, env, damage):
        state = FakeState(env.tmp_path / "state.json")
        task, respond, _row = self._stop_a_partial_batch(env, state)
        report_path = env.phase_dir / "write_batch_report.json"
        report = json.loads(report_path.read_text())
        report["tasks"][0].update(damage)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        kept = env.scheduler([task], kind="apply_code_fixes", state=state, respond=respond,
                             resume=True, partial_skips=["apply-fix-batch-0"]).run()
        assert kept.exit_code == 0, kept.messages
        outcome = kept.outcomes[0]
        assert outcome.ok and outcome.suppressions == []
        assert outcome.suppressions_not_scanned == [
            {"path": "src/a.py", "why": bs.KEPT_NOT_ON_RECORD}]

    def test_a_scan_that_raises_never_fails_the_batch(self, env, monkeypatch):
        def boom(*_args, **_kwargs):
            raise RuntimeError("scanner broke")

        monkeypatch.setattr(bs, "executor_scan", boom)
        lines = []
        state = FakeState(env.tmp_path / "state.json")
        respond = env.native_edits(
            [{"path": "src/a.py", "mode": "patch", "content": "b = 2  # noqa\n"}],
            {"status": "applied", "ids": ["f0"], "files_touched": ["src/a.py"],
             "summary": "fixed", "check": capture.BATCH_CHECK})
        result = env.scheduler([capture.fix_task(env, 0)], kind="apply_code_fixes",
                               state=state, respond=respond, log=lines.append).run()
        assert result.exit_code == 0 and result.outcomes[0].ok
        assert any("suppression scan did not run" in line for line in lines), lines
        # Not silent in the summary either: its files are named as not scanned.
        assert result.outcomes[0].suppressions_not_scanned == [
            {"path": "src/a.py", "why": "the suppression scan did not run (RuntimeError)"}]


# ---------------------------------------------------------------------------
# The reports a person reads
# ---------------------------------------------------------------------------

HIT = {"path": "src/a.py", "line": 4, "kind": "comment-marker:noqa",
       "marker": "# noqa: BLE001", "text": "except Exception:  # noqa: BLE001 - isolate",
       "reason": "isolate"}


class TestTheReports:

    def test_the_apply_summary_lists_an_earlier_run_s_hits_too(self, tmp_path):
        rows = [
            {"task_id": "apply-fix-batch-0", "status": "ok", "write_outcome": "applied",
             "expected_ids": ["a"], "files_touched": ["src/a.py"], "from_earlier_run": True,
             "suppressions": [HIT]},
            {"task_id": "apply-fix-batch-1", "status": "ok", "write_outcome": "applied",
             "expected_ids": ["b"], "files_touched": ["src/b.py"],
             "suppressions_not_scanned": [{"path": "src/b.py", "why": "it is not a regular file"}]},
        ]
        report = _report(tmp_path, phase_name="apply-fixes", write_kind="apply_code_fixes",
                         item_noun="fix", item_noun_plural="fixes", target_file=None,
                         write_result=_Result(rows))
        text = Path(report.write_apply_summary_report()).read_text(encoding="utf-8")
        assert CHECK_THESE_HEADING in text
        assert ("- `src/a.py:4` `# noqa: BLE001` — reason: isolate — "
                "apply-fix-batch-0 (earlier run) — ") in text
        assert "- `src/b.py` — apply-fix-batch-1: not scanned, it is not a regular file" in text

    def test_the_outcome_row_is_additive(self):
        outcome = TaskOutcome(task_id="T001", kind="implement", runner="codex",
                              status="ok", write_outcome="applied")
        assert "suppressions" not in outcome.to_dict()
        outcome.suppressions = [HIT]
        assert outcome.to_dict()["suppressions"] == [HIT]


# ---------------------------------------------------------------------------
# The writers are told the rule
# ---------------------------------------------------------------------------

RULE = ("never add a lint, type-checker or coverage suppression (`# noqa`, "
        "`# type: ignore`, a config ignore entry) to make a check pass, unless "
        "the code is right as written and you add a comment saying why")


class TestTheWritersAreToldTheRule:

    def test_an_implement_code_task_is_told_once(self):
        from utils.batch_prompt import with_result_contract

        task = {"task_id": "T001", "expected_ids": ["T001"],
                "files_to_modify": ["src/a.py"], "files": ["src/a.py"]}
        prompt = with_result_contract("implement", task, "Do the task.")
        assert prompt.count(RULE) == 1
        assert "Never make a check pass by weakening it (deleting or loosening an " \
            "assertion, skipping a test)" in prompt

    def test_a_check_task_is_not(self):
        from utils.batch_prompt import with_result_contract

        assert RULE not in with_result_contract(
            "implement", {"task_id": "T002", "expected_ids": ["T002"]}, "Run it.")

    def test_an_apply_batch_reads_it_in_its_known_checks_section_only(self):
        from utils.batch_prompt import with_result_contract
        from utils.known_checks import known_checks_section

        section = known_checks_section([("pytest -q", "driver.verify_command")])
        assert section.count(RULE) == 1
        batch = {"task_id": "apply-fix-batch-0", "expected_ids": ["a1"],
                 "files_to_modify": ["src/x.py"], "files": ["src/x.py"]}
        prompt = with_result_contract("apply_code_fixes", batch, "Fix it.",
                                      checks_section=section)
        assert prompt.count(RULE) == 1


# ---------------------------------------------------------------------------
# improve-code: what the composite's own report and result carry
# ---------------------------------------------------------------------------

class TestImproveCode:

    def test_the_composite_names_what_its_apply_added(self, tmp_path):
        """A real ``improve-code`` subprocess whose writers add a reasonless
        ``# noqa``: the apply's summary lists it, and the composite's result
        line names it."""
        from tests.test_e2e_improve_modes import WRITE_RULES, World

        rules = [dict(rule) for rule in WRITE_RULES]
        for rule in rules:
            if rule["when"] == "Fix the following":
                rule["content"] = "\nimport os  # noqa: F401\n# improve-modes-e2e fixed {id}\n"
        world = World(tmp_path, working_diff=True)
        run = world.composite("improve-code",
                              env=world.env(IMPROVE_E2E_WRITES=json.dumps(rules)))
        assert run.exit_code == 0, run.explain()
        report = world.report("apply-fixes")
        found = [hit for task in report["tasks"] for hit in task.get("suppressions") or []]
        assert found and all(hit["marker"] == "# noqa: F401" and hit["reason"] is None
                             for hit in found), report["tasks"]
        summary = (world.out / "apply-fixes" / "summary.md").read_text(encoding="utf-8")
        assert CHECK_THESE_HEADING in summary
        assert "check these: the batches added" in (run.result.get("notice") or ""), (
            run.result)
