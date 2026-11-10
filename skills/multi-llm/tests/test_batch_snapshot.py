"""``util write-prompt``'s pre-batch snapshot of a shared-artifact batch.

PERF-1 / WRITE-13, the first slice of plans/apply-phase-throughput.md D1: the
host used to copy the plan (the tasks file) in a Bash turn of its own before
each apply-suggestions (apply-task-suggestions) batch; ``util write-prompt``
now takes that copy as it binds the prompt. Its two Codex reviews (U31, U66)
pinned what the copy must survive: a phase dir link, rebinding a batch, a
failed publication, two hosts at once, damaged evidence, batches bound out of
order and two plans of one name. The rules are in
docs/development/write-safety.md; no command restores from the copy (it lies
where a batch's subagent can write), so the apply instructions give the
manual way back instead.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.support.write_prompt_world import (  # noqa: E402
    PROMPT,
    _materialize,
    _suggestion_batches,
    prompt_plan_dir,  # noqa: F401 - the plan_dir fixture
)
from utils import batch_done, process_control, prompt_files, write_prompt  # noqa: E402
from utils.write_phase_bridge import (  # noqa: E402
    WRITE_TASK_FILENAME, materialize_write_tasks, native_run_record,
    write_task_file_for_phase,
)

INSTRUCTIONS = Path(__file__).parent.parent / "instructions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_state(out, plan, **extra):
    """The plan dir's ``state.json`` as a phase's emit leaves it: nothing
    recorded yet."""
    (out / "state.json").write_text(
        json.dumps({"plan_path": str(plan), "task_status": {}, **extra}), encoding="utf-8")


@pytest.fixture
def phase(plan_dir):
    """An apply-suggestions phase of two batches, emitted and not run."""
    plan, out = plan_dir
    phase_dir, _tasks = _materialize(
        plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
    _seed_state(out, plan)
    return plan, out, phase_dir


def _bind(phase_dir, task_id, capsys, *extra):
    code = write_prompt.main(["--phase-dir", str(phase_dir), "--task-id", task_id, *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _marker(text, marker):
    lines = [line for line in text.splitlines() if line.startswith(marker + " ")]
    assert len(lines) == 1, text
    return Path(lines[0][len(marker) + 1:])


def _tmp(root):
    # Under the repository root as pinned: realpath'd (macOS /var).
    return Path(os.path.realpath(root)) / ".multi-llm" / "tmp"


def _slot(phase_dir, artifact, kind="apply_suggestions", stem="feature"):
    return write_prompt.snapshot_slot(
        kind, stem, os.path.realpath(phase_dir), os.path.normpath(str(artifact)))


def _binding_path(root, phase_dir, artifact, **kwargs):
    return _tmp(root) / write_prompt.snapshot_binding_name(_slot(phase_dir, artifact, **kwargs))


def _binding(root, phase_dir, artifact, **kwargs):
    return json.loads(_binding_path(root, phase_dir, artifact, **kwargs).read_text(encoding="utf-8"))


def _copies(root, phase_dir, artifact, **kwargs):
    slot = _slot(phase_dir, artifact, **kwargs)
    folder = _tmp(root)
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.iterdir()
                  if p.name.startswith(slot + ".") and p.name.endswith(".md"))


def _record(phase_dir, task_id, capsys, status="completed"):
    """The host's step-7 record, by the real ``util batch-done``."""
    assert batch_done.main([
        "--phase-dir", str(phase_dir), "--task-id", task_id, "--status", status,
    ]) == 0
    capsys.readouterr()


def _root(out):
    return out.parent.parent


# ---------------------------------------------------------------------------
# The copy
# ---------------------------------------------------------------------------

class TestThePreBatchSnapshot:
    """Byte for byte, after every refusal, before the prompt is handed out."""

    def test_the_roles_are_the_shared_artifact_kinds(self):
        from utils.subagent_tasks import SHARED_ARTIFACT_WRITE_KINDS
        assert set(write_prompt.SNAPSHOT_ROLES) == set(SHARED_ARTIFACT_WRITE_KINDS)

    def test_the_plan_is_copied_and_both_paths_printed_after_the_markers(
        self, phase, capsys
    ):
        plan, out, phase_dir = phase
        code, stdout, _err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 0
        lines = stdout.splitlines()
        slot = _slot(phase_dir, plan)
        copy = _tmp(_root(out)) / write_prompt.snapshot_copy_name(
            slot, write_prompt._sha256(plan.read_bytes()))
        assert copy.read_bytes() == plan.read_bytes()
        assert (copy.parent / ".gitignore").read_text() == "*\n"
        # The three prompt markers keep their lines; the snapshot's follow.
        assert lines[0].startswith(write_prompt.MARKER_PATH + " ")
        assert lines[3:] == [
            f"{write_prompt.MARKER_SNAPSHOT} {copy}",
            f"{write_prompt.MARKER_ARTIFACT} {plan.resolve()}",
        ]
        # Not an executor rollback slot.
        assert not (phase_dir / "pre_batch").exists()

    def test_the_tasks_file_is_copied_under_the_plan_s_stem(self, plan_dir, capsys):
        plan, out = plan_dir
        tasks_md = out / "tasks" / "tasks.md"
        tasks_md.parent.mkdir()
        tasks_md.write_text("# Tasks\n\n## T001: First\n", encoding="utf-8")
        phase_dir = out / "apply-task-suggestions"
        phase_dir.mkdir()
        tasks = materialize_write_tasks(
            kind="apply_task_suggestions", batches=_suggestion_batches(),
            phase_dir=str(phase_dir), plan_file=str(plan), tasks_file=str(tasks_md),
        )
        write_task_file_for_phase(
            kind="apply_task_suggestions", tasks=tasks, phase_dir=str(phase_dir),
            plan_file=str(plan), tasks_file=str(tasks_md),
        )
        _seed_state(out, plan.resolve())
        code, stdout, _err = _bind(phase_dir, "apply-task-batch-0", capsys)
        assert code == 0
        copy = _marker(stdout, write_prompt.MARKER_SNAPSHOT)
        assert copy.name.startswith("batch_snapshot_tasks_feature-")
        assert copy.read_bytes() == tasks_md.read_bytes()
        assert _marker(stdout, write_prompt.MARKER_ARTIFACT) == tasks_md.resolve()

    @pytest.mark.parametrize("kind,subdir,task_id", [
        ("implement", "implement", "T001"),
        ("apply_code_fixes", "apply-fixes", "apply-fix-batch-0"),
    ])
    def test_the_other_write_kinds_take_none(self, plan_dir, capsys, kind, subdir, task_id):
        from tests.support.write_prompt_world import _implement_batches
        plan, out = plan_dir
        batches = _implement_batches() if kind == "implement" else [{
            "file_key": "src/a.py", "prompt": PROMPT,
            "fixes": [{"group_id": "g-fix", "title": "Fix it"}],
        }]
        phase_dir, _tasks = _materialize(plan, out, kind, batches, subdir)
        code, stdout, err = _bind(phase_dir, task_id, capsys)
        assert code == 0
        assert write_prompt.MARKER_SNAPSHOT not in stdout + err
        assert not (_root(out) / ".multi-llm").exists()

    def test_the_copy_is_byte_for_byte(self, phase, capsys):
        """A host that retyped the plan could change its whitespace and read
        as a false "changed"; the copy keeps every byte, invalid UTF-8 too."""
        plan, _out, phase_dir = phase
        raw = b"# Feature\r\n\r\n## Step 2 \t\r\nBody \xff\xfe trailing   \n\n"
        plan.write_bytes(raw)
        code, stdout, _err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 0
        assert _marker(stdout, write_prompt.MARKER_SNAPSHOT).read_bytes() == raw

    def test_the_prompt_then_the_copy_then_its_binding_then_the_markers(
        self, phase, capsys, monkeypatch
    ):
        """The order the check rests on: the prompt is published, then the
        copy (pre-batch bytes) and the binding naming it, and only then do the
        markers hand the batch out; copying the copy back restores the plan
        exactly, by hand, where a human finds it after a stop."""
        plan, out, phase_dir = phase
        before = plan.read_bytes()
        slot = _slot(phase_dir, plan)
        copy_name = write_prompt.snapshot_copy_name(slot, write_prompt._sha256(before))
        seen = []
        folder_class = prompt_files.PromptFolder
        real_write_bytes = folder_class.write_bytes
        real_print = print

        def watching(folder, name, data):
            seen.append(name)
            return real_write_bytes(folder, name, data)

        def printing(*args, **kwargs):
            line = " ".join(str(arg) for arg in args)
            if line.startswith(write_prompt.MARKER_PATH):
                seen.append(("markers", (_tmp(_root(out)) / copy_name).read_bytes()))
            return real_print(*args, **kwargs)

        monkeypatch.setattr(folder_class, "write_bytes", watching)
        monkeypatch.setattr("builtins.print", printing)
        assert write_prompt.main(
            ["--phase-dir", str(phase_dir), "--task-id", "apply-batch-0"]) == 0
        monkeypatch.undo()
        assert seen == [
            ".gitignore", "apply-batch-0.prompt.txt", copy_name,
            write_prompt.snapshot_binding_name(slot), ("markers", before),
        ]
        copy = _tmp(_root(out)) / copy_name
        # The subagent runs: it edits the plan, never the copy.
        plan.write_bytes(before + b"\n## Email validation\n")
        assert copy.read_bytes() == before != plan.read_bytes()
        # The batch stops (a partial application): a human copies it back.
        plan.write_bytes(copy.read_bytes())
        assert plan.read_bytes() == before

    def test_the_next_batch_retakes_it_and_the_old_copy_goes(self, phase, capsys):
        plan, out, phase_dir = phase
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        edited = plan.read_bytes() + b"\nbatch 0 landed\n"
        plan.write_bytes(edited)
        _record(phase_dir, "apply-batch-0", capsys)
        code, stdout, _err = _bind(phase_dir, "apply-batch-1", capsys)
        assert code == 0
        copy = _marker(stdout, write_prompt.MARKER_SNAPSHOT)
        assert copy.read_bytes() == edited
        assert not _marker(first, write_prompt.MARKER_SNAPSHOT).exists()
        assert _copies(_root(out), phase_dir, plan) == [copy.name]

    @pytest.mark.parametrize("refusal", ["unknown task", "write policy"])
    def test_a_refused_call_leaves_the_previous_copy_alone(
        self, phase, capsys, monkeypatch, refusal
    ):
        plan, out, phase_dir = phase
        before = plan.read_bytes()
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        binding = _binding_path(_root(out), phase_dir, plan).read_bytes()
        plan.write_bytes(before + b"\nedited by batch 0\n")
        task_id = "apply-batch-1"
        if refusal == "unknown task":
            task_id = "apply-batch-9"
        else:
            # Refused after the prompt rendered: the last check before output.
            monkeypatch.setattr(write_prompt, "policy_refusal", lambda *_a: "denied.")
            monkeypatch.setattr(write_prompt, "policy_refused_hits", lambda *_a: [])
        code, stdout, _err = _bind(phase_dir, task_id, capsys)
        assert code == 3
        assert write_prompt.MARKER_SNAPSHOT not in stdout
        assert _marker(first, write_prompt.MARKER_SNAPSHOT).read_bytes() == before
        assert _binding_path(_root(out), phase_dir, plan).read_bytes() == binding

    def test_no_snapshot_no_prompt(self, phase, capsys):
        """The artifact cannot be read: exit 3, and nothing is handed out."""
        plan, _out, phase_dir = phase
        plan.unlink()
        code, stdout, err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 3
        assert write_prompt.MARKER_PATH not in stdout
        assert "pre-batch snapshot" in err and str(plan.resolve()) in err
        assert not (phase_dir / "prompts" / "apply-batch-0.prompt.txt").exists()

    @staticmethod
    def _outside_with_two_repos(tmp_path_factory):
        """A plan and its phase dir in no repository, and two repositories
        A and B a host could run the loop from."""
        import subprocess

        base = tmp_path_factory.mktemp("two-repos")
        repos = []
        for name in ("A", "B"):
            repo = base / name
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
            repos.append(repo)
        plan = base / "notes" / "feature.md"
        plan.parent.mkdir()
        plan.write_text("ORIGINAL\n", encoding="utf-8")
        out = base / "notes" / "feature"
        out.mkdir()
        phase_dir, _tasks = _materialize(
            plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
        _seed_state(out, plan)
        return base, plan, out, phase_dir, repos

    def test_a_plan_outside_every_repository_keeps_its_slot_in_the_phase_dir(
        self, tmp_path_factory, capsys, monkeypatch,
    ):
        """Fable review 2026-11-10, F2: every batch of a plan outside every
        repository was refused. Its slot lives in the phase dir's own
        `snapshots/` folder, whatever the working directory (none here), and
        the loop runs batch after batch."""
        base, plan, out, phase_dir, _repos = self._outside_with_two_repos(tmp_path_factory)
        monkeypatch.chdir(base)

        code, stdout, err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 0, err
        snapshot = _marker(stdout, write_prompt.MARKER_SNAPSHOT)
        assert snapshot.parent == Path(os.path.realpath(phase_dir)) / write_prompt.PHASE_SNAPSHOTS
        assert snapshot.read_text(encoding="utf-8") == "ORIGINAL\n"
        assert list(base.rglob(".multi-llm")) == []

        # The batch's record carries the generation, and the next batch binds.
        plan.write_text("APPLIED\n", encoding="utf-8")
        _record(phase_dir, "apply-batch-0", capsys)
        state = json.loads((out / "state.json").read_text(encoding="utf-8"))
        assert write_prompt.RECORD_GENERATION_KEY in (
            state[write_prompt.SNAPSHOT_RECORDS_KEY]["apply-suggestions"]["apply-batch-0"])
        code, stdout, err = _bind(phase_dir, "apply-batch-1", capsys)
        assert code == 0, err
        assert _marker(stdout, write_prompt.MARKER_SNAPSHOT).read_text(
            encoding="utf-8") == "APPLIED\n"

    def test_binds_and_records_from_any_repository_reach_the_one_slot(
        self, tmp_path_factory, capsys, monkeypatch,
    ):
        """Codex reviews of F02 (#1 and its re-check): with the slot under the
        working directory's repository, batch 0 bound again from B took a new
        baseline of its own half-applied bytes, a record from B missed A's
        generation, and the next batch moved the slot; a kept record of the
        chosen repository could itself be replaced. The slot is fixed by the
        phase dir: from A, then B, every call reaches it and the original
        baseline stays."""
        base, plan, out, phase_dir, (repo_a, repo_b) = self._outside_with_two_repos(
            tmp_path_factory)
        monkeypatch.chdir(repo_a)
        code, first, err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 0, err
        baseline = _marker(first, write_prompt.MARKER_SNAPSHOT)
        plan.write_text("PARTLY APPLIED\n", encoding="utf-8")

        monkeypatch.chdir(repo_b)
        code, again, err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 0, err
        assert _marker(again, write_prompt.MARKER_SNAPSHOT) == baseline
        assert baseline.read_text(encoding="utf-8") == "ORIGINAL\n"

        _record(phase_dir, "apply-batch-0", capsys)
        state = json.loads((out / "state.json").read_text(encoding="utf-8"))
        assert write_prompt.RECORD_GENERATION_KEY in (
            state[write_prompt.SNAPSHOT_RECORDS_KEY]["apply-suggestions"]["apply-batch-0"])
        monkeypatch.chdir(repo_a)
        code, nxt, err = _bind(phase_dir, "apply-batch-1", capsys)
        assert code == 0, err
        moved = _marker(nxt, write_prompt.MARKER_SNAPSHOT)
        assert moved.parent == baseline.parent
        assert moved.read_text(encoding="utf-8") == "PARTLY APPLIED\n"
        assert not (repo_a / ".multi-llm").exists() and not (repo_b / ".multi-llm").exists()

    def test_a_first_bind_from_another_repository_waits_for_the_one_lock(
        self, tmp_path_factory, capsys, monkeypatch,
    ):
        """Codex re-check of F02, #2: two first binds from different
        repositories each chose their own repository, lock and baseline. B's
        bind, interleaved while A holds the slot (before A reads it), now
        meets A's lock in the phase dir, and once A is done it keeps A's
        baseline."""
        base, plan, out, phase_dir, (repo_a, repo_b) = self._outside_with_two_repos(
            tmp_path_factory)
        monkeypatch.setattr(write_prompt, "SNAPSHOT_LOCK_TIMEOUT_S", 0.2)
        decide = write_prompt._decide
        interleaved = []

        def b_binds_meanwhile(*args, **kwargs):
            monkeypatch.setattr(write_prompt, "_decide", decide)
            os.chdir(repo_b)
            interleaved.append(write_prompt.main(
                ["--phase-dir", str(phase_dir), "--task-id", "apply-batch-0"]))
            os.chdir(repo_a)
            if interleaved[-1] == 0:
                # B was handed the batch: its subagent edits the plan.
                plan.write_text("PARTLY APPLIED AFTER B\n", encoding="utf-8")
            return decide(*args, **kwargs)

        monkeypatch.setattr(write_prompt, "_decide", b_binds_meanwhile)
        monkeypatch.chdir(repo_a)
        code, first, err = _bind(phase_dir, "apply-batch-0", capsys)

        assert interleaved == [3]
        assert f"holds {os.path.realpath(phase_dir)}" in " ".join(err.split())
        assert code == 0
        baseline = _marker(first, write_prompt.MARKER_SNAPSHOT)
        assert baseline.read_text(encoding="utf-8") == "ORIGINAL\n"
        monkeypatch.chdir(repo_b)
        code, again, err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 0, err
        assert _marker(again, write_prompt.MARKER_SNAPSHOT) == baseline
        assert not (repo_b / ".multi-llm").exists()

    def test_a_repository_git_disagrees_with_is_not_passed_over(
        self, plan_dir, capsys, monkeypatch, tmp_path_factory,
    ):
        """The plan's repository cannot be pinned (a `core.worktree` that
        disagrees): refused, never answered from the working directory's."""
        import subprocess

        plan, out = plan_dir
        phase_dir, _tasks = _materialize(
            plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
        _seed_state(out, plan)
        elsewhere = tmp_path_factory.mktemp("elsewhere")
        subprocess.run(["git", "-C", str(_root(out)), "config", "core.worktree",
                        str(elsewhere)], check=True)
        other = tmp_path_factory.mktemp("cwd-repo")
        subprocess.run(["git", "init", "-q", str(other)], check=True, capture_output=True)
        monkeypatch.chdir(other)
        code, stdout, err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 3
        assert write_prompt.MARKER_PATH not in stdout
        assert "pre-batch snapshot" in err
        assert not (other / ".multi-llm").exists()

    def test_stdout_mode_keeps_stdout_the_prompt_and_still_takes_it(self, phase, capsys):
        plan, _out, phase_dir = phase
        code, stdout, err = _bind(phase_dir, "apply-batch-0", capsys, "--stdout")
        assert code == 0
        text, _task, _info = write_prompt.render_write_prompt(phase_dir, "apply-batch-0")
        assert stdout == text
        assert _marker(err, write_prompt.MARKER_SNAPSHOT).read_bytes() == plan.read_bytes()

    @pytest.mark.parametrize("host_copies", ["before", "after"])
    def test_a_host_that_still_copies_it_itself_breaks_nothing(
        self, phase, capsys, host_copies
    ):
        """The old instructions' step 2 (`cp <plan> <its fixed name>`), run
        before or after this command: the host's copy and the command's both
        hold the plan, and neither touches the other."""
        plan, out, phase_dir = phase
        legacy = _tmp(_root(out)) / "batch_snapshot_plan_feature.md"

        def host_cp():
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_bytes(plan.read_bytes())

        if host_copies == "before":
            host_cp()
        code, stdout, _err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 0
        if host_copies == "after":
            host_cp()
        assert legacy.read_bytes() == plan.read_bytes()
        assert _marker(stdout, write_prompt.MARKER_SNAPSHOT).read_bytes() == plan.read_bytes()


# ---------------------------------------------------------------------------
# One batch's baseline
# ---------------------------------------------------------------------------

class TestTheBaselineBelongsToItsBatch:
    """The first Codex review of U31 (#2): every call replaced the copy, so
    binding batch 0 again after its subagent had edited the plan made the
    edited plan its baseline. The copy is bound to its batch and run, kept
    when that batch is bound again, and replaced for another batch of the run
    only once the bound one is recorded."""

    def test_the_binding_names_the_batch_and_the_run(self, phase, capsys):
        import hashlib
        plan, out, phase_dir = phase
        _code, stdout, _err = _bind(phase_dir, "apply-batch-0", capsys)
        binding = _binding(_root(out), phase_dir, plan)
        digest = hashlib.sha256(plan.read_bytes()).hexdigest()
        assert binding["version"] == write_prompt.SNAPSHOT_BINDING_VERSION
        assert binding["task_id"] == "apply-batch-0"
        assert binding["run"] == hashlib.sha256(
            (phase_dir / WRITE_TASK_FILENAME).read_bytes()).hexdigest()
        assert binding["phase_dir"] == os.path.realpath(phase_dir)
        assert binding["artifact"] == str(plan.resolve())
        assert binding["expected_ids"] == ["g-email", "g-password"]
        assert binding["sha256"] == digest
        assert binding["copy"] == _marker(stdout, write_prompt.MARKER_SNAPSHOT).name

    def test_binding_the_same_batch_again_keeps_its_baseline(self, phase, capsys):
        plan, out, phase_dir = phase
        before = plan.read_bytes()
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        binding = _binding_path(_root(out), phase_dir, plan).read_bytes()
        # Its subagent edits part of the plan; the host binds batch 0 again.
        plan.write_bytes(before + b"\n## Email validation (half)\n")

        code, again, _err = _bind(phase_dir, "apply-batch-0", capsys)

        assert code == 0
        assert again.splitlines()[3:] == first.splitlines()[3:]
        assert _marker(again, write_prompt.MARKER_SNAPSHOT).read_bytes() == before
        assert _binding_path(_root(out), phase_dir, plan).read_bytes() == binding

    @pytest.mark.parametrize("rebind", ["same batch", "next batch"])
    def test_a_failed_prompt_publication_keeps_the_baseline(
        self, phase, tmp_path, capsys, rebind
    ):
        """The rebind-then-fail sequence: the prompt cannot be written
        (``prompts`` is a link), so nothing is handed out and the bound
        generation stays as it was."""
        plan, out, phase_dir = phase
        before = plan.read_bytes()
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        binding = _binding_path(_root(out), phase_dir, plan).read_bytes()
        plan.write_bytes(before + b"\nedited by batch 0\n")
        task_id = "apply-batch-0"
        if rebind == "next batch":
            _record(phase_dir, "apply-batch-0", capsys)
            task_id = "apply-batch-1"
        prompts = phase_dir / "prompts"
        prompts.rename(phase_dir / "prompts.moved")
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir()
        os.symlink(outside, prompts, target_is_directory=True)

        code, stdout, _err = _bind(phase_dir, task_id, capsys)

        assert code == 1
        assert write_prompt.MARKER_SNAPSHOT not in stdout
        assert list(outside.iterdir()) == []
        assert _marker(first, write_prompt.MARKER_SNAPSHOT).read_bytes() == before
        assert _binding_path(_root(out), phase_dir, plan).read_bytes() == binding
        assert _copies(_root(out), phase_dir, plan) == [
            _marker(first, write_prompt.MARKER_SNAPSHOT).name]

    def test_another_batch_waits_for_the_bound_one_s_record(self, phase, capsys):
        """A transition within a run needs the bound batch's `util
        batch-done` record of this run: a status an earlier run left under
        the same id (a `--resume` numbers its batches afresh) is not one."""
        plan, out, phase_dir = phase
        # An earlier run recorded an `apply-batch-0` (no record of this run).
        _seed_state(out, plan, task_status={"apply-batch-0": "completed"})
        before = plan.read_bytes()
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        edited = before + b"\nhalf of batch 0\n"
        plan.write_bytes(edited)

        code, stdout, err = _bind(phase_dir, "apply-batch-1", capsys)

        assert code == 3
        assert write_prompt.MARKER_PATH not in stdout
        assert ("apply-batch-0, an earlier batch of this run, has no `util batch-done` "
                "record from this run") in " ".join(err.split())
        assert _marker(first, write_prompt.MARKER_SNAPSHOT).read_bytes() == before
        assert _binding(_root(out), phase_dir, plan)["task_id"] == "apply-batch-0"
        assert not (phase_dir / "prompts" / "apply-batch-1.prompt.txt").exists()

        # Recorded now: the transition goes ahead and batch 1 gets its baseline.
        _record(phase_dir, "apply-batch-0", capsys)
        code, stdout, _err = _bind(phase_dir, "apply-batch-1", capsys)
        assert code == 0
        assert _marker(stdout, write_prompt.MARKER_SNAPSHOT).read_bytes() == edited
        assert _binding(_root(out), phase_dir, plan)["task_id"] == "apply-batch-1"

    def test_a_new_run_takes_a_new_baseline(self, phase, capsys):
        """A re-emitted `write_tasks.json` (a `--resume`, a new run) is
        another run: batch 0 gets the plan as it is now, recorded or not."""
        plan, out, phase_dir = phase
        _bind(phase_dir, "apply-batch-0", capsys)
        old_run = _binding(_root(out), phase_dir, plan)["run"]
        edited = plan.read_bytes() + b"\nkept by the user, then resumed\n"
        plan.write_bytes(edited)
        _materialize(plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
        # A native emit stamps its time; this one writes the same bytes, so
        # one more byte makes it another run.
        (phase_dir / WRITE_TASK_FILENAME).write_text(
            (phase_dir / WRITE_TASK_FILENAME).read_text() + "\n", encoding="utf-8")

        code, stdout, _err = _bind(phase_dir, "apply-batch-0", capsys)

        assert code == 0
        assert _marker(stdout, write_prompt.MARKER_SNAPSHOT).read_bytes() == edited
        assert _binding(_root(out), phase_dir, plan)["run"] != old_run

    @pytest.mark.parametrize("damage", ["altered", "removed"])
    def test_a_bound_baseline_that_changed_is_never_retaken(self, phase, capsys, damage):
        plan, _out, phase_dir = phase
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        copy = _marker(first, write_prompt.MARKER_SNAPSHOT)
        plan.write_bytes(plan.read_bytes() + b"\nbatch 0 edit\n")
        if damage == "altered":
            copy.write_bytes(b"something else\n")
        else:
            copy.unlink()

        code, stdout, err = _bind(phase_dir, "apply-batch-0", capsys)

        assert code == 3
        assert write_prompt.MARKER_PATH not in stdout
        assert "no longer holds the bytes it was taken with" in err
        if damage == "altered":
            assert copy.read_bytes() == b"something else\n"
        else:
            assert not copy.exists()


# ---------------------------------------------------------------------------
# Codex review 2, #3: one generation at a time
# ---------------------------------------------------------------------------

class TestAGenerationIsPublishedWhole:
    """The copy used to replace batch 0's before batch 1's binding was
    written, so a failure between the two left a binding naming bytes that
    were gone. A generation is a new copy and the binding naming it, the
    binding renamed into place last, the old copy removed only after."""

    @pytest.mark.parametrize("failing", ["binding", "copy"])
    def test_a_failed_write_keeps_the_previous_generation(
        self, phase, capsys, monkeypatch, failing
    ):
        plan, out, phase_dir = phase
        before = plan.read_bytes()
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        old_copy = _marker(first, write_prompt.MARKER_SNAPSHOT)
        binding_path = _binding_path(_root(out), phase_dir, plan)
        bound = binding_path.read_bytes()
        _record(phase_dir, "apply-batch-0", capsys)
        edited = before + b"\nbatch 0 landed\n"
        plan.write_bytes(edited)
        real_write_bytes = prompt_files.PromptFolder.write_bytes

        def failing_write(folder, name, data):
            if (failing == "binding" and name.endswith(write_prompt.SNAPSHOT_BINDING_SUFFIX)
                    or failing == "copy" and name.endswith(".md")
                    and name.startswith("batch_snapshot_")):
                raise OSError("injected I/O failure")
            return real_write_bytes(folder, name, data)

        monkeypatch.setattr(prompt_files.PromptFolder, "write_bytes", failing_write)
        code, stdout, err = _bind(phase_dir, "apply-batch-1", capsys)
        monkeypatch.undo()

        assert code == 1
        assert write_prompt.MARKER_PATH not in stdout
        assert "The previous one is unchanged" in err
        # Batch 0's generation, whole: its binding, and the copy it names.
        assert binding_path.read_bytes() == bound
        assert old_copy.read_bytes() == before
        assert json.loads(bound)["sha256"] == write_prompt._sha256(old_copy.read_bytes())
        assert _copies(_root(out), phase_dir, plan) == [old_copy.name]
        # Nothing was lost: the next attempt publishes batch 1's generation.
        code, stdout, _err = _bind(phase_dir, "apply-batch-1", capsys)
        assert code == 0
        assert _marker(stdout, write_prompt.MARKER_SNAPSHOT).read_bytes() == edited
        assert not old_copy.exists()


# ---------------------------------------------------------------------------
# Codex review 2, #4: one call at a time holds the slot
# ---------------------------------------------------------------------------

class TestOneCallAtATimeHoldsTheSlot:
    """Two hosts: B read "no binding", A bound batch 0 and its subagent edited
    the plan, then B went on and took those edits as batch 0's baseline. The
    read, the checks, the capture and the publication now run under the
    slot's lock, held until the call ends."""

    def test_a_second_host_is_held_off_while_the_first_holds_the_slot(
        self, phase, capsys, monkeypatch
    ):
        plan, out, phase_dir = phase
        before = plan.read_bytes()
        real_read_slot = write_prompt._read_slot
        other = {}

        def host_a_runs_meanwhile(*args, **kwargs):
            now = real_read_slot(*args, **kwargs)
            if not other:
                # Host B has read the slot (no binding yet); host A binds the
                # same batch now, and its subagent edits the plan if A won.
                other["running"] = True
                monkeypatch.setattr(write_prompt, "SNAPSHOT_LOCK_TIMEOUT_S", 0)
                other["code"] = write_prompt.main(
                    ["--phase-dir", str(phase_dir), "--task-id", "apply-batch-0"])
                other["streams"] = capsys.readouterr()
                if other["code"] == 0:
                    plan.write_bytes(before + b"\nhost A's subagent, half done\n")
            return now

        monkeypatch.setattr(write_prompt, "_read_slot", host_a_runs_meanwhile)
        code, stdout, _err = _bind(phase_dir, "apply-batch-0", capsys)  # host B

        assert other["code"] == 3
        assert "Another `util write-prompt` holds" in other["streams"].err
        assert write_prompt.MARKER_PATH not in other["streams"].out
        assert code == 0
        assert _marker(stdout, write_prompt.MARKER_SNAPSHOT).read_bytes() == before

        # B done, the lock free: A's retry keeps B's baseline.
        monkeypatch.setattr(write_prompt, "_read_slot", real_read_slot)
        code, again, _err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 0
        assert again.splitlines()[3:] == stdout.splitlines()[3:]

    def test_the_lock_is_dropped_when_a_call_ends_refused(self, phase, capsys):
        plan, out, phase_dir = phase
        plan.unlink()  # refused under the lock: the plan cannot be read
        assert _bind(phase_dir, "apply-batch-0", capsys)[0] == 3
        lock = _tmp(_root(out)) / (_slot(phase_dir, plan) + write_prompt.SNAPSHOT_LOCK_SUFFIX)
        fd = os.open(lock, os.O_RDWR)
        try:
            assert process_control.wait_for_fd_lock(fd, timeout_s=0)
            process_control.release_fd_lock(fd)
        finally:
            os.close(fd)

    def test_a_link_at_the_lock_name_is_never_followed(self, phase, tmp_path, capsys):
        plan, out, phase_dir = phase
        folder = _tmp(_root(out))
        folder.mkdir(parents=True)
        victim = tmp_path.parent / f"{tmp_path.name}-victim"
        os.symlink(victim, folder / (_slot(phase_dir, plan) + write_prompt.SNAPSHOT_LOCK_SUFFIX))
        code, stdout, err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 1
        assert "is a symlink" in err and write_prompt.MARKER_PATH not in stdout
        assert not victim.exists()


# ---------------------------------------------------------------------------
# Codex review 2, #5: damaged evidence is not a new slot
# ---------------------------------------------------------------------------

class TestDamagedEvidenceIsNotANewSlot:
    """A deleted, unreadable or malformed binding, or a deleted or unreadable
    state file, took the permissive branch: the same batch's baseline was
    retaken with its edits, or the next batch replaced an unrecorded one's.
    Now each refuses, naming the files and the manual way out."""

    def test_a_deleted_binding_after_a_partial_application(self, phase, capsys):
        plan, out, phase_dir = phase
        before = plan.read_bytes()
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        copy = _marker(first, write_prompt.MARKER_SNAPSHOT)
        plan.write_bytes(before + b"\nhalf of batch 0\n")
        _binding_path(_root(out), phase_dir, plan).unlink()

        code, stdout, err = _bind(phase_dir, "apply-batch-0", capsys)

        assert code == 3
        assert write_prompt.MARKER_PATH not in stdout
        flat = " ".join(err.split())
        assert "with no binding naming them" in flat and str(copy) in flat
        assert "put the right bytes back by hand; then delete those files" in flat
        assert copy.read_bytes() == before
        # The way out: the human puts the plan back and clears the slot.
        plan.write_bytes(copy.read_bytes())
        copy.unlink()
        code, stdout, _err = _bind(phase_dir, "apply-batch-0", capsys)
        assert code == 0
        assert _marker(stdout, write_prompt.MARKER_SNAPSHOT).read_bytes() == before

    @pytest.mark.parametrize("damage", [
        "not json", "not an object", "another version", "names another file", "a link",
    ])
    def test_a_damaged_binding_refuses(self, phase, tmp_path, capsys, damage):
        plan, out, phase_dir = phase
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        copy = _marker(first, write_prompt.MARKER_SNAPSHOT)
        before = copy.read_bytes()
        path = _binding_path(_root(out), phase_dir, plan)
        bound = json.loads(path.read_text(encoding="utf-8"))
        plan.write_bytes(plan.read_bytes() + b"\nhalf of batch 0\n")
        if damage == "not json":
            path.write_text("{ not json", encoding="utf-8")
        elif damage == "not an object":
            path.write_text("[]", encoding="utf-8")
        elif damage == "another version":
            path.write_text(json.dumps({**bound, "version": 1}), encoding="utf-8")
        elif damage == "names another file":
            path.write_text(json.dumps({**bound, "artifact": "/elsewhere/feature.md"}),
                            encoding="utf-8")
        else:
            elsewhere = tmp_path.parent / f"{tmp_path.name}-binding.json"
            elsewhere.write_text(json.dumps(bound), encoding="utf-8")
            path.unlink()
            os.symlink(elsewhere, path)

        code, stdout, err = _bind(phase_dir, "apply-batch-0", capsys)

        assert code == 3
        assert write_prompt.MARKER_PATH not in stdout
        assert "nothing says whose pre-batch baseline the slot holds" in " ".join(err.split())
        assert copy.read_bytes() == before

    @pytest.mark.parametrize("damage", ["deleted", "not json", "not a state file"])
    def test_a_damaged_state_file_never_lets_the_next_batch_in(
        self, phase, capsys, damage
    ):
        plan, out, phase_dir = phase
        before = plan.read_bytes()
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        plan.write_bytes(before + b"\nhalf of batch 0\n")
        state = out / "state.json"
        if damage == "deleted":
            state.unlink()
        elif damage == "not json":
            state.write_text("{", encoding="utf-8")
        else:
            state.write_text(json.dumps({"task_status": []}), encoding="utf-8")

        code, stdout, err = _bind(phase_dir, "apply-batch-1", capsys)

        assert code == 3
        assert write_prompt.MARKER_PATH not in stdout
        assert "cannot be told" in err
        assert _marker(first, write_prompt.MARKER_SNAPSHOT).read_bytes() == before

    def test_a_deleted_state_file_keeps_a_later_bound_batch_s_baseline(self, phase, capsys):
        """Batch 1 bound (batch 0 recorded) and half applied; with the state
        file gone, binding batch 0 again cannot tell that batch 1 finished."""
        plan, out, phase_dir = phase
        _bind(phase_dir, "apply-batch-0", capsys)
        _record(phase_dir, "apply-batch-0", capsys)
        _code, second, _err = _bind(phase_dir, "apply-batch-1", capsys)
        copy = _marker(second, write_prompt.MARKER_SNAPSHOT)
        before = copy.read_bytes()
        plan.write_bytes(before + b"\nhalf of batch 1\n")
        (out / "state.json").unlink()

        code, _stdout, err = _bind(phase_dir, "apply-batch-0", capsys)

        assert code == 3
        assert "whether apply-batch-1 has its `util batch-done` record cannot be told" in (
            " ".join(err.split()))
        assert copy.read_bytes() == before
        assert _binding(_root(out), phase_dir, plan)["task_id"] == "apply-batch-1"


# ---------------------------------------------------------------------------
# Codex review 2, #6: the batches go in order
# ---------------------------------------------------------------------------

class TestTheBatchesGoInOrder:
    """Binding batch 1 first exited 0 telling the host to finish batch 0;
    binding batch 0 then exited 3 telling it to finish batch 1, and the two
    loops could cycle forever. A batch is now refused before its slot is
    touched while an earlier batch of its run has no record."""

    def test_a_later_batch_is_refused_before_the_slot_is_touched(self, phase, capsys):
        plan, out, phase_dir = phase
        code, stdout, err = _bind(phase_dir, "apply-batch-1", capsys)
        assert code == 3 and stdout == ""
        assert "apply-batch-0, an earlier batch of this run, has no `util batch-done` " \
               "record from this run" in " ".join(err.split())
        assert not (_root(out) / ".multi-llm").exists()
        assert not (phase_dir / "prompts" / "apply-batch-1.prompt.txt").exists()

    def test_the_host_following_each_message_never_cycles(self, phase, capsys):
        """The host does what each refusal says; every step moves forward."""
        plan, out, phase_dir = phase
        # Out of order: told to run apply-batch-0 first.
        code, _stdout, err = _bind(phase_dir, "apply-batch-1", capsys)
        assert code == 3 and "Run apply-batch-0 first" in " ".join(err.split())
        # It does: batch 0 binds.
        assert _bind(phase_dir, "apply-batch-0", capsys)[0] == 0
        plan.write_bytes(plan.read_bytes() + b"\nbatch 0 landed\n")
        # Too early for batch 1 again: the same batch is named, not batch 1.
        code, _stdout, err = _bind(phase_dir, "apply-batch-1", capsys)
        assert code == 3 and "Run apply-batch-0 first" in " ".join(err.split())
        # Batch 0 again (a re-dispatch) keeps its baseline.
        assert _bind(phase_dir, "apply-batch-0", capsys)[0] == 0
        # Recorded: batch 1 binds.
        _record(phase_dir, "apply-batch-0", capsys)
        code, stdout, _err = _bind(phase_dir, "apply-batch-1", capsys)
        assert code == 0
        assert _binding(_root(out), phase_dir, plan)["task_id"] == "apply-batch-1"

    def test_an_earlier_run_s_record_under_the_same_id_does_not_count(
        self, plan_dir, capsys
    ):
        """A `--resume` numbers its batches afresh: `apply-batch-0` of this run
        is not the one an earlier run recorded, and only a record made since
        this emit (its `native_run.emitted_at`) lets batch 1 through."""
        plan, out = plan_dir
        phase_dir = out / "apply-suggestions"
        phase_dir.mkdir()
        tasks = materialize_write_tasks(
            kind="apply_suggestions", batches=_suggestion_batches(),
            phase_dir=str(phase_dir), plan_file=str(plan))
        write_task_file_for_phase(
            kind="apply_suggestions", tasks=tasks, phase_dir=str(phase_dir),
            plan_file=str(plan), native_host="claude-code",
            native_run=native_run_record(str(phase_dir), tasks, resume=False))
        earlier = (datetime.now() - timedelta(hours=1)).isoformat()
        _seed_state(
            out, plan, task_status={"apply-batch-0": "completed"},
            **{"processed_apply-suggestions": {
                item: {"status": "applied", "timestamp": earlier,
                       "details": {"task_id": "apply-batch-0"}}
                for item in ("g-email", "g-password")}})

        code, _stdout, err = _bind(phase_dir, "apply-batch-1", capsys)
        # A record with no stamp at all: its own message, both ways named.
        flat = " ".join(err.split())
        assert code == 3 and "carries no run stamp" in flat
        assert "If this loop has not run apply-batch-0, run it first" in flat

        assert _bind(phase_dir, "apply-batch-0", capsys)[0] == 0
        _record(phase_dir, "apply-batch-0", capsys)
        assert _bind(phase_dir, "apply-batch-1", capsys)[0] == 0

    def test_an_earlier_run_s_stamped_record_keeps_the_general_message(
        self, phase, capsys
    ):
        """A record stamped by another run (a `--resume` renumbered its
        batches): the batch has not run in this one, so run it first."""
        plan, out, phase_dir = phase
        state = json.loads((out / "state.json").read_text(encoding="utf-8"))
        state["task_status"] = {"apply-batch-0": "completed"}
        state[write_prompt.SNAPSHOT_RECORDS_KEY] = {"apply-suggestions": {
            "apply-batch-0": {write_prompt.RECORD_RUN_KEY: "f" * 64}}}
        (out / "state.json").write_text(json.dumps(state), encoding="utf-8")
        code, _stdout, err = _bind(phase_dir, "apply-batch-1", capsys)
        flat = " ".join(err.split())
        assert code == 3 and "no `util batch-done` record from this run" in flat
        assert "carries no run stamp" not in flat

    def test_a_loop_begun_before_records_were_stamped_is_sent_to_resume(
        self, phase, capsys,
    ):
        """A native loop begun on a plugin version whose `util batch-done`
        kept no stamp: its next batch was refused with advice to record the
        recorded batch again (Fable review 2026-11-10, F10). The message
        names `--resume`, which emits what is left as a new run."""
        plan, out, phase_dir = phase
        # The loop as the older version left it: batch 0 bound, applied and
        # recorded, with no stamp and no binding of the slot.
        plan.write_bytes(plan.read_bytes() + b"\nbatch 0 landed\n")
        state = json.loads((out / "state.json").read_text(encoding="utf-8"))
        task = json.loads((phase_dir / WRITE_TASK_FILENAME).read_text(encoding="utf-8"))
        batch0 = next(t for t in task.get("batches") or task["tasks"] if t["task_id"] == "apply-batch-0")
        state["task_status"] = {"apply-batch-0": "completed"}
        state["processed_apply-suggestions"] = {
            item: {"status": "applied", "timestamp": datetime.now().isoformat(),
                   "details": {"task_id": "apply-batch-0", "leg": "native"}}
            for item in batch0["expected_ids"]}
        (out / "state.json").write_text(json.dumps(state), encoding="utf-8")

        code, stdout, err = _bind(phase_dir, "apply-batch-1", capsys)
        flat = " ".join(err.split())
        assert code == 3 and stdout == ""
        assert "carries no run stamp" in flat
        assert "re-run the phase's orchestrator with --resume" in flat
        assert "record it with util batch-done" not in flat
        assert not (_root(out) / ".multi-llm").exists()

    def test_the_loops_name_the_stampless_record_s_way_forward(self):
        for core in ("apply-suggestions", "apply-task-suggestions"):
            flat = _flat(core)
            assert ("an earlier batch whose record carries no run stamp (an older "
                    "plugin version made it)") in flat
            assert "re-run the phase's orchestrator with `--resume`" in flat


# ---------------------------------------------------------------------------
# Codex review 3: identity, not clocks; a lock that cannot be taken
# ---------------------------------------------------------------------------

def _native_phase(plan, out):
    """An apply-suggestions phase as a native emit writes it (`native_run`)."""
    phase_dir = out / "apply-suggestions"
    phase_dir.mkdir()
    tasks = materialize_write_tasks(
        kind="apply_suggestions", batches=_suggestion_batches(),
        phase_dir=str(phase_dir), plan_file=str(plan))
    write_task_file_for_phase(
        kind="apply_suggestions", tasks=tasks, phase_dir=str(phase_dir),
        plan_file=str(plan), native_host="claude-code",
        native_run=native_run_record(str(phase_dir), tasks, resume=False))
    _seed_state(out, plan)
    return phase_dir


class _AnHourBack(datetime):
    """The host's clock, set back an hour (a correction, a time zone)."""

    @classmethod
    def now(cls, tz=None):
        return datetime.now(tz) - timedelta(hours=1)


class TestEveryIdentityFieldIsChecked:
    """A binding with no `phase_identity` read as a new run: binding batch 0
    again took its partial edits as the baseline and deleted the original."""

    @pytest.mark.parametrize("field", [*write_prompt.BINDING_IDENTITY, "generation"])
    def test_a_binding_missing_a_field_refuses(self, phase, capsys, field):
        plan, out, phase_dir = phase
        before = plan.read_bytes()
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        copy = _marker(first, write_prompt.MARKER_SNAPSHOT)
        path = _binding_path(_root(out), phase_dir, plan)
        bound = json.loads(path.read_text(encoding="utf-8"))
        assert field in bound
        del bound[field]
        path.write_text(json.dumps(bound), encoding="utf-8")
        plan.write_bytes(before + b"\nhalf of batch 0\n")

        code, stdout, err = _bind(phase_dir, "apply-batch-0", capsys)

        assert code == 3
        assert write_prompt.MARKER_PATH not in stdout
        flat = " ".join(err.split())
        assert f"has no {field}" in flat
        assert "nothing says whose pre-batch baseline the slot holds" in flat
        assert copy.read_bytes() == before
        assert _copies(_root(out), phase_dir, plan) == [copy.name]

    @pytest.mark.parametrize("field,value", [
        ("phase_identity", "123:456"),
        ("phase_identity", [1]),
        ("phase_identity", [True, 2]),
        ("expected_ids", ["g-email", 7]),
        ("run", "not a digest"),
        ("generation", "0" * 12),
    ])
    def test_a_binding_field_of_the_wrong_shape_refuses(self, phase, capsys, field, value):
        plan, out, phase_dir = phase
        before = plan.read_bytes()
        _code, first, _err = _bind(phase_dir, "apply-batch-0", capsys)
        path = _binding_path(_root(out), phase_dir, plan)
        path.write_text(json.dumps({**json.loads(path.read_text()), field: value}))
        plan.write_bytes(before + b"\nhalf of batch 0\n")

        code, _stdout, err = _bind(phase_dir, "apply-batch-0", capsys)

        assert code == 3 and f"has no {field}" in err
        assert _marker(first, write_prompt.MARKER_SNAPSHOT).read_bytes() == before


class TestARecordIsTiedToItsRunNotItsTime:
    """Completion was decided by comparing local wall-clock stamps: a clock
    set back between binding and `util batch-done` hid the record. The
    record now names the run and the binding's generation."""

    @pytest.mark.parametrize("emit", ["native", "plain"])
    def test_a_clock_set_back_an_hour_does_not_hide_the_record(
        self, plan_dir, capsys, monkeypatch, emit
    ):
        plan, out = plan_dir
        if emit == "native":
            phase_dir = _native_phase(plan, out)
        else:
            phase_dir, _tasks = _materialize(
                plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
            _seed_state(out, plan)
        assert _bind(phase_dir, "apply-batch-0", capsys)[0] == 0
        generation = _binding(_root(out), phase_dir, plan)["generation"]
        edited = plan.read_bytes() + b"\nbatch 0 landed\n"
        plan.write_bytes(edited)
        from utils import state_manager
        for module in (state_manager, batch_done, write_prompt):
            monkeypatch.setattr(module, "datetime", _AnHourBack)

        _record(phase_dir, "apply-batch-0", capsys)
        code, stdout, err = _bind(phase_dir, "apply-batch-1", capsys)

        assert code == 0, err
        assert _marker(stdout, write_prompt.MARKER_SNAPSHOT).read_bytes() == edited
        state = json.loads((out / "state.json").read_text(encoding="utf-8"))
        run = write_prompt._sha256((phase_dir / WRITE_TASK_FILENAME).read_bytes())
        assert state[write_prompt.SNAPSHOT_RECORDS_KEY]["apply-suggestions"][
            "apply-batch-0"] == {"run": run, "snapshot_generation": generation}
        # The item records keep the one shape both legs write.
        assert "run" not in state["processed_apply-suggestions"]["g-email"]["details"]

    def test_a_record_made_for_an_earlier_snapshot_of_the_batch_does_not_count(
        self, phase, capsys
    ):
        """Batch 0 bound again after batch 1 finished gets a new snapshot; its
        old record was not made for it, so batch 1 waits for a new one."""
        plan, out, phase_dir = phase
        _bind(phase_dir, "apply-batch-0", capsys)
        _record(phase_dir, "apply-batch-0", capsys)
        _bind(phase_dir, "apply-batch-1", capsys)
        _record(phase_dir, "apply-batch-1", capsys)
        assert _bind(phase_dir, "apply-batch-0", capsys)[0] == 0
        plan.write_bytes(plan.read_bytes() + b"\nbatch 0, dispatched again, half\n")

        code, _stdout, err = _bind(phase_dir, "apply-batch-1", capsys)
        assert code == 3
        assert "apply-batch-0 has no `util batch-done` record made for this snapshot" in (
            " ".join(err.split()))

        _record(phase_dir, "apply-batch-0", capsys)
        assert _bind(phase_dir, "apply-batch-1", capsys)[0] == 0

    def test_other_write_kinds_get_no_stamp(self, plan_dir):
        plan, out = plan_dir
        phase_dir, tasks = _materialize(plan, out, "apply_code_fixes", [{
            "file_key": "src/a.py", "prompt": PROMPT,
            "fixes": [{"group_id": "g-fix", "title": "Fix it"}],
        }], "apply-fixes")
        assert write_prompt.completion_stamp(phase_dir, "apply_code_fixes", tasks[0]) == {}


class TestALockThatCannotBeTakenIsNotContention:
    """`EOPNOTSUPP` read as "another call holds it" and the host was told to
    retry, which can never help."""

    def test_an_unsupported_lock_names_the_cause_and_is_not_retry_advice(
        self, phase, capsys, monkeypatch
    ):
        import errno
        plan, out, phase_dir = phase
        monkeypatch.setattr(write_prompt, "SNAPSHOT_LOCK_TIMEOUT_S", 0.5)

        def unsupported(_fd):
            raise OSError(errno.EOPNOTSUPP, os.strerror(errno.EOPNOTSUPP))

        monkeypatch.setattr(process_control, "_take_fd_lock", unsupported)
        code, stdout, err = _bind(phase_dir, "apply-batch-0", capsys)

        assert code == 3
        assert write_prompt.MARKER_PATH not in stdout
        flat = " ".join(err.split())
        assert "Another `util write-prompt` holds" not in flat
        assert flat.startswith("ERROR: Cannot lock ")
        assert f"{os.strerror(errno.EOPNOTSUPP)} ({errno.errorcode[errno.EOPNOTSUPP]})" in flat
        assert "running this command again gives the same answer" in flat
        assert not (phase_dir / "prompts" / "apply-batch-0.prompt.txt").exists()

    def test_the_wait_raises_a_permanent_error_and_waits_out_contention(
        self, tmp_path, monkeypatch
    ):
        import errno
        fd = os.open(tmp_path / "slot.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            for code in (errno.EOPNOTSUPP, errno.ENOLCK):
                monkeypatch.setattr(
                    process_control, "_take_fd_lock",
                    lambda _fd, code=code: (_ for _ in ()).throw(OSError(code, "no")))
                with pytest.raises(OSError) as raised:
                    process_control.wait_for_fd_lock(fd, timeout_s=30)
                assert raised.value.errno == code
            monkeypatch.setattr(
                process_control, "_take_fd_lock",
                lambda _fd: (_ for _ in ()).throw(OSError(errno.EWOULDBLOCK, "held")))
            assert process_control.wait_for_fd_lock(fd, timeout_s=0.05) is False
        finally:
            os.close(fd)

    def test_file_mutex_never_raises_on_a_lock_that_cannot_be_taken(
        self, tmp_path, monkeypatch
    ):
        import errno
        monkeypatch.setattr(
            process_control, "_take_fd_lock",
            lambda _fd: (_ for _ in ()).throw(OSError(errno.EOPNOTSUPP, "no")))
        mutex = process_control.FileMutex(str(tmp_path / "m.lock"))
        assert mutex.acquire(timeout_s=30) is False
        assert mutex.held is False

    def test_file_mutex_held_follows_acquire_and_release(self, tmp_path):
        mutex = process_control.FileMutex(str(tmp_path / "m.lock"))
        assert mutex.held is False
        assert mutex.acquire(timeout_s=1)
        assert mutex.held is True
        mutex.release()
        assert mutex.held is False


# ---------------------------------------------------------------------------
# Codex review 2, #8: two plans of one name
# ---------------------------------------------------------------------------

def test_two_plans_of_one_name_keep_their_own_baselines(tmp_path, capsys):
    """`plans/a/feature.md` and `plans/b/feature.md` shared one slot: binding
    the second destroyed the first's baseline, and rebinding the first then
    took its partial edits. The slot name carries the phase dir and the
    artifact."""
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    phases = {}
    for name in ("a", "b"):
        plan = tmp_path / "plans" / name / "feature.md"
        plan.parent.mkdir(parents=True)
        plan.write_text(f"# Feature {name}\n", encoding="utf-8")
        out = plan.parent / "feature"
        out.mkdir()
        phase_dir, _tasks = _materialize(
            plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
        _seed_state(out, plan)
        phases[name] = (plan, phase_dir)
    plan_a, phase_a = phases["a"]
    before_a = plan_a.read_bytes()
    _code, first, _err = _bind(phase_a, "apply-batch-0", capsys)
    plan_a.write_bytes(before_a + b"\nhalf of a's batch 0\n")

    code, other, _err = _bind(phases["b"][1], "apply-batch-0", capsys)

    assert code == 0
    assert _marker(other, write_prompt.MARKER_SNAPSHOT) != _marker(
        first, write_prompt.MARKER_SNAPSHOT)
    assert _marker(first, write_prompt.MARKER_SNAPSHOT).read_bytes() == before_a
    code, again, _err = _bind(phase_a, "apply-batch-0", capsys)
    assert code == 0
    assert _marker(again, write_prompt.MARKER_SNAPSHOT).read_bytes() == before_a


# ---------------------------------------------------------------------------
# The instructions
# ---------------------------------------------------------------------------

def _flat(core):
    return " ".join((INSTRUCTIONS / f"{core}.md").read_text(encoding="utf-8").split())


@pytest.mark.parametrize("core,role", [
    ("apply-suggestions", "plan"), ("apply-task-suggestions", "tasks"),
])
def test_the_shared_artifact_loops_read_the_snapshot_the_command_took(core, role):
    """Step 2 is a note, not a copy: no root lookup, no Read/Write of the
    artifact, no shell variables; step 5 diffs the two printed paths."""
    text = (INSTRUCTIONS / f"{core}.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    for stale in (
        "Capture the snapshot with your host's file tools",
        "PROJECT_ROOT=", "BATCH_SNAPSHOT", "TASKS_PATH", "{plan_abs}",
        "Resolve the project root first",
        f"batch_snapshot_{role}_{{plan_stem}}.md",
    ):
        assert stale not in text, (core, stale)
    assert "2. **Note the pre-batch snapshot**" in text
    assert "are the `[WRITE_SNAPSHOT]` and `[WRITE_ARTIFACT]` paths from step 1" in flat
    assert ("run step 1's command again for this same batch: it keeps this batch's "
            "snapshot and prints the same two lines") in flat


def _partial_handling(core):
    flat = _flat(core)
    start = flat.index("**Partial application handling**")
    return flat[start:flat.index("A resume that does neither", start)]


def _fresh(core):
    flat = _flat(core)
    start = flat.index("- `--fresh`:")
    ends = [flat.find(mark, start + 1) for mark in (" - `--", " ### ")]
    return flat[start:min(end for end in ends if end > 0)]


@pytest.mark.parametrize("core", [
    "apply-suggestions", "apply-task-suggestions", "apply-code-fixes",
])
def test_no_native_loop_advises_a_restore_that_does_not_exist(core):
    """U66, with the native restore cut from U31: `util restore-batch` reads
    only the executor's rollback store, so the native loops' partial
    application and `--fresh` advice name the manual way back, and promise
    no rollback a native run lacks."""
    handling = _partial_handling(core)
    assert "util restore-batch" not in handling.replace(
        "`util restore-batch` has nothing", "")
    fresh = _fresh(core)
    # `util restore-batch` only for an `--exec-subagents` run.
    assert "after an `--exec-subagents` run" in fresh
    assert "native loop" in fresh
    if core == "apply-code-fixes":
        assert "backup" not in fresh
        assert "this loop takes no snapshot" in handling
    else:
        assert "backup.md" in fresh
        assert "copy it back by hand" in handling
        assert "not tamper-proof" in handling


def test_the_code_fix_loop_claims_no_snapshot():
    flat = _flat("apply-code-fixes")
    assert "every batch was already verified against its own pre-batch snapshot" not in flat
    assert "each batch was verified when it finished (step 4)" in flat


@pytest.mark.parametrize("core", ["apply-suggestions", "apply-task-suggestions"])
def test_the_loops_say_which_write_prompt_refusals_to_retry(core):
    """Codex review 3 of U31, #3: a lock the file system cannot take was
    reported as another call holding it, and the loop retried it."""
    flat = _flat(core)
    start = flat.index("**Error handling**: an exit 3 names the problem")
    handling = flat[start:flat.index("[WRITE_PROMPT_ORDER]", start)]
    assert "a snapshot lock the file system cannot take at all (`Cannot lock …`)" in handling
    assert ("Only two exits 3 are worth running this command again for; every other "
            "one is a stop.") in handling
    assert "Another `util write-prompt` holds the snapshot's lock" in handling


def _fresh_option(help_text):
    """The `--fresh` entry of an apply orchestrator's `--help`, flattened (a
    word argparse wrapped at its hyphen joined again)."""
    import re
    flat = re.sub(r"(\w)- (\w)", r"\1-\2", " ".join(help_text.split()))
    start = flat.index("--fresh Discard")
    return flat[start:flat.index(" --human-task", start)]


@pytest.mark.parametrize("golden", [
    "apply_suggestions_help.txt", "apply_task_suggestions_help.txt",
    "apply_code_fixes_help.txt",
])
def test_the_fresh_help_names_each_path_s_way_back(golden):
    """U73: `--fresh` offered `util restore-batch` to native runs, promised a
    refusal they never get, and named backup.md for task suggestions too."""
    from utils.apply_orchestrator_base import build_common_arg_parser
    live = _fresh_option(build_common_arg_parser("d", "e").format_help())
    assert "After an --exec-subagents run, `util restore-batch` can roll" in live
    assert "it holds nothing of a native run's" in live
    assert "for apply-suggestions copy the plan back from {phase}/backup.md" in live
    assert "for apply-task-suggestions copy tasks.md back from {phase}/tasks-backup.md" in live
    assert "apply-code-fixes keeps no backup" in live
    assert ("Only after an --exec-subagents run of the two suggestion phases is a target "
            "still carrying that run's edits refused (exit 3)") in live
    # The checked-in --help fixture says the same.
    fixture = (Path(__file__).parent / "golden" / golden).read_text(encoding="utf-8")
    assert _fresh_option(fixture) == live
