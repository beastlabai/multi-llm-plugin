"""The commit apply guard (plans/review-commits.md §2.7, T6a).

apply-code-fixes on a commit review (a commit brief's ``commit_range``, or a
plan's ``head_ref`` after a direct ``--head-ref`` review) decides, before any
batch, which validated fixes may still land: a group whose fix files changed
since the reviewed ``head`` is skipped and named, a group naming the context
plan is skipped as read-only, the base ref is the recorded one, and the
decision is saved as expectations, never approvals. ``status`` renders the
``[COMMIT_FIXES]`` block from that saved decision. ``select_review`` is the
no-argument ``--apply-commit-fixes`` lookup.

Every fixture is a throwaway repository under ``tmp_path``; the review is
seeded by hand (``state.json`` with ``commit_range``/``head_ref`` and a
completed code-review phase, ``code-review/grouped.json`` and
``validation.json``), the shapes apply-code-fixes reads. No model runs: the
resume tests drive the write executor over the suite's mock codex, with a
scratch plugin standing in for the batches' edits.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import apply_code_fixes_orchestrator as acf  # noqa: E402
from utils import cli_common, commit_apply, run_status, status_file  # noqa: E402
from utils.code_fix_batcher import (  # noqa: E402
    PROTECTED_CLAUSE_OPEN,
    code_only_rule,
    without_protected_clause,
)
from utils.commit_apply import (  # noqa: E402
    CommitReviewError,
    mark_stale,
    protected_violations,
    recheck_files,
    repin_after_batch,
    require_completed_review,
    save_guard,
    select_review,
    take_guard,
)
from utils.commit_range import DATA_UNDER_REVIEW, RangeError, pinned_head_read_rule  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import compute_plan_hash, stamp_stable_ids  # noqa: E402
from utils.write_safety import file_sha256  # noqa: E402

FOLDER = "plans/commit-reviews"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(root), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _commit(root: Path, message: str, files: Optional[Dict[str, Optional[str]]] = None) -> str:
    for name, text in (files or {}).items():
        if text is None:
            _git(root, "rm", "-q", name)
        else:
            _write(root / name, text)
            _git(root, "add", "--", name)
    _git(root, "commit", "-q", "--allow-empty", "-m", message)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "master")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    monkeypatch.chdir(root)
    return root


def _suggestion(file: str, title: str, *, line=(1, 2), extra: str = "") -> Dict[str, Any]:
    return {
        "title": title, "desc": f"Fix {title} {extra}".strip(), "file": file,
        "line_range": list(line), "anchor_text": "x", "importance": "MEDIUM",
        "type": "bug", "model": "m1",
    }


def _group(theme: str, *files: str, introduced_by: Optional[str] = None) -> Dict[str, Any]:
    group: Dict[str, Any] = {
        "theme": theme,
        "suggestions": [_suggestion(f, f"{theme} {i}") for i, f in enumerate(files)],
        "models": ["m1"],
    }
    if introduced_by:
        group["introduced_by"] = introduced_by
    return group


def _hashes(groups: Sequence[Dict[str, Any]]) -> List[str]:
    stamped = copy.deepcopy(list(groups))
    stamp_stable_ids(stamped)
    return [g["group_hash"] for g in stamped]


def _commits_record(root: Path, base: Optional[str], head: str) -> List[Dict[str, Any]]:
    revs = f"{base}..{head}" if base else head
    try:
        out = _git(root, "log", "--first-parent", "--format=%H%x09%cs%x09%s", revs)
    except subprocess.CalledProcessError:
        return []  # a recorded SHA that no longer names a commit
    records = []
    for line in out.splitlines():
        sha, date, subject = line.split("\t", 2)
        records.append({"sha": sha, "date": date, "subject": subject, "merge": False,
                        "outside_window": False})
    return records


def _brief_path(root: Path, base: Optional[str], head: str, folder: str = FOLDER) -> Path:
    slug = f"{base[:12] if base else 'root'}-{head[:12]}"
    return root / folder / f"{slug}.md"


def _seed(
    root: Path,
    groups: Sequence[Dict[str, Any]],
    *,
    base: Optional[str],
    head: str,
    plan: Optional[Path] = None,
    key: str = "commit_range",
    context_plan: Optional[str] = None,
    completed: Optional[str] = "2026-10-26T10:00:00",
    resolved_at: str = "2026-10-26T09:00:00Z",
    statuses: Optional[Sequence[str]] = None,
    folder: str = FOLDER,
) -> Path:
    """Seed a completed review of ``base..head``; returns the plan/brief path."""
    if plan is None:
        plan = _brief_path(root, base, head, folder)
        ignore = root / folder / ".gitignore"
        if not ignore.exists():
            _write(ignore, "*\n")
        _write(plan, "<!-- multi-llm:commit-brief v1 -->\n# Commit brief\n")
    out = output_dir_path(plan)
    review = out / "code-review"
    review.mkdir(parents=True, exist_ok=True)
    (review / "grouped.json").write_text(
        json.dumps({"format_version": "2.0", "groups": list(groups)}), encoding="utf-8")
    ids = _hashes(groups)
    statuses = list(statuses or ["valid"] * len(groups))
    (review / "validation.json").write_text(json.dumps([
        {"group_id": gid, "status": status, "reason": "seeded", "confidence": 0.9}
        for gid, status in zip(ids, statuses)
    ]), encoding="utf-8")
    record: Dict[str, Any] = {"base": base, "head": head, "drifted": [],
                              "resolved_at": resolved_at}
    if key == "commit_range":
        record.update({
            "commits": _commits_record(root, base, head), "phrase": None, "window": None,
            "notes": [], "brief": str(plan.relative_to(root)), "plan": context_plan,
            "plan_snapshot": None, "plan_sha256": None, "focus": None,
            "header_sha256": "h", "reviewed_header_sha256": "h",
        })
    state: Dict[str, Any] = {
        "schema_version": "2.0", "plan_path": str(plan.resolve()),
        "plan_hash": compute_plan_hash(plan.resolve()), "phases_completed": {},
        "created_at": "2026-10-26T08:00:00", "updated_at": "2026-10-26T08:00:00",
        "head_at_start": head, "branch_name": "master", "review_phase_completed": False,
        "tracked_files": [], "task_status": {}, "phases_skipped": {},
        key: record,
    }
    if completed:
        state["phases_completed"]["code-review"] = completed
    (out / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return plan


def _state(plan: Path) -> Dict[str, Any]:
    return json.loads((output_dir_path(plan) / "state.json").read_text(encoding="utf-8"))


def _guard(plan: Path, key: str = "commit_range") -> Dict[str, Any]:
    return _state(plan)[key]["guard"]


def _run_apply(plan: Path, *flags: str):
    args = acf.ApplyCodeFixesOrchestrator.parse_args(
        ["--plan-file", str(plan), "--no-confirm", *flags])
    orch = acf.ApplyCodeFixesOrchestrator(args)
    try:
        rc = orch.run()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 0
    return rc, orch


def _output(plan: Path) -> Dict[str, Any]:
    path = output_dir_path(plan) / "apply-fixes" / "orchestrator_output.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _prompts(plan: Path) -> str:
    return "\n".join(b["prompt"] for b in _output(plan)["batches"])


def _block(out: str) -> List[str]:
    lines = out.splitlines()
    start = [i for i, line in enumerate(lines) if line.startswith("[COMMIT_FIXES] ")]
    assert len(start) == 1, out
    end = lines.index("[COMMIT_FIXES_END]", start[0])
    return lines[start[0]:end + 1]


@pytest.fixture
def ranged(repo):
    """``base..head`` touching a.py, b.py and c.py in two commits."""
    base = _commit(repo, "init", {"a.py": "a = 1\n", "b.py": "b = 1\n", "c.py": "c = 1\n",
                                  "README.md": "# r\n"})
    first = _commit(repo, "touch a and b", {"a.py": "a = 2\n", "b.py": "b = 2\n"})
    head = _commit(repo, "touch c", {"c.py": "c = 2\n"})
    return repo, base, first, head


# ---------------------------------------------------------------------------
# The guard through a plain apply-code-fixes <brief>
# ---------------------------------------------------------------------------


class TestTheGuardOnAPlainApply:

    def test_a_drifted_file_is_skipped_and_named(self, ranged, capsys):
        repo, base, first, head = ranged
        groups = [_group("A", "a.py", introduced_by=first),
                  _group("B", "b.py", introduced_by=first),
                  _group("C", "c.py", introduced_by=head)]
        plan = _seed(repo, groups, base=base, head=head)
        (repo / "b.py").write_text("b = 99\n", encoding="utf-8")
        rc, _ = _run_apply(plan)
        out = capsys.readouterr().out
        assert rc == 0
        block = _block(out)
        assert block[0] == "[COMMIT_FIXES] 2 fixes to apply"
        assert any(line.startswith(f"  {first[:7]}  1 fix") for line in block)
        assert any(line.startswith(f"  {head[:7]}  1 fix") for line in block)
        assert f"  skipped  1 (file changed since {head[:7]}: b.py)" in block
        output = _output(plan)
        assert sorted(f["file"] for f in output["to_apply"]) == ["a.py", "c.py"]
        skipped = [s for s in output["skipped_items"] if s["status"] == "commit_guard"]
        assert len(skipped) == 1 and skipped[0]["files"] == ["b.py"]
        assert skipped[0]["reason"] == f"file changed since reviewed commit {head[:7]}"
        guard = _guard(plan)
        b_id = _hashes(groups)[1]
        assert guard["skipped"] == {b_id: {"reason": "changed", "files": ["b.py"]}}
        assert guard["head"] == head
        # The drifted file is protected at its disk content, with a copy.
        assert "b.py" in guard["protected"]
        copies = repo / guard["protected_copies"]
        assert (copies / "b.py").read_text(encoding="utf-8") == "b = 99\n"

    def test_the_base_ref_is_the_recorded_sha_after_head_moved(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        _commit(repo, "later", {"README.md": "# later\n"})
        rc, _ = _run_apply(plan)
        capsys.readouterr()
        assert rc == 0
        output = _output(plan)
        assert output["base_ref"] == base
        prompts = _prompts(plan)
        assert f"git diff {base} -- a.py" in prompts
        assert "HEAD~1" not in prompts
        assert DATA_UNDER_REVIEW in prompts

    def test_the_block_is_printed_exactly_once(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        context = cli_common.activate_result(cli_common.ResultContext())
        try:
            _run_apply(plan)
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        assert out.count("[COMMIT_FIXES] ") == 1
        assert out.count("[COMMIT_FIXES_END]") == 1
        # The result line carries the same decision.
        assert context.commit_fixes["lines"] == _block(out)
        assert context.payload(0)["commit_fixes"]["to_apply"] == 1

    def test_an_all_skipped_run_still_prints_the_block_once(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py"), _group("B", "b.py")], base=base, head=head)
        (repo / "a.py").write_text("x\n", encoding="utf-8")
        (repo / "b.py").write_text("y\n", encoding="utf-8")
        rc, _ = _run_apply(plan)
        out = capsys.readouterr().out
        assert rc == 0
        assert out.count("[COMMIT_FIXES] ") == 1
        block = _block(out)
        assert block[0] == "[COMMIT_FIXES] 0 fixes to apply"
        assert f"  skipped  2 (file changed since {head[:7]}: a.py, b.py)" in block
        state = _state(plan)
        assert state["phases_completed"].get("apply-code-fixes")
        assert len(state["commit_range"]["guard"]["skipped"]) == 2

    def test_a_detach_launcher_leaves_the_decision_to_its_worker(
        self, ranged, capsys, monkeypatch
    ):
        from utils import review_orchestrator_base

        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        monkeypatch.setattr(review_orchestrator_base, "maybe_detach", lambda **kw: 0)
        rc, _ = _run_apply(plan, "--detach")
        assert rc == 0
        assert "[COMMIT_FIXES]" not in capsys.readouterr().out
        assert "guard" not in _state(plan)["commit_range"]

    def test_a_dry_run_prints_the_block_and_saves_nothing(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        _write(repo / "untracked.txt", "u\n")
        _run_apply(plan, "--dry-run")
        assert _block(capsys.readouterr().out)[0] == "[COMMIT_FIXES] 1 fix to apply"
        assert "guard" not in _state(plan)["commit_range"]
        assert not (output_dir_path(plan) / "apply-fixes" / "protected").exists()

    def test_a_plan_without_pinned_endpoints_prints_no_block(self, repo, capsys):
        _commit(repo, "init", {"a.py": "a = 1\n"})
        plan = _write(repo / "plans" / "p.md", "# Plan\n")
        out_dir = output_dir_path(plan)
        groups = [_group("A", "a.py")]
        (out_dir / "code-review").mkdir(parents=True)
        (out_dir / "code-review" / "grouped.json").write_text(json.dumps(groups))
        (out_dir / "code-review" / "validation.json").write_text(json.dumps(
            [{"group_id": _hashes(groups)[0], "status": "valid", "reason": "r"}]))
        rc, _ = _run_apply(plan, "--base-ref", "HEAD")
        out = capsys.readouterr().out
        assert rc == 0
        assert "[COMMIT_FIXES]" not in out
        assert "files" not in _output(plan)["to_apply"][0]
        assert DATA_UNDER_REVIEW not in _prompts(plan)


# ---------------------------------------------------------------------------
# The context plan is read-only
# ---------------------------------------------------------------------------


class TestTheContextPlan:

    @pytest.mark.parametrize("changed_by_range", [True, False])
    def test_a_group_naming_it_is_skipped_before_any_batch(self, repo, capsys, changed_by_range):
        base = _commit(repo, "init", {"a.py": "a = 1\n", "docs/x.md": "# x\n"})
        head = _commit(repo, "work", {"a.py": "a = 2\n",
                                      **({"docs/x.md": "# x2\n"} if changed_by_range else {})})
        groups = [_group("Plan", "docs/x.md"), _group("A", "a.py"),
                  _group("Both", "docs/x.md", "a.py")]
        plan = _seed(repo, groups, base=base, head=head, context_plan="docs/x.md")
        # A group both rules skip is counted once, under the context plan.
        (repo / "a.py").write_text("drift\n", encoding="utf-8")
        rc, orch = _run_apply(plan)
        out = capsys.readouterr().out
        assert rc == 0
        block = _block(out)
        assert any(line.startswith("  skipped  2 (context plan, read-only: docs/x.md)")
                   for line in block), block
        assert f"  skipped  1 (file changed since {head[:7]}: a.py)" in block
        guard = _guard(plan)
        ids = _hashes(groups)
        assert guard["skipped"][ids[0]] == {"reason": "context_plan", "files": ["docs/x.md"]}
        assert guard["skipped"][ids[2]]["reason"] == "context_plan"
        assert guard["context_plan"] == "docs/x.md"
        summary = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "context plan docs/x.md is read-only in this review" in summary
        assert "**Files**: `docs/x.md`" in summary

    def test_the_fix_prompts_name_it_as_a_file_not_to_edit(self, repo, capsys):
        base = _commit(repo, "init", {"a.py": "a = 1\n", "docs/x.md": "# x\n"})
        head = _commit(repo, "work", {"a.py": "a = 2\n"})
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head,
                     context_plan="docs/x.md")
        _run_apply(plan)
        capsys.readouterr()
        prompts = " ".join(_prompts(plan).split())
        assert ("or the context plan this review checks the commits against "
                "(`docs/x.md`)") in prompts
        assert "this run's plan, its task list or the context plan" in prompts

    def test_code_only_rule_with_a_context_plan(self):
        rule = code_only_rule("plans/b.md", context_plan="docs/x.md")
        assert ("Do NOT edit this run's plan (`plans/b.md`) or the task list generated "
                "from it, or the context plan this review checks the commits against "
                "(`docs/x.md`)") in rule
        assert "changing this run's plan, its task list or the context plan" in rule
        assert "not applied: needs a plan decision" in rule
        assert code_only_rule("plans/b.md") == code_only_rule("plans/b.md", context_plan=None)

    def test_without_protected_clause_takes_out_the_list_and_nothing_else(self):
        bare = code_only_rule("plans/b.md")
        for paths in (["a.py"], [f"m{i}.py" for i in range(25)]):
            rule = code_only_rule("plans/b.md", protected=paths, protected_head="abcdef0123",
                                  protected_list="plans/x/apply-fixes/protected_paths.txt")
            assert rule.startswith(bare) and PROTECTED_CLAUSE_OPEN in rule
            assert without_protected_clause(f"head\n7. {rule}\n\nReturn") == (
                f"head\n7. {bare}\n\nReturn")
        assert without_protected_clause(bare) == bare
        # A fix description quoting the opening, ahead of the rule, stays.
        quoted = f"Description:{PROTECTED_CLAUSE_OPEN} x\n"
        rule = code_only_rule("plans/b.md", protected=["a.py"])
        assert without_protected_clause(quoted + rule) == quoted + bare


# ---------------------------------------------------------------------------
# The protected set
# ---------------------------------------------------------------------------


class TestTheProtectedSet:

    def test_a_fix_whose_targets_include_a_protected_path_is_skipped(self, ranged, capsys):
        repo, base, first, head = ranged
        # README.md is outside the range and has no finding of its own.
        (repo / "README.md").write_text("# edited\n", encoding="utf-8")
        groups = [_group("A", "a.py", "README.md"), _group("C", "c.py")]
        plan = _seed(repo, groups, base=base, head=head)
        rc, _ = _run_apply(plan)
        out = capsys.readouterr().out
        assert rc == 0
        block = _block(out)
        assert f"  skipped  1 (file changed since {head[:7]}: README.md)" in block
        assert block[0] == "[COMMIT_FIXES] 1 fix to apply"
        guard = _guard(plan)
        assert "README.md" in guard["protected"]
        # Every later batch is told not to edit it.
        assert "`README.md`" in _prompts(plan)

    def test_a_protected_path_that_matches_head_is_still_skipped(self, ranged, monkeypatch):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("C", "c.py")], base=base, head=head)
        monkeypatch.setattr(commit_apply, "protected_paths",
                            lambda *a, **k: ["c.py"])
        outcome = take_guard(_state(plan), [_group("C", "c.py")], cwd=str(repo),
                             plan_path=str(plan), out_dir=str(output_dir_path(plan)),
                             phase_dir=str(output_dir_path(plan) / "apply-fixes"),
                             persist=False)
        assert list(outcome.skipped.values()) == [{"reason": "protected", "files": ["c.py"]}]
        assert outcome.payload["skipped_changed"] == 1

    def test_more_than_twenty_protected_paths_go_to_a_list_file(self, ranged, capsys):
        repo, base, first, head = ranged
        for i in range(25):
            _write(repo / "untracked" / f"u{i:02}.txt", "u\n")
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        _run_apply(plan)
        capsys.readouterr()
        listed = output_dir_path(plan) / "apply-fixes" / "protected_paths.txt"
        names = listed.read_text(encoding="utf-8").splitlines()
        assert len(names) == 25 and names[0] == "untracked/u00.txt"
        prompts = " ".join(_prompts(plan).split())
        rel = listed.relative_to(repo).as_posix()
        assert f"every one is listed in `{rel}`" in prompts
        assert "… and 5 more" in prompts
        # Workflow paths are left out of the set.
        assert not any(name.startswith(FOLDER) for name in _guard(plan)["protected"])

    def test_an_edit_behind_skip_worktree_or_assume_unchanged_is_protected(self, repo):
        # `git diff` never reads these entries' disk copies, so the set takes
        # them from `git ls-files -v`, as the change test's third read does: a
        # batch for another file that rewrites one is a violation with a copy
        # to restore. A skip-worktree entry absent on disk (a sparse checkout's
        # out-of-cone path) has no edit to protect and stays out.
        base = _commit(repo, "init", {"a.py": "a = 1\n", "sw.cfg": "sw\n", "au.cfg": "au\n",
                                      "gone.cfg": "g\n"})
        head = _commit(repo, "work", {"a.py": "a = 2\n"})
        _git(repo, "update-index", "--skip-worktree", "sw.cfg", "gone.cfg")
        _git(repo, "update-index", "--assume-unchanged", "au.cfg")
        (repo / "sw.cfg").write_text("sw local\n", encoding="utf-8")
        (repo / "au.cfg").write_text("au local\n", encoding="utf-8")
        (repo / "gone.cfg").unlink()
        assert _git(repo, "diff", "--name-only", head) == ""
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        outcome = take_guard(_state(plan), [_group("A", "a.py")], cwd=str(repo),
                             plan_path=str(plan), out_dir=str(output_dir_path(plan)),
                             phase_dir=str(output_dir_path(plan) / "apply-fixes"))
        g = outcome.guard
        assert sorted(g["protected"]) == ["au.cfg", "sw.cfg"]
        assert outcome.skipped == {}
        assert (repo / g["protected_copies"] / "sw.cfg").read_text() == "sw local\n"
        assert protected_violations(g, cwd=str(repo)) == []
        (repo / "sw.cfg").write_text("overwritten\n", encoding="utf-8")
        (repo / "au.cfg").write_text("overwritten\n", encoding="utf-8")
        assert protected_violations(g, cwd=str(repo)) == ["au.cfg", "sw.cfg"]
        report = commit_apply.restore_watched(g, ["sw.cfg"], cwd=str(repo))
        assert report == {"sw.cfg": commit_apply.RESTORED_FROM_COPY}
        assert (repo / "sw.cfg").read_text() == "sw local\n"

    def test_a_path_the_deny_list_names_is_protected_but_never_copied(self, repo):
        # The executor's rollback store never retains a `driver.write_paths.deny`
        # path, to keep secrets out of the run's artifacts; the guard's copies
        # follow the same rule. The path stays protected and watched (a batch
        # writing it still fails), with no copy as its way back, and a copy an
        # earlier run left of one is removed when it is protected again.
        base = _commit(repo, "init", {"a.py": "a = 1\n", "b.py": "b = 1\n",
                                      "keys/server.pem": "pem\n"})
        head = _commit(repo, "work", {"a.py": "a = 2\n"})
        _write(repo / ".env", "TOKEN=secret\n")
        (repo / "keys/server.pem").write_text("pem local\n", encoding="utf-8")
        (repo / "b.py").write_text("b = 9\n", encoding="utf-8")
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        phase = output_dir_path(plan) / "apply-fixes"
        outcome = take_guard(_state(plan), [_group("A", "a.py")], cwd=str(repo),
                             plan_path=str(plan), out_dir=str(output_dir_path(plan)),
                             phase_dir=str(phase))
        g = outcome.guard
        assert {".env", "b.py", "keys/server.pem"} <= set(g["protected"])
        copies = repo / g["protected_copies"]
        assert (copies / "b.py").read_text() == "b = 9\n"
        assert not (copies / ".env").exists()
        assert not (copies / "keys/server.pem").exists()
        (repo / ".env").write_text("TOKEN=overwritten\n", encoding="utf-8")
        assert ".env" in protected_violations(g, cwd=str(repo))
        assert commit_apply.restore_watched(g, [".env"], cwd=str(repo)) == {
            ".env": "skipped: the commit review kept no copy of it"}
        # A failing file the re-check protects later gets no copy either, and
        # one a run before this rule left behind goes.
        _write(repo / "c" / ".env", "OTHER=1\n")
        _write(copies / "c" / ".env", "OTHER=old\n")
        mark_stale(g, {"g9": ["c/.env"]}, ["c/.env"], cwd=str(repo))
        assert "c/.env" in g["protected"]
        assert not (copies / "c" / ".env").exists()


# ---------------------------------------------------------------------------
# Deleted files, git's view of a file, group granularity
# ---------------------------------------------------------------------------


class TestDeletedFiles:

    @pytest.fixture
    def deleted(self, repo):
        base = _commit(repo, "init", {"d.py": "d = 1\n", "keep.py": "k = 1\n"})
        deleting = _commit(repo, "drop d", {"d.py": None})
        head = _commit(repo, "more", {"keep.py": "k = 2\n"})
        return repo, base, deleting, head

    def test_still_absent_passes_and_the_prompt_names_its_history(self, deleted, capsys):
        repo, base, deleting, head = deleted
        plan = _seed(repo, [_group("D", "d.py")], base=base, head=head)
        rc, _ = _run_apply(plan)
        out = capsys.readouterr().out
        assert rc == 0
        assert _block(out)[0] == "[COMMIT_FIXES] 1 fix to apply"
        prompt = _prompts(plan)
        assert f"git show {deleting[:12]}^:d.py" in prompt
        assert "absent at the reviewed commit" in prompt and "and on disk" in prompt
        assert f"git diff {base} -- d.py" in prompt
        assert _guard(plan)["expect"]["d.py"] == {"kind": "head", "blob": None}

    def test_a_new_file_no_range_commit_deleted_says_so(self, deleted, capsys):
        repo, base, deleting, head = deleted
        plan = _seed(repo, [_group("N", "tests/test_new.py")], base=base, head=head)
        _run_apply(plan)
        capsys.readouterr()
        assert "no commit in the range deleted it" in _prompts(plan)

    def test_recreated_on_disk_is_skipped(self, deleted, capsys):
        repo, base, deleting, head = deleted
        plan = _seed(repo, [_group("D", "d.py")], base=base, head=head)
        (repo / "d.py").write_text("back\n", encoding="utf-8")
        _run_apply(plan)
        block = _block(capsys.readouterr().out)
        assert f"  skipped  1 (file changed since {head[:7]}: d.py)" in block

    def test_present_at_head_but_deleted_on_disk_is_skipped(self, deleted, capsys):
        repo, base, deleting, head = deleted
        plan = _seed(repo, [_group("K", "keep.py")], base=base, head=head)
        (repo / "keep.py").unlink()
        _run_apply(plan)
        block = _block(capsys.readouterr().out)
        assert f"  skipped  1 (file changed since {head[:7]}: keep.py)" in block

    def test_a_root_range_says_the_diff_shows_whole_files(self, repo, capsys):
        head = _commit(repo, "root", {"a.py": "a = 1\n"})
        plan = _seed(repo, [_group("A", "a.py")], base=None, head=head)
        _run_apply(plan)
        capsys.readouterr()
        empty = _git(repo, "hash-object", "-t", "tree", "/dev/null")
        prompt = _prompts(plan)
        assert f"git diff {empty} -- a.py" in prompt
        assert "starts at the repository's first commit" in prompt


class TestGitsViewOfAFile:

    def test_crlf_under_text_auto_passes(self, repo, capsys):
        base = _commit(repo, "init", {".gitattributes": "* text=auto\n", "a.py": "a\nb\n"})
        head = _commit(repo, "work", {"a.py": "a\nb\nc\n"})
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        (repo / "a.py").write_bytes(b"a\r\nb\r\nc\r\n")
        _run_apply(plan)
        assert _block(capsys.readouterr().out)[0] == "[COMMIT_FIXES] 1 fix to apply"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_chmod_alone_is_skipped(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        os.chmod(repo / "a.py", 0o755)
        _run_apply(plan)
        block = _block(capsys.readouterr().out)
        assert f"  skipped  1 (file changed since {head[:7]}: a.py)" in block

    def test_a_skip_worktree_file_is_skipped(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        _git(repo, "update-index", "--skip-worktree", "a.py")
        _run_apply(plan)
        block = _block(capsys.readouterr().out)
        assert f"  skipped  1 (file changed since {head[:7]}: a.py)" in block


class TestGroupGranularity:

    def test_one_group_on_an_unchanged_and_a_drifted_file_is_skipped_once(self, ranged, capsys):
        repo, base, first, head = ranged
        groups = [_group("AB", "a.py", "b.py"), _group("C", "c.py")]
        plan = _seed(repo, groups, base=base, head=head)
        ab = _hashes(groups)[0]
        review = output_dir_path(plan) / "code-review"
        (review / "report_data.json").write_text(json.dumps({
            "templateStyle": "pr",
            "groups": [{"groupHash": ab, "theme": "AB"}, {"groupHash": _hashes(groups)[1]}],
        }), encoding="utf-8")
        (repo / "b.py").write_text("drift\n", encoding="utf-8")
        rc, orch = _run_apply(plan)
        out = capsys.readouterr().out
        assert rc == 0
        block = _block(out)
        assert f"  skipped  1 (file changed since {head[:7]}: b.py)" in block
        summary = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert summary.count(f"**Group**: `{ab}`") == 1
        assert "**Files**: `b.py`" in summary
        assert orch.regenerate_decisions_overlay() is not None
        data = json.loads((review / "report_data.json").read_text(encoding="utf-8"))
        stamped = [g for g in data["groups"] if g.get("guardSkip")]
        assert len(stamped) == 1
        assert stamped[0]["guardSkip"] == {
            "reason": "changed", "files": ["b.py"],
            "label": f"skipped: file changed since {head[:7]}",
        }
        assert f"skipped: file changed since {head[:7]}" in (
            review / "report.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# What state saves, resume, the shared re-check functions
# ---------------------------------------------------------------------------


class TestTheSavedGuard:

    def test_expect_holds_a_head_identity_for_every_file_let_through(self, ranged, capsys):
        repo, base, first, head = ranged
        groups = [_group("A", "a.py", "c.py"), _group("N", "new_module.py"),
                  _group("B", "b.py")]
        plan = _seed(repo, groups, base=base, head=head)
        (repo / "b.py").write_text("drift\n", encoding="utf-8")
        _run_apply(plan)
        capsys.readouterr()
        guard = _guard(plan)
        assert set(guard["expect"]) == {"a.py", "c.py", "new_module.py"}
        assert guard["expect"]["a.py"] == {
            "kind": "head", "blob": _git(repo, "rev-parse", f"{head}:a.py")}
        assert guard["expect"]["new_module.py"] == {"kind": "head", "blob": None}
        text = json.dumps(guard)
        for word in ("passed", "approved", '"ok"', "pass_"):
            assert word not in text
        assert set(guard) >= {"head", "decided_at", "skipped", "expect", "protected",
                              "protected_copies", "context_plan"}

    def test_resume_reuses_the_decision_and_skips_a_file_edited_since(self, ranged, capsys):
        repo, base, first, head = ranged
        groups = [_group("A", "a.py"), _group("C", "c.py")]
        plan = _seed(repo, groups, base=base, head=head)
        _run_apply(plan)
        first_guard = _guard(plan)
        capsys.readouterr()
        (repo / "c.py").write_text("hand edit\n", encoding="utf-8")
        rc, _ = _run_apply(plan, "--resume")
        out = capsys.readouterr().out
        assert rc == 0
        guard = _guard(plan)
        assert guard["decided_at"] == first_guard["decided_at"]
        c_id = _hashes(groups)[1]
        assert guard["skipped"][c_id] == {"reason": "changed", "files": ["c.py"]}
        assert "c.py" in guard["protected"] and "c.py" not in guard["expect"]
        assert f"  skipped  1 (file changed since {head[:7]}: c.py)" in _block(out)

    def test_resume_accepts_a_file_an_earlier_batch_fixed(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        _run_apply(plan)
        capsys.readouterr()
        (repo / "a.py").write_text("a = fixed\n", encoding="utf-8")
        guard = _guard(plan)
        assert repin_after_batch(guard, "apply-fix-batch-0",
                                 {"a.py": file_sha256(str(repo / "a.py"))}, cwd=str(repo))
        save_guard(output_dir_path(plan) / "state.json", guard)
        _run_apply(plan, "--resume")
        assert _block(capsys.readouterr().out)[0] == "[COMMIT_FIXES] 1 fix to apply"
        # A hand edit after that batch fails its saved sha256.
        (repo / "a.py").write_text("a = hand\n", encoding="utf-8")
        _run_apply(plan, "--resume")
        assert _block(capsys.readouterr().out)[0] == "[COMMIT_FIXES] 0 fixes to apply"


# ---------------------------------------------------------------------------
# A resume after a hand edit: the completed batch is never applied again
# ---------------------------------------------------------------------------

#: A scratch mock plugin (``MOCK_LLM_PLUGIN``): every write batch appends one
#: marker line naming its first expected id to the file its prompt names, and
#: logs the ids it was dispatched with; a prompt holding
#: ``$COMMIT_APPLY_FAIL`` is a runner that tried and could not.
_WRITE_PLUGIN = '''\
import importlib.util, json, os, re, sys

_spec = importlib.util.spec_from_file_location("commit_apply_orig_mock", {mock!r})
_orig = importlib.util.module_from_spec(_spec)
sys.modules["commit_apply_orig_mock"] = _orig
_spec.loader.exec_module(_orig)

_IDS = re.compile(r"(?:suggestion|fix|task) \\d+ = `([^`]+)`")
_FILE = re.compile(r"\\*\\*File\\*\\*: (\\S+)|related issues in `([^`]+)`")


def parse_provider_args(provider):
    args, prompt = _orig.parse_provider_args(provider)
    if not prompt:
        prompt = _orig.read_stdin_prompt()
    ids = _IDS.findall(prompt)
    with open(os.environ["COMMIT_APPLY_CALLS"], "a", encoding="utf-8") as log:
        log.write(json.dumps(ids) + "\\n")
    fail = os.environ.get("COMMIT_APPLY_FAIL")
    if fail and fail in prompt:
        os.environ["MOCK_LLM_WRITE_RESULT"] = json.dumps({{
            "status": "failed", "ids": [], "files_touched": [],
            "summary": "scripted failure"}})
        return args, prompt
    found = _FILE.search(prompt)
    path = next(group for group in found.groups() if group)
    os.environ["MOCK_LLM_WRITE_IDS"] = json.dumps(ids)
    os.environ["MOCK_LLM_EDITS"] = json.dumps([{{
        "path": path, "mode": "patch", "content": "# fixed %s\\n" % ids[0]}}])
    return args, prompt
'''


#: A review completed before the runs below, so every pre-batch snapshot and
#: report row they leave is this review cycle's (``_seed``'s default
#: completion time is later in the day the suite was written on).
_REVIEWED_EARLIER = {"completed": "2026-01-01T00:00:00",
                     "resolved_at": "2025-12-31T23:00:00Z"}


def _marker(gid: str) -> str:
    return f"# fixed {gid}\n"


def _protected_clause_names(prompt: str, path: str) -> bool:
    """Whether *prompt*'s ``code_only_rule`` lists *path* as protected."""
    _, found, clause = prompt.partition(PROTECTED_CLAUSE_OPEN)
    return bool(found) and f"`{path}`" in clause


@pytest.fixture
def write_rig(ranged, monkeypatch, tmp_path):
    """``ranged`` with the mock codex as the write runner and the plugin above."""
    from tests.mock_launchers import MOCK_LLM, install_mock_llm, mock_path
    from utils import provider_registry as registry
    from utils import subagent_executor as executor

    bin_dir = tmp_path / "bin"
    install_mock_llm(bin_dir, "codex")
    plugin = tmp_path / "commit_apply_plugin.py"
    plugin.write_text(_WRITE_PLUGIN.format(mock=str(MOCK_LLM)), encoding="utf-8")
    config = tmp_path / "driver.yaml"
    config.write_text("driver:\n  subagent_runner: codex:gpt-5.5\n", encoding="utf-8")
    calls = tmp_path / "write-calls.jsonl"
    monkeypatch.setenv("PATH", mock_path(bin_dir, ("codex",)))
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
    monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
    monkeypatch.setenv("MOCK_LLM_PLUGIN", str(plugin))
    monkeypatch.setenv("COMMIT_APPLY_CALLS", str(calls))
    registry._config = None
    registry._config_key = None
    executor.reset_version_cache()
    yield ranged, calls
    registry._config = None
    registry._config_key = None


def _dispatched(calls: Path) -> List[List[str]]:
    if not calls.exists():
        return []
    return [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]


class TestAResumeAfterAHandEdit:
    """§2.8: an apply stopped after its first batch landed, a fix file edited by
    hand, then ``--resume``. The guard's re-check skips the edited file's group
    and protects the file, which grows the protected list every fix prompt
    carries (``code_only_rule``). The completed batch is kept by its result
    and its saved ``batch`` sha256, never dispatched again; the batches left
    run, and their prompts name the newly protected file."""

    @pytest.mark.parametrize("edited", ["b.py", "c.py"],
                             ids=["the stopped batch's file", "an undispatched batch's file"])
    def test_on_the_executor(self, write_rig, capsys, monkeypatch, edited):
        (repo, base, first, head), calls = write_rig
        groups = [_group("A", "a.py"), _group("B", "b.py"), _group("C", "c.py")]
        plan = _seed(repo, groups, base=base, head=head, **_REVIEWED_EARLIER)
        ids = dict(zip(("a.py", "b.py", "c.py"), _hashes(groups)))
        at_head = {name: (repo / name).read_text(encoding="utf-8") for name in ids}

        # -- the stop: batch 0 (a.py) lands, batch 1 (b.py) fails -------------
        monkeypatch.setenv("COMMIT_APPLY_FAIL", "**File**: b.py")
        rc, _ = _run_apply(plan, "--yes", "--agent-decide", "--exec-subagents")
        capsys.readouterr()
        assert rc == 3
        assert [call[0] for call in _dispatched(calls)] == [ids["a.py"], ids["b.py"]]
        assert (repo / "a.py").read_text(encoding="utf-8") == (
            at_head["a.py"] + _marker(ids["a.py"]))
        assert _guard(plan)["expect"]["a.py"]["kind"] == "batch"

        # -- the hand edit, then the resume -----------------------------------
        monkeypatch.delenv("COMMIT_APPLY_FAIL")
        hand = at_head[edited] + "# edited by hand\n"
        (repo / edited).write_text(hand, encoding="utf-8")
        before = len(_dispatched(calls))
        rc, _ = _run_apply(plan, "--yes", "--agent-decide", "--exec-subagents", "--resume")
        captured = capsys.readouterr()
        logged = captured.out + captured.err
        assert rc == 0, logged

        # The edited file's group is skipped and named, and the file protected.
        assert f"  skipped  1 (file changed since {head[:7]}: {edited})" in _block(
            captured.out)
        guard = _guard(plan)
        assert guard["skipped"][ids[edited]] == {"reason": "changed", "files": [edited]}
        assert edited in guard["protected"]
        assert (repo / edited).read_text(encoding="utf-8") == hand

        # The completed batch is kept, not dispatched again: one application.
        assert "(D2g)" not in logged
        assert "resume: kept 1 task(s) an earlier run finished — apply-fix-batch-0" in logged
        resumed = _dispatched(calls)[before:]
        assert all(ids["a.py"] not in call for call in resumed), resumed
        assert (repo / "a.py").read_text(encoding="utf-8") == (
            at_head["a.py"] + _marker(ids["a.py"]))

        # The batch left runs, and its prompt names the newly protected file.
        left = "c.py" if edited == "b.py" else "b.py"
        assert [call[0] for call in resumed] == [ids[left]]
        assert (repo / left).read_text(encoding="utf-8") == at_head[left] + _marker(ids[left])
        prompts = {b["fixes"][0]["file"]: b["prompt"] for b in _output(plan)["batches"]}
        assert _protected_clause_names(prompts[left], edited)

    def test_a_hand_edit_to_a_completed_groups_second_file(self, write_rig, capsys,
                                                           monkeypatch):
        """Batch 0 holds A (a.py) and AB (a.py, b.py) and lands; b.py is then
        edited by hand. Skipping AB on the resume would re-make batch 0 as A
        alone, and A would be applied a second time: AB stays, batch 0 is
        kept, and b.py is protected from every batch left."""
        (repo, base, first, head), calls = write_rig
        groups = [_group("A", "a.py"), _group("AB", "a.py", "b.py"), _group("C", "c.py")]
        plan = _seed(repo, groups, base=base, head=head, **_REVIEWED_EARLIER)
        a_id, ab_id, c_id = _hashes(groups)
        at_head = (repo / "a.py").read_text(encoding="utf-8")

        monkeypatch.setenv("COMMIT_APPLY_FAIL", "**File**: c.py")
        assert _run_apply(plan, "--yes", "--agent-decide", "--exec-subagents")[0] == 3
        assert _dispatched(calls) == [[a_id, ab_id], [c_id]]
        monkeypatch.delenv("COMMIT_APPLY_FAIL")
        capsys.readouterr()
        (repo / "b.py").write_text("b = 'by hand'\n", encoding="utf-8")

        rc, _ = _run_apply(plan, "--yes", "--agent-decide", "--exec-subagents", "--resume")
        captured = capsys.readouterr()
        logged = captured.out + captured.err
        assert rc == 0, logged
        assert "(D2g)" not in logged
        assert _dispatched(calls)[2:] == [[c_id]]
        assert (repo / "a.py").read_text(encoding="utf-8") == at_head + _marker(a_id)
        assert (repo / "b.py").read_text(encoding="utf-8") == "b = 'by hand'\n"
        guard = _guard(plan)
        assert ab_id not in guard["skipped"]
        assert "b.py" in guard["protected"] and "b.py" not in guard["expect"]
        prompts = {b["fixes"][0]["file"]: b["prompt"] for b in _output(plan)["batches"]}
        assert _protected_clause_names(prompts["c.py"], "b.py")

    def test_on_the_native_leg(self, ranged, capsys, monkeypatch):
        """The native leg has no such hazard: ``util batch-done`` records the
        completed batch's items, so the resume leaves its group out before the
        guard and batching (step 23) and no prompt of it is written again."""
        import multi_llm

        repo, base, first, head = ranged
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
        groups = [_group("A", "a.py"), _group("B", "b.py"), _group("C", "c.py")]
        plan = _seed(repo, groups, base=base, head=head, **_REVIEWED_EARLIER)
        ids = dict(zip(("a.py", "b.py", "c.py"), _hashes(groups)))
        phase_dir = str(output_dir_path(plan) / "apply-fixes")

        def util(*argv: str):
            code = multi_llm.main(["util", *argv, "--phase-dir", phase_dir])
            return code, capsys.readouterr().out

        assert _run_apply(plan, "--yes", "--agent-decide")[0] == 0
        assert [b["fixes"][0]["file"] for b in _output(plan)["batches"]] == [
            "a.py", "b.py", "c.py"]
        capsys.readouterr()
        code, out = util("write-prompt", "--task-id", "apply-fix-batch-0")
        assert code == 0 and out.startswith("[WRITE_PROMPT] "), out
        with open(repo / "a.py", "a", encoding="utf-8") as handle:
            handle.write(_marker(ids["a.py"]))
        code, out = util("batch-done", "--task-id", "apply-fix-batch-0",
                         "--status", "completed", "--own-check", "passed")
        assert code == 0, out
        # The host stops here; the user edits b.py.
        (repo / "b.py").write_text("b = 'by hand'\n", encoding="utf-8")

        assert _run_apply(plan, "--yes", "--agent-decide", "--resume")[0] == 0
        out = capsys.readouterr().out
        assert f"  skipped  1 (file changed since {head[:7]}: b.py)" in _block(out)
        batches = _output(plan)["batches"]
        assert [b["fixes"][0]["file"] for b in batches] == ["c.py"]
        assert _protected_clause_names(batches[0]["prompt"], "b.py")
        code, out = util("write-prompt", "--task-id", "apply-fix-batch-0")
        assert code == 0 and out.startswith("[WRITE_PROMPT] "), out
        assert ids["c.py"] in Path(out.split()[1]).read_text(encoding="utf-8")


class TestTheSharedFunctions:

    @pytest.fixture
    def guard(self, ranged):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py"), _group("C", "c.py")], base=base, head=head)
        outcome = take_guard(_state(plan), [_group("A", "a.py"), _group("C", "c.py")],
                             cwd=str(repo), plan_path=str(plan),
                             out_dir=str(output_dir_path(plan)),
                             phase_dir=str(output_dir_path(plan) / "apply-fixes"))
        return repo, outcome.guard

    def test_recheck_files(self, guard):
        repo, g = guard
        assert recheck_files(g, ["a.py", "c.py"], cwd=str(repo), file_sha256=file_sha256) == []
        (repo / "a.py").write_text("edit\n", encoding="utf-8")
        assert recheck_files(g, ["a.py", "c.py"], cwd=str(repo),
                             file_sha256=file_sha256) == ["a.py"]
        g["expect"]["a.py"] = {"kind": "batch", "sha256": file_sha256(str(repo / "a.py")),
                               "task_id": "t1"}
        assert recheck_files(g, ["a.py"], cwd=str(repo), file_sha256=file_sha256) == []
        (repo / "a.py").write_text("edit 2\n", encoding="utf-8")
        assert recheck_files(g, ["a.py"], cwd=str(repo), file_sha256=file_sha256) == ["a.py"]
        g["protected"]["c.py"] = "x"
        assert recheck_files(g, ["c.py"], cwd=str(repo), file_sha256=file_sha256) == ["c.py"]

    def test_repin_after_batch(self, guard):
        repo, g = guard
        sha = file_sha256(str(repo / "c.py"))
        assert not repin_after_batch(g, "t1", {"c.py": sha}, cwd=str(repo))
        assert g["expect"]["c.py"]["kind"] == "head"
        (repo / "c.py").write_text("batch edit\n", encoding="utf-8")
        sha = file_sha256(str(repo / "c.py"))
        assert repin_after_batch(g, "t1", {"c.py": sha}, cwd=str(repo))
        assert g["expect"]["c.py"] == {"kind": "batch", "sha256": sha, "task_id": "t1"}
        assert not repin_after_batch(g, "t2", {"c.py": sha}, cwd=str(repo))
        assert not repin_after_batch(g, "t2", {"elsewhere.py": "abc"}, cwd=str(repo))

    def test_protected_violations(self, guard):
        repo, g = guard
        assert protected_violations(g, cwd=str(repo)) == []
        _write(repo / "scratch.txt", "one\n")
        g["protected"]["scratch.txt"] = commit_apply.disk_blob_ids(
            ["scratch.txt"], cwd=str(repo))["scratch.txt"]
        g["protected"]["gone.txt"] = None
        assert protected_violations(g, cwd=str(repo)) == []
        (repo / "scratch.txt").write_text("two\n", encoding="utf-8")
        _write(repo / "gone.txt", "now here\n")
        assert protected_violations(g, cwd=str(repo)) == ["gone.txt", "scratch.txt"]

    def test_mark_stale(self, guard):
        repo, g = guard
        (repo / "a.py").write_text("stale\n", encoding="utf-8")
        mark_stale(g, {"g1": ["a.py", "c.py"], "g2": ["c.py"]}, ["a.py"], cwd=str(repo))
        assert g["skipped"]["g1"] == {"reason": "changed", "files": ["a.py"]}
        assert "g2" not in g["skipped"]
        assert g["protected"]["a.py"] == _git(repo, "hash-object", "a.py")
        assert "a.py" not in g["expect"]
        assert (repo / g["protected_copies"] / "a.py").read_text(encoding="utf-8") == "stale\n"

    def test_save_guard_writes_under_the_pinned_record(self, ranged):
        repo, base, first, head = ranged
        plan = _write(repo / "plans" / "p.md", "# P\n")
        _seed(repo, [_group("A", "a.py")], base=base, head=head, plan=plan, key="head_ref")
        state_path = output_dir_path(plan) / "state.json"
        save_guard(state_path, {"head": head, "skipped": {}})
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["head_ref"]["guard"] == {"head": head, "skipped": {}}
        assert state["phases_completed"]["code-review"]


# ---------------------------------------------------------------------------
# Strict endpoints, the direct run, the judge, status
# ---------------------------------------------------------------------------


class TestStrictEndpoints:

    @pytest.mark.parametrize("which", ["head", "base"])
    def test_an_unresolvable_recorded_sha_exits_3_before_the_guard(self, ranged, capsys, which):
        repo, base, first, head = ranged
        gone = "0123456789abcdef" * 2 + "01234567"
        plan = _seed(repo, [_group("A", "a.py")],
                     base=gone if which == "base" else base,
                     head=gone if which == "head" else head)
        rc, _ = _run_apply(plan)
        out = capsys.readouterr().out
        assert rc == 3
        assert f"[RANGE_UNRESOLVED] {which} {gone[:12]} recorded in" in out
        assert "--review-commits" in out
        assert "[COMMIT_FIXES]" not in out
        assert not (output_dir_path(plan) / "apply-fixes" / "orchestrator_output.json").exists()
        assert "guard" not in _state(plan)["commit_range"]


class TestTheDirectRunApply:

    def test_apply_after_a_direct_head_ref_review(self, repo, capsys):
        a = _commit(repo, "A", {"x.py": "x = 1\n", "y.py": "y = 1\n",
                                "plans/p.md": "# Plan\n"})
        b = _commit(repo, "B", {"x.py": "x = 2\n", "y.py": "y = 2\n"})
        plan = repo / "plans" / "p.md"
        groups = [_group("X", "x.py"), _group("Y", "y.py")]
        _seed(repo, groups, base=a, head=b, plan=plan, key="head_ref")
        _commit(repo, "on top", {"z.py": "z\n"})
        (repo / "y.py").write_text("hand\n", encoding="utf-8")
        rc, _ = _run_apply(plan)
        captured = capsys.readouterr()
        assert rc == 0
        assert _output(plan)["base_ref"] == a
        block = _block(captured.out)
        assert block == [
            "[COMMIT_FIXES] 1 fix to apply",
            f"  skipped  1 (file changed since {b[:7]}: y.py)",
            "[COMMIT_FIXES_END]",
        ]
        assert "HEAD~1" not in captured.out + captured.err
        assert _guard(plan, "head_ref")["skipped"]


class TestTheJudgeTasks:

    def test_a_pinned_runs_judge_tasks_carry_the_trust_sentence(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head,
                     statuses=["needs-human-decision"])
        _run_apply(plan, "--agent-decide")
        capsys.readouterr()
        tasks = json.loads((output_dir_path(plan) / "apply-fixes"
                            / "human_decision_tasks.json").read_text(encoding="utf-8"))
        assert all(DATA_UNDER_REVIEW in t["prompt"] for t in tasks["tasks"])


class TestTheRevalidationTasks:
    """§2.5 "The rule goes into every prompt … and the revalidation tasks", §2.6
    trust boundary: driven through apply-code-fixes' own ``--revalidate`` call,
    whose batches carry the plan fallback (nothing writes ``diff_context.txt``),
    so the guidance cannot hang on the context kind being ``diff``."""

    @staticmethod
    def _prompts(plan: Path) -> List[str]:
        path = output_dir_path(plan) / "code-review" / "revalidation_tasks.json"
        tasks = json.loads(path.read_text(encoding="utf-8"))
        assert tasks["context_kind"] == "plan", tasks["context_kind"]
        assert tasks["batches"], tasks
        return [batch["prompt"] for batch in tasks["batches"]]

    @pytest.mark.parametrize("key", ["commit_range", "head_ref"])
    def test_a_pinned_runs_revalidation_carries_the_rule_and_the_sentence(
        self, ranged, capsys, key
    ):
        repo, base, first, head = ranged
        plan = None
        if key == "head_ref":
            plan = _write(repo / "plans" / "p.md", "# Plan\n")
        plan = _seed(repo, [_group("A", "a.py"), _group("B", "b.py")], base=base,
                     head=head, plan=plan, key=key,
                     statuses=["validation_failed", "valid"])
        rc, _ = _run_apply(plan, "--revalidate")
        out = capsys.readouterr().out
        assert rc == 0 and "[REVALIDATION_PENDING]" in out, out
        rule = pinned_head_read_rule(base, head)
        for prompt in self._prompts(plan):
            assert rule in prompt, prompt
            assert DATA_UNDER_REVIEW in prompt, prompt

    def test_an_unpinned_runs_revalidation_carries_neither(self, repo, capsys):
        _commit(repo, "init", {"a.py": "a = 1\n", "plans/p.md": "# Plan\n"})
        head = _commit(repo, "touch a", {"a.py": "a = 2\n"})
        plan = _seed(repo, [_group("A", "a.py")], base=None, head=head,
                     plan=repo / "plans" / "p.md", key="head_ref",
                     statuses=["validation_failed"])
        state = _state(plan)
        del state["head_ref"]
        (output_dir_path(plan) / "state.json").write_text(json.dumps(state), encoding="utf-8")
        rc, _ = _run_apply(plan, "--revalidate")
        capsys.readouterr()
        assert rc == 0
        for prompt in self._prompts(plan):
            assert DATA_UNDER_REVIEW not in prompt
            assert "never in the working tree" not in prompt


class TestStatusRendersTheBlock:

    def _status(self, plan: Path, capsys, *argv: str):
        capsys.readouterr()
        context = cli_common.activate_result(cli_common.ResultContext())
        try:
            run_status.main(["--plan-file", str(plan), "--view", "run", *argv])
        finally:
            cli_common.deactivate_result()
        return capsys.readouterr().out, context.commit_fixes

    def test_status_prints_the_block_from_the_saved_guard(self, ranged, capsys):
        repo, base, first, head = ranged
        groups = [_group("A", "a.py", introduced_by=first), _group("B", "b.py")]
        plan = _seed(repo, groups, base=base, head=head)
        (repo / "b.py").write_text("drift\n", encoding="utf-8")
        _run_apply(plan)
        applied = _block(capsys.readouterr().out)
        # A detached apply's worker ran to the end.
        status_file.claim_run(str(output_dir_path(plan) / "apply-fixes"), {
            "run_id": "run-apply-0001", "state": "done", "exit_code": 0,
            "detached": True, "mode": "apply-code-fixes", "replay_flags": [],
        })
        out, fixes = self._status(plan, capsys)
        assert _block(out) == applied
        assert fixes["to_apply"] == 1 and fixes["skipped_changed"] == 1
        assert fixes["changed_files"] == ["b.py"]
        assert fixes["per_commit"] == [{"sha": first, "groups": 1, "subject": "touch a and b"}]
        assert fixes["lines"] == applied

    def test_no_guard_no_block(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        out, fixes = self._status(plan, capsys)
        assert "[COMMIT_FIXES]" not in out and fixes is None


# ---------------------------------------------------------------------------
# The no-argument lookup (select_review)
# ---------------------------------------------------------------------------


def _complete(root: Path, base: Optional[str], head: str, when: str, **kw) -> Path:
    return _seed(root, [_group("A", "a.py")], base=base, head=head, completed=when, **kw)


class TestSelectReview:

    @pytest.fixture
    def history(self, repo):
        shas = [_commit(repo, f"c{i}", {f"f{i}.py": f"{i}\n"}) for i in range(5)]
        return repo, shas

    def test_shared_head_takes_the_newest_and_names_the_other(self, history):
        repo, s = history
        older = _complete(repo, s[3], s[4], "2026-10-26T10:00:00")
        newer = _complete(repo, s[1], s[4], "2026-10-26T11:00:00")
        picked = select_review(cwd=str(repo))
        assert picked.brief == newer
        assert picked.line == (
            f"[COMMIT_REVIEW_SELECTED] {s[1][:12]}..{s[4][:12]} completed 2026-10-26 11:00, "
            f"ends at HEAD; also: {s[3][:12]}..{s[4][:12]}")
        assert picked.payload == {"completed_at": "2026-10-26T11:00:00", "behind": 0}
        # File times are never read.
        os.utime(older, None)
        os.utime(output_dir_path(older) / "state.json", None)
        assert select_review(cwd=str(repo)).brief == newer

    def test_head_moved_on(self, history, capsys):
        repo, s = history
        brief = _seed(repo, [_group("A", "f3.py"), _group("B", "f4.py")],
                      base=s[2], head=s[4])
        _commit(repo, "unrelated", {"other.py": "o\n"})
        _commit(repo, "touch a fix file", {"f3.py": "changed\n"})
        picked = select_review(cwd=str(repo))
        assert picked.brief == brief
        assert picked.line.endswith(
            "2 commits behind HEAD; the guard skips fix files they changed")
        assert picked.payload["behind"] == 2
        _run_apply(brief)
        block = _block(capsys.readouterr().out)
        assert block[0] == "[COMMIT_FIXES] 1 fix to apply"
        assert f"  skipped  1 (file changed since {s[4][:7]}: f3.py)" in block

    def test_a_range_below_head_is_picked_over_an_older_review(self, history):
        repo, s = history
        _complete(repo, s[3], s[4], "2026-10-26T10:00:00")
        below = _complete(repo, s[0], s[3], "2026-10-26T12:00:00")
        picked = select_review(cwd=str(repo))
        assert picked.brief == below
        assert "1 commit behind HEAD; the guard skips fix files it changed" in picked.line

    def test_newest_off_the_history(self, history):
        repo, s = history
        _git(repo, "checkout", "-q", "-b", "feature")
        f = _commit(repo, "feature work", {"feat.py": "f\n"})
        _git(repo, "checkout", "-q", "master")
        _complete(repo, s[4], f, "2026-10-26T12:00:00")
        with pytest.raises(CommitReviewError) as caught:
            select_review(cwd=str(repo))
        message = caught.value.message
        assert message.splitlines()[0] == (
            f"[COMMIT_REVIEW_NOT_FOUND] no completed commit review in {FOLDER} ends at "
            f"HEAD {s[4][:7]}, and the newest is not in its history")
        row = message.splitlines()[1]
        assert "not in HEAD's history" in row
        assert f"--apply-commit-fixes {s[4][:12]}..{f[:12]}" in row
        assert "1 valid" in row
        assert message.splitlines()[-1] == (
            "  next: --review-commits or --improve-commits to review HEAD")
        at_head = _complete(repo, s[3], s[4], "2026-10-26T09:00:00")
        assert select_review(cwd=str(repo)).brief == at_head

    def test_an_amended_head_is_listed_as_off_the_history(self, history):
        repo, s = history
        _complete(repo, s[3], s[4], "2026-10-26T12:00:00")
        _git(repo, "commit", "-q", "--amend", "-m", "amended")
        with pytest.raises(CommitReviewError) as caught:
            select_review(cwd=str(repo))
        assert "not in HEAD's history" in caught.value.message

    def test_unfinished_and_unreadable(self, history):
        repo, s = history
        unfinished = _complete(repo, s[3], s[4], None)
        bad = _complete(repo, s[2], s[4], "2026-10-26T10:00:00")
        (output_dir_path(bad) / "state.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(CommitReviewError) as caught:
            select_review(cwd=str(repo))
        lines = caught.value.message.splitlines()
        assert lines[0] == f"[COMMIT_REVIEW_NOT_FOUND] no completed commit review in {FOLDER}"
        rel = unfinished.relative_to(repo).as_posix()
        assert (f"  1 unfinished commit review not listed; newest: "
                f"{s[3][:7]}..{s[4][:7]} (status {rel})") in lines
        assert any(line.startswith("  1 unreadable state.json skipped: ") for line in lines)
        # The explicit range of the unfinished one is refused too.
        pinned = f"{s[3][:12]}..{s[4][:12]}"
        with pytest.raises(CommitReviewError) as refused:
            require_completed_review(unfinished, pinned=pinned, cwd=str(repo))
        text = refused.value.message
        assert text.startswith(f"[COMMIT_REVIEW_NOT_FOUND] {pinned}")
        assert f"--review-commits {pinned}" in text
        assert f"--improve-commits {pinned}" in text
        assert f"status {rel}" in text
        assert refused.value.exit_code == 3

    def test_require_completed_review_without_a_brief(self, history):
        repo, s = history
        brief = _brief_path(repo, s[3], s[4])
        with pytest.raises(CommitReviewError) as refused:
            require_completed_review(brief, pinned="p..q", cwd=str(repo))
        assert "no commit brief at" in refused.value.message
        assert "status " not in refused.value.message
        done = _complete(repo, s[3], s[4], "2026-10-26T10:00:00")
        require_completed_review(done, pinned="p..q", cwd=str(repo))

    @pytest.mark.config_override
    def test_a_configured_folder(self, history):
        import utils.provider_registry as registry

        repo, s = history
        _write(repo / ".multi-llm" / "providers.yaml",
               "code_review:\n  commit_reviews_dir: reviews/commits\n")
        registry._config = None
        registry._config_key = None
        _complete(repo, s[1], s[4], "2026-10-26T12:00:00")  # plans/commit-reviews
        configured = _complete(repo, s[3], s[4], "2026-10-26T09:00:00",
                               folder="reviews/commits")
        picked = select_review(cwd=str(repo))
        assert picked.brief == configured
        assert "also" not in picked.line

    def test_an_unborn_head_exits_2_first(self, repo):
        with pytest.raises(RangeError) as caught:
            select_review(cwd=str(repo))
        assert caught.value.exit_code == 2
        assert caught.value.message.startswith("[NOTHING_TO_REVIEW] no commits yet on master")

    def test_no_review_at_all(self, history):
        repo, s = history
        with pytest.raises(CommitReviewError) as caught:
            select_review(cwd=str(repo))
        assert caught.value.message.splitlines() == [
            f"[COMMIT_REVIEW_NOT_FOUND] no completed commit review in {FOLDER}",
            "  next: --review-commits or --improve-commits to review HEAD",
        ]


# ---------------------------------------------------------------------------
# Review fix 3 (adversarial review, area C): items 2 to 6
# ---------------------------------------------------------------------------


class TestAStateTheApplyCannotTrust:
    """Item 2: ``StateManager`` replaces a ``state.json`` that does not parse
    with a fresh state, so a plain ``apply-code-fixes <brief>`` ran unguarded
    against ``head_at_start`` and then saved over the file. Now it exits 3
    before anything is written."""

    def _bytes(self, plan: Path) -> bytes:
        return (output_dir_path(plan) / "state.json").read_bytes()

    @pytest.mark.parametrize("flags", [(), ("--mark-completed",)])
    def test_a_corrupt_state_on_a_brief_exits_3_and_writes_nothing(self, ranged, capsys, flags):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        state_path = output_dir_path(plan) / "state.json"
        state_path.write_text("{not json", encoding="utf-8")
        (repo / "a.py").write_text("a = 3\n", encoding="utf-8")
        rc, _orch = _run_apply(plan, *flags)
        captured = capsys.readouterr()
        assert rc == 3
        assert "[COMMIT_FIXES]" not in captured.out
        assert "commit brief's state.json" in captured.err
        assert "it does not parse" in captured.err
        assert f"--review-commits {base[:12]}..{head[:12]}" in captured.err
        assert state_path.read_text(encoding="utf-8") == "{not json"
        assert not (output_dir_path(plan) / "apply-fixes" / "orchestrator_output.json").exists()

    def test_a_corrupt_state_on_a_plan_exits_3_too(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _write(repo / "plans" / "p.md", "# P\n")
        _seed(repo, [_group("A", "a.py")], base=base, head=head, plan=plan, key="head_ref")
        state_path = output_dir_path(plan) / "state.json"
        state_path.write_text("[]", encoding="utf-8")
        rc, _orch = _run_apply(plan)
        err = capsys.readouterr().err
        assert rc == 3
        assert "it does not hold a JSON object" in err
        assert state_path.read_text(encoding="utf-8") == "[]"

    def test_a_brief_whose_state_records_no_range_exits_3(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        state_path = output_dir_path(plan) / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        del state["commit_range"]
        state_path.write_text(json.dumps(state), encoding="utf-8")
        before = state_path.read_bytes()
        rc, _orch = _run_apply(plan)
        assert rc == 3
        assert "it records no reviewed commit range" in capsys.readouterr().err
        assert state_path.read_bytes() == before

    def test_a_plan_with_a_good_state_is_untouched(self, ranged, capsys):
        repo, base, first, head = ranged
        plan = _write(repo / "plans" / "p.md", "# P\n")
        _seed(repo, [_group("A", "a.py")], base=base, head=head, plan=plan, key="head_ref")
        rc, _orch = _run_apply(plan)
        assert rc == 0
        assert "cannot be used" not in capsys.readouterr().err


class TestAnArchivedRoundIsNotACompletedReview:
    """Item 3 (the apply half): a restart archives ``code-review/*``; if its
    phase never runs to pop ``phases_completed["code-review"]``, the review
    must still not be picked or accepted, because its results are gone."""

    def test_select_review_and_require_completed_review_refuse_it(self, ranged):
        from utils.commit_brief import archive_round

        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        assert select_review(cwd=str(repo)).brief == plan
        assert archive_round(output_dir_path(plan) / "code-review") is not None
        with pytest.raises(CommitReviewError) as caught:
            select_review(cwd=str(repo))
        lines = caught.value.message.splitlines()
        assert lines[0] == f"[COMMIT_REVIEW_NOT_FOUND] no completed commit review in {FOLDER}"
        assert any("1 unfinished commit review not listed" in line for line in lines)
        with pytest.raises(CommitReviewError) as caught:
            require_completed_review(plan, pinned="p", cwd=str(repo))
        message = caught.value.message
        # Review R2, item 5: it says what is missing, and offers a restart
        # as one explanation, never as the fact.
        assert ("its state records the code review as completed, but its results "
                "(code-review/grouped.json) are not on disk (a restart that has "
                "not completed since moves them out, for one)") in message
        assert "it was restarted" not in message
        assert "review it first: --review-commits p, then --apply-commit-fixes p; " \
               "or both in one run: --improve-commits p" in message

    def test_a_skip_validation_round_still_counts(self, ranged):
        repo, base, first, head = ranged
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head)
        (output_dir_path(plan) / "code-review" / "validation.json").unlink()
        assert select_review(cwd=str(repo)).brief == plan
        require_completed_review(plan, pinned="p", cwd=str(repo))


class TestTheContextPlanIsWatched:
    """Item 4: the guard saves the context plan's disk id (and a copy) when
    it is not already protected, so both write legs catch a batch that
    writes it (``watched_ids`` / ``protected_violations``)."""

    def test_take_guard_records_it_and_a_write_to_it_is_a_violation(self, repo):
        base = _commit(repo, "init", {"a.py": "a = 1\n", "docs/x.md": "# x\n"})
        head = _commit(repo, "work", {"a.py": "a = 2\n"})
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head,
                     context_plan="docs/x.md")
        phase_dir = output_dir_path(plan) / "apply-fixes"
        outcome = take_guard(_state(plan), [_group("A", "a.py")], cwd=str(repo),
                             plan_path=str(plan), out_dir=str(output_dir_path(plan)),
                             phase_dir=str(phase_dir))
        g = outcome.guard
        assert "docs/x.md" not in g["protected"]
        assert g["context_plan_id"] == _git(repo, "hash-object", "docs/x.md")
        assert commit_apply.watched_ids(g)["docs/x.md"] == g["context_plan_id"]
        assert (repo / g["protected_copies"] / "docs/x.md").read_text() == "# x\n"
        assert protected_violations(g, cwd=str(repo)) == []
        (repo / "docs" / "x.md").write_text("# rewritten\n", encoding="utf-8")
        assert protected_violations(g, cwd=str(repo)) == ["docs/x.md"]
        report = commit_apply.restore_watched(g, ["docs/x.md"], cwd=str(repo))
        assert report == {"docs/x.md": commit_apply.RESTORED_FROM_COPY}
        assert (repo / "docs" / "x.md").read_text() == "# x\n"

    def test_a_protected_context_plan_is_watched_once_under_the_protected_set(self, repo):
        base = _commit(repo, "init", {"a.py": "a = 1\n", "docs/x.md": "# x\n"})
        head = _commit(repo, "work", {"a.py": "a = 2\n"})
        (repo / "docs" / "x.md").write_text("# edited since\n", encoding="utf-8")
        plan = _seed(repo, [_group("A", "a.py")], base=base, head=head,
                     context_plan="docs/x.md")
        g = take_guard(_state(plan), [_group("A", "a.py")], cwd=str(repo),
                       plan_path=str(plan), out_dir=str(output_dir_path(plan)),
                       phase_dir=str(output_dir_path(plan) / "apply-fixes")).guard
        assert "docs/x.md" in g["protected"] and "context_plan_id" not in g
        assert list(commit_apply.watched_ids(g)) == list(g["protected"])


class TestTheContextPlanAcrossAResume:
    """Review R2, item 1: the user edits their own context plan between a
    stopped apply and its ``--resume``. The reused decision kept the stopped
    run's ``context_plan_id``, so on the native leg every later ``util
    batch-done --status completed`` refused with ``[COMMIT_PROTECTED_WRITE]
    … docs/x.md`` and told the host to copy the guard's copy back over the
    user's edit. The resumed decision now refreshes the plan's id and copy,
    and each native batch is compared with the tree as its prompt was bound
    (the executor's per-batch window); a write made during a batch is still
    refused."""

    @staticmethod
    def _plan_repo(repo):
        base = _commit(repo, "init", {"a.py": "a = 1\n", "b.py": "b = 1\n", "c.py": "c = 1\n",
                                      "docs/x.md": "# the plan\n"})
        _commit(repo, "touch a and b", {"a.py": "a = 2\n", "b.py": "b = 2\n"})
        head = _commit(repo, "touch c", {"c.py": "c = 2\n"})
        return base, head

    @pytest.mark.parametrize("protected", [False, True],
                             ids=["context_plan_id", "in the protected set"])
    def test_a_resumed_decision_refreshes_the_plans_id_and_copy(self, repo, protected):
        base, head = self._plan_repo(repo)
        if protected:
            (repo / "docs" / "x.md").write_text("# the plan, edited before\n", encoding="utf-8")
        groups = [_group("A", "a.py"), _group("B", "b.py")]
        plan = _seed(repo, groups, base=base, head=head, context_plan="docs/x.md")
        out_dir = output_dir_path(plan)
        kw = dict(cwd=str(repo), plan_path=str(plan), out_dir=str(out_dir),
                  phase_dir=str(out_dir / "apply-fixes"))
        first = take_guard(_state(plan), groups, **kw).guard
        assert ("docs/x.md" in first["protected"]) is protected
        first["bound"] = {"apply-fix-batch-0": {"docs/x.md": "stale"}}

        edited = "# the plan, with a typo fixed\n"
        (repo / "docs" / "x.md").write_text(edited, encoding="utf-8")
        state = _state(plan)
        state["commit_range"]["guard"] = first
        resumed = take_guard(state, groups, resume=True, **kw)
        g = resumed.guard
        assert resumed.reused
        now = _git(repo, "hash-object", "docs/x.md")
        assert commit_apply.watched_ids(g)["docs/x.md"] == now
        assert (repo / g["protected_copies"] / "docs/x.md").read_text() == edited
        # The stopped run's bind records never stand for this run's batches.
        assert "bound" not in g
        # Nothing ran since: no violation.
        assert protected_violations(g, cwd=str(repo)) == []
        # A batch that writes it is still caught, and put back to the user's
        # edit, not to the stopped run's copy.
        (repo / "docs" / "x.md").write_text("# rewritten by a batch\n", encoding="utf-8")
        assert protected_violations(g, cwd=str(repo)) == ["docs/x.md"]
        assert commit_apply.restore_watched(g, ["docs/x.md"], cwd=str(repo)) == {
            "docs/x.md": commit_apply.RESTORED_FROM_COPY}
        assert (repo / "docs" / "x.md").read_text() == edited

    def test_on_the_native_leg(self, repo, capsys, monkeypatch):
        import multi_llm

        base, head = self._plan_repo(repo)
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
        groups = [_group("A", "a.py"), _group("B", "b.py"), _group("C", "c.py")]
        plan = _seed(repo, groups, base=base, head=head, context_plan="docs/x.md",
                     **_REVIEWED_EARLIER)
        ids = dict(zip(("a.py", "b.py", "c.py"), _hashes(groups)))
        phase_dir = str(output_dir_path(plan) / "apply-fixes")

        def util(*argv: str):
            capsys.readouterr()
            code = multi_llm.main(["util", *argv, "--phase-dir", phase_dir])
            return code, capsys.readouterr().out

        def run_batch(task_id: str, path: str):
            code, out = util("write-prompt", "--task-id", task_id)
            assert code == 0 and out.startswith("[WRITE_PROMPT] "), out
            with open(repo / path, "a", encoding="utf-8") as handle:
                handle.write(_marker(ids[path]))

        def done(task_id: str):
            return util("batch-done", "--task-id", task_id,
                        "--status", "completed", "--own-check", "passed")

        assert _run_apply(plan, "--yes", "--agent-decide")[0] == 0
        capsys.readouterr()
        run_batch("apply-fix-batch-0", "a.py")
        assert done("apply-fix-batch-0")[0] == 0
        # The host stops here. The user fixes a typo in THEIR plan, then resumes.
        edited = "# the plan, with a typo fixed\n"
        (repo / "docs" / "x.md").write_text(edited, encoding="utf-8")
        assert _run_apply(plan, "--yes", "--agent-decide", "--resume")[0] == 0
        capsys.readouterr()
        assert [b["fixes"][0]["file"] for b in _output(plan)["batches"]] == ["b.py", "c.py"]

        run_batch("apply-fix-batch-0", "b.py")
        code, out = done("apply-fix-batch-0")
        assert code == 0, out
        assert out.startswith("[BATCH_DONE] apply-fix-batch-0 completed")
        assert (repo / "docs" / "x.md").read_text(encoding="utf-8") == edited

        # A batch that does write the plan is still refused, and the way back
        # is a copy holding the user's edit.
        run_batch("apply-fix-batch-1", "c.py")
        (repo / "docs" / "x.md").write_text("# rewritten by the batch\n", encoding="utf-8")
        code, out = done("apply-fix-batch-1")
        assert code == 3, out
        first_line, way_back = out.splitlines()[:2]
        assert first_line == "[COMMIT_PROTECTED_WRITE] apply-fix-batch-1: docs/x.md"
        copies = _guard(plan)["protected_copies"]
        assert f"copy {copies}/docs/x.md back over docs/x.md" in way_back
        assert (repo / copies / "docs" / "x.md").read_text(encoding="utf-8") == edited


class TestAProtectedPathThatCannotBeHashed:
    """Item 5 hashed a newline in a name with the name in argv, and made an
    ``unreadable`` read always count. Review R2, item 2: "always" failed
    every batch on both legs, whatever it wrote, with a reason that was not
    true, for a protected file that simply cannot be read (mode 000, owned
    by root). The guard now decides about such a path once, when it is
    taken: it names it in the block and skips every group naming it, and
    the write legs count only a move into or out of readability."""

    def test_unreadable_is_its_own_identity(self):
        assert not commit_apply.identity_changed("unreadable", "unreadable")
        assert commit_apply.identity_changed("abc", "unreadable")
        assert commit_apply.identity_changed("unreadable", "abc")
        assert commit_apply.identity_changed("unreadable", None)
        assert commit_apply.identity_changed(None, "unreadable")
        assert not commit_apply.identity_changed("abc", "abc")
        assert not commit_apply.identity_changed(None, None)

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file names and modes")
    def test_a_newline_name_gets_a_real_id_and_a_write_is_caught(self, repo):
        name = "odd\nname.py"
        _write(repo / name, "one\n")
        ids = commit_apply.disk_blob_ids([name], cwd=str(repo))
        assert ids[name] != "unreadable" and len(ids[name]) == 40
        g = {"protected": dict(ids)}
        assert protected_violations(g, cwd=str(repo)) == []
        (repo / name).write_text("two\n", encoding="utf-8")
        assert protected_violations(g, cwd=str(repo)) == [name]

    @pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                        reason="needs POSIX modes and a non-root user")
    def test_an_unreadable_protected_file_counts_only_when_it_moves_in_or_out(self, repo):
        _write(repo / "secret.txt", "one\n")
        os.chmod(repo / "secret.txt", 0)
        try:
            ids = commit_apply.disk_blob_ids(["secret.txt"], cwd=str(repo))
            assert ids["secret.txt"] == "unreadable"
            g = {"protected": dict(ids)}
            # Unreadable then and now: nothing to compare, not a batch's write.
            assert protected_violations(g, cwd=str(repo)) == []
            assert protected_violations(g, cwd=str(repo), baseline=dict(ids)) == []
            # Made readable (and rewritten): a move out of it is a change.
            os.chmod(repo / "secret.txt", 0o644)
            (repo / "secret.txt").write_text("two\n", encoding="utf-8")
            assert protected_violations(g, cwd=str(repo)) == ["secret.txt"]
            # A readable protected file that becomes unreadable is one too.
            readable = {"protected": commit_apply.disk_blob_ids(["secret.txt"], cwd=str(repo))}
            os.chmod(repo / "secret.txt", 0)
            assert protected_violations(readable, cwd=str(repo)) == ["secret.txt"]
        finally:
            os.chmod(repo / "secret.txt", 0o644)

    @pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                        reason="needs POSIX modes and a non-root user")
    def test_the_guard_names_an_unreadable_protected_file_once_and_skips_its_groups(
        self, ranged, capsys,
    ):
        repo, base, first, head = ranged
        _write(repo / "secret.env", "TOKEN=1\n")           # untracked since head
        os.chmod(repo / "secret.env", 0)
        try:
            groups = [_group("A", "a.py"), _group("S", "secret.env"), _group("C", "c.py")]
            plan = _seed(repo, groups, base=base, head=head)
            rc, _orch = _run_apply(plan, "--dry-run")
            block = _block(capsys.readouterr().out)
            assert rc == 0
            assert block[0] == "[COMMIT_FIXES] 2 fixes to apply"
            assert f"  skipped  1 (file changed since {head[:7]}: secret.env)" in block
            assert block[-2] == (
                "  unreadable  1 (secret.env): no fix naming it is applied, and a "
                "batch write to it cannot be seen while it stays unreadable")

            out_dir = output_dir_path(plan)
            outcome = take_guard(_state(plan), groups, cwd=str(repo), plan_path=str(plan),
                                 out_dir=str(out_dir), phase_dir=str(out_dir / "apply-fixes"))
            g = outcome.guard
            assert g["unreadable"] == ["secret.env"]
            assert g["protected"]["secret.env"] == "unreadable"
            assert outcome.payload["unreadable"] == ["secret.env"]
            # Untracked, so absent at head: the change test skips its group
            # first (its being protected would skip it too).
            assert outcome.skipped[_hashes(groups)[1]] == {
                "reason": "changed", "files": ["secret.env"]}
            # Left out of the per-batch comparison while it stays unreadable.
            assert protected_violations(g, cwd=str(repo)) == []
        finally:
            os.chmod(repo / "secret.env", 0o644)

    def test_a_readable_tree_records_no_unreadable_key(self, ranged, capsys):
        repo, base, first, head = ranged
        _write(repo / "notes.txt", "n\n")
        groups = [_group("A", "a.py")]
        plan = _seed(repo, groups, base=base, head=head)
        out_dir = output_dir_path(plan)
        outcome = take_guard(_state(plan), groups, cwd=str(repo), plan_path=str(plan),
                             out_dir=str(out_dir), phase_dir=str(out_dir / "apply-fixes"))
        assert "unreadable" not in outcome.guard
        assert outcome.payload["unreadable"] == []
        assert not any(line.startswith("  unreadable") for line in outcome.lines)

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file names")
    @pytest.mark.parametrize("name", ['"x"', '"x', 'back\\slash', 'tab\there'],
                             ids=["quoted", "leading quote", "backslash", "control"])
    def test_a_name_git_would_quote_gets_its_own_id_and_a_write_is_caught(self, repo, name):
        """Review R2, item 4: ``--stdin-paths`` C-unquotes a line that starts
        with ``"``, so ``"x"`` was hashed as ``x`` (whose id was saved) and a
        write to it was missed. Every name git would quote is hashed alone."""
        twin = name.strip('"').replace("\\", "").replace("\t", "")
        _write(repo / twin, "the twin\n")
        _write(repo / name, "the protected file\n")
        ids = commit_apply.disk_blob_ids([name, twin, "other.txt"], cwd=str(repo))
        real = subprocess.run(["git", "hash-object", "--no-filters", "--", name], cwd=repo,
                              check=True, capture_output=True, text=True).stdout.strip()
        assert ids[name] == real != ids[twin]
        assert ids["other.txt"] is None
        g = {"protected": {name: ids[name]}}
        assert protected_violations(g, cwd=str(repo)) == []
        (repo / name).write_text("clobbered\n", encoding="utf-8")
        assert protected_violations(g, cwd=str(repo)) == [name]
        assert protected_violations(g, cwd=str(repo), baseline={name: ids[name]}) == [name]

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
    def test_a_fifo_is_its_kind_not_unreadable(self, repo):
        os.mkfifo(repo / "pipe")
        ids = commit_apply.disk_blob_ids(["pipe"], cwd=str(repo))
        assert ids["pipe"].startswith("special:")
        assert protected_violations({"protected": dict(ids)}, cwd=str(repo)) == []


class TestDotDotSegments:
    """Item 6: a fix naming the context plan as ``docs/../docs/x.md`` was
    skipped as "file changed since" instead of under the context plan."""

    def test_the_guards_names_resolve_dot_dot_lexically(self, repo):
        assert commit_apply._git_name("docs/../docs/x.md", str(repo)) == "docs/x.md"
        assert commit_apply._git_name("a/../../x.md", str(repo)) == "../x.md"
        assert not commit_apply._inside(commit_apply._git_name("a/../../x.md", str(repo)))
        assert commit_apply._git_name("docs/x.md", str(repo)) == "docs/x.md"

    def test_a_dot_dot_spelling_of_the_context_plan_is_skipped_as_the_context_plan(
        self, repo, capsys,
    ):
        base = _commit(repo, "init", {"a.py": "a = 1\n", "docs/x.md": "# plan\n"})
        head = _commit(repo, "touch a", {"a.py": "a = 2\n"})
        groups = [_group("fix plan", "docs/../docs/x.md"), _group("fix a", "a.py")]
        plan = _seed(repo, groups, base=base, head=head, context_plan="docs/x.md")
        rc, _orch = _run_apply(plan)
        block = _block(capsys.readouterr().out)
        assert rc == 0
        assert any(line.startswith("  skipped  1 (context plan, read-only: docs/x.md)")
                   for line in block), block
        assert _guard(plan)["skipped"][_hashes(groups)[0]] == {
            "reason": "context_plan", "files": ["docs/x.md"]}


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
class TestASymlinkedContextPlan:
    """A ``--context-plan docs/plan-link.md`` symlink to ``docs/plan.md``
    keeps the link's own name (``normalize_plan_path``), so a fix naming
    ``docs/plan.md`` passed the context-plan skip, the watcher saved only
    the link's target text, and the executor denied only the link: editing
    the plan's real text escaped every layer. The guard now saves the
    chain's in-repository targets (``context_plan_targets``), skips a group
    naming one as the context plan, and both write legs watch and deny
    them."""

    @staticmethod
    def _linked(repo):
        base = _commit(repo, "init", {"a.py": "a = 1\n", "docs/plan.md": "# plan\n"})
        os.symlink("plan.md", repo / "docs" / "plan-link.md")
        _git(repo, "add", "--", "docs/plan-link.md")
        _git(repo, "commit", "-q", "-m", "link")
        head = _commit(repo, "touch a", {"a.py": "a = 2\n"})
        return base, head

    def test_symlink_targets_follows_the_chain_inside_the_repository(self, repo, tmp_path):
        self._linked(repo)
        os.symlink("plan-link.md", repo / "docs" / "second-link.md")
        assert commit_apply.symlink_targets("docs/plan-link.md", cwd=str(repo)) == [
            "docs/plan.md"]
        assert commit_apply.symlink_targets("docs/second-link.md", cwd=str(repo)) == [
            "docs/plan-link.md", "docs/plan.md"]
        assert commit_apply.symlink_targets("docs/plan.md", cwd=str(repo)) == []
        outside = tmp_path / "outside.md"
        outside.write_text("# out\n", encoding="utf-8")
        os.symlink(str(outside), repo / "docs" / "out-link.md")
        assert commit_apply.symlink_targets("docs/out-link.md", cwd=str(repo)) == []
        os.symlink("loop-b.md", repo / "docs" / "loop-a.md")
        os.symlink("loop-a.md", repo / "docs" / "loop-b.md")
        assert commit_apply.symlink_targets("docs/loop-a.md", cwd=str(repo)) == [
            "docs/loop-b.md"]

    def test_a_fix_naming_the_target_is_skipped_and_the_target_is_watched(self, repo):
        base, head = self._linked(repo)
        groups = [_group("fix the plan", "docs/plan.md"), _group("fix a", "a.py")]
        ids = _hashes(groups)
        plan = _seed(repo, groups, base=base, head=head, context_plan="docs/plan-link.md")
        out_dir = output_dir_path(plan)
        outcome = take_guard(_state(plan), groups, cwd=str(repo), plan_path=str(plan),
                             out_dir=str(out_dir), phase_dir=str(out_dir / "apply-fixes"))
        g = outcome.guard
        assert outcome.skipped == {
            ids[0]: {"reason": "context_plan", "files": ["docs/plan.md"]}}
        assert g["context_plan"] == "docs/plan-link.md"
        target_id = _git(repo, "hash-object", "docs/plan.md")
        assert g["context_plan_targets"] == {"docs/plan.md": target_id}
        assert commit_apply.context_plan_paths(g) == ["docs/plan-link.md", "docs/plan.md"]
        assert commit_apply.watched_ids(g)["docs/plan.md"] == target_id
        assert (repo / g["protected_copies"] / "docs/plan.md").read_text() == "# plan\n"
        assert protected_violations(g, cwd=str(repo)) == []
        # Writing through the link lands in the target: caught under its name.
        (repo / "docs" / "plan-link.md").write_text("# rewritten\n", encoding="utf-8")
        assert protected_violations(g, cwd=str(repo)) == ["docs/plan.md"]
        assert commit_apply.restore_watched(g, ["docs/plan.md"], cwd=str(repo)) == {
            "docs/plan.md": commit_apply.RESTORED_FROM_COPY}
        assert (repo / "docs" / "plan.md").read_text() == "# plan\n"
        assert os.path.islink(repo / "docs" / "plan-link.md")

    def test_the_block_counts_it_under_the_context_plan(self, repo, capsys):
        base, head = self._linked(repo)
        groups = [_group("fix the plan", "docs/plan.md"), _group("fix a", "a.py")]
        plan = _seed(repo, groups, base=base, head=head, context_plan="docs/plan-link.md")
        rc, _orch = _run_apply(plan)
        block = _block(capsys.readouterr().out)
        assert rc == 0
        assert any(
            line.startswith("  skipped  1 (context plan, read-only: docs/plan-link.md)")
            for line in block), block

    def test_a_resume_refreshes_the_targets_id(self, repo):
        base, head = self._linked(repo)
        groups = [_group("fix a", "a.py")]
        plan = _seed(repo, groups, base=base, head=head, context_plan="docs/plan-link.md")
        out_dir = output_dir_path(plan)
        kw = dict(cwd=str(repo), plan_path=str(plan), out_dir=str(out_dir),
                  phase_dir=str(out_dir / "apply-fixes"))
        first = take_guard(_state(plan), groups, **kw).guard
        edited = "# plan, edited by the user\n"
        (repo / "docs" / "plan.md").write_text(edited, encoding="utf-8")
        state = _state(plan)
        state["commit_range"]["guard"] = first
        g = take_guard(state, groups, resume=True, **kw).guard
        assert g["context_plan_targets"] == {
            "docs/plan.md": _git(repo, "hash-object", "docs/plan.md")}
        assert protected_violations(g, cwd=str(repo)) == []
        assert (repo / g["protected_copies"] / "docs/plan.md").read_text() == edited

    @pytest.mark.parametrize("persist", [True, False])
    def test_resume_protects_a_retargeted_context_plan(self, repo, persist):
        base, _head = self._linked(repo)
        head = _commit(repo, "another document", {"docs/next.md": "# next plan\n"})
        groups = [_group("fix next", "docs/next.md"), _group("fix a", "a.py")]
        plan = _seed(repo, groups, base=base, head=head, context_plan="docs/plan-link.md")
        out_dir = output_dir_path(plan)
        kw = dict(cwd=str(repo), plan_path=str(plan), out_dir=str(out_dir),
                  phase_dir=str(out_dir / "apply-fixes"), persist=persist)
        state = _state(plan)
        state["commit_range"]["guard"] = take_guard(state, groups, **kw).guard
        link = repo / "docs/plan-link.md"
        link.unlink()
        link.symlink_to("next.md")

        resumed = take_guard(state, groups, resume=True, **kw)
        assert resumed.skipped[_hashes(groups)[0]] == {
            "reason": "context_plan", "files": ["docs/next.md"]}
        # Keep earlier protections too: a resume never releases the old target.
        assert set(commit_apply.context_plan_paths(resumed.guard)) == {
            "docs/plan-link.md", "docs/plan.md", "docs/next.md"}
        assert protected_violations(resumed.guard, cwd=str(repo)) == []
        if persist:
            assert (repo / resumed.guard["protected_copies"] / "docs/next.md").read_text() == (
                "# next plan\n")
        link.write_text("# overwritten by a batch\n", encoding="utf-8")
        assert protected_violations(resumed.guard, cwd=str(repo)) == ["docs/next.md"]

    def test_the_executor_denies_the_target_as_the_context_plan(self, repo):
        import types

        from utils.write_batch_scheduler import WriteBatchScheduler
        from utils.write_safety import CONTEXT_PLAN_RULE, PolicyViolation, WritePolicy

        policy = WritePolicy(root=str(repo), allow=["**"], deny=[])
        WriteBatchScheduler._commit_protect(types.SimpleNamespace(policy=policy), {
            "head": "0" * 40, "protected": {}, "context_plan": "docs/plan-link.md",
            "context_plan_targets": {"docs/plan.md": "id"},
        })
        for path in ("docs/plan-link.md", "docs/plan.md"):
            with pytest.raises(PolicyViolation) as caught:
                policy.check([path], when="post-run")
            assert CONTEXT_PLAN_RULE in str(caught.value)

    def test_batch_done_names_the_target_as_the_context_plan(self, repo):
        from utils.batch_done import _protected_write_lines

        guard = {"head": "abcdef1234", "protected": {}, "context_plan": "docs/plan-link.md",
                 "context_plan_id": "symlink:plan.md",
                 "context_plan_targets": {"docs/plan.md": "id"},
                 "protected_copies": "plans/x/apply-fixes/protected"}
        lines = _protected_write_lines(
            Path("plans/x/apply-fixes"), "apply-fix-batch-0", ["docs/plan.md"], guard,
            "[COMMIT_PROTECTED_WRITE]", before={"docs/plan.md": "id"},
            copy_held=lambda path, was: True)
        assert lines[0] == "[COMMIT_PROTECTED_WRITE] apply-fix-batch-0: docs/plan.md"
        assert ("docs/plan.md is the context plan this review checks the commits "
                "against (through the symlink docs/plan-link.md)") in lines[1]
        assert "changed since reviewed commit" not in lines[1]
        assert "copy plans/x/apply-fixes/protected/docs/plan.md back over docs/plan.md" \
            in lines[1]
        assert '--context-plan ""' in lines[1]
