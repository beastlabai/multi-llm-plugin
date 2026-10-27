"""review-code with ``--head-ref``: a review pinned to two commits.

plans/review-commits.md §2.5 (and §2.4's strict re-resolution of recorded
SHAs): with a head ref the change under review is ``git diff <base> <head>``,
never the working tree, so

* both refs resolve strictly (no ``validate_git_ref`` and no ``HEAD~1``), an
  empty-tree base included, and a recorded SHA that stops resolving is
  ``[RANGE_UNRESOLVED]`` exit 3 before anything is written;
* the scope is the range's own diff (``SCOPE_SOURCE_COMMIT_RANGE``), which
  ``--files`` narrows and never widens, with none of the working-tree guesses
  (baseline, intent-to-add, untracked announcement, staleness);
* every prompt that asks a model to read code states the pinned-head read
  rule, and every read Python makes (validation excerpts, report context) is
  at ``head``, so ``reviewed_code_sha256`` survives a moved tree;
* a direct run records ``head_ref`` (and its drift) when a round starts, a
  fronted run updates ``commit_range`` instead, and the recorded invocation
  carries the pair as full SHAs.

Every test drives the real ``code_review_orchestrator.main`` in-process on a
temporary git repository. The one seam is ``utils.llm_client.invoke_with_provider``
— the real ``invoke_with_file_output`` formats the real prompt (so a missing
placeholder would be ``PROMPT_FORMAT_ERROR``) and the fake hands back findings,
so no model is ever called.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro  # noqa: E402
from harness import SkillRunner  # noqa: E402
from utils import cli_common  # noqa: E402
from utils.commit_range import DATA_UNDER_REVIEW, pinned_head_read_rule  # noqa: E402
from utils.git_utils import empty_tree_id, paths_changed_since  # noqa: E402
from utils.llm_client import render_diff_access  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.subagent_prompts import (  # noqa: E402
    CODE_REVIEW_VALIDATION_GUIDANCE,
    code_review_validation_guidance,
)
from utils.subagent_tasks import (  # noqa: E402
    TaskFileError,
    build_inputs,
    check_task_file_inputs,
    content_hash,
)
from utils.validation import prepare_batched_revalidation_tasks  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "code_review_head_ref_prompt.txt"
UPDATE_GOLDEN = os.environ.get("MULTI_LLM_UPDATE_GOLDEN") == "1"
MODEL = "claude-code:sonnet"

#: Fixed identity and clock, so the repository's SHAs (and the golden) are
#: the same on every machine.
GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_CONFIG_NOSYSTEM": "1",
}


# ---------------------------------------------------------------------------
# Repository and run helpers
# ---------------------------------------------------------------------------


class Repo:
    def __init__(self, root: Path):
        self.root = root
        self._tick = 0

    def git(self, *args: str, check: bool = True) -> str:
        env = {**os.environ, **GIT_ENV, "HOME": str(self.root.parent / "home")}
        result = subprocess.run(
            ["git", "-c", "commit.gpgsign=false", *args], cwd=self.root,
            capture_output=True, text=True, env=env,
        )
        if check and result.returncode != 0:
            raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
        return result.stdout.strip()

    def write(self, path: str, text: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def commit(self, message: str) -> str:
        self._tick += 1
        stamp = f"2026-01-01T00:{self._tick:02d}:00+00:00"
        self.git("add", "-A")
        env = {**os.environ, **GIT_ENV, "GIT_AUTHOR_DATE": stamp,
               "GIT_COMMITTER_DATE": stamp, "HOME": str(self.root.parent / "home")}
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", message],
            cwd=self.root, check=True, capture_output=True, text=True, env=env,
        )
        return self.git("rev-parse", "HEAD")

    @property
    def plan(self) -> Path:
        return self.root / "plans" / "feature.md"

    def state(self, plan: Optional[Path] = None) -> Dict[str, Any]:
        plan = plan or self.plan
        path = plan.parent / plan.stem / "state.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def phase_dir(self, plan: Optional[Path] = None) -> Path:
        plan = plan or self.plan
        return plan.parent / plan.stem / "code-review"


def lines(prefix: str, count: int, *, marked: Sequence[int] = (), mark: str = "X") -> str:
    return "".join(
        f"{prefix}{i}{mark if i in marked else ''}\n" for i in range(1, count + 1)
    )


def new_repo(tmp_path: Path) -> Repo:
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "home").mkdir(exist_ok=True)
    repo = Repo(root)
    repo.git("init", "-q", "-b", "main")
    repo.write("plans/feature.md", "# Feature\n\nChange a.py and b.py.\n")
    return repo


def issue(path: str, start: int, end: int, title: str, anchor: str = "x") -> Dict[str, Any]:
    return {
        "title": title, "desc": f"{title}: what is wrong", "importance": "medium",
        "file": path, "line_range": [start, end], "type": "bug", "anchor_text": anchor,
    }


class Reviewer:
    """The fake behind ``invoke_with_provider``: records every prompt, and
    answers with the findings a test gave it (or what *respond* returns)."""

    def __init__(self, findings: Optional[List[Dict[str, Any]]] = None,
                 respond: Optional[Callable[[str], List[Dict[str, Any]]]] = None):
        self.findings = findings or []
        self.respond = respond
        self.prompts: List[str] = []

    def __call__(self, prompt: str, model_spec: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.prompts.append(prompt)
        data = self.respond(prompt) if self.respond else list(self.findings)
        return {
            "success": True, "data": data,
            "details": {"provider": "claude-code", "model": "sonnet",
                        "duration_seconds": 0.1},
        }


def run_review(argv: List[str], reviewer: Optional[Reviewer] = None) -> int:
    """``review-code <argv>`` as the dispatcher would run it: the invocation is
    recorded first (so the run can rewrite it), then ``main`` in-process."""
    cli_common.record_invocation("review-code", list(argv))
    with patch("utils.llm_client.invoke_with_provider", side_effect=reviewer or Reviewer()):
        try:
            code = asyncio.run(cro.main(list(argv)))
        except SystemExit as exc:
            code = exc.code
    return 0 if code is None else int(code)


def review_argv(repo: Repo, *extra: str, plan: Optional[Path] = None) -> List[str]:
    return ["--plan-file", str(plan or repo.plan), "--models", MODEL, *extra]


def validate_all(repo: Repo, plan: Optional[Path] = None) -> None:
    """Play the host's half of validation: every batch answered ``valid``."""
    SkillRunner.simulate_subagents(repo.phase_dir(plan) / "validation_tasks.json")


def diff_access_of(prompt: str) -> str:
    return prompt.split("## Diff Access", 1)[1].split("## Your Task", 1)[0]


@pytest.fixture(autouse=True)
def _clean_invocation():
    cli_common.clear_invocation()
    yield
    cli_common.clear_invocation()


@pytest.fixture
def ranged(tmp_path, monkeypatch):
    """base: a.py, b.py, old.py; feature: edits a.py and b.py (one commit)."""
    repo = new_repo(tmp_path)
    repo.write("a.py", lines("a", 20))
    repo.write("b.py", lines("b", 20))
    repo.write("old.py", lines("old", 10))
    base = repo.commit("base")
    repo.git("checkout", "-q", "-b", "feature")
    repo.write("a.py", lines("a", 20, marked=(5,)))
    repo.write("b.py", lines("b", 20, marked=(7,)))
    head = repo.commit("feature")
    monkeypatch.chdir(repo.root)
    repo.base, repo.head = base, head
    return repo


# ---------------------------------------------------------------------------
# The parser and strict refs
# ---------------------------------------------------------------------------


class TestStrictRefs:
    def test_the_parser_takes_head_ref(self, tmp_path):
        args = cro.parse_args(["--plan-file", "x.md", "--base-ref", "A", "--head-ref", "B"])
        assert args.head_ref == "B"
        assert cro.parse_args(["--plan-file", "x.md"]).head_ref is None

    def test_an_unresolvable_head_ref_is_exit_3_with_nothing_written(self, ranged, capsys):
        reviewer = Reviewer()
        code = run_review(
            review_argv(ranged, "--base-ref", ranged.base, "--head-ref", "no-such-ref"),
            reviewer,
        )
        out = capsys.readouterr()
        assert code == 3
        assert "[RANGE_BAD_REF] no-such-ref does not name a commit" in out.out
        assert "HEAD~1" not in out.err + out.out
        assert reviewer.prompts == []
        assert not (ranged.plan.parent / "feature" / "state.json").exists()

    def test_an_unresolvable_base_ref_never_falls_back_to_head_1(self, ranged, capsys):
        code = run_review(
            review_argv(ranged, "--base-ref", "nope", "--head-ref", ranged.head)
        )
        out = capsys.readouterr()
        assert code == 3
        assert "[RANGE_BAD_REF] nope" in out.out
        assert "falling back to HEAD~1" not in out.err

    def test_a_head_ref_without_a_base_never_takes_head_at_start(self, ranged, capsys):
        # A plan whose state was created at the reviewed tip: StateManager
        # stamped `head_at_start` from HEAD, so taking it as the base would be
        # an empty `head..head` review (§2.2 "No site tests base for truth").
        args = cro.parse_args(["--plan-file", str(ranged.plan), "--head-ref", ranged.head])
        with pytest.raises(cro.RangeError) as caught:
            cro.resolve_pinned_range(
                {"head_at_start": ranged.head}, args,
                cwd=str(ranged.root), plan_path=ranged.plan,
            )
        assert caught.value.tag == "[RANGE_BAD_REF]" and caught.value.exit_code == 3
        assert "--head-ref needs a base" in caught.value.message
        # implement's own `head_before_implement` still names the base.
        pinned = cro.resolve_pinned_range(
            {"head_at_start": ranged.head, "head_before_implement": ranged.base}, args,
            cwd=str(ranged.root), plan_path=ranged.plan,
        )
        assert (pinned.base, pinned.head, pinned.source) == (
            ranged.base, ranged.head, cro.PIN_TYPED,
        )

        reviewer = Reviewer()
        code = run_review(review_argv(ranged, "--head-ref", ranged.head), reviewer)
        out = capsys.readouterr()
        assert code == 3
        assert "[RANGE_BAD_REF] --head-ref needs a base" in out.out
        assert reviewer.prompts == []

    def test_an_empty_tree_base_reviews_a_root_range_strictly(self, tmp_path, monkeypatch, capsys):
        repo = new_repo(tmp_path)
        repo.write("a.py", lines("a", 5))
        root_commit = repo.commit("root")
        monkeypatch.chdir(repo.root)
        empty = empty_tree_id(str(repo.root))

        reviewer = Reviewer()
        code = run_review(
            review_argv(repo, "--base-ref", empty, "--head-ref", root_commit,
                        "--skip-validation"),
            reviewer,
        )
        out = capsys.readouterr()
        assert code == 0, out.out + out.err
        assert "HEAD~1" not in out.err
        assert f"Git base reference: {empty}" in out.out
        # The whole root commit is the change; the plan is the workflow's own.
        assert "Using the 1 file(s) changed in root.." in out.out
        # The empty tree is never abbreviated: git knows it only by its full id.
        assert f"`git diff --name-only {empty} {root_commit[:12]}`" in out.out
        record = repo.state()["head_ref"]
        assert record["base"] is None and record["head"] == root_commit
        manifest = (repo.phase_dir() / "review-scope.md").read_text(encoding="utf-8")
        assert "empty tree: the range starts at the root commit" in manifest
        access = diff_access_of(reviewer.prompts[0])
        assert f"`root..{root_commit[:7]}`" in access
        assert "the diff shows each whole file as new" in access


# ---------------------------------------------------------------------------
# Scope: the range decides, and nothing guesses at the working tree
# ---------------------------------------------------------------------------


class TestCommitRangeScope:
    def test_the_scope_is_the_ranges_diff_and_nothing_on_disk(self, ranged, capsys):
        # Uncommitted and untracked work, a baseline and a base far behind HEAD:
        # each would change today's working-tree review, none may touch this one.
        ranged.write("old.py", lines("old", 10, marked=(1,)))
        ranged.write("untracked.py", "print('new')\n")
        ranged.write("c.py", "x\n")
        ranged.commit("later work on HEAD")
        ranged.write("c.py", "y\n")
        sm = StateManager(ranged.plan)
        sm.state["pre_existing_changes"] = ["a.py"]
        sm.save()
        index_before = ranged.git("ls-files", "-s")

        reviewer = Reviewer()
        # Two commits need no intent-to-add marking: never entered.
        with patch.object(cro, "intent_to_add_untracked",
                          side_effect=AssertionError("intent-to-add ran")):
            code = run_review(
                review_argv(ranged, "--base-ref", ranged.base, "--head-ref",
                            ranged.head, "--skip-validation"),
                reviewer,
            )
        out = capsys.readouterr().out
        assert code == 0
        assert "Files to review: 2" in out
        assert "[SCOPE_FALLBACK]" not in out
        assert "pre-existing files" not in out
        assert "--base-ref HEAD" not in out
        assert "behind HEAD" not in out
        assert ranged.git("ls-files", "-s") == index_before  # no intent-to-add
        manifest = (ranged.phase_dir() / "review-scope.md").read_text(encoding="utf-8")
        assert "`a.py`" in manifest and "`b.py`" in manifest
        assert "untracked.py" not in manifest and "c.py" not in manifest
        assert f"Head ref: `{ranged.head}`" in manifest
        scope_line = next(
            line for line in reviewer.prompts[0].splitlines()
            if line.startswith("Where this list came from:")
        )
        assert f"`git diff --name-only {ranged.base[:12]} {ranged.head[:12]}`" in scope_line
        assert f"ends at commit `{ranged.head[:7]}`, not at the working tree" in scope_line

    def test_files_narrow_the_range_and_never_widen_it(self, ranged, capsys):
        argv = review_argv(
            ranged, "--base-ref", ranged.base, "--head-ref", ranged.head,
            "--files", "a.py", "--files", "z.py", "--skip-validation",
        )
        reviewer = Reviewer([issue("a.py", 5, 5, "A")])
        assert run_review(argv, reviewer) == 0
        out = capsys.readouterr().out
        shown = f"{ranged.base[:7]}..{ranged.head[:7]}"
        assert "Files to review: 1" in out
        assert f"  - z.py: no diff: not changed in {shown}" in out
        assert ranged.state()["review_scope_files"] == ["a.py", "z.py"]
        manifest = (ranged.phase_dir() / "review-scope.md").read_text(encoding="utf-8")
        assert "`b.py`" not in manifest.split("## Source", 1)[1]
        assert f"`z.py` (no diff: not changed in {shown})" in manifest
        assert f"`z.py` (no diff: not changed in {shown})" in reviewer.prompts[0]
        # The replay the status resume line is built from keeps --files and
        # carries the pair as SHAs.
        flags = cli_common.current_replay_flags(mode="review-code")
        assert flags[flags.index("--head-ref") + 1] == ranged.head
        assert "--files" in flags

        # --reaggregate regroups that same narrowed scope.
        assert run_review(review_argv(
            ranged, "--base-ref", ranged.base, "--head-ref", ranged.head,
            "--files", "a.py", "--reaggregate", "--skip-validation",
        )) == 0
        out = capsys.readouterr().out
        assert "[SCOPE_FLAGS_IGNORED]" in out
        assert "Files to review: 1" in out

        # A resume typed without --files clears the record, reviews the whole
        # range and re-runs the narrowed result.
        again = Reviewer([issue("b.py", 7, 7, "B")])
        assert run_review(review_argv(
            ranged, "--base-ref", ranged.base, "--head-ref", ranged.head,
            "--force", "--skip-validation",
        ), again) == 0
        out = capsys.readouterr().out
        assert "review_scope_files" not in ranged.state()
        assert "Files to review: 2" in out
        assert "[SCOPE_CHANGED]" in out
        assert len(again.prompts) == 1


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


class TestDrift:
    def test_a_direct_run_prints_and_records_exactly_the_change_test(
        self, tmp_path, monkeypatch, capsys,
    ):
        repo = new_repo(tmp_path)
        for name in ("edited", "skipped", "moved", "renamed_onto", "clean"):
            repo.write(f"{name}.py", lines(name, 5))
        repo.write("gone_untracked.py", "g\n")
        repo.write("gone_intent.py", "i\n")
        repo.write("deleted_target.py", "d\n")
        base = repo.commit("base")
        for name in ("edited", "skipped", "moved", "renamed_onto", "clean"):
            repo.write(f"{name}.py", lines(name, 5, marked=(1,)))
        for name in ("gone_untracked.py", "gone_intent.py", "deleted_target.py"):
            (repo.root / name).unlink()
        head = repo.commit("range")
        monkeypatch.chdir(repo.root)
        repo.git("config", "diff.renames", "true")
        # After head: an edit, a deleted file back untracked and one back with
        # `git add -N`, a skip-worktree entry edited on disk, a `git mv`, and a
        # scope file moved onto a path the range deleted.
        repo.write("edited.py", "edited later\n")
        repo.write("gone_untracked.py", "back\n")
        repo.write("gone_intent.py", "back\n")
        repo.git("add", "-N", "gone_intent.py")
        repo.git("update-index", "--skip-worktree", "skipped.py")
        repo.write("skipped.py", "hidden edit\n")
        repo.git("mv", "moved.py", "moved_elsewhere.py")
        repo.git("mv", "renamed_onto.py", "deleted_target.py")

        reviewer = Reviewer()
        assert run_review(review_argv(
            repo, "--base-ref", base, "--head-ref", head, "--skip-validation",
        ), reviewer) == 0
        out = capsys.readouterr().out
        scope = sorted(
            line.split("`")[1] for line in
            (repo.phase_dir() / "review-scope.md").read_text(encoding="utf-8").splitlines()
            if re.match(r"^\d+\. `", line)
        )
        expected = sorted(paths_changed_since(head, scope, cwd=str(repo.root)))
        assert expected == [
            "deleted_target.py", "edited.py", "gone_intent.py", "gone_untracked.py",
            "moved.py", "renamed_onto.py", "skipped.py",
        ]
        drift = [line for line in out.splitlines() if line.startswith("[SCOPE_DRIFT]")]
        assert drift == [
            f"[SCOPE_DRIFT] {len(expected)} of {len(scope)} reviewed files differ on "
            f"disk from {head[:7]}: " + ", ".join(expected)
        ]
        assert repo.state()["head_ref"]["drifted"] == expected
        access = diff_access_of(reviewer.prompts[0])
        assert f"have changed since `{head[:7]}`" in access
        assert "`edited.py`" in access and "`clean.py`" not in access

    def test_a_fronted_run_takes_the_front_doors_list_and_prints_none(
        self, ranged, capsys,
    ):
        brief = seed_commit_brief(ranged, drifted=["b.py"])
        ranged.write("a.py", "edited on disk, not in the recorded list\n")
        reviewer = Reviewer()
        assert run_review(review_argv(
            ranged, "--base-ref", ranged.base, "--head-ref", ranged.head,
            "--skip-validation", plan=brief,
        ), reviewer) == 0
        out = capsys.readouterr().out
        assert "[SCOPE_DRIFT]" not in out
        access = diff_access_of(reviewer.prompts[0])
        assert "`b.py`" in access.split("have changed since", 1)[1].split("\n", 1)[0]
        assert "`a.py`" not in access.split("have changed since", 1)[1].split("\n", 1)[0]


# ---------------------------------------------------------------------------
# Fronted runs: the commit_range record
# ---------------------------------------------------------------------------


def seed_commit_brief(
    repo: Repo, *, drifted: Sequence[str] = (), base: Optional[str] = "<base>",
    head: Optional[str] = None, plan_snapshot: Optional[str] = None,
    plan_sha256: Optional[str] = None,
) -> Path:
    """What the front door (another agent's) leaves before the phase runs: the
    brief and its ``state.json`` with a ``commit_range`` record."""
    base_sha = repo.base if base == "<base>" else base
    head_sha = head or repo.head
    slug = f"{(base_sha or 'root')[:12]}-{head_sha[:12]}"
    brief = repo.root / "plans" / "commit-reviews" / f"{slug}.md"
    brief.parent.mkdir(parents=True, exist_ok=True)
    (brief.parent / ".gitignore").write_text("*\n", encoding="utf-8")
    brief.write_text(
        "<!-- multi-llm:commit-brief v1 -->\n# Commit review\n"
        "<!-- /multi-llm:commit-brief -->\n", encoding="utf-8",
    )
    sm = StateManager(brief)
    sm.state["commit_range"] = {
        "base": base_sha, "head": head_sha,
        "commits": [{"sha": head_sha, "date": "2026-01-01", "subject": "feature",
                     "merge": False, "outside_window": False}],
        "phrase": None, "window": None, "notes": [],
        "size": {"commits": 1, "files": 2, "added": 2, "deleted": 2, "lines": 4,
                 "large": False},
        "brief": str(brief.relative_to(repo.root)),
        "plan": "docs/context.md" if plan_snapshot else None,
        "plan_snapshot": plan_snapshot, "plan_sha256": plan_sha256,
        "focus": None, "header_sha256": "header-1",
        "reviewed_header_sha256": None, "drifted": list(drifted),
        "resolved_at": "2026-01-01T00:00:00Z",
        "guard": {"head": head_sha, "decided_at": "x", "skipped": {}, "expect": {},
                  "protected": {}, "protected_copies": "x", "context_plan": None},
    }
    sm.save()
    return brief


class TestFrontedRecords:
    def test_a_round_start_records_the_header_and_clears_the_guard(self, ranged):
        brief = seed_commit_brief(ranged)
        argv = review_argv(
            ranged, "--base-ref", ranged.base, "--head-ref", ranged.head,
            "--skip-validation", plan=brief,
        )
        assert run_review(argv, Reviewer([issue("a.py", 5, 5, "A")])) == 0
        state = ranged.state(brief)
        assert state["commit_range"]["reviewed_header_sha256"] == "header-1"
        assert "guard" not in state["commit_range"]
        assert "head_ref" not in state  # a fronted run writes none

        # A resume of its own round (the result is reused) writes neither.
        sm = StateManager(brief)
        sm.state["commit_range"]["header_sha256"] = "header-2"
        sm.state["commit_range"]["guard"] = {"head": ranged.head}
        sm.save()
        assert run_review(argv[:] + ["--force"], Reviewer()) == 0
        record = ranged.state(brief)["commit_range"]
        assert record["reviewed_header_sha256"] == "header-1"
        assert record["guard"] == {"head": ranged.head}

        # --force --rerun-all restarts the round: both fields move.
        assert run_review(argv + ["--force", "--rerun-all"], Reviewer()) == 0
        record = ranged.state(brief)["commit_range"]
        assert record["reviewed_header_sha256"] == "header-2"
        assert "guard" not in record

    def test_a_typed_pair_that_is_not_the_briefs_is_refused(self, ranged, capsys):
        brief = seed_commit_brief(ranged)
        code = run_review(review_argv(
            ranged, "--base-ref", ranged.base, "--head-ref", ranged.base, plan=brief,
        ))
        out = capsys.readouterr().out
        assert code == 3
        assert "[HEAD_REF_MISMATCH]" in out
        assert f"--base-ref {ranged.base} --head-ref {ranged.head}" in out

    def test_the_frozen_context_plan_is_named_and_bound(self, ranged, capsys):
        snapshot_rel = "plans/commit-reviews/ctx/context-plan.md"
        snapshot = ranged.root / snapshot_rel
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"# Context plan\n")
        digest = hashlib.sha256(b"# Context plan\n").hexdigest()
        brief = seed_commit_brief(ranged, plan_snapshot=snapshot_rel, plan_sha256=digest)
        argv = review_argv(ranged, "--base-ref", ranged.base, "--head-ref", ranged.head,
                           plan=brief)
        assert run_review(argv, Reviewer([issue("a.py", 5, 5, "A")])) == 0
        tasks = json.loads(
            (ranged.phase_dir(brief) / "validation_tasks.json").read_text(encoding="utf-8")
        )
        assert tasks["inputs"]["context_plan_sha256"] == digest
        assert (ranged.phase_dir(brief) / tasks["context_plan_file"]).resolve() == snapshot.resolve()
        prompt = tasks["batches"][0]["prompt"]
        assert f"`{snapshot_rel}`" in prompt
        assert "docs/context.md" not in prompt

        validate_all(ranged, brief)
        snapshot.write_bytes(b"# Context plan, edited\n")
        code = run_review(argv + ["--reaggregate"])
        err = capsys.readouterr().err
        assert code == 3
        assert "context_plan_sha256 (the context plan's frozen snapshot" in err

    @pytest.mark.parametrize("in_memory", [False, True])
    @pytest.mark.parametrize("damage", ["changed", "missing"])
    def test_executor_refuses_a_changed_context_snapshot(self, ranged, monkeypatch, in_memory, damage):
        from utils import subagent_executor as executor

        snapshot = ranged.root / "context-plan.md"
        snapshot.write_bytes(b"# Frozen context\r\n")
        brief = seed_commit_brief(ranged, plan_snapshot=snapshot.name,
                                  plan_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
        argv = review_argv(ranged, "--base-ref", ranged.base, "--head-ref", ranged.head, plan=brief)
        assert run_review(argv, Reviewer([issue("a.py", 5, 5, "A")])) == 0
        phase = ranged.phase_dir(brief)
        task_file = phase / "validation_tasks.json"
        doc = json.loads(task_file.read_text(encoding="utf-8"))
        calls = []
        monkeypatch.setattr(executor, "run_tasks", lambda *a, **kw: calls.append(a))
        kwargs = {"document": doc} if in_memory else {}
        executor.run_task_file(str(task_file), phase_dir=str(phase), **kwargs)
        assert len(calls) == 1
        if damage == "changed":
            snapshot.write_bytes(b"# Different context\n")
        else:
            snapshot.unlink()
        with pytest.raises(executor.ExecutorError, match="context_plan_sha256"):
            executor.run_task_file(str(task_file), phase_dir=str(phase), **kwargs)
        assert len(calls) == 1

    def test_a_new_fanout_refuses_a_changed_context_snapshot(self, ranged):
        snapshot = ranged.root / "context-plan.md"
        snapshot.write_bytes(b"# Frozen context\n")
        brief = seed_commit_brief(ranged, plan_snapshot=snapshot.name,
                                  plan_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
        snapshot.write_bytes(b"# Changed snapshot\n")
        reviewer = Reviewer()
        assert run_review(review_argv(
            ranged, "--base-ref", ranged.base, "--head-ref", ranged.head, plan=brief,
        ), reviewer) == 3
        assert reviewer.prompts == []

    def test_revalidation_binds_the_frozen_context_plan(self, ranged):
        snapshot = ranged.root / "context-plan.md"
        snapshot.write_bytes(b"# Frozen context\n")
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        brief = seed_commit_brief(ranged, plan_snapshot=snapshot.name, plan_sha256=digest)
        out = ranged.root / "apply"
        out.mkdir()
        tasks = prepare_batched_revalidation_tasks(
            groups=[{"theme": "T", "suggestions": [issue("a.py", 5, 5, "A")]}],
            validation_results=[{"status": "validation_failed"}],
            context="diff --git a/a.py b/a.py\n", output_dir=str(out),
            plan_file=str(brief), context_kind="diff",
        )
        assert tasks["inputs"]["context_plan_sha256"] == digest
        assert (out / tasks["context_plan_file"]).resolve() == snapshot.resolve()

    def test_an_inline_lane_is_shown_the_frozen_context_plan(self, ranged, capsys):
        """A lane that cannot open files gets the snapshot's text in its
        payload (§2.4 "Everyone reads the snapshot"); when it cannot be shown,
        the lane is not dispatched rather than judging the commits alone."""
        snapshot_rel = "plans/commit-reviews/ctx/context-plan.md"
        snapshot = ranged.root / snapshot_rel
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        body = b"# Context plan\n\nEvery edit must keep b.py untouched.\n"
        snapshot.write_bytes(body)
        brief = seed_commit_brief(ranged, plan_snapshot=snapshot_rel,
                                  plan_sha256=hashlib.sha256(body).hexdigest())
        argv = ["--plan-file", str(brief), "--models", "qwen-code:qwen3-coder",
                "--base-ref", ranged.base, "--head-ref", ranged.head, "--skip-validation"]

        reviewer = Reviewer()
        assert run_review(argv, reviewer) == 0
        assert len(reviewer.prompts) == 1
        prompt = reviewer.prompts[0]
        access = diff_access_of(prompt)
        assert ("===== BEGIN context-plan.md =====\n# Context plan\n\n"
                "Every edit must keep b.py untouched.\n===== END context-plan.md =====") in access
        assert "`context-plan.md` block under Diff Access below" in " ".join(prompt.split())
        assert "commits alone" not in prompt
        assert str(snapshot) not in prompt

        # The same lane with the snapshot gone is refused, not dispatched.
        snapshot.unlink()
        reviewer = Reviewer()
        assert run_review(argv + ["--force", "--rerun-all"], reviewer) == 3
        assert reviewer.prompts == []
        assert "[CONTEXT_PLAN_SNAPSHOT_CHANGED]" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Unresolvable recorded SHAs
# ---------------------------------------------------------------------------


def prune_unreachable(repo: Repo) -> None:
    repo.git("reflog", "expire", "--expire=now", "--all")
    repo.git("gc", "-q", "--prune=now")


class TestRecordedShasResolveStrictly:
    def _assert_refused(self, repo: Repo, brief: Path, argv: List[str], sha: str,
                        which: str, capsys, recovery: str) -> None:
        reviewer = Reviewer()
        code = run_review(argv, reviewer)
        out = capsys.readouterr()
        assert code == 3, out.out + out.err
        assert f"[RANGE_UNRESOLVED] {which} {sha[:12]} recorded in" in out.out
        assert recovery in out.out
        assert "HEAD~1" not in out.out + out.err
        assert reviewer.prompts == []
        phase = repo.phase_dir(brief)
        for name in ("review-scope.md", "report.md", "validation_tasks.json"):
            assert not (phase / name).exists(), name

    def test_a_pruned_head_refuses_resume_and_reaggregate(self, ranged, capsys):
        ranged.write("a.py", lines("a", 20, marked=(5, 6)))
        doomed = ranged.commit("amended away")
        brief = seed_commit_brief(ranged, head=doomed)
        ranged.git("reset", "-q", "--hard", ranged.head)
        prune_unreachable(ranged)
        assert ranged.git("cat-file", "-t", doomed, check=False) == ""

        argv = review_argv(ranged, "--base-ref", ranged.base, "--head-ref", doomed,
                           plan=brief)
        self._assert_refused(ranged, brief, argv, doomed, "head", capsys,
                             "--review-commits")
        self._assert_refused(ranged, brief, argv + ["--reaggregate"], doomed, "head",
                             capsys, "--review-commits")

    def test_a_pruned_base_refuses_too(self, ranged, capsys):
        ranged.git("checkout", "-q", "--orphan", "scratch")
        ranged.write("z.py", "z\n")
        doomed_base = ranged.commit("orphan base")
        ranged.git("checkout", "-q", "-f", "feature")
        ranged.git("branch", "-q", "-D", "scratch")
        brief = seed_commit_brief(ranged, base=doomed_base)
        prune_unreachable(ranged)

        argv = review_argv(ranged, "--base-ref", doomed_base, "--head-ref", ranged.head,
                           plan=brief)
        self._assert_refused(ranged, brief, argv, doomed_base, "base", capsys,
                             "--review-commits")
        self._assert_refused(ranged, brief, argv + ["--reaggregate"], doomed_base,
                             "base", capsys, "--review-commits")

    def test_a_direct_runs_pruned_head_names_review_code(self, ranged, capsys):
        ranged.write("a.py", lines("a", 20, marked=(9,)))
        doomed = ranged.commit("will vanish")
        assert run_review(review_argv(
            ranged, "--base-ref", ranged.base, "--head-ref", doomed, "--skip-validation",
        )) == 0
        replay = list(cli_common.invocation()["argv"])
        ranged.git("reset", "-q", "--hard", ranged.head)
        prune_unreachable(ranged)
        capsys.readouterr()
        for extra in (["--force"], ["--reaggregate"]):
            code = run_review(replay + extra)
            out = capsys.readouterr()
            assert code == 3
            assert f"[RANGE_UNRESOLVED] head {doomed[:12]} recorded in" in out.out
            assert "review-code" in out.out and "--head-ref <rev>" in out.out
            assert "HEAD~1" not in out.err


# ---------------------------------------------------------------------------
# Direct runs: the head_ref record, replay and reaggregate after a move
# ---------------------------------------------------------------------------


class TestDirectRun:
    def test_head_ref_record_replay_and_a_moved_tree(self, ranged, capsys):
        reviewer = Reviewer([issue("a.py", 5, 5, "A"), issue("b.py", 7, 7, "B")])
        assert run_review(review_argv(
            ranged, "--base-ref", "main", "--head-ref", "feature",
        ), reviewer) == 0
        out = capsys.readouterr().out
        state = ranged.state()
        assert state["head_ref"]["base"] == ranged.base
        assert state["head_ref"]["head"] == ranged.head
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", state["head_ref"]["resolved_at"])
        assert "commit_range" not in state
        argv = cli_common.invocation()["argv"]
        assert argv[argv.index("--base-ref") + 1] == ranged.base
        assert argv[argv.index("--head-ref") + 1] == ranged.head
        assert "feature" not in argv and "main" not in argv
        # The printed continuation is review-code's own, with the SHAs, and
        # calls the plan by its path.
        continuation = next(
            line for line in out.splitlines() if "--reaggregate" in line
        )
        assert f"review-code {ranged.plan} " in continuation
        assert f"--base-ref {ranged.base} --head-ref {ranged.head}" in continuation
        tasks_path = ranged.phase_dir() / "validation_tasks.json"
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert tasks["inputs"]["diff_base"] == f"{ranged.base}..{ranged.head}"
        recorded_hash = tasks["inputs"]["reviewed_code_sha256"]
        validate_all(ranged)

        # The branch moves on and a reviewed file is edited on disk.
        ranged.write("a.py", lines("a", 20, marked=(1, 2, 3)))
        moved = ranged.commit("feature moves")
        ranged.write("b.py", "rewritten on disk\n")
        replay = list(argv)

        # A --reaggregate naming the moved branch is refused, naming the pair.
        code = run_review(review_argv(
            ranged, "--head-ref", "feature", "--reaggregate",
        ))
        out = capsys.readouterr().out
        assert code == 3
        assert "[HEAD_REF_MISMATCH]" in out
        assert f"--base-ref {ranged.base} --head-ref {ranged.head}" in out

        # The replayed reaggregate uses the recorded pair: the excerpt is
        # re-cut at head, so its hash still binds the stored task file.
        assert run_review(replay + ["--reaggregate"]) == 0
        out = capsys.readouterr().out
        assert "Reaggregation complete!" in out
        assert ranged.state()["phases_completed"].get("code-review")
        assert json.loads(tasks_path.read_text(encoding="utf-8"))["inputs"][
            "reviewed_code_sha256"] == recorded_hash
        pinned = cro.PinnedRange(ranged.base, ranged.head, ranged.base, cro.PIN_TYPED)
        grouped = json.loads((ranged.phase_dir() / "grouped.json").read_text())
        groups = grouped["groups"] if isinstance(grouped, dict) else grouped
        assert content_hash(cro.build_reviewed_code_excerpt(
            groups, project_root=str(ranged.root), pinned=pinned,
        )) == recorded_hash

        # A resume replays the same pair, not the moved branch.
        assert run_review(replay + ["--force", "--skip-validation"], Reviewer()) == 0
        manifest = (ranged.phase_dir() / "review-scope.md").read_text(encoding="utf-8")
        assert f"Head ref: `{ranged.head}`" in manifest and moved not in manifest
        assert ranged.state()["head_ref"]["head"] == ranged.head

        # A later round without --head-ref is a working-tree review again.
        capsys.readouterr()
        assert run_review(review_argv(
            ranged, "--base-ref", ranged.base, "--force", "--rerun-all",
            "--skip-validation",
        ), Reviewer()) == 0
        assert "head_ref" not in ranged.state()


# ---------------------------------------------------------------------------
# The pinned-head read rule, and an out-of-scope dependency
# ---------------------------------------------------------------------------


class TestPinnedHeadReadRule:
    def test_an_out_of_scope_dependency_is_read_at_head(self, tmp_path, monkeypatch, capsys):
        repo = new_repo(tmp_path)
        repo.write("b.py", "VALUE = 'at head'\n")
        repo.write("a.py", "import b\n")
        base = repo.commit("base")
        repo.write("a.py", "import b\nprint(b.VALUE)\n")
        head = repo.commit("a uses b")
        monkeypatch.chdir(repo.root)
        repo.write("b.py", "VALUE = 'edited on disk'\n")

        seen: Dict[str, str] = {}

        def follow_the_rule(prompt: str) -> List[Dict[str, Any]]:
            command = re.search(r"`git show ([0-9a-f]{12}):<path>`", prompt)
            assert command, "no pinned-head read rule in the prompt"
            seen["b.py"] = subprocess.run(
                ["git", "show", f"{command.group(1)}:b.py"], cwd=repo.root,
                capture_output=True, text=True, check=True,
            ).stdout
            return [issue("a.py", 2, 2, "Uses b", "print(b.VALUE)")]

        reviewer = Reviewer(respond=follow_the_rule)
        assert run_review(review_argv(
            repo, "--base-ref", base, "--head-ref", head,
        ), reviewer) == 0
        out = capsys.readouterr().out
        assert "[SCOPE_DRIFT]" not in out
        assert seen["b.py"] == "VALUE = 'at head'\n"

        rule = pinned_head_read_rule(base, head)
        prompts = [reviewer.prompts[0]] + [
            task["prompt"] for task in json.loads(
                (repo.phase_dir() / "validation_tasks.json").read_text(encoding="utf-8")
            )["batches"]
        ]
        for prompt in prompts:
            assert rule in prompt
            assert DATA_UNDER_REVIEW in prompt
            assert "reviewed content" not in prompt
            assert "cat path/to/file" not in prompt
            assert "current content" not in prompt
        access = diff_access_of(reviewer.prompts[0])
        assert access.strip().startswith(
            f"The change under review is `{base[:7]}..{head[:7]}`"
        )
        assert f"ending at commit `{head[:7]}`, not at the working tree" in access

    @pytest.mark.parametrize("model", ["codex:gpt-5", "claude-code:sonnet", "qwen-code:qwen3"])
    def test_every_diff_access_state_states_the_rule(self, tmp_path, model):
        phase = tmp_path / "plans" / "p" / "code-review"
        (phase / "diff").mkdir(parents=True)
        scope = phase / "review-scope.md"
        scope.write_text("# Review scope\n", encoding="utf-8")
        plan = tmp_path / "plans" / "p.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        base, head = "1" * 40, "2" * 40
        access = render_diff_access(
            model, scope, phase / "diff", phase, tmp_path, base_ref=base,
            coverage="complete", plan_path=plan, head_ref=head, drifted=["x.py"],
        )
        assert access.mode in ("paths", "inline"), access.reason
        text = access.text
        assert text.startswith(f"The change under review is `{base[:7]}..{head[:7]}`")
        assert pinned_head_read_rule(base, head) in text
        assert "`x.py`" in text.split("\n", 1)[0]
        assert "cat path/to/file" not in text and "reviewed content" not in text
        if model.startswith("codex"):
            assert f"`git diff {base} {head} -- path/to/file`" in text
            assert f"`git show {head[:12]}:path/to/file`" in text

    def test_an_unpinned_diff_access_is_unchanged(self, tmp_path):
        phase = tmp_path / "code-review"
        (phase / "diff").mkdir(parents=True)
        scope = phase / "review-scope.md"
        scope.write_text("x", encoding="utf-8")
        kwargs = dict(base_ref="abc", coverage="complete", plan_path=tmp_path / "p.md")
        plain = render_diff_access("codex:gpt-5", scope, phase / "diff", phase, tmp_path, **kwargs)
        assert "cat path/to/file" in plain.text
        assert "Read code at commit" not in plain.text


# ---------------------------------------------------------------------------
# Paths absent at head
# ---------------------------------------------------------------------------


class TestPathsAbsentAtHead:
    def test_excerpts_labels_report_and_reaggregate(self, tmp_path, monkeypatch, capsys):
        repo = new_repo(tmp_path)
        repo.write("old.py", lines("old", 12))
        repo.write("a.py", lines("a", 12))
        repo.write("keep.py", lines("keep", 3))
        base = repo.commit("base")
        repo.git("rm", "-q", "old.py")
        repo.git("mv", "a.py", "b.py")
        repo.write("new.py", lines("new", 4))
        repo.write("keep.py", lines("keep", 3, marked=(2,)))
        head = repo.commit("delete, rename, add")
        monkeypatch.chdir(repo.root)
        findings = [
            issue("old.py", 4, 5, "Removed guard", "old4"),
            issue("a.py", 2, 2, "Renamed away", "a2"),
            issue("b.py", 3, 3, "In the new name", "a3"),
            issue("new.py", 1, 2, "New file", "new1"),
            issue("ghost.py", 1, 1, "Made up", "ghost"),
        ]
        argv = review_argv(repo, "--base-ref", base, "--head-ref", head)
        assert run_review(argv, Reviewer(findings)) == 0
        replay = list(cli_common.invocation()["argv"])
        tasks_path = repo.phase_dir() / "validation_tasks.json"
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        excerpt = "\n".join(task["prompt"] for task in tasks["batches"])
        rng = f"{base[:7]}..{head[:7]}"
        assert f"### old.py (deleted in {rng}; lines as at {base[:7]})" in excerpt
        assert f"### a.py (renamed to b.py in {rng}; lines as at {base[:7]})" in excerpt
        assert "\n1: old1\n" in excerpt and "\n12: old12\n" in excerpt
        assert "### b.py\n\n```\n1: a1\n" in excerpt
        assert "### new.py\n\n```\n1: new1\n" in excerpt
        assert f"### ghost.py\n\n(absent at {base[:7]} and {head[:7]}; no code to show)" in excerpt
        recorded_hash = tasks["inputs"]["reviewed_code_sha256"]
        validate_all(repo)

        # old.py comes back on disk with other bytes: no report may show them,
        # and the excerpt (so its hash) cannot move.
        repo.write("old.py", "WORKTREE_BYTES = 1\n" * 20)
        repo.write("a.py", "WORKTREE_BYTES = 2\n" * 20)
        assert run_review(replay + ["--reaggregate"]) == 0
        assert "Reaggregation complete!" in capsys.readouterr().out
        assert json.loads(tasks_path.read_text(encoding="utf-8"))["inputs"][
            "reviewed_code_sha256"] == recorded_hash
        report_data = (repo.phase_dir() / "report_data.json").read_text(encoding="utf-8")
        html = (repo.phase_dir() / "report.html").read_text(encoding="utf-8")
        assert "WORKTREE_BYTES" not in report_data
        assert "WORKTREE_BYTES" not in html
        contexts = json.loads(report_data)["fileContexts"]
        assert contexts["old.py"]["label"] == f"deleted in {rng}"
        assert contexts["a.py"]["label"] == f"renamed to b.py in {rng}"
        assert contexts["ghost.py"]["label"] == f"absent at {head[:7]}"
        assert "label" not in contexts.get("new.py", {})


# ---------------------------------------------------------------------------
# The golden: a direct run's code_review.txt rendering
# ---------------------------------------------------------------------------


class TestGolden:
    def test_the_direct_run_prompt_matches_its_golden(self, ranged):
        reviewer = Reviewer()
        assert run_review(review_argv(
            ranged, "--base-ref", ranged.base, "--head-ref", ranged.head,
            "--skip-validation",
        ), reviewer) == 0
        prompt = reviewer.prompts[0].replace(str(ranged.root), "<ROOT>")
        assert "PROMPT_FORMAT_ERROR" not in prompt
        access = diff_access_of(prompt)
        assert f"`{ranged.base[:7]}..{ranged.head[:7]}`" in access
        assert f"ending at commit `{ranged.head[:7]}`" in access
        assert pinned_head_read_rule(ranged.base, ranged.head) in access
        assert DATA_UNDER_REVIEW in access
        if UPDATE_GOLDEN or not GOLDEN.exists():
            GOLDEN.write_text(prompt, encoding="utf-8")
            if not UPDATE_GOLDEN:
                pytest.fail(f"created missing golden {GOLDEN.name}; re-run")
        assert prompt == GOLDEN.read_text(encoding="utf-8"), (
            f"the direct run's prompt drifted from {GOLDEN}. If the change is "
            f"intended, re-run with MULTI_LLM_UPDATE_GOLDEN=1."
        )


# ---------------------------------------------------------------------------
# Units: the guidance, the inputs block, the recorded argv
# ---------------------------------------------------------------------------


class TestUnits:
    def test_unpinned_guidance_is_the_constant(self):
        assert code_review_validation_guidance() == CODE_REVIEW_VALIDATION_GUIDANCE

    def test_pinned_guidance_states_the_rule_and_names_the_snapshot(self):
        rule = pinned_head_read_rule("a" * 40, "b" * 40)
        text = code_review_validation_guidance(
            read_rule=rule, head="b" * 40, change="aaaaaaa..bbbbbbb",
            plan_snapshot="plans/x/context-plan.md", plan_sha256="c" * 64,
        )
        assert rule in text and DATA_UNDER_REVIEW in text
        assert "`plans/x/context-plan.md` (sha256 `cccccccccccc`)" in text
        assert "open those locations and compare" not in text

    def test_context_plan_sha256_has_its_own_refusal_reason(self, tmp_path):
        inputs = build_inputs(plan_file=None, context_plan_sha256="a" * 64)
        assert inputs["context_plan_sha256"] == "a" * 64
        assert "context_plan_sha256" not in build_inputs(plan_file=None)
        with pytest.raises(TaskFileError) as raised:
            check_task_file_inputs(
                {"inputs": inputs}, "t.json", str(tmp_path),
                {"context_plan_sha256": "b" * 64},
            )
        assert "the context plan's frozen snapshot no longer hashes" in str(raised.value)

    def test_a_pinned_diff_revalidation_carries_the_rule(self, ranged):
        StateManager(ranged.plan).save()
        sm = StateManager(ranged.plan)
        sm.state["head_ref"] = {"base": ranged.base, "head": ranged.head,
                                "drifted": [], "resolved_at": "x"}
        sm.save()
        out = ranged.root / "apply"
        out.mkdir()
        groups = [{"theme": "T", "suggestions": [issue("a.py", 5, 5, "A")]}]
        tasks = prepare_batched_revalidation_tasks(
            groups=groups,
            validation_results=[{"status": "validation_failed"}],
            context="diff --git a/a.py b/a.py\n", output_dir=str(out),
            plan_file=str(ranged.plan), context_kind="diff",
        )
        prompt = tasks["batches"][0]["prompt"]
        assert pinned_head_read_rule(ranged.base, ranged.head) in prompt

    def test_the_recorded_argv_carries_full_shas_in_every_spelling(self):
        pinned = cro.PinnedRange("a" * 40, "b" * 40, "a" * 40, cro.PIN_TYPED)
        cli_common.record_invocation(
            "review-code", ["--plan-file", "p.md", "--base=main", "--head", "feature", "-q"],
        )
        assert cro.pin_recorded_refs(pinned) is True
        assert cli_common.invocation()["argv"] == [
            "--plan-file", "p.md", "--base-ref", "a" * 40, "--head-ref", "b" * 40, "-q",
        ]
        assert cro.pin_recorded_refs(pinned) is False
        cli_common.record_invocation("review-code", ["--plan-file", "p.md"])
        cro.pin_recorded_refs(pinned)
        assert cli_common.invocation()["argv"][-4:] == [
            "--base-ref", "a" * 40, "--head-ref", "b" * 40,
        ]


# ---------------------------------------------------------------------------
# Attributes at head, and a detached worker's landed stage
# ---------------------------------------------------------------------------


class TestReadsAtHead:
    def test_the_prompt_order_reads_gitattributes_at_head(self, tmp_path, monkeypatch):
        repo = new_repo(tmp_path)
        repo.write("gen.py", "g = 1\n")
        repo.write("z.py", "z = 1\n")
        base = repo.commit("base")
        repo.write(".gitattributes", "gen.py linguist-generated\n")
        repo.write("gen.py", "g = 2\n")
        repo.write("z.py", "z = 2\n")
        head = repo.commit("generated at head")
        monkeypatch.chdir(repo.root)
        # The disk says otherwise (an unstaged edit: check-attr would read it).
        repo.write(".gitattributes", "# nothing is generated here\n")

        reviewer = Reviewer()
        assert run_review(review_argv(
            repo, "--base-ref", base, "--head-ref", head, "--skip-validation",
        ), reviewer) == 0
        listed = [
            line[3:-1] for line in reviewer.prompts[0].splitlines()
            if line.startswith("- `") and line.endswith("`")
        ]
        # Source first, then generated: gen.py sorts last only if its
        # attribute was read as head has it.
        assert listed.index("z.py") < listed.index("gen.py")
        manifest = (repo.phase_dir() / "review-scope.md").read_text(encoding="utf-8")
        assert "## Generated (1 file, collapsed)" in manifest


class TestALandedStage:
    """A detached worker of a commit front door: its stage (new header,
    ``commit_range``) lands inside ``maybe_detach``, after review-code's
    pre-flight loaded ``state`` and read the brief (plans/review-commits.md
    §2.1). The run must use what the stage brought and save it intact."""

    def test_the_worker_uses_and_keeps_what_the_stage_brought(self, ranged, monkeypatch):
        # A new brief: the stage carries its commit_range, so the state the
        # worker loads before its boundary has none, and the brief holds the
        # old header.
        brief = seed_commit_brief(ranged)
        sm = StateManager(brief)
        landed = dict(sm.state.pop("commit_range"))
        sm.save()
        brief.write_text("# OLD HEADER\n", encoding="utf-8")
        landed.update(header_sha256="header-NEW", drifted=["b.py"])
        stage = brief.parent / "stage.json"

        real_parse = cro.parse_args

        def parse_with_stage(argv=None):
            args = real_parse(argv)
            args.commit_stage = str(stage)  # the front door's hidden option
            return args

        def boundary_applies_the_stage(*_args, **kwargs):
            if kwargs.get("mode") == "review-code":
                writer = StateManager(brief)
                writer.state["commit_range"] = landed
                writer.save()
                brief.write_text(
                    "<!-- multi-llm:commit-brief v1 -->\n# NEW HEADER\n"
                    "<!-- /multi-llm:commit-brief -->\n", encoding="utf-8",
                )
            return None

        monkeypatch.setattr(cro, "parse_args", parse_with_stage)
        monkeypatch.setattr(cro, "maybe_detach", boundary_applies_the_stage)
        reviewer = Reviewer([issue("a.py", 5, 5, "A")])
        assert run_review(review_argv(
            ranged, "--base-ref", ranged.base, "--head-ref", ranged.head,
            "--files", "a.py", plan=brief,
        ), reviewer) == 0

        state = ranged.state(brief)
        record = state["commit_range"]
        assert record["header_sha256"] == "header-NEW"
        assert record["reviewed_header_sha256"] == "header-NEW"
        assert record["drifted"] == ["b.py"]
        assert "head_ref" not in state  # fronted, once the stage is seen
        assert state["review_scope_files"] == ["a.py"]  # the staged record, redone
        snapshot = (ranged.phase_dir(brief) / "validation-snapshot.md").read_text(
            encoding="utf-8"
        )
        assert "NEW HEADER" in snapshot and "OLD HEADER" not in snapshot
        access = diff_access_of(reviewer.prompts[0])
        assert "`b.py`" in access.split("have changed since", 1)[1].split("\n", 1)[0]

    def test_a_new_briefs_launch_calls_it_the_commit_brief(self, ranged, monkeypatch, capsys):
        # Live run 2026-10-26 (codex, opencode): a detached first launch's
        # log read "Plan file: <brief>". A launcher writes no state, so the
        # brief's `commit_range` is only in the stage while its pre-flight
        # prints, in the launcher and again in the worker.
        brief = seed_commit_brief(ranged)
        state_file = Path(StateManager(brief).state_file)
        state_file.unlink()
        stage = brief.parent / "stage.json"
        real_parse = cro.parse_args

        def parse_with_stage(argv=None):
            args = real_parse(argv)
            args.commit_stage = str(stage)  # the front door's hidden option
            return args

        monkeypatch.setattr(cro, "parse_args", parse_with_stage)
        monkeypatch.setattr(cro, "maybe_detach", lambda *_args, **_kwargs: 0)
        assert run_review(review_argv(
            ranged, "--base-ref", ranged.base, "--head-ref", ranged.head, "--detach",
            plan=brief,
        )) == 0
        out = capsys.readouterr().out
        assert f"Commit brief: {brief}" in out
        assert "Plan file" not in out
        assert not state_file.exists()  # a launcher writes no state
