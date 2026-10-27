#!/usr/bin/env python3
"""``review-commits``, ``apply-commit-fixes`` and ``improve-commits`` end to end.

**What this proves, and how** (plans/review-commits.md §4 T11, §5). Every test
here is a ``multi_llm.py`` **subprocess** in a throwaway git repository: the real
dispatcher, the real front door (``utils/commit_review.py``), the real
review-code and apply-code-fixes phases it fronts and the real write executor
(``--exec-subagents``), with the suite's mock provider binaries on ``PATH`` and
nothing stubbed inside the skill. No model is called and nothing is spent.

**One world, borrowed rather than rebuilt.** ``tests/test_e2e_improve_modes.py``
already drives ``improve-commits`` through the same harness, and its
``CommitWorld`` (a repository of three commits: ``init``, then
``security: rate limiter`` and ``auth: login records attempts``, or the root
commit alone), its mock plugin (write batches append one marker line per batch
to the file they name; ``IMPROVE_E2E_FAIL_CALL=<n>`` makes the n-th write call a
clean failure) and its scenario (the reviewer answers the commit review prompt,
``prompts/commit_review.txt``, from ``code_review/container_issues.json``;
validation subagents echo every group valid) are imported, not copied. The
cases that module already covers fully are referenced, not repeated: the two
steps of ``improve-commits`` on the pinned refs
(``TestImproveCommits.test_review_commits_then_apply_commit_fixes_on_the_pinned_refs``),
a stop in its apply and the continuation, ``--detach`` from a phrase, and a
root range's detached child.

**What is covered here**, each through the real phases:

* ``--review-commits`` with no range (``HEAD^..HEAD``, one commit), ``last 2
  commits`` then ``-2`` (the same brief, no restart, nothing re-run), the
  shims' quoted slot (``"last 2 commits"``, one argv word: the same range as
  the bare words), and ``A..B``: the block before any model call,
  ``[COMMIT_BRIEF]`` naming a brief on disk, the result line's
  ``commit_range``, ``report.md``/``report.html`` opening with the commit
  block and carrying "Introduced by", and the next step
  ``--apply-commit-fixes <pinned>``;
* a dirty working tree: the review's diff files are byte-identical to a clean
  run's, and ``[SCOPE_DRIFT]`` names every drifted file in scope;
* ``--apply-commit-fixes <pinned>`` and with no argument
  (``[COMMIT_REVIEW_SELECTED]``): ``[COMMIT_FIXES]`` once, before any batch, the
  fixes landed as uncommitted edits and nothing committed; a file edited after
  the review skipped and named; a ``report.md`` skip and a
  ``user_selections.json`` skip honoured;
* ``--improve-commits last 2 commits`` typed again after it finished: exit 2
  ``[ALREADY_COMPLETE]`` naming both commit modes, no provider call;
* the one-commit repository: review, apply, a stopped apply and its ``--resume``,
  each through the command the step before printed (the brief path is a root
  range's pinned spelling), no ``HEAD~1`` warning or ``Diff failed:`` anywhere,
  and the fix prompts carrying the full empty-tree id as their base;
* the refusals before any write: ``[RANGE_AFTER_FLAGS]``, ``[RANGE_DIVERGED]``,
  an empty window and an empty ``A..A`` (``[NOTHING_TO_REVIEW]``), a plan path
  (``[RANGE_UNRECOGNIZED]`` naming ``--context-plan``) and ``--plan-file``
  (``[COMMIT_FLAG_CONFLICT]``);
* the container lane's commit scenario
  (``fixtures/e2e/container_scenarios/container_commit_modes.yaml``, read by
  ``tests/test_container_commit_modes.py``), checked against the prompts these
  runs really send, so the container case cannot fail on a pattern.

Usage:
    uv run --project skills/multi-llm -- pytest tests/test_e2e_commit_modes.py -v
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_e2e_improve_modes import (  # noqa: E402
    COMMIT_REVIEW_PROMPT,
    COMMITS_FOLDER,
    MODEL,
    CommitWorld,
    Run,
    _commit_marked,
    _git,
    _poll_terminal,
    _stub_uv,
    _write_ids,
    classify,
)
from utils.commit_range import DATA_UNDER_REVIEW, pinned_head_read_rule  # noqa: E402
from utils.commit_review import BRIEF_MARKER  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
SCENARIOS = TESTS_DIR / "fixtures" / "e2e" / "scenarios"
CHAIN_SCENARIO = SCENARIOS / "container_full_chain.yaml"
COMMIT_SCENARIO = (TESTS_DIR / "fixtures" / "e2e" / "container_scenarios"
                   / "container_commit_modes.yaml")

#: The one reviewer, unattended, with the executor dispatching every subagent.
REVIEW_FLAGS = ("--models", MODEL, "--yes", "--exec-subagents")
#: What an unattended apply is typed with (``--yes`` implies the judge; the
#: executor runs the batches in this one call).
APPLY_FLAGS = ("--yes", "--no-confirm", "--exec-subagents")
#: The reviewer's fixture raises five findings: four in ``src/security.py``,
#: one in ``src/auth.py``.
FINDINGS = 5
AUTH_FINDING = "login does not record the lockout rejection"
#: What a command a commit mode prints never carries (§2.8).
NEVER_PRINTED = ("--plan-file", "--base-ref", "--head-ref", "--commit-stage", "--revs")
#: The commit block's commit lines: ``  <sha7>  <date>  <subject>``.
COMMIT_LINE = re.compile(r"^  ([0-9a-f]{7})  \d{4}-\d{2}-\d{2}  (.*)$")


# --------------------------------------------------------------------------- #
# Driving the dispatcher
# --------------------------------------------------------------------------- #

def dispatch(world: CommitWorld, mode: str, *tokens: str, env: Optional[dict] = None) -> Run:
    """``multi_llm.py <mode> <tokens...>`` in the world's repository.

    ``World.run`` puts a plan right after the mode; the commit modes take the
    range there (or nothing), so the first token simply takes that slot.
    """
    assert tokens, "every call here carries at least one token"
    return world.run(mode, *tokens[1:], plan=tokens[0], env=env)


def review(world: CommitWorld, *words: str, env: Optional[dict] = None) -> Run:
    """``review-commits <range words> --models <one reviewer> --yes --exec-subagents``."""
    return dispatch(world, "review-commits", *words, *REVIEW_FLAGS, env=env)


def apply_fixes(world: CommitWorld, *words: str, env: Optional[dict] = None) -> Run:
    """``apply-commit-fixes <range words> --yes --no-confirm --exec-subagents``."""
    return dispatch(world, "apply-commit-fixes", *words, *APPLY_FLAGS, env=env)


def printed(run: Run) -> List[str]:
    """The stdout lines a user reads: everything but the result line, which also
    carries each block's lines (``commit_range.lines``) for the tool routes."""
    return [line for line in run.stdout.splitlines()
            if not line.startswith("MULTI_LLM_RESULT")]


def block(run: Run, start: str = "[COMMIT_RANGE] ",
          end: str = "[COMMIT_RANGE_END]") -> List[str]:
    """The first printed block from *start* through *end*, both included."""
    lines = printed(run)
    first = next((i for i, line in enumerate(lines) if line.startswith(start)), None)
    assert first is not None, f"no {start.strip()} block\n{run.explain()}"
    last = next(i for i in range(first, len(lines)) if lines[i].startswith(end))
    return lines[first:last + 1]


def commit_lines(lines: Sequence[str]) -> List[Tuple[str, str]]:
    """``(sha7, subject)`` of each commit line of a commit block."""
    return [(m.group(1), m.group(2)) for m in map(COMMIT_LINE.match, lines) if m]


def marker_line(run: Run, tag: str) -> str:
    """The one printed line starting with *tag*."""
    found = [line for line in printed(run) if line.startswith(tag)]
    assert len(found) == 1, (tag, found, run.explain())
    return found[0]


def command_tokens(command: str) -> Tuple[str, List[str]]:
    """A printed command (``/multi-llm:multi-llm --apply-commit-fixes X --resume``,
    in whatever host spelling) as ``(mode, the tokens after the mode word)``."""
    tokens = shlex.split(command)
    at = next(i for i, token in enumerate(tokens)
              if token.startswith("--") and token[2:] in (
                  "review-commits", "apply-commit-fixes", "improve-commits"))
    return tokens[at][2:], tokens[at + 1:]


def git_out(world: CommitWorld, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=world.repo, capture_output=True,
                          text=True, check=True).stdout


def tree_snapshot(world: CommitWorld) -> Dict[str, bytes]:
    """Every file under the repository but ``.git``, by bytes."""
    return {
        path.relative_to(world.repo).as_posix(): path.read_bytes()
        for path in sorted(world.repo.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(world.repo).parts
    }


def display(base: Optional[str], head: str) -> str:
    return f"{base[:7] if base else 'root'}..{head[:7]}"


def brief_for(base: Optional[str], head: str) -> str:
    return f"{COMMITS_FOLDER}/{base[:12] if base else 'root'}-{head[:12]}.md"


def pinned_for(base: Optional[str], head: str) -> str:
    return f"{base[:12]}..{head[:12]}" if base else brief_for(base, head)


def assert_block_before_any_model_call(run: Run) -> None:
    """The commit block, then ``[COMMIT_BRIEF]``, then the fan-out (§2.1)."""
    lines = printed(run)
    opened = next(i for i, line in enumerate(lines) if line.startswith("[COMMIT_RANGE] "))
    brief = next(i for i, line in enumerate(lines) if line.startswith("[COMMIT_BRIEF] "))
    fan_out = next(i for i, line in enumerate(lines) if "fan-out over" in line)
    assert opened < brief < fan_out, run.explain()


def assert_next_step_is_the_apply(run: Run, pinned: str, out_rel: str) -> None:
    """After a commit review: ``--apply-commit-fixes <pinned>``, the reports named."""
    step = run.result["user_next_step"]
    assert step["phase"] == "apply-code-fixes" and step["complete"] is False, step
    assert step["command"].endswith(f"--apply-commit-fixes {pinned}"), step
    for token in NEVER_PRINTED:
        assert token not in step["command"], (token, step)
    for report in ("report.md", "report.html"):
        assert f"{out_rel}/code-review/{report}" in step["why"], step["why"]
    assert marker_line(run, "[NEXT_COMMAND] ").endswith(f"--apply-commit-fixes {pinned}")


def assert_commit_range_payload(run: Run, base: Optional[str], head: str,
                                commits: Sequence[str]) -> dict:
    payload = run.result["commit_range"]
    assert (payload["base"], payload["head"]) == (base, head), payload
    assert payload["brief"] == brief_for(base, head), payload
    assert [entry["sha"] for entry in payload["commits"]] == list(commits), payload
    assert payload["size"]["commits"] == len(commits), payload
    return payload


# --------------------------------------------------------------------------- #
# 1. --review-commits
# --------------------------------------------------------------------------- #

class TestReviewCommits:

    def test_no_range_reviews_the_last_commit_alone(self, tmp_path):
        world = CommitWorld(tmp_path)
        base, head = world.rev("HEAD~1"), world.head
        run = review(world)
        assert run.exit_code == 0, run.explain()

        # -- the block: HEAD^..HEAD, exactly one commit, then the brief ---------
        lines = block(run)
        assert lines[0].startswith(f"[COMMIT_RANGE] {display(base, head)} (1 commit, "), lines
        assert commit_lines(lines) == [(head[:7], "auth: login records attempts")], lines
        brief = brief_for(base, head)
        assert marker_line(run, "[COMMIT_BRIEF] ") == f"[COMMIT_BRIEF] {brief}"
        assert (world.repo / brief).read_text(encoding="utf-8").startswith(BRIEF_MARKER)
        assert_block_before_any_model_call(run)
        # one commit, so nothing calls it "two commits"
        assert marker_line(run, "Git head reference: ") == (
            f"Git head reference: {head} (reviewing {display(base, head)} as committed, "
            f"not the working tree)")

        # -- the result line and the record -------------------------------------
        assert (run.result["mode"], run.result["phase"], run.result["state"]) == (
            "review-commits", "code-review", "done")
        assert_commit_range_payload(run, base, head, [head])
        out = world.repo / brief[:-len(".md")]
        state = json.loads((out / "state.json").read_text(encoding="utf-8"))
        assert (state["commit_range"]["base"], state["commit_range"]["head"]) == (base, head)
        assert "code-review" in state["phases_completed"]

        # -- one commit review, on the commit prompt, with the pinned-head rule --
        reviews = world.calls_of("review")
        assert len(reviews) == 1 and reviews[0]["prompt"].startswith(COMMIT_REVIEW_PROMPT)
        assert pinned_head_read_rule(base, head) in reviews[0]["prompt"]
        manifest = (out / "code-review" / "review-scope.md").read_text(encoding="utf-8")
        assert "`src/auth.py`" in manifest and "`src/security.py`" not in manifest, manifest

        # -- both reports open with the commit block; findings name their commit -
        report_md = (out / "code-review" / "report.md").read_text(encoding="utf-8")
        assert "**Commit brief:**" in report_md and "**Plan:**" not in report_md
        for line in lines:
            assert line in report_md, (line, report_md[:1500])
        assert f"Introduced by `{head[:7]}` — auth: login records attempts" in report_md
        html = (out / "code-review" / "report.html").read_text(encoding="utf-8")
        assert "Commit brief:" in html
        assert f"[COMMIT_RANGE] {display(base, head)} (" in html
        assert "Introduced by" in html and head in html

        assert_next_step_is_the_apply(run, pinned_for(base, head), brief[:-len(".md")])

    def test_last_two_commits_then_minus_two_reuse_the_same_brief(self, tmp_path):
        world = CommitWorld(tmp_path)
        base, head = world.base, world.head
        first = review(world, "last", "2", "commits")
        assert first.exit_code == 0, first.explain()
        lines = block(first)
        assert commit_lines(lines) == [(head[:7], "auth: login records attempts"),
                                       (world.rev("HEAD~1")[:7], "security: rate limiter")]
        brief = world.repo / world.brief_rel
        assert marker_line(first, "[COMMIT_BRIEF] ") == f"[COMMIT_BRIEF] {world.brief_rel}"
        assert_commit_range_payload(first, base, head, [head, world.rev("HEAD~1")])
        assert_next_step_is_the_apply(first, world.pinned, world.brief_rel[:-len(".md")])
        record = world.state()["commit_range"]
        assert record["phrase"] == "last 2 commits"
        brief_bytes, calls = brief.read_bytes(), len(world.calls())

        again = review(world, "-2")

        # the same range, the same brief, the review on record: nothing re-runs
        assert again.exit_code == 2, again.explain()
        assert marker_line(again, "[COMMIT_BRIEF] ") == f"[COMMIT_BRIEF] {world.brief_rel}"
        assert block(again)[0] == lines[0]
        assert "[COMMIT_REVIEW_RESTARTED]" not in again.stdout
        assert len(world.calls()) == calls, "a model was asked again"
        assert brief.read_bytes() == brief_bytes
        after = world.state()["commit_range"]
        for key in ("base", "head", "header_sha256", "reviewed_header_sha256", "phrase"):
            assert after[key] == record[key], key
        assert again.result["state"] == "nothing-to-do"
        assert_next_step_is_the_apply(again, world.pinned, world.brief_rel[:-len(".md")])

    def test_the_shims_quoted_slot_is_the_same_range_as_bare_words(self, tmp_path):
        """The codex and opencode shims put the range in the dispatcher's quoted
        ``"<plan-path>"`` slot, so a model following them literally sends
        ``last 2 commits`` as ONE argv element, spaces inside. That word reaches
        ``_normalize_positionals`` and ``commit_range`` whole and resolves to
        the range three bare words do: the same brief, and the bare-word call
        after it is the review on record (nothing re-runs)."""
        world = CommitWorld(tmp_path)
        base, head = world.base, world.head
        first = review(world, "last 2 commits")
        assert first.exit_code == 0, first.explain()
        assert first.argv[3] == "last 2 commits", first.argv
        assert marker_line(first, "[COMMIT_BRIEF] ") == f"[COMMIT_BRIEF] {world.brief_rel}"
        assert_commit_range_payload(first, base, head, [head, world.rev("HEAD~1")])
        assert world.state()["commit_range"]["phrase"] == "last 2 commits"
        assert_next_step_is_the_apply(first, world.pinned, world.brief_rel[:-len(".md")])
        calls = len(world.calls())

        again = review(world, "last", "2", "commits")

        assert again.exit_code == 2, again.explain()
        assert marker_line(again, "[COMMIT_BRIEF] ") == f"[COMMIT_BRIEF] {world.brief_rel}"
        assert "[COMMIT_REVIEW_RESTARTED]" not in again.stdout
        assert len(world.calls()) == calls, "a model was asked again"

    def test_a_rerun_of_a_finished_review_names_the_commit_brief_and_its_mode(self, tmp_path):
        """§2.8: no line a user reads calls the brief a plan, and the mode word is
        the one the user typed (``review_orchestrator_base.already_completed_lines``)."""
        world = CommitWorld(tmp_path)
        assert review(world, "last", "2", "commits").exit_code == 0
        again = review(world, world.pinned)
        assert again.exit_code == 2, again.explain()
        notice = again.result["notice"]
        assert notice.startswith("review-commits has already been completed for this "
                                 "commit brief, so nothing ran this time"), notice
        assert "for this plan" not in again.stdout

    def test_the_reports_open_with_the_block_the_front_door_printed(self, tmp_path):
        """A file outside the range, edited on disk and cited by a finding, is
        badged as changed (the guard will skip it) but is not a reviewed file."""
        world = CommitWorld(tmp_path)
        (world.repo / "src/security.py").write_bytes(
            world.committed["src/security.py"] + b"# edited, outside the range\n")
        base, head = world.rev("HEAD~1"), world.head
        run = review(world)
        assert run.exit_code == 0, run.explain()
        lines = block(run)
        assert not any("[SCOPE_DRIFT]" in line for line in lines), "src/security.py is not in scope"
        assert run.result["commit_range"]["drifted"] == []

        review_dir = world.repo / brief_for(base, head)[:-len(".md")] / "code-review"
        report_md = (review_dir / "report.md").read_text(encoding="utf-8").splitlines()
        fence = report_md.index("```text")
        assert report_md[fence + 1:fence + 1 + len(lines)] == lines, report_md[:16]
        data = json.loads((review_dir / "report_data.json").read_text(encoding="utf-8"))
        assert data["commitRange"]["blockLines"] == lines
        # ...while the badges still name it: the apply's guard will skip it.
        assert data["commitRange"]["drifted"] == ["src/security.py"]

    def test_a_detached_launch_on_a_new_brief_prints_the_drift_line_once(self, tmp_path):
        """§2.5 "Drift": the front door prints ``[SCOPE_DRIFT]`` inside its
        block; the review-code it fronts neither recomputes nor prints it. A
        detached launch on a new brief hands review-code the pair as typed refs
        with ``commit_range`` still in the stage, and its pre-flight is fronted
        all the same: nothing more on the launcher's terminal, nothing in the
        worker's log."""
        world = CommitWorld(tmp_path)
        (world.repo / "src/auth.py").write_text("# edited by hand\n", encoding="utf-8")
        env = world.env(MULTI_LLM_UV=str(_stub_uv(world.tmp_path)))
        launch = dispatch(world, "review-commits", "last", "2", "--models", MODEL, "--yes",
                          "--detach", env=env)
        assert launch.exit_code == 0, launch.explain()
        line = (f"  [SCOPE_DRIFT] 1 of 2 reviewed files differ on disk from "
                f"{world.head[:7]}: src/auth.py")
        assert [l for l in printed(launch) if "[SCOPE_DRIFT]" in l] == [line], launch.explain()
        assert line in block(launch)
        _poll_terminal(world.out / "code-review")
        log = (world.out / "code-review" / "orchestrator-run.log").read_text(encoding="utf-8")
        assert "[SCOPE_DRIFT]" not in log, log[-2000:]
        assert world.state()["commit_range"]["drifted"] == ["src/auth.py"]
        # Both pre-flights call the brief the commit brief, never a plan file
        # (commit modes' live run, 2026-10-26: the worker's log said "Plan file").
        brief = str(world.out) + ".md"
        assert f"Commit brief: {brief}" in printed(launch), launch.explain()
        assert f"Commit brief: {brief}" in log.splitlines(), log[:2000]
        assert "Plan file" not in "\n".join(printed(launch)) + log

    def test_an_a_dot_dot_b_range(self, tmp_path):
        world = CommitWorld(tmp_path)
        base, head = world.rev("HEAD~2"), world.rev("HEAD~1")
        run = review(world, "HEAD~2..HEAD~1")
        assert run.exit_code == 0, run.explain()
        lines = block(run)
        assert lines[0].startswith(f"[COMMIT_RANGE] {display(base, head)} (1 commit, "), lines
        assert commit_lines(lines) == [(head[:7], "security: rate limiter")]
        brief = brief_for(base, head)
        assert marker_line(run, "[COMMIT_BRIEF] ") == f"[COMMIT_BRIEF] {brief}"
        assert (world.repo / brief).is_file()
        assert_commit_range_payload(run, base, head, [head])
        # the scope is that commit's file, read at that commit, not HEAD's
        out = world.repo / brief[:-len(".md")]
        manifest = (out / "code-review" / "review-scope.md").read_text(encoding="utf-8")
        assert "`src/security.py`" in manifest and "`src/auth.py`" not in manifest, manifest
        diff = (out / "code-review" / "diff" / "1-src-security.py.diff").read_text(encoding="utf-8")
        assert "+def record_attempt(email, success):" in diff
        assert_next_step_is_the_apply(run, pinned_for(base, head), brief[:-len(".md")])

    def test_a_dirty_tree_changes_nothing_the_reviewer_reads(self, tmp_path):
        """§5 "The diff": with a dirty tree the diff files are byte-identical to a
        clean run's, and ``[SCOPE_DRIFT]`` names every drifted file in scope."""
        (tmp_path / "clean").mkdir()
        (tmp_path / "dirty").mkdir()
        clean = CommitWorld(tmp_path / "clean")
        dirty = CommitWorld(tmp_path / "dirty")
        # unstaged, staged, and untracked: only the first two are in scope
        (dirty.repo / "src/auth.py").write_text("# edited by hand\n", encoding="utf-8")
        (dirty.repo / "src/security.py").write_text(
            dirty.committed["src/security.py"].decode("utf-8") + "# staged\n", encoding="utf-8")
        _git(dirty.repo, "add", "src/security.py")
        (dirty.repo / "src/extra.py").write_text("EXTRA = 1\n", encoding="utf-8")

        ran = {name: review(world, "last", "2", "commits")
               for name, world in (("clean", clean), ("dirty", dirty))}
        for name, run in ran.items():
            assert run.exit_code == 0, (name, run.explain())

        def diffs(world: CommitWorld) -> Dict[str, bytes]:
            folder = world.out / "code-review" / "diff"
            return {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}

        assert diffs(clean) and diffs(dirty) == diffs(clean), (
            "a dirty tree moved the bytes the reviewer reads")

        drift = [line.strip() for line in block(ran["dirty"]) if "[SCOPE_DRIFT]" in line]
        assert drift == [f"[SCOPE_DRIFT] 2 of 2 reviewed files differ on disk from "
                         f"{dirty.head[:7]}: src/auth.py, src/security.py"], drift
        assert ran["dirty"].result["commit_range"]["drifted"] == ["src/auth.py", "src/security.py"]
        assert not any("[SCOPE_DRIFT]" in line for line in block(ran["clean"]))
        assert ran["clean"].result["commit_range"]["drifted"] == []
        assert dirty.state()["commit_range"]["drifted"] == ["src/auth.py", "src/security.py"]
        # the review left the dirty tree as it found it
        assert (dirty.repo / "src/auth.py").read_text(encoding="utf-8") == "# edited by hand\n"
        # every reviewer and validator prompt carries the read rule, drift or none
        for world in (clean, dirty):
            rule = pinned_head_read_rule(world.base, world.head)
            prompts = [c["prompt"] for c in world.calls()
                       if classify(c) in ("review", "validate")]
            assert len(prompts) >= 2
            for prompt in prompts:
                assert rule in prompt and DATA_UNDER_REVIEW in prompt, prompt[:200]

        # §5 "Drift and the guard": the apply skips exactly the files the drift
        # line named (both run the one change test), and moves none of them
        on_disk = dirty.bytes_of()
        applied = apply_fixes(dirty, dirty.pinned)
        assert applied.exit_code == 0, applied.explain()
        fixes = applied.result["commit_fixes"]
        assert fixes["changed_files"] == ["src/auth.py", "src/security.py"], fixes
        assert (fixes["to_apply"], fixes["skipped_changed"]) == (0, FINDINGS), fixes
        assert not dirty.calls_of("write"), "a batch ran on a drifted file"
        assert dirty.bytes_of() == on_disk


# --------------------------------------------------------------------------- #
# 2. --apply-commit-fixes
# --------------------------------------------------------------------------- #

def _reviewed(tmp_path: Path) -> CommitWorld:
    world = CommitWorld(tmp_path)
    run = review(world, "last", "2", "commits")
    assert run.exit_code == 0, run.explain()
    return world


def _groups(world: CommitWorld) -> List[dict]:
    return json.loads((world.out / "code-review" / "grouped.json").read_text(
        encoding="utf-8"))["groups"]


class TestApplyCommitFixes:

    @pytest.mark.parametrize("typed", ["pinned", "no argument"])
    def test_fixes_land_as_uncommitted_edits_and_nothing_is_committed(self, tmp_path, typed):
        world = _reviewed(tmp_path)
        log_before = git_out(world, "log", "--format=%H")
        run = apply_fixes(world, *([world.pinned] if typed == "pinned" else []))
        assert run.exit_code == 0, run.explain()

        # -- which review: the pinned one, or the one the lookup selected -------
        lines = block(run)
        selected = [line.strip() for line in lines if "[COMMIT_REVIEW_SELECTED]" in line]
        if typed == "no argument":
            assert len(selected) == 1, lines
            assert selected[0].startswith(f"[COMMIT_REVIEW_SELECTED] {world.pinned} completed ")
            assert selected[0].endswith(", ends at HEAD"), selected
            assert run.result["commit_range"]["selected"]["behind"] == 0
        else:
            assert not selected, lines
        assert marker_line(run, "[COMMIT_BRIEF] ") == f"[COMMIT_BRIEF] {world.brief_rel}"

        # -- [COMMIT_FIXES]: once, before the first batch -----------------------
        fixes = block(run, "[COMMIT_FIXES] ", "[COMMIT_FIXES_END]")
        assert fixes[0] == f"[COMMIT_FIXES] {FINDINGS} fixes to apply", fixes
        assert sum(line.startswith("[COMMIT_FIXES] ") for line in printed(run)) == 1
        out = printed(run)
        first_batch = next(i for i, line in enumerate(out) if "[write-subagents] batch " in line)
        assert out.index(fixes[0]) < first_batch
        payload = run.result["commit_fixes"]
        assert (payload["to_apply"], payload["skipped_changed"], payload["changed_files"]) == (
            FINDINGS, 0, [])
        assert payload["lines"] == fixes

        # -- the bytes: every batch landed once, on the committed files ----------
        report = world.report("apply-fixes")
        assert report["exit"] == 0 and report["stop"] is None and report["tasks"]
        assert all(task["write_outcome"] == "applied" for task in report["tasks"])
        assert world.bytes_of() == _commit_marked(world, report["tasks"])
        assert len(world.calls_of("write")) == len(report["tasks"])

        # -- nothing committed: the history is as the review found it ------------
        assert git_out(world, "log", "--format=%H") == log_before
        status = git_out(world, "status", "--porcelain", "--untracked-files=no")
        assert {line[3:] for line in status.splitlines()} == {"src/auth.py", "src/security.py"}

        state = world.state()
        assert {"code-review", "apply-code-fixes"} <= set(state["phases_completed"])
        assert (run.result["mode"], run.result["phase"], run.result["state"]) == (
            "apply-commit-fixes", "apply-code-fixes", "done")
        assert run.result["user_next_step"]["complete"] is True

    def test_status_after_a_foreground_apply_prints_the_saved_block(self, tmp_path):
        # Review D, finding 1: a phase that ran in the foreground has no run
        # record (`no_run`), and one such view used to suppress the block for
        # the whole poll, so the hand-typed `status <pinned>` never showed it.
        world = _reviewed(tmp_path)
        applied = apply_fixes(world, world.pinned)
        assert applied.exit_code == 0, applied.explain()
        fixes = block(applied, "[COMMIT_FIXES] ", "[COMMIT_FIXES_END]")
        status = world.run("status", plan=world.pinned)
        assert status.exit_code in (0, 2), status.explain()
        assert block(status, "[COMMIT_FIXES] ", "[COMMIT_FIXES_END]") == fixes
        assert status.result["commit_fixes"]["lines"] == fixes

    def test_a_file_edited_after_the_review_is_skipped_and_named(self, tmp_path):
        world = _reviewed(tmp_path)
        hand = world.committed["src/auth.py"] + b"# edited by hand\n"
        (world.repo / "src/auth.py").write_bytes(hand)
        run = apply_fixes(world, world.pinned)
        assert run.exit_code == 0, run.explain()

        fixes = block(run, "[COMMIT_FIXES] ", "[COMMIT_FIXES_END]")
        assert fixes[0] == f"[COMMIT_FIXES] {FINDINGS - 1} fixes to apply", fixes
        assert (f"  skipped  1 (file changed since {world.head[:7]}: src/auth.py)"
                in fixes), fixes
        payload = run.result["commit_fixes"]
        assert payload["changed_files"] == ["src/auth.py"] and payload["skipped_changed"] == 1

        # the hand edit is untouched; the other file got every batch on it
        assert (world.repo / "src/auth.py").read_bytes() == hand
        report = world.report("apply-fixes")
        assert all(task["files_touched"] == ["src/security.py"] for task in report["tasks"])
        assert world.bytes_of()["src/security.py"] == _commit_marked(
            world, report["tasks"])["src/security.py"]
        assert not any(AUTH_FINDING in call["prompt"] for call in world.calls_of("write"))
        summary = (world.out / "apply-fixes" / "summary.md").read_text(encoding="utf-8")
        assert f"### {AUTH_FINDING} [commit_guard]" in summary
        assert "**Files**: `src/auth.py`" in summary

    def test_the_summary_names_the_commit_brief(self, tmp_path):
        """§2.8: ``summary.md`` is a file the user reads, and the apply's core
        names it in the report; it never calls the brief a plan."""
        world = _reviewed(tmp_path)
        assert apply_fixes(world, world.pinned).exit_code == 0
        summary = (world.out / "apply-fixes" / "summary.md").read_text(encoding="utf-8")
        assert f"**Commit brief**: {world.repo / world.brief_rel}" in summary, summary[:400]
        assert "**Plan**:" not in summary

    @pytest.mark.parametrize("where", ["report.md", "user_selections.json"])
    def test_a_finding_skipped_in_the_reports_is_not_applied(self, tmp_path, where):
        """§5 "Reports": a finding skipped in either report is not applied."""
        world = _reviewed(tmp_path)
        review_dir = world.out / "code-review"
        if where == "report.md":
            skipped = "Token comparison is not constant time"
            text = (review_dir / "report.md").read_text(encoding="utf-8")
            at = text.index("- [ ] Skip", text.index(f". {skipped}\n"))
            (review_dir / "report.md").write_text(
                text[:at] + "- [x] Skip" + text[at + len("- [ ] Skip"):], encoding="utf-8")
        else:
            skipped = "Attempt store grows without bound"
            group = next(g for g in _groups(world) if g["theme"] == skipped)
            (review_dir / "user_selections.json").write_text(json.dumps({
                "format_version": 2,
                "plan_path": str(world.repo / world.brief_rel),
                "phase": "code-review",
                "exported_at": "2026-10-26T00:00:00Z",
                "skipped_groups": [group["group_hash"]],
                "skipped_suggestions": [],
                "edited_descriptions": {},
            }), encoding="utf-8")

        run = apply_fixes(world, world.pinned)
        assert run.exit_code == 0, run.explain()

        assert block(run, "[COMMIT_FIXES] ", "[COMMIT_FIXES_END]")[0] == (
            f"[COMMIT_FIXES] {FINDINGS - 1} fixes to apply")
        output = json.loads((world.out / "apply-fixes" / "orchestrator_output.json").read_text(
            encoding="utf-8"))
        assert skipped not in [fix["title"] for fix in output["to_apply"]]
        assert [item["title"] for item in output["user_skipped_items"]] == [skipped]
        writes = world.calls_of("write")
        assert writes and not any(skipped in call["prompt"] for call in writes)
        report = world.report("apply-fixes")
        assert all(task["write_outcome"] == "applied" for task in report["tasks"])
        assert world.bytes_of() == _commit_marked(world, report["tasks"])


# --------------------------------------------------------------------------- #
# 3. --improve-commits typed again after it finished
# --------------------------------------------------------------------------- #

class TestARefusedDetachedRestart:
    """Review fix 3, A1: ``--review-commits <pinned> --focus X --detach`` is a
    restart whose launch ``detach.launch`` refuses here (the detached-run
    ceiling, made 1 with one other live run registered). review-code's own
    pre-flight in the launcher used to pop ``phases_completed["code-review"]``
    and save before the refusal, so the completed review read as incomplete
    and the apply refused it. A refused launch now leaves state.json as it
    was, and the review it kept still applies."""

    def test_the_completed_review_survives_and_still_applies(self, tmp_path):
        world = _reviewed(tmp_path)
        state_path = world.out / "state.json"
        before = state_path.read_bytes()
        brief_before = (world.repo / world.brief_rel).read_bytes()
        # Another live detached run in this project (a process that is alive,
        # registered elsewhere): with a ceiling of 1 the launch is refused.
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        try:
            registry = world.repo / ".multi-llm" / "runs"
            registry.mkdir(parents=True, exist_ok=True)
            (registry / "other-run.json").write_text(json.dumps({
                "run_id": "other-run", "pid": other.pid, "mode": "review-code",
                "phase_dir": str(tmp_path / "elsewhere" / "code-review"),
            }), encoding="utf-8")
            run = dispatch(world, "review-commits", world.pinned, "--focus", "X", "--detach",
                           *REVIEW_FLAGS, env=world.env(MULTI_LLM_MAX_DETACHED="1"))
        finally:
            other.kill()
            other.wait()
            (world.repo / ".multi-llm" / "runs" / "other-run.json").unlink()
        assert run.exit_code == 3, run.explain()
        assert "detached runs are already in flight" in run.stdout + run.stderr
        assert state_path.read_bytes() == before
        assert (world.repo / world.brief_rel).read_bytes() == brief_before
        assert not list((world.out / "code-review").glob(".commit-stage-*"))

        applied = apply_fixes(world, world.pinned)
        assert applied.exit_code == 0, applied.explain()
        assert "[COMMIT_REVIEW_NOT_FOUND]" not in applied.stdout


class TestImproveCommitsTypedAgain:
    """The two steps themselves are graded by
    ``test_e2e_improve_modes.TestImproveCommits.test_review_commits_then_apply_commit_fixes_on_the_pinned_refs``;
    its ``[ALREADY_COMPLETE]`` follows a ``--resume`` of the pinned range. This is
    the plain case: the same phrase typed again, no flag."""

    def test_the_rerun_is_already_complete_and_names_the_commit_modes(self, tmp_path):
        world = CommitWorld(tmp_path)
        first = world.improve()
        assert first.exit_code == 0, first.explain()
        assert list(first.step_flags()) == ["review-code", "apply-code-fixes"]
        assert set(world.state()["phases_completed"]) == {"code-review", "apply-code-fixes"}
        calls, edited = len(world.calls()), world.bytes_of()

        again = world.improve()

        assert again.exit_code == 2, again.explain()
        line = marker_line(again, "[ALREADY_COMPLETE] ")
        assert line.startswith("[ALREADY_COMPLETE] improve-commits: review-commits completed "), line
        assert ", apply-commit-fixes completed " in line, line
        assert "review-code" not in line and "apply-code-fixes" not in line, line
        assert marker_line(again, "[COMMIT_BRIEF] ") == f"[COMMIT_BRIEF] {world.brief_rel}"
        assert "[COMMIT_REVIEW_RESTARTED]" not in again.stdout
        assert "$ uv run" not in again.stdout, "a phase was started"
        assert len(world.calls()) == calls
        assert world.bytes_of() == edited
        assert (again.result["mode"], again.result["exit"], again.result["state"]) == (
            "improve-commits", 2, "nothing-to-do")


# --------------------------------------------------------------------------- #
# 4. The one-commit repository
# --------------------------------------------------------------------------- #

def _no_fallback_or_diff_failure(run: Run) -> None:
    """§5 "Root ranges": no step prints a ``HEAD~1`` warning or ``Diff failed:``."""
    for stream in (run.stdout, run.stderr):
        assert "HEAD~1" not in stream, run.explain()
        assert "Diff failed:" not in stream, run.explain()


class TestTheOneCommitRepository:

    def test_review_apply_stop_and_resume_through_the_printed_commands(self, tmp_path):
        world = CommitWorld(tmp_path, root_only=True)
        head, empty_tree = world.head, world.empty_tree
        brief = f"{COMMITS_FOLDER}/root-{head[:12]}.md"
        assert world.brief_rel == brief

        # -- the review: every file of the first commit, added, against the
        #    computed empty tree --------------------------------------------------
        reviewed = review(world)
        assert reviewed.exit_code == 0, reviewed.explain()
        _no_fallback_or_diff_failure(reviewed)
        lines = block(reviewed)
        assert lines[0].startswith(f"[COMMIT_RANGE] root..{head[:7]} (1 commit, 4 files, "), lines
        assert commit_lines(lines) == [(head[:7], "init")]
        assert marker_line(reviewed, "[COMMIT_BRIEF] ") == f"[COMMIT_BRIEF] {brief}"
        assert (world.repo / brief).is_file()
        assert_commit_range_payload(reviewed, None, head, [head])
        assert world.state()["commit_range"]["base"] is None
        assert f"Git base reference: {empty_tree}" in reviewed.stdout
        diffs = sorted((world.out / "code-review" / "diff").glob("*.diff"))
        assert [p.name.split("-", 1)[1] for p in diffs] == [
            "plans-auth-feature.md.diff", "src-__init__.py.diff", "src-auth.py.diff",
            "src-security.py.diff"]
        for path in diffs:
            text = path.read_text(encoding="utf-8")
            assert "new file mode" in text and "--- /dev/null" in text, path.name
        assert empty_tree in world.calls_of("review")[0]["prompt"]

        # -- the apply, typed as the review printed it, stopped at batch 2 -------
        mode, words = command_tokens(reviewed.result["user_next_step"]["command"])
        assert (mode, words) == ("apply-commit-fixes", [brief])
        stopped = apply_fixes(world, *words, env=world.env(IMPROVE_E2E_FAIL_CALL="2"))
        assert stopped.exit_code == 3, stopped.explain()
        _no_fallback_or_diff_failure(stopped)
        assert stopped.result["state"] == "needs-human"
        report = world.report("apply-fixes")
        assert [task["status"] for task in report["tasks"]] == ["ok", "batch_failed"]
        landed, failed = report["tasks"]
        assert "apply-code-fixes" not in world.state()["phases_completed"]

        # -- the way back, as the dispatcher prints it after an exit 3 ------------
        lookup = world.run("util", brief, "--blocked", "apply-commit-fixes", plan="next-command")
        assert lookup.exit_code == 0, lookup.explain()
        resume_line = marker_line(lookup, "[RESUME_COMMAND] ")
        assert resume_line.endswith(f"--apply-commit-fixes {brief} --resume"), resume_line
        assert lookup.result["user_resume"]["mode"] == "apply-commit-fixes"
        mode, words = command_tokens(resume_line[len("[RESUME_COMMAND] "):])
        assert (mode, words) == ("apply-commit-fixes", [brief, "--resume"])
        writes = len(world.calls_of("write"))

        resumed = apply_fixes(world, *words)
        assert resumed.exit_code == 0, resumed.explain()
        _no_fallback_or_diff_failure(resumed)

        # only the stopped batch and the one never reached were dispatched
        redispatched = [_write_ids(call) for call in world.calls_of("write")[writes:]]
        assert failed["expected_ids"] in redispatched, redispatched
        assert all(not set(ids) & set(landed["expected_ids"]) for ids in redispatched)
        final = world.report("apply-fixes")
        assert [task["write_outcome"] for task in final["tasks"]] == ["applied"] * 3
        assert world.bytes_of() == _commit_marked(world, final["tasks"])
        assert "apply-code-fixes" in world.state()["phases_completed"]
        assert git_out(world, "rev-list", "--count", "HEAD").strip() == "1", "a commit was made"

        # -- every fix prompt diffs against the full, computed empty-tree id -------
        fix_prompts = [call["prompt"] for call in world.calls_of("write")]
        assert len(fix_prompts) == 4
        for prompt in fix_prompts:
            assert f"git diff {empty_tree} -- " in prompt, prompt[:400]
            assert "HEAD~1" not in prompt
        assert f"Git base reference: {empty_tree}" in resumed.stderr


# --------------------------------------------------------------------------- #
# 5. Refusals, before any write
# --------------------------------------------------------------------------- #

REFUSALS = [
    # (id, tokens, exit, the stdout marker line's start)
    ("range after a flag", ("--focus", "x", "last", "2", "commits"), 3,
     "[RANGE_AFTER_FLAGS] last 2 commits"),
    ("reversed range", ("HEAD..HEAD~2", *REVIEW_FLAGS), 3,
     "[RANGE_DIVERGED] HEAD is not an ancestor of HEAD~2"),
    ("empty window", ("yesterday", *REVIEW_FLAGS), 2, "[NOTHING_TO_REVIEW] "),
    ("empty A..A", ("HEAD..HEAD", *REVIEW_FLAGS), 2, "[NOTHING_TO_REVIEW] "),
    ("a plan path", ("plans/auth-feature.md", *REVIEW_FLAGS), 3,
     "[RANGE_UNRECOGNIZED] plans/auth-feature.md: a file, but not a commit brief"),
    ("--plan-file", ("x", "--plan-file", "plans/auth-feature.md", *REVIEW_FLAGS), 3,
     "[COMMIT_FLAG_CONFLICT] --plan-file"),
]


class TestRefusedBeforeAnyWrite:

    @pytest.mark.parametrize("tokens, code, marker",
                             [case[1:] for case in REFUSALS],
                             ids=[case[0] for case in REFUSALS])
    def test_the_refusal_writes_nothing(self, tmp_path, tokens, code, marker):
        world = CommitWorld(tmp_path)
        before = tree_snapshot(world)
        status_before = git_out(world, "status", "--porcelain", "--ignored")

        run = dispatch(world, "review-commits", *tokens)

        if tokens[0] == "yesterday" and run.exit_code != 2:
            today = datetime.now().astimezone().date().isoformat()
            if git_out(world, "log", "-1", "--format=%cd", "--date=short").strip() != today:
                pytest.skip("the commits were made before midnight and the run after it")
        assert run.exit_code == code, run.explain()
        found = [line for line in printed(run) if line.startswith(marker)]
        assert len(found) == 1, run.explain()
        assert (run.result["exit"], run.result["state"]) == (
            code, "nothing-to-do" if code == 2 else "needs-human")
        # nothing ran and nothing was written: no block, no brief, no state.json
        for never in ("[COMMIT_RANGE] ", "[COMMIT_BRIEF]", "$ uv run"):
            assert never not in run.stdout, (never, run.explain())
        assert not world.calls(), "a provider was called"
        assert not (world.repo / COMMITS_FOLDER).exists()
        assert not list(world.repo.rglob("state.json"))
        assert tree_snapshot(world) == before
        assert git_out(world, "status", "--porcelain", "--ignored") == status_before
        if marker.startswith("[RANGE_UNRECOGNIZED]"):
            assert ("To review commits against that plan: --review-commits <range> "
                    "--context-plan plans/auth-feature.md") in run.stdout
        if marker.startswith("[RANGE_AFTER_FLAGS]"):
            assert "Run: --review-commits last 2 commits --focus x" in run.stderr


# --------------------------------------------------------------------------- #
# 6. The container lane's scenario, checked against the prompts really sent
# --------------------------------------------------------------------------- #

def _load(path: Path):
    import scenario_matcher

    return scenario_matcher.load_scenario(path)


class TestTheContainerCommitScenario:
    """``tests/test_container_commit_modes.py`` only runs with ``-m container``;
    this keeps its scenario honest in every plain run, at no cost."""

    def test_it_is_the_chain_scenario_plus_one_commit_review_rule(self):
        commit = yaml.safe_load(COMMIT_SCENARIO.read_text(encoding="utf-8"))
        chain = yaml.safe_load(CHAIN_SCENARIO.read_text(encoding="utf-8"))
        assert commit["prompts"][:-1] == chain["prompts"], (
            "copy container_full_chain.yaml's `prompts:` list across verbatim")
        assert commit["on_unmatched"] == chain["on_unmatched"] == "fail"
        last = commit["prompts"][-1]
        assert last["fixture"] == "../responses/code_review/container_issues.json"
        assert "You are reviewing the code changes made by a range of git commits" in (
            last["pattern"])
        scenario = _load(COMMIT_SCENARIO)
        for rule in scenario.rules:
            path = rule.response.path(scenario.dir)
            assert path is not None and path.is_file(), rule.response.fixture

    def test_each_prompt_a_commit_review_sends_has_exactly_one_owner(self, tmp_path):
        world = CommitWorld(tmp_path)
        assert review(world, "last", "2", "commits").exit_code == 0
        (world.repo / "src/auth.py").write_bytes(world.committed["src/auth.py"] + b"# x\n")
        assert apply_fixes(world, world.pinned).exit_code == 0
        owner = {
            "review": "../responses/code_review/container_issues.json",
            "validate": "../responses/validation/code_review_valid.json",
            "write": "../responses/write_batches/apply_code_fixes.json",
        }
        scenario, chain = _load(COMMIT_SCENARIO), _load(CHAIN_SCENARIO)
        seen = set()
        for call in world.calls():
            kind = classify(call)
            matches = scenario.matching_rules(call["prompt"])
            assert [m.response.fixture for m in matches] == [owner[kind]], (
                kind, [m.response.fixture for m in matches], call["prompt"][:200])
            seen.add(kind)
            if kind == "review":
                # the chain's own review-code rule must not claim it
                assert chain.matching_rules(call["prompt"]) == [], call["prompt"][:200]
        assert seen == set(owner), seen

        # The verdict sheet the validation rule serves joins BY GROUP HASH; a
        # commit review's groups must be among its rows, or every group of the
        # container row lands on validation_failed and its guard grades nothing.
        sheet = json.loads((COMMIT_SCENARIO.parent / owner["validate"]).read_text(
            encoding="utf-8"))
        hashes = {str(row.get("group_hash") or row.get("group_id"))
                  for row in sheet["groups"]}
        grouped = {group["group_hash"] for group in _groups(world)}
        assert len(grouped) == FINDINGS and grouped <= hashes, sorted(grouped - hashes)
