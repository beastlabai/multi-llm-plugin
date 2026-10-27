"""Provider lane: `--review-commits` and `--apply-commit-fixes` against real CLIs.

The commit modes (plans/review-commits.md) are front doors over review-code and
apply-code-fixes: a front door resolves a commit range, prints the commit
block, writes a commit brief and runs the fronted phase on it, pinned to the
range's two commits. `test_e2e_commit_modes.py` drives them end to end through
the real dispatcher and phases with the suite's fake provider binaries; what it
cannot show is a REAL CLI answering the commit review prompt, a real runner
dispatching its validation, and the guard deciding over a workspace a real
seed built. These rows close that gap at zero cost, on the provider lane's
terms (the loopback stub, `--network none`, no credential), exactly as
`test_container_improve_modes.py` does for the two other composites. Each row
runs, in one container:

    review-commits --quick --yes --exec-subagents
    apply-commit-fixes --yes --no-confirm --exec-subagents

**The workspace makes this the one-commit case.** `seed_workspace.py` commits
the sample project ONCE ("Initial auth service") and then writes an
uncommitted working diff: `src/auth.py` modified, `src/security.py` new and
untracked. So a bare `review-commits` reviews a ROOT range: every committed
file as added, against the computed empty-tree id, with the brief at
`plans/commit-reviews/root-<head12>.md` (§2.2 "Root ranges and the empty
tree"). The working diff is drift (§2.5): `src/auth.py` is in scope and differs
on disk from the reviewed commit, so the block names it in `[SCOPE_DRIFT]`.
The bare `apply-commit-fixes` then takes that review (`[COMMIT_REVIEW_SELECTED]`,
§2.7), and the guard compares every fix file with the reviewed commit: the
reviewer's fixture (`code_review/container_issues.json`, the same one the
review-code rows serve) cites only `src/auth.py` and `src/security.py`, both
of which differ from `head` (a file absent at `head` and present on disk does
not match), so EVERY fix is skipped and named in `[COMMIT_FIXES]` and no write
batch is scheduled. That is the claim graded here: the guard holds on a real
CLI's findings, the run still completes, and no byte of the sample project
moves. The write path itself is the review-code/apply-code-fixes one, graded
per CLI by the chain rows and by `test_container_improve_modes.py`.

**The scenario** is `fixtures/e2e/container_scenarios/container_commit_modes.yaml`:
the chain scenario's rules verbatim plus one rule for the commit review's
fan-out prompt. Its header says why it is a separate file;
`test_e2e_commit_modes.py::TestTheContainerCommitScenario` pins it to the chain
scenario and to the prompts a commit review really sends, in every plain run.

**The evidence** is read from what the row left, never from exit codes alone:
the dispatcher's receipts (each result line carries `commit_range`, and the
apply's `commit_fixes`), the stub log (which turns were served, and that no
write batch ever was), the reports the receipts named (exported under
`artifacts/commit-reviews/…`), and the workspace census (the brief and its
folder appear; nothing under `src/` or `tests/` moves). `state.json` is not in
the bundle: the entrypoint exports the plan's output dir and the files the
receipts name, and a commit review writes under `plans/commit-reviews/`.

Why these two CLIs: the same reasoning as `test_container_improve_modes.py`
(the orchestration under test does not change per reviewer CLI, and each CLI's
parse is graded per CLI by the chain rows); the two rows reuse its entries.
"""

from __future__ import annotations

import re

import pytest

from .container.docker_probe import requires_docker
from .container import harness
from .container import seed_workspace
from . import test_container_improve_modes as improve_lane

pytestmark = [pytest.mark.container, requires_docker()]

#: The commit scenario, as the image holds it (`tests/fixtures/e2e` ships whole).
SCENARIO = ("/opt/mllm/skill/tests/fixtures/e2e/container_scenarios/"
            "container_commit_modes.yaml")

COMMANDS = [
    "review-commits --quick --yes --exec-subagents",
    "apply-commit-fixes --yes --no-confirm --exec-subagents",
]

#: The one commit the seed makes, and the files it holds: the source tree plus
#: the copied plan. `.multi-llm/providers.yaml` is written after the commit.
SEED_SUBJECT = "Initial auth service"
COMMITTED_FILES = len(seed_workspace.COMMITTED) + 1
#: In scope and different on disk; `src/security.py` is not in the commit.
DRIFTED = ["src/auth.py"]
#: The files the reviewer's fixture cites, both different from `head`.
CHANGED_FIX_FILES = ["src/auth.py", "src/security.py"]
#: `container_issues.json` raises five findings: five groups on a one-CLI panel.
FINDINGS = 5

BRIEF = re.compile(r"^plans/commit-reviews/root-([0-9a-f]{12})\.md$")

SERVED = {
    "../responses/code_review/container_issues.json": "review-commits fan-out",
    "../responses/validation/code_review_valid.json": "commit review validation",
}
#: Never answered: no phase the commit modes do not front, and no write batch
#: (every fix was skipped by the guard before a batch existed).
NEVER_SERVED = {
    "../responses/review_plan/raw_suggestions.json": "review-plan fan-out",
    "../responses/generate_tasks/tasks_v1.json": "generate-tasks",
    "../responses/review_tasks/coverage_suggestions.json": "review-tasks fan-out",
    "../responses/write_batches/apply_suggestions.json": "apply-suggestions batch",
    "../responses/write_batches/apply_task_suggestions.json": "apply-task-suggestions",
    "../responses/write_batches/implement.json": "implement",
    "../responses/write_batches/apply_code_fixes.json": "apply-code-fixes batch",
    "../responses/ask/plan_question.md": "ask",
}

COMMIT_CLIS = improve_lane.IMPROVE_CLIS
ENTRIES = improve_lane.ENTRIES


def _row(entry: dict) -> harness.Row:
    cli = entry["cli"]
    spec = f"{cli}:{entry.get('model_arg', 'stub-model')}"
    return harness.Row(
        row_id=f"commit-{cli}",
        lane="provider",
        cli=cli,
        spec=[spec],
        # The reviewer is its own runner, as on the chain and improve rows.
        runner=spec,
        helpers=[],
        command=COMMANDS[0],
        commands=COMMANDS[1:],
        scenario=SCENARIO,
        # One review and an apply with no batch: the improve rows' cap bounds it.
        row_timeout_s=int(entry.get(
            "chain_row_timeout_s",
            max(600, int(entry.get("row_timeout_s", 300)) * 2))),
    )


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def utc():
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


@pytest.fixture(scope="module", params=ENTRIES, ids=list(COMMIT_CLIS))
def commit(request, posture, utc):
    """Run one CLI's commit row once and share its evidence."""
    return harness.run_row(_row(request.param), posture=posture, utc=utc)


def _results(result: harness.RowResult) -> dict:
    """``{mode word: result line}`` of the two dispatcher invocations."""
    return {mode: payload for mode, payload in improve_lane._receipts_by_mode(result)}


def _review(result: harness.RowResult) -> dict:
    return _results(result).get("review-commits") or {}


def _apply(result: harness.RowResult) -> dict:
    return _results(result).get("apply-commit-fixes") or {}


def _brief(result: harness.RowResult) -> str:
    brief = str((_review(result).get("commit_range") or {}).get("brief") or "")
    assert BRIEF.match(brief), f"{result.row.cli}: brief {brief!r} is not a root brief"
    return brief


class TestTheRowRan:
    def test_the_row_did_not_hit_its_cap(self, commit):
        assert not commit.timed_out, (
            f"{commit.row.cli}: {commit.cap_reason} after "
            f"{commit.wall_clock_s}s (cap {commit.row.row_timeout_s}s)"
        )

    def test_every_command_exited_zero(self, commit):
        failed = [record for record in commit.commands if record["exit"] != 0]
        assert not failed, (
            f"{commit.row.cli}: {failed}\n"
            f"--- container stderr (tail) ---\n{commit.stderr[-3000:]}"
        )
        assert [r["command"] for r in commit.commands] == COMMANDS, (
            f"{commit.row.cli}: ran {[r['command'] for r in commit.commands]}"
        )

    def test_no_prompt_fell_through_and_no_route_went_unserved(self, commit):
        unmatched = [r["prompt_sha"] for r in commit.stub_log if r.get("unmatched")]
        unhandled = [(r["method"], r["path"]) for r in commit.stub_log
                     if r.get("unhandled")]
        assert not unmatched, (
            f"{commit.row.cli}: {len(unmatched)} prompt(s) matched no pattern "
            f"in container_commit_modes.yaml: {unmatched}"
        )
        assert not unhandled, f"{commit.row.cli}: unserved routes {unhandled}"

    def test_the_call_log_names_only_this_row_s_cli(self, commit):
        seen = {record.get("cli") for record in commit.call_log}
        seen.discard("uv")
        assert seen == {commit.row.cli}, (
            f"{commit.row.cli}: call log names {sorted(s for s in seen if s)}"
        )

    def test_one_receipt_per_command_in_order(self, commit):
        modes = [mode for mode, _result in improve_lane._receipts_by_mode(commit)]
        assert modes == ["review-commits", "apply-commit-fixes"], (
            f"{commit.row.cli}: dispatcher receipts name {modes}"
        )


class TestTheReviewTookTheRootRange:
    def test_the_review_finished_on_the_seed_commit_alone(self, commit):
        result = _review(commit)
        assert (result.get("mode"), result.get("phase"), result.get("state"),
                result.get("exit")) == ("review-commits", "code-review", "done", 0), result
        record = result.get("commit_range") or {}
        assert record.get("base") is None, (
            f"{commit.row.cli}: a one-commit repository is a root range; base="
            f"{record.get('base')!r}")
        head = str(record.get("head") or "")
        assert BRIEF.match(_brief(commit)).group(1) == head[:12], record
        assert [c.get("subject") for c in record.get("commits") or []] == [SEED_SUBJECT]
        assert (record.get("size") or {}).get("files") == COMMITTED_FILES, record

    def test_the_block_names_the_root_range_and_the_drift(self, commit):
        record = _review(commit).get("commit_range") or {}
        head7 = str(record.get("head") or "")[:7]
        lines = record.get("lines") or []
        assert lines and lines[0].startswith(f"[COMMIT_RANGE] root..{head7} (1 commit, "), lines
        assert record.get("drifted") == DRIFTED, record
        assert (f"  [SCOPE_DRIFT] 1 of {COMMITTED_FILES} reviewed files differ on disk "
                f"from {head7}: src/auth.py") in lines, lines

    def test_every_finding_was_validated(self, commit):
        counts = _review(commit).get("validation_counts") or {}
        assert counts.get("valid") == FINDINGS, (
            f"{commit.row.cli}: {counts} — the code-review verdict sheet did not "
            f"join this review's groups, so the guard below grades nothing")

    def test_the_next_step_is_the_apply_of_the_brief(self, commit):
        step = _review(commit).get("user_next_step") or {}
        assert step.get("phase") == "apply-code-fixes", step
        assert str(step.get("command") or "").endswith(
            f"--apply-commit-fixes {_brief(commit)}"), step
        for token in ("--plan-file", "--base-ref", "--head-ref"):
            assert token not in str(step.get("command")), (token, step)

    def test_the_report_opens_with_the_commit_block(self, commit):
        folder = _brief(commit)[len("plans/"):-len(".md")]
        report = commit.out_dir / "artifacts" / folder / "code-review" / "report.md"
        assert report.is_file(), (
            f"{commit.row.cli}: the review's receipt named report.md, but it is "
            f"not in the bundle at {report}")
        text = report.read_text(encoding="utf-8")
        assert "**Commit brief:**" in text and "**Plan:**" not in text
        head7 = str((_review(commit).get("commit_range") or {}).get("head") or "")[:7]
        assert f"[COMMIT_RANGE] root..{head7} (1 commit, " in text


class TestTheApplyTookTheReviewAndSkippedEveryChangedFile:
    def test_the_apply_selected_the_review_and_finished(self, commit):
        result = _apply(commit)
        assert (result.get("mode"), result.get("phase"), result.get("state"),
                result.get("exit")) == ("apply-commit-fixes", "apply-code-fixes", "done", 0), result
        record = result.get("commit_range") or {}
        assert record.get("brief") == _brief(commit), record
        assert record.get("head") == (_review(commit).get("commit_range") or {}).get("head")
        assert (record.get("selected") or {}).get("behind") == 0, record
        assert any("[COMMIT_REVIEW_SELECTED]" in line for line in record.get("lines") or []), (
            record.get("lines"))

    def test_every_fix_was_skipped_and_named(self, commit):
        fixes = _apply(commit).get("commit_fixes") or {}
        assert fixes.get("to_apply") == 0, fixes
        assert fixes.get("skipped_changed") == FINDINGS, fixes
        assert fixes.get("changed_files") == CHANGED_FIX_FILES, fixes
        head7 = str(fixes.get("head") or "")[:7]
        assert (fixes.get("lines") or [None])[0] == "[COMMIT_FIXES] 0 fixes to apply", fixes
        assert (f"  skipped  {FINDINGS} (file changed since {head7}: "
                f"{', '.join(CHANGED_FIX_FILES)})") in (fixes.get("lines") or []), fixes

    def test_no_write_batch_reached_the_stub(self, commit):
        served = improve_lane._served_fixtures(commit)
        missing = {path: what for path, what in SERVED.items() if path not in served}
        assert not missing, (
            f"{commit.row.cli}: no turn reached the stub for "
            f"{sorted(missing.values())} (served: {sorted(served)})"
        )
        extra = {path: what for path, what in NEVER_SERVED.items() if path in served}
        assert not extra, (
            f"{commit.row.cli}: the stub answered {sorted(extra.values())}, which "
            f"this row never owes"
        )

    def test_the_sample_project_did_not_move(self, commit):
        moved = sorted(name for name in commit.census_delta()
                       if name.startswith(("src/", "tests/")))
        assert not moved, (
            f"{commit.row.cli}: the commit modes moved bytes in {moved}; every fix "
            f"was skipped, and nothing here may edit or commit")

    def test_the_brief_and_its_folder_are_new_in_the_workspace(self, commit):
        delta = commit.census_delta()
        brief = _brief(commit)
        for path in (brief, "plans/commit-reviews/.gitignore",
                     f"{brief[:-len('.md')]}/state.json"):
            assert path in delta and delta[path][0] is None and delta[path][1], (
                f"{commit.row.cli}: {path} was not written by the row "
                f"({delta.get(path)})")
