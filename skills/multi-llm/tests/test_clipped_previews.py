"""Every bounded line that shows a model's words cuts them with ``text_clip.clip``.

Fix round 22 (live run 9, #4). ``status``'s progress line cut a declined fix's
reason mid-word ("… so this fix would reverse that dec)") and on OpenCode left
"(catch-up may reach 60 m/s), )"; the finished run's ``notice`` and
``status``'s "check findings for review-code" line cut T002's finding to
"… but scripts/c)", which a Codex host relayed as "truncated at 'but
scripts/c'". The texts below are the run's own (``/tmp/mllm-race-r9/``).
Each row asserts the shown preview is a word-boundary cut ending in "…",
with no separator before the mark and every bracket paired.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import run_status  # noqa: E402
from utils.text_clip import ELLIPSIS, clip, unclosed  # noqa: E402

CODEX_REASON = (
    "The implementation record deliberately allows the catch-up boost to exceed "
    "MAX_SPEED, so this fix would reverse that decision"
)
OPENCODE_REASON = (
    "Implementation record T001 and the plan's Decisions deliberately leave "
    "nextSpeed unclamped (catch-up may reach 60 m/s), and test 'a trailing "
    "opponent may exceed MAX_SPEED' pins it; adding Math.min(..., MAX_SPEED) "
    "would break it"
)
T002_FINDING = (
    "node scripts/check.mjs exited 1: check failed: retries must be 3 (found 2); "
    "src/config.json has retries 2, but scripts/check.mjs requires 3."
)
LONG_FINDINGS = [T002_FINDING, OPENCODE_REASON, CODEX_REASON]


def assert_clean_cut(line: str, source: str) -> None:
    """*line* shows a cut of *source*: ends of words, a mark, paired brackets."""
    assert ELLIPSIS in line, line
    before = line.split(ELLIPSIS)[0]
    # The last kept word is a whole word of the source.
    last = before.split()[-1].lstrip("(\"'")
    assert re.search(rf"(^|\s|\(|\"){re.escape(last)}(\s|$|[,;:.)\"'])", source), (last, line)
    assert before[-1] not in " ,;:—-/(", line
    assert unclosed(line) == [], line


class TestStatusProgressLine:
    def _line(self, reason: str) -> str:
        mirror = {
            "kind": "apply_code_fixes", "total": 2, "ok": 2,
            "items_total": 3, "items_applied": 2,
            "tasks": [{"task_id": "apply-fix-batch-0", "status": "ok"},
                      {"task_id": "apply-fix-batch-1", "status": "ok"}],
            "declined_items": [{"task_id": "apply-fix-batch-0",
                                "id": "c8fe719158c8d2be", "reason": reason}],
        }
        text = " ".join(run_status._mirror_text("write_subagents", mirror))
        assert "declined by its batch: c8fe7191 in apply-fix-batch-0 (" in text
        return text[text.index("declined by its batch"):]

    def test_codex_run_9_declined_reason(self):
        line = self._line(CODEX_REASON)
        assert "reverse that dec)" not in line
        assert line.endswith("so this fix would reverse that…)"), line
        assert_clean_cut(line, CODEX_REASON)

    def test_opencode_run_9_declined_reason_leaves_no_dangling_comma(self):
        line = self._line(OPENCODE_REASON)
        assert ", )" not in line
        assert_clean_cut(line, OPENCODE_REASON)


class TestStatusCheckFindingLines:
    def _payload(self, key: str, findings) -> dict:
        return {"phases": {"implement": {"status": "in_progress", key: [
            {"id": "T002", "findings": list(findings)}]}}}

    def test_check_findings_for_review_code(self):
        lines = run_status._workflow_trailer_lines(
            self._payload("check_findings", [T002_FINDING]), prefix="")
        line = next(item for item in lines if "check findings for review-code" in item)
        assert "but scripts/c)" not in line
        assert line.endswith("has retries 2, but…)"), line
        assert_clean_cut(line, T002_FINDING)

    def test_a_tasks_own_check_quote(self):
        lines = run_status._workflow_trailer_lines(
            self._payload("batch_check_findings", [T002_FINDING]), prefix="")
        line = next(item for item in lines if "own report says a check" in item)
        assert_clean_cut(line, T002_FINDING)

    def test_the_native_implement_block(self):
        view = {"in_progress_since": None, "completed": False,
                "phase_dir": "plans/p/implement",
                "tally": {"done": 1, "total": 3, "buckets": {}},
                "reasons": {}, "answers": {}, "resumable": False,
                "findings": [{"id": "T002", "findings": LONG_FINDINGS}],
                "batch_checks": [{"id": "T003", "findings": LONG_FINDINGS}]}
        lines = run_status.render_state_implement_lines(view)
        for word in ("check findings T002:", "own check fails T003:"):
            line = next(item for item in lines if word in item)
            assert len(line.split(": ", 1)[1]) <= 300
            assert_clean_cut(line, "; ".join(LONG_FINDINGS))

    def test_the_native_apply_block(self):
        view = {"phase_dir": "plans/p/apply-fixes", "completed": True,
                "batches_recorded": 1, "batches_total": 1,
                "batch_checks": [{"id": "apply-fix-batch-0", "findings": LONG_FINDINGS}]}
        lines = run_status.render_state_apply_lines(view)
        line = next(item for item in lines if "own check fails" in item)
        assert_clean_cut(line, "; ".join(LONG_FINDINGS))


def test_clip_keeps_the_run_9_budgets():
    """The budgets stay what they were: 120 for one finding or reason."""
    assert len(clip(CODEX_REASON, 120)) <= 120


# ===========================================================================
# Notices, the stop `error`, the batch lines (write_batch_scheduler et al.)
# ===========================================================================

from utils import known_checks, write_batch_scheduler as wbs  # noqa: E402
from utils.cli_common import ERROR_MAX_CHARS, ResultContext  # noqa: E402

TESTS_LINE = (
    "Nothing ran this repository's tests: driver.verify_command is not set. "
    "To have every code phase run them, add verify_command: \"npm test\" under "
    "driver: in .multi-llm/providers.yaml (package.json defines scripts.test)."
)


def _quote_in_brackets(headline: str) -> str:
    return headline[headline.index(" [") + 2:headline.index("]")]


class TestNotices:
    def test_a_batch_check_summary_cuts_each_finding(self):
        text = wbs.batch_check_summary({"T003": [OPENCODE_REASON]})
        first = text[text.index('("') + 2:text.index('")')]
        assert ELLIPSIS in first and len(first) <= 160
        assert_clean_cut(text, OPENCODE_REASON)

    def test_quoted_findings_close_the_quote_they_cut(self):
        shown = wbs.quoted_findings(LONG_FINDINGS, 200)
        assert len(shown) <= 200 and shown.endswith(ELLIPSIS + '"')
        assert unclosed(shown) == []

    def test_the_batch_check_line(self):
        could_not = ["its check could not run: " + " ".join(LONG_FINDINGS * 3)]
        line = wbs.batch_check_line("T003", could_not)
        assert_clean_cut(line, could_not[0])

    def test_the_write_result_notice(self):
        class Result:
            unverified = TESTS_LINE

            @staticmethod
            def batch_check_findings():
                return {f"T00{n}": [OPENCODE_REASON] for n in range(1, 9)}

        notice = wbs.write_result_notice(Result())
        assert len(notice) <= 1200
        assert_clean_cut(notice, " ".join(LONG_FINDINGS) + " " + TESTS_LINE)

    def test_implement_check_findings_notice(self):
        import implement_orchestrator

        notice = implement_orchestrator.check_findings_notice(
            {"T002": [OPENCODE_REASON, T002_FINDING], "T005": [T002_FINDING]})
        assert f"T005: {T002_FINDING}" in notice  # fits whole: unchanged
        cut = notice[notice.index("T002: ") + 6:notice.index(" (+1 more)")]
        assert len(cut) <= 160 and ELLIPSIS in cut
        assert_clean_cut(cut, OPENCODE_REASON)

    def test_the_known_checks_tally(self):
        checks = [
            (f"node scripts/check-{'the-very-long-name-' * 4}{n}.mjs --reporter spec", "T001")
            for n in range(6)
        ]
        state = {"task_status": {"apply-fix-batch-0": "completed"}}
        tally = known_checks.checks_tally(checks, state, [{"task_id": "apply-fix-batch-0"}])
        assert len(tally) <= known_checks.TALLY_MAX_CHARS and ELLIPSIS in tally
        assert unclosed(tally) == [], tally
        assert tally.split(ELLIPSIS)[0][-1] not in " ,;:"


class TestTheStopError:
    def test_the_models_quote_gives_way_at_a_word(self):
        reported = " ".join([OPENCODE_REASON] * 4)
        headline = wbs.fit_stop_headline(
            "implement stopped at T002 (Release check): its check reported 4 problem(s)",
            reported, ["T006", "T007"],
            ["To go on: fix what the check reported and resume, or accept its "
             "findings with --human-task T002=done."],
            tail=TESTS_LINE, budget=600,
        )
        assert len(headline) <= 600 and headline.endswith(TESTS_LINE)
        quote = _quote_in_brackets(headline)
        assert ELLIPSIS in quote
        assert_clean_cut(quote, reported)

    def test_the_head_cut_as_a_last_resort(self):
        head = "implement stopped at T002 (" + " ".join([OPENCODE_REASON] * 3) + ")"
        headline = wbs.fit_stop_headline(
            head, None, [], ["Resume with --resume."], tail=TESTS_LINE, budget=400,
        )
        assert len(headline) <= 400 and headline.endswith(TESTS_LINE)
        assert_clean_cut(headline.split(" Resume with")[0], head)
        assert unclosed(headline) == []

    def test_the_whole_line_kept_at_error_max_chars(self):
        """Round 19's rule holds: the tests line stays whole at 1200."""
        headline = wbs.fit_stop_headline(
            "implement stopped at T002", " ".join([OPENCODE_REASON] * 8), [],
            ["To go on: resume."], tail=TESTS_LINE,
        )
        assert len(headline) <= ERROR_MAX_CHARS and headline.endswith(TESTS_LINE)

    def test_the_result_lines_error_is_cut_at_a_word(self):
        context = ResultContext(mode="implement", phase="implement")
        text = " ".join([OPENCODE_REASON] * 8)
        context.set_error(text)
        assert len(context.error) <= ERROR_MAX_CHARS
        assert_clean_cut(context.error, text)


# ===========================================================================
# What both legs record: a declined reason, a check's findings (the executor's
# reading and `util batch-done`'s), and the native leg's printed plan of work
# ===========================================================================

from utils.subagent_executor import (  # noqa: E402
    CHECK_FINDING_MAX_CHARS,
    DECLINED_REASON_MAX_CHARS,
    check_findings_from,
    declined_reason,
)

LONG_REASON = " ".join([OPENCODE_REASON, CODEX_REASON]) + "."


class TestRecords:
    def test_a_declined_reason_is_cut_at_a_word(self):
        reason = declined_reason(LONG_REASON + " " + LONG_REASON)
        assert len(reason) <= DECLINED_REASON_MAX_CHARS
        assert_clean_cut(reason, LONG_REASON)

    def test_a_reason_that_fits_keeps_round_21s_one_mark_rule(self):
        assert declined_reason(CODEX_REASON + ".") == CODEX_REASON

    def test_a_checks_findings(self):
        found = check_findings_from(
            {"check": {"passed": False, "findings": [LONG_REASON * 2, T002_FINDING]}})
        assert found[1] == T002_FINDING
        assert len(found[0]) <= CHECK_FINDING_MAX_CHARS
        assert_clean_cut(found[0], LONG_REASON)

    def test_a_check_that_could_not_run(self):
        found = check_findings_from({"check": {
            "passed": False, "could_not_run": True, "findings": [LONG_REASON * 2]}})
        assert len(found) == 1 and len(found[0]) <= CHECK_FINDING_MAX_CHARS
        assert_clean_cut(found[0], LONG_REASON)

    def test_a_check_that_is_not_an_object(self):
        found = check_findings_from({"check": LONG_REASON})
        shown = found[0][found[0].index("was str ") + 9:found[0].index(", not an object") - 1]
        assert len(shown) <= 80 and ELLIPSIS in shown

    def test_native_batch_done_prints_and_records_the_cut_reason(
        self, tmp_path, monkeypatch, capsys,
    ):
        """`util batch-done --declined` (the native leg) prints `[BATCH_DECLINED]`
        and records the reason through the same cut, and its stderr line for
        a garbled `--listed-check` is cut the same way."""
        import json
        import subprocess

        from utils import batch_done
        from utils import provider_registry as registry
        from utils.subagent_tasks import PRIOR_CHANGES_TOKEN
        from utils.write_phase_bridge import (
            materialize_write_tasks,
            write_task_file_for_phase,
        )

        root = tmp_path / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        config = tmp_path / "providers.yaml"
        config.write_text("driver:\n  subagent_runner: codex:gpt-5.5\n", encoding="utf-8")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
        monkeypatch.setattr(registry, "_config", None)
        monkeypatch.setattr(registry, "_config_key", None)
        plan = root / "plans" / "kart.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Kart\n", encoding="utf-8")
        out = root / "plans" / "kart"
        phase_dir = out / "apply-fixes"
        phase_dir.mkdir(parents=True)
        batches = [{"file_key": "src/opponent.js",
                    "prompt": f"Fix.\n\n{PRIOR_CHANGES_TOKEN}\n",
                    "fixes": [{"group_id": "a1", "title": "A"},
                              {"group_id": "b2", "title": "B"}]}]
        tasks = materialize_write_tasks(
            kind="apply_code_fixes", batches=batches, phase_dir=str(phase_dir),
            plan_file=str(plan))
        write_task_file_for_phase(
            kind="apply_code_fixes", tasks=tasks, phase_dir=str(phase_dir),
            plan_file=str(plan))
        (out / "state.json").write_text(json.dumps({"plan_path": str(plan)}), encoding="utf-8")
        declined_id = tasks[0]["expected_ids"][1]
        reason = LONG_REASON + " " + LONG_REASON
        assert batch_done.main([
            "--phase-dir", str(phase_dir), "--task-id", "apply-fix-batch-0",
            "--status", "completed", "--declined", f"{declined_id}={reason}",
            "--listed-check", "garbled " + LONG_REASON * 3,
        ]) == 0
        captured = capsys.readouterr()
        line = next(item for item in captured.out.splitlines()
                    if item.startswith("[BATCH_DECLINED]"))
        shown = line.split(" — ", 1)[1].split(". List it")[0]
        assert len(shown) <= DECLINED_REASON_MAX_CHARS
        assert_clean_cut(shown, reason)
        ignored = next(item for item in captured.err.splitlines()
                       if "--listed-check not recorded" in item)
        assert_clean_cut(ignored.split("): ", 1)[1], "garbled " + LONG_REASON * 3)
        state = json.loads((out / "state.json").read_text())
        recorded = json.dumps(state, ensure_ascii=False)
        assert shown in recorded


def _bare(cls, **attrs):
    orch = cls.__new__(cls)
    defaults = {"args": type("A", (), {"output_format": "text"})(),
                "plan_path": "plans/p.md", "valid": [], "needs_human": [],
                "skipped": [], "batching_stats": {}, "formatted_human": [],
                "_base_ref": "HEAD", "tasks_file": "plans/p/tasks/tasks.md"}
    for key, value in {**defaults, **attrs}.items():
        setattr(orch, key, value)
    return orch


class TestTheNativePlanOfWork:
    """The apply phases' native emit prints each item's description, bounded."""

    def _item(self, description: str) -> dict:
        return {"title": "Clamp speed", "type": "bug", "importance": "HIGH",
                "file": "src/opponent.js", "reference": "## Decisions",
                "description": description}

    def _shown(self, out: str) -> str:
        return out.split("Description:\n  ", 1)[1].splitlines()[0]

    def test_code_fixes(self, capsys):
        from apply_code_fixes_orchestrator import ApplyCodeFixesOrchestrator

        orch = _bare(ApplyCodeFixesOrchestrator)
        orch.print_text_summary([{"fixes": [self._item(LONG_REASON)]}], "/dev/null")
        shown = self._shown(capsys.readouterr().out)
        assert len(shown) <= 200
        assert_clean_cut(shown, LONG_REASON)

    def test_suggestions(self, capsys):
        from apply_suggestions_orchestrator import ApplySuggestionsOrchestrator

        orch = _bare(ApplySuggestionsOrchestrator)
        orch.print_text_summary([{"suggestions": [self._item(LONG_REASON)]}], "/dev/null")
        assert_clean_cut(self._shown(capsys.readouterr().out), LONG_REASON)

    def test_a_short_description_is_shown_whole_without_a_mark(self, capsys):
        from apply_code_fixes_orchestrator import ApplyCodeFixesOrchestrator

        orch = _bare(ApplyCodeFixesOrchestrator)
        orch.print_text_summary([{"fixes": [self._item(T002_FINDING)]}], "/dev/null")
        assert self._shown(capsys.readouterr().out) == T002_FINDING


def test_the_review_fixes_section_of_the_plan(tmp_path):
    import code_review_orchestrator

    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n")
    code_review_orchestrator.update_plan_with_fixes(
        plan, [{"fixed": False, "issue_title": "Clamp", "reason": LONG_REASON}])
    line = next(item for item in plan.read_text().splitlines() if "**Clamp**" in item)
    assert_clean_cut(line, LONG_REASON)
