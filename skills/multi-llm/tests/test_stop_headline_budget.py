"""One rule fits every stop headline into the result line's ``error``
(fix round 19, the review of fix round 18: decline #5, implement #2).

The headline is the line the hosts relay whole, bounded at
``cli_common.ERROR_MAX_CHARS``. The bound used to cut its end: an apply
stop's "Or keep what it applied … --human-task <batch>=skip …" sentence (the
600-character quote of the batch's notes came first), and an implement
check-findings stop's remedy and its ``--human-task T002=done`` answer once a
check had a longer title, more tasks waited on it, and the tests line took its
reserved room — while the model's quoted words were kept whole. Now the
model's quoted words give way first, then the not-run list, then the task
titles, and only then the head; the remedy, the answers and the tests line
stay whole.
"""

from __future__ import annotations

import json

import multi_llm  # noqa: E402
from utils.cli_common import ERROR_MAX_CHARS, plain_line
from utils.write_batch_scheduler import (
    STOP_HEADLINE_QUOTE_MIN,
    fit_stop_headline,
)
from tests.test_check_gate_fixture import ALLOW, PLAN, _policy, repo  # noqa: F401
from tests.test_write_executor import FakeState, env  # noqa: F401
from tests.test_write_batch_declined import (
    CODEX_08_REPLY,
    _codex_08_respond,
    _codex_08_tasks,
    _respond_with,
    _two_item_task,
)

UNVERIFIED = "Nothing ran this repository's tests: driver.verify_command is not set."
REMEDY = "To go on: fix it, then resume this phase."
ANSWER = "Or add --human-task T1=done to the resume command."
TAIL = "Nothing ran the tests."


class TestTheRule:
    """``fit_stop_headline`` itself, at a small budget."""

    def test_a_headline_that_fits_is_the_plain_composition(self):
        assert fit_stop_headline(
            "x stopped at T1 (Title): failed", "the words", ["T2", "T3"],
            [REMEDY, ANSWER], tail=TAIL, budget=400,
        ) == (
            f"x stopped at T1 (Title): failed [the words]. Not run: T2, T3. "
            f"{REMEDY} {ANSWER} {TAIL}"
        )

    def test_the_model_s_words_give_way_first(self):
        words = "w" * 300
        text = fit_stop_headline(
            "x stopped at T1: failed", words, ["T2"], [REMEDY, ANSWER],
            tail=TAIL, budget=250,
        )
        assert len(text) == 250, len(text)
        assert text.startswith("x stopped at T1: failed [www")
        assert "w…]. Not run: T2." in text
        assert text.endswith(f"{REMEDY} {ANSWER} {TAIL}")

    def test_a_quote_too_short_to_say_anything_is_left_out(self):
        bare = len(f"x stopped at T1: failed. {REMEDY} {ANSWER} {TAIL}")
        text = fit_stop_headline(
            "x stopped at T1: failed", "w" * 300, [], [REMEDY, ANSWER],
            tail=TAIL, budget=bare + STOP_HEADLINE_QUOTE_MIN,
        )
        assert text == f"x stopped at T1: failed. {REMEDY} {ANSWER} {TAIL}"

    def test_then_the_not_run_list(self):
        ids = [f"T{n:03d}" for n in range(2, 40)]
        text = fit_stop_headline(
            "x stopped at T1: failed", "words", ids, [REMEDY, ANSWER],
            tail=TAIL, budget=200,
        )
        assert len(text) <= 200
        assert "[words]" not in text
        assert "Not run: T002, " in text and " more." in text
        assert text.endswith(f"{REMEDY} {ANSWER} {TAIL}")

    def test_a_not_run_list_with_no_room_is_left_to_the_summary(self):
        head = "x stopped at T1: failed"
        fits = len(f"{head}. {REMEDY} {ANSWER} {TAIL}")
        text = fit_stop_headline(
            head, None, ["T-with-a-long-id-" + "9" * 40], [REMEDY, ANSWER],
            tail=TAIL, budget=fits + 5,
        )
        assert text == f"{head}. {REMEDY} {ANSWER} {TAIL}"

    def test_a_caller_with_shorter_titles_to_try_gets_none(self):
        assert fit_stop_headline(
            "x stopped at T1 (" + "t" * 300 + "): failed", None, [],
            [REMEDY], tail=TAIL, budget=200, last_resort=False,
        ) is None

    def test_the_last_resort_cuts_the_head_never_the_answers(self):
        text = fit_stop_headline(
            "x stopped at T1 (" + "t" * 300 + "): failed", None, [],
            [REMEDY, ANSWER], tail=TAIL, budget=200,
        )
        assert len(text) <= 200
        assert text.startswith("x stopped at T1 (ttt")
        # Fix round 22: the cut closes the bracket the head opened.
        assert text.endswith(f"t…). {REMEDY} {ANSWER} {TAIL}")

    def test_only_a_frame_longer_than_the_budget_cuts_into_it(self):
        text = fit_stop_headline(
            "x", None, [], ["a" * 300], tail=TAIL, budget=100,
        )
        assert len(text) <= 100
        assert text.endswith(f"… {TAIL}")

    def test_it_is_one_plain_line(self):
        text = fit_stop_headline(
            "x stopped at T1 (a\n title\x1b[2J): failed", "two\n  lines",
            [], [REMEDY], tail=TAIL, budget=400,
        )
        assert text == plain_line(text)
        assert "\x1b" not in text and "\n" not in text


class TestAnApplyStopKeepsItsSkipSentence:
    """Decline #5: bounded notes (600) and a 200-character summary already
    put the apply-suggestions headline past the bound, and the cut went
    through the `=skip` sentence."""

    def test_the_skip_sentence_survives_long_notes(self, env):
        reply = dict(CODEX_08_REPLY, summary="s" * 200,
                     notes="Fix 1: not applied, deliberate per T011. " * 15)
        result = env.scheduler(
            [_two_item_task(env)], respond=_respond_with(env, reply),
        ).run()
        head = result.stop["headline"]
        assert len(head) <= ERROR_MAX_CHARS, len(head)
        assert head == plain_line(head)
        assert head.endswith(
            "Or keep what it applied and go on without suggestion 1 (`s0`): "
            "add --human-task apply-batch-0=skip to the resume command "
            "(recorded as not applied, skipped at your request; the batches "
            "after it run)."
        ), head[-300:]
        assert "To go on: inspect the half-applied state, restore or keep it, " \
            "fix the cause, then resume this phase." in head
        assert "deliberate per T011" in head, "the quote is cut, not dropped"
        assert "…]" in head

    def test_apply_code_fixes_with_long_ids_keeps_its_answer_and_tests_line(self, env):
        """16-character fix ids, a long batch title, several batches not run
        and the tests line on an apply-code-fixes stop."""
        (env.root / "package.json").write_text(
            json.dumps({"scripts": {"test": "vitest run"}}), encoding="utf-8")
        ids = ["0123456789abcdef", "fedcba9876543210", "00112233aabbccdd"]

        def task(index, expected):
            return env.apply_task(
                index, kind="apply_code_fixes", task_id=f"apply-fix-batch-{index}",
                output_name=f"apply_fix_batch_{index}.json",
                expected_ids=expected,
                task_title=(
                    f"apply {len(expected)} fix(es) to "
                    "src/simulation/physics/integrators/semi_implicit_euler.ts"
                ),
            )

        tasks = [task(0, ["aaaaaaaaaaaaaaaa"]), task(1, ids)] + [
            task(n, [f"{n:016x}"]) for n in range(2, 9)
        ]

        def respond(task, prompt):
            env.plan.write_text(env.plan.read_text() + f"{task['task_id']}\n")
            if task["task_id"] == "apply-fix-batch-0":
                return {"status": "applied", "ids": ["aaaaaaaaaaaaaaaa"],
                        "files_touched": ["plans/p.md"], "summary": "fixed it"}
            return {"status": "applied", "ids": [ids[1]],
                    "files_touched": ["plans/p.md"], "summary": "s" * 200,
                    "notes": "Fix 1 and fix 3: not applied. " * 30}

        result = env.scheduler(
            tasks, kind="apply_code_fixes", respond=respond,
            state=FakeState(env.tmp_path / "state.json"),
        ).run()

        assert result.exit_code == 3
        head = result.stop["headline"]
        assert len(head) <= ERROR_MAX_CHARS, len(head)
        assert (
            f"Or keep what it applied and go on without fix 1 (`{ids[0]}`), fix 3 "
            f"(`{ids[2]}`): add --human-task apply-fix-batch-1=skip to the resume "
            f"command (recorded as not applied, skipped at your request; the "
            f"batches after it run)."
        ) in head, head
        assert UNVERIFIED in head
        assert head.endswith("(package.json defines scripts.test)."), head[-200:]
        assert "To go on: inspect the half-applied state" in head


class TestACheckFindingsStopKeepsItsAnswer:
    """Implement #2: a longer check title, more tasks waiting on it and the
    tests line pushed the cut into the remedy and the `T002=done` answer."""

    def test_long_titles_and_several_blocked_tasks(self, repo, capsys):
        _policy(repo, ALLOW)
        (repo.root / "package.json").write_text(
            json.dumps({"scripts": {"test": "node --test"}}), encoding="utf-8")
        title = (
            "Check the module: verify the physics step against the render "
            "loop, the HUD timers and the lap counter"
        )
        plan = PLAN.replace("### T002: Check the module", f"### T002: {title}")
        # T003 waits on the check too, so T003-T007 are all held back.
        plan = plan.replace(
            "### T003: Edit: add the greeting module\nAdd the greeting.\n"
            "- Depends on: T001",
            "### T003: Edit: add the greeting module\nAdd the greeting.\n"
            "- Depends on: T001, T002",
        )
        repo.plan.write_text(plan, encoding="utf-8")
        findings = [
            "src/game.js:142 the physics step reads dt in seconds but the "
            "renderer passes milliseconds " + "x" * 220,
            "index.html:33 the module is loaded before the vendor script "
            + "y" * 220,
        ]
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
            {"passed": False, "findings": findings}))
        capsys.readouterr()
        assert multi_llm.main([
            "implement", str(repo.plan), "--yes", "--exec-subagents",
        ]) == 3
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
        error = json.loads(line.split(" ", 1)[1])["error"]

        assert len(error) <= ERROR_MAX_CHARS, len(error)
        assert error.startswith(
            "implement did not run T003, T004, T005, T006, T007: they depend "
            "on T002 (check: Check the module: verify the physics step"), error
        assert (
            "To go on: fix what the check reported (by hand, or review-code then "
            "apply-code-fixes), then resume this phase — the check runs again; "
            "or accept its findings by adding --human-task T002=done to the "
            "resume command (the tasks after it then run, told that you "
            "accepted the findings)."
        ) in error, error
        # Fix round 24: led by the check that ran.
        assert (
            "The tasks ran checks: reported problems in T002. The plugin "
            "itself ran no tests: driver.verify_command is not set."
        ) in error, error
        assert error.endswith("(package.json defines scripts.test)."), error[-200:]
        # The findings are quoted as far as they fit, and cut there.
        assert "[src/game.js:142 the physics step reads dt" in error
        assert "…]" in error
        report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
        assert report["stop"]["kind"] == "check-findings"
        assert report["stop"]["headline"] == error
