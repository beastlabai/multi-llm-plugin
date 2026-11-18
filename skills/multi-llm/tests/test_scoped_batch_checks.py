"""``driver.batch_checks: scoped``: narrowed batch checks and the full run they owe.

Opt-in. Under ``scoped`` an apply-code-fixes batch on the executor leg runs a
pytest ``driver.verify_command`` narrowed to its own tests, and the phase owes
one full run of that exact command before it can be recorded completed. These
tests hold every completion path to that debt (the executor's finalize, an
empty resume, a no-op last writer, ``--mark-completed``, the no-items exit, a
``preserve`` deferral, the composites), the fallbacks that narrow nothing
(another command, the native leg, another phase), config drift, the scoped
identity on every surface, and the default. For the default, what is proved
is: on the ``FakeState`` path, absent or ``full`` leaves the same normalised
artifacts as the golden captured on the commit before the setting; on the
production ``StateManager`` path, absent and ``full`` leave the same
normalised artifacts on this commit. Neither is a byte comparison
(docs/development/checks.md#scoped-batch-checks).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shlex
from datetime import datetime
from pathlib import Path

import pytest

import apply_code_fixes_orchestrator as acf
import full_workflow_orchestrator as fwo
from tests.scope_apply_fixtures import CODE_CASES, _issue, run_apply, write_code_review
from tests.support import scoped_checks_capture as capture
from tests.support.write_orchestrators import orchestrator_repo  # noqa: F401 - the repo fixture
from tests.test_write_executor import FakeState, env  # noqa: F401 - the env fixture
from utils import improve_workflow as iw
from utils import known_checks, scoped_checks, verification_lines, write_prompt
from utils import provider_registry as registry
from utils.provider_registry import ConfigError
from utils.state_manager import (
    ScopedCheckOwedError,
    StateManager,
    mark_phase_completed_in_state,
    owed_scoped_check_is_current,
)
from utils.unattended import NeedsHuman
from utils.write_batch_scheduler import dispatched_prompt_path

PYTEST = capture.PYTEST_COMMAND
PHASE = scoped_checks.PHASE
OWED = scoped_checks.OWED_KEY
SCOPED_SENTENCE = "run it scoped to your edits"
GOLDEN = Path(__file__).parent / "golden" / "scoped_checks_default_run.json"


def _scoped(env, command=PYTEST, *, covering=True, **driver):
    """The env's config with *command* as verify_command and scoped checks;
    with *covering*, ``tests/test_a.py`` exists, the test file known to
    cover ``src/a.py`` (a batch with no covering test runs in full)."""
    driver.setdefault("batch_checks", "scoped")
    if covering:
        (env.root / "tests").mkdir(exist_ok=True)
        (env.root / "tests" / "test_a.py").write_text("def test_a():\n    pass\n",
                                                      encoding="utf-8")
    return env.use(verify_command=f"'{command}'", **driver)


def _prompt(env, task_id):
    return Path(dispatched_prompt_path(str(env.phase_dir), task_id)).read_text(
        encoding="utf-8")


def _two_batches(env, monkeypatch, state, *, summary=None, lines=None, **over):
    capture.program_two_batches(env, monkeypatch)
    return env.scheduler(
        [capture.fix_task(env, 0), capture.fix_task(env, 1)], kind="apply_code_fixes",
        state=state, summary_file=str(summary) if summary else None,
        log=(lines.append if lines is not None else None), **over,
    ).run()


def _seed_owed(state, command=PYTEST, batches=("apply-fix-batch-0",)):
    for task_id in batches:
        assert scoped_checks.record_owed(state, command, task_id)
    state.save()


#: A code review's completion older than any a fixture stamps: written
#: explicitly, so no test needs two clock reads to differ.
EARLIER_REVIEW = "2020-01-01T00:00:00"


def _seed_owed_of_an_earlier_review(state):
    """The debt as batches admitted under an earlier code review left it:
    its ``review_completed_at`` is not the review on record now."""
    _seed_owed(state)
    state.set(OWED, {**scoped_checks.owed(state), "review_completed_at": EARLIER_REVIEW})
    state.save()


def _listed_mentions(document, command=PYTEST):
    """Every ``listed`` map in *document* (any depth) that names *command*:
    a scoped run must never appear as a run of the command as listed."""
    found = []
    if isinstance(document, dict):
        for key, value in document.items():
            if key == "listed" and isinstance(value, dict) and command in value:
                found.append(value)
            found += _listed_mentions(value, command)
    elif isinstance(document, list):
        for item in document:
            found += _listed_mentions(item, command)
    return found


@pytest.fixture
def pytest_cli(env):
    """The stand-in ``pytest`` on the env's mock PATH (exit 0 unless set)."""
    paths = capture.install_pytest(env.bin_dir, env.tmp_path)
    paths["set"] = lambda code: paths["exit"].write_text(str(code), encoding="utf-8")
    paths["calls"] = lambda: capture.pytest_calls(paths)
    return paths


# ---------------------------------------------------------------------------
# The setting, and when it narrows
# ---------------------------------------------------------------------------

class TestTheSetting:

    @pytest.mark.parametrize("command", [
        "pytest", "pytest -q", "python -m pytest", "python3 -m pytest -x --tb=short",
        "pytest -n 8 -m 'not slow'", ".venv/bin/python -m pytest -q",
        "uv run --project skills/x -- pytest -q", "uv run pytest", "py.test -k parser",
    ])
    def test_a_pytest_command_with_options_only_narrows(self, command):
        assert scoped_checks.narrowing_refusal(command) is None

    @pytest.mark.parametrize("command, why", [
        ("make test", "not a pytest command"),
        ("npm test", "not a pytest command"),
        ("pytest tests/unit", "already selects"),
        ("python -m pytest -x tests", "already selects"),
        ("pytest -q && ruff check", "more than one plain command"),
        ("FOO=1 pytest", "not a pytest command"),
        ("pytest 'unclosed", "does not parse"),
    ])
    def test_anything_else_does_not(self, command, why):
        assert why in scoped_checks.narrowing_refusal(command)

    def test_an_unknown_word_is_refused_at_load(self, tmp_path, monkeypatch):
        config = tmp_path / "c.yaml"
        config.write_text("driver:\n  batch_checks: narrow\n", encoding="utf-8")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
        registry._config = None
        with pytest.raises(ConfigError, match="driver.batch_checks is 'narrow'"):
            registry.get_batch_check_settings()
        registry._config = None

    def test_absent_is_full_from_the_base_layer(self, env):
        settings = registry.get_batch_check_settings(anchor=str(env.plan))
        assert (settings.mode, settings.mode_layer) == ("full", "base")

    def test_full_says_nothing_and_narrows_nothing(self, env):
        env.use(verify_command=f"'{PYTEST}'", batch_checks="full")
        assert scoped_checks.narrowing(
            "apply_code_fixes", executor=True, anchor=str(env.plan)) == scoped_checks.Narrowing()

    def test_each_reason_not_to_narrow_is_one_line(self, env):
        _scoped(env)
        anchor = str(env.plan)
        implement = scoped_checks.narrowing(
            "implement", executor=True, anchor=anchor, phase="implement")
        assert implement.command is None
        assert "applies to apply-code-fixes only; this implement phase" in implement.line
        native = scoped_checks.narrowing("apply_code_fixes", executor=False, anchor=anchor)
        assert native.command is None and "applies under --exec-subagents only" in native.line
        _scoped(env, "make test")
        other = scoped_checks.narrowing("apply_code_fixes", executor=True, anchor=anchor)
        assert other.command is None
        assert "`make test` cannot be narrowed" in other.line and "not a pytest" in other.line
        env.use(batch_checks="scoped")
        unset = scoped_checks.narrowing("apply_code_fixes", executor=True, anchor=anchor)
        assert unset.command is None and "driver.verify_command is not set" in unset.line

    def test_an_owed_run_of_another_command_stops_narrowing(self, env):
        _scoped(env, "pytest -x")
        state = {}
        scoped_checks.record_owed(state, PYTEST, "apply-fix-batch-0")
        answer = scoped_checks.narrowing(
            "apply_code_fixes", executor=True, anchor=str(env.plan), state=state)
        assert answer.command is None
        assert "`pytest -q`" in answer.line and "`pytest -x`" in answer.line

    def test_the_record_matches_the_exact_string_only(self):
        state = {"phases_completed": {PHASE: "2026-11-01T10:00:00"}}
        assert scoped_checks.record_owed(state, PYTEST, "b0")
        record = scoped_checks.owed(state)
        assert PHASE not in state["phases_completed"]
        assert scoped_checks.defers_to(record, PYTEST)
        for near in ("pytest  -q", " pytest -q", "pytest -q ", "pytest -Q"):
            assert not scoped_checks.defers_to(record, near)
        # Another command is never added to the same record.
        assert not scoped_checks.record_owed(state, "pytest -x", "b1")
        assert scoped_checks.owed(state)["batches"] == ["b0"]


# ---------------------------------------------------------------------------
# The prompt and the record of what a batch reported
# ---------------------------------------------------------------------------

class TestTheScopedPrompt:

    CHECKS = [(PYTEST, "the repository's driver.verify_command"),
              ("npm test", "package.json defines scripts.test")]

    def test_no_scoped_command_is_the_old_text_byte_for_byte(self):
        earlier = [("apply-fix-batch-0", ["pytest -q: 1 failed"])]
        assert known_checks.known_checks_section(self.CHECKS, earlier, scoped=None) == \
            known_checks.known_checks_section(self.CHECKS, earlier)

    def test_the_scoped_command_gets_its_own_paragraph(self):
        text = known_checks.known_checks_section(
            self.CHECKS, scoped=PYTEST, scoped_tests=("tests/test_a.py",))
        assert "The plugin never runs them itself." not in text
        assert known_checks._SCOPED_INTRO in text
        assert SCOPED_SENTENCE in text and "`pytest -q tests/test_a.py`" in text
        assert "include each of them: `tests/test_a.py`" in text
        # A run that tested nothing is never a pass.
        assert "exits with code 5" in text and "skips every test it selects" in text
        assert "Run every other command above as it is written." in text
        assert "- `npm test` (package.json defines scripts.test)" in text

    def test_a_command_not_listed_is_never_scoped(self):
        assert known_checks.known_checks_section(self.CHECKS, scoped="pytest -x") == \
            known_checks.known_checks_section(self.CHECKS)

    def test_a_scoped_outcome_is_kept_apart_and_named_in_the_tally(self):
        tasks = [{"task_id": "b0", "prior_changes_id": "p0"},
                 {"task_id": "b1", "prior_changes_id": "p1"}]
        state = {"task_status": {"b0": "completed", "b1": "completed"}}
        known_checks.set_batch_check_report_in_state(
            state, "b0", prior_changes_id="p0", findings=None, listed={},
            scoped={PYTEST: "passed"})
        known_checks.set_batch_check_report_in_state(
            state, "b1", prior_changes_id="p1", findings=None, listed={PYTEST: "passed"})
        record = state[known_checks.APPLY_FIX_BATCH_CHECKS_KEY]["b0"]
        assert record["scoped"] == {PYTEST: "passed"} and record["listed"] == {}
        assert "scoped" not in state[known_checks.APPLY_FIX_BATCH_CHECKS_KEY]["b1"]
        tally = known_checks.checks_tally([(PYTEST, "x")], state, tasks)
        assert "passed in b1" in tally and "passed scoped to its tests in b0" in tally

    def test_a_scoped_failure_is_a_finding_named_as_scoped(self):
        findings = scoped_checks.with_scoped_failures(
            None, {PYTEST: "failed"}, known_checks.LISTED_FAILED)
        assert findings == [f"{PYTEST} ({scoped_checks.SCOPED_LABEL}): failed "
                            "(the batch's report lists it as failed)"]

    def test_the_native_leg_s_prompt_never_narrows(self, env, monkeypatch):
        """`util write-prompt` binds the same task with the full text."""
        from utils.write_phase_bridge import write_task_file_for_phase

        _scoped(env)
        tasks = [capture.fix_task(env, 0)]
        write_task_file_for_phase(kind="apply_code_fixes", tasks=tasks,
                                  phase_dir=str(env.phase_dir), plan_file=str(env.plan))
        text, _task, _info = write_prompt.render_write_prompt(env.phase_dir, "apply-fix-batch-0")
        assert "The plugin never runs them itself." in text
        assert SCOPED_SENTENCE not in text


# ---------------------------------------------------------------------------
# The executor leg
# ---------------------------------------------------------------------------

class TestExecutorScopedRun:

    def test_batches_narrow_and_the_full_run_completes_the_phase(
        self, env, monkeypatch, pytest_cli,
    ):
        _scoped(env)
        state = FakeState(env.tmp_path / "state.json")
        state.mark_phase_completed(PHASE)  # an earlier cycle's completion
        state.save()
        lines, summary = [], env.phase_dir / "summary.md"
        result = _two_batches(env, monkeypatch, state, summary=summary, lines=lines)

        assert result.exit_code == 0 and result.finalized, result.messages
        for task_id in ("apply-fix-batch-0", "apply-fix-batch-1"):
            assert SCOPED_SENTENCE in _prompt(env, task_id)
        # Only the one full run: the batches' runs are the writers' own.
        assert pytest_cli["calls"]() == ["pytest -q"]
        assert OWED not in state.state and PHASE in state.state["phases_completed"]
        assert result.verification["scoped_batches"] == [
            "apply-fix-batch-0", "apply-fix-batch-1"]
        report = state.state[known_checks.APPLY_FIX_BATCH_CHECKS_KEY]["apply-fix-batch-0"]
        assert report["scoped"] == {PYTEST: "passed"} and PYTEST not in report["listed"]
        assert any("driver.batch_checks (env): scoped: each batch runs `pytest -q`" in line
                   for line in lines), lines
        text = summary.read_text(encoding="utf-8")
        assert "`pytest -q` — pass" in text
        assert "the full run 2 batch check(s) scoped to the batch's tests deferred to" in text
        # No surface records the scoped runs as runs of the command as listed.
        for document in (
            state.state,
            json.loads((env.phase_dir / "write_batch_report.json").read_text(encoding="utf-8")),
            json.loads((env.phase_dir / ".status.json").read_text(encoding="utf-8")),
        ):
            assert _listed_mentions(document) == []
        assert "pytest -q: passed" not in text and "`pytest -q`: passed" not in text

    def test_the_debt_is_on_disk_before_each_scoped_prompt_is_sent(
        self, env, monkeypatch, pytest_cli,
    ):
        """Saved before the runner is spawned, with the old completion
        dropped in the same save: a run that dies mid-batch still owes the
        full run. Observed at the spawn itself, past every refusal."""
        from utils import task_dispatch

        _scoped(env)
        state = FakeState(env.tmp_path / "state.json")
        state.mark_phase_completed(PHASE)
        state.save()
        capture.program_two_batches(env, monkeypatch)
        scheduler = env.scheduler(
            [capture.fix_task(env, 0), capture.fix_task(env, 1)],
            kind="apply_code_fixes", state=state)
        real, seen = task_dispatch.spawn_attempt, []

        def spy(plan, hooks, attempt, *args, **kwargs):
            on_disk = json.loads(Path(state.state_file).read_text(encoding="utf-8"))
            seen.append((plan.task["task_id"], SCOPED_SENTENCE in attempt.prompt,
                         (on_disk.get(OWED) or {}).get("batches"),
                         PHASE in (on_disk.get("phases_completed") or {})))
            return real(plan, hooks, attempt, *args, **kwargs)

        monkeypatch.setattr(task_dispatch, "spawn_attempt", spy)
        assert scheduler.run().exit_code == 0
        assert seen == [
            ("apply-fix-batch-0", True, ["apply-fix-batch-0"], False),
            ("apply-fix-batch-1", True, ["apply-fix-batch-0", "apply-fix-batch-1"], False),
        ]

    def test_a_failing_full_run_stops_at_exit_3_and_a_resume_runs_only_it(
        self, env, monkeypatch, pytest_cli,
    ):
        _scoped(env)
        pytest_cli["set"](1)
        state = FakeState(env.tmp_path / "state.json")
        result = _two_batches(env, monkeypatch, state)
        assert result.exit_code == 3 and not result.finalized
        assert scoped_checks.owed(state) is not None
        assert PHASE not in (state.state.get("phases_completed") or {})
        blob = "\n".join(result.messages)
        assert "`pytest -q` exited 1" in blob and "collected 3 items" in blob
        assert "scoped to the batch's tests deferred to" in blob

        # Fixed by hand; the resume dispatches nothing and runs only the check.
        pytest_cli["set"](0)
        again = env.scheduler(
            [capture.fix_task(env, 0), capture.fix_task(env, 1)],
            kind="apply_code_fixes", state=state, resume=True).run()
        assert again.exit_code == 0 and again.finalized, again.messages
        assert all(outcome.skipped for outcome in again.outcomes)
        assert pytest_cli["calls"]() == ["pytest -q", "pytest -q"]
        assert OWED not in state.state and PHASE in state.state["phases_completed"]

    def test_an_empty_resume_runs_the_owed_check(self, env, pytest_cli):
        """Nothing left to dispatch (every batch handled, or the rest skipped
        in the report): the finalize still owes and runs the full check."""
        _scoped(env)
        state = FakeState(env.tmp_path / "state.json")
        _seed_owed(state)
        pytest_cli["set"](2)
        failed = env.scheduler([], kind="apply_code_fixes", state=state, resume=True).run()
        assert failed.exit_code == 3 and scoped_checks.owed(state) is not None
        pytest_cli["set"](0)
        passed = env.scheduler([], kind="apply_code_fixes", state=state, resume=True).run()
        assert passed.exit_code == 0 and OWED not in state.state
        assert pytest_cli["calls"]() == ["pytest -q", "pytest -q"]

    def test_a_no_op_last_writer_still_owes_the_full_run(self, env, monkeypatch, pytest_cli):
        """With no edit on record the default skips the check; an owed full
        run is run anyway."""
        _scoped(env)
        state = FakeState(env.tmp_path / "state.json")
        _seed_owed(state)
        env.edits([])
        monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["f1"]))
        pytest_cli["set"](1)
        result = env.scheduler([capture.fix_task(env, 1)], kind="apply_code_fixes",
                               state=state).run()
        assert result.outcomes[0].write_outcome == "no-op"
        assert result.exit_code == 3 and pytest_cli["calls"]() == ["pytest -q"]
        assert scoped_checks.owed(state) is not None

    def test_a_stop_says_the_full_run_is_owed(self, env, monkeypatch, pytest_cli):
        _scoped(env)
        monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Apply fix 1")
        state = FakeState(env.tmp_path / "state.json")
        result = _two_batches(env, monkeypatch, state)
        assert result.exit_code == 3 and result.stopped_at_batch == 1
        assert result.unverified.startswith(scoped_checks.OWED_LEAD)
        assert scoped_checks.OWED_LEAD in result.stop["headline"]
        assert verification_lines.verification_line(result).startswith("not run. ")
        assert pytest_cli["calls"]() == []
        assert scoped_checks.owed(state)["batches"] == ["apply-fix-batch-0", "apply-fix-batch-1"]


class TestAScopedPassIsNeverVacuous:
    """A batch runs the full check when a scoped one would test nothing of
    what it changes, and a scoped ``passed`` stands only with the scoped
    run itself in the reply."""

    @pytest.mark.parametrize("paths,known,why", [
        (["src/a.py"], ["tests/test_a.py", "src/a.py"], None),
        (["tests/test_a.py"], [], None),
        (["src/a.py"], ["src/a.py"], "no test file is known to cover src/a.py"),
        (["src/a.py", "README.md"], ["tests/test_a.py"], "no test file is known to cover README.md"),
        (["tests/conftest.py"], ["tests/test_conftest.py"], "it changes tests/conftest.py"),
        (["pytest.ini"], [], "it changes pytest.ini"),
        (["tox.ini"], [], "it changes tox.ini"),
        (["src/a.py"], None, "the repository's test files could not be listed"),
        ([], [], "it declares no file a test could cover"),
    ])
    def test_when_a_batch_runs_in_full(self, tmp_path, paths, known, why):
        found, tests = scoped_checks.full_check_reason(paths, str(tmp_path), known)
        assert (found is None) == (why is None) and (why is None or found.startswith(why))
        if why is None:
            assert tests and all(t.endswith(".py") for t in tests)

    @pytest.mark.parametrize("name,text,shared", [
        ("pyproject.toml", "[tool.pytest.ini_options]\naddopts = '-q'\n", True),
        ("pyproject.toml", "[project]\nname = 'x'\n", False),
        ("setup.cfg", "[tool:pytest]\naddopts = -q\n", True),
        ("setup.cfg", "[metadata]\nname = x\n", False),
    ])
    def test_pytest_config_in_a_shared_file(self, tmp_path, name, text, shared):
        (tmp_path / name).write_text(text, encoding="utf-8")
        found = scoped_checks.shared_collection_file([name], str(tmp_path))
        assert (found == name) is shared

    def test_a_batch_changing_conftest_runs_the_full_check(self, env, monkeypatch, pytest_cli):
        _scoped(env)
        (env.root / "tests" / "conftest.py").write_text("", encoding="utf-8")
        capture.program_two_batches(env, monkeypatch)
        task = capture.fix_task(env, 0)
        task["declared_files"] = [str(env.root / "src/a.py"), str(env.root / "tests/conftest.py")]
        lines = []
        state = FakeState(env.tmp_path / "state.json")
        result = env.scheduler([task], kind="apply_code_fixes", state=state,
                               log=lines.append).run()
        assert SCOPED_SENTENCE not in _prompt(env, "apply-fix-batch-0")
        assert scoped_checks.owed(state) is None and result.verification.get("scoped_batches") is None
        assert any("in full, not scoped to its tests: it changes tests/conftest.py" in line
                   for line in lines), lines

    def test_a_batch_no_test_covers_runs_the_full_check(self, env, monkeypatch, pytest_cli):
        _scoped(env, covering=False)
        lines = []
        state = FakeState(env.tmp_path / "state.json")
        result = _two_batches(env, monkeypatch, state, lines=lines)
        assert result.exit_code == 0, result.messages
        assert SCOPED_SENTENCE not in _prompt(env, "apply-fix-batch-0")
        assert scoped_checks.owed(state) is None
        assert any("no test file is known to cover src/a.py" in line for line in lines), lines

    @pytest.mark.parametrize("ran", [
        {},                                  # nothing named at all
        {PYTEST: "passed"},                  # the full command, not a scoped run
        {"pytest -q --lf": "passed"},        # options, no test selected
        {"pytest -q tests/test_a.py": "failed"},
        {"pytest -q tests/test_a.py || true": "passed"},
        {"pytest -q tests/test_a.py tests/test_b.py -k test_b": "passed"},
    ], ids=["none", "full", "options-only", "failed-run", "or-true", "filtered"])
    def test_a_pass_the_reply_does_not_back_is_not_run(self, env, monkeypatch, pytest_cli, ran):
        _scoped(env)
        capture.program_two_batches(env, monkeypatch)
        monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
            {"passed": True, "findings": [], "ran": ran, "listed": {PYTEST: "passed"}}))
        lines = []
        state = FakeState(env.tmp_path / "state.json")
        result = env.scheduler([capture.fix_task(env, 0)], kind="apply_code_fixes",
                               state=state, log=lines.append).run()
        assert result.exit_code == 0, result.messages
        report = state.state[known_checks.APPLY_FIX_BATCH_CHECKS_KEY]["apply-fix-batch-0"]
        assert report["scoped"] == {PYTEST: "not run"}
        assert any("is recorded as not run" in line for line in lines), lines

    def test_a_scoped_run_must_include_the_named_tests(self):
        told = ("tests/test_a.py", "tests/test_b.py")
        long_run = PYTEST + " " + " ".join(f"tests/test_{i:03d}_long_name.py" for i in range(40))
        cases = [
            ({"pytest -q tests/test_a.py": "passed"}, "not run", "left out tests/test_b.py"),
            ({"pytest -q tests/test_a.py": "passed", "pytest -q tests/test_b.py": "passed"},
             "passed", None),
            ({"pytest -q -x ./tests/test_a.py::test_x tests/test_b.py": "passed"},
             "passed", None),
            ({"pytest -q -k fast ./tests/test_a.py::test_x tests/test_b.py": "passed"},
             "not run", "it adds -k"),
            ({long_run + " tests/test_a.py tests/test_b.py": "passed"}, "passed", None),
        ]
        for ran, word, why in cases:
            backed, found = scoped_checks.backed_scoped(
                {PYTEST: "passed"}, PYTEST, ran, ["src/a.py"], "/nonexistent", told)
            assert backed == {PYTEST: word}, (ran, found)
            assert (found is None) == (why is None) and (why is None or why in found)

    def test_an_absolute_path_to_the_named_test_backs_the_pass(self, env, monkeypatch, pytest_cli):
        _scoped(env)
        capture.program_two_batches(env, monkeypatch)
        state = FakeState(env.tmp_path / "state.json")
        scheduler = env.scheduler([capture.fix_task(env, 0)], kind="apply_code_fixes",
                                  state=state)
        # The root the scheduler pinned, spelled absolute as a batch would.
        absolute = shlex.quote(Path(scheduler.repo_root, "tests", "test_a.py").as_posix())
        monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
            {"passed": True, "findings": [], "ran": {f"{PYTEST} {absolute}": "passed"},
             "listed": {PYTEST: "passed"}}))
        result = scheduler.run()
        assert result.exit_code == 0, result.messages
        report = state.state[known_checks.APPLY_FIX_BATCH_CHECKS_KEY]["apply-fix-batch-0"]
        assert report["scoped"] == {PYTEST: "passed"}

    def test_a_backed_pass_stands(self):
        backed, why = scoped_checks.backed_scoped(
            {PYTEST: "passed"}, PYTEST, {"pytest -q tests/test_a.py::test_x": "passed"},
            ["src/a.py"], "/nonexistent")
        assert backed == {PYTEST: "passed"} and why is None
        changed, why = scoped_checks.backed_scoped(
            {PYTEST: "passed"}, PYTEST, {"pytest -q tests/test_a.py": "passed"},
            ["src/a.py", "conftest.py"], "/nonexistent")
        assert changed == {PYTEST: "not run"} and "conftest.py" in why


class TestScopedEvidenceIsReadAsArgv:
    """A scoped pass is credited from the reported run's argv: the
    configured command's argv, then test selectors canonical against the
    repository root, with no shell composition and no option that selects
    what pytest runs. Anything else is recorded ``not run``."""

    TOLD = ("tests/test_keep.py", "tests/test_drop.py")

    def _backed(self, ran, root="/nonexistent", command=PYTEST, told=TOLD):
        return scoped_checks.backed_scoped(
            {command: "passed"}, command, {ran: "passed"}, ["src/keep.py"], root, told)

    def test_a_filtered_run_credits_neither_file(self):
        backed, why = self._backed("pytest -q tests/test_keep.py tests/test_drop.py -k keep")
        assert backed == {PYTEST: "not run"} and "it adds -k" in why

    @pytest.mark.parametrize("tail", [
        "|| true", "&& true", "; true", "| tee out.txt", "> out.txt", "2>&1",
        "&", "$(true)", "`true`", "# tests/test_more.py",
    ])
    def test_shell_composition_credits_nothing(self, tail):
        backed, why = self._backed(f"pytest -q tests/test_keep.py tests/test_drop.py {tail}")
        assert backed == {PYTEST: "not run"}, tail
        assert "more than one plain command" in why, (tail, why)

    def test_a_line_break_credits_nothing(self):
        backed, why = self._backed("pytest -q tests/test_keep.py tests/test_drop.py\ntrue")
        assert backed == {PYTEST: "not run"} and "more than one plain command" in why

    @pytest.mark.parametrize("option", [
        "-m slow", "-mslow", "-kkeep", "-qk keep", "--deselect tests/test_drop.py::t",
        "--deselect=tests/test_drop.py::t", "--lf", "--last-failed", "--ff",
        "--failed-first", "--sw", "--stepwise", "--co", "--collect-only", "--collectonly",
        "--ignore tests/test_drop.py", "--ignore-glob=*drop*", "-o addopts=-kkeep",
        "--override-ini=addopts=-kkeep", "-c other.ini", "--pyargs", "--setup-only",
    ])
    def test_a_selecting_option_in_any_spelling_credits_nothing(self, option):
        backed, why = self._backed(f"pytest -q {option} tests/test_keep.py tests/test_drop.py")
        assert backed == {PYTEST: "not run"}, option
        assert "it adds -" in why, (option, why)

    @pytest.mark.parametrize("extra", ["@args.txt", "-- @args.txt", "-W @args.txt"])
    def test_an_argument_file_credits_nothing(self, extra):
        """pytest reads ``@file`` as more arguments, which may hold a ``-k``."""
        backed, why = self._backed(f"pytest -q tests/test_keep.py tests/test_drop.py {extra}")
        assert backed == {PYTEST: "not run"} and "more arguments from @args.txt" in why

    @pytest.mark.parametrize("option", [
        "-x", "--exitfirst", "-rA", "-n 4", "-n4", "-p no:cacheprovider", "--tb=short",
        "-W error", "-vv", "--maxfail 2",
    ])
    def test_an_option_that_selects_nothing_still_counts(self, option):
        backed, why = self._backed(f"pytest -q {option} tests/test_keep.py tests/test_drop.py")
        assert backed == {PYTEST: "passed"} and why is None, (option, why)

    def test_quoted_shell_characters_are_one_word(self):
        backed, why = self._backed(
            "pytest -q tests/test_keep.py tests/test_drop.py -W 'error::a|b'")
        assert backed == {PYTEST: "passed"} and why is None, why

    def test_the_configured_command_s_own_filter_is_not_the_batch_s(self):
        command = 'pytest -q -m "not slow"'
        backed, why = self._backed(
            'pytest -q -m "not slow" tests/test_keep.py tests/test_drop.py', command=command)
        assert backed == {command: "passed"} and why is None, why

    @pytest.mark.parametrize("spell", [
        "{root}/tests/test_keep.py {root}/tests/test_drop.py",
        "./tests/test_keep.py ./tests/test_drop.py",
        "tests//test_keep.py tests/sub/../test_drop.py",
        "{root}/tests/test_keep.py::test_x ./tests/test_drop.py::TestY::test_z",
        "'{root}/tests/test_keep.py' -- tests/test_drop.py",
    ])
    def test_every_spelling_of_one_file_matches(self, tmp_path, spell):
        root = tmp_path / "repo root"
        text = spell.replace("{root}/", shlex.quote(root.as_posix()) + "/") \
            if "'{root}" not in spell else spell.replace("{root}", root.as_posix())
        backed, why = self._backed(f"pytest -q {text}", root=str(root))
        assert backed == {PYTEST: "passed"} and why is None, (text, why)

    def test_a_told_test_spelled_with_dot_slash_matches(self, tmp_path):
        backed, why = self._backed("pytest -q tests/test_keep.py tests/test_drop.py",
                                   root=str(tmp_path),
                                   told=("./tests/test_keep.py", "tests/./test_drop.py"))
        assert backed == {PYTEST: "passed"} and why is None, why

    def test_a_file_outside_the_repository_is_left_out(self, tmp_path):
        root, other = tmp_path / "repo", tmp_path / "other"
        outside = shlex.quote((other / "tests" / "test_keep.py").as_posix())
        backed, why = self._backed(f"pytest -q {outside} tests/test_drop.py", root=str(root))
        assert backed == {PYTEST: "not run"} and "left out tests/test_keep.py" in why

    def test_canonical_path(self, tmp_path):
        root = str(tmp_path)
        for spelled in ("tests/test_a.py", "./tests/test_a.py", "tests/x/../test_a.py",
                        (tmp_path / "tests" / "test_a.py").as_posix(),
                        str(tmp_path / "tests" / "test_a.py")):
            assert scoped_checks.canonical_path(spelled, root) == "tests/test_a.py", spelled

    def test_a_rejected_run_beside_a_credited_one_credits_only_that_one(self):
        backed, why = scoped_checks.backed_scoped(
            {PYTEST: "passed"}, PYTEST,
            {"pytest -q tests/test_keep.py": "passed",
             "pytest -q tests/test_drop.py --lf": "passed"},
            ["src/keep.py"], "/nonexistent", self.TOLD)
        assert backed == {PYTEST: "not run"}
        assert "left out tests/test_drop.py" in why and "it adds --lf" in why


class TestARefusedBatchOwesNothing:
    """The debt is written immediately before the narrowed prompt is sent:
    a batch refused before it (the write policy, the runner preflight) owes
    no full run, and leaves no debt for a later run to trip on."""

    def test_a_write_policy_denial_records_no_debt(self, env, monkeypatch, pytest_cli):
        _scoped(env, write_paths={"allow": ["src/b.py"]})
        state = FakeState(env.tmp_path / "state.json")
        state.mark_phase_completed(PHASE)
        state.save()
        result = _two_batches(env, monkeypatch, state)
        assert result.exit_code == 3, result.messages
        assert result.outcomes[0].write_outcome == "policy-violation"
        assert scoped_checks.owed(state) is None
        on_disk = json.loads(Path(state.state_file).read_text(encoding="utf-8"))
        assert OWED not in on_disk

    def test_a_runner_preflight_refusal_records_no_debt(self, env, monkeypatch, pytest_cli):
        import utils.write_grant as write_grant
        from utils import subagent_executor as executor

        _scoped(env)

        def refused(spec, **kwargs):
            raise executor.RunnerRejected(
                "The runner cannot serve this batch.", key="driver.subagent_runner",
                spec=spec, reason_code="missing_profile")

        monkeypatch.setattr(write_grant, "preflight_runner", refused)
        state = FakeState(env.tmp_path / "state.json")
        result = _two_batches(env, monkeypatch, state)
        assert result.exit_code != 0 and not result.finalized
        assert scoped_checks.owed(state) is None
        assert pytest_cli["calls"]() == []

    def test_a_refused_batch_then_a_changed_command_does_not_block_a_fresh_run(
        self, env, monkeypatch, pytest_cli,
    ):
        """Refused by the policy, then ``verify_command`` changed: a
        ``--fresh`` run that narrows nothing completes, with no debt from
        the batch that never started."""
        _scoped(env, write_paths={"allow": ["src/b.py"]})
        state = FakeState(env.tmp_path / "state.json")
        assert _two_batches(env, monkeypatch, state).exit_code == 3
        env.use(verify_command="'pytest -x'", batch_checks="scoped")
        again = env.scheduler([], kind="apply_code_fixes", state=state).run()
        assert again.exit_code == 0 and again.finalized, again.messages
        assert PHASE in state.state["phases_completed"]

    def test_a_record_that_cannot_be_saved_refuses_the_batch(
        self, env, monkeypatch, pytest_cli,
    ):
        """The prompt is already narrowed when the debt is written: a save
        that fails refuses the batch with nothing spawned."""
        from utils import task_dispatch

        _scoped(env)
        state = FakeState(env.tmp_path / "state.json")
        spawned = []
        real = task_dispatch.spawn_attempt
        monkeypatch.setattr(task_dispatch, "spawn_attempt",
                            lambda *a, **k: spawned.append(1) or real(*a, **k))

        save = state.save

        def broken():
            if OWED in state.state:  # the save that records the debt
                state.state.pop(OWED)
                raise OSError("disk full")
            save()

        monkeypatch.setattr(state, "save", broken)
        result = _two_batches(env, monkeypatch, state)
        assert result.exit_code == 3 and spawned == []
        assert "could not be recorded in state.json (disk full)" in result.outcomes[0].reason


class TestTheTallyNamesEachScopedOutcome:
    """The full verdict carries what each batch's scoped run recorded, so a
    ``not run`` never counts as a ``passed``: on the summary's Verification
    line, on a stop and on a resume."""

    def test_the_verification_line_names_each_batch(self, env, monkeypatch, pytest_cli):
        _scoped(env)
        state = FakeState(env.tmp_path / "state.json")
        summary = env.phase_dir / "summary.md"
        result = _two_batches(env, monkeypatch, state, summary=summary)
        assert result.exit_code == 0, result.messages
        assert result.verification["scoped_outcomes"] == {
            "apply-fix-batch-0": "passed", "apply-fix-batch-1": "passed"}
        line = verification_lines.verification_line(result)
        assert line.endswith("(scoped: passed in apply-fix-batch-0, apply-fix-batch-1))"), line
        assert "(scoped: passed in apply-fix-batch-0, apply-fix-batch-1)" in summary.read_text(
            encoding="utf-8")

    def test_a_not_run_batch_is_named_apart(self, env, monkeypatch, pytest_cli):
        _scoped(env)
        state = FakeState(env.tmp_path / "state.json")
        # Batch 0 reports no scoped run at all: recorded "not run".
        capture.program_two_batches(env, monkeypatch)
        monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
            {"passed": True, "findings": [], "ran": {}, "listed": {PYTEST: "passed"}}))
        first = env.scheduler([capture.fix_task(env, 0)], kind="apply_code_fixes",
                              state=state, finalize=False).run()
        assert first.exit_code == 0, first.messages
        capture.program_two_batches(env, monkeypatch)
        result = env.scheduler([capture.fix_task(env, 0), capture.fix_task(env, 1)],
                               kind="apply_code_fixes", state=state, resume=True).run()
        assert result.exit_code == 0, result.messages
        assert result.verification["scoped_outcomes"] == {
            "apply-fix-batch-0": "not run", "apply-fix-batch-1": "passed"}
        line = verification_lines.verification_line(result)
        assert "(scoped: passed in apply-fix-batch-1; not run in apply-fix-batch-0)" in line, line

    def test_a_stop_names_each_batch(self, env, monkeypatch, pytest_cli):
        _scoped(env)
        monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Apply fix 1")
        state = FakeState(env.tmp_path / "state.json")
        result = _two_batches(env, monkeypatch, state)
        assert result.exit_code == 3
        assert "(scoped: passed in apply-fix-batch-0; not reported in apply-fix-batch-1)" in \
            result.unverified, result.unverified
        assert "scoped: passed in apply-fix-batch-0" in verification_lines.verification_line(result)

    def test_a_resume_after_a_failed_full_run_names_the_earlier_batches(
        self, env, monkeypatch, pytest_cli,
    ):
        _scoped(env)
        pytest_cli["set"](1)
        state = FakeState(env.tmp_path / "state.json")
        failed = _two_batches(env, monkeypatch, state)
        assert failed.exit_code == 3
        assert "(scoped: passed in apply-fix-batch-0, apply-fix-batch-1)" in "\n".join(
            failed.messages)
        pytest_cli["set"](0)
        again = env.scheduler([], kind="apply_code_fixes", state=state, resume=True).run()
        assert again.exit_code == 0, again.messages
        assert again.verification["scoped_outcomes"] == {
            "apply-fix-batch-0": "passed", "apply-fix-batch-1": "passed"}


class TestFallbacksNarrowNothing:

    def test_another_command_runs_full_checks_with_one_line(self, env, monkeypatch, pytest_cli):
        _scoped(env, "pytest tests/unit")
        lines = []
        state = FakeState(env.tmp_path / "state.json")
        result = _two_batches(env, monkeypatch, state, lines=lines)
        assert result.exit_code == 0
        prompt = _prompt(env, "apply-fix-batch-0")
        assert SCOPED_SENTENCE not in prompt and "The plugin never runs them itself." in prompt
        assert OWED not in state.state and "scoped_batches" not in result.verification
        said = [line for line in lines if "driver.batch_checks" in line]
        assert len(said) == 1 and "cannot be narrowed" in said[0], said

    def test_the_simulated_native_invoker_never_narrows(self, env, pytest_cli):
        _scoped(env)
        lines = []
        state = FakeState(env.tmp_path / "state.json")
        respond = env.native_edits(
            [{"path": "src/a.py", "mode": "patch", "content": "a = 2\n"}],
            {"status": "applied", "ids": ["f0"], "files_touched": ["src/a.py"],
             "summary": "fixed", "check": capture.BATCH_CHECK})
        result = env.scheduler([capture.fix_task(env, 0)], kind="apply_code_fixes",
                               state=state, respond=respond, log=lines.append).run()
        assert result.exit_code == 0
        assert OWED not in state.state
        assert any("applies under --exec-subagents only" in line for line in lines), lines

    def test_another_phase_narrows_nothing(self, env):
        _scoped(env)
        lines = []
        env.scheduler([], kind="implement", log=lines.append,
                      state=FakeState(env.tmp_path / "state.json")).run()
        assert any("applies to apply-code-fixes only; this implement phase" in line
                   for line in lines), lines


class TestConfigDrift:

    def test_a_changed_command_refuses_and_runs_nothing(self, env, pytest_cli):
        _scoped(env, "pytest -x")
        state = FakeState(env.tmp_path / "state.json")
        _seed_owed(state)
        summary = env.phase_dir / "summary.md"
        result = env.scheduler([], kind="apply_code_fixes", state=state, resume=True,
                               summary_file=str(summary)).run()
        assert result.exit_code == 3 and not result.finalized
        assert pytest_cli["calls"]() == []
        blob = "\n".join(result.messages)
        assert "`pytest -q`" in blob and "`pytest -x` (env)" in blob
        assert "Set driver.verify_command back to `pytest -q`" in blob
        assert scoped_checks.owed(state) is not None
        assert "**Verification**: not run. The full check is owed" in summary.read_text(
            encoding="utf-8")

    def test_an_unset_command_refuses_too(self, env, pytest_cli):
        env.use(batch_checks="scoped")
        state = FakeState(env.tmp_path / "state.json")
        _seed_owed(state)
        result = env.scheduler([], kind="apply_code_fixes", state=state, resume=True).run()
        assert result.exit_code == 3 and pytest_cli["calls"]() == []
        assert "now reads nothing (it is unset)" in "\n".join(result.messages)

    def test_back_to_full_with_the_same_command_runs_it(self, env, pytest_cli):
        env.use(verify_command=f"'{PYTEST}'", batch_checks="full")
        state = FakeState(env.tmp_path / "state.json")
        _seed_owed(state)
        lines = []
        result = env.scheduler([], kind="apply_code_fixes", state=state, resume=True,
                               log=lines.append).run()
        assert result.exit_code == 0 and OWED not in state.state
        assert pytest_cli["calls"]() == ["pytest -q"]
        assert any("driver.batch_checks is now full" in line for line in lines), lines

    def test_a_command_changed_mid_run_is_not_narrowed_and_blocks_completion(
        self, env, monkeypatch, pytest_cli,
    ):
        _scoped(env)
        state = FakeState(env.tmp_path / "state.json")
        lines = []

        def change_it(index, _tasks):
            if index == 1:
                _scoped(env, "pytest -x")
            return None

        result = _two_batches(env, monkeypatch, state, lines=lines, before_batch=change_it)
        assert SCOPED_SENTENCE in _prompt(env, "apply-fix-batch-0")
        assert SCOPED_SENTENCE not in _prompt(env, "apply-fix-batch-1")
        assert any("apply-fix-batch-1: driver.batch_checks" in line for line in lines), lines
        assert result.exit_code == 3 and pytest_cli["calls"]() == []
        assert scoped_checks.owed(state)["batches"] == ["apply-fix-batch-0"]


# ---------------------------------------------------------------------------
# The completions the apply orchestrator records itself (both legs)
# ---------------------------------------------------------------------------

#: The batch check the orchestrator repo's batches report: the scoped run of
#: the test file known to cover `src/main.py`, which they change.
MAIN_BATCH_CHECK = {
    "passed": True, "findings": [],
    "ran": {"pytest -q tests/test_main.py": "passed"},
    "listed": {PYTEST: "passed"},
}


def _orchestrator_config(repo, command=PYTEST, batch_checks="scoped"):
    config = repo.tmp_path / "driver.yaml"
    config.write_text(
        "driver:\n  subagent_runner: codex:gpt-5.5\n"
        f"  verify_command: '{command}'\n  batch_checks: {batch_checks}\n",
        encoding="utf-8")
    registry._config = None
    # The test file known to cover `src/main.py`, which the batches change.
    (repo.root / "tests").mkdir(exist_ok=True)
    (repo.root / "tests" / "test_main.py").write_text("def test_main():\n    pass\n",
                                                      encoding="utf-8")
    paths = capture.install_pytest(repo.tmp_path / "bin", repo.tmp_path)
    paths["set"] = lambda code: paths["exit"].write_text(str(code), encoding="utf-8")
    return paths


def _seed_plan_owed(plan, *, completed=True):
    state = StateManager(plan)
    if completed:
        state.mark_phase_completed(PHASE)  # a stale completion
    _seed_owed(state)
    assert not StateManager(plan).is_phase_completed(PHASE)


def _plan_state(plan):
    return StateManager(plan)


class TestOrchestratorCompletions:

    def _review(self, repo, cases):
        return write_code_review(repo.root, cases=cases, review_policy=None)["plan"]

    def test_mark_completed_runs_the_owed_check(self, repo, capsys):
        plan = self._review(repo, {"in_scope": CODE_CASES["in_scope"]})
        paths = _orchestrator_config(repo)
        _seed_plan_owed(plan)
        paths["set"](1)
        assert run_apply(acf, plan, "--mark-completed") == 3
        err = capsys.readouterr().err
        assert "NOT marked completed" in err and "`pytest -q`" in err
        assert not _plan_state(plan).is_phase_completed(PHASE)
        assert scoped_checks.owed(_plan_state(plan)) is not None
        paths["set"](0)
        assert run_apply(acf, plan, "--mark-completed") == 0
        state = _plan_state(plan)
        assert state.is_phase_completed(PHASE) and scoped_checks.owed(state) is None
        assert capture.pytest_calls(paths) == ["pytest -q", "pytest -q"]
        # It is step 10 itself: no line telling the host to skip it.
        assert "[SCOPED_CHECK_SETTLED]" not in capsys.readouterr().err

    def test_mark_completed_refuses_a_changed_command(self, repo, capsys):
        plan = self._review(repo, {"in_scope": CODE_CASES["in_scope"]})
        paths = _orchestrator_config(repo, command="pytest -x")
        _seed_plan_owed(plan)
        assert run_apply(acf, plan, "--mark-completed") == 3
        assert "Set driver.verify_command back to `pytest -q`" in capsys.readouterr().err
        assert capture.pytest_calls(paths) == []
        assert not _plan_state(plan).is_phase_completed(PHASE)

    @pytest.mark.parametrize("flags", [
        ("--yes",),                       # nothing left: the no-items exit
        ("--yes", "--fresh"),             # --fresh keeps the debt
        ("--yes", "--exec-subagents"),    # the executor leg reaches the same exit
    ])
    def test_the_no_items_exit_runs_the_owed_check(self, repo, capsys, flags):
        cases = {"bad": ([_issue("Wrong", "Not a bug.")], "invalid", ("legacy",))}
        plan = self._review(repo, cases)
        paths = _orchestrator_config(repo)
        _seed_plan_owed(plan)
        paths["set"](4)
        assert run_apply(acf, plan, *flags, "--base-ref", "HEAD") == 3
        assert "NOT marked completed" in capsys.readouterr().err
        assert scoped_checks.owed(_plan_state(plan)) is not None
        paths["set"](0)
        assert run_apply(acf, plan, *flags, "--base-ref", "HEAD") == 0
        state = _plan_state(plan)
        assert state.is_phase_completed(PHASE) and scoped_checks.owed(state) is None

    def test_a_preserve_deferral_of_everything_left_runs_the_owed_check(self, repo, capsys):
        review = write_code_review(
            repo.root, cases={"extension": CODE_CASES["extension"]}, review_policy="preserve")
        plan = review["plan"]
        paths = _orchestrator_config(repo)
        _seed_plan_owed(plan)
        paths["set"](1)
        assert run_apply(acf, plan, "--yes", "--base-ref", "HEAD") == 3
        assert "[SCOPED_CHECK_SETTLED]" not in capsys.readouterr().err
        assert not _plan_state(plan).is_phase_completed(PHASE)
        paths["set"](0)
        assert run_apply(acf, plan, "--yes", "--base-ref", "HEAD") == 0
        state = _plan_state(plan)
        assert state.is_phase_completed(PHASE) and scoped_checks.owed(state) is None
        assert capture.pytest_calls(paths) == ["pytest -q", "pytest -q"]
        err = capsys.readouterr().err
        assert len([line for line in err.splitlines()
                    if line.startswith("[SCOPED_CHECK_SETTLED] ")]) == 1, err

    @pytest.mark.parametrize("flags", [("--yes",), ("--yes", "--fresh")])
    def test_the_no_items_exit_says_the_check_is_settled(self, repo, capsys, flags):
        """The no-items exit ran the owed check and recorded the phase
        completed: one ``[SCOPED_CHECK_SETTLED]`` line says so, so the native
        host's step 7 and step 10 run nothing and the suite runs once."""
        cases = {"bad": ([_issue("Wrong", "Not a bug.")], "invalid", ("legacy",))}
        plan = self._review(repo, cases)
        paths = _orchestrator_config(repo)
        _seed_plan_owed(plan)
        assert run_apply(acf, plan, *flags, "--base-ref", "HEAD") == 0
        err = capsys.readouterr().err
        marked = [line for line in err.splitlines()
                  if line.startswith("[SCOPED_CHECK_SETTLED] ")]
        assert len(marked) == 1, err
        assert "`pytest -q`" in marked[0] and "passed in full" in marked[0]
        assert "do not run --mark-completed" in marked[0]
        assert "do not run driver.verify_command" in marked[0]
        assert "[SCOPED_CHECK_OWED]" not in err
        assert capture.pytest_calls(paths) == ["pytest -q"]
        state = _plan_state(plan)
        assert state.is_phase_completed(PHASE) and scoped_checks.owed(state) is None

    def test_no_line_without_a_debt_or_after_a_failed_check(self, repo, capsys):
        cases = {"bad": ([_issue("Wrong", "Not a bug.")], "invalid", ("legacy",))}
        plan = self._review(repo, cases)
        paths = _orchestrator_config(repo)
        assert run_apply(acf, plan, "--yes", "--base-ref", "HEAD") == 0
        assert "[SCOPED_CHECK_SETTLED]" not in capsys.readouterr().err
        assert capture.pytest_calls(paths) == []
        _seed_plan_owed(plan)
        paths["set"](1)
        assert run_apply(acf, plan, "--yes", "--base-ref", "HEAD") == 3
        assert "[SCOPED_CHECK_SETTLED]" not in capsys.readouterr().err

    def _executor_run(self, repo, *flags):
        review = write_code_review(
            repo.root, cases={k: CODE_CASES[k] for k in ("in_scope", "legacy")},
            review_policy=None)
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps(
            [{"path": "src/main.py", "mode": "patch", "content": "y = 2\n"}]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(list(review["ids"].values())))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(MAIN_BATCH_CHECK))
        return review["plan"]

    def test_the_executor_leg_end_to_end_on_the_real_state(self, repo, capsys):
        """The orchestrator's own run under ``--exec-subagents``: narrowed
        prompts, the debt in the real ``state.json``, one full run, then the
        completion with the record gone in the same save."""
        plan = self._executor_run(repo)
        paths = _orchestrator_config(repo)
        assert run_apply(acf, plan, "--yes", "--exec-subagents", "--base-ref", "HEAD") == 0
        phase_dir = plan.parent / plan.stem / "apply-fixes"
        prompts = sorted((phase_dir / "prompts").glob("*.prompt.txt"))
        assert prompts and all(SCOPED_SENTENCE in p.read_text(encoding="utf-8") for p in prompts)
        assert capture.pytest_calls(paths) == ["pytest -q"]
        state = _plan_state(plan)
        assert state.is_phase_completed(PHASE) and scoped_checks.owed(state) is None
        on_disk = json.loads((plan.parent / plan.stem / "state.json").read_text(encoding="utf-8"))
        reports = on_disk[known_checks.APPLY_FIX_BATCH_CHECKS_KEY]
        assert reports and all(r.get("scoped") == {PYTEST: "passed"} for r in reports.values())
        assert _listed_mentions(on_disk) == []
        summary = (phase_dir / "summary.md").read_text(encoding="utf-8")
        assert "scoped to the batch's tests deferred to" in summary

    def test_the_executor_leg_fails_at_exit_3_and_resume_completes(self, repo, capsys):
        plan = self._executor_run(repo)
        paths = _orchestrator_config(repo)
        paths["set"](1)
        assert run_apply(acf, plan, "--yes", "--exec-subagents", "--base-ref", "HEAD") == 3
        state = _plan_state(plan)
        assert not state.is_phase_completed(PHASE) and scoped_checks.owed(state) is not None
        paths["set"](0)
        assert run_apply(acf, plan, "--yes", "--resume", "--exec-subagents",
                         "--base-ref", "HEAD") == 0
        state = _plan_state(plan)
        assert state.is_phase_completed(PHASE) and scoped_checks.owed(state) is None
        assert capture.pytest_calls(paths) == ["pytest -q", "pytest -q"]

    def test_a_completion_no_gate_settled_is_exit_3_not_a_crash(self, repo, capsys, monkeypatch):
        """The state writer's own refusal, reached past a gate (a dry run,
        or a writer added later without one), is exit 3 on the orchestrator's
        own entry point too."""
        plan = self._review(repo, {"in_scope": CODE_CASES["in_scope"]})
        paths = _orchestrator_config(repo)
        _seed_plan_owed(plan)
        monkeypatch.setattr(acf.ApplyCodeFixesOrchestrator, "_settle_scoped_check",
                            lambda self, state: None)
        assert run_apply(acf, plan, "--mark-completed") == 3
        assert "apply-code-fixes is NOT marked completed" in capsys.readouterr().err
        assert capture.pytest_calls(paths) == []
        assert not _plan_state(plan).is_phase_completed(PHASE)

    def test_a_native_resume_over_a_debt_tells_the_host_not_to_verify(self, repo, capsys):
        """The executor left the full run owed; the native emit hands out
        batches and says, as ``[SCOPED_CHECK_OWED]``, that ``--mark-completed``
        runs it: the host's own step 7 runs nothing."""
        plan = self._review(repo, {"in_scope": CODE_CASES["in_scope"]})
        paths = _orchestrator_config(repo)
        _seed_plan_owed(plan)
        assert run_apply(acf, plan, "--yes", "--base-ref", "HEAD") == 0
        err = capsys.readouterr().err
        marked = [line for line in err.splitlines() if line.startswith("[SCOPED_CHECK_OWED] ")]
        assert len(marked) == 1, err
        assert scoped_checks.OWED_LEAD in marked[0]
        assert "Do not run driver.verify_command" in marked[0] and "--mark-completed runs" in marked[0]
        assert capture.pytest_calls(paths) == []
        # And --mark-completed is the one full run.
        assert run_apply(acf, plan, "--mark-completed") == 0
        assert capture.pytest_calls(paths) == ["pytest -q"]

    def test_a_handoff_of_only_human_decisions_also_tells_the_host_not_to_verify(
        self, repo, capsys,
    ):
        """No batch, one item left to the host's decision: the phase still
        closes through the host's ``--mark-completed``, so the emit says the
        check is owed there rather than letting step 7 run it first."""
        issues, _status, scope = CODE_CASES["in_scope"]
        plan = self._review(repo, {"in_scope": (issues, "needs-human-decision", scope)})
        paths = _orchestrator_config(repo)
        _seed_plan_owed(plan)
        run_apply(acf, plan, "--yes", "--base-ref", "HEAD")
        err = capsys.readouterr().err
        assert "Needs human review: 1" in err, err
        marked = [line for line in err.splitlines() if line.startswith("[SCOPED_CHECK_OWED] ")]
        assert len(marked) == 1, err
        assert "[SCOPED_CHECK_SETTLED]" not in err
        assert capture.pytest_calls(paths) == []

    def test_no_debt_no_marker(self, repo, capsys):
        plan = self._review(repo, {"in_scope": CODE_CASES["in_scope"]})
        _orchestrator_config(repo)
        assert run_apply(acf, plan, "--yes", "--base-ref", "HEAD") == 0
        assert "[SCOPED_CHECK_OWED]" not in capsys.readouterr().err

    def test_the_native_emit_says_scoped_does_not_apply(self, repo, capsys):
        plan = self._review(repo, {"in_scope": CODE_CASES["in_scope"]})
        _orchestrator_config(repo)
        assert run_apply(acf, plan, "--yes", "--base-ref", "HEAD") == 0
        err = capsys.readouterr().err
        assert err.count("driver.batch_checks") == 1
        assert "applies under --exec-subagents only" in err


class TestEveryCompletionWriterSettlesFirst:
    """Structural: each place that records apply-code-fixes completed settles
    the owed full check before it, and the state writer refuses otherwise."""

    SITES = {
        "utils/apply_orchestrator_base.py": "_settle_scoped_check",
        "apply_code_fixes_orchestrator.py": "_settle_scoped_check",
        "utils/write_batch_scheduler.py": "_verify.run",
    }

    def test_each_mark_phase_completed_follows_the_gate(self):
        root = Path(__file__).resolve().parent.parent
        found = 0
        for name, gate in self.SITES.items():
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            for function in ast.walk(tree):
                if not isinstance(function, ast.FunctionDef):
                    continue
                source = ast.get_source_segment((root / name).read_text(encoding="utf-8"),
                                                function) or ""
                if ".mark_phase_completed(" not in source:
                    continue
                found += 1
                assert gate in source and source.index(gate) < source.index(
                    ".mark_phase_completed("), f"{name}:{function.name} completes ungated"
        assert found == 5

    def test_the_state_writer_refuses_while_the_debt_stands(self):
        state = {OWED: {"command_sha256": "x" * 64}}
        with pytest.raises(NeedsHuman):
            mark_phase_completed_in_state(state, PHASE)
        mark_phase_completed_in_state(state, "implement")
        assert "implement" in state["phases_completed"]

    def test_the_record_survives_a_stale_writer_and_its_discharge_does_too(self, tmp_path):
        plan = tmp_path / "p.md"
        plan.write_text("# P\n", encoding="utf-8")
        stale = StateManager(plan)
        writer = StateManager(plan)
        _seed_owed(writer)
        stale.set("unrelated", 1)
        stale.save()
        assert scoped_checks.owed(StateManager(plan)) is not None
        later_stale = StateManager(plan)
        settler = StateManager(plan)
        scoped_checks.discharge(settler, scoped_checks.owed(settler))
        settler.mark_phase_completed(PHASE)
        settler.save()
        later_stale.set("unrelated", 2)
        later_stale.save()
        final = StateManager(plan)
        assert scoped_checks.owed(final) is None and final.is_phase_completed(PHASE)


class TestTwoWritersInterleaved:
    """Two ``StateManager``s over one plan, the second saving between the
    first's load and its save: the state writer's merge, under the state
    file's lock, never leaves a completion beside a debt, and a settlement
    drops only the exact debt it checked (its token)."""

    @staticmethod
    def _plan(tmp_path):
        plan = tmp_path / "p.md"
        plan.write_text("# P\n", encoding="utf-8")
        return plan

    def test_a_completion_over_a_debt_saved_since_its_load_is_refused(self, tmp_path):
        plan = self._plan(tmp_path)
        completer = StateManager(plan)  # loads: nothing owed
        _seed_owed(StateManager(plan))  # another run's batch defers the full run
        assert scoped_checks.owed(completer) is None
        completer.set("unrelated", 1)
        completer.mark_phase_completed(PHASE)
        with pytest.raises(ScopedCheckOwedError):
            completer.save()
        final = StateManager(plan)
        assert scoped_checks.owed(final) is not None
        assert PHASE not in (final.state.get("phases_completed") or {})
        assert not final.is_phase_completed(PHASE)
        # The rest of its save stands, and it holds the file's state now.
        assert final.state["unrelated"] == 1
        assert scoped_checks.owed(completer) is not None
        completer.set("unrelated", 2)
        completer.save()

    def test_a_stale_settlement_keeps_a_newer_batch_s_debt(self, tmp_path):
        plan = self._plan(tmp_path)
        _seed_owed(StateManager(plan))
        settler = StateManager(plan)
        checked = scoped_checks.owed(settler)
        newer = StateManager(plan)
        _seed_owed(newer, batches=("apply-fix-batch-1",))
        scoped_checks.discharge(settler, checked)
        settler.mark_phase_completed(PHASE)
        with pytest.raises(ScopedCheckOwedError):
            settler.save()
        final = StateManager(plan)
        record = scoped_checks.owed(final)
        assert record is not None and record["token"] != checked["token"]
        assert record["batches"] == ["apply-fix-batch-0", "apply-fix-batch-1"]
        assert not final.is_phase_completed(PHASE)

    def test_a_batch_dispatched_again_since_the_check_keeps_the_debt(self, tmp_path):
        """The same batch id recorded again (a re-dispatch) is a newer debt
        too: the batch list alone would not tell it apart."""
        plan = self._plan(tmp_path)
        _seed_owed(StateManager(plan))
        settler = StateManager(plan)
        checked = scoped_checks.owed(settler)
        _seed_owed(StateManager(plan))
        scoped_checks.discharge(settler, checked)
        settler.mark_phase_completed(PHASE)
        with pytest.raises(ScopedCheckOwedError):
            settler.save()
        assert scoped_checks.owed(StateManager(plan))["batches"] == ["apply-fix-batch-0"]

    def test_a_settlement_of_the_debt_on_file_completes(self, tmp_path):
        plan = self._plan(tmp_path)
        _seed_owed(StateManager(plan))
        settler = StateManager(plan)
        bystander = StateManager(plan)
        scoped_checks.discharge(settler, scoped_checks.owed(settler))
        settler.mark_phase_completed(PHASE)
        bystander.set("unrelated", 1)
        bystander.save()
        settler.save()
        final = StateManager(plan)
        assert scoped_checks.owed(final) is None and final.is_phase_completed(PHASE)

    def test_two_writers_recording_batches_keep_both(self, tmp_path):
        plan = self._plan(tmp_path)
        first, second = StateManager(plan), StateManager(plan)
        _seed_owed(first)
        _seed_owed(second, batches=("apply-fix-batch-1",))
        assert scoped_checks.owed(StateManager(plan))["batches"] == [
            "apply-fix-batch-0", "apply-fix-batch-1"]

    def test_the_executor_finalize_refuses_a_debt_recorded_during_its_full_run(
        self, env, monkeypatch, pytest_cli,
    ):
        """The scheduler's own full run passes while another run records a
        batch: the finalize stops at exit 3, nothing completed, the newer
        batch's debt kept and named on the stop."""
        from utils import write_verify

        _scoped(env)
        state = StateManager(env.plan)
        real = write_verify.VerifyCommand.run

        def run_then_race(self, command, timeout):
            outcome = real(self, command, timeout)
            _seed_owed(StateManager(env.plan), batches=("apply-fix-batch-9",))
            return outcome

        monkeypatch.setattr(write_verify.VerifyCommand, "run", run_then_race)
        result = _two_batches(env, monkeypatch, state)
        assert result.exit_code == 3 and not result.finalized, result.messages
        assert result.unverified.startswith(scoped_checks.OWED_LEAD)
        assert any("NOT marked completed" in line for line in result.messages)
        final = StateManager(env.plan)
        assert scoped_checks.owed(final)["batches"] == [
            "apply-fix-batch-0", "apply-fix-batch-1", "apply-fix-batch-9"]
        assert not final.is_phase_completed(PHASE)

    def test_mark_completed_refuses_a_debt_recorded_after_it_settled(
        self, repo, capsys, monkeypatch,
    ):
        """``--mark-completed`` loads the state, finds nothing owed, and
        another run's batch records a debt before it saves: exit 3, no
        completion, the debt kept, nothing run."""
        plan = write_code_review(
            repo.root, cases={"in_scope": CODE_CASES["in_scope"]}, review_policy=None)["plan"]
        paths = _orchestrator_config(repo)
        real = acf.ApplyCodeFixesOrchestrator._settle_scoped_check

        def settle_then_race(self, state):
            real(self, state)
            _seed_owed(StateManager(plan))

        monkeypatch.setattr(acf.ApplyCodeFixesOrchestrator, "_settle_scoped_check",
                            settle_then_race)
        assert run_apply(acf, plan, "--mark-completed") == 3
        assert "NOT marked completed" in capsys.readouterr().err
        state = _plan_state(plan)
        assert not state.is_phase_completed(PHASE) and scoped_checks.owed(state) is not None
        assert capture.pytest_calls(paths) == []
        # The next --mark-completed settles it.
        monkeypatch.setattr(acf.ApplyCodeFixesOrchestrator, "_settle_scoped_check", real)
        assert run_apply(acf, plan, "--mark-completed") == 0
        state = _plan_state(plan)
        assert state.is_phase_completed(PHASE) and scoped_checks.owed(state) is None
        assert capture.pytest_calls(paths) == ["pytest -q"]


# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------

class TestComposites:

    def _owed_plan(self, tmp_path, *, skip=False):
        plan = tmp_path / "wf.md"
        plan.write_text("# Plan\n\n## Step 1\nbody\n", encoding="utf-8")
        state = StateManager(plan)
        state.mark_phase_completed("code-review")
        if skip:
            state.mark_phase_skipped(PHASE, "Skipped: an earlier zero-finding review")
        state.mark_phase_completed(PHASE)
        _seed_owed(state)
        return plan

    def _workflow(self, plan):
        workflow = fwo.FullWorkflow(fwo.parse_args(["--plan-file", str(plan), "--yes"]))
        ran = []
        workflow.run_mode = lambda mode, extra: ran.append(mode) or 0  # type: ignore[assignment]
        workflow._verify = lambda *a, **k: ("ok", "stub", None)  # type: ignore[assignment]
        return workflow, ran

    def test_full_runs_the_apply_over_a_zero_finding_review(self, tmp_path):
        plan = self._owed_plan(tmp_path)
        review = plan.parent / plan.stem / "code-review"
        review.mkdir(parents=True)
        (review / "grouped.json").write_text(json.dumps({"groups": []}), encoding="utf-8")
        workflow, ran = self._workflow(plan)
        workflow.step_apply_code_fixes()
        assert ran == [PHASE]
        assert not StateManager(plan).is_phase_skipped(PHASE)

    def test_full_does_not_pass_over_a_stale_skip(self, tmp_path):
        plan = self._owed_plan(tmp_path, skip=True)
        workflow, _ran = self._workflow(plan)
        step = next(s for s in workflow.steps() if s.phase == PHASE)
        assert workflow.skip_reason(step) is None

    @pytest.mark.parametrize("mode", ["improve-code", "improve-commits"])
    def test_improve_runs_the_apply_while_the_debt_stands(self, mode):
        """``skip_reason`` is what improve-code, improve-commits and every
        ``--cascade`` stage (``improve_cascade._stage_done``) pass an apply
        over with."""
        args = argparse.Namespace(force=False, scope_policy=None)
        apply = iw._row(mode).sequence[1]
        done = iw.Records(mode=mode, review_completed=True, apply_word="completed")
        assert iw.skip_reason(mode, apply, args, done) == "completed"
        for word in ("completed", "skipped"):
            owed = iw.Records(mode=mode, review_completed=True, apply_word=word,
                              scoped_check_owed=True)
            assert iw.skip_reason(mode, apply, args, owed) is None

    def test_a_cascade_stage_is_not_done_while_the_debt_stands(self, tmp_path):
        """``improve_cascade._stage_done`` over a completion newer than the
        review, written past the gate (raw: ``StateManager.save`` never
        writes one beside the debt): the stage runs its apply again."""
        from utils import improve_cascade

        plan = self._owed_plan(tmp_path)
        state_file = Path(StateManager(plan).state_file)
        document = json.loads(state_file.read_text(encoding="utf-8"))
        document.setdefault("phases_completed", {})[PHASE] = datetime.now().isoformat()
        state_file.write_text(json.dumps(document), encoding="utf-8")
        args = argparse.Namespace(force=False, scope_policy=None, files=None, files_from=None)
        assert improve_cascade._stage_done(plan, "improve-code", args) is False
        state = StateManager(plan)
        scoped_checks.discharge(state, scoped_checks.owed(state))
        state.save()
        assert improve_cascade._stage_done(plan, "improve-code", args) is True

    def test_read_records_sees_the_debt(self, tmp_path):
        plan = self._owed_plan(tmp_path)
        args = argparse.Namespace(force=False, scope_policy=None, files=None, files_from=None)
        assert iw.read_records("improve-code", plan, args).scoped_check_owed is True


class TestWorkflowStatusShowsTheDebt:
    """The shared status reader: the debt beats a recorded skip, the skip
    inferred from a newer review with no findings, and a completion, so
    ``status`` and the next step offer the apply's ``--resume``."""

    def _owed_after_a_clean_review(self, repo, *, recorded_skip):
        plan = write_code_review(repo.root, cases={}, review_policy=None)["plan"]
        state = StateManager(plan)
        if recorded_skip:
            state.mark_phase_skipped(PHASE, "Skipped: code review produced zero actionable findings")
        _seed_owed(state)
        return plan

    @pytest.mark.parametrize("recorded_skip", [False, True], ids=["inferred", "recorded"])
    def test_the_apply_is_in_progress_and_its_resume_is_next(self, repo, capsys, recorded_skip):
        import check_workflow_prerequisites as prereq
        from utils import next_phase as np_

        plan = self._owed_after_a_clean_review(repo, recorded_skip=recorded_skip)
        paths = _orchestrator_config(repo)
        status = prereq.get_workflow_status(
            StateManager(plan, read_only=True), plan, plan.parent / plan.stem)
        entry = status["phases"][PHASE]
        assert entry["status"] == "in_progress", entry
        assert entry["scoped_check_owed"].startswith(scoped_checks.OWED_LEAD)
        assert status["suggested_next"] == PHASE
        assert entry["scoped_check_owed"] in status["notices"]
        answer = np_.next_phase_commands(
            "plans/feat.md", "claude-code", None, "code-review", plan_path=plan)
        assert answer.phase == PHASE and "--resume" in answer.command, answer.command
        # The command it offers settles the debt.
        assert run_apply(acf, plan, "--resume", "--yes", "--base-ref", "HEAD") == 0
        state = _plan_state(plan)
        assert state.is_phase_completed(PHASE) and scoped_checks.owed(state) is None
        assert capture.pytest_calls(paths) == ["pytest -q"]
        after = prereq.get_workflow_status(
            StateManager(plan, read_only=True), plan, plan.parent / plan.stem)
        assert after["phases"][PHASE]["status"] == "completed"

    def test_an_earlier_review_s_debt_over_a_recorded_skip_is_a_fresh_apply(self, repo):
        """A skip recorded for an earlier clean review stays on file, a later
        review left an owed check, and a newer review has findings: the apply
        reopens as ``pending`` with the bare command. ``in_progress`` made
        ``next-command`` offer ``--resume``, which replayed the earlier
        review's decisions against the new findings."""
        import check_workflow_prerequisites as prereq
        from utils import next_phase as np_

        plan = write_code_review(repo.root, cases={"in_scope": CODE_CASES["in_scope"]},
                                 review_policy=None)["plan"]
        state = StateManager(plan)
        state.mark_phase_skipped(PHASE, "Skipped: code review produced zero actionable findings")
        _seed_owed_of_an_earlier_review(state)
        reader = StateManager(plan, read_only=True)
        entry = prereq.get_workflow_status(reader, plan, plan.parent / plan.stem)["phases"][PHASE]
        assert entry["status"] == "pending", entry
        assert entry["scoped_check_owed"].startswith(scoped_checks.OWED_LEAD)
        answer = np_.next_phase_commands(
            "plans/feat.md", "claude-code", None, "code-review", plan_path=plan)
        assert answer.phase == PHASE and "--resume" not in answer.command, answer.command
        after = np_.next_phase_commands(
            "plans/feat.md", "claude-code", None, None, plan_path=plan, after_mode="improve-code")
        assert not after.complete and "--resume" not in after.command, after

    def test_an_earlier_cycle_s_batches_under_a_newer_review_stay_a_fresh_apply(self, repo):
        """A newer review with findings, an earlier cycle's batches and the
        debt: the apply stays ``pending`` with the bare command (``--resume``
        would replay that cycle's decisions against the new findings), and
        the entry names the debt its run settles."""
        import check_workflow_prerequisites as prereq
        from utils import next_phase as np_

        plan = write_code_review(repo.root, cases={"in_scope": CODE_CASES["in_scope"]},
                                 review_policy=None)["plan"]
        state = StateManager(plan)
        state.update_task_status("apply-fix-batch-0", "completed")
        state.state.setdefault("metrics", {})[PHASE] = [
            {"label": "apply-fix-batch-0", "timestamp": "2020-01-01T00:00:00"}]
        _seed_owed_of_an_earlier_review(state)
        reader = StateManager(plan, read_only=True)
        assert not prereq.apply_phase_in_progress(reader, PHASE)
        entry = prereq.get_workflow_status(reader, plan, plan.parent / plan.stem)["phases"][PHASE]
        assert entry["status"] == "pending", entry
        assert entry["scoped_check_owed"].startswith(scoped_checks.OWED_LEAD)
        answer = np_.next_phase_commands(
            "plans/feat.md", "claude-code", None, "code-review", plan_path=plan)
        assert answer.phase == PHASE and "--resume" not in answer.command, answer.command

    def test_no_debt_leaves_the_inferred_skip(self, repo):
        import check_workflow_prerequisites as prereq

        plan = write_code_review(repo.root, cases={}, review_policy=None)["plan"]
        status = prereq.get_workflow_status(
            StateManager(plan, read_only=True), plan, plan.parent / plan.stem)
        assert status["phases"][PHASE] == {"status": "skipped", "reason": "skipped (no findings)"}
        assert not prereq.apply_phase_in_progress(StateManager(plan, read_only=True), PHASE)


class TestTheDebtKeepsItsReviewCycle:
    """Each admission copies the code review's completion stamp into the
    debt. A debt of the review on record is resumed (``--resume``) even
    before any batch is recorded; one of an earlier review is a fresh apply;
    and neither is walked past at a composite's last phase. Both through
    ``util next-command --after`` and the improve step's flag builder."""

    def test_each_admission_stamps_the_review_on_record(self):
        state = {"phases_completed": {"code-review": "2026-01-01T10:00:00"}}
        assert scoped_checks.record_owed(state, PYTEST, "apply-fix-batch-0")
        assert state[OWED]["review_completed_at"] == "2026-01-01T10:00:00"
        assert owed_scoped_check_is_current(state)
        state["phases_completed"]["code-review"] = "2026-01-02T10:00:00"
        assert not owed_scoped_check_is_current(state)
        assert scoped_checks.record_owed(state, PYTEST, "apply-fix-batch-1")
        assert state[OWED]["review_completed_at"] == "2026-01-02T10:00:00"
        assert state[OWED]["batches"] == ["apply-fix-batch-0", "apply-fix-batch-1"]
        assert owed_scoped_check_is_current(state)
        bare = {}
        assert scoped_checks.record_owed(bare, PYTEST, "apply-fix-batch-0")
        assert "review_completed_at" not in bare[OWED]

    @pytest.mark.parametrize("admitted,reviewed,current", [
        ("2026-01-01T10:00:00", "2026-01-01T10:00:00", True),
        ("2026-01-01T10:00:00.000000", "2026-01-01T10:00:00", True),
        ("2026-01-01T10:00:00", "2026-01-02T10:00:00", False),
        (None, "2026-01-01T10:00:00", False),
        ("2026-01-01T10:00:00", None, False),
        (None, None, False),
    ])
    def test_current_only_under_the_review_on_record(self, admitted, reviewed, current):
        state = {OWED: {"command_sha256": "x" * 64}, "phases_completed": {}}
        if admitted:
            state[OWED]["review_completed_at"] = admitted
        if reviewed:
            state["phases_completed"]["code-review"] = reviewed
        assert owed_scoped_check_is_current(state) is current
        assert not owed_scoped_check_is_current({"phases_completed": state["phases_completed"]})

    def _plan(self, repo, *, earlier):
        plan = write_code_review(repo.root, cases={"in_scope": CODE_CASES["in_scope"]},
                                 review_policy=None)["plan"]
        state = StateManager(plan)
        if earlier:
            # An earlier cycle's batch, dated before the review on record.
            state.update_task_status("apply-fix-batch-0", "completed")
            state.state.setdefault("metrics", {})[PHASE] = [
                {"label": "apply-fix-batch-0", "timestamp": EARLIER_REVIEW}]
            _seed_owed_of_an_earlier_review(state)
        else:
            # Admitted under this review; the batch never recorded an outcome.
            _seed_owed(state)
        return plan

    def _after(self, monkeypatch, capsys, plan, word):
        import modes
        from utils import cli_common
        from utils import next_phase as np_

        for name in ("MULTI_LLM_HOST", modes.ROUTE_ENV):
            monkeypatch.delenv(name, raising=False)
        recorded = []
        monkeypatch.setattr(cli_common, "record_user_next_step", recorded.append)
        assert np_.main([str(plan), "--after", word, "--host", "claude-code"]) == 0
        out = capsys.readouterr().out
        assert len(recorded) == 1, out
        return recorded[0], out

    def _apply_flags(self, plan):
        args = iw.parse_args("improve-code", ["--plan-file", str(plan)])
        records = iw.read_records("improve-code", plan, args)
        assert records.scoped_check_owed
        assert iw.skip_reason("improve-code", "review-code", args, records) == "completed"
        assert iw.skip_reason("improve-code", "apply-code-fixes", args, records) is None
        return iw.step_flags("improve-code", "apply-code-fixes", args, records)

    def test_an_earlier_review_s_debt_is_a_fresh_apply_never_complete(
        self, repo, monkeypatch, capsys,
    ):
        import check_workflow_prerequisites as prereq

        plan = self._plan(repo, earlier=True)
        reader = StateManager(plan, read_only=True)
        assert not prereq.apply_phase_in_progress(reader, PHASE)
        status = prereq.get_workflow_status(reader, plan, plan.parent / plan.stem)
        assert status["phases"][PHASE]["status"] == "pending"
        assert status["suggested_next"] == PHASE
        for word in ("improve-code", "apply-code-fixes"):
            step, out = self._after(monkeypatch, capsys, plan, word)
            assert step["complete"] is False and step["phase"] == PHASE, (word, step)
            assert "--apply-code-fixes" in step["command"], step
            assert "--resume" not in step["command"], step
            assert step["command"] in out
        assert "--resume" not in self._apply_flags(plan)

    def test_this_review_s_debt_resumes_before_any_batch_is_recorded(
        self, repo, monkeypatch, capsys,
    ):
        import check_workflow_prerequisites as prereq

        plan = self._plan(repo, earlier=False)
        reader = StateManager(plan, read_only=True)
        assert not reader.get_all_task_statuses()
        assert not (reader.state.get("metrics") or {}).get(PHASE)
        assert prereq.apply_phase_in_progress(reader, PHASE)
        status = prereq.get_workflow_status(reader, plan, plan.parent / plan.stem)
        assert status["phases"][PHASE]["status"] == "in_progress"
        for word in ("improve-code", "apply-code-fixes"):
            step, out = self._after(monkeypatch, capsys, plan, word)
            assert step["complete"] is False and step["phase"] == PHASE, (word, step)
            assert step["command"].endswith("--resume"), step
            assert step["command"] in out
        assert "--resume" in self._apply_flags(plan)


# ---------------------------------------------------------------------------
# The default: what the commit before the setting did, normalised (not bytes)
# ---------------------------------------------------------------------------

def _captured(env, monkeypatch, extra):
    state = FakeState(env.tmp_path / "state.json")
    captured = capture.capture_default_run(env, monkeypatch, state, extra_driver=extra)
    return json.loads(json.dumps(captured, sort_keys=True))


class TestDefaultUnchanged:

    @pytest.mark.skipif(os.name == "nt", reason=(
        "the golden holds POSIX paths and LF bytes (prompts, rollback-store "
        "hashes); the absent-equals-full test below runs everywhere"))
    @pytest.mark.parametrize("extra", [None, {"batch_checks": "full"}],
                             ids=["absent", "full"])
    def test_the_run_matches_the_golden_captured_before_the_setting(
        self, env, monkeypatch, extra,
    ):
        """Prompts, state.json, the batch report, the summary, the status
        mirror, the messages and the verify run, through ``FakeState`` and
        normalised the same way (``tests/support/scoped_checks_capture.py``):
        equal after normalisation, not byte for byte, and not the production
        ``StateManager`` path (the next tests take that one). After an intended
        change to what a default run leaves, write the golden again with
        ``MULTI_LLM_UPDATE_GOLDEN=1`` and ``-k "golden and absent"``."""
        captured = _captured(env, monkeypatch, extra)
        if extra is None and os.environ.get("MULTI_LLM_UPDATE_GOLDEN") == "1":
            GOLDEN.write_text(json.dumps(captured, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
        assert captured == json.loads(GOLDEN.read_text(encoding="utf-8"))

    def test_full_is_the_same_run_as_absent_on_the_real_state(
        self, env, monkeypatch, tmp_path_factory,
    ):
        """The production path, which the golden's ``FakeState`` does not
        take: a real ``StateManager`` (its merge, its journal, its own
        ``state.json``). Absent and ``full`` leave the same normalised
        ``state.json``, batch report, status mirror, summary, prompts,
        messages and verify run, and none of them a scoped key. Equality on
        this commit, normalised as the golden is: not byte identity, and not
        a comparison with the commit before the setting."""
        from tests.test_write_executor import WriteEnv

        def real_run(world, extra):
            # A new state reads the branch and HEAD at the cwd.
            monkeypatch.chdir(world.root)
            captured = capture.capture_default_run(
                world, monkeypatch, StateManager(world.plan), extra_driver=extra)
            return json.loads(json.dumps(captured, sort_keys=True))

        absent = real_run(env, None)
        second = WriteEnv(tmp_path_factory.mktemp("full-real"), monkeypatch)
        try:
            full = real_run(second, {"batch_checks": "full"})
        finally:
            registry._config = None
        assert absent["exit_code"] == 0 and absent["finalized"], absent["messages"]
        assert full == absent
        assert OWED not in absent["state"]
        assert absent["state"]["phases_completed"].get(PHASE)
        for record in absent["state"].get(known_checks.APPLY_FIX_BATCH_CHECKS_KEY, {}).values():
            assert "scoped" not in record
        assert not {"scoped_batches", "scoped_outcomes"} & set(absent["verification"] or {})

    def test_full_is_the_same_run_as_absent(self, env, monkeypatch, tmp_path_factory):
        """Everywhere, the golden's platforms included: two fresh worlds, one
        with the key absent and one with ``full``, leave the same run."""
        from tests.test_write_executor import WriteEnv

        absent = _captured(env, monkeypatch, None)
        second = WriteEnv(tmp_path_factory.mktemp("full"), monkeypatch)
        try:
            assert _captured(second, monkeypatch, {"batch_checks": "full"}) == absent
        finally:
            registry._config = None
