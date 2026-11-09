"""The machine-readable contracts have schemas, and the code keeps to them (LLM-6).

Three outputs are parsed by something other than a person: the
``MULTI_LLM_RESULT`` line every invocation ends with, ``{phase}/.status.json``,
and ``status --json``. Their schemas are ``schemas/result_line.schema.json``,
``schemas/status_file.schema.json`` and ``schemas/status_json.schema.json``
(docs/development/dispatcher.md, "Result schemas"). Two halves hold them to the
code:

* :class:`TestRealOutputsValidate` produces each output from real code paths
  (a dispatcher refusal, a crash, a missing plan, a range refusal, a detached
  launch, a live fan-out poll, partial, failed and refused runs, a finished
  review relaying its counts, ``doctor``) and validates what came out;
* :class:`TestEveryWrittenKeyIsDeclared` reads the emitters' source
  (``tests/contract_inventory.py``) and requires every key they can write to be
  declared, so a new field fails here even when no scenario reaches it;
  :class:`TestTheWalker` runs that source walker on made-up files, one dict
  form at a time.

Nothing validates at run time (the schemas describe; the tests enforce): a
schema check in the dispatcher could only turn a host-facing line into a
crash or a warning nobody reads, and ``.status.json`` outlives the version
that wrote it, so a strict check there would reject a newer plugin's file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
import pytest
from referencing import Registry, Resource

sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
from tests import contract_inventory as inventory  # noqa: E402
from utils import (  # noqa: E402
    cli_common, detach, output_handler, progress_tracker, status_file,
)
from utils.schema_validator import SCHEMA_DIR  # noqa: E402

RESULT_LINE = "result_line.schema.json"
STATUS_FILE = "status_file.schema.json"
STATUS_JSON = "status_json.schema.json"
CONTRACT_SCHEMAS = (RESULT_LINE, STATUS_FILE, STATUS_JSON)

DOCS = Path(__file__).resolve().parents[3] / "docs" / "development" / "dispatcher.md"


def _schema(name: str) -> Dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


_REGISTRY = Registry().with_resources(
    [(name, Resource.from_contents(_schema(name))) for name in CONTRACT_SCHEMAS]
)


def problems(name: str, document: Any, definition: Optional[str] = None) -> List[str]:
    """Every way *document* breaks schema *name* (or one of its definitions)."""
    schema: Dict[str, Any] = (
        {"$ref": f"{name}#/definitions/{definition}"} if definition else _schema(name)
    )
    validator = jsonschema.Draft7Validator(schema, registry=_REGISTRY)
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(document), key=lambda e: list(e.path))
    ]


def assert_valid(name: str, document: Any, definition: Optional[str] = None) -> None:
    found = problems(name, document, definition)
    target = f"{name}#/definitions/{definition}" if definition else name
    assert not found, (
        f"output does not match {target}:\n  " + "\n  ".join(found)
        + f"\n\ndocument:\n{json.dumps(document, indent=2, default=str)[:4000]}"
    )


# ---------------------------------------------------------------------------
# Producing real outputs
# ---------------------------------------------------------------------------

def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\n## Tasks\n- [ ] 1. Do the thing\n", encoding="utf-8")
    return plan


def _phase_dir(plan: Path, phase: str) -> Path:
    directory = output_handler.phase_dir_path(plan, phase)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _main(argv: List[str], capsys) -> Tuple[int, Dict[str, Any], str]:
    """Run the dispatcher in process: (exit, result payload, stdout before the line)."""
    capsys.readouterr()  # whatever a fixture printed first (`[DETACHED_RUN]`)
    code = multi_llm.main(argv)
    out = capsys.readouterr().out
    payload = cli_common.parse_result_line(out)
    assert payload is not None, f"no {cli_common.RESULT_MARKER} line in:\n{out}"
    lines = out.rstrip("\n").splitlines()
    assert lines[-1].startswith(cli_common.RESULT_MARKER), "the result line is not last"
    assert payload["exit"] == code
    return code, payload, out.rsplit(cli_common.RESULT_MARKER, 1)[0]


def _status_json(plan: Path, capsys, *extra: str) -> Tuple[int, Dict[str, Any], Any]:
    code, payload, before = _main(["status", str(plan), "--json", *extra], capsys)
    return code, payload, json.loads(before)


def _assert_checkpoint_valid(directory: Path) -> Dict[str, Any]:
    document = json.loads((directory / status_file.STATUS_FILENAME).read_text("utf-8"))
    assert_valid(STATUS_FILE, document)
    return document


@pytest.fixture
def handle_factory():
    """``RunHandle``s started in process (``detached=False``), finished at teardown."""
    started: List[detach.RunHandle] = []

    def start(plan: Path, phase_dir: Path, mode: str, run_id: str) -> detach.RunHandle:
        handle = detach.RunHandle(
            str(phase_dir), run_id=run_id, mode=mode, plan_file=str(plan),
            detached=False,
        )
        handle.start()
        started.append(handle)
        return handle

    yield start
    for handle in started:
        handle.finish(0)


def _fan_out(phase_dir: Path, outcomes: Dict[str, Optional[str]]) -> None:
    """Seed a fan-out over the specs in *outcomes* and move each slot: ``None``
    leaves it running, ``"done"`` finishes it, any other text fails it with
    that error."""
    tracker = progress_tracker.FanoutProgress(
        phase_dir, list(outcomes), interval_s=0, timeout=600,
    )
    tracker.seed()
    for spec, outcome in outcomes.items():
        tracker.running(spec, attempt=1)
        if outcome == "done":
            tracker.done(spec, source="file")
        elif outcome is not None:
            tracker.failed(spec, error=outcome)


class TestRealOutputsValidate:
    def test_an_unknown_mode_is_refused_on_a_valid_line(self, capsys):
        code, payload, _ = _main(["no-such-mode"], capsys)
        assert code == cli_common.EXIT_NEEDS_HUMAN
        assert payload["state"] == "needs-human"
        assert_valid(RESULT_LINE, payload)

    def test_a_crash_still_ends_with_a_valid_line(self, tmp_path, monkeypatch, capsys):
        plan = _plan(tmp_path)

        def boom(argv):
            raise RuntimeError("kaboom in the phase")

        monkeypatch.setattr(multi_llm, "_load_target", lambda mode, argv: boom)
        code, payload, _ = _main(["review-plan", str(plan)], capsys)
        assert code == cli_common.EXIT_ERROR
        assert payload["state"] == "error"
        assert_valid(RESULT_LINE, payload)

    def test_a_missing_plan(self, tmp_path, capsys):
        code, payload, document = _status_json(tmp_path / "missing.md", capsys)
        assert code == cli_common.EXIT_ERROR
        assert_valid(RESULT_LINE, payload)
        assert_valid(STATUS_JSON, document, "error")

    def test_a_refused_range_spelling(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        code, payload, document = _status_json(Path("HEAD~2..HEAD"), capsys)
        assert code == cli_common.EXIT_NEEDS_HUMAN
        assert payload.get("error")
        assert_valid(RESULT_LINE, payload)
        assert_valid(STATUS_JSON, document, "error")

    def test_a_plan_with_no_run(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        code, payload, document = _status_json(plan, capsys)
        assert code == cli_common.EXIT_NOTHING_TO_DO
        assert payload["state"] == "nothing-to-do"
        assert_valid(RESULT_LINE, payload)
        assert_valid(STATUS_JSON, document, "runs_document")
        assert document["runs"] == [] and document["workflow"] is not None

        code, payload, document = _status_json(plan, capsys, "--view", "workflow")
        assert code == cli_common.EXIT_OK
        assert_valid(RESULT_LINE, payload)
        assert_valid(STATUS_JSON, document, "workflow")

    def test_a_detached_launch(self, tmp_path, monkeypatch, capsys):
        """The launch's own line: the poll command as ``next``, the roster."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-plan")
        uv = tmp_path / "fake-uv"
        uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        uv.chmod(0o755)
        monkeypatch.setenv(detach.UV_ENV, str(uv))
        monkeypatch.setattr(detach, "missing_provider_binaries", lambda *a, **k: {})

        def fake_spawn(command, *, cwd, env, log_path):
            run_id = command[command.index("--internal-run-id") + 1]
            status_file.claim_run(str(phase), {
                "run_id": run_id, "state": "running", "pid": os.getpid(),
                "heartbeat_at": status_file.now_iso(),
            })
            return os.getpid()

        monkeypatch.setattr(detach.process_control, "spawn_detached", fake_spawn)
        context = cli_common.activate_result(cli_common.ResultContext(mode="review-plan"))
        try:
            code = detach.launch(
                mode="review-plan",
                argv=["--plan-file", str(plan), "--detach", "--yes"],
                phase_dir=str(phase), plan_file=str(plan), phase="review-plan",
                args=argparse.Namespace(force=False, if_not_running=False, timeout=600),
                model_specs=["codex:gpt-6-luna:high", "opencode:auto"],
            )
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert code == cli_common.EXIT_OK
        payload = context.payload(code)
        assert payload["next"] and payload["progress_lines"]
        assert_valid(RESULT_LINE, payload)
        _assert_checkpoint_valid(phase)

    def test_a_live_fan_out_poll(self, tmp_path, capsys, handle_factory):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-plan")
        handle_factory(plan, phase, "review-plan", "run-contract-live")
        _fan_out(phase, {"codex:gpt-6-luna:high": None, "opencode:auto": "done"})
        _assert_checkpoint_valid(phase)

        code, payload, document = _status_json(plan, capsys, "--phase", "review-plan")
        assert code == cli_common.EXIT_NOTHING_TO_DO
        assert payload["state"] == "running"
        assert payload["progress"]["models"] and payload["progress_lines"]
        assert_valid(RESULT_LINE, payload)
        assert_valid(STATUS_JSON, document, "runs_document")
        assert document["runs"][0]["verdict"] == "running"

    def test_a_partial_run(self, tmp_path, capsys, handle_factory):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-plan")
        handle = handle_factory(plan, phase, "review-plan", "run-contract-partial")
        _fan_out(phase, {"codex:gpt-6-luna:high": "done", "opencode:auto": "timed out"})
        handle.finish(0, result_state=cli_common.STATE_PARTIAL)
        _assert_checkpoint_valid(phase)

        code, payload, _ = _main(["status", str(plan), "--phase", "review-plan"], capsys)
        assert code == cli_common.EXIT_OK
        assert payload["state"] == cli_common.STATE_PARTIAL
        assert_valid(RESULT_LINE, payload)
        _, _, document = _status_json(plan, capsys, "--phase", "review-plan")
        assert_valid(STATUS_JSON, document, "runs_document")

    def test_a_failed_run(self, tmp_path, capsys, handle_factory):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-plan")
        handle = handle_factory(plan, phase, "review-plan", "run-contract-failed")
        handle.finish(1, error="the reviewer panel broke")
        _assert_checkpoint_valid(phase)

        code, payload, document = _status_json(plan, capsys, "--phase", "review-plan")
        assert code == cli_common.EXIT_NEEDS_HUMAN
        assert payload.get("error")
        assert_valid(RESULT_LINE, payload)
        assert_valid(STATUS_JSON, document, "runs_document")

    def test_a_refused_runner(self, tmp_path, capsys, handle_factory):
        """A pre-dispatch refusal's close-out keys, on disk and in the view."""
        from utils.review_orchestrator_base import _print_executor_refusal
        from utils.subagent_executor import RunnerRejected

        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-plan")
        handle = handle_factory(plan, phase, "review-plan", "run-contract-refused")
        _print_executor_refusal(
            RunnerRejected("no such runner", key="driver.subagent_runner",
                           spec="nope:model", reason_code="unknown_provider"),
            phase_dir=str(phase), retry_command=None,
        )
        handle.finish(3, error="refused before dispatch")
        document = _assert_checkpoint_valid(phase)
        assert document["state"] == "runner_rejected"

        code, payload, view = _status_json(plan, capsys, "--phase", "review-plan")
        assert code == cli_common.EXIT_NEEDS_HUMAN
        assert_valid(RESULT_LINE, payload)
        assert_valid(STATUS_JSON, view, "runs_document")
        assert view["runs"][0]["reason_code"] == "unknown_provider"

    def test_a_finished_review_relays_its_counts(self, tmp_path, capsys, handle_factory):
        """The review writers' own objects, on disk, in the view and on the line."""
        from utils import review_orchestrator_base as base
        from utils.validation import publish_validation_context

        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "code-review")
        handle = handle_factory(plan, phase, "review-code", "run-contract-review")
        groups = [
            {"issues": [{"importance": "HIGH"}], "validation_status": "valid"},
            {"issues": [{"importance": "LOW"}], "validation_status": "needs-human-decision"},
        ]
        status_file.update_status(str(phase), {"phase": "code-review", "state": "complete"})
        assert base.record_high_priority_groups(str(phase), groups) == 1
        assert base.record_validation_counts(str(phase), groups)
        assert publish_validation_context(str(phase), [{
            "label": "plan", "mode": "truncated", "chars": 100, "bytes": 5000,
            "path": str(plan), "sha256": "0" * 64, "dropped_sections": ["Later"],
            "reason": "over the limit",
        }], phase="code-review")
        handle.finish(0)
        document = _assert_checkpoint_valid(phase)
        assert {"high_priority_groups", "validation_counts", "validation_context"} <= set(document)

        code, payload, _ = _main(["status", str(plan), "--phase", "code-review"], capsys)
        assert code == cli_common.EXIT_OK
        assert payload["high_priority_groups"] == {"count": 1}
        assert payload["validation_counts"]["needs_human_decision"] == 1
        assert payload["validation_context"]["mode"] == "truncated"
        assert_valid(RESULT_LINE, payload)
        _, _, view = _status_json(plan, capsys, "--phase", "code-review")
        assert_valid(STATUS_JSON, view, "runs_document")

    @pytest.mark.parametrize("failures, code, state", [
        ([], cli_common.EXIT_OK, "done"),
        (["codex: sandbox"], cli_common.EXIT_NEEDS_HUMAN, "needs-human"),
    ], ids=["healthy", "a-failing-row"])
    def test_doctor(self, monkeypatch, capsys, failures, code, state):
        """``doctor``'s line is the dispatcher's own and in this schema's scope.
        Its ``--json`` report, printed before the line, is a document of its
        own that none of the three schemas covers; the probes are stubbed out
        here (``tests/test_doctor.py`` owns them)."""
        from utils import doctor

        monkeypatch.setattr(doctor, "collect", lambda **kwargs: {
            "host": "claude-code", "providers": [], "failures": list(failures),
        })
        exit_code, payload, before = _main(["doctor", "--json"], capsys)
        assert exit_code == code
        assert (payload["mode"], payload["phase"], payload["state"]) == ("doctor", None, state)
        assert_valid(RESULT_LINE, payload)
        assert json.loads(before)["failures"] == failures

    def test_every_optional_key_at_once(self, tmp_path):
        """One context carrying every optional key, each built by its real
        builder where one exists, validates — and between an exit-0 and an
        exit-3 payload every declared key is reached."""
        import ask_orchestrator
        from utils import review_orchestrator_base as base
        from utils.commit_apply import fix_payload
        from utils.commit_range import FixBlock
        from utils.next_phase import NextPhase
        from utils.validation_context import build_validation_context

        phase = tmp_path / "progress"
        phase.mkdir()
        _fan_out(phase, {"codex:gpt-6-luna:high": "done"})
        progress = status_file.read_status(str(phase))["progress"]
        context = cli_common.ResultContext(mode="review-code", phase="code-review")
        context.add_artifact(tmp_path / "report.md")
        context.set_consolidation_recommended(
            {"valid_groups": 30, "threshold": 25, "command": "--consolidate"})
        context.set_high_priority_groups({"count": 2})
        context.set_validation_counts(base.validation_counts(
            [{"validation_status": "valid"}, {"validation_status": "invalid"}]))
        context.set_validation_context(build_validation_context(
            phase="code-review", limit=1000, sources=[{
                "label": "plan", "mode": "path", "chars": 10, "path": "plan.md",
            }]))
        context.set_answer_origins(ask_orchestrator.answer_origins(str(tmp_path), [], []))
        context.set_notice("run --apply-task-suggestions, or skip to --implement")
        context.set_progress(progress)
        context.set_progress_lines(["[PROGRESS] fan-out over 1 model: codex"])
        context.set_user_next_step(NextPhase(phase=None, command=None, complete=True).to_dict())
        context.set_user_resume({
            "phase": "implement", "mode": "implement", "command": "multi-llm implement x",
            "flags": ["--resume"], "markdown": "## Resume",
        })
        context.set_commit_range({
            "brief": "brief.md", "base": "a" * 12, "head": "b" * 12, "commits": [],
            "more": 0, "full_list": None, "window": None, "notes": [], "drifted": [],
            "size": {"files": 1},
        })
        context.set_commit_fixes(fix_payload(FixBlock(
            to_apply=1, per_commit=[("c" * 12, 1, "subject")], unattributed=0,
            skipped_changed=0, changed_files=["a.py"], head="b" * 12,
        ), source="pinned"))
        context.set_error("the reviewer panel broke")
        context.ran_from_root = str(tmp_path)

        done = context.payload(cli_common.EXIT_OK)
        stopped = context.payload(cli_common.EXIT_NEEDS_HUMAN)
        assert_valid(RESULT_LINE, done)
        assert_valid(RESULT_LINE, stopped)
        declared = set(_schema(RESULT_LINE)["properties"])
        assert set(done) | set(stopped) == declared


# ---------------------------------------------------------------------------
# The inventory: every key the source can write is declared
# ---------------------------------------------------------------------------

#: Functions whose writes the inventory cannot follow, by (file, function),
#: each with the enforced check that covers the data instead. An entry is
#: allowed only when such a check exists; a new one is a decision, so prefer a
#: literal dict at the write. ``test_no_exemption_is_stale`` drops dead ones.
PASS_THROUGH: Dict[Tuple[str, str], str] = {
    ("utils/status_file.py", "update_status"):
        "merges into the document it read from disk (json.load), which earlier "
        "writers wrote, and re-stores the _MERGED_OBJECTS keys the caller's dict "
        "already carries; every update_status call is itself a writer scanned "
        "here, and test_the_status_file checks _MERGED_OBJECTS are declared",
    ("utils/status_file.py", "load_status"):
        "returns the document read from disk (json.load) for update_status to "
        "merge into; its keys are the earlier writers', each scanned here",
    ("utils/status_file.py", "write_status"):
        "copies the _RESERVED_OBJECTS (`run`, `progress`) over from disk; "
        "test_the_status_file checks they are declared",
    ("utils/status_file.py", "remove_status_keys"):
        "writes back the document it read, minus the keys it names: it adds none",
    ("utils/detach.py", "report_progress"):
        "forwards its `**updates`; progress_keys inventories the keywords of "
        "every report_progress call and flags a `**` spread at one",
    ("utils/detach.py", "progress"):
        "RunHandle.progress forwards its `**updates`; progress_keys inventories "
        "the keywords of every .progress() call and flags a `**` spread at one",
    ("utils/review_orchestrator_base.py", "base_round_steps"):
        "hands write_status, update_status and report_progress to the shared "
        "review lane as RoundSteps fields; every call through them is an "
        "attribute call (`steps.update_status(...)`, utils/review_lane.py), "
        "which callers() follows by attribute name, so its dict and keywords "
        "are inventoried here like any other writer's",
    ("code_review_orchestrator.py", "code_round_steps"):
        "review-code's RoundSteps, the same hand-over as base_round_steps: its "
        "calls are the shared lane's `steps.<writer>(...)` attribute calls",
}


def _declared(name: str, *path: str) -> set:
    node: Dict[str, Any] = _schema(name)
    for part in path:
        node = node[part]
    return set(node["properties"])


def _check(found: inventory.Inventory, declared: set, what: str,
           *, exact: bool = True) -> None:
    unknown = sorted(set(found.keys) - declared)
    assert not unknown, (
        f"{what}: written but not declared in its schema: "
        + ", ".join(f"{key} ({found.keys[key]})" for key in unknown)
        + ". Declare each key (type and meaning) in the schema, and say what it "
        "means in docs/development/dispatcher.md or the status_file docstring."
    )
    if exact:
        stale = sorted(declared - set(found.keys))
        assert not stale, (
            f"{what}: declared but no writer in the source writes it: {stale}. "
            "Remove it from the schema, or the inventory lost track of a writer."
        )
    unexpected = sorted(set(found.unresolved) - set(PASS_THROUGH))
    assert not unexpected, (
        f"{what}: writes the inventory cannot follow:\n  "
        + "\n  ".join(f"{site}: " + "; ".join(found.unresolved[site]) for site in unexpected)
        + "\nBuild the dict as a literal at the write, or add (file, function) to "
        "PASS_THROUGH, naming the enforced check that covers its keys."
    )


SHARED_OBJECTS = ("consolidation_recommended", "high_priority_groups", "validation_context",
                  "answer_origins", "step_results")

#: Every inventory the class below checks: (function in contract_inventory, args).
SCANS: List[Tuple[str, ...]] = [
    ("result_line_keys",), ("status_file_top_level",), ("run_keys",), ("progress_keys",),
    ("model_entry_keys",), *[("status_member_keys", m) for m in SHARED_OBJECTS],
    ("user_next_step_keys",), ("user_resume_keys",), ("commit_range_keys",),
    ("commit_fixes_keys",), ("status_json_top_level",), ("run_view_keys",), ("workflow_keys",),
]
_FOUND: Dict[Tuple[str, ...], inventory.Inventory] = {}


def _found(*scan: str) -> inventory.Inventory:
    """One inventory over the real source, walked once per test session."""
    if scan not in _FOUND:
        name, *args = scan
        _FOUND[scan] = getattr(inventory, name)(*args)
    return _FOUND[scan]


class TestEveryWrittenKeyIsDeclared:
    def test_the_result_line(self):
        _check(_found("result_line_keys"), _declared(RESULT_LINE), "MULTI_LLM_RESULT")

    def test_the_result_line_states(self):
        enum = set(_schema(RESULT_LINE)["properties"]["state"]["enum"])
        words = set(cli_common._STATE_BY_EXIT.values()) | {
            cli_common.STATE_RUNNING, cli_common.STATE_DIED, cli_common.STATE_PARTIAL,
        }
        assert enum == words

    def test_the_status_file(self):
        declared = _declared(STATUS_FILE)
        _check(_found("status_file_top_level"), declared, ".status.json")
        # The keys the status_file merge helpers write on their own (PASS_THROUGH).
        assert set(status_file._MERGED_OBJECTS) <= declared
        assert set(status_file._RESERVED_OBJECTS) <= declared

    def test_the_run_record(self):
        _check(_found("run_keys"), _declared(STATUS_FILE, "definitions", "run"),
               ".status.json run")
        states = _schema(STATUS_FILE)["definitions"]["run"]["properties"]["state"]["enum"]
        assert set(states) == {status_file.RUN_STATE_STARTING, status_file.RUN_STATE_RUNNING,
                               *status_file.TERMINAL_RUN_STATES}

    def test_the_progress_object(self):
        _check(_found("progress_keys"), _declared(STATUS_FILE, "definitions", "progress"),
               ".status.json progress")

    def test_a_fan_out_slot(self):
        _check(_found("model_entry_keys"),
               _declared(STATUS_FILE, "definitions", "model_entry"), "progress.models entry")
        states = _schema(STATUS_FILE)["definitions"]["model_entry"]["properties"]["state"]["enum"]
        assert set(states) == (
            set(progress_tracker.TERMINAL_STATES) | set(progress_tracker.WAITING_STATES)
            | {progress_tracker.STATE_RUNNING, progress_tracker.STATE_RETRYING}
        )

    @pytest.mark.parametrize("member", SHARED_OBJECTS)
    def test_the_shared_objects(self, member):
        _check(_found("status_member_keys", member),
               _declared(STATUS_FILE, "definitions", member), f".status.json {member}")

    def test_the_validation_counts(self):
        from utils.review_orchestrator_base import VALIDATION_COUNT_KEYS

        definition = _schema(STATUS_FILE)["definitions"]["validation_counts"]
        assert set(definition["properties"]) == set(VALIDATION_COUNT_KEYS.values())
        assert set(definition["required"]) == set(VALIDATION_COUNT_KEYS.values())

    @pytest.mark.parametrize("builder, definition", [
        ("user_next_step_keys", "user_next_step"),
        ("user_resume_keys", "user_resume"),
        ("commit_range_keys", "commit_range"),
        ("commit_fixes_keys", "commit_fixes"),
    ])
    def test_the_result_line_objects(self, builder, definition):
        _check(_found(builder), _declared(RESULT_LINE, "definitions", definition),
               f"MULTI_LLM_RESULT {definition}")

    def test_the_status_json_document(self):
        declared = (_declared(STATUS_JSON, "definitions", "runs_document")
                    | _declared(STATUS_JSON, "definitions", "error"))
        _check(_found("status_json_top_level"), declared, "status --json")

    def test_a_run_view(self):
        _check(_found("run_view_keys"), _declared(STATUS_JSON, "definitions", "run_view"),
               "status --json runs[]")
        verdicts = _schema(STATUS_JSON)["definitions"]["run_view"]["properties"]["verdict"]["enum"]
        assert set(verdicts) == {
            detach.VERDICT_NO_RUN, detach.VERDICT_STARTING, detach.VERDICT_RUNNING,
            detach.VERDICT_STALE, *detach.TERMINAL_VERDICTS,
        }

    def test_the_workflow_view(self):
        _check(_found("workflow_keys"), _declared(STATUS_JSON, "definitions", "workflow"),
               "status --view workflow")

    def test_no_exemption_is_stale(self):
        """An exemption no write needs any more would excuse the next one there."""
        reached = set()
        for scan in SCANS:
            reached |= set(_found(*scan).unresolved)
        stale = sorted(set(PASS_THROUGH) - reached)
        assert not stale, f"PASS_THROUGH entries no inventory reaches: {stale}; remove them"


# ---------------------------------------------------------------------------
# The walker itself, on made-up sources
# ---------------------------------------------------------------------------

#: The smallest ``utils/status_file.py`` and ``utils/detach.py`` the progress
#: inventory needs: the writers it looks for, forwarding the way the real ones do.
STUB_STATUS_FILE = """
def write_status(phase_dir, data):
    pass

def update_status(phase_dir, updates):
    write_status(phase_dir, updates)

def update_progress(phase_dir, updates):
    update_status(phase_dir, {"progress": dict(updates)})

def claim_run(phase_dir, record):
    update_status(phase_dir, {"run": dict(record)})
"""

STUB_DETACH = """
from utils import status_file

class RunHandle:
    def progress(self, **updates):
        status_file.update_progress(self.phase_dir, updates)

def report_progress(phase_dir, **updates):
    status_file.update_progress(phase_dir, updates)
"""

#: What the made-up ``ResultContext.payload`` bodies below may call on.
PAYLOAD_MODULE = """
NEW_KEY = "new_key"


def _base(context):
    result = {"mode": 1}
    result["new_key"] = 2
    return result


class _Filler:
    def fill(self, target):
        target["new_key"] = 1


class _OtherFiller:
    def fill(self, target):
        pass


class ResultContext:
    def payload(self, exit_code):
{body}
        return payload

    def _close(self, payload):
        payload["new_key"] = 1
"""


def _sources(**files: str) -> Dict[str, str]:
    """The two stubs plus made-up files, keyed ``utils_x`` for ``utils/x.py``."""
    sources = {"utils/status_file.py": STUB_STATUS_FILE, "utils/detach.py": STUB_DETACH}
    for name, text in files.items():
        sources[name.replace("_", "/", 1) + ".py"] = textwrap.dedent(text)
    return sources


def _payload_keys(body: str) -> inventory.Inventory:
    """``result_line_keys`` over a made-up ``ResultContext.payload`` doing *body*."""
    source = PAYLOAD_MODULE.replace("{body}", textwrap.indent(body, " " * 8))
    return inventory.result_line_keys(
        inventory._Corpus.from_sources({"utils/cli_common.py": source}))


def _progress_keys(writer: str) -> inventory.Inventory:
    """``progress_keys`` over the stubs and one made-up writer module."""
    head = "from utils.status_file import update_status\n"
    return inventory.progress_keys(inventory._Corpus.from_sources(
        _sources(utils_writer=head + textwrap.dedent(writer))))


#: A function that hands its ``progress`` parameter to a status writer.
CLOSE = """
def close(directory, *, progress=None):
    updates = {"state": "done"}
    if progress:
        updates["progress"] = progress
    update_status(directory, updates)
"""


class TestTheWalker:
    """Each dict form the inventory claims to follow, run on the walker itself."""

    @pytest.mark.parametrize("body", [
        'payload = {"mode": 1}\npayload.update({"new_key": True})',
        'payload = {"mode": 1}\nif self.mode == "doctor":\n    payload.update({"new_key": True})',
        'payload = {"mode": 1}\npayload.update(new_key=True)',
        'payload = {"mode": 1}\nextra = {"new_key": 1}\npayload.update(**extra)',
        'payload = {"mode": 1}\npayload.update(dict(new_key=1))',
        'payload = {"mode": 1}\npayload.setdefault("new_key", [])',
        'payload = {"mode": 1}\npayload |= {"new_key": 1}',
        'payload = {"mode": 1}\npayload = payload | {"new_key": 1}',
        'payload = dict(mode=1, **{"new_key": 2})',
        'payload = dict({"mode": 1}, new_key=2)',
        'payload = dict(dict(new_key=1))',
        'payload = {"mode": 1}\npayload[NEW_KEY] = 1',
        'payload = {"mode": 1}\npayload["new_key"], payload["mode"] = 1, 2',
        'payload = {"mode": 1}\nself._close(payload)',
        'payload = _base(self)',
    ], ids=[
        "update-positional", "update-under-a-condition", "update-keyword", "update-spread",
        "update-dict-call", "setdefault", "or-equals", "union", "dict-call-spread",
        "dict-call-positional", "dict-call-nested", "constant-key", "tuple-target",
        "handed-to-a-method", "helper-return",
    ])
    def test_a_key_written_in_place_is_found(self, body):
        found = _payload_keys(body)
        assert "new_key" in found.keys, found.keys
        assert not found.unresolved, found.unresolved

    @pytest.mark.parametrize("body, reason", [
        ('payload = {"mode": 1}\npayload.update(self.extra)', "self.extra"),
        ('payload = {"mode": 1}\npayload.update(json.loads(text))', "json.loads"),
        ('payload = {"mode": 1}\npayload[name] = 1', "cannot evaluate: name"),
        ('payload = {"mode": 1}\npayload.setdefault(name, 1)', "cannot evaluate: name"),
        ('payload = {"mode": 1}\npayload.update({name: 1})', "cannot evaluate: name"),
        ('payload = dict(zip(names, values))', "zip"),
        ('payload = {k: 1 for k in names}', "cannot follow"),
        ('payload, other = build()', "bound by a Assign"),
        ('payload = {}\nfor payload in rows:\n    pass', "bound by a For"),
        ('payload = {"mode": 1}\npayload.update({"a": 1}, {"b": 2})', "more than one"),
        ('payload = {"mode": 1}\nfiller.fill(payload)', "2 skill functions are named"),
    ], ids=[
        "update-attribute", "update-outside-call", "subscript-variable-key",
        "setdefault-variable-key", "update-variable-key", "dict-of-zip", "comprehension",
        "tuple-unpacking", "loop-variable", "update-two-positional", "handed-to-one-of-two",
    ])
    def test_a_form_it_cannot_follow_is_reported_not_skipped(self, body, reason):
        found = _payload_keys(body)
        site = ("utils/cli_common.py", "payload")
        assert site in found.unresolved, f"not reported: {body!r} -> {found.keys}"
        assert any(reason in why for why in found.unresolved[site]), found.unresolved[site]
        with pytest.raises(AssertionError, match="cannot follow"):
            _check(found, set(found.keys), "made-up line")

    @pytest.mark.parametrize("writer", [
        'def record(d):\n    update_status(d, dict(progress={"new_nested_field": 1}))',
        'def record(d):\n    update_status(d, dict(progress=dict(new_nested_field=1)))',
        'def record(d):\n    update_status(d, {"progress": {"new_nested_field": 1}})',
        'def record(d):\n    u = {"state": "x"}\n    u["progress"] = {"new_nested_field": 1}\n'
        '    update_status(d, u)',
        'def record(d):\n    u = {"progress": {}}\n    u["progress"]["new_nested_field"] = 1\n'
        '    update_status(d, u)',
        'def record(d):\n    u = {}\n    u.setdefault("progress", {})["new_nested_field"] = 1\n'
        '    update_status(d, u)',
        'def record(d):\n    u = {"progress": {}}\n    u["progress"].update(new_nested_field=1)\n'
        '    update_status(d, u)',
        'def record(d):\n    u = {}\n    u.update(progress={"new_nested_field": 1})\n'
        '    update_status(d, u)',
        'def record(d):\n    u = {}\n    u |= {"progress": {"new_nested_field": 1}}\n'
        '    update_status(d, u)',
        'def _build():\n    return {"progress": {"new_nested_field": 1}}\n\n'
        'def record(d):\n    update_status(d, _build())',
    ], ids=[
        "dict-call", "nested-dict-call", "literal", "subscript", "nested-subscript",
        "setdefault-chain", "member-update", "update-keyword", "or-equals", "helper",
    ])
    def test_a_nested_member_is_found(self, writer):
        found = _progress_keys(writer)
        assert "new_nested_field" in found.keys, found.keys
        assert ("utils/writer.py", "record") not in found.unresolved, found.unresolved

    def test_a_nested_member_it_cannot_follow_is_reported(self):
        found = _progress_keys(
            "def record(d, snapshot):\n"
            "    update_status(d, dict(progress=snapshot))\n\n"
            "def caller(d, self):\n"
            "    record(d, self.snapshot)\n")
        assert ("utils/writer.py", "caller") in found.unresolved, found.unresolved
        assert any("self.snapshot" in why
                   for why in found.unresolved[("utils/writer.py", "caller")])

    # -- a dict handed through a parameter ----------------------------------

    @pytest.mark.parametrize("caller", [
        "def accept(directory):\n"
        "    progress_update = {}\n"
        '    progress_update["new_acceptance_field"] = "operator"\n'
        "    return close(directory, progress=progress_update or None)\n\n"
        "def plain(directory):\n"
        "    return close(directory)\n",
        "def accept(directory):\n"
        "    close(directory, progress=dict(new_acceptance_field=1))\n",
    ], ids=["keyword-local", "keyword-dict-call"])
    def test_a_parameter_is_read_at_every_call(self, caller):
        found = _progress_keys(CLOSE + "\n" + caller)
        assert "new_acceptance_field" in found.keys, found.keys
        assert not {site for site in found.unresolved if site[0] == "utils/writer.py"}

    def test_a_positional_parameter_and_a_method_are_read_at_every_call(self):
        found = _progress_keys(
            "class Closer:\n"
            "    def close(self, directory, progress):\n"
            '        update_status(directory, {"progress": progress})\n\n'
            "    def accept(self, directory):\n"
            '        self.close(directory, {"new_acceptance_field": 1})\n\n'
            "def outside(closer, directory):\n"
            '    Closer.close(closer, directory, {"new_outside_field": 1})\n')
        assert {"new_acceptance_field", "new_outside_field"} <= set(found.keys), found.keys
        assert not {site for site in found.unresolved if site[0] == "utils/writer.py"}

    @pytest.mark.parametrize("caller, site, reason", [
        ("def accept(directory, options):\n    close(directory, **options)\n",
         "accept", "may carry `progress`"),
        ("CALLBACKS = [close]\n", "close", "uses as a value"),
        ("", "close", "nothing calls"),
    ], ids=["spread-at-the-call", "passed-as-a-value", "never-called"])
    def test_a_parameter_whose_callers_it_cannot_see_is_reported(self, caller, site, reason):
        found = _progress_keys(CLOSE + "\n" + caller)
        assert ("utils/writer.py", site) in found.unresolved, found.unresolved
        assert any(reason in why for why in found.unresolved[("utils/writer.py", site)])

    def test_a_parameter_of_a_name_defined_twice_is_reported(self):
        sources = _sources(
            utils_writer="from utils.status_file import update_status\n" + CLOSE
            + "\ndef accept(d):\n    close(d, progress={})\n",
            utils_other="def close(directory):\n    return directory\n",
        )
        found = inventory.progress_keys(inventory._Corpus.from_sources(sources))
        assert any("cannot be told apart" in why
                   for why in found.unresolved[("utils/writer.py", "close")])

    def test_a_writer_used_as_a_value_is_reported(self):
        found = _progress_keys("WRITERS = [update_status]\n")
        assert any("out of sight" in why
                   for why in found.unresolved[("utils/writer.py", "<module>")])

    def test_a_re_exported_writer_is_still_scanned(self):
        sources = _sources(
            utils_base="from utils import status_file as _status_file\n"
            "update_status = _status_file.update_status\n",
            utils_writer="from utils import base\n"
            "from utils.base import update_status as _update_status\n\n"
            "def record(d):\n"
            "    base.update_status(d, {'progress': {'via_module': 1}})\n"
            "    _update_status(d, {'progress': {'via_alias': 1}})\n",
        )
        found = inventory.progress_keys(inventory._Corpus.from_sources(sources))
        assert {"via_module", "via_alias"} <= set(found.keys), found.keys
        assert not {site for site in found.unresolved if site[0] != "utils/detach.py"}


class TestTheReviewedHoles:
    """Writes injected into the real source that the first inventory let
    through (Codex review of U27): each now fails the check."""

    @staticmethod
    def _injected(rel: str, anchor: str, line: str) -> "inventory._Corpus":
        text = (inventory.SKILL_DIR / rel).read_text(encoding="utf-8")
        assert text.count(anchor) == 1, f"{rel}: the anchor moved: {anchor!r}"
        return inventory.corpus().with_sources({rel: text.replace(anchor, anchor + line)})

    def test_a_doctor_only_update_of_the_result_payload(self):
        anchor = "            payload[RAN_FROM_ROOT_KEY] = self.ran_from_root\n"
        corpus = self._injected("utils/cli_common.py", anchor, (
            '        if self.mode == "doctor":\n'
            '            payload.update({"new_diagnostic": True})\n'))
        found = inventory.result_line_keys(corpus)
        assert "new_diagnostic" in found.keys
        with pytest.raises(AssertionError, match="new_diagnostic"):
            _check(found, _declared(RESULT_LINE), "MULTI_LLM_RESULT")

    def test_a_key_added_to_the_accepted_partials_progress(self):
        anchor = '        progress_update["models_succeeded"] = len(completed)\n'
        corpus = self._injected("ask_orchestrator.py", anchor,
                                '        progress_update["new_acceptance_field"] = "operator"\n')
        found = inventory.progress_keys(corpus)
        assert "new_acceptance_field" in found.keys
        with pytest.raises(AssertionError, match="new_acceptance_field"):
            _check(found, _declared(STATUS_FILE, "definitions", "progress"),
                   ".status.json progress")

    def test_a_nested_dict_call_on_a_real_writer(self):
        anchor = "def update_progress(phase_dir: str, updates: Dict[str, Any]) -> None:\n"
        corpus = self._injected("utils/status_file.py", anchor, (
            '    update_status(phase_dir, dict(progress={"new_nested_field": 1}))\n'))
        found = inventory.progress_keys(corpus)
        assert "new_nested_field" in found.keys
        with pytest.raises(AssertionError, match="new_nested_field"):
            _check(found, _declared(STATUS_FILE, "definitions", "progress"),
                   ".status.json progress")


class TestTheSchemasThemselves:
    @pytest.mark.parametrize("name", CONTRACT_SCHEMAS)
    def test_each_is_a_draft_07_schema_named_by_its_file(self, name):
        schema = _schema(name)
        jsonschema.Draft7Validator.check_schema(schema)
        assert schema["$id"] == name

    def test_the_inventory_rejects_a_key_nobody_declared(self):
        """The check this file exists for, seen failing once."""
        found = inventory.Inventory(keys={"mode": "x", "brand_new_key": "utils/x.py:1"})
        with pytest.raises(AssertionError, match="brand_new_key"):
            _check(found, {"mode"}, "MULTI_LLM_RESULT", exact=False)

    def test_the_strict_documents_reject_an_unknown_top_level_key(self):
        line = {"mode": None, "phase": None, "state": "done", "exit": 0,
                "artifacts": [], "next": None, "brand_new_key": 1}
        assert any("brand_new_key" in p for p in problems(RESULT_LINE, line))

    def test_the_dispatcher_doc_names_each_schema(self):
        text = DOCS.read_text(encoding="utf-8")
        for name in CONTRACT_SCHEMAS:
            assert f"schemas/{name}" in text, f"dispatcher.md does not name {name}"
