"""M7's second phase caller — the schema leg on Lane B (§4.2.1 / §4.2.2).

Lane A (``llm_client.invoke_with_provider``) reads the flag before dispatch and
applies §4.2.1's failure policy to the envelope. This pins the same contract on
the **agent-profile** lane, ``subagent_executor.dispatch_task``, where three
things differ and each of them is a test below:

* **The phase word is not the document.** §4.2.2's table is keyed by *phase*; a
  Lane B dispatch's document contract is its *kind* (``validate_result``). The
  two are the same file for exactly two kinds, and
  :data:`SCHEMA_PHASE_BY_KIND` is that intersection — asserted here against
  ``validate_result`` itself rather than by eye, because a table that drifts
  would hand a runner one schema and validate its answer against another.
* **This lane retries.** The sidecar is therefore materialized **per attempt**,
  inside ``_spawn``'s own ``finally`` — the one scope that also covers the
  timeout kill path — and the tests below assert every attempt's directory is
  gone afterwards, on the success path, on the timeout path and across a retry.
* **This lane has rungs.** §4.2.1's verdicts land on D3k rungs rather than on a
  result dict: a configuration error is rung 7 (``rejected``, never retried,
  nothing billed) and the CLI's own exhausted structured-output budget is rung 6
  (``schema_invalid``, never retried) — not rungs 4/5, which *are* retried and
  would re-run a refusal that made zero requests.

The off switch is the first class: a kind with no schema phase renders the argv
it always did, byte for byte, and its report row grows no key.

Streams are canned through a fake ``Popen`` rather than driven through a real
``tests/mocks/mock_llm.py`` subprocess, because the shapes that matter here are
claude's measured 2.1.233 envelopes — ``structured_output`` present, absent, and
the ``error_max_structured_output_retries`` subtype — which no mock CLI emits.
The codex and opencode streams still come from ``mock_llm``'s own formatters, so
there is one definition of what each CLI's output looks like.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "mocks"))

import mock_llm  # noqa: E402 - the one definition of each CLI's stream shape

from utils import structured_output as so  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.provider_registry import get_provider  # noqa: E402
from utils.providers.base import PROFILE_AGENT_READONLY  # noqa: E402
from utils.providers.claude_code import (  # noqa: E402
    JSON_SCHEMA_FLAG,
    reset_flag_gate_cache,
)
from utils.providers.codex import OUTPUT_SCHEMA_FLAG  # noqa: E402
from utils.schema_validator import validate_with_schema_file  # noqa: E402
from utils.subagent_tasks import KINDS, build_inputs, make_task  # noqa: E402


# --------------------------------------------------------------------------- #
# Documents — the two kinds whose result IS a phase document
# --------------------------------------------------------------------------- #

#: A valid ``task_generation`` result: byte-for-byte the shape
#: ``tests/mocks/mock_llm.py`` returns, so this file and the subprocess suite
#: agree about what a conforming document looks like.
GENERATED_TASKS_DOC = {
    "schema_version": "1.0",
    "plan_preamble": (
        "Simulated preamble: implement the plan in two steps, foundation first, "
        "then the wiring that consumes it."
    ),
    "tasks": [
        {
            "id": "T001",
            "title": "Create the foundation module",
            "description": "Add the module the plan names.",
            "depends_on": [],
            "files_to_create": ["src/foundation.py"],
            "files_to_modify": [],
            "acceptance_criteria": ["src/foundation.py exports the interface"],
            "estimated_complexity": "low",
            "executor_role": "agent",
            "subagent_type": None,
        },
    ],
}

#: A valid ``consolidation`` batch result.
CONSOLIDATION_DOC = {
    "verification": "passed",
    "clusters": [],
    "singletons": [
        {"group_id": "aaaa000000000000", "group_index": 0,
         "reasoning": "simulated singleton"},
    ],
}


def _envelope(result_text: str, *, structured=None, subtype="success"):
    """A claude ``--output-format json`` result envelope, as 2.1.233 emits it."""
    body = {
        "type": "result",
        "subtype": subtype,
        "is_error": subtype != "success",
        "num_turns": 2,
        "result": result_text,
    }
    if structured is not None:
        body["structured_output"] = structured
    return json.dumps(body)


def _codex_stream(text: str) -> str:
    """The same NDJSON shape ``tests/mocks/mock_llm.py`` emits for codex."""
    return mock_llm.format_codex_output(text)


def _opencode_stream(text: str) -> str:
    return mock_llm.format_opencode_output(text)


#: claude's malformed-schema refusal, verbatim at 2.1.233 (0 requests, unbilled).
STDERR_SCHEMA_NOT_JSON = (
    "Error: --json-schema is not valid JSON: JSON Parse error: Expected '}'"
)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #

class _CannedPopen:
    """A child that never really runs: canned streams, recorded argv.

    ``pid = -1`` for the same reason ``_NeverFinishingPopen`` uses it: the
    timeout path signals the child's *process group*, and ``kill_tree`` refuses
    any pid ``<= 0``, so this fake can never signal a real process.
    """

    calls = []
    stdout_text = ""
    stderr_text = ""
    exit_code = 0
    timing_out = False

    def __init__(self, args, **kwargs):
        self.args = list(args)
        self.kwargs = kwargs
        self.pid = -1
        self.returncode = None
        self.stdin = self.stdout = self.stderr = None
        self._drains = 0
        type(self).calls.append(self)

    def communicate(self, input=None, timeout=None):
        self._drains += 1
        cls = type(self)
        if cls.timing_out and self._drains == 1:
            raise subprocess.TimeoutExpired(
                cmd=self.args, timeout=timeout or 1, output=b"", stderr=b""
            )
        self.returncode = cls.exit_code
        return (cls.stdout_text, cls.stderr_text)

    def kill(self):  # pragma: no cover - only the teardown fallback calls this
        self.returncode = -9


class _Lane:
    """Small helper: configure the canned child, build tasks, dispatch."""

    def __init__(self, tmp_path):
        self.phase_dir = tmp_path
        self.tmp_path = tmp_path

    @staticmethod
    def respond(stdout="", *, stderr="", exit_code=0, timing_out=False):
        _CannedPopen.stdout_text = stdout
        _CannedPopen.stderr_text = stderr
        _CannedPopen.exit_code = exit_code
        _CannedPopen.timing_out = timing_out

    @property
    def calls(self):
        return _CannedPopen.calls

    @property
    def argv(self):
        assert _CannedPopen.calls, "nothing was spawned"
        return list(_CannedPopen.calls[-1].args)

    def task(self, kind, **overrides):
        defaults = {
            "validation": ("validation-batch-0", "validation_batch_0.json"),
            "consolidation": ("consolidation-batch-0", "consolidation_batch_0.json"),
            "task_generation": ("task-generation", "task_generation_result.json"),
        }
        task_id, output_name = defaults[kind]
        extra = {}
        context_mode = "inline"
        if kind in ("validation", "consolidation"):
            extra = {
                "batch_index": 0,
                "group_indices": [0],
                "group_ids": ["aaaa000000000000"],
                "groups_count": 1,
            }
        if kind == "consolidation":
            extra["section_key"] = "s"
        if kind == "task_generation":
            # The one kind whose context is the repository (D3e), so its task
            # carries the pinned root and the workspace-read mode its schema
            # requires.
            context_mode = "workspace-read"
            extra = {"repo_root": str(self.tmp_path)}
        return make_task(
            kind=kind,
            task_id=task_id,
            prompt=overrides.get("prompt", "Do the thing and print one JSON object."),
            output_name=output_name,
            inputs=build_inputs(grouped_file=str(self.tmp_path / "grouped.json")),
            phase_dir=str(self.phase_dir),
            context_mode=context_mode,
            extra=extra,
        )

    def dispatch(self, kind, *, runner="claude-code:sonnet", retries=0,
                 timeout=1800, **kwargs):
        return executor.dispatch_task(
            self.task(kind),
            phase_dir=str(self.phase_dir),
            runner=runner,
            retries=retries,
            timeout=timeout,
            sleep=lambda _seconds: None,
            **kwargs,
        )


@pytest.fixture
def lane(tmp_path, monkeypatch):
    """A dispatch lane whose child is canned and whose CLIs are never resolved."""
    _CannedPopen.calls = []
    _Lane.respond("")
    reset_flag_gate_cache()
    so.reset_declined_cache()
    monkeypatch.setattr(executor.subprocess, "Popen", _CannedPopen)
    monkeypatch.setattr(
        executor, "_resolve_executable", lambda name: ([f"/usr/bin/{name}"], False)
    )
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
    with patch(
        "shutil.which", side_effect=lambda name, *a, **k: f"/usr/bin/{name}"
    ), patch(
        # M14: the gate has one implementation, so the sniff is pinned there.
        "utils.providers.base.read_help_tokens",
        lambda binary, **_kwargs: frozenset({JSON_SCHEMA_FLAG}),
    ):
        yield _Lane(tmp_path)
    reset_flag_gate_cache()
    so.reset_declined_cache()


@pytest.fixture
def strict_probe():
    """The shipped ``generated_tasks.schema.json``, which codex can now carry.

    Until 2026-10-02 no shipped schema satisfied OpenAI's strict structured
    output (§15.2 finding 5), so this fixture handed the sidecar tests a
    hand-stripped copy. ``subagent_type`` is required-but-nullable now, so the
    real file is the probe and the tests below exercise the file transport
    through it.
    """
    return json.loads(
        (Path(__file__).parent.parent / "schemas" / "generated_tasks.schema.json")
        .read_text(encoding="utf-8")
    )


# --------------------------------------------------------------------------- #
# The kind → phase table
# --------------------------------------------------------------------------- #

class TestKindToPhaseTable:
    """§4.2.2's table is keyed by phase; a Lane B dispatch is keyed by kind."""

    def test_the_table_is_the_two_kinds_it_claims(self):
        assert executor.SCHEMA_PHASE_BY_KIND == {
            "task_generation": "generate-tasks",
            "consolidation": "consolidation",
        }

    @pytest.mark.parametrize("kind,phase", [
        ("task_generation", "generate-tasks"),
        ("consolidation", "consolidation"),
    ])
    @pytest.mark.parametrize("document", [
        {"totally": "wrong"},
        [],
        GENERATED_TASKS_DOC,
        CONSOLIDATION_DOC,
    ])
    def test_the_phase_schema_is_the_file_validate_result_uses(
        self, kind, phase, document
    ):
        """The invariant that makes this lane fail-closed without re-validating.

        Lane B validates every persisted document against its *kind's* result
        shape. That is §4.2.1 case 2's "validates against the same schema file"
        **only** while the two resolve to one file — so it is asserted
        functionally, over conforming and non-conforming documents alike, rather
        than by comparing two names.
        """
        schema_file = so.PHASE_SCHEMA_FILE[phase]
        assert schema_file
        assert executor.validate_result(kind, document) == validate_with_schema_file(
            document, schema_file
        )

    @pytest.mark.parametrize(
        "kind", [k for k in KINDS
                 if k not in ("task_generation", "consolidation")]
    )
    def test_every_other_kind_carries_no_schema_phase(self, kind):
        """The off switch, stated per kind: no phase, no schema, no flag."""
        assert executor.schema_phase_for_kind(kind) is None


# --------------------------------------------------------------------------- #
# Green stays green
# --------------------------------------------------------------------------- #

class TestOffSwitch:
    """A kind with no schema phase is byte-identical to the pre-M7 lane."""

    def test_the_argv_is_the_one_the_adapter_has_always_rendered(self, lane):
        task = lane.task("validation")
        lane.respond(_envelope(json.dumps({"groups": [
            {"group_hash": "aaaa000000000000", "status": "valid"}
        ]})))

        outcome = lane.dispatch("validation")

        provider = get_provider("claude-code")
        expected, _stdin = provider.build_agent_command_for_transport(
            task["prompt"], "sonnet", PROFILE_AGENT_READONLY, "stdin"
        )
        assert lane.argv == ["/usr/bin/claude", *expected[1:]]
        assert JSON_SCHEMA_FLAG not in lane.argv
        assert outcome.status == executor.STATUS_OK

    def test_the_report_row_grows_no_key(self, lane):
        lane.respond(_envelope(json.dumps({"groups": [
            {"group_hash": "aaaa000000000000", "status": "valid"}
        ]})))

        outcome = lane.dispatch("validation")

        assert "structured_output" not in outcome.to_dict()
        assert outcome.structured_output is None

    def test_an_adapter_without_support_renders_the_same_argv(self, lane):
        """A schema-bearing kind on an adapter that declines: no flag, no change."""
        task = lane.task("task_generation")
        lane.respond(_opencode_stream(json.dumps(GENERATED_TASKS_DOC)))

        outcome = lane.dispatch(
            "task_generation", runner="opencode:opencode/big-pickle", timeout=600
        )

        provider = get_provider("opencode")
        expected, _stdin = provider.build_agent_command_for_transport(
            task["prompt"], "opencode/big-pickle", PROFILE_AGENT_READONLY, "stdin"
        )
        assert lane.argv == ["/usr/bin/opencode", *expected[1:]]
        assert JSON_SCHEMA_FLAG not in lane.argv
        assert OUTPUT_SCHEMA_FLAG not in lane.argv
        assert outcome.status == executor.STATUS_OK
        # A declined gate is §4.2.1's recovery branch, never `native`: the
        # document still validated against the same schema file, through this
        # lane's own `validate_result`.
        assert outcome.structured_output == so.MARKER_RECOVERED

    def test_a_failed_dispatch_claims_no_path_at_all(self, lane):
        """The marker describes a document that exists, never an intention."""
        lane.respond(_opencode_stream("I could not do this, sorry."))

        outcome = lane.dispatch(
            "task_generation", runner="opencode:opencode/big-pickle", timeout=600
        )

        assert outcome.status != executor.STATUS_OK
        assert outcome.structured_output is None
        assert "structured_output" not in outcome.to_dict()


# --------------------------------------------------------------------------- #
# The flag, and §4.2.1's verdicts on this lane's rungs
# --------------------------------------------------------------------------- #

class TestSchemaReachesTheRunner:

    def test_the_flag_lands_in_the_policy_region(self, lane):
        lane.respond(_envelope(
            json.dumps(GENERATED_TASKS_DOC), structured=GENERATED_TASKS_DOC
        ))

        lane.dispatch("task_generation")

        argv = lane.argv
        assert JSON_SCHEMA_FLAG in argv
        at = argv.index(JSON_SCHEMA_FLAG)
        # §4.2.2's splice point: immediately after `--output-format json`, never
        # adjacent to a variadic tool flag.
        assert argv[at - 2:at] == ["--output-format", "json"]
        assert json.loads(argv[at + 1])["$id"] == "generated_tasks.schema.json"

    def test_a_native_answer_is_marked_native_and_is_the_persisted_document(
        self, lane
    ):
        lane.respond(_envelope("ignored prose", structured=GENERATED_TASKS_DOC))

        outcome = lane.dispatch("task_generation")

        assert outcome.status == executor.STATUS_OK
        assert outcome.structured_output == so.MARKER_NATIVE
        assert outcome.to_dict()["structured_output"] == so.MARKER_NATIVE
        assert json.loads(
            Path(outcome.output_path).read_text(encoding="utf-8")
        ) == GENERATED_TASKS_DOC

    def test_a_prose_answer_that_validates_is_marked_recovered(self, lane):
        lane.respond(_envelope(json.dumps(GENERATED_TASKS_DOC)))

        outcome = lane.dispatch("task_generation")

        assert outcome.status == executor.STATUS_OK
        assert outcome.structured_output == so.MARKER_RECOVERED
        assert json.loads(
            Path(outcome.output_path).read_text(encoding="utf-8")
        ) == GENERATED_TASKS_DOC

    def test_a_prose_answer_that_does_not_validate_is_rung_6(self, lane):
        """Exit 0 is never on its own a success for a schema-carrying dispatch."""
        lane.respond(_envelope(json.dumps({"verdict": "looks good"})))

        outcome = lane.dispatch("task_generation", retries=2)

        assert outcome.status == executor.STATUS_SCHEMA_INVALID
        assert outcome.rung == 6
        # Rung 6 spends no transport retry; task_generation's one
        # schema-repair re-ask (2026-10-05) is the only second attempt.
        assert [a.repair for a in outcome.attempts] == [
            None, executor.REPAIR_SCHEMA_INVALID,
        ]
        assert outcome.structured_output != so.MARKER_NATIVE

    def test_the_clis_exhausted_budget_is_rung_6_and_is_never_re_run(self, lane):
        lane.respond(
            _envelope("", subtype="error_max_structured_output_retries"),
            exit_code=1,
        )

        outcome = lane.dispatch("task_generation", retries=2)

        assert outcome.status == executor.STATUS_SCHEMA_INVALID
        assert outcome.rung == 6
        assert len(outcome.attempts) == 1
        assert "budget" in (outcome.reason or "")

    def test_a_malformed_schema_is_a_configuration_error_never_retried(self, lane):
        lane.respond("", stderr=STDERR_SCHEMA_NOT_JSON, exit_code=1)

        outcome = lane.dispatch("task_generation", retries=2)

        assert outcome.status == executor.STATUS_REJECTED
        assert outcome.rung == 7
        assert len(outcome.attempts) == 1
        assert "generated_tasks.schema.json" in (outcome.reason or "")


# --------------------------------------------------------------------------- #
# The sidecar — per attempt, removed in a `finally`
# --------------------------------------------------------------------------- #

class TestSidecarLifetime:

    @staticmethod
    def _schema_paths():
        paths = []
        for call in _CannedPopen.calls:
            if OUTPUT_SCHEMA_FLAG in call.args:
                paths.append(call.args[call.args.index(OUTPUT_SCHEMA_FLAG) + 1])
        return paths

    def test_the_sidecar_exists_during_the_spawn_and_is_removed_after(
        self, lane, strict_probe
    ):
        seen = {}

        class _Checking(_CannedPopen):
            def __init__(self, args, **kwargs):
                super().__init__(args, **kwargs)
                at = self.args.index(OUTPUT_SCHEMA_FLAG)
                seen["path"] = self.args[at + 1]
                seen["existed"] = os.path.isfile(seen["path"])
                seen["document"] = json.loads(
                    Path(seen["path"]).read_text(encoding="utf-8")
                )

        with patch.object(executor.subprocess, "Popen", _Checking):
            lane.respond(_codex_stream(json.dumps(GENERATED_TASKS_DOC)))
            outcome = lane.dispatch(
                "task_generation", runner="codex:gpt-5.5", timeout=600
            )

        assert outcome.status == executor.STATUS_OK
        assert seen["existed"] is True
        assert seen["document"] == strict_probe
        assert not os.path.exists(seen["path"])
        assert not os.path.exists(os.path.dirname(seen["path"]))

    def test_a_retry_gets_its_own_sidecar_and_leaves_none_behind(
        self, lane, strict_probe
    ):
        """This lane retries, so the `finally` has to fire once per attempt."""
        lane.respond("not JSON at all, and no envelope either")

        outcome = lane.dispatch(
            "task_generation", runner="codex:gpt-5.5", timeout=600, retries=1
        )

        paths = self._schema_paths()
        assert len(outcome.attempts) == 2
        assert len(set(paths)) == 2
        assert [p for p in paths if os.path.exists(p)] == []

    def test_a_timeout_removes_the_sidecar_and_stays_a_timeout(
        self, lane, strict_probe
    ):
        lane.respond("", timing_out=True)

        outcome = lane.dispatch(
            "task_generation", runner="codex:gpt-5.5", timeout=600
        )

        paths = self._schema_paths()
        assert outcome.status == executor.STATUS_TIMEOUT
        assert paths and [p for p in paths if os.path.exists(p)] == []


# --------------------------------------------------------------------------- #
# Nothing is spawned for a schema this run can never satisfy
# --------------------------------------------------------------------------- #

class TestFailsBeforeSpawn:

    def test_a_missing_schema_file_fails_with_nothing_dispatched(
        self, lane, monkeypatch
    ):
        table = dict(so.PHASE_SCHEMA_FILE)
        table["generate-tasks"] = "does_not_exist.schema.json"
        monkeypatch.setattr(so, "PHASE_SCHEMA_FILE", table)

        outcome = lane.dispatch("task_generation")

        assert outcome.status == executor.STATUS_REJECTED
        assert lane.calls == []
        assert "does_not_exist.schema.json" in (outcome.reason or "")

    def test_an_unmaterialized_sidecar_fails_with_no_partial_argv(
        self, lane, strict_probe, monkeypatch
    ):
        """§4.2.1's sixth case: codex takes a FILE, and there is not one."""
        import contextlib

        @contextlib.contextmanager
        def _no_sidecar(schema):
            yield schema  # never materialized: `schema.path` stays None

        monkeypatch.setattr(so, "schema_sidecar", _no_sidecar)
        lane.respond(_codex_stream(json.dumps(GENERATED_TASKS_DOC)))

        outcome = lane.dispatch(
            "task_generation", runner="codex:gpt-5.5", timeout=600, retries=1
        )

        assert outcome.status == executor.STATUS_REJECTED
        assert outcome.rung == 7
        assert lane.calls == []  # no partial argv reached the CLI
        assert len(outcome.attempts) <= 1


class TestGeneratedTasksSchemaIsStrictClean:
    """#9 (2026-10-01 three-host run): codex declined the schema on
    `<root>.tasks[]: every property must be required; missing subagent_type`
    and generate-tasks fell back to a `recovered` parse."""

    def test_codex_can_carry_the_shipped_schema(self, strict_probe):
        assert so._strict_openai_errors(strict_probe) == []

    def test_a_named_subagent_type_must_still_agree_with_the_role(self):
        from utils.task_decomposer import validate_generated_tasks
        tasks = copy.deepcopy(GENERATED_TASKS_DOC["tasks"])
        tasks[0]["subagent_type"] = "human"
        assert any("disagrees" in p for p in validate_generated_tasks(tasks))
        tasks[0]["subagent_type"] = "general-purpose"
        assert validate_generated_tasks(tasks) == []


class TestSchemaRepairIsTaskGenerationOnly:
    """The one schema-repair re-ask (2026-10-05) is granted per kind:
    `task_generation` and the three apply kinds (whose re-ask restates a report
    and is pinned in test_write_executor.py) hold a grant; every other kind's
    rung 6 is exactly as terminal as it was."""

    def test_the_grant_table_is_the_one_kind_it_claims(self):
        from utils.subagent_prompts import SCHEMA_REPAIR_SHAPES

        assert executor.SCHEMA_REPAIRS_BY_KIND == {
            "task_generation": 1,
            "apply_suggestions": 1,
            "apply_task_suggestions": 1,
            "apply_code_fixes": 1,
        }
        assert "implement" not in executor.SCHEMA_REPAIRS_BY_KIND
        # A granted kind must have a reply shape to restate.
        assert set(executor.SCHEMA_REPAIRS_BY_KIND) <= set(SCHEMA_REPAIR_SHAPES)
        assert set(executor.SCHEMA_REPAIRS_BY_KIND) <= set(KINDS)

    @pytest.mark.parametrize("kind", ["validation", "consolidation"])
    def test_another_kinds_rung_6_is_still_one_attempt(self, lane, kind):
        lane.respond(_envelope(json.dumps({"verdict": "looks good"})))

        outcome = lane.dispatch(kind, retries=2)

        assert outcome.status == executor.STATUS_SCHEMA_INVALID
        assert len(outcome.attempts) == 1
        assert len(lane.calls) == 1
        assert outcome.attempts[0].repair is None
        assert "schema-repair" not in (outcome.reason or "")

    def test_task_generation_re_asks_once_with_the_errors(self, lane):
        lane.respond(_envelope(json.dumps({"verdict": "looks good"})))

        outcome = lane.dispatch("task_generation", retries=2)

        assert outcome.status == executor.STATUS_SCHEMA_INVALID
        assert len(lane.calls) == 2
        assert [a.repair for a in outcome.attempts] == [
            None, executor.REPAIR_SCHEMA_INVALID,
        ]
        assert outcome.reason.endswith("(after the one schema-repair re-ask)")


class TestAReplyThatIsNotJson:
    """2026-10-05, opencode/big-pickle, generate-tasks, twice: the reply had the
    right keys, but a description held unescaped double quotes, so the extractor
    served one task object from inside it. The reason and the re-ask said
    "missing required field 'schema_version'"; they now say what was wrong."""

    BROKEN = (
        '{\n  "schema_version": "1.0",\n  "plan_preamble": "One page.",\n'
        '  "tasks": [\n    {\n      "id": "T001",\n      "title": "Countdown",\n'
        '      "description": "the overlay shows "3", "2", "1" and "GO" in turn",\n'
        '      "depends_on": [],\n      "files_to_create": ["index.html"],\n'
        '      "files_to_modify": [],\n      "acceptance_criteria": ["it counts"],\n'
        '      "estimated_complexity": "low",\n      "executor_role": "agent"\n'
        '    },\n    {\n      "id": "T002",\n      "title": "Laps",\n'
        '      "description": "count laps",\n      "depends_on": ["T001"],\n'
        '      "files_to_create": [],\n      "files_to_modify": ["index.html"],\n'
        '      "acceptance_criteria": ["laps count"],\n'
        '      "estimated_complexity": "low",\n      "executor_role": "agent"\n'
        '    }\n  ]\n}'
    )

    def test_the_reason_and_the_re_ask_name_the_syntax_error(
        self, lane, monkeypatch, capsys
    ):
        sent = []
        real = executor.build_schema_repair_prompt

        def spy(kind, original, errors, **kwargs):
            sent.append(kwargs)
            return real(kind, original, errors, **kwargs)

        monkeypatch.setattr(executor, "build_schema_repair_prompt", spy)
        lane.respond(_opencode_stream(self.BROKEN))

        outcome = lane.dispatch(
            "task_generation", runner="opencode:opencode/big-pickle", timeout=600
        )

        assert outcome.status == executor.STATUS_SCHEMA_INVALID
        assert outcome.reason.startswith(
            "the reply is not valid JSON (line 8, column 42: Expecting ',' delimiter"
        )
        assert "only a fragment of it could be read" in outcome.reason
        assert len(sent) == 1
        assert sent[0]["syntax_error"].startswith("line 8, column 42")
        assert "is not valid JSON; re-asking" in capsys.readouterr().out

    def test_a_whole_document_of_the_wrong_shape_is_still_its_own_errors(self, lane):
        lane.respond(_opencode_stream(json.dumps({"verdict": "looks good"})))

        outcome = lane.dispatch(
            "task_generation", runner="opencode:opencode/big-pickle", timeout=600
        )

        assert outcome.reason.startswith(
            "the extracted result does not match the task_generation result shape"
        )


class TestCodexNativeAnswer:
    """codex carries the schema natively (`--output-schema`) and answers in its
    final agent_message, so a conforming answer is `native`, with no WARN."""

    def test_a_codex_task_generation_is_native_and_silent(self, lane, capsys):
        lane.respond(_codex_stream(json.dumps(GENERATED_TASKS_DOC)))

        outcome = lane.dispatch("task_generation", runner="codex:gpt-5.5", timeout=600)

        assert outcome.status == executor.STATUS_OK
        assert outcome.structured_output == so.MARKER_NATIVE
        assert "no `structured_output` key" not in capsys.readouterr().err
