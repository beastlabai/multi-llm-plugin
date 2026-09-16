"""D2 materialized-task tests: schema, refusals, inputs binding, budget, roles.

Covers, per the Phase-1 task-3 done-when rows:

* per-kind schema validation, positive and negative
* the ``inputs`` hash binding (mismatch -> exit 3) and the v1 version window
  (missing/wrong ``tasks_schema_version`` -> exit 3)
* ``result_channel: inline-json`` refused before dispatch with the ``task_id``
  named, ``executor_role: human`` likewise, and ``workspace-read`` against a
  runner with no workspace-read capability
* ``executor_role`` derivation and the schema's ``subagent_type`` disagreement
  rejection
* the D2d UTF-16 prompt budget and its split-or-fail-loudly resolution
"""

import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.schema_validator import (  # noqa: E402
    TASKS_SCHEMA_VERSION,
    validate_subagent_task,
    validate_subagent_task_file,
)
from utils.subagent_tasks import (  # noqa: E402
    KINDS,
    PromptBudget,
    TaskFileError,
    TaskRefusal,
    assert_dispatchable,
    build_inputs,
    check_task_file_inputs,
    check_task_file_version,
    content_hash,
    derive_executor_role,
    file_content_hash,
    make_task,
    make_task_file,
    preferred_prompt_transport,
    prompt_budget_utf16,
    read_task_file,
    refuse_reason,
    resolve_output_path,
    split_indices,
    subagent_type_for_role,
    tasks_of,
    utf16_units,
    write_task_file,
)
from utils.subagent_tasks import (  # noqa: E402
    PRIOR_CHANGES_TOKEN,
    READ_KINDS,
    WRITE_KINDS,
    declared_file_set,
    is_write_kind,
    make_write_task,
    snapshot_file_set,
    write_output_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inputs(**over):
    base = {"plan_sha256": "a" * 16, "grouped_sha256": "b" * 16}
    base.update(over)
    return base


def _kind_extras(kind: str):
    """The per-kind required fields for a minimal valid task of each kind."""
    return {
        "validation": {"batch_index": 0, "group_ids": ["g1"]},
        "revalidation": {"batch_index": 0, "group_ids": ["g1"]},
        "consolidation": {"batch_index": 0, "group_ids": ["g1"]},
        "salvage": {"suggestion_id": "gemini_pro"},
        "human_decision_judge": {"group_id": "g1"},
        "task_generation": {"repo_root": "/repo"},
        # Write kinds (task 6): batch_index orders the loop, declared_files drives
        # the disjointness admission and the write policy, prior_changes_id keys the
        # idempotent prior-changes append.
        "apply_suggestions": {"batch_index": 0, "declared_files": ["plan.md"],
                              "prior_changes_id": "sec-intro"},
        "apply_task_suggestions": {"batch_index": 0, "declared_files": ["tasks.md"],
                                   "prior_changes_id": "sec-tasks"},
        "apply_code_fixes": {"batch_index": 0, "declared_files": ["src/a.py"],
                             "prior_changes_id": "src/a.py"},
        "implement": {"batch_index": 0, "declared_files": ["src/a.py"],
                      "prior_changes_id": "T001"},
    }[kind]


def _kind_inputs(kind: str):
    if kind == "salvage":
        return {"plan_sha256": "a" * 16, "salvage_request_sha256": "c" * 16}
    if kind in ("human_decision_judge", "task_generation"):
        return {"plan_sha256": "a" * 16}
    return _inputs()


def make_minimal(kind: str, **over):
    kwargs = dict(
        kind=kind,
        task_id=f"{kind}-0",
        prompt="do the thing",
        output_name="out.json",
        inputs=_kind_inputs(kind),
        phase_dir="/phase",
        extra=_kind_extras(kind),
    )
    if kind == "task_generation" or kind in WRITE_KINDS:
        kwargs["context_mode"] = "workspace-read"
    if kind in WRITE_KINDS:
        kwargs["output_name"] = write_output_name(kind, 0)
    kwargs.update(over)
    return make_task(**kwargs)


# ---------------------------------------------------------------------------
# Per-kind schema validation
# ---------------------------------------------------------------------------

class TestPerKindSchema:
    """The schema discriminates on ``kind`` and pins each kind's required set."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_minimal_task_of_every_kind_validates(self, kind):
        task = make_minimal(kind)
        assert validate_subagent_task(task) == []
        assert task["tasks_schema_version"] == TASKS_SCHEMA_VERSION
        assert task["result_channel"] == "file"
        assert task["executor_role"] == "agent"

    @pytest.mark.parametrize(
        "kind,missing",
        [
            ("validation", "batch_index"),
            ("validation", "group_ids"),
            ("revalidation", "group_ids"),
            ("consolidation", "batch_index"),
            ("salvage", "suggestion_id"),
            ("human_decision_judge", "group_id"),
            ("task_generation", "repo_root"),
        ],
    )
    def test_kind_specific_required_field_is_enforced(self, kind, missing):
        """A kind-agnostic union would let a judging task validate with no
        group_id — exactly the misattribution the stable-id mapping prevents."""
        task = make_minimal(kind)
        task.pop(missing)
        errors = validate_subagent_task(task)
        assert any(missing in e for e in errors), errors

    @pytest.mark.parametrize(
        "field",
        [
            "task_id", "kind", "prompt", "output_name", "result_channel",
            "context_mode", "capability_tier", "executor_role", "inputs",
            "tasks_schema_version",
        ],
    )
    def test_common_required_fields_are_enforced(self, field):
        task = make_minimal("validation")
        task.pop(field)
        assert any(field in e for e in validate_subagent_task(task))

    def test_unknown_field_is_rejected(self):
        """additionalProperties: false — a typo fails at materialization rather
        than dispatching a task that is merely shaped right."""
        task = make_minimal("validation")
        task["output_nmae"] = "typo.json"
        errors = validate_subagent_task(task)
        assert any("output_nmae" in e for e in errors), errors

    @pytest.mark.parametrize(
        "kind",
        ["validation", "revalidation", "consolidation", "salvage",
         "human_decision_judge"],
    )
    def test_cheap_tier_is_refused_where_agents_md_forbids_downgrading(self, kind):
        task = make_minimal(kind)
        task["capability_tier"] = "cheap"
        errors = validate_subagent_task(task)
        assert any("capability_tier" in e for e in errors), errors

    def test_make_task_refuses_a_cheap_validation_task_on_write(self):
        with pytest.raises(TaskFileError) as excinfo:
            make_minimal("validation", capability_tier="cheap")
        assert "capability_tier" in str(excinfo.value)

    @pytest.mark.parametrize("bad", ["full", "cheap"])
    def test_model_hint_never_shares_capability_tier_vocabulary(self, bad):
        """The two fields never share a vocabulary, so no producer or reader can
        conflate the layers."""
        task = make_minimal("validation")
        task["model_hint"] = bad
        assert any("model_hint" in e for e in validate_subagent_task(task))

    def test_model_hint_keeps_free_form_values(self):
        for value in ("auto", "gpt-5.2", "claude-code:sonnet"):
            task = make_minimal("validation", model_hint=value)
            assert validate_subagent_task(task) == []

    def test_output_name_must_be_a_bare_basename(self):
        task = make_minimal("validation")
        task["output_name"] = "../escape.json"
        assert any("output_name" in e for e in validate_subagent_task(task))

    def test_validation_inputs_require_grouped_hash(self):
        task = make_minimal("validation")
        task["inputs"] = {"plan_sha256": "a" * 16}
        assert any("grouped_sha256" in e for e in validate_subagent_task(task))

    def test_salvage_inputs_require_the_request_hash(self):
        task = make_minimal("salvage")
        task["inputs"] = {"plan_sha256": "a" * 16}
        assert any("salvage_request_sha256" in e for e in validate_subagent_task(task))

    def test_unknown_inputs_key_is_rejected(self):
        task = make_minimal("validation")
        task["inputs"]["plan_sha257"] = "x"
        assert any("plan_sha257" in e for e in validate_subagent_task(task))


class TestTaskFileEnvelope:
    def test_file_envelope_validates_and_carries_the_version(self):
        task = make_minimal("validation")
        document = make_task_file(
            kind="validation", tasks=[task], inputs=task["inputs"]
        )
        assert document["tasks_schema_version"] == 2
        assert validate_subagent_task_file(document) == []
        assert tasks_of(document) == [task]

    def test_batched_kinds_carry_the_list_once_under_batches(self):
        task = make_minimal("validation")
        document = make_task_file(
            kind="validation",
            tasks=[task],
            inputs=task["inputs"],
            task_list_key="batches",
        )
        assert "tasks" not in document
        assert tasks_of(document) == [task]
        assert validate_subagent_task_file(document) == []

    def test_file_with_no_task_list_is_rejected(self):
        document = {
            "tasks_schema_version": 2,
            "kind": "validation",
            "inputs": _inputs(),
        }
        assert validate_subagent_task_file(document)

    def test_write_refuses_an_invalid_document(self, tmp_path):
        task = make_minimal("validation")
        task["capability_tier"] = "cheap"
        document = {
            "tasks_schema_version": 2,
            "kind": "validation",
            "inputs": task["inputs"],
            "tasks": [task],
        }
        with pytest.raises(TaskFileError):
            write_task_file(str(tmp_path / "validation_tasks.json"), document)


# ---------------------------------------------------------------------------
# executor_role (D2h)
# ---------------------------------------------------------------------------

class TestExecutorRole:
    @pytest.mark.parametrize(
        "subagent_type,expected",
        [
            ("human", "human"),
            ("general-purpose", "agent"),
            ("", "agent"),
            (None, "agent"),
            ("something-else", "agent"),
        ],
    )
    def test_derivation(self, subagent_type, expected):
        assert derive_executor_role(subagent_type) == expected

    def test_round_trip(self):
        for role in ("agent", "human"):
            assert derive_executor_role(subagent_type_for_role(role)) == role

    def test_schema_rejects_disagreement_human_role_agent_type(self):
        task = make_minimal("validation", executor_role="human")
        task["subagent_type"] = "general-purpose"
        errors = validate_subagent_task(task)
        assert any("subagent_type" in e for e in errors), errors

    def test_schema_rejects_disagreement_agent_role_human_type(self):
        task = make_minimal("validation")
        task["subagent_type"] = "human"
        errors = validate_subagent_task(task)
        assert any("subagent_type" in e for e in errors), errors

    def test_subagent_type_may_be_absent_entirely(self):
        """A host with no agent-type vocabulary gets task files without it;
        additionalProperties: false makes its absence the normal case."""
        task = make_minimal("validation", subagent_type="")
        assert "subagent_type" not in task
        assert validate_subagent_task(task) == []

    def test_producers_derive_the_pair_so_it_cannot_drift(self):
        assert make_minimal("validation")["subagent_type"] == "general-purpose"
        human = make_minimal("validation", executor_role="human")
        assert human["subagent_type"] == "human"

    def test_only_the_two_roles_validate(self):
        task = make_minimal("validation")
        task["executor_role"] = "coordinator"
        assert any("executor_role" in e for e in validate_subagent_task(task))


# ---------------------------------------------------------------------------
# Pre-dispatch refusals (D2a / D2c)
# ---------------------------------------------------------------------------

class TestPreDispatchRefusals:
    def test_inline_json_is_refused_and_names_the_task_id(self):
        task = make_minimal("human_decision_judge")
        task["result_channel"] = "inline-json"
        reason = refuse_reason(task)
        assert reason is not None
        assert task["task_id"] in reason
        assert "inline-json" in reason
        assert "Nothing was dispatched" in reason

    def test_inline_json_raises_a_refusal_carrying_exit_code_3(self):
        task = make_minimal("human_decision_judge")
        task["result_channel"] = "inline-json"
        with pytest.raises(TaskRefusal) as excinfo:
            assert_dispatchable(task)
        assert excinfo.value.exit_code == 3
        assert excinfo.value.task_id == task["task_id"]

    def test_human_executor_role_is_never_dispatched(self):
        task = make_minimal("validation", executor_role="human")
        with pytest.raises(TaskRefusal) as excinfo:
            assert_dispatchable(task)
        assert "executor_role is 'human'" in str(excinfo.value)
        assert excinfo.value.exit_code == 3

    def test_workspace_read_refused_without_the_capability(self):
        task = make_minimal("task_generation")
        assert task["context_mode"] == "workspace-read"
        with pytest.raises(TaskRefusal) as excinfo:
            assert_dispatchable(
                task, supports_workspace_reads=False, runner="opencode:auto"
            )
        message = str(excinfo.value)
        assert "workspace-read" in message
        assert "opencode:auto" in message

    def test_workspace_read_allowed_with_the_capability(self):
        task = make_minimal("task_generation")
        assert_dispatchable(task, supports_workspace_reads=True)

    def test_an_ordinary_inline_file_task_is_dispatchable(self):
        assert refuse_reason(make_minimal("validation")) is None


# ---------------------------------------------------------------------------
# Read side: version window and inputs binding, both exit 3
# ---------------------------------------------------------------------------

class TestReadSideRefusals:
    def _write(self, tmp_path, document):
        path = tmp_path / "validation_tasks.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def _valid_document(self, tmp_path):
        task = make_minimal("validation", phase_dir=str(tmp_path))
        return make_task_file(
            kind="validation",
            tasks=[task],
            inputs=task["inputs"],
            task_list_key="batches",
        )

    def test_v1_file_is_refused_with_exit_3_naming_the_phase_dir(self, tmp_path):
        path = self._write(tmp_path, {"batches": [], "total_batches": 0})
        with pytest.raises(TaskFileError) as excinfo:
            read_task_file(str(path), phase_dir=str(tmp_path))
        message = str(excinfo.value)
        assert excinfo.value.exit_code == 3
        assert "version 1" in message
        assert str(tmp_path) in message
        assert "--force" in message

    def test_future_version_is_refused_the_same_way(self, tmp_path):
        path = self._write(tmp_path, {"tasks_schema_version": 3, "batches": []})
        with pytest.raises(TaskFileError) as excinfo:
            read_task_file(str(path), phase_dir=str(tmp_path))
        assert "3" in str(excinfo.value)
        assert excinfo.value.exit_code == 3

    def test_version_check_is_reusable_on_a_parsed_document(self):
        with pytest.raises(TaskFileError):
            check_task_file_version({}, "/p/validation_tasks.json", "/p")
        check_task_file_version({"tasks_schema_version": 2}, "/p/x.json", "/p")

    def test_matching_inputs_round_trip(self, tmp_path):
        document = self._valid_document(tmp_path)
        path = self._write(tmp_path, document)
        read = read_task_file(
            str(path), phase_dir=str(tmp_path), current_inputs=dict(document["inputs"])
        )
        assert tasks_of(read)[0]["task_id"] == "validation-0"

    def test_changed_plan_hash_is_refused_with_exit_3(self, tmp_path):
        document = self._valid_document(tmp_path)
        path = self._write(tmp_path, document)
        moved = dict(document["inputs"])
        moved["plan_sha256"] = "d" * 16
        with pytest.raises(TaskFileError) as excinfo:
            read_task_file(str(path), phase_dir=str(tmp_path), current_inputs=moved)
        message = str(excinfo.value)
        assert excinfo.value.exit_code == 3
        assert "plan_sha256" in message
        assert "--force" in message
        assert "Nothing was dispatched" in message

    def test_changed_grouped_hash_is_refused(self, tmp_path):
        document = self._valid_document(tmp_path)
        path = self._write(tmp_path, document)
        moved = dict(document["inputs"])
        moved["grouped_sha256"] = "e" * 16
        with pytest.raises(TaskFileError) as excinfo:
            read_task_file(str(path), phase_dir=str(tmp_path), current_inputs=moved)
        assert "grouped_sha256" in str(excinfo.value)

    def test_the_refusal_says_which_file_moved_and_not_only_which_key(self):
        """F8: `review-tasks --reaggregate` is exit 3 once tasks.md is edited.

        That is what the hash is for — apply-task-suggestions rewrites the file
        — but `tasks_sha256: materialized from 'a1b2', now 'c3d4'` names a key
        nothing on screen explains, so the message has to say which artifact
        moved before "--force" means anything.
        """
        with pytest.raises(TaskFileError) as excinfo:
            check_task_file_inputs(
                {"inputs": {"plan_sha256": "a" * 16, "tasks_sha256": "b" * 16}},
                "/p/validation_tasks.json",
                "/p",
                {"plan_sha256": "a" * 16, "tasks_sha256": "c" * 16},
            )
        message = str(excinfo.value)
        assert "tasks.md changed since validation" in message
        assert "apply-task-suggestions rewrites it" in message
        assert "--force" in message
        # The key and both hashes are still there for a machine reader.
        assert "tasks_sha256" in message and "'b" in message and "'c" in message

    def test_every_recorded_input_key_can_explain_itself(self):
        """A new `build_inputs` key with no reason reads as an unexplained hash."""
        from utils.subagent_tasks import _INPUT_CHANGE_REASONS

        recorded = build_inputs(
            plan_file="", grouped_file="", tasks_file="x", diff_base="ref",
            diff_text="", reviewed_code="", salvage_request_file="", items=[],
        )
        assert set(recorded) <= set(_INPUT_CHANGE_REASONS)

    def test_inputs_a_caller_cannot_recompute_are_not_checked(self):
        """Only keys present in both blocks are compared, so a code path with no
        diff available does not fail the file spuriously."""
        check_task_file_inputs(
            {"inputs": {"plan_sha256": "a" * 16, "diff_sha256": "z" * 16}},
            "/p/x.json",
            "/p",
            {"plan_sha256": "a" * 16},
        )

    def test_unparseable_file_is_refused_not_crashed(self, tmp_path):
        path = tmp_path / "validation_tasks.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(TaskFileError) as excinfo:
            read_task_file(str(path), phase_dir=str(tmp_path))
        assert excinfo.value.exit_code == 3

    def test_output_name_resolves_against_the_dir_it_was_loaded_from(self, tmp_path):
        """A moved or copied phase dir still works."""
        task = make_minimal("validation", phase_dir="/original/phase")
        assert resolve_output_path(task, str(tmp_path)) == str(tmp_path / "out.json")


# ---------------------------------------------------------------------------
# inputs hashing
# ---------------------------------------------------------------------------

class TestInputsHashing:
    def test_content_hash_matches_state_managers_plan_hash(self, tmp_path):
        """The same content hash StateManager already uses for the plan, so a
        task's inputs.plan_sha256 and state.plan_hash are comparable."""
        from utils.state_manager import StateManager

        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n\nbody\n", encoding="utf-8")
        state = StateManager(plan)
        assert file_content_hash(str(plan)) == state.state["plan_hash"]

    def test_missing_file_hashes_to_empty_string(self, tmp_path):
        assert file_content_hash(str(tmp_path / "nope.md")) == ""
        assert file_content_hash("") == ""
        assert file_content_hash(None) == ""

    def test_build_inputs_always_carries_plan_sha256(self):
        assert build_inputs() == {"plan_sha256": ""}

    def test_build_inputs_optional_members(self, tmp_path):
        salvage = tmp_path / "salvage_x.json"
        salvage.write_text("{}", encoding="utf-8")
        inputs = build_inputs(
            plan_file="",
            grouped_file="",
            diff_base="abc123",
            diff_text="diff --git",
            salvage_request_file=str(salvage),
            items=["g1", "g2"],
        )
        assert inputs["diff_base"] == "abc123"
        assert inputs["diff_sha256"] == content_hash("diff --git")
        assert inputs["salvage_request_sha256"] == content_hash("{}")
        assert "items_sha256" in inputs

    def test_reviewed_code_hashes_the_excerpt_the_prompt_embeds(self):
        """What a code-review prompt carries, not the ref it was taken against.

        `diff_base` names the far side of the diff and sits still for any edit
        on the near side of it, so a reviewed file edited between fan-out and
        validation used to invalidate nothing.
        """
        inputs = build_inputs(diff_base="abc123", reviewed_code="98: accel: 32")

        assert inputs["reviewed_code_sha256"] == content_hash("98: accel: 32")
        assert "reviewed_code_sha256" not in build_inputs(diff_base="abc123")
        assert build_inputs(reviewed_code="")["reviewed_code_sha256"] == (
            content_hash("")
        )

    def test_items_hash_is_order_independent_of_dict_key_order(self):
        a = build_inputs(items={"b": 1, "a": 2})
        b = build_inputs(items={"a": 2, "b": 1})
        assert a["items_sha256"] == b["items_sha256"]


# ---------------------------------------------------------------------------
# D2d — transport and the UTF-16 prompt budget
# ---------------------------------------------------------------------------

class TestPromptTransportAndBudget:
    @pytest.mark.parametrize("provider", ["claude-code", "codex", "opencode"])
    def test_subagent_runner_eligible_providers_have_a_non_argv_transport(self, provider):
        assert preferred_prompt_transport(provider) == "stdin"

    @pytest.mark.parametrize("provider", ["gemini", "cursor-agent", "aider", None])
    def test_other_providers_stay_on_argv(self, provider):
        assert preferred_prompt_transport(provider) == "argv"

    def test_provider_spec_with_model_still_resolves(self):
        assert preferred_prompt_transport("codex:gpt-5.5") == "stdin"

    def test_stdin_and_file_transports_are_unbounded(self):
        assert prompt_budget_utf16(transport="stdin") is None
        assert prompt_budget_utf16(transport="file") is None

    def test_argv_budget_is_the_cap_minus_headroom_minus_argv(self):
        from utils.llm_client import (
            CMDLINE_CAP_UTF16_NATIVE,
            CMDLINE_UTF16_HEADROOM,
        )

        bare = prompt_budget_utf16(transport="argv", enforce_on_posix=True)
        assert bare == CMDLINE_CAP_UTF16_NATIVE - CMDLINE_UTF16_HEADROOM
        with_argv = prompt_budget_utf16(
            transport="argv",
            non_prompt_argv=["claude", "-p", "--model", "sonnet"],
            enforce_on_posix=True,
        )
        assert with_argv < bare

    def test_batch_shim_budget_is_the_smaller_cap(self):
        from utils.llm_client import (
            CMDLINE_CAP_UTF16_BATCH,
            CMDLINE_UTF16_HEADROOM,
        )

        assert prompt_budget_utf16(
            transport="argv", is_batch_shim=True, enforce_on_posix=True
        ) == CMDLINE_CAP_UTF16_BATCH - CMDLINE_UTF16_HEADROOM

    def test_budget_is_measured_in_utf16_code_units_not_code_points(self):
        """Astral characters count double — the unit Windows actually limits."""
        astral = "\U0001F600" * 10          # 10 code points, 20 UTF-16 units
        assert len(astral) == 10
        assert utf16_units(astral) == 20
        budget = PromptBudget(15)
        assert not budget.fits(astral)
        assert budget.exceeded_by(astral) == 5
        assert PromptBudget(15).fits("x" * 15)

    def test_over_budget_prefers_splitting(self):
        budget = PromptBudget(4)
        assert budget.resolution("x" * 10, splittable=True) == "split"

    def test_over_budget_downgrades_when_the_runner_supports_workspace_reads(self):
        budget = PromptBudget(4, supports_workspace_reads=True)
        assert budget.resolution("x" * 10, splittable=False) == "workspace-read"

    def test_over_budget_never_truncates_silently(self):
        budget = PromptBudget(4, transport="argv")
        with pytest.raises(TaskFileError) as excinfo:
            budget.resolution("x" * 10, splittable=False)
        message = str(excinfo.value)
        assert "never truncated" in message
        assert "subagent_runner" in message

    def test_unbounded_budget_fits_everything(self):
        budget = PromptBudget(None)
        assert budget.unbounded
        assert budget.fits("x" * 10_000_000)
        assert budget.resolution("x" * 10_000_000, splittable=False) == "ok"

    def test_split_indices_halves_and_bottoms_out(self):
        assert split_indices([0, 1, 2, 3]) == ([0, 1], [2, 3])
        assert split_indices([0, 1, 2]) == ([0], [1, 2])
        assert split_indices([7]) == ([7], [])
        assert split_indices([]) == ([], [])


class TestAdapterTransports:
    """The adapters that are eligible as ``subagent_runner`` carry a real
    non-argv transport, and the fan-out default is unchanged."""

    @pytest.mark.parametrize("provider_name", ["claude-code", "codex", "opencode"])
    def test_stdin_omits_the_prompt_from_argv(self, provider_name):
        from utils.provider_registry import get_provider

        provider = get_provider(provider_name)
        prompt = "PROMPT-BODY"
        argv, stdin_text = provider.build_command_for_transport(
            prompt, "some-model", "stdin"
        )
        assert stdin_text == prompt
        assert prompt not in argv
        # Only the prompt is dropped; every flag survives.
        assert argv == provider.build_command(prompt, "some-model")[:-1]

    @pytest.mark.parametrize("provider_name", ["claude-code", "codex", "opencode"])
    def test_argv_transport_is_still_the_default(self, provider_name):
        from utils.provider_registry import get_provider

        provider = get_provider(provider_name)
        assert provider.prompt_transport == "argv"
        assert provider.supports_prompt_transport("stdin")
        argv, stdin_text = provider.build_command_for_transport("P", "m", "argv")
        assert stdin_text is None
        assert argv[-1] == "P"

    def test_an_argv_only_adapter_refuses_stdin(self):
        from utils.provider_registry import get_provider

        provider = get_provider("gemini")
        assert not provider.supports_prompt_transport("stdin")
        with pytest.raises(ValueError):
            provider.build_command_for_transport("P", "m", "stdin")


# ---------------------------------------------------------------------------
# Write kinds (task 6): the materialized shape a write dispatch needs
# ---------------------------------------------------------------------------

class TestWriteKinds:
    """The write kinds are part of the ONE delegation contract, not a side shape."""

    def test_the_kind_table_is_read_plus_write_with_no_overlap(self):
        assert set(KINDS) == set(READ_KINDS) | set(WRITE_KINDS)
        assert not set(READ_KINDS) & set(WRITE_KINDS)
        assert all(is_write_kind(k) for k in WRITE_KINDS)
        assert not any(is_write_kind(k) for k in READ_KINDS)

    def test_the_output_name_templates_match_the_executors_patterns(self):
        """One fact, declared in two modules; they must not drift.

        The template names the file a write batch's result is persisted under; the
        executor's regex is what D3f admits. A mismatch means every write dispatch
        is rejected at containment — or, worse, that a widened regex admits a name
        no producer writes.
        """
        from utils.subagent_executor import OUTPUT_NAME_PATTERN_BY_KIND

        for kind in WRITE_KINDS:
            pattern = OUTPUT_NAME_PATTERN_BY_KIND[kind]
            for index in (0, 1, 17):
                assert pattern.match(write_output_name(kind, index)), (
                    f"{kind}: {write_output_name(kind, index)!r} does not match "
                    f"{pattern.pattern!r}"
                )

    def test_no_write_kinds_output_pattern_admits_anothers_name(self):
        """A hand-edited task from one phase can never write into another's set."""
        from utils.subagent_executor import OUTPUT_NAME_PATTERN_BY_KIND

        for kind in WRITE_KINDS:
            name = write_output_name(kind, 0)
            others = [
                other for other in WRITE_KINDS
                if other != kind and OUTPUT_NAME_PATTERN_BY_KIND[other].match(name)
            ]
            assert others == [], f"{name} is also accepted by {others}"

    def test_a_write_task_is_workspace_read_and_full_tier(self):
        task = make_write_task(
            kind="apply_suggestions", task_id="apply-0",
            prompt=f"do it\n{PRIOR_CHANGES_TOKEN}\n",
            inputs=_inputs(), batch_index=0, prior_changes_id="sec-1",
            phase_dir="/phase", declared_files=["plan.md"],
        )
        assert task["context_mode"] == "workspace-read"
        assert task["capability_tier"] == "full"
        assert task["executor_role"] == "agent"
        assert task["output_name"] == "apply_batch_0.json"
        assert task["is_human"] is False
        assert task["can_parallelize"] is False

    def test_a_write_task_without_a_stable_prior_changes_key_is_refused(self):
        """The append is idempotent on (id, phase), so the id may not be an index."""
        with pytest.raises(TaskFileError) as excinfo:
            make_write_task(
                kind="apply_suggestions", task_id="apply-0", prompt="p",
                inputs=_inputs(), batch_index=0, prior_changes_id="",
                phase_dir="/phase", declared_files=["plan.md"],
            )
        assert "prior_changes_id" in str(excinfo.value)

    def test_a_read_kind_cannot_be_built_as_a_write_task(self):
        with pytest.raises(TaskFileError):
            make_write_task(
                kind="validation", task_id="v-0", prompt="p", inputs=_inputs(),
                batch_index=0, prior_changes_id="x", phase_dir="/phase",
            )

    def test_declared_file_set_falls_back_to_the_batchers_own_spellings(self):
        """A producer predating `declared_files` still declares something."""
        assert declared_file_set({"declared_files": ["a.py"]}) == ["a.py"]
        assert declared_file_set({"file_key": "src/x.py"}) == ["src/x.py"]
        assert declared_file_set({
            "files_to_modify": ["m.py"], "files_to_create": ["c.py"],
        }) == ["m.py", "c.py"]
        # Empty means UNDECLARED, which the scheduler runs alone — never "nothing".
        assert declared_file_set({}) == []

    def test_snapshot_file_set_defaults_to_the_declared_set(self):
        assert snapshot_file_set({"declared_files": ["a.py"]}) == ["a.py"]
        assert snapshot_file_set({
            "declared_files": ["a.py"], "snapshot_files": ["a.py", "b.py"],
        }) == ["a.py", "b.py"]

    def test_the_prior_changes_token_spelling_is_shared(self):
        """One literal, three consumers: the batchers, the executor, the scheduler."""
        from utils.subagent_executor import _PRIOR_CHANGES_TOKEN
        from utils.suggestion_batcher import format_batch_for_prompt
        from utils.suggestion_batcher import SuggestionBatch

        assert _PRIOR_CHANGES_TOKEN == PRIOR_CHANGES_TOKEN
        batch = SuggestionBatch()
        batch.add({"title": "T", "desc": "d", "description": "d",
                   "importance": "MEDIUM", "type": "addition", "reference": "## S"})
        assert PRIOR_CHANGES_TOKEN in format_batch_for_prompt(batch, "plan.md")

    def test_a_human_write_task_agrees_across_all_three_role_fields(self):
        task = make_write_task(
            kind="implement", task_id="T001", prompt="manual step",
            inputs={"plan_sha256": "a" * 16}, batch_index=0,
            prior_changes_id="T001", phase_dir="/phase",
            declared_files=["src/a.py"], executor_role="human",
        )
        assert task["executor_role"] == "human"
        assert task["subagent_type"] == "human"
        assert task["is_human"] is True

    def test_a_write_task_file_round_trips_through_the_schema(self, tmp_path):
        tasks = [
            make_write_task(
                kind="implement", task_id=f"T00{i}",
                prompt=f"task {i}", inputs={"plan_sha256": "a" * 16},
                batch_index=i, prior_changes_id=f"T00{i}",
                phase_dir=str(tmp_path), declared_files=[f"src/{i}.py"],
                depends_on=[f"T00{i-1}"] if i else [],
                output_name=write_output_name("implement", i),
            )
            for i in range(3)
        ]
        document = make_task_file(
            kind="implement", tasks=tasks, inputs={"plan_sha256": "a" * 16},
            task_list_key="batches",
        )
        path = tmp_path / "write_tasks.json"
        write_task_file(str(path), document)
        reread = read_task_file(str(path), phase_dir=str(tmp_path))
        assert [t["task_id"] for t in tasks_of(reread)] == ["T000", "T001", "T002"]
        assert tasks_of(reread)[2]["depends_on"] == ["T001"]
