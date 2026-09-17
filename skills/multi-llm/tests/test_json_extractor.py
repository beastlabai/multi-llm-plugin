"""Tests for JSON extractor utilities."""

import json
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.json_extractor import (
    build_unsanitize_map,
    extract_json_from_text,
    generate_output_path,
    has_salvageable_json,
    is_answer_shaped,
    is_refused_result_envelope,
    sanitize_model_name,
)

CLAUDE_CODE_FIXTURES = Path(__file__).parent / "fixtures" / "claude-code"

#: The 2026-09-17 envelope, re-keyed. ``claude-code:claude-fable-5-1:high``
#: reviewed for 491 s, spawned four subagents, spent $25.05 and 137 348 output
#: tokens, and then hit the account's usage cap: ``api_error_status: 429`` with
#: the vendor's sentence in ``result`` and **no** ``API Error:`` prefix, no
#: ``429`` and no "rate limit" anywhere in the text — the text ladder cannot
#: read it and only the status rung can. The session and event ids are fresh;
#: nothing else is changed, because the bookkeeping shape (``iterations: []``,
#: ``permission_denials: []``, the four-entry ``subagent_stats``) is what the
#: gate misread. The top-level ``usage`` block being all zeros beside a
#: populated ``modelUsage`` is measured too, not an edit.
USAGE_CAP_ENVELOPE = (
    '{"duration_api_ms":1738676,"stop_reason":"stop_sequence",'
    '"session_id":"00000000-0000-4000-8000-000000000001",'
    '"total_cost_usd":25.05318824999999,'
    '"usage":{"output_tokens_details":{"thinking_tokens":0},"input_tokens":0,'
    '"cache_creation_input_tokens":0,"cache_read_input_tokens":0,'
    '"output_tokens":0,"server_tool_use":{"web_search_requests":0,'
    '"web_fetch_requests":0},"service_tier":"standard",'
    '"cache_creation":{"ephemeral_1h_input_tokens":0,'
    '"ephemeral_5m_input_tokens":0},"inference_geo":"","iterations":[],'
    '"speed":"standard"},'
    '"modelUsage":{"claude-haiku-4-5-20251001":{"inputTokens":1944,'
    '"outputTokens":13,"cacheReadInputTokens":0,"cacheCreationInputTokens":0,'
    '"webSearchRequests":0,"costUSD":0.002009,"contextWindow":200000,'
    '"maxOutputTokens":32000,"thinkingTokens":0,'
    '"canonicalModel":"claude-haiku-4-5","provider":"firstParty",'
    '"costBasis":"list"},"claude-fable-5-1":{"inputTokens":1936,'
    '"outputTokens":137348,"cacheReadInputTokens":37205114,'
    '"cacheCreationInputTokens":1065254,"webSearchRequests":0,'
    '"costUSD":25.051179249999993,"contextWindow":1000000,'
    '"maxOutputTokens":64000,"thinkingTokens":75679,'
    '"canonicalModel":"claude-fable-5-1","provider":"firstParty",'
    '"costBasis":"list"}},"permission_denials":[],'
    '"terminal_reason":"api_error","fast_mode_state":"off",'
    '"fast_mode_disabled_reason":"sdk_opt_in_required",'
    '"origin":{"kind":"task-notification"},'
    '"subagent_stats":{"spawned":4,"requested":{"background":4,'
    '"foreground":0,"unset":0},"started_in_background":4,"max_depth":1,'
    '"spawned_by_subagents":0,"completed":2,"failed":2,'
    '"killed":{"parent":0,"user":0,"system":0},"refused":{"depth_limit":0,'
    '"concurrency_limit":0,"budget":0},"by_type":{"general-purpose":4}},'
    '"is_error":true,"num_turns":1,"subtype":"success","api_error_status":429,'
    '"result":"You\'ve reached your Fable limit. Switch to another model, or '
    'manage usage credits at claude.ai/settings/usage?from=cc_cli_limit_message, '
    'to continue.","type":"result","duration_ms":409,'
    '"uuid":"00000000-0000-4000-8000-000000000002","queued_turn_count":0,'
    '"result_index":3}'
)


def _checkpoint(phase_dir: Path, **fields) -> Path:
    """Write a ``.status.json`` by hand, which is the shape under test.

    ``--reaggregate`` reads this file and resolves nothing, so the pre-dispatch
    gate in ``resolve_models`` (``utils/interactive.py``) never runs for it. A
    checkpoint written before that gate existed — or written by hand, which is a
    recovery the stem-attribution warning actively recommends — is therefore the
    only way a colliding pair can still reach ``build_unsanitize_map``, and
    writing the bytes directly is the only honest way to reproduce it.
    """
    path = phase_dir / ".status.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


class TestSanitizeModelName:
    """Tests for sanitize_model_name function."""

    def test_replaces_colon(self):
        """Test that colons are replaced with underscore."""
        assert sanitize_model_name("cursor-agent:auto") == "cursor-agent_auto"

    def test_replaces_dot(self):
        """Test that dots are replaced with underscore."""
        assert sanitize_model_name("gpt-5.2-high") == "gpt-5_2-high"

    def test_replaces_slash(self):
        """Test that slashes are replaced with underscore."""
        assert sanitize_model_name("moonshotai/kimi-k2.5") == "moonshotai_kimi-k2_5"

    def test_preserves_hyphen(self):
        """Test that hyphens are preserved."""
        assert sanitize_model_name("my-model-name") == "my-model-name"

    def test_preserves_underscore(self):
        """Test that underscores are preserved."""
        assert sanitize_model_name("my_model_name") == "my_model_name"

    def test_preserves_alphanumeric(self):
        """Test that alphanumeric characters are preserved."""
        assert sanitize_model_name("model123ABC") == "model123ABC"

    def test_complex_model_spec(self):
        """Test a complex model specification."""
        assert sanitize_model_name("cursor-agent:gpt-5.2-high") == "cursor-agent_gpt-5_2-high"
        assert sanitize_model_name("kilocode:moonshotai/kimi-k2.5") == "kilocode_moonshotai_kimi-k2_5"


class TestGenerateOutputPath:
    """Tests for generate_output_path function."""

    def test_sanitizes_colon(self, tmp_path):
        """Test that colons are replaced."""
        path = generate_output_path(tmp_path, "test", "review-plan", "cursor-agent:auto")
        assert ":" not in path.name
        assert path.name == "cursor-agent_auto.json"

    def test_sanitizes_dot(self, tmp_path):
        """Test that dots are replaced (consistency with sanitize_model_name)."""
        path = generate_output_path(tmp_path, "test", "review-plan", "cursor-agent:gpt-5.2-high")
        assert "." not in path.stem  # stem excludes .json extension
        assert path.name == "cursor-agent_gpt-5_2-high.json"

    def test_sanitizes_slash(self, tmp_path):
        """Test that slashes are replaced."""
        path = generate_output_path(tmp_path, "test", "review-plan", "kilocode:moonshotai/kimi-k2.5")
        assert "/" not in path.name
        assert path.name == "kilocode_moonshotai_kimi-k2_5.json"

    def test_preserves_hyphen_and_underscore(self, tmp_path):
        """Test that hyphens and underscores are preserved."""
        path = generate_output_path(tmp_path, "test", "review-plan", "my-provider_test")
        assert path.name == "my-provider_test.json"

    def test_uses_sanitize_model_name(self, tmp_path):
        """Test that generate_output_path uses sanitize_model_name consistently."""
        test_cases = [
            "cursor-agent:gpt-5.2-high",
            "kilocode:moonshotai/kimi-k2.5",
            "gemini:gemini-2.5-pro",
            "simple-model",
        ]

        for model_spec in test_cases:
            path = generate_output_path(tmp_path, "test", "review-plan", model_spec)
            expected_name = sanitize_model_name(model_spec) + ".json"
            assert path.name == expected_name, f"Mismatch for {model_spec}: {path.name} != {expected_name}"


class TestLastDocumentWins:
    """The extractor's one selection rule, stated as a test rather than a convention.

    **The last document wins, at every level** — the last envelope record (each
    adapter's own rule), then the last top-level candidate inside that record's
    text — with shape preference consulted before position, and a contained
    candidate never outranking the one that contains it. Four clauses make that
    implementable, and there is one case per clause below, on fixed strings
    rather than on a capture so the rule is pinned independently of any vendor's
    bytes.
    """

    def test_the_last_top_level_document_is_the_answer(self):
        """The rule itself: a draft followed by an answer resolves to the answer."""
        two_documents = (
            '[{"id": "DRAFT", "title": "FIRST ANSWER (must NOT be selected)"}]\n\n'
            '[{"id": "S001", "title": "Bound the retry ladder"}]'
        )
        result = extract_json_from_text(two_documents)

        assert result["success"]
        assert result["data"] == [{"id": "S001", "title": "Bound the retry ladder"}]

    def test_clause_1_a_nested_array_never_outranks_its_container(self):
        """Reversing the raw candidate list would return the inner ``[1, 2]``."""
        result = extract_json_from_text('[{"a": [1, 2]}]')

        assert result["success"]
        assert result["data"] == [{"a": [1, 2]}]

    def test_clause_1_an_inner_document_is_still_salvaged(self):
        """Contained candidates are ranked after, never dropped.

        The outer array carries a trailing comma and does not parse; the object
        inside it does, and ``TestMalformedJson``'s truncated and unbalanced
        cases depend on that fallback surviving the flip.
        """
        result = extract_json_from_text('[{"id": "S001"}, ]')

        assert result["success"]
        assert result["data"] == {"id": "S001"}

    def test_clause_2_a_trailing_object_does_not_displace_a_preceding_array(self):
        """Shape before position: every array is tried before every object."""
        array_then_object = (
            '[{"id": "S001"}]\n\n'
            'For example, one suggestion looks like {"id": "S001"}.'
        )
        result = extract_json_from_text(array_then_object, prefer_arrays=True)

        assert result["success"]
        assert result["data"] == [{"id": "S001"}]

    def test_clause_3_an_empty_trailing_array_does_not_displace_a_full_one(self):
        """Non-empty before empty: the last candidate that parses to a value."""
        result = extract_json_from_text('[{"id": "S001"}]\n\n[]')

        assert result["success"]
        assert result["data"] == [{"id": "S001"}]

    def test_clause_3_an_empty_document_is_still_served_when_it_is_all_there_is(self):
        """Else the last that parses at all: an empty answer is not a failure."""
        result = extract_json_from_text("[]")

        assert result["success"]
        assert result["data"] == []

    def test_clause_4_the_last_fenced_block_wins(self):
        """The fenced path and the bare-text path are on one rule."""
        two_fences = (
            "```json\n"
            '[{"id": "DRAFT"}]\n'
            "```\n\n"
            "```json\n"
            '[{"id": "S001"}]\n'
            "```"
        )
        result = extract_json_from_text(two_fences)

        assert result["success"]
        assert result["data"] == [{"id": "S001"}]

    def test_clause_4_an_empty_trailing_fence_does_not_displace_a_full_one(self):
        """Clause 3 binds the fenced path too, not just the bare-text one."""
        answer_then_empty = (
            "```json\n"
            '[{"id": "S001"}]\n'
            "```\n\n"
            "```json\n"
            "[]\n"
            "```"
        )
        result = extract_json_from_text(answer_then_empty)

        assert result["success"]
        assert result["data"] == [{"id": "S001"}]

    def test_clause_4_an_empty_fence_is_still_served_when_it_is_all_there_is(self):
        """Else the last fence that parses at all: an empty answer is not a failure."""
        result = extract_json_from_text("```json\n[]\n```")

        assert result["success"]
        assert result["data"] == []


#: One real kilocode stderr, verbatim, from a `review-code` run against an
#: account the key no longer belongs to: the CLI's own colours, its banner line,
#: and a four-word failure. Non-empty, non-zero exit, and not one character of
#: JSON — which is exactly the combination that used to materialize a salvage
#: request whose only possible answer was `[]`.
KILOCODE_AUTH_FAILURE = (
    "\x1b[0m\n> code · moonshotai/kimi-k3\n\x1b[0m\n"
    "\x1b[91m\x1b[1mError: \x1b[0mUser not found.\n"
)


class TestHasSalvageableJson:
    """The gate in front of every salvage request: is there anything to repair?"""

    def test_a_provider_auth_failure_is_not_salvageable(self):
        """The defect, as the run produced it — colours and all."""
        assert has_salvageable_json(KILOCODE_AUTH_FAILURE) is False

    def test_the_same_failure_with_its_escapes_already_stripped(self):
        """A log that lost its ESC bytes leaves `[0m`/`[91m` as bare text.

        The token requirement, not the ANSI strip, is what rejects those — so
        the verdict may not depend on which of the two the text arrived as.
        """
        stripped = KILOCODE_AUTH_FAILURE.replace("\x1b", "")
        assert has_salvageable_json(stripped) is False

    def test_a_bracketed_log_level_is_not_a_json_array(self):
        """`[ERROR]` and `[docs]` balance; neither is a document."""
        assert has_salvageable_json("[ERROR] User not found.\nSee [docs].") is False

    def test_prose_refusal_is_not_salvageable(self):
        """§40's write refusal: a sentence, on a slot that exited 0."""
        assert has_salvageable_json(
            "I can't write the requested file while Ask mode is active."
        ) is False

    def test_empty_output_is_not_salvageable(self):
        assert has_salvageable_json("") is False
        assert has_salvageable_json("   \n\t  \n  ") is False

    def test_an_array_with_prose_either_side_is_salvageable(self):
        """The case the gate must never cost us: a real answer plus chatter."""
        assert has_salvageable_json(
            'Here is what I found:\n[{"title": "x", "desc": "y"}]\nHope that helps!'
        ) is True

    def test_a_fenced_array_is_salvageable(self):
        assert has_salvageable_json(
            'Suggestions:\n\n```json\n[{"title": "Test"}]\n```\n'
        ) is True

    def test_a_truncated_array_is_salvageable(self):
        """Clause 2's whole reason: an unclosed bracket is not a candidate."""
        assert has_salvageable_json(
            '[{"title": "Test", "desc": "Description"}, {"title": "Anoth'
        ) is True

    def test_an_array_truncated_before_its_first_object_closes(self):
        """Nothing balances anywhere in this one — not even an inner object."""
        assert has_salvageable_json('[{"title": "Test"') is True

    def test_an_object_wrapping_the_array_is_salvageable(self):
        """A model that wrapped the array is the ordinary salvage case."""
        assert has_salvageable_json(
            '{"theme": "Add validation", "suggestions": [{"title": "Test"'
        ) is True

    def test_an_unquoted_key_inside_a_started_structure_is_salvageable(self):
        """Mechanical damage is precisely what the salvage prompt repairs.

        The opening `{"` is what admits this document; the unquoted key is the
        damage sitting behind it, and repairing that is salvage's whole job.
        """
        assert has_salvageable_json(
            '{"title": "Test", desc: "Missing quotes on key"}'
        ) is True

    def test_an_empty_array_is_a_document_the_model_really_wrote(self):
        """`[]` reaching salvage is a model saying nothing, not a model missing."""
        assert has_salvageable_json("[]") is True

    def test_a_json_error_envelope_is_not_salvageable(self):
        """Well-formed JSON, and not one word of it is an answer.

        Parsing is not the test — this document parses. A provider that answers
        its own failure in JSON must land on the same failed-model path as one
        that answers in prose.
        """
        assert has_salvageable_json(
            '{"error": {"code": 401, "message": "User not found"}}'
        ) is False

    def test_a_provider_error_event_is_not_salvageable(self):
        """The record opencode and codex print on their event stream.

        `type` is a required field of both answer schemas, which is exactly why
        the schema-field set leaves it out: it is the one name this envelope and
        a review record share.
        """
        assert has_salvageable_json(
            '{"type":"error","message":"Unauthorized"}'
        ) is False

    def test_a_colon_tagged_log_line_is_not_a_started_structure(self):
        """`[ERROR: …]` balances and looks like an unquoted key. It is neither."""
        assert has_salvageable_json("[ERROR: User not found]") is False

    def test_this_tools_own_provider_model_prefix_is_not_salvageable(self):
        """`[provider:model]` is the format this skill prints its own lines in.

        The provider that motivated the gate announces its failure behind one,
        so a rule that reads the prefix as a started structure re-admits the
        defect through the front door.
        """
        assert has_salvageable_json(
            "[kilocode:openrouter/moonshotai/kimi-k3] Agent failed"
        ) is False

    def test_a_brace_dict_written_by_something_other_than_json(self):
        """`{code: 401, …}` never parses, and never held an answer either."""
        assert has_salvageable_json('{code: 401, message: "x"}') is False

    def test_an_envelope_followed_by_a_truncated_array_is_salvageable(self):
        """A parsed-and-turned-down candidate must not end the search.

        The envelope closed; the array after it did not. Clause 2 has to reach
        that second opening, which it only does by treating the envelope's own
        span — and nothing else — as already settled.
        """
        assert has_salvageable_json(
            '{"type":"error","message":"rate limited"}\n'
            '[{"title": "Race in worker", "severity": "high"'
        ) is True

    def test_a_truncated_array_carrying_severity_is_salvageable(self):
        """`severity` is the near-miss a model writes for `importance`."""
        assert has_salvageable_json(
            '[{"title": "Race in worker", "severity": "high"'
        ) is True

    def test_a_balanced_object_wrapping_the_array_is_salvageable(self):
        """The wrapped array, closed rather than truncated: still the answer."""
        assert has_salvageable_json(
            '{"suggestions": [{"title": "x", "desc": "y", "importance": "high"}]}'
        ) is True

    def test_a_wrapper_whose_array_is_empty_is_still_an_answer(self):
        """Same document as `[]`, one key deep: a model that found nothing."""
        assert has_salvageable_json('{"issues": []}') is True

    def test_a_single_record_where_an_array_was_asked_for_is_salvageable(self):
        """No array anywhere, and the schema's own field names all over it."""
        assert has_salvageable_json(
            '{"title": "Race in worker", "desc": "Two writers", "importance": "high"}'
        ) is True

    def test_the_usage_cap_envelope_is_not_salvageable(self):
        """The 2026-09-17 defect: a 429 usage cap offered to the host as a document.

        ``permission_denials: []`` makes the envelope "an object with an
        array-valued key", and ``usage.iterations: []`` is a nested candidate
        that parses to ``[]`` — two independent routes to ``True``, and the
        orchestrator printed ``[SALVAGE_NEEDED]`` over a run that produced no
        review at all.
        """
        assert json.loads(USAGE_CAP_ENVELOPE)["is_error"] is True
        assert has_salvageable_json(USAGE_CAP_ENVELOPE) is False

    def test_the_usage_cap_envelope_is_not_answer_shaped_either(self):
        """The persist end (``_salvage_gave_up``) asks the same predicate."""
        assert is_answer_shaped(json.loads(USAGE_CAP_ENVELOPE)) is False
        assert is_refused_result_envelope(json.loads(USAGE_CAP_ENVELOPE)) is True

    @pytest.mark.parametrize(
        "row",
        [
            "rate-limit",
            "auth-error",
            "error-envelope",
            "invalid-model",
            "stdout-error",
            "context-overflow",
            "truncated",
            "zero-byte-response",
        ],
    )
    def test_every_committed_claude_code_failure_is_refused(self, row):
        """All eight ``is_error: true`` captures carry the same empty arrays."""
        text = (CLAUDE_CODE_FIXTURES / f"{row}.stdout.txt").read_text(encoding="utf-8")
        assert has_salvageable_json(text) is False

    @pytest.mark.parametrize("row", ["success", "ephemeral-only", "multi-assistant"])
    def test_every_committed_claude_code_success_still_qualifies(self, row):
        """The verdict, not the envelope shape, is what the gate reads."""
        text = (CLAUDE_CODE_FIXTURES / f"{row}.stdout.txt").read_text(encoding="utf-8")
        assert has_salvageable_json(text) is True

    def test_a_refused_envelope_followed_by_a_truncated_array_still_qualifies(self):
        """Scoping the refusal to the envelope's own span is what keeps this."""
        assert has_salvageable_json(
            USAGE_CAP_ENVELOPE + '\n[{"title": "Race in worker", "severity": "high"'
        ) is True

    def test_an_ordinary_turned_down_object_still_lets_its_children_speak(self):
        """Only a *refused envelope* silences its nested candidates."""
        assert has_salvageable_json(
            '{"status": "partial", "data": {"issues": [{"title": "x"}]}}'
        ) is True

    def test_only_the_verdict_changes_the_outcome(self):
        """The same bookkeeping with ``is_error: false`` reads exactly as before.

        An envelope that was *not* refused still qualifies on its array-valued
        key: the shape rule is untouched, and it is the CLI's own verdict — not
        the presence of ``permission_denials`` — that turns the gate.
        """
        served = {"type": "result", "is_error": False, "permission_denials": []}
        assert is_refused_result_envelope(served) is False
        assert is_answer_shaped(served) is True
        assert has_salvageable_json(json.dumps(served)) is True


class TestBuildUnsanitizeMapReportsStemCollisions:
    """Two specs, one stem, in a checkpoint the pre-dispatch gate never saw.

    ``sanitize_model_name`` is lossy in one direction only, so the map this
    function returns is an *inverse* only while it is one-to-one. Two colliding
    specs make one entry and the loop is a plain dict write, so the spec listed
    last wins in silence and every group under that stem is then attributed to a
    model that may not have produced it — a misattribution, not an omission, and
    the report reads as plausible either way.

    **The posture under test is warn-and-continue, not refuse**, and that is a
    decision rather than an oversight. It matches ``warn_stem_attribution``
    (``code_review_orchestrator.py``), which lets a phase dir whose checkpoint
    records no models at all reaggregate while saying attribution has fallen back
    to stems. The two degradations arrive from the same lossy function and have
    to tell the operator one story; refusing here would also discard a completed
    fan-out's results in order to protect the naming of them, which is the wrong
    trade at this end of the run. Fail-closed belongs at ``resolve_models``,
    before the spend — see
    ``tests/test_provider_registry.py::TestCollidingModelStemsAreRefusedBeforeDispatch``.
    """

    def test_two_colliding_specs_in_models_requested_are_reported_on_stderr(
        self, tmp_path, capsys
    ):
        """The core case: the warning names both specs and the stem they share.

        All three are asserted because all three are what the operator acts on.
        The stem is the string that appears nowhere in anything they typed, so
        without it the message describes a file name they cannot search for; the
        two specs are what tells them *which* pair of a longer list collided.
        """
        _checkpoint(tmp_path, models_requested=["p:a.b", "p:a_b"])

        mapping = build_unsanitize_map(str(tmp_path))

        err = capsys.readouterr().err
        assert "p:a.b" in err and "p:a_b" in err, (
            f"the warning has to name both colliding specs; got: {err}"
        )
        assert "p_a_b" in err, (
            f"the warning has to name the stem they share; got: {err}"
        )
        # And the consequence, not merely the fact: a message that says two specs
        # collided without saying what the report will now claim leaves the
        # operator with a curiosity rather than a decision.
        assert "attributed" in err, (
            f"the warning never says what happens to the groups: {err}"
        )
        # The mapping itself is deliberately unchanged — last spec still wins.
        # Returning `{}` or dropping the entry would turn a misattribution into a
        # *total* attribution failure for every other model in the dir, which is
        # strictly worse than the thing being reported.
        assert mapping == {"p_a_b": "p:a_b"}

    def test_the_warning_goes_to_stderr_and_never_to_stdout(self, tmp_path, capsys):
        """stdout is the reaggregate path's machine-readable channel.

        ``[PARTIAL_RESULTS: ...]`` and ``[REAGGREGATE_AFTER_SALVAGE]`` are parsed
        off it by hosts, and a degradation nobody chose does not belong in a
        stream something else is parsing. Same rule ``warn_stem_attribution``
        follows for the sibling case, asserted here because the two warnings are
        emitted by different modules and nothing else would keep them aligned.
        """
        _checkpoint(tmp_path, models_requested=["p:a.b", "p:a_b"])

        build_unsanitize_map(str(tmp_path))

        captured = capsys.readouterr()
        assert captured.out == "", (
            f"the collision warning leaked into the parsed stream: {captured.out}"
        )
        assert "p_a_b" in captured.err

    def test_a_collision_split_across_the_two_keys_is_reported_identically(
        self, tmp_path, capsys
    ):
        """The shape a resumed run really has.

        The function unions ``models_requested`` with ``models_completed``, and a
        run that resumed lists its models under both — so a colliding spec's twin
        is as likely to sit in the other key as beside it. Detecting per key
        would miss the commonest real shape entirely while passing the test
        above.
        """
        _checkpoint(
            tmp_path,
            models_requested=["p:a.b"],
            models_completed=["p:a_b"],
        )

        mapping = build_unsanitize_map(str(tmp_path))

        err = capsys.readouterr().err
        assert "p:a.b" in err and "p:a_b" in err and "p_a_b" in err, (
            f"a cross-key collision went unreported; got: {err}"
        )
        assert mapping == {"p_a_b": "p:a_b"}

    def test_the_same_spec_in_both_keys_is_the_union_doing_its_job(
        self, tmp_path, capsys
    ):
        """The negative that matters most: the ordinary checkpoint.

        Every completed run lists each model under ``models_requested`` and again
        under ``models_completed``. That is one spec written twice, not two specs
        sharing a stem, and a check keyed on arrival rather than on identity
        would warn on literally every reaggregation in the suite.
        """
        _checkpoint(
            tmp_path,
            models_requested=["cursor-agent:auto", "gemini:gemini-2.5-flash"],
            models_completed=["cursor-agent:auto", "gemini:gemini-2.5-flash"],
        )

        mapping = build_unsanitize_map(str(tmp_path))

        assert capsys.readouterr().err == ""
        assert mapping == {
            "cursor-agent_auto": "cursor-agent:auto",
            "gemini_gemini-2_5-flash": "gemini:gemini-2.5-flash",
        }

    def test_distinct_specs_whose_stems_differ_are_not_reported(
        self, tmp_path, capsys
    ):
        """The other negative: lossy stems are normal, collisions are not.

        Nearly every real spec sanitizes to something it cannot be recovered
        from on its own — that is the whole reason the checkpoint records the
        originals. Losing separators is not the defect; two specs landing on one
        stem is.
        """
        _checkpoint(
            tmp_path,
            models_requested=[
                "claude-code:sonnet-4.6",
                "cursor-agent:cursor-grok-4.6-xhigh",
            ],
        )

        mapping = build_unsanitize_map(str(tmp_path))

        assert capsys.readouterr().err == ""
        assert mapping == {
            "claude-code_sonnet-4_6": "claude-code:sonnet-4.6",
            "cursor-agent_cursor-grok-4_6-xhigh": "cursor-agent:cursor-grok-4.6-xhigh",
        }

    def test_three_specs_on_one_stem_produce_one_warning_naming_all_three(
        self, tmp_path, capsys
    ):
        """Reporting per stem rather than per arrival, and why it matters.

        Warning inside the loop would chain this into two pairwise warnings, the
        second of which has already forgotten ``p:a.b``. An operator cannot act
        on "these two collided" twice; they can act on "these three are one
        file".
        """
        _checkpoint(tmp_path, models_requested=["p:a.b", "p:a_b", "p:a/b"])

        mapping = build_unsanitize_map(str(tmp_path))

        err = capsys.readouterr().err
        assert err.count("WARNING:") == 1, (
            f"one colliding stem must produce exactly one warning; got: {err}"
        )
        for spec in ("p:a.b", "p:a_b", "p:a/b"):
            assert spec in err, f"the warning dropped {spec!r}: {err}"
        assert mapping == {"p_a_b": "p:a/b"}

    def test_a_missing_checkpoint_still_returns_an_empty_map_in_silence(
        self, tmp_path, capsys
    ):
        """The collision report must not disturb the file-absent contract.

        ``{}`` for a missing or unreadable ``.status.json`` is what the callers
        branch on to decide whether to emit the *other* stem warning
        (``code_review_orchestrator.py`` keys that on this function's result, not
        on the file's existence). A stray diagnostic here would double up with
        it.
        """
        assert build_unsanitize_map(str(tmp_path)) == {}
        assert capsys.readouterr().err == ""

