"""Tests for JSON extractor utilities."""

import json
import random
import re
import subprocess
import sys
import textwrap
import time
import tracemalloc

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import json_extractor
from utils.json_extractor import (
    _balanced_spans,
    _fenced_spans,
    build_unsanitize_map,
    describe_syntax_error,
    extract_json_from_text,
    find_json_candidates,
    generate_output_path,
    has_salvageable_json,
    is_answer_shaped,
    is_refused_result_envelope,
    looks_like_ndjson,
    sanitize_model_name,
)
from utils.providers.result_envelope import parse_result_envelope

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


#: The tail of one real `cursor-agent:kimi-k3-high` review, verbatim, from
#: ``plans/uv-preflight-per-run-cost/code-review/log_cursor-agent_kimi-k3-high.txt``
#: (2026-09-20): a clean review that ends on the empty array, having written two
#: bracketed indices in its prose on the way there. `[0]` is the one that used to
#: be served as the model's answer — the result file it produced held the integer
#: `0`, the schema validator warned `'[0]: expected object, got int'`, and
#: aggregation then died on `issue["model"] = model_spec` and took the other four
#: models' eleven findings with it. Kept as a capture rather than paraphrased,
#: because the ordering (index in prose, THEN the answer) is the whole bug.
KIMI_CLEAN_REVIEW_TAIL = (
    "- `legacy/` untouched, `ROW_DOCS[1]` unchanged, `unattended.ROWS[0].site` "
    "correctly left as-is (probe still precedes the first `uv run` under "
    "A-only), no shim/lint/`SHIM_VERSION` changes — all consistent with A-only "
    "scope.\n\nDeviations from the plan's letter (no renumbering, item-4 second "
    "branch, multi-line remainder) are documented in the record's DECISIONS "
    "section with sound reasoning tied to the acceptance criteria. No bugs, "
    "missing functionality, security issues, or scope creep found.\n\n[]"
)


class TestLastDocumentWins:
    """The extractor's one selection rule, stated as a test rather than a convention.

    **The last document wins, at every level** — the last envelope record (each
    adapter's own rule), then the last top-level candidate inside that record's
    text — with shape preference consulted before position, and a contained
    candidate never outranking the one that contains it. Five clauses make that
    implementable, and there is one case per clause below, on fixed strings
    rather than on a capture so the rule is pinned independently of any vendor's
    bytes — except clause 5, whose case *is* the capture that produced it.
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

    def test_clause_1_a_document_that_parses_beats_the_pieces_inside_it(self):
        """A stray `{` in the prose before a document and a `}` after it make
        one balanced span around it, so the whole document is only
        "contained". The pieces inside it used to be tried first (the last by
        position), and a task-generation reply was read as its last task."""
        document = {
            "schema_version": "1.0",
            "plan_preamble": "p",
            "tasks": [{"id": "T001", "depends_on": []}],
        }
        text = (
            "Opening brace here { and the document follows:\n"
            + json.dumps(document)
            + "\nclosing }"
        )

        result = extract_json_from_text(text, prefer_arrays=False)

        assert result["data"] == document
        # Shape before position still holds: an array-preferring caller gets
        # what it gets from the same document without the stray braces.
        assert extract_json_from_text(text, prefer_arrays=True)["data"] == (
            extract_json_from_text(
                "Here: " + json.dumps(document), prefer_arrays=True
            )["data"]
        )

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

    def test_clause_5_a_bracketed_index_in_prose_is_not_the_answer(self):
        """The real 2026-09-20 failure: `[0]` scraped out of a clean review.

        `KIMI_CLEAN_REVIEW_TAIL` is the model's own final paragraph, so the two
        bracketed indices and the `[]` sit exactly where it put them. Under
        clause 3 alone the `[0]` was non-empty and the `[]` was not, so `[0]`
        won and the phase died aggregating an int.
        """
        result = extract_json_from_text(KIMI_CLEAN_REVIEW_TAIL, prefer_arrays=True)

        assert result["success"]
        assert result["data"] == []

    def test_clause_5_the_whole_captured_envelope_resolves_to_the_empty_review(self):
        """End to end through the adapter's own unwrapping, not just the prose.

        `cursor-agent`'s `--output-format json` envelope is what reached
        `parse_result_envelope`, which hands the inner `result` string to this
        extractor — the path that produced the `[\\n  0\\n]` result file.
        """
        envelope = json.dumps({
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": KIMI_CLEAN_REVIEW_TAIL,
            "session_id": "674cab5f-e7b6-4c75-9854-2f65972857f9",
        })

        assert parse_result_envelope(envelope) == {"success": True, "data": []}

    def test_clause_5_records_still_outrank_a_bracketed_index_written_later(self):
        """Rank before position: prose after the answer never displaces it."""
        answer_then_prose = (
            '[{"id": "S001", "title": "Bound the retry ladder"}]\n\n'
            "For the record, `unattended.ROWS[0].site` was left as-is."
        )
        result = extract_json_from_text(answer_then_prose, prefer_arrays=True)

        assert result["success"]
        assert result["data"] == [{"id": "S001", "title": "Bound the retry ladder"}]

    def test_clause_5_a_bare_scalar_array_is_still_served_when_it_is_all_there_is(self):
        """Demoted, never dropped: it is the only JSON in the reply."""
        result = extract_json_from_text("The offsets are [1, 2].")

        assert result["success"]
        assert result["data"] == [1, 2]

    def test_clause_5_binds_the_fenced_path_too(self):
        """A fenced index does not beat the empty array the model ended on."""
        fenced_index_then_empty = (
            "The row it reads is:\n\n"
            "```json\n"
            "[0]\n"
            "```\n\n"
            "Nothing to report.\n\n"
            "[]"
        )
        result = extract_json_from_text(fenced_index_then_empty)

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



class TestLooksLikeNdjson:
    """D2.4 (plans/ask-answer-recovery-and-false-success.md): the truncation
    marker, a torn last line and ``role`` no longer let a stream pass as prose.
    """

    def test_the_log_truncation_marker_does_not_vouch_for_a_stream(self):
        """A bounded log's marker is prose; the 20 events under it are a stream."""
        events = "\n".join(
            json.dumps({"type": "thinking_delta", "delta": f"d{i}"}) for i in range(20)
        )
        text = "... [63073 earlier line(s) truncated; last 20 kept]\n" + events
        assert looks_like_ndjson(text) is True

    def test_a_torn_final_line_does_not_make_a_short_stream_prose(self):
        """Five cline event lines plus a sixth cut mid-write is still a stream."""
        lines = [
            json.dumps({"type": "content_start", "contentType": "text", "i": i})
            for i in range(5)
        ]
        lines.append('{"type":"content_delta","contentType":"text","te')
        assert looks_like_ndjson("\n".join(lines)) is True

    def test_role_counts_as_a_typed_key(self):
        """An assistant-only kimi stream has no ``type`` on any record."""
        text = (
            json.dumps({"role": "assistant", "content": "one"})
            + "\n"
            + json.dumps({"role": "assistant", "content": "two"})
        )
        assert looks_like_ndjson(text) is True

    def test_a_prose_answer_ending_in_a_lone_brace_is_still_prose(self):
        """Only the *last* line is forgiven; the first prose line still decides."""
        text = "# Answer\n\nSome prose about the plan.\nMore prose.\n{"
        assert looks_like_ndjson(text) is False

    def test_a_single_event_after_the_marker_is_not_a_stream(self):
        text = "... [5 earlier line(s) truncated; last 20 kept]\n" + json.dumps(
            {"type": "x"}
        )
        assert looks_like_ndjson(text) is False

    def test_a_torn_line_outside_the_last_position_is_prose(self):
        text = '{"type":"a"}\n{"type":"b"\n{"type":"c"}'
        assert looks_like_ndjson(text) is False


#: apply-suggestions batch 6 of the 2026-10-05 OpenCode run (opencode/big-pickle),
#: trimmed: the closing message's report object carries a raw NUL byte in a key.
#: The strict parse failed the whole object and the extractor served the report's
#: own `files_touched` array, so the batch was rejected as "expected type object,
#: got list".
BATCH_6_REPLY = (
    "Changes applied to `plans/racing-game.md` (Race logic + Game loop):\n\n"
    "- **Countdown timing**: each of \"3\", \"2\", \"1\", \"GO\" lasts exactly 1 s.\n\n"
    "```json\n"
    '{"status": "applied", "ids\x00": ["cce38d9dab8f0315"], '
    '"ids": ["cce38d9dab8f0315"], "files_touched": ["plans/racing-game.md"], '
    '"summary": "Race logic + Game loop now specify the 4s countdown.", '
    '"notes": "Timer start at GO was already specified by Batch 1."}\n'
    "```"
)


class TestRawControlCharacters:
    """A tab, line break or NUL byte typed inside a string is kept, not fatal."""

    def test_a_nul_byte_in_a_key_no_longer_hands_back_an_inner_array(self):
        result = extract_json_from_text(BATCH_6_REPLY)

        assert result["success"]
        assert isinstance(result["data"], dict)
        assert result["data"]["status"] == "applied"
        assert result["data"]["files_touched"] == ["plans/racing-game.md"]
        # The byte is kept exactly as written, never cleaned up.
        assert "ids\x00" in result["data"]

    def test_a_raw_line_break_inside_a_string_is_kept(self):
        result = extract_json_from_text('{"summary": "line one\nline two"}')

        assert result["success"]
        assert result["data"] == {"summary": "line one\nline two"}

    def test_every_other_syntax_error_still_fails_the_document(self):
        """Only control characters are tolerated: an unescaped quote still breaks
        the outer object, so the extractor still falls back to a fragment."""
        broken = '{"a": "the overlay shows "GO" now", "b": [{"c": 1}]}'

        result = extract_json_from_text(broken)

        assert result["data"] != json.loads(
            '{"a": "the overlay shows \\"GO\\" now", "b": [{"c": 1}]}'
        )

    def test_the_opencode_adapter_reads_the_report_object(self):
        sys.path.insert(0, str(Path(__file__).parent / "mocks"))
        import mock_llm
        from utils.providers.opencode import OpenCodeProvider

        stream = mock_llm.format_opencode_output(BATCH_6_REPLY)
        parsed = OpenCodeProvider().parse_output(stream, "")

        assert parsed["success"]
        assert isinstance(parsed["data"], dict)
        assert parsed["data"]["status"] == "applied"


class TestEnclosingSyntaxError:
    """Say why the document did not parse when only a piece of it was served."""

    #: The 2026-10-05 big-pickle task-generation failure in miniature: right
    #: keys, one description with unescaped double quotes.
    BROKEN = (
        '{\n  "schema_version": "1.0",\n  "plan_preamble": "One page.",\n'
        '  "tasks": [\n    {\n      "id": "T001",\n'
        '      "description": "the overlay shows "3", "2", "1" and "GO" in turn",\n'
        '      "depends_on": []\n    },\n'
        '    {\n      "id": "T002",\n      "description": "fine",\n'
        '      "depends_on": ["T001"]\n    }\n  ]\n}'
    )

    def test_a_fragment_of_a_broken_document_gets_the_documents_error(self):
        result = extract_json_from_text(self.BROKEN, prefer_arrays=True)
        assert "schema_version" not in result["data"]  # a fragment was served

        found = result["enclosing_syntax_error"]

        assert found["line"] == 7
        assert found["message"] == "Expecting ',' delimiter"
        assert '"<<HERE>>3", "2"' in found["excerpt"]
        assert describe_syntax_error(found).startswith(
            "line 7, column 42: Expecting ',' delimiter — the text there reads:"
        )

    def test_a_broken_document_inside_prose_brackets_is_still_named(self):
        """The stray bracket is skipped, the broken document under it is not."""
        text = "Opening brace here { the tasks:\n" + self.BROKEN + "\nclosing }"

        found = extract_json_from_text(text)["enclosing_syntax_error"]

        assert (found["line"], found["message"]) == (7, "Expecting ',' delimiter")
        assert '"<<HERE>>3", "2"' in found["excerpt"]

    def test_a_whole_document_has_no_enclosing_error(self):
        result = extract_json_from_text('Here it is: {"a": [1, 2]}', prefer_arrays=False)
        assert result["data"] == {"a": [1, 2]}
        assert "enclosing_syntax_error" not in result

    def test_a_piece_of_a_valid_document_has_none_either(self):
        result = extract_json_from_text('{"a": [{"b": 1}]}', prefer_arrays=True)
        assert result["data"] == [{"b": 1}]
        assert "enclosing_syntax_error" not in result

    DOC = '{"status": "applied", "ids": ["s0"], "files_touched": ["p.md"]}'

    @pytest.mark.parametrize("text", [
        DOC + "\n\nMy earlier draft was: {draft: " + DOC + "}",
        "The object looks like { and then the fields.\n```json\n" + DOC
        + "\n```\nand closes with }",
        "Opening brace here { and the report follows:\n" + DOC + "\nclosing }",
        "[see the report: " + DOC + " ]",
    ], ids=["quoted-again-in-a-wrapper", "stray-braces-around-a-fence",
            "stray-braces-around-the-document", "stray-bracket-around-the-document"])
    @pytest.mark.parametrize("prefer_arrays", [True, False])
    def test_a_whole_document_inside_prose_brackets_has_none(self, text, prefer_arrays):
        """Fix round 8: the old search found the LAST span equal to the served
        value and asked whether that one was top-level, so each of these was
        reported "not valid JSON" at the stray bracket."""
        result = extract_json_from_text(text, prefer_arrays=prefer_arrays)

        assert result["data"] == json.loads(self.DOC)
        assert "enclosing_syntax_error" not in result

    def test_a_reply_with_no_json_has_none(self):
        assert "enclosing_syntax_error" not in extract_json_from_text("")


# ---------------------------------------------------------------------------
# The balanced-run scan: one linear pass that must equal walking every opener
# ---------------------------------------------------------------------------


def _walk_each_opener(text, char_type="array"):
    """The oracle: ``find_json_candidates`` as it was before it became linear.

    Kept verbatim (only renamed) so the differential tests below compare the
    one-pass scan against the walk it replaces, not against a restatement of it.
    It walks from every opener to its match, so it is quadratic: keep the inputs
    handed to it small.
    """
    candidates = []
    chars = []

    if char_type in ("array", "both"):
        chars.append(("[", "]"))
    if char_type in ("object", "both"):
        chars.append(("{", "}"))

    for start_char, end_char in chars:
        i = 0
        while i < len(text):
            if text[i] == start_char:
                depth = 1
                in_string = False
                escape_next = False
                j = i + 1

                while j < len(text) and depth > 0:
                    c = text[j]
                    if escape_next:
                        escape_next = False
                    elif c == "\\" and in_string:
                        escape_next = True
                    elif c == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if c == start_char:
                            depth += 1
                        elif c == end_char:
                            depth -= 1
                    j += 1

                if depth == 0:
                    candidates.append((i, text[i:j]))
            i += 1

    candidates.sort(key=lambda x: x[0])
    return [c[1] for c in candidates]


def _walk_spans(text, char_type):
    """The oracle's spans, recovered the way the extractor used to: a
    forward-only search for each candidate, since at most one run starts at
    any index."""
    spans = []
    pos = 0
    for candidate in _walk_each_opener(text, char_type):
        start = text.find(candidate, pos)
        assert start >= 0
        spans.append((start, start + len(candidate), candidate))
        pos = start + 1
    return spans


def _assert_scan_matches_walk(text):
    for char_type in ("array", "object", "both"):
        expected = _walk_spans(text, char_type)
        assert find_json_candidates(text, char_type) == [c for _, _, c in expected], (
            f"char_type={char_type!r} text={text!r}")
        assert _balanced_spans(text, char_type) == [(s, e) for s, e, _ in expected], (
            f"char_type={char_type!r} text={text!r}")


#: Hand-written edges: escapes outside strings, runs of ``\\"``, unterminated
#: strings, closers inside strings, mixed and crossed nesting, ANSI, newlines
#: inside strings.
_WALK_EDGE_CASES = [
    "", "[", "]", '"', "\\", "[\\]", '["\\"]"]', '[\\"]', '"[1]"', '[" \\\\"]',
    '["\\\\"]', '["a\\', '["\\u005d]"]', '[{"a":[1,2]},{"b":"]"}]', '{"a":"\\"}"}',
    '{"a":"\\\\"}', "[1] [2] [3]", '{"x":[1,{"y":[2]}]}', "[\\\\]", '"\\"[1]"',
    '[]"[]"[]', '["]"]', '["\\"]', '[["]]"]]', '\x1b[32mINFO\x1b[0m [12/400] "quoted\n',
    'x\\"[1]', '["\\\\\\"]"]', "\\[1]", "[1]\\", '["\\\\"]"]', '{"a":"\\u0022}"}',
    '[\n"\n]\n', '["\\\n"]', '[""]', '["""]', '[""""]', '["\\""]', '["\\"\\""]',
    "{{}}{}", "[[[]]]", "[[]][[]]", '{"a":{"b":{}}}', "[{]}", "{[}]", "[}", "{]",
    '["\\', '["\\"', "[\\", '{"a": "\U0001F600]"}', '{"a": [1, 2]} trailing [',
]

#: Token alphabets for random strings. A token lands whole, so multi-character
#: noise (an ANSI colour code, an escaped quote or backslash) stays intact.
_WALK_ALPHABETS = (
    ("[", "]", "{", "}", '"', "\\", " ", "a", "x", ",", ":", "\n"),
    ("[", "]", '"', "\\"),
    ("{", "}", '"', "\\"),
    ("[", "]", "{", "}", '"', "\\", '\\"', "\\\\", '""'),
    ("[", "]", "{", "}", '"', "\\", "\x1b[32m", "\x1b[0m", "é", "\U0001F600", "x"),
)

_TRICKY_STRINGS = (
    'he said "hi"', "back\\slash", "] [ } {", '\\"', "\\", '"',
    "\x1b[31mred\x1b[0m", "é\U0001F600", "",
)

_PROSE_NOISE = (
    "\n", " prose [not json ", ' x "unterminated ', "\\",
    "\x1b[33mWARN\x1b[0m step [3/9] ", "{ and then } ", "",
)


def _random_value(rng, depth=0):
    roll = rng.random()
    if depth >= 4 or roll < 0.35:
        return rng.choice(
            [rng.randint(-5, 500), rng.choice(_TRICKY_STRINGS), True, None, 1.5])
    if roll < 0.7:
        return [_random_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {
        rng.choice(_TRICKY_STRINGS) + str(i): _random_value(rng, depth + 1)
        for i in range(rng.randint(0, 4))
    }


def _document_mix(rng):
    """Prose, ANSI noise and several JSON documents, some fenced and some cut
    short anywhere, the last one included."""
    parts = []
    for _ in range(rng.randint(1, 5)):
        doc = json.dumps(
            _random_value(rng),
            ensure_ascii=rng.random() < 0.5,
            indent=rng.choice([None, 2]),
        )
        shape = rng.random()
        if shape < 0.25:
            doc = doc[: rng.randint(0, len(doc))]
        elif shape < 0.4:
            doc = "```json\n" + doc + "\n```"
        parts.append(doc)
        parts.append(rng.choice(_PROSE_NOISE))
    return "".join(parts)


class TestBalancedScanMatchesTheWalk:
    """``find_json_candidates`` solves the bracket walk in one pass. It must
    return exactly what walking from every opener returns — the same
    candidates in the same order — because "last document wins", the
    contained-candidate rule and ``only_object`` all read that list."""

    @pytest.mark.parametrize("text", _WALK_EDGE_CASES)
    def test_hand_written_edges(self, text):
        _assert_scan_matches_walk(text)

    @pytest.mark.parametrize("alphabet_index", range(len(_WALK_ALPHABETS)))
    def test_seeded_random_strings(self, alphabet_index):
        rng = random.Random(20261108 + alphabet_index)
        alphabet = _WALK_ALPHABETS[alphabet_index]
        for _ in range(4000):
            length = rng.choice(
                [rng.randint(0, 12), rng.randint(0, 60), rng.randint(0, 160)])
            _assert_scan_matches_walk(
                "".join(rng.choice(alphabet) for _ in range(length)))

    def test_seeded_document_mixes(self):
        rng = random.Random(4242)
        for _ in range(1500):
            _assert_scan_matches_walk(_document_mix(rng))

    def test_extraction_serves_the_same_document_either_way(self, monkeypatch):
        """Downstream of the scan: the extractor and the salvage check give the
        same answer on the oracle's spans as on the scan's."""
        rng = random.Random(777)
        texts = [_document_mix(rng) for _ in range(200)]

        def results():
            return [
                (
                    extract_json_from_text(text, prefer_arrays=True),
                    extract_json_from_text(text, prefer_arrays=False),
                    has_salvageable_json(text),
                )
                for text in texts
            ]

        scanned = results()
        monkeypatch.setattr(
            json_extractor,
            "_balanced_spans",
            lambda text, char_type: [(s, e) for s, e, _ in _walk_spans(text, char_type)],
        )
        assert results() == scanned


class TestBalancedScanIsLinear:
    """The walk from each opener was quadratic: a 112 KB coloured log took
    about 10 s and 448 KB about 160 s, on a worker thread holding the GIL.
    The one-pass scan does each input here in milliseconds, so the ceiling
    leaves three orders of magnitude for a slow or loaded runner. The inputs
    are sized so a quadratic scan fails in under a minute instead of hanging
    the suite for an hour."""

    CEILING_S = 5.0
    LOG_LINE = '\x1b[32mINFO\x1b[0m compiling module foo.bar [12/400] "quoted\n'

    @pytest.mark.parametrize("text", [
        LOG_LINE * 2000,
        "[ " * 20_000,
        '{"a":' * 10_000,
        '\\"[' * 15_000,
    ], ids=["ansi-log-112KB", "unmatched-openers-40KB", "unclosed-objects-50KB",
            "escaped-quotes-45KB"])
    def test_a_large_unbalanced_reply_scans_in_linear_time(self, text):
        started = time.perf_counter()
        for char_type in ("array", "object", "both"):
            find_json_candidates(text, char_type)
        extract_json_from_text(text, prefer_arrays=True)
        elapsed = time.perf_counter() - started
        assert elapsed < self.CEILING_S, (
            f"scanning {len(text)} characters took {elapsed:.1f}s, over the "
            f"{self.CEILING_S}s ceiling: the balanced-run scan has gone "
            f"quadratic again (see _balanced_spans in utils/json_extractor.py)"
        )


def _traced_peak(call):
    """The most memory *call* held at once beyond what was held before it, by
    ``tracemalloc``; an outer trace (``-X tracemalloc``) is left running."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        call()
        return tracemalloc.get_traced_memory()[1] - before
    finally:
        if not was_tracing:
            tracemalloc.stop()


class TestBalancedScanMemory:
    """The scan keeps a few bytes per bracket of the pair it walks and nothing
    per quote or backslash, and reads nothing before the first opener or after
    the last closer. A reply made of quotes and backslashes, with no bracket or
    with two, costs the scan no memory. Measured 2026-11-08: with a table entry
    per quote and backslash, 8 MiB of ``"`` peaked at about 242 MiB and raised
    ``MemoryError`` under a 256 MiB limit; walking each opener had needed about
    22 MiB. These inputs cost about 25 MiB each that way, and about 2 KiB now."""

    SIZE = 256 * 1024
    SCAN_PEAK_BYTES = 64 * 1024
    TEXTS = {
        "quotes": '"' * SIZE,
        "backslashes": "\\" * SIZE,
        "escaped-quotes": '\\"' * (SIZE // 2),
        "spaced-quotes": 'a"' * (SIZE // 2),
        "one-array-of-quotes": "[" + '"' * SIZE + "]",
        "one-object-of-escapes": '{"' + '\\"' * (SIZE // 2) + '"}',
    }

    @pytest.mark.parametrize("name", list(TEXTS))
    def test_quotes_and_backslashes_cost_the_scan_nothing(self, name):
        text = self.TEXTS[name]
        peak = _traced_peak(
            lambda: [_balanced_spans(text, kind) for kind in ("array", "object", "both")])
        assert peak < self.SCAN_PEAK_BYTES, (
            f"scanning {len(text)} characters of {name} held {peak} bytes: the "
            f"scan is storing quotes or backslashes again (see _pair_runs)")

    @pytest.mark.parametrize("name", list(TEXTS))
    def test_the_readers_hold_no_more_than_a_copy_of_the_reply(self, name):
        text = self.TEXTS[name]
        peak = _traced_peak(lambda: (
            extract_json_from_text(text, prefer_arrays=True),
            extract_json_from_text(text, prefer_arrays=False),
            has_salvageable_json(text),
        ))
        assert peak < 2 * len(text), (
            f"reading {len(text)} characters of {name} held {peak} bytes")


class TestCandidateBudget:
    """Nested candidates add up to a quadratic length: 1 MiB of ``[`` then
    ``]`` holds 256 GiB of them, and every reader slices and parses what it
    tries. Past the budget only the outermost runs are returned — by nesting
    level, then position, up to the first that does not fit — so a reply that
    nests brackets thousands deep is still read, in bounded time and memory."""

    @staticmethod
    def _budget(monkeypatch, floor, per_char=0):
        monkeypatch.setattr(json_extractor, "_CANDIDATE_BUDGET_FLOOR", floor)
        monkeypatch.setattr(json_extractor, "_CANDIDATE_BUDGET_PER_CHAR", per_char)

    NESTED = "[" * 6 + "]" * 6  # runs of 12, 10, 8, 6, 4 and 2: 42 in all

    def test_within_the_budget_every_run_is_returned(self, monkeypatch):
        self._budget(monkeypatch, 42)
        assert len(_balanced_spans(self.NESTED, "array")) == 6

    def test_past_it_the_outermost_levels_are_kept(self, monkeypatch):
        self._budget(monkeypatch, 30)
        assert _balanced_spans(self.NESTED, "array") == [(0, 12), (1, 11), (2, 10)]

    def test_the_budget_grows_with_the_text(self, monkeypatch):
        self._budget(monkeypatch, 0, per_char=2)
        assert _balanced_spans(self.NESTED, "array") == [(0, 12), (1, 11)]

    def test_a_level_that_does_not_fit_is_cut_at_its_first_run_that_does_not(
        self, monkeypatch
    ):
        # Three runs of 5 at level 0 (15), three of 3 inside them at level 1.
        self._budget(monkeypatch, 15 + 3 + 2)
        assert _balanced_spans("[[1]] [[2]] [[3]]", "array") == [
            (0, 5), (1, 4), (6, 11), (12, 17)]

    def test_both_bracket_pairs_share_one_budget(self, monkeypatch):
        self._budget(monkeypatch, 13 + 11)
        assert _balanced_spans('[{"a": [{}]}]', "both") == [(0, 13), (1, 12)]

    def test_the_readers_copy_only_the_candidate_they_parse(self):
        # 2,048 levels: 4 MiB of candidates, all within the budget.
        text = "[" * 2048 + "]" * 2048
        peak = _traced_peak(lambda: (
            extract_json_from_text(text, prefer_arrays=True),
            extract_json_from_text(text, prefer_arrays=False),
            has_salvageable_json(text),
        ))
        assert peak < 1024 * 1024, f"reading the nesting held {peak} bytes"

    def test_the_document_after_deep_nesting_is_still_served(self, monkeypatch):
        answer = '[{"title": "x"}]'
        text = "[" * 40 + "]" * 40 + " " + answer
        self._budget(monkeypatch, 200)
        candidates = find_json_candidates(text)
        assert sum(map(len, candidates)) <= 200
        assert candidates[-1] == answer
        assert extract_json_from_text(text)["data"] == [{"title": "x"}]

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="the child's address-space limit needs RLIMIT_AS and /proc",
    )
    def test_a_megabyte_of_nesting_reads_in_bounded_time_and_memory(self):
        """The review's input, read by every reader, in a child process under
        an address-space limit so a regression fails here rather than taking
        the worker down. Its outer candidates are nested deeper than the JSON
        parser's stack allows; the readers skip those rather than raise."""
        script = textwrap.dedent(f"""
            import json, resource, sys, time
            sys.path.insert(0, {str(Path(__file__).parent.parent)!r})
            from utils.json_extractor import (
                extract_json_from_text, find_json_candidates, has_salvageable_json)
            text = "[" * 524288 + "]" * 524288
            with open("/proc/self/statm") as statm:
                mapped = int(statm.read().split()[0]) * resource.getpagesize()
            limit = mapped + 160 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            started = time.perf_counter()
            extract_json_from_text(text, prefer_arrays=True)
            extract_json_from_text(text, prefer_arrays=False)
            has_salvageable_json(text)
            chars = sum(map(len, find_json_candidates(text, "both")))
            print(json.dumps({{"elapsed": time.perf_counter() - started, "chars": chars}}))
        """)
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            pytest.fail("reading 1 MiB of nesting took over 60 s: the candidate "
                        "budget is not bounding the readers (see _balanced_spans)")
        assert completed.returncode == 0, completed.stderr[-3000:]
        result = json.loads(completed.stdout)
        budget = max(json_extractor._CANDIDATE_BUDGET_FLOOR,
                     json_extractor._CANDIDATE_BUDGET_PER_CHAR * 2 * 524288)
        assert result["chars"] <= budget
        assert result["elapsed"] < 30, result


# ---------------------------------------------------------------------------
# The fenced-block scan: one forward pass that must equal the fence pattern
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


def _fenced_spans_by_pattern(text):
    """The oracle: strategy 1's fence search as it was before it became linear,
    returning each match's first-group span instead of its text. Quadratic on
    fences with no closing bracket after them: keep the inputs handed to it
    small."""
    return [
        match.span(1)
        for match in re.finditer(r"```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```", text)
    ]


def _assert_fences_match_pattern(text):
    assert _fenced_spans(text) == _fenced_spans_by_pattern(text), f"text={text!r}"


#: Hand-written edges: language tags, unclosed fences, fences inside a block,
#: backticks that are not three, closing brackets not followed by a fence,
#: whitespace the pattern's ``\s`` counts (CRLF, NBSP, U+2028, U+001C).
_FENCE_EDGE_CASES = [
    "", "```", "``````", "```[", "```[]", "```[]```", "```{}```", "```json[1]```",
    "```json\n[1]\n```", "```json\n{\"a\": 1}\n```", "```JSON\n[1]\n```",
    "```jsonc\n[1]\n```", "```python\n[1]\n```", "```js\n[1]\n```", "```json json[1]```",
    "````[1]````", "````json\n[1]\n````", "``[1]``", "~~~json\n[1]\n~~~",
    "```[1]````", "```[1] ```x```", "```[1] x ```[2]```", "```[1]\n```\n```[2]\n```",
    "```python\nprint(1)\n```\n```json\n[2]\n```", "```[``` ]```", "```[ ```[1]``` ]```",
    "```[1]``````[2]```", "```[1]```[2]```", "```[1}```", "```{1]```", "```[[1],[2]]```",
    "```[\"]```\"]```", "```json\r\n[1]\r\n```", "```json [1] ```",
    "```json\x1c[1]\x1f```", "```\t\n [1] \t\n```", "] ``` [1] ```", "```json\n[1]\n",
    "```json\n[1]", "prose ```json\n[{\"a\": [1, 2]}]\n``` more prose ``` tail",
    "```json\n[1]\n``` ```json\n[\n``` ```json\n[3]\n```", "```" + "json" * 3 + "[1]```",
    "x```[]```y```{}```z", "```json\n[1]\n```json\n[2]\n```", "```[]　```",
]

#: Token alphabets for random strings, each heavy in what the fence rule reads.
_FENCE_ALPHABETS = (
    ("```", "`", "json", "[", "]", "{", "}", " ", "\n", "x"),
    ("```", "``", "````", "[", "]", " "),
    ("```json\n", "```", "\n```", "[1]", "{}", "]", "}", "[", "{", "python\n", "\r\n"),
    ("```", "json", "JSON", "[", "}", "\t", " ", " ", "\x1c", "\x85", "é", "a"),
)


def _fence_mix(rng):
    """Prose and documents fenced every way a model fences one: tagged and not,
    with another language, unclosed, cut short, closing on the same line."""
    parts = []
    for _ in range(rng.randint(1, 6)):
        doc = json.dumps(_random_value(rng), indent=rng.choice([None, 2]))
        opening = rng.choice(["```json\n", "```\n", "```", "```json ", "```python\n",
                              "````json\n", "``` json\n", "```JSON\n"])
        closing = rng.choice(["\n```", "```", "\n````", "\n", " ``` ", ""])
        if rng.random() < 0.2:
            doc = doc[: rng.randint(0, len(doc))]
        parts.append(opening + doc + closing)
        parts.append(rng.choice(_PROSE_NOISE + ("\n\n", " see ```x``` ", "`[1]`")))
    return "".join(parts)


def _fenced_fixture_texts():
    """Every committed fixture holding a fence, decoded leniently: captured
    provider output is the corpus fences come from."""
    texts = []
    for path in sorted(FIXTURES.rglob("*")):
        if path.is_file():
            text = path.read_bytes().decode("utf-8", errors="replace")
            if "```" in text:
                texts.append(text)
    return texts


class TestFencedSpansMatchThePattern:
    """Strategy 1 finds fenced blocks in one forward pass. It must return
    exactly the blocks the fence pattern matched — the same blocks in the same
    order — because the last fenced block that parses is the answer
    (clause 4)."""

    @pytest.mark.parametrize("text", _FENCE_EDGE_CASES)
    def test_hand_written_edges(self, text):
        _assert_fences_match_pattern(text)
        _assert_fences_match_pattern(text.strip())

    @pytest.mark.parametrize("alphabet_index", range(len(_FENCE_ALPHABETS)))
    def test_seeded_random_strings(self, alphabet_index):
        rng = random.Random(20261109 + alphabet_index)
        alphabet = _FENCE_ALPHABETS[alphabet_index]
        for _ in range(5000):
            length = rng.choice(
                [rng.randint(0, 8), rng.randint(0, 30), rng.randint(0, 100)])
            _assert_fences_match_pattern(
                "".join(rng.choice(alphabet) for _ in range(length)))

    def test_seeded_fence_mixes(self):
        rng = random.Random(9090)
        for _ in range(2000):
            _assert_fences_match_pattern(_fence_mix(rng))
        for _ in range(500):
            _assert_fences_match_pattern(_document_mix(rng))

    def test_every_fixture(self):
        texts = _fenced_fixture_texts()
        assert texts, f"no fixture under {FIXTURES} holds a fence"
        for text in texts:
            _assert_fences_match_pattern(text)
            _assert_fences_match_pattern(text.strip())

    def test_extraction_serves_the_same_document_either_way(self, monkeypatch):
        """Downstream of the scan: the extractor gives the same answer on the
        pattern's spans as on the scan's."""
        rng = random.Random(31337)
        texts = [_fence_mix(rng) for _ in range(300)] + _fenced_fixture_texts()

        def results():
            return [
                (extract_json_from_text(text, prefer_arrays=True),
                 extract_json_from_text(text, prefer_arrays=False))
                for text in texts
            ]

        scanned = results()
        monkeypatch.setattr(json_extractor, "_fenced_spans", _fenced_spans_by_pattern)
        assert results() == scanned


class TestFencedSpansAreLinear:
    """The fence pattern was quadratic: every fence with no closing bracket
    after it searched the rest of the reply, so 80 KB of `````[`` took about
    4 s and 1 MiB about 12 minutes. The scan does 1 MiB in milliseconds. It
    runs in a child process under a timeout, so a regression fails here
    instead of hanging the suite."""

    def test_a_megabyte_of_unclosed_fences_reads_in_bounded_time(self):
        script = textwrap.dedent(f"""
            import json, sys, time
            sys.path.insert(0, {str(Path(__file__).parent.parent)!r})
            from utils.json_extractor import extract_json_from_text
            size = 1024 * 1024
            texts = [
                "```[" * (size // 4),
                "```json\\n{{" * (size // 9),
                "```[" * (size // 8) + "] x" * (size // 6),
                "```[" + ("]" + " " * 64 + "x") * (size // 66),
            ]
            started = time.perf_counter()
            for text in texts:
                for prefer_arrays in (True, False):
                    extract_json_from_text(text, prefer_arrays=prefer_arrays)
            print(json.dumps({{"elapsed": time.perf_counter() - started}}))
        """)
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            pytest.fail("reading 1 MiB of unclosed fences took over 60 s: the fence "
                        "scan has gone quadratic again (see _fenced_spans)")
        assert completed.returncode == 0, completed.stderr[-3000:]
        assert json.loads(completed.stdout)["elapsed"] < 20
