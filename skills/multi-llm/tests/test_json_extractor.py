"""Tests for JSON extractor utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.json_extractor import (
    extract_json_from_text,
    generate_output_path,
    sanitize_model_name,
)


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
