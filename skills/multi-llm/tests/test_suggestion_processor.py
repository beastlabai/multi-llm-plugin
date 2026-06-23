"""Tests for suggestion processor module."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.suggestion_processor import (
    SuggestionGroup,
    extract_suggestions_from_review,
    compute_similarity,
    group_similar_suggestions,
    deduplicate_suggestions,
    merge_suggestions_by_model,
    filter_by_importance,
    export_groups_to_json,
    import_groups_from_json,
)


class TestSuggestionGroup:
    """Tests for SuggestionGroup class."""

    def test_create_group(self):
        """Test creating a suggestion group."""
        group = SuggestionGroup("bug", "Error handling")
        assert group.category == "bug"
        assert group.theme == "Error handling"
        assert len(group.suggestions) == 0

    def test_add_suggestion(self):
        """Test adding suggestions to a group."""
        group = SuggestionGroup("bug", "Error handling")
        suggestion = {"title": "Add try-catch", "importance": "high"}

        group.add_suggestion(suggestion, "model-a")

        assert len(group.suggestions) == 1
        assert "model-a" in group.models

    def test_priority_score_increases_with_models(self):
        """Test that priority score increases with more models."""
        group = SuggestionGroup("bug", "Error handling")

        group.add_suggestion({"title": "Test", "importance": "medium"}, "model-a")
        score_1 = group.priority_score

        group.add_suggestion({"title": "Test2", "importance": "medium"}, "model-b")
        score_2 = group.priority_score

        assert score_2 > score_1

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        group = SuggestionGroup("improvement", "Performance")
        group.add_suggestion({"title": "Add cache", "importance": "low"}, "model-a")

        d = group.to_dict()
        restored = SuggestionGroup.from_dict(d)

        assert restored.category == group.category
        assert restored.theme == group.theme
        assert len(restored.suggestions) == len(group.suggestions)


class TestExtractSuggestions:
    """Tests for suggestion extraction."""

    def test_extract_from_json(self):
        """Test extracting suggestions from JSON format."""
        review_text = '''
Here are my suggestions:
```json
[
  {"title": "Add error handling", "desc": "Need try-catch", "importance": "high", "type": "bug"}
]
```
'''
        suggestions = extract_suggestions_from_review(review_text, "test-model")
        assert len(suggestions) == 1
        assert suggestions[0]["title"] == "Add error handling"
        assert suggestions[0]["source_model"] == "test-model"

    def test_extract_from_markdown_list(self):
        """Test extracting suggestions from markdown list."""
        review_text = '''
Here are the issues I found:
- Missing error handling in the API endpoints which could cause crashes
- Need to add input validation for user data
'''
        suggestions = extract_suggestions_from_review(review_text, "test-model")
        assert len(suggestions) >= 1


class TestComputeSimilarity:
    """Tests for similarity computation."""

    def test_identical_suggestions(self):
        """Test similarity of identical suggestions."""
        s1 = {"title": "Add error handling", "desc": "Need try-catch blocks", "type": "bug"}
        s2 = {"title": "Add error handling", "desc": "Need try-catch blocks", "type": "bug"}

        similarity = compute_similarity(s1, s2)
        assert similarity > 0.8

    def test_different_suggestions(self):
        """Test similarity of different suggestions."""
        s1 = {"title": "Add error handling", "desc": "Need try-catch", "type": "bug"}
        s2 = {"title": "Improve performance", "desc": "Add caching", "type": "improvement"}

        similarity = compute_similarity(s1, s2)
        assert similarity < 0.5

    def test_similar_but_not_identical(self):
        """Test similarity of similar suggestions."""
        s1 = {"title": "Add error handling", "desc": "Need error handling for edge cases", "type": "bug"}
        s2 = {"title": "Error handling needed", "desc": "Add error handling for failures", "type": "bug"}

        similarity = compute_similarity(s1, s2)
        assert 0.3 < similarity < 0.9


class TestGroupSimilarSuggestions:
    """Tests for suggestion grouping."""

    def test_group_similar_suggestions(self, sample_suggestions):
        """Test grouping similar suggestions."""
        groups = group_similar_suggestions(sample_suggestions)

        # Should group the two error handling suggestions
        assert len(groups) <= len(sample_suggestions)

    def test_groups_sorted_by_priority(self, sample_suggestions):
        """Test that groups are sorted by priority."""
        groups = group_similar_suggestions(sample_suggestions)

        if len(groups) >= 2:
            assert groups[0].priority_score >= groups[1].priority_score

    def test_grouping_is_order_independent(self):
        """Grouping must be a pure function of the suggestion *set*.

        Regression for the validation-misalignment bug: the greedy clusterer
        seeds groups in input order, so the same suggestions arriving in a
        different order (live review's results-dict order vs. --reaggregate's
        glob order) could split/merge differently — yielding a different group
        count and different content hashes. Validation, tasked against the
        original grouping, then mis-joined the re-grouped output. The grouping
        must now produce identical clusters regardless of input order.
        """
        import itertools
        from utils.state_manager import stamp_stable_ids

        # A mix of near-duplicate and distinct findings across importance tiers
        # — the kind of input whose clustering used to depend on arrival order.
        suggestions = [
            {"title": "Streaming underbills cache tokens", "desc": "Encode with responses shape",
             "importance": "high", "type": "bug", "file": "streaming.py", "line_range": [2000, 2005]},
            {"title": "Streaming under-bills unpriced", "desc": "Encode with the responses shape",
             "importance": "high", "type": "bug", "file": "streaming.py", "line_range": [2000, 2010]},
            {"title": "Add version guard test", "desc": "Pin litellm version",
             "importance": "medium", "type": "missing", "file": "test_v.py", "line_range": [1, 2]},
            {"title": "Request metrics miss cache grain", "desc": "Capture rowcount",
             "importance": "medium", "type": "missing", "file": "db_logger.py", "line_range": [50, 60]},
            {"title": "Stale comment about router", "desc": "Comment is wrong",
             "importance": "low", "type": "nit", "file": "cost_tracker.py", "line_range": [10, 11]},
            {"title": "Nano routing out of scope", "desc": "Unrelated change",
             "importance": "medium", "type": "scope", "file": "beast-nano.yaml", "line_range": [3, 4]},
        ]

        def canonical(groups):
            # Cluster membership as a set of frozensets of titles — independent
            # of group order and within-group order.
            return frozenset(
                frozenset(s.get("title", "") for s in g.suggestions) for g in groups
            )

        def hashes(groups):
            gl = [g.to_dict() if hasattr(g, "to_dict") else g for g in groups]
            stamp_stable_ids(gl)
            return tuple(sorted(g.get("group_hash") for g in gl))

        baseline = group_similar_suggestions(suggestions)
        baseline_clusters = canonical(baseline)
        baseline_hashes = hashes(baseline)

        # Every permutation must yield the identical clustering and hashes.
        for perm in itertools.permutations(suggestions):
            groups = group_similar_suggestions(list(perm))
            assert len(groups) == len(baseline), "group count changed with input order"
            assert canonical(groups) == baseline_clusters, "cluster membership changed with input order"
            assert hashes(groups) == baseline_hashes, "group_hash changed with input order"


class TestDeduplicate:
    """Tests for deduplication."""

    def test_removes_exact_duplicates(self):
        """Test that exact duplicates are removed."""
        suggestions = [
            {"title": "Add test", "desc": "Need more tests"},
            {"title": "Add test", "desc": "Need more tests"},
            {"title": "Fix bug", "desc": "Bug in line 10"},
        ]

        unique = deduplicate_suggestions(suggestions)
        assert len(unique) == 2


class TestMergeSuggestions:
    """Tests for merging suggestions from multiple models."""

    def test_merge_from_multiple_models(self):
        """Test merging suggestions from multiple model reviews."""
        reviews = {
            "model-a": '''```json
[{"title": "Add tests", "desc": "Need tests", "importance": "high", "type": "missing"}]
```''',
            "model-b": '''```json
[{"title": "Fix bug", "desc": "Bug found", "importance": "high", "type": "bug"}]
```'''
        }

        merged = merge_suggestions_by_model(reviews)
        assert len(merged) == 2


class TestFilterByImportance:
    """Tests for importance filtering."""

    def test_filter_high_only(self, sample_suggestions):
        """Test filtering for high importance only."""
        groups = group_similar_suggestions(sample_suggestions)
        filtered = filter_by_importance(groups, "high")

        for group in filtered:
            has_high = any(
                s.get("importance") == "high"
                for s in group.suggestions
            )
            assert has_high


class TestExportImport:
    """Tests for export/import functionality."""

    def test_export_import_roundtrip(self, sample_suggestions):
        """Test that export and import preserve data."""
        groups = group_similar_suggestions(sample_suggestions)

        json_str = export_groups_to_json(groups)
        restored = import_groups_from_json(json_str)

        assert len(restored) == len(groups)
