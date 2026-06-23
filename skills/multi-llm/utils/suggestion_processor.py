"""Suggestion processor module for handling multi-LLM review suggestions."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class SuggestionGroup:
    """A group of similar suggestions from multiple models."""

    def __init__(self, category: str, theme: str):
        self.category = category
        self.theme = theme
        self.suggestions: List[Dict[str, Any]] = []
        self.models: List[str] = []
        self.priority_score: float = 0.0
        self.validation_status: Optional[str] = None
        self.validation_reason: Optional[str] = None

    def add_suggestion(self, suggestion: Dict[str, Any], model: str) -> None:
        """Add a suggestion to this group."""
        self.suggestions.append(suggestion)
        if model not in self.models:
            self.models.append(model)
        self._recalculate_priority()

    def _recalculate_priority(self) -> None:
        """Recalculate priority based on model consensus and importance."""
        base_score = len(self.models)

        importance_weights = {"high": 3, "medium": 2, "low": 1}
        for suggestion in self.suggestions:
            importance = suggestion.get("importance", "medium")
            base_score += importance_weights.get(importance, 1)

        self.priority_score = base_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category,
            "theme": self.theme,
            "suggestions": self.suggestions,
            "models": self.models,
            "priority_score": self.priority_score,
            "validation_status": self.validation_status,
            "validation_reason": self.validation_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuggestionGroup":
        """Create from dictionary."""
        group = cls(data["category"], data["theme"])
        group.suggestions = data.get("suggestions", [])
        group.models = data.get("models", [])
        group.priority_score = data.get("priority_score", 0.0)
        group.validation_status = data.get("validation_status")
        group.validation_reason = data.get("validation_reason")
        return group


def extract_suggestions_from_review(
    review_text: str,
    model_name: str
) -> List[Dict[str, Any]]:
    """
    Extract structured suggestions from a model's review text.

    Args:
        review_text: Raw review output from a model
        model_name: Name of the model that produced the review

    Returns:
        List of suggestion dictionaries
    """
    suggestions = []

    # Try to extract JSON if present
    json_match = re.search(r'```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```', review_text)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, list):
                for item in parsed:
                    item["source_model"] = model_name
                    suggestions.append(item)
                return suggestions
        except json.JSONDecodeError:
            pass

    # Fall back to pattern-based extraction
    suggestion_patterns = [
        r'(?:Suggestion|Issue|Problem|Concern|Improvement)[\s:]+(.+?)(?=\n\n|\n-|\n\d+\.|\Z)',
        r'-\s+(.+?)(?=\n-|\n\n|\Z)',
        r'\d+\.\s+(.+?)(?=\n\d+\.|\n\n|\Z)',
    ]

    for pattern in suggestion_patterns:
        matches = re.findall(pattern, review_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            text = match.strip()
            if len(text) > 20:  # Filter out too-short matches
                suggestions.append({
                    "title": text[:100] + "..." if len(text) > 100 else text,
                    "desc": text,
                    "importance": _infer_importance(text),
                    "type": _infer_type(text),
                    "source_model": model_name,
                })

    return suggestions


def _infer_importance(text: str) -> str:
    """Infer importance level from suggestion text."""
    text_lower = text.lower()

    high_indicators = ["critical", "must", "required", "security", "bug", "error", "fail"]
    low_indicators = ["minor", "optional", "consider", "could", "nice to have", "style"]

    for indicator in high_indicators:
        if indicator in text_lower:
            return "high"

    for indicator in low_indicators:
        if indicator in text_lower:
            return "low"

    return "medium"


def _infer_type(text: str) -> str:
    """Infer suggestion type from text."""
    text_lower = text.lower()

    type_indicators = {
        "bug": ["bug", "error", "fix", "broken", "incorrect"],
        "missing": ["missing", "add", "include", "need"],
        "improvement": ["improve", "enhance", "better", "optimize"],
        "style": ["style", "format", "naming", "convention"],
        "scope": ["scope", "out of scope", "beyond", "different"],
    }

    for stype, indicators in type_indicators.items():
        for indicator in indicators:
            if indicator in text_lower:
                return stype

    return "improvement"


def compute_similarity(s1: Dict[str, Any], s2: Dict[str, Any]) -> float:
    """
    Compute similarity score between two suggestions.

    Args:
        s1: First suggestion
        s2: Second suggestion

    Returns:
        Similarity score between 0 and 1
    """
    title1 = s1.get("title", "").lower()
    title2 = s2.get("title", "").lower()
    desc1 = s1.get("desc", "").lower()
    desc2 = s2.get("desc", "").lower()

    # Word overlap for titles
    words1 = set(title1.split())
    words2 = set(title2.split())
    if words1 and words2:
        title_similarity = len(words1 & words2) / max(len(words1), len(words2))
    else:
        title_similarity = 0

    # Word overlap for descriptions
    words1 = set(desc1.split())
    words2 = set(desc2.split())
    if words1 and words2:
        desc_similarity = len(words1 & words2) / max(len(words1), len(words2))
    else:
        desc_similarity = 0

    # Type match bonus
    type_bonus = 0.2 if s1.get("type") == s2.get("type") else 0

    return 0.3 * title_similarity + 0.5 * desc_similarity + type_bonus


def group_similar_suggestions(
    suggestions: List[Dict[str, Any]],
    similarity_threshold: float = 0.4
) -> List[SuggestionGroup]:
    """
    Group similar suggestions together.

    Args:
        suggestions: List of all suggestions from all models
        similarity_threshold: Minimum similarity to group suggestions

    Returns:
        List of SuggestionGroup objects
    """
    groups: List[SuggestionGroup] = []
    assigned = set()

    # Sort by importance (HIGH first) for better grouping, then by a stable
    # content signature so the seed order is a pure function of the suggestion
    # *set* — never of input order.
    #
    # This greedy clusterer is order-sensitive: the first suggestion of a
    # cluster seeds the group and ties go to the earliest-seen group. With an
    # importance-only key, ``sorted`` is stable and falls back to input
    # position, so the same suggestions arriving in a different order (e.g.
    # ``results.items()`` order in the live review vs. ``glob`` order during
    # ``--reaggregate``) can split/merge differently — producing a different
    # group count and different content hashes. That desynchronizes validation
    # (tasked against the original grouping) from the re-grouped output, which
    # then mis-joins by stale index. A deterministic secondary key makes
    # grouping reproducible so both paths emit identical group_hashes.
    importance_order = {"high": 0, "medium": 1, "low": 2}

    def _stable_sort_key(item: Tuple[int, Dict[str, Any]]):
        _, s = item
        return (
            importance_order.get(s.get("importance", "medium"), 1),
            str(s.get("file", "")),
            str(s.get("line_range", s.get("line", ""))),
            str(s.get("title", "")),
            str(s.get("desc", "")),
            str(s.get("source_model", s.get("model", ""))),
        )

    sorted_suggestions = sorted(enumerate(suggestions), key=_stable_sort_key)

    for idx, suggestion in sorted_suggestions:
        if idx in assigned:
            continue

        # Find or create a group
        best_group = None
        best_similarity = similarity_threshold

        for group in groups:
            for existing in group.suggestions:
                sim = compute_similarity(suggestion, existing)
                if sim > best_similarity:
                    best_similarity = sim
                    best_group = group

        if best_group:
            best_group.add_suggestion(suggestion, suggestion.get("source_model", "unknown"))
        else:
            # Create new group
            group = SuggestionGroup(
                category=suggestion.get("type", "improvement"),
                theme=suggestion.get("title", "Untitled")[:50]
            )
            group.add_suggestion(suggestion, suggestion.get("source_model", "unknown"))
            groups.append(group)

        assigned.add(idx)

    # Sort groups by priority
    groups.sort(key=lambda g: g.priority_score, reverse=True)

    return groups


def deduplicate_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate suggestions.

    Args:
        suggestions: List of suggestions

    Returns:
        Deduplicated list
    """
    seen_hashes = set()
    unique = []

    for suggestion in suggestions:
        # Create a hash from title and description
        hash_str = f"{suggestion.get('title', '')}|{suggestion.get('desc', '')}"
        if hash_str not in seen_hashes:
            seen_hashes.add(hash_str)
            unique.append(suggestion)

    return unique


def merge_suggestions_by_model(
    reviews: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Extract and merge suggestions from multiple model reviews.

    Args:
        reviews: Dictionary mapping model names to their review text

    Returns:
        Merged list of suggestions
    """
    all_suggestions = []

    for model_name, review_text in reviews.items():
        suggestions = extract_suggestions_from_review(review_text, model_name)
        all_suggestions.extend(suggestions)

    return deduplicate_suggestions(all_suggestions)


def filter_by_importance(
    groups: List[SuggestionGroup],
    min_importance: str = "low"
) -> List[SuggestionGroup]:
    """
    Filter suggestion groups by minimum importance.

    Args:
        groups: List of suggestion groups
        min_importance: Minimum importance level ("low", "medium", "high")

    Returns:
        Filtered list of groups
    """
    importance_levels = {"low": 0, "medium": 1, "high": 2}
    min_level = importance_levels.get(min_importance, 0)

    def group_meets_threshold(group: SuggestionGroup) -> bool:
        for suggestion in group.suggestions:
            importance = suggestion.get("importance", "medium")
            if importance_levels.get(importance, 1) >= min_level:
                return True
        return False

    return [g for g in groups if group_meets_threshold(g)]


def export_groups_to_json(groups: List[SuggestionGroup]) -> str:
    """Export suggestion groups to JSON string."""
    return json.dumps([g.to_dict() for g in groups], indent=2)


def import_groups_from_json(json_str: str) -> List[SuggestionGroup]:
    """Import suggestion groups from JSON string."""
    data = json.loads(json_str)
    return [SuggestionGroup.from_dict(item) for item in data]
