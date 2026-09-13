"""Tests for suggestion processor module."""

import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.suggestion_processor import (
    DEFAULT_SIMILARITY_THRESHOLD,
    SuggestionGroup,
    extract_suggestions_from_review,
    compute_similarity,
    compute_section_weights,
    compute_token_weights,
    group_similar_suggestions,
    deduplicate_suggestions,
    merge_suggestions_by_model,
    normalize_tokens,
    filter_by_importance,
    summarize_theme,
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

    def test_matching_type_cannot_carry_a_merge(self):
        """A shared ``type`` is a tiebreak, never evidence.

        ``type`` is model-assigned and noisy — three reviewers call the same
        finding "clarification", "addition" and "modification". The previous
        score handed a matching type a flat 0.2 against a 0.4 threshold, so
        two suggestions with a matching type and one accidental word in
        common were already halfway to a merge.
        """
        s1 = {"title": "Particle trail scope unbounded",
              "desc": "No capacity, lifetime, or emission rate for the trail.",
              "type": "clarification"}
        s2 = {"title": "Screen shake conflicts with camera lerp",
              "desc": "Jitter written into cam.position feeds back into the next lerp.",
              "type": "clarification"}

        assert compute_similarity(s1, s2) < 0.1

    def test_shared_boilerplate_is_not_agreement(self):
        """Review boilerplate and stopwords carry no similarity.

        Every suggestion in every review says "the plan does not specify" and
        "add a section that defines"; a score built on raw ``.split()`` word
        overlap read that shared genre vocabulary as agreement.
        """
        s1 = {"title": "Document the retry budget",
              "desc": "The plan does not specify the retry budget. Add a section that defines it.",
              "type": "addition"}
        s2 = {"title": "Document the favicon size",
              "desc": "The plan does not specify the favicon size. Add a section that defines it.",
              "type": "addition"}

        assert compute_similarity(s1, s2) < 0.2

    def test_token_weights_discount_ubiquitous_words(self):
        """Rarity weighting is what replaces a per-project stopword list."""
        suggestions = [
            {"title": f"Boost finding {n}", "desc": "boost boost boost"}
            for n in range(11)
        ] + [{"title": "Boost camera lerp", "desc": "boost lerp"}]
        weights = compute_token_weights(suggestions)

        # "boost" is in every suggestion, "lerp" in one of twelve: the rare
        # word is worth several of the ubiquitous one.
        assert weights["boost"] < weights["lerp"] / 2


class TestNormalizeTokens:
    """Tests for token normalization — the input every score is built on."""

    def test_splits_identifiers_inside_and_outside_code_spans(self):
        tokens = normalize_tokens(
            "Applying a temporary maxSpeedMultiplier through Physics.integrate() "
            "while `keys.boost` is held"
        )
        assert "max" in tokens and "speed" in tokens and "multiplier" in tokens
        assert "physic" in tokens and "integrat" in tokens
        # A backticked identifier is the most specific thing a reviewer can
        # name, so it is split like any other identifier rather than dropped.
        assert "keys" in tokens and "boost" in tokens

    def test_a_code_span_cannot_weld_two_words_together(self):
        assert normalize_tokens("reset`Car`state") == normalize_tokens(
            "reset Car state"
        )

    def test_one_model_backticks_an_identifier_and_another_does_not(self):
        """The pair the dropped-code-span rule used to keep apart."""
        quoted = {
            "title": "Car.reset boost reset not tasked",
            "desc": "Task 2 does not require resetting boost in `Car.reset()`.",
        }
        bare = {
            "title": "Game-state edge cases not addressed",
            "desc": "State that Car.reset() clears all boost fields on restart.",
        }
        shared = set(normalize_tokens(quoted["desc"])) & set(
            normalize_tokens(bare["desc"])
        )
        assert {"car", "reset"} <= shared

    def test_stems_word_families_together(self):
        for surface in ("charge", "charges", "charging", "charged"):
            assert normalize_tokens(surface) == normalize_tokens("charge"), surface
        assert normalize_tokens("dragged") == normalize_tokens("drag")

    def test_drops_stopwords_boilerplate_and_short_words(self):
        assert normalize_tokens("The plan does not specify the task") == []
        assert normalize_tokens("an ad") == []


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


#: Eight suggestions from a real three-reviewer plan review (a racing game's
#: "turbo boost mode" plan), verbatim. They are the two cases the old scorer
#: got backwards, and they are here because synthetic near-duplicates cannot
#: reproduce either: real reviewers restate the same finding in disjoint
#: words, and unrelated findings share the plan's own vocabulary.
#:
#: Under the old score — 0.3·title + 0.5·desc word overlap + a flat 0.2 for a
#: matching ``type``, threshold 0.4, greedy single-link — the first three
#: below landed in ONE group at 0.414 (0.214 of it text, the rest the type
#: bonus; the third joined by chaining off the second at 0.267), while the
#: edge-trigger pair scored 0.195 and the HUD trio 0.181–0.195 and stayed
#: apart.
_REAL_ACTIVATION_AND_HUD = [
    {
        "title": "Boost activation semantics undefined",
        "desc": "The plan does not specify tap vs hold, edge-trigger vs level-trigger, "
                "or whether holding the key after cooldown immediately re-fires. Current "
                "code level-triggers each physics step while the key is held. Document "
                "intended behavior (recommend edge-trigger on keydown or single activation "
                "per press) to avoid accidental multi-burn with maxCharges > 1.",
        "importance": "MEDIUM",
        "reference": "### 3. Input Handling / ### 4. Physics Integration",
        "type": "clarification",
        "source_model": "cursor-agent_composer-2_5",
    },
    {
        "title": "Activation is level-triggered, giving permanent boost",
        "desc": "The plan specifies `Input.boost` only as a 'boolean (0/1) for current "
                "frame' — a held-key state, not a press event. Combined with the natural "
                "activation rule (`if (Input.boost && charges > 0 && !active)`), simply "
                "holding the key re-fires boost the instant a charge returns, producing "
                "continuous boost with zero player skill and defeating the 'limited "
                "resource' premise in the Overview. Specify edge-triggered activation: "
                "latch a `pressed` flag on the keydown transition, consume it in the "
                "physics step, and clear it after read.",
        "importance": "HIGH",
        "reference": "### 3. Input Handling",
        "type": "addition",
        "source_model": "claude-code_opus",
    },
    {
        "title": "Boost activation needs press-edge semantics",
        "desc": "The plan treats Input.boost as a current-frame boolean, which risks "
                "repeatedly reactivating boost while the key is held once charges return. "
                "Specify an edge-triggered activation, for example by tracking "
                "wasBoostDown or handling keydown only when !repeat, so one key press "
                "consumes at most one charge.",
        "importance": "HIGH",
        "reference": "### 3. Input Handling",
        "type": "modification",
        "source_model": "cursor-agent_gpt-5_5-extra-high",
    },
    {
        "title": "Opponent boost scope undefined",
        "desc": "Task 8 is optional with no decision criteria, success metric, or AI "
                "design (random chance vs scripted straights). Either cut it from v1 scope "
                "with a follow-up note or define when opponents boost, how it affects "
                "rubber-banding, and that it must not confuse the existing 'rubber-band "
                "boost' terminology in opponent AI tests.",
        "importance": "LOW",
        "reference": "### 8. Opponent Parity (Optional)",
        "type": "clarification",
        "source_model": "cursor-agent_composer-2_5",
    },
    {
        "title": "Opponent parity is left as unresolved scope",
        "desc": "The plan leaves opponent boost as optional without deciding whether "
                "player-only boost is an intentional balance advantage. Either declare "
                "opponents out of scope for this feature, or define deterministic AI boost "
                "rules, tuning, and tests so opponent behavior stays reproducible.",
        "importance": "MEDIUM",
        "reference": "### 8. Opponent Parity (Optional)",
        "type": "clarification",
        "source_model": "cursor-agent_gpt-5_5-extra-high",
    },
    {
        "title": "HUD meter design underspecified",
        "desc": "Task 6 mentions #boost-meter and charge/cooldown display but gives no "
                "placement, layout, or behavior for maxCharges > 1, active glow, or "
                "cooldown-as-bar vs numeric timer. prevCharges is in the state schema with "
                "no HUD task describing its use. Add wireframe/placement (e.g. bottom-left "
                "near speedo) and rules for single vs multi-charge UI.",
        "importance": "MEDIUM",
        "reference": "### 6. HUD Updates",
        "type": "addition",
        "source_model": "cursor-agent_composer-2_5",
    },
    {
        "title": "HUD requirements lack DOM and state details",
        "desc": "The plan says to add #boost-meter but does not specify the HTML "
                "structure, CSS placement, cached HUD fields, or display mode for charges "
                "versus cooldown progress. Add the exact HUD elements, text/progress "
                "states, active styling, disabled styling, and how it behaves in time-trial "
                "versus race mode.",
        "importance": "MEDIUM",
        "reference": "### 6. HUD Updates",
        "type": "addition",
        "source_model": "cursor-agent_gpt-5_5-extra-high",
    },
    {
        "title": "HUD task lacks markup, states, and caching convention",
        "desc": "Task 6 says to add `#boost-meter` but specifies no markup, placement "
                "relative to the existing HUD nodes, or styling; no decision between a "
                "charge pip and a cooldown bar for a 1-charge design; no behavior for "
                "raceMode vs classic time-trial; nothing about what shows when maxCharges "
                "is 0; and no mention of HUD's write-caching convention (`_txt`/`_dsp`) "
                "that every other readout follows.",
        "importance": "MEDIUM",
        "reference": "### 6. HUD Updates / ### 2. Car Module State",
        "type": "clarification",
        "source_model": "claude-code_opus",
    },
]


class TestRealWorldGrouping:
    """Grouping against suggestions a real multi-model review produced.

    The whole 56-suggestion review regroups from 46 groups (7 multi-member,
    one of them incoherent) to 28 (15 multi-member, every one of them about
    one thing). These tests pin the two cases that named the bug.
    """

    @staticmethod
    def _clusters(groups):
        return {
            frozenset(s["title"] for s in g.suggestions) for g in groups
        }

    def test_activation_and_opponent_findings_separate(self):
        """The incoherent group splits, and both halves land where they belong.

        "Boost activation semantics undefined" is about edge- vs
        level-triggered activation and belongs with the two findings that say
        exactly that. "Opponent boost scope undefined" is about whether
        opponents boost at all; it shares the word "undefined", the word
        "boost" (which 43 of the review's 56 suggestions use) and a
        ``clarification`` type, and nothing else.
        """
        groups = group_similar_suggestions(_REAL_ACTIVATION_AND_HUD)
        clusters = self._clusters(groups)

        assert frozenset({
            "Boost activation semantics undefined",
            "Activation is level-triggered, giving permanent boost",
            "Boost activation needs press-edge semantics",
        }) in clusters

        assert frozenset({
            "Opponent boost scope undefined",
            "Opponent parity is left as unresolved scope",
        }) in clusters

    def test_hud_findings_merge(self):
        """Three reviewers describing the same missing HUD spec, in three vocabularies."""
        groups = group_similar_suggestions(_REAL_ACTIVATION_AND_HUD)

        assert frozenset({
            "HUD meter design underspecified",
            "HUD requirements lack DOM and state details",
            "HUD task lacks markup, states, and caching convention",
        }) in self._clusters(groups)

    def test_three_coherent_groups_and_nothing_else(self):
        """No chaining: eight suggestions, three groups, no strays."""
        groups = group_similar_suggestions(_REAL_ACTIVATION_AND_HUD)

        assert len(groups) == 3
        assert sorted(len(g.suggestions) for g in groups) == [2, 3, 3]

    def test_edge_trigger_pair_outscores_the_accidental_pair(self):
        """The two scores the old arithmetic had in the wrong order.

        Old: the accidental pair 0.414 (merged), the real pair 0.195 (split).
        """
        by_title = {s["title"]: s for s in _REAL_ACTIVATION_AND_HUD}
        weights = compute_token_weights(_REAL_ACTIVATION_AND_HUD)

        real_pair = compute_similarity(
            by_title["Activation is level-triggered, giving permanent boost"],
            by_title["Boost activation needs press-edge semantics"],
            weights,
        )
        accidental_pair = compute_similarity(
            by_title["Boost activation semantics undefined"],
            by_title["Opponent boost scope undefined"],
            weights,
        )

        assert real_pair > accidental_pair
        assert accidental_pair < DEFAULT_SIMILARITY_THRESHOLD <= real_pair

    def test_grouping_is_order_independent_on_real_data(self):
        """The reaggregate-identity property, on input that actually clusters."""
        import random

        baseline = self._clusters(group_similar_suggestions(_REAL_ACTIVATION_AND_HUD))
        shuffler = random.Random(20260911)
        for _ in range(10):
            shuffled = list(_REAL_ACTIVATION_AND_HUD)
            shuffler.shuffle(shuffled)
            assert self._clusters(group_similar_suggestions(shuffled)) == baseline


_TURBO_BOOST = json.loads(
    (Path(__file__).parent / "fixtures" / "review_plan_turbo_boost.json").read_text(
        encoding="utf-8"
    )
)


class TestTurboBoostReview:
    """A second real review, audited group by group (2026-09-13).

    Fifty-five suggestions from three reviewers over an eight-task plan. An
    audit of the shipped 32-group output found fourteen of the sixteen
    multi-member groups coherent, one mis-merge and a handful of paraphrase
    pairs left apart; these tests pin the ones the current rules fix, and the
    coherent groups they had to leave alone.
    """

    @staticmethod
    def _clusters(groups):
        return {frozenset(s["title"] for s in g.suggestions) for g in groups}

    def test_the_whole_review_regroups_to_thirty(self):
        groups = group_similar_suggestions(_TURBO_BOOST)
        assert len(_TURBO_BOOST) == 55
        assert len(groups) == 30
        assert sum(1 for g in groups if len(g.suggestions) > 1) == 16

    def test_the_particles_finding_leaves_the_nojuice_group(self):
        """The one mis-merge the audit found, and where the stray belongs.

        "Visual feedback requirements contradict optional particles" asks
        whether particles are in v1. It was filed with two findings about
        gating effects behind ``?nojuice`` — it shares "visual" with one of
        their titles and nothing else — while the two findings that are also
        about whether particles ship sat in their own group.
        """
        clusters = self._clusters(group_similar_suggestions(_TURBO_BOOST))

        assert frozenset({
            "Particle trail ship criteria missing",
            "Particle trail under-specified and likely over-scope",
            "Visual feedback requirements contradict optional particles",
        }) in clusters

    def test_the_juice_convention_finding_joins_the_nojuice_group(self):
        """Three models on one question: must boost cosmetics respect ?nojuice."""
        clusters = self._clusters(group_similar_suggestions(_TURBO_BOOST))

        assert frozenset({
            "No ?nojuice handling for boost FX",
            "Visual effects ignore nojuice gating",
            "Cosmetic boost values bypass the juice convention",
        }) in clusters

    def test_the_car_reset_findings_merge(self):
        """One model backticked ``Car.reset()``; the other did not.

        The fifth bullet of "Game-state and track edge cases not addressed" is
        "that Car.reset() clears all boost fields" — the whole subject of
        "Car.reset boost reset not tasked".
        """
        clusters = self._clusters(group_similar_suggestions(_TURBO_BOOST))

        assert frozenset({
            "Game-state and track edge cases not addressed",
            "Pause and reset behavior is underspecified",
            "Car.reset boost reset not tasked",
        }) in clusters

    def test_the_coherent_groups_are_left_alone(self):
        """Every group the audit passed still comes out whole.

        The lap-hook pair is the one to watch: "Boosted lap times invalidate
        persisted best lap" and "First-lap charge grant unspecified" both share
        its section and much of its vocabulary, and a threshold a hair lower
        pulls them in.
        """
        clusters = self._clusters(group_similar_suggestions(_TURBO_BOOST))

        for coherent in (
            {"Activation edge vs. held-key semantics left undefined",
             "Activation needs edge-trigger semantics",
             "Activation semantics not defined"},
            {"Boost key binding still ambiguous",
             "Shift key cannot reach the input handler",
             "Shift key conflicts with existing input flow"},
            {"Opponent balance impact is unresolved",
             "Opponent parity left as an open coin flip",
             "Optional opponent boost unresolved"},
            {"Recharge model left undecided",
             "Recharge model never actually decided",
             "Recharge model remains undecided"},
            {"Acceptance criteria are missing",
             "No acceptance criteria or definition of determinism"},
            {"Timing hook integration may break existing behavior",
             "Wrong lap-completion hook named"},
        ):
            assert frozenset(coherent) in clusters, sorted(coherent)

    def test_grouping_is_order_independent_on_the_whole_review(self):
        import random

        baseline = self._clusters(group_similar_suggestions(_TURBO_BOOST))
        shuffler = random.Random(20260913)
        for _ in range(5):
            shuffled = list(_TURBO_BOOST)
            shuffler.shuffle(shuffled)
            assert self._clusters(group_similar_suggestions(shuffled)) == baseline


class TestSectionRarity:
    """The ``reference`` tiebreak is worth what the section is worth."""

    def test_a_rare_section_outweighs_a_crowded_one(self):
        crowded = [
            {"title": f"Finding {i}", "desc": f"Body {i}", "reference": "### 4. Physics"}
            for i in range(10)
        ]
        rare = [
            {"title": "Finding A", "desc": "Body A", "reference": "### 6. HUD"},
            {"title": "Finding B", "desc": "Body B", "reference": "### 6. HUD"},
        ]
        weights = compute_section_weights(crowded + rare)

        assert weights["6. hud"] > weights["4. physics"]
        # Same texts, same sections shared: only the rarity differs.
        assert compute_similarity(rare[0], rare[1], {}, weights) > compute_similarity(
            crowded[0], crowded[1], {}, weights
        )

    def test_a_bare_slash_inside_a_heading_is_not_a_separator(self):
        one = {"reference": "### 5. Visual/Audio Effects (Juice)"}
        weights = compute_section_weights([one, dict(one)])

        assert list(weights) == ["5. visual/audio effects (juice)"]

    def test_a_delimited_slash_still_names_two_sections(self):
        composite = {"reference": "## Design Requirements / ### 7. Lap Integration"}

        assert set(compute_section_weights([composite])) == {
            "design requirements",
            "7. lap integration",
        }


class TestSummarizeTheme:
    """A theme is a label a reader can use, not a title cut mid-word."""

    def test_a_short_title_is_unchanged(self):
        assert summarize_theme("Boost key binding still ambiguous") == (
            "Boost key binding still ambiguous"
        )

    def test_a_title_that_exactly_fits_keeps_its_last_character(self):
        title = "T003 does not name the lockout threshold under test"
        assert summarize_theme(title) == title

    def test_a_long_title_is_cut_at_a_word_boundary(self):
        theme = summarize_theme(
            "Activation edge vs. held-key semantics left undefined for opponents"
        )

        assert theme == "Activation edge vs. held-key semantics left undefined for…"
        assert not theme.rstrip("…").endswith(" ")
        # The old hard slice cut "undefined" in half at 50 characters.
        assert "undefi…" not in theme

    def test_an_empty_title_falls_back(self):
        assert summarize_theme("") == "Untitled"
        assert summarize_theme(None) == "Untitled"

    def test_a_title_with_no_usable_boundary_is_still_bounded(self):
        theme = summarize_theme("x" * 200)

        assert theme.endswith("…")
        assert len(theme) <= 61


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


class TestThemeIsDisplayNotIdentity:
    """`summarize_theme` changed what a theme looks like, not what a group is.

    `state_manager.generate_group_id` hashes the theme at the legacy
    50-character cut, so a group whose theme grew from `title[:50]` to the
    word-boundary form keeps the id the container goldens bake in
    (`coverage_suggestions.json` → `ef911087fcf05304`, measured 2026-09-13).
    """

    def test_a_longer_theme_hashes_like_its_fifty_char_cut(self):
        from utils.state_manager import generate_group_id

        title = "T003 does not name the lockout threshold under test"  # 51 chars
        assert len(title) == 51
        suggestions = [{"type": "addition", "reference": "T003", "desc": "x"}]
        new_style = {"theme": title, "suggestions": suggestions}
        legacy = {"theme": title[:50], "suggestions": suggestions}
        assert generate_group_id(new_style) == generate_group_id(legacy)

    def test_a_theme_change_inside_the_cut_still_changes_the_id(self):
        from utils.state_manager import generate_group_id

        suggestions = [{"type": "addition", "reference": "T003", "desc": "x"}]
        a = {"theme": "Recharge model left undecided", "suggestions": suggestions}
        b = {"theme": "Recharge model decided", "suggestions": suggestions}
        assert generate_group_id(a) != generate_group_id(b)
