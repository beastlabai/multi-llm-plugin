#!/usr/bin/env python3
"""Per-batch validation context: cited plan sections and per-batch code windows.

A validation batch carries the plan sections its groups cite and, on
review-code, the windows its own issues cite, instead of the round-wide blocks
every batch used to repeat (docs/development/delegation.md, "Cited plan
sections"). These tests pin the scoping rule and its fallbacks, the prompt it
renders, and the contracts around it that must not move: batches, verdict
parsing, the resume hashes and snapshot re-verification.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro  # noqa: E402
from utils import validation_context as vc  # noqa: E402
from utils import validation_sections as vs  # noqa: E402
from utils.subagent_tasks import content_hash  # noqa: E402
from utils.validation import (  # noqa: E402
    PLAN_SNAPSHOT_NAME,
    VALIDATION_CONTEXT_LIMIT,
    ValidationContextError,
    bind_document_file,
    merge_batched_validation_results,
    prepare_batched_validation_tasks,
    verify_context_sources,
)


def _filler(word: str, chars: int = 2400) -> str:
    """A paragraph of distinct text, so a section's body is recognisable."""
    line = f"{word} " * 12
    return "\n".join(line.strip() for _ in range(chars // len(line) + 1)) + "\n"


PLAN = (
    "# Fixture plan\n\nPREAMBLE the goal of this plan.\n\n"
    "## 1. Uploads\n\nUPLOADSINTRO framing for every upload section.\n\n"
    "### 1.1 Retry\n\n" + _filler("RETRYBODY")
    + "\n### 1.2 Backoff\n\n" + _filler("BACKOFFBODY")
    + "\n## 2. Config, paths and defaults\n\n" + _filler("CONFIGBODY")
    + "\n### 2.1 Location\n\n" + _filler("LOCATIONBODY")
    + "\n## 3. Testing\n\n" + _filler("TESTINGBODY")
    + "\n```bash\n# not a heading\n```\n"
    + "\n## 4. Rollout\n\n" + _filler("ROLLOUTBODY")
)


def _group(reference, importance="HIGH", theme=None):
    suggestion = {
        "title": f"About {reference}", "desc": "A claim.", "importance": importance,
        "type": "modification",
    }
    if reference is not None:
        suggestion["reference"] = reference
    return {
        "theme": theme or f"theme {reference}", "category": "test",
        "models": ["m"], "suggestions": [suggestion],
    }


def _records(*references):
    return [
        {"group_index": i, "suggestions": [{"reference": ref}]}
        for i, ref in enumerate(references)
    ]


@pytest.fixture
def cited_env(monkeypatch):
    """The opt-in: plan sections are cited only under ``cited``."""
    monkeypatch.setenv(vs.CITED_CONTEXT_ENV, vs.CONTEXT_CITED)


@pytest.fixture
def default_env(monkeypatch):
    monkeypatch.delenv(vs.CITED_CONTEXT_ENV, raising=False)


def _materialize(tmp_path, groups, *, name="run", plan=PLAN, **kwargs):
    root = tmp_path / name
    phase = root / "review-plan"
    phase.mkdir(parents=True)
    plan_path = root / "plan.md"
    plan_path.write_text(plan, encoding="utf-8")
    document = bind_document_file(str(plan_path), str(phase))
    result = prepare_batched_validation_tasks(
        groups=groups, context=document.text, output_dir=str(phase),
        plan_file=str(plan_path), context_document=document, **kwargs,
    )
    return result, phase


def _plan_block(prompt):
    return prompt.split("## Context (Plan content)\n\n", 1)[1].split(
        "\n## Suggestion groups to validate", 1
    )[0]


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------

class TestReferenceParts:
    def test_slash_and_semicolon_split(self):
        assert vs.split_reference("## 2 / ### 3.1; Stage A") == ["## 2", "### 3.1", "Stage A"]

    def test_a_comma_inside_a_heading_stays(self):
        ref = "### Stage A — adapter, registry, config"
        assert vs.split_reference(ref) == [ref]

    def test_a_comma_before_another_reference_splits(self):
        assert vs.split_reference("## 2, ## 4") == ["## 2", "## 4"]
        assert vs.split_reference("§2.3, §4") == ["§2.3", "§4"]

    def test_empty_and_list_references(self):
        assert vs.split_reference("") == []
        assert vs.split_reference(None) == []
        assert vs.split_reference(["## 1", "## 2"]) == ["## 1", "## 2"]


class TestResolvePart:
    HEADINGS = vs.markdown_headings(
        "# Title\n"
        "## 1. Intro\n### 1.1 Scope\n"
        "## 2. Uploads\n### 2.3 Retry\n### 2.4 Backoff\n### 2.5 Limits\n"
        "## 3. Retry Design\n"
        "### Stage A — adapter, registry\n"
        "### T4.2 Hooks\n"
        "## D5 — The surfaced reason\n### Fix\n"
        "### 9.2 Open (P0 of §5)\n"
        "## 10. Appendix\n"
        "## Testing strategy\n"
        "```\n## 7. Inside a fence\n```\n"
    )

    def names(self, part):
        return [self.HEADINGS[i].line for i in vs.resolve_part(part, self.HEADINGS)]

    def test_heading_text_and_its_word_prefix(self):
        assert self.names("## 2. Uploads") == ["## 2. Uploads"]
        # `## 1` names section 1 and its subsections, never `## 10`.
        assert self.names("## 1") == ["## 1. Intro", "### 1.1 Scope"]

    def test_section_key(self):
        assert self.names("### 9.2 P5") == ["### 9.2 Open (P0 of §5)"]
        assert self.names("Stage A fixtures row") == ["### Stage A — adapter, registry"]
        assert self.names("T4.2") == ["### T4.2 Hooks"]
        assert self.names("§3") == ["## 3. Retry Design"]

    def test_range(self):
        assert self.names("§2.3-2.5") == ["### 2.3 Retry", "### 2.4 Backoff", "### 2.5 Limits"]

    def test_title_without_its_number(self):
        assert self.names("Retry Design") == ["## 3. Retry Design"]

    def test_a_heading_title_at_the_start_of_the_part(self):
        assert self.names("Testing strategy table") == ["## Testing strategy"]

    def test_a_parenthesized_key_names_its_section(self):
        assert self.names("### Fix (D5)") == ["## D5 — The surfaced reason"]

    def test_unresolvable_and_fenced(self):
        assert self.names("the whole plan") == []
        assert self.names("## 7. Inside a fence") == []

    def test_an_unqualified_title_names_every_heading_with_it(self):
        """``Testing`` with ``## Testing`` and ``## 3. Testing``: both, never
        the first by rule order."""
        headings = vs.markdown_headings(
            "## Testing\n\na\n\n## 2. Other\n\nb\n\n## 3. Testing\n\nc\n"
        )
        found = [headings[i].line for i in vs.resolve_part("Testing", headings)]
        assert found == ["## Testing", "## 3. Testing"]
        # A numbered reference stays narrow.
        found = [headings[i].line for i in vs.resolve_part("§3", headings)]
        assert found == ["## 3. Testing"]


class TestHeadingScan:
    def test_a_longer_fence_is_not_closed_by_a_shorter_one(self):
        """A four-backtick example showing a three-backtick block stays one
        fence: nothing inside it is a heading."""
        text = (
            "## Example\n\n````md\n```bash\n## Not a heading\n```\n## Nor this\n````\n\n"
            "CONSTRAINT after the example\n\n## Next\n"
        )
        assert [h.line for h in vs.markdown_headings(text)] == ["## Example", "## Next"]

    def test_a_fence_closes_only_on_its_own_character(self):
        text = "~~~\n```\n## Inside\n~~~\n## Outside\n"
        assert [h.line for h in vs.markdown_headings(text)] == ["## Outside"]

    def test_a_closing_line_with_text_does_not_close(self):
        text = "```\n```python\n## Inside\n```\n## Outside\n"
        assert [h.line for h in vs.markdown_headings(text)] == ["## Outside"]

    def test_inline_code_is_not_a_fence(self):
        text = "```x``` is inline\n## Heading\n"
        assert [h.line for h in vs.markdown_headings(text)] == ["## Heading"]

    def test_the_cited_section_keeps_what_follows_a_nested_fence(self):
        """The review's reproduction: the nested block moved a section
        boundary and the constraint after it disappeared from the selection."""
        plan = vs.CitedPlan.parse(
            "# P\n\n## A\n\n" + _filler("ABODY", 6000)
            + "## Example\n\n````md\n```markdown\n## Inner\n```\n````\n\n"
            + "LATERCONSTRAINT never retry a 4xx.\n\n"
            + "## B\n\n" + _filler("BBODY", 9000)
        )
        selection = plan.select(_records("## Example"), limit=10**6)
        assert selection.excerpt is not None
        assert "LATERCONSTRAINT" in selection.excerpt
        assert "## Inner\n```\n````" in selection.excerpt

    def test_a_setext_heading_leaves_the_plan_unscoped(self):
        """Its sections end where the ATX scan cannot see: never cited."""
        text = (
            "# P\n\n## A\n\n" + _filler("ABODY", 6000)
            + "Testing\n-------\n\n" + _filler("SETEXT", 6000)
            + "## Z\n\n" + _filler("ZBODY", 9000)
        )
        plan = vs.CitedPlan.parse(text)
        assert plan.unscoped and "setext" in plan.unscoped
        selection = plan.select(_records("## A"), limit=10**6)
        assert selection.excerpt is None and "setext" in selection.reason

    def test_a_rule_after_a_blank_line_is_not_setext(self):
        text = "# P\n\n## A\n\nbody\n\n---\n\n## B\n\nmore\n"
        assert vs.CitedPlan.parse(text).unscoped is None


# ---------------------------------------------------------------------------
# Selecting one batch's sections
# ---------------------------------------------------------------------------

class TestCitedPlanSelect:
    plan = vs.CitedPlan.parse(PLAN)

    def test_a_cited_section_comes_with_the_preamble_and_its_parent_intro(self):
        selection = self.plan.select(_records("### 1.1 Retry"), limit=VALIDATION_CONTEXT_LIMIT)
        excerpt = selection.excerpt
        assert excerpt is not None and selection.reason is None
        assert "PREAMBLE" in excerpt and "UPLOADSINTRO" in excerpt and "RETRYBODY" in excerpt
        for body in ("BACKOFFBODY", "CONFIGBODY", "LOCATIONBODY", "TESTINGBODY", "ROLLOUTBODY"):
            assert body not in excerpt
        assert excerpt.rstrip().endswith(
            "[... not cited by this batch, not shown: ### 1.2 Backoff, "
            "## 2. Config, paths and defaults, ### 2.1 Location (+2 more) ...]"
        )
        assert (selection.parts, selection.resolved) == (1, 1)

    def test_separate_sections_are_joined_by_a_gap_marker_in_plan_order(self):
        one = self.plan.select(_records("## 3. Testing", "### 1.2 Backoff"), limit=10**6)
        two = self.plan.select(_records("### 1.2 Backoff", "## 3. Testing"), limit=10**6)
        assert one.excerpt == two.excerpt
        excerpt = one.excerpt
        assert excerpt.index("BACKOFFBODY") < excerpt.index("TESTINGBODY")
        assert "[... not cited by this batch, not shown: ### 1.1 Retry ...]" in excerpt
        assert "## 2. Config, paths and defaults, ### 2.1 Location ..." in excerpt

    def test_a_group_citing_nothing_resolvable_falls_back(self):
        selection = self.plan.select(
            _records("### 1.1 Retry", "the whole thing"), limit=VALIDATION_CONTEXT_LIMIT
        )
        assert selection.excerpt is None
        assert selection.reason == "group 1 cites a part that names no section of the plan"

    def test_a_group_with_no_reference_falls_back(self):
        records = [{"group_index": 4, "suggestions": [{"title": "t"}]}]
        selection = self.plan.select(records, limit=VALIDATION_CONTEXT_LIMIT)
        assert selection.excerpt is None and "group 4" in selection.reason

    def test_every_part_of_every_reference_must_resolve(self):
        """One resolved part never vouches for another: a batch is cited only
        when every section its suggestions name is shown, else the prompt
        would claim sections it does not carry."""
        selection = self.plan.select(_records("## 3. Testing / ## 9. Missing"), limit=10**6)
        assert selection.excerpt is None
        assert selection.reason == "group 0 cites a part that names no section of the plan"
        assert (selection.parts, selection.resolved) == (2, 1)

    def test_an_unresolved_suggestion_beside_a_resolved_one_falls_back(self):
        records = [{"group_index": 0, "suggestions": [
            {"reference": "### 1.1 Retry"}, {"reference": "the whole plan"},
        ]}]
        selection = self.plan.select(records, limit=10**6)
        assert selection.excerpt is None and "group 0 cites a part" in selection.reason

    def test_a_suggestion_with_no_reference_beside_a_cited_one_falls_back(self):
        records = [{"group_index": 2, "suggestions": [
            {"reference": "### 1.1 Retry"}, {"title": "no reference"},
        ]}]
        selection = self.plan.select(records, limit=10**6)
        assert selection.excerpt is None
        assert selection.reason == "group 2 has a suggestion with no reference"

    def test_an_anchor_reference_that_names_no_heading_falls_back(self):
        """The review's reproduction: ``Uploads / plan.md#retry-policy``
        selected Uploads and dropped the Retry policy section it also cites."""
        plan = vs.CitedPlan.parse(
            "# P\n\nintro\n\n## Uploads\n\n" + _filler("UPLOADS", 4000)
            + "## Retry policy\n\n" + _filler("RETRY", 4000)
            + "## Rollout\n\n" + _filler("ROLLOUT", 12000)
        )
        selection = plan.select(_records("Uploads / plan.md#retry-policy"), limit=10**6)
        assert selection.excerpt is None

    def test_a_short_plan_is_never_cited(self):
        plan = vs.CitedPlan.parse("# T\n\n## 1. A\nbody\n## 2. B\nbody\n")
        selection = plan.select(_records("## 1. A"), limit=VALIDATION_CONTEXT_LIMIT)
        assert selection.excerpt is None and "under" in selection.reason

    def test_a_selection_most_of_the_plan_falls_back(self):
        selection = self.plan.select(
            _records("## 1. Uploads", "## 2. Config", "## 3. Testing"), limit=10**6
        )
        assert selection.excerpt is None
        assert selection.reason == "the cited sections are 60% of the plan or more"

    def test_a_selection_over_the_limit_falls_back(self):
        selection = self.plan.select(_records("### 1.1 Retry"), limit=100)
        assert selection.excerpt is None and "over 100 chars" in selection.reason

    def test_global_sections_come_along_whatever_is_cited(self):
        """The review's reproduction: citing Uploads dropped a Goals section
        that forbids logging credentials. Top-level goal, non-goal, scope,
        constraint, definition and decision sections ride in every cited
        batch; a nested one, or a feature section, does not."""
        plan = vs.CitedPlan.parse(
            "# P\n\nintro\n\n## 1. Goals\n\nGOALS never log credentials.\n\n"
            "## 2. Uploads\n\n" + _filler("UPLOADS", 4000)
            + "### 2.1 Constraints of the upload path\n\n" + _filler("NESTED", 300)
            + "## 3. Non-goals and follow-ups\n\nNONGOALS no resumable uploads.\n\n"
            "## 4. Retry\n\n" + _filler("RETRY", 9000)
            + "## 5. Definitions\n\nDEFINITIONS a part is 8 MiB.\n\n"
            "## 6. Locked design decisions\n\nDECISIONS D1 keep the bucket.\n\n"
            "## 7. Rollout\n\n" + _filler("ROLLOUT", 9000)
        )
        assert [plan.headings[i].line for i in plan.global_sections] == [
            "## 1. Goals", "## 3. Non-goals and follow-ups", "## 5. Definitions",
            "## 6. Locked design decisions",
        ]
        selection = plan.select(_records("## 7. Rollout"), limit=10**6)
        excerpt = selection.excerpt
        assert excerpt is not None
        for marker in ("GOALS never log", "NONGOALS", "DEFINITIONS", "DECISIONS", "ROLLOUT"):
            assert marker in excerpt, marker
        for marker in ("UPLOADS", "NESTED", "RETRY"):
            assert marker not in excerpt, marker

    def test_global_sections_count_towards_the_share(self):
        """Global sections that make the selection most of the plan send the
        batch back to the whole plan."""
        plan = vs.CitedPlan.parse(
            "# P\n\n## Goals\n\n" + _filler("GOALS", 9000)
            + "## Uploads\n\n" + _filler("UPLOADS", 2000)
            + "## Retry\n\n" + _filler("RETRY", 2000)
        )
        selection = plan.select(_records("## Uploads"), limit=10**6)
        assert selection.excerpt is None and "60%" in selection.reason

    def test_no_section_headings_means_no_cited_plan(self):
        assert vs.CitedPlan.parse("x" * 20000) is None
        assert vs.CitedPlan.parse("# Only a title\n\n" + "x" * 20000) is None


class TestSwitchAndSummary:
    def test_the_switch(self, capsys):
        """Plan sections are opt-in (``cited``); per-batch code windows are on
        unless ``full``; an unknown value is reported once and read as
        ``full``."""
        unset, cited, full = {}, {vs.CITED_CONTEXT_ENV: "cited"}, {vs.CITED_CONTEXT_ENV: "full"}
        assert vs.cited_context_enabled(unset) is False
        assert vs.cited_context_enabled(cited) is True
        assert vs.cited_context_enabled(full) is False
        assert vs.batch_code_windows_enabled(unset) is True
        assert vs.batch_code_windows_enabled(cited) is True
        assert vs.batch_code_windows_enabled(full) is False
        assert capsys.readouterr().err == ""
        odd = {vs.CITED_CONTEXT_ENV: "scoped-for-this-test"}
        assert vs.cited_context_enabled(odd) is False
        assert vs.batch_code_windows_enabled(odd) is False
        err = capsys.readouterr().err
        assert err.count("embedding the whole plan") == 1

    def test_the_summary_line(self):
        plan = vs.CitedPlan.parse(PLAN)
        selections = [
            plan.select(_records("### 1.1 Retry"), limit=10**6),
            plan.select(_records("nothing here"), limit=10**6),
        ]
        line = vs.summary_line(selections, len(PLAN))
        assert line.startswith("[VALIDATION_CONTEXT_CITED] plan: 1 of 2 batches carry")
        assert "whole plan in 1 (a group cites a part that names no section of the plan)" in line
        assert line.endswith("1/2 reference parts resolved")

    def test_silent_when_nothing_carried_a_reference(self):
        plan = vs.CitedPlan.parse(PLAN)
        selection = plan.select([{"group_index": 0, "suggestions": [{}]}], limit=10**6)
        assert vs.summary_line([selection], len(PLAN)) is None


# ---------------------------------------------------------------------------
# prepare_batched_validation_tasks: the materialized round
# ---------------------------------------------------------------------------

class TestPerBatchPlanContext:
    GROUPS = [
        _group("### 1.1 Retry"),
        _group("## 3. Testing"),
        _group("the overall approach"),
    ]

    def test_each_batch_carries_the_sections_its_group_cites(self, tmp_path, cited_env, capsys):
        result, phase = _materialize(tmp_path, self.GROUPS)
        batches = result["batches"]
        assert [b["group_indices"] for b in batches] == [[0], [1], [2]]
        snapshot = str(phase / PLAN_SNAPSHOT_NAME)

        retry, testing, whole = (b["prompt"] for b in batches)
        assert "RETRYBODY" in _plan_block(retry) and "TESTINGBODY" not in _plan_block(retry)
        assert "TESTINGBODY" in _plan_block(testing) and "RETRYBODY" not in _plan_block(testing)
        # The validator sees which section each suggestion names.
        assert '"reference": "### 1.1 Retry"' in retry
        for batch in batches[:2]:
            assert batch["context_mode"] == "workspace-read"
            (source,) = batch["context_sources"]
            assert source["mode"] == "path" and source["selection"] == "cited"
            assert source["abs_path"] == snapshot and source["path"] == PLAN_SNAPSHOT_NAME
            assert batch["prompt"].count(snapshot) == 1
            assert "## Plan file" in batch["prompt"]
            assert "and the sections the suggestions in this batch cite" in batch["prompt"]
            assert "read the file before answering such a\nclaim" in batch["prompt"]
        # The group citing nothing resolvable gets the whole plan, inline.
        assert _plan_block(whole).rstrip("\n") == PLAN.rstrip("\n")
        assert batches[2]["context_mode"] == "inline"
        assert "context_sources" not in batches[2]
        assert "## Plan file" not in whole
        err = capsys.readouterr().err
        assert "[VALIDATION_CONTEXT_CITED] plan: 2 of 3 batches carry" in err

    def test_a_fallback_batch_is_byte_identical_to_the_full_switch(self, tmp_path, monkeypatch):
        monkeypatch.setenv(vs.CITED_CONTEXT_ENV, vs.CONTEXT_CITED)
        cited, cited_phase = _materialize(tmp_path, self.GROUPS, name="cited")
        monkeypatch.setenv(vs.CITED_CONTEXT_ENV, "full")
        full, full_phase = _materialize(tmp_path, self.GROUPS, name="full")
        for batch in full["batches"]:
            assert _plan_block(batch["prompt"]).rstrip("\n") == PLAN.rstrip("\n")
            assert batch["context_mode"] == "inline" and "context_sources" not in batch
        assert cited["batches"][2]["prompt"] == full["batches"][2]["prompt"]
        # The switch never moves a batch: same groups, same files.
        assert [b["group_ids"] for b in cited["batches"]] == [
            b["group_ids"] for b in full["batches"]
        ]

    def test_the_default_embeds_the_whole_plan_in_every_batch(self, tmp_path, monkeypatch):
        """Plan scoping is opt-in until a live check: unset, every batch is the
        ``full`` switch's batch byte for byte, and no cited line is printed."""
        monkeypatch.delenv(vs.CITED_CONTEXT_ENV, raising=False)
        default, default_phase = _materialize(tmp_path, self.GROUPS, name="default")
        monkeypatch.setenv(vs.CITED_CONTEXT_ENV, vs.CONTEXT_FULL)
        full, full_phase = _materialize(tmp_path, self.GROUPS, name="full")
        for a, b in zip(default["batches"], full["batches"]):
            assert _plan_block(a["prompt"]).rstrip("\n") == PLAN.rstrip("\n")
            assert a["context_mode"] == "inline" and "context_sources" not in a
            assert a["prompt"].replace(str(default_phase), "<phase>") == (
                b["prompt"].replace(str(full_phase), "<phase>")
            )

    def test_a_partly_resolved_group_claims_no_cited_coverage(self, tmp_path, cited_env):
        """A group whose second part names no heading gets the whole plan, and
        its prompt says nothing about cited sections."""
        result, _ = _materialize(tmp_path, [_group("## 3. Testing / §9 Missing")])
        (batch,) = result["batches"]
        assert _plan_block(batch["prompt"]).rstrip("\n") == PLAN.rstrip("\n")
        assert "not cited by this batch" not in batch["prompt"]
        assert "## Plan file" not in batch["prompt"]
        assert batch["context_mode"] == "inline"

    def test_identical_inputs_render_identical_prompts(self, tmp_path, cited_env):
        one, one_phase = _materialize(tmp_path, self.GROUPS, name="one")
        two, two_phase = _materialize(tmp_path, self.GROUPS, name="two")
        for a, b in zip(one["batches"], two["batches"]):
            assert a["prompt"].replace(str(one_phase), "<phase>") == (
                b["prompt"].replace(str(two_phase), "<phase>")
            )

    def test_a_runner_that_cannot_read_gets_the_whole_plan(self, tmp_path, cited_env):
        result, _ = _materialize(
            tmp_path, self.GROUPS, runner_read_refusal="runner x cannot read files",
        )
        for batch in result["batches"]:
            assert _plan_block(batch["prompt"]).rstrip("\n") == PLAN.rstrip("\n")
            assert batch["context_mode"] == "inline"

    def test_a_snapshot_outside_the_native_hosts_root_gets_the_whole_plan(
        self, tmp_path, cited_env
    ):
        class Advisory:
            advisory_only = True
            root = str(tmp_path / "elsewhere")

            def __call__(self, path):
                return None

            def advisory(self, path):
                return None

        result, _ = _materialize(tmp_path, self.GROUPS, runner_read_refusal=Advisory())
        assert all("context_sources" not in b for b in result["batches"])

    def test_a_round_whose_only_partial_block_is_cited_publishes_nothing(
        self, tmp_path, cited_env
    ):
        _, phase = _materialize(tmp_path, self.GROUPS[:2], publish=True)
        status = phase / ".status.json"
        assert not status.exists() or "validation_context" not in json.loads(
            status.read_text(encoding="utf-8")
        )

    def test_resume_re_verifies_the_named_snapshot(self, tmp_path, cited_env):
        result, phase = _materialize(tmp_path, self.GROUPS)
        assert verify_context_sources(result, str(phase)) is False
        (phase / PLAN_SNAPSHOT_NAME).write_text(PLAN + "\nedited\n", encoding="utf-8")
        with pytest.raises(ValidationContextError, match="changed"):
            verify_context_sources(result, str(phase))

    def test_a_moved_phase_dir_rebinds_the_cited_prompt(self, tmp_path, cited_env):
        result, phase = _materialize(tmp_path, self.GROUPS)
        moved = tmp_path / "moved"
        phase.rename(moved)
        assert verify_context_sources(result, str(moved)) is True
        prompt = result["batches"][0]["prompt"]
        assert str(moved / PLAN_SNAPSHOT_NAME) in prompt
        assert str(phase / PLAN_SNAPSHOT_NAME) not in prompt

    def test_verdicts_join_their_groups_exactly_as_before(self, tmp_path, cited_env):
        groups = self.GROUPS + [_group("## 4. Rollout", importance="MEDIUM")]
        result, phase = _materialize(tmp_path, groups)
        statuses = {0: "valid", 1: "invalid", 2: "needs-human-decision", 3: "valid"}
        assert any("context_sources" in b for b in result["batches"])
        for batch in result["batches"]:
            reply = {"groups": [
                {"group_index": index, "group_hash": group_hash,
                 "status": statuses[index], "reason": "checked", "confidence": 0.9}
                for index, group_hash in zip(batch["group_indices"], batch["group_ids"])
            ]}
            Path(batch["output_path"]).write_text(json.dumps(reply), encoding="utf-8")
        merged = merge_batched_validation_results(str(phase), result, len(groups), groups)
        assert [m["status"] for m in merged] == [statuses[i] for i in range(4)]

    def test_the_resume_inputs_do_not_hash_the_prompt(self, tmp_path, monkeypatch):
        monkeypatch.setenv(vs.CITED_CONTEXT_ENV, vs.CONTEXT_CITED)
        cited, _ = _materialize(tmp_path, self.GROUPS, name="cited")
        monkeypatch.setenv(vs.CITED_CONTEXT_ENV, "full")
        full, _ = _materialize(tmp_path, self.GROUPS, name="full")
        assert cited["inputs"]["plan_sha256"] == full["inputs"]["plan_sha256"]
        assert set(cited["inputs"]) == set(full["inputs"])


# ---------------------------------------------------------------------------
# The published validation_context object
# ---------------------------------------------------------------------------

class TestPublishedObject:
    CITED = {"label": "plan", "mode": "path", "selection": "cited",
             "chars_shown": 5000, "bytes": 20000, "abs_path": "/p/s.md"}
    HEAD = {"label": "plan", "mode": "path", "chars_shown": 60000, "bytes": 90000,
            "abs_path": "/p/s.md"}

    def test_all_cited_publishes_nothing_unless_a_verdict_could_not_read(self):
        assert vc.build_validation_context(phase="p", sources=[self.CITED], limit=1) is None
        obj = vc.build_validation_context(
            phase="p", sources=[self.CITED], limit=1, insufficient=2,
        )
        assert obj and obj["mode"] == "path"
        assert "cited sections inline (smallest 5000 chars)" in vc.validation_context_text(obj)

    def test_one_head_cut_batch_makes_it_an_ordinary_path_entry(self):
        (entry,) = vc.aggregate_sources([dict(self.CITED), dict(self.HEAD)])
        assert "selection" not in entry and entry["chars"] == 5000
        (entry,) = vc.aggregate_sources([dict(self.HEAD), dict(self.CITED)])
        assert "selection" not in entry
        obj = vc.build_validation_context(
            phase="p", sources=[self.CITED, self.HEAD], limit=1,
        )
        assert obj is not None
        assert "excerpt inline + full snapshot by path" in vc.validation_context_text(obj)

    def test_batches_that_dropped_different_files_name_them_all(self):
        one = {"label": "reviewed-code", "mode": "truncated", "chars_shown": 50000,
               "bytes": 70000, "dropped_files": ["a.py"], "sections_total": 3,
               "sections_shown": 2}
        two = dict(one, dropped_files=["b.py", "a.py"], sections_total=4, sections_shown=2)
        (entry,) = vc.aggregate_sources([one, two])
        assert entry["dropped_files"] == ["a.py", "b.py"]
        assert "sections_total" not in entry
        assert vc.truncation_sentence(entry) == (
            "Validation saw 50000 of 70000 chars of the reviewed code (dropped: a.py, b.py)"
        )
        (same,) = vc.aggregate_sources([dict(one), dict(one)])
        assert same["sections_total"] == 3 and same["dropped_files"] == ["a.py"]

    def test_inline_batches_beside_cited_ones_stay_cited(self):
        inline = {"label": "plan", "mode": "inline", "chars_shown": 20000}
        (entry,) = vc.aggregate_sources([dict(inline), dict(self.CITED)])
        assert vc.is_cited(entry)


# ---------------------------------------------------------------------------
# review-code: each batch's own code windows
# ---------------------------------------------------------------------------

class TestPerBatchCodeWindows:
    @staticmethod
    def _repo(tmp_path):
        for name in ("a.py", "b.py"):
            (tmp_path / name).write_text(
                "\n".join(f"{name} line {n}" for n in range(1, 401)), encoding="utf-8"
            )
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n\n## 1. Do it\n", encoding="utf-8")
        return plan

    @staticmethod
    def _groups():
        return [
            {"suggestions": [{"file": "a.py", "line_range": [100, 101]}]},
            {"suggestions": [{"file": "b.py", "line_range": [50, 50]}]},
            {"suggestions": [{"file": "a.py", "line_range": [300, 302]}]},
        ]

    def test_a_batch_carries_only_its_own_files_and_windows(self, tmp_path, default_env):
        plan = self._repo(tmp_path)
        groups = self._groups()
        per_batch = {}
        secondary, reviewed_code = cro.build_validation_secondary_context(
            groups, plan, project_root=str(tmp_path), per_batch=per_batch,
        )
        # The resume hash still names the round's whole, untrimmed excerpt.
        assert reviewed_code == cro.build_reviewed_code_excerpt(
            groups, project_root=str(tmp_path)
        )
        assert content_hash(reviewed_code)
        for text in ("a.py line 100", "a.py line 300", "b.py line 50"):
            assert text in secondary

        first, report = per_batch["for_batch"]([0])
        assert "a.py line 100" in first
        assert "a.py line 300" not in first and "### b.py" not in first
        assert report["files_total"] == 1 and report["dropped_files"] == []
        assert first.startswith("#### Implementation record")

        both, report = per_batch["for_batch"]([1, 2])
        assert "b.py line 50" in both and "a.py line 300" in both
        assert "a.py line 100" not in both
        assert report["files_total"] == 2

    def test_the_view_builds_each_batchs_source_and_instructions(self, tmp_path, default_env):
        plan = self._repo(tmp_path)
        per_batch = {}
        cro.build_validation_secondary_context(
            self._groups(), plan, project_root=str(tmp_path), per_batch=per_batch,
        )
        view = cro.reviewed_code_batch_view(per_batch)
        text, document, extra = view([0])
        assert document.label == "reviewed-code" and not document.forced_truncated
        assert extra == cro.CODE_REVIEW_VALIDATION_GUIDANCE
        assert text == per_batch["for_batch"]([0])[0]

    def test_the_full_switch_keeps_the_round_wide_block(self, tmp_path, monkeypatch):
        monkeypatch.setenv(vs.CITED_CONTEXT_ENV, "full")
        plan = self._repo(tmp_path)
        per_batch = {}
        cro.build_validation_secondary_context(
            self._groups(), plan, project_root=str(tmp_path), per_batch=per_batch,
        )
        assert per_batch == {}
        assert cro.reviewed_code_batch_view(per_batch) is None

    def test_prepare_embeds_the_view_per_batch(self, tmp_path, default_env):
        plan = self._repo(tmp_path)
        groups = [dict(g, theme=f"t{i}", category="c", models=["m"]) for i, g in enumerate(self._groups())]
        for group in groups:
            group["suggestions"][0].update(
                title="issue", desc="d", importance="HIGH", type="bug"
            )
        per_batch = {}
        secondary, reviewed_code = cro.build_validation_secondary_context(
            groups, plan, project_root=str(tmp_path), per_batch=per_batch,
        )
        phase = tmp_path / "code-review"
        result = prepare_batched_validation_tasks(
            groups=groups, context=plan.read_text(encoding="utf-8"),
            output_dir=str(phase), plan_file=str(plan),
            secondary_context=secondary,
            secondary_context_label=cro.REVIEWED_CODE_CONTEXT_LABEL,
            reviewed_code=reviewed_code,
            secondary_for_batch=cro.reviewed_code_batch_view(per_batch),
        )
        prompts = [b["prompt"] for b in result["batches"]]
        assert len(prompts) == 3
        assert "a.py line 100" in prompts[0] and "a.py line 300" not in prompts[0]
        assert "b.py line 50" in prompts[1] and "### a.py" not in prompts[1]
        assert "a.py line 300" in prompts[2] and "a.py line 100" not in prompts[2]
        assert result["inputs"]["reviewed_code_sha256"] == content_hash(reviewed_code)
