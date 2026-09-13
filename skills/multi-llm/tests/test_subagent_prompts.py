"""D2i prompt-generator tests: purity, golden text, and the no-paths invariant.

The five generators are **pure** — no I/O, no clock, no config lookup — so
identical inputs render byte-identical prompt text. That is the property the D3a
result-contract equivalence test diffs across the two invokers, and it is what
makes the per-kind golden prompt fixtures in ``tests/golden/prompts/`` possible at
all.

The other invariant tested here spans all five: a materialized prompt names no
path the runner may be unable to open. ``plan_file``, ``grouped_file`` and
``output_path`` are provenance and invoker plumbing, never subagent instructions.
"""

import ast
import json
import os
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import subagent_prompts  # noqa: E402
from utils.subagent_prompts import (  # noqa: E402
    CODE_REVIEW_VALIDATION_GUIDANCE,
    HUMAN_DECISION_JUDGE_RULES,
    REFERENCE_SECTIONS,
    TASK_AUTHORING_RULES,
    build_consolidation_prompt,
    build_human_decision_judge_prompt,
    build_salvage_prompt,
    build_task_generation_prompt,
    build_validation_prompt,
    extract_reference_section,
    render_reference_section,
)

GOLDEN_DIR = SKILL_ROOT / "tests" / "golden" / "prompts"

#: Set MULTI_LLM_UPDATE_GOLDEN=1 to rewrite the golden prompt fixtures.
UPDATE_GOLDEN = os.environ.get("MULTI_LLM_UPDATE_GOLDEN") == "1"


# ---------------------------------------------------------------------------
# Fixture inputs — deliberately fixed values, since the generators are pure
# ---------------------------------------------------------------------------

PLAN_EXCERPT = "# Fixture plan\n\n## Step 1\nDo the thing.\n"

SUGGESTION_RECORDS = [
    {
        "index": 0,
        "group_hash": "aaaa000000000000",
        "group_id": "aaaa000000000000",
        "theme": "Add retry to the upload call",
        "category": "reliability",
        "models": ["model-a", "model-b"],
        "suggestions": [
            {
                "title": "Retry uploads",
                "desc": "A transient 503 currently loses the file.",
                "importance": "MEDIUM",
                "type": "addition",
            }
        ],
    },
    {
        "index": 1,
        "group_hash": "bbbb000000000000",
        "group_id": "bbbb000000000000",
        "theme": "Document the config location",
        "category": "docs",
        "models": ["model-a"],
        "suggestions": [
            {
                "title": "Say where config lives",
                "desc": "The plan never states the directory.",
                "importance": "LOW",
                "type": "clarification",
            }
        ],
    },
]

GROUP_RECORDS = [
    {
        "index": 0,
        "group_id": "aaaa000000000000",
        "theme": "Add retry to the upload call",
        "importance": "MEDIUM",
        "type": "addition",
        "reference": "Step 1",
        "suggestions": [{"title": "Retry uploads", "description": "No retry today."}],
    },
    {
        "index": 1,
        "group_id": "bbbb000000000000",
        "theme": "Uploads need backoff",
        "importance": "HIGH",
        "type": "modification",
        "reference": "Step 1",
        "suggestions": [{"title": "Backoff", "description": "Use exponential backoff."}],
    },
]

SALVAGE_REQUEST = {
    "model": "gemini:gemini-3-pro",
    "phase": "review_plan",
    "raw_output": 'Here you go:\n[{"title": "T", "desc": "D",}]\nHope that helps!',
    "expected_type": "array",
    "expected_schema": {
        "fields": ["title", "desc", "importance", "reference", "type"],
        "importance_values": ["high", "medium", "low"],
        "type_values": ["addition", "modification", "deletion", "clarification"],
    },
    "output_path": "/phase/gemini_gemini-3-pro.json",
    "timestamp": "2026-01-01T00:00:00",
}

JUDGE_ITEMS = [
    {
        "group_id": "aaaa000000000000",
        "title": "Retry uploads",
        "description": "A transient 503 currently loses the file, and the plan "
                       "also speculates about a queue nobody asked for.",
        "importance": "medium",
        "validation_reason": "Could go either way on the queue part.",
        "validation_confidence": 0.4,
        "models": ["model-a", "model-b"],
    }
]

PLAN_TEXT = "# Fixture plan\n\nBuild the thing.\n"


def _prompts():
    """Every generator's output, keyed by golden-fixture name."""
    return {
        "validation": build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        ),
        "revalidation": build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT,
            suggestion_records=SUGGESTION_RECORDS,
            kind="revalidation",
        ),
        "validation_code_review": build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT,
            suggestion_records=SUGGESTION_RECORDS,
            context_label="Plan content",
            extra_instructions=CODE_REVIEW_VALIDATION_GUIDANCE,
        ),
        "consolidation": build_consolidation_prompt(
            template=(SKILL_ROOT / "prompts" / "consolidate_suggestions.txt").read_text(
                encoding="utf-8"
            ),
            group_records=GROUP_RECORDS,
        ),
        "salvage": build_salvage_prompt(SALVAGE_REQUEST),
        "human_decision_judge": build_human_decision_judge_prompt(JUDGE_ITEMS),
        "task_generation": build_task_generation_prompt(
            plan_text=PLAN_TEXT, repo_root="/repo", plan_name="fixture-plan.md"
        ),
    }


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------

class TestGeneratorsArePure:
    """No I/O, no clock, no config lookup — inputs in, text out."""

    FORBIDDEN_CALLS = {
        "open", "read_text", "write_text", "now", "utcnow", "time", "monotonic",
        "getenv", "system", "run", "popen", "load_config", "load_prompt",
        "read_bytes", "glob", "iterdir",
    }

    def _generator_functions(self):
        tree = ast.parse(
            (SKILL_ROOT / "utils" / "subagent_prompts.py").read_text(encoding="utf-8")
        )
        wanted = {
            "build_validation_prompt",
            "build_consolidation_prompt",
            "build_salvage_prompt",
            "build_human_decision_judge_prompt",
            "build_task_generation_prompt",
        }
        return [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]

    def test_all_five_generators_exist(self):
        assert len(self._generator_functions()) == 5

    def test_no_generator_calls_io_a_clock_or_config(self):
        for func in self._generator_functions():
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                assert name not in self.FORBIDDEN_CALLS, (
                    f"{func.name} calls {name}() — the generators must stay pure "
                    f"so identical inputs render byte-identical prompts."
                )

    def test_module_imports_nothing_that_does_io(self):
        """The module's own imports are `json` and typing only, so importing it
        cannot read a file, a clock or the config."""
        tree = ast.parse(
            (SKILL_ROOT / "utils" / "subagent_prompts.py").read_text(encoding="utf-8")
        )
        top_level = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        modules = set()
        for node in top_level:
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif node.module:
                modules.add(node.module)
        assert modules <= {"json", "typing", "__future__"}, modules

    @pytest.mark.parametrize("name", sorted(_prompts()))
    def test_identical_inputs_render_byte_identical_text(self, name):
        assert _prompts()[name] == _prompts()[name]

    def test_key_insertion_order_does_not_change_the_prompt(self):
        """Records built with keys inserted in a different order must render the
        same text, or the two invokers could not be diffed."""
        reordered = [
            {key: record[key] for key in reversed(list(record))}
            for record in SUGGESTION_RECORDS
        ]
        assert build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        ) == build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=reordered
        )


# ---------------------------------------------------------------------------
# The no-paths invariant
# ---------------------------------------------------------------------------

class TestNoPathsInPromptText:
    """One invariant spans all five generators: a materialized prompt names no
    path the runner may be unable to open."""

    PLAN_FILE = "/abs/plans/fixture-plan.md"
    GROUPED_FILE = "/abs/plans/fixture-plan/review-plan/grouped.json"
    OUTPUT_PATH = "/abs/plans/fixture-plan/review-plan/validation_batch_0.json"

    @pytest.mark.parametrize("name", sorted(_prompts()))
    def test_no_generator_interpolates_the_task_path_fields(self, name):
        prompt = _prompts()[name]
        for path in (self.PLAN_FILE, self.GROUPED_FILE, self.OUTPUT_PATH):
            assert path not in prompt
        for token in ("grouped.json", "validation_batch_", "output_path",
                      "validation_tasks.json"):
            assert token not in prompt, f"{name} prompt names {token}"

    def test_the_materialized_validation_task_never_leaks_its_paths(self, tmp_path):
        """Same assertion at the producer level, where the paths actually exist."""
        from utils.validation import prepare_batched_validation_tasks

        plan = tmp_path / "plan.md"
        plan.write_text(PLAN_EXCERPT, encoding="utf-8")
        groups = [
            {
                "theme": f"Theme {i}",
                "category": "x",
                "models": ["m"],
                "suggestions": [
                    {"title": "t", "desc": "d", "importance": "MEDIUM",
                     "type": "addition"}
                ],
            }
            for i in range(6)
        ]
        result = prepare_batched_validation_tasks(
            groups=groups,
            context=PLAN_EXCERPT,
            output_dir=str(tmp_path),
            plan_file=str(plan),
        )
        for task in result["batches"]:
            # The fields exist on the task (provenance / invoker plumbing) ...
            assert task["plan_file"] == str(plan)
            assert task["output_path"].endswith(task["output_name"])
            # ... and appear nowhere in the prompt.
            assert task["plan_file"] not in task["prompt"]
            assert task["grouped_file"] not in task["prompt"]
            assert task["output_path"] not in task["prompt"]
            assert task["output_name"] not in task["prompt"]

    def test_the_materialized_consolidation_task_never_leaks_its_paths(self, tmp_path):
        from utils.consolidation import prepare_consolidation_tasks

        groups = [
            {"theme": f"T{i}", "importance": "MEDIUM", "type": "addition",
             "suggestions": [{"title": "a", "description": "b"}]}
            for i in range(4)
        ]
        result = prepare_consolidation_tasks(
            groups, {"section": [0, 1, 2, 3]}, str(tmp_path), "/abs/plan.md"
        )
        for task in result["batches"]:
            assert task["output_path"] not in task["prompt"]
            assert task["grouped_file"] not in task["prompt"]
            assert "/abs/plan.md" not in task["prompt"]

    def test_task_generation_names_exactly_one_path_the_repo_root(self):
        prompt = build_task_generation_prompt(
            plan_text=PLAN_TEXT, repo_root="/abs/repo", plan_name="p.md"
        )
        assert "/abs/repo" in prompt
        # ... and instructs exploration in capability-neutral terms, since
        # Glob/Grep/Read exist on no other host.
        for claude_tool in ("Glob", "Grep(", "Read tool", "Write tool", "Task tool"):
            assert claude_tool not in prompt

    @pytest.mark.parametrize("name", sorted(_prompts()))
    def test_every_prompt_ends_on_the_print_not_write_contract(self, name):
        prompt = _prompts()[name]
        assert "Do not write any file" in prompt
        assert "persists your reply" in prompt


# ---------------------------------------------------------------------------
# Golden prompt fixtures, one per kind
# ---------------------------------------------------------------------------

class TestGoldenPrompts:
    @pytest.mark.parametrize("name", sorted(_prompts()))
    def test_prompt_matches_its_golden_fixture(self, name):
        rendered = _prompts()[name]
        path = GOLDEN_DIR / f"{name}.txt"
        if UPDATE_GOLDEN or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
            if not UPDATE_GOLDEN:
                pytest.fail(f"created missing golden fixture {path.name}; re-run")
            return
        assert rendered == path.read_text(encoding="utf-8"), (
            f"{name} prompt drifted from tests/golden/prompts/{name}.txt. If the "
            f"change is intended, re-record with MULTI_LLM_UPDATE_GOLDEN=1."
        )


# ---------------------------------------------------------------------------
# Rendered narrative sections must not diverge from their constants
# ---------------------------------------------------------------------------

class TestRenderedReferenceSections:
    """One source of truth: the reference/instruction section is *rendered from*
    the generator's rule constant, and this test fails when the two diverge."""

    @pytest.mark.parametrize("name", sorted(REFERENCE_SECTIONS))
    def test_checked_in_section_matches_the_constant(self, name):
        rel_path = REFERENCE_SECTIONS[name][0]
        document = (SKILL_ROOT / rel_path).read_text(encoding="utf-8")
        current = extract_reference_section(document, name)
        assert current is not None, (
            f"{rel_path} carries no BEGIN/END GENERATED markers for {name!r}. "
            f"Run `python utils/subagent_prompts.py` after adding them."
        )
        assert current == render_reference_section(name), (
            f"{rel_path}'s {name!r} block has diverged from "
            f"utils/subagent_prompts.py. Edit the constant, then run "
            f"`python utils/subagent_prompts.py` to re-render."
        )

    def test_judge_rules_travel_inside_the_prompt_verbatim(self):
        prompt = build_human_decision_judge_prompt(JUDGE_ITEMS)
        assert HUMAN_DECISION_JUDGE_RULES in prompt

    def test_task_authoring_rules_travel_inside_the_prompt_verbatim(self):
        prompt = build_task_generation_prompt(plan_text=PLAN_TEXT, repo_root="/r")
        assert TASK_AUTHORING_RULES in prompt

    def test_rendering_is_idempotent(self):
        from utils.subagent_prompts import replace_reference_section

        for name in REFERENCE_SECTIONS:
            rel_path = REFERENCE_SECTIONS[name][0]
            document = (SKILL_ROOT / rel_path).read_text(encoding="utf-8")
            assert replace_reference_section(document, name) == document


# ---------------------------------------------------------------------------
# Per-kind embedded context and required reply (the D2i table)
# ---------------------------------------------------------------------------

class TestEmbeddedContextAndReply:
    def test_validation_embeds_the_excerpt_and_this_batch_only(self):
        prompt = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS[:1]
        )
        assert PLAN_EXCERPT in prompt
        assert "aaaa000000000000" in prompt
        assert "bbbb000000000000" not in prompt
        assert "groups to validate (1)" in prompt
        # Reply is the validation result array in today's shape.
        assert '"schema_version": "2.1"' in prompt

    def test_validation_tells_the_subagent_to_check_code_claims_when_it_can_read(self):
        """A validator that can read its own repository is told to use that,
        rather than answering a claim about existing code from the plan
        excerpt alone — and to reserve needs-human-decision for genuine
        judgment calls, not for claims it simply did not check."""
        prompt = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        )
        assert "existing code" in prompt
        assert "open it and check the claim before answering" in prompt
        assert "never for a claim you could have checked but did not look at" in prompt
        # Capability-neutral: still correct for a runner that cannot read.
        assert "If\nyou cannot read the repository, judge from the records" in prompt

    def test_reply_contract_tells_the_subagent_to_echo_each_groups_group_index(self):
        """The reply contract explicitly names `group_index` as the value to
        echo back, rather than leaving the subagent to guess which of the
        request record's fields the reply's `group_index` refers to.

        The request record itself carries the group's position under that same
        `group_index` key in production (`build_validation_batch_records`,
        tested via `prepare_batched_validation_tasks` in
        test_validation_group_id.py) — this generator treats records as
        opaque, so that key rename is verified at the producer, not here.
        """
        prompt = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        )
        assert "echo that group's" in prompt
        assert "`group_index` value back into its result" in prompt

    def test_revalidation_only_changes_the_framing_sentence(self):
        first = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        )
        second = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT,
            suggestion_records=SUGGESTION_RECORDS,
            kind="revalidation",
        )
        assert first != second
        assert first.split("\n", 1)[1] == second.split("\n", 1)[1]

    def test_consolidation_embeds_the_filled_template(self):
        template = (SKILL_ROOT / "prompts" / "consolidate_suggestions.txt").read_text(
            encoding="utf-8"
        )
        prompt = build_consolidation_prompt(
            template=template, group_records=GROUP_RECORDS
        )
        for placeholder in (
            "{GROUPS_JSON}", "{GROUP_COUNT}", "{OUTPUT_SCHEMA}",
            "{EXAMPLE_INPUT}", "{EXAMPLE_OUTPUT}", "{NON_OVERLAP_CONSTRAINTS}",
        ):
            assert placeholder not in prompt, f"{placeholder} left unfilled"
        assert "Total input groups: 2" in prompt
        assert "underlying_group_ids" in prompt

    def test_salvage_embeds_the_values_not_the_request_path(self):
        prompt = build_salvage_prompt(SALVAGE_REQUEST)
        assert SALVAGE_REQUEST["raw_output"] in prompt
        assert "gemini:gemini-3-pro" in prompt
        assert "importance_values" in prompt
        assert SALVAGE_REQUEST["output_path"] not in prompt
        # Extraction only: backup and request deletion are invoker duties now.
        assert "backup" not in prompt.lower()
        assert "Delete the salvage request" not in prompt

    def test_salvage_raw_output_is_never_truncated(self):
        big = {**SALVAGE_REQUEST, "raw_output": "x" * 200_000}
        assert "x" * 200_000 in build_salvage_prompt(big)

    def test_judge_prompt_embeds_the_seven_item_fields(self):
        prompt = build_human_decision_judge_prompt(JUDGE_ITEMS)
        for field in ("group_id", "title", "description", "importance",
                      "validation_reason", "confidence", "models"):
            assert f'"{field}"' in prompt
        assert '"importance": "MEDIUM"' in prompt  # normalized upper-case
        # Required reply is the verdict document.
        assert '"schema_version": "1.0"' in prompt
        assert '"verdicts"' in prompt

    def test_judge_prompt_verdicts_are_matched_by_id_not_position(self):
        prompt = build_human_decision_judge_prompt(JUDGE_ITEMS)
        assert "never by position" in prompt

    def test_task_generation_reply_is_the_document(self):
        prompt = build_task_generation_prompt(plan_text=PLAN_TEXT, repo_root="/r")
        assert '"plan_preamble"' in prompt
        assert '"executor_role": "agent|human"' in prompt
        assert "subagent_type" not in prompt
