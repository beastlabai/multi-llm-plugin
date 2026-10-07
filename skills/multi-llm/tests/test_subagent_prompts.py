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
import re
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

TASKS_EXCERPT = "# Implementation Tasks\n\n## T001: Do the thing\nWrite the code.\n"

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
        # review-tasks' two-context shape: the plan its suggestions are measured
        # against, plus the task list they actually name.
        "validation_review_tasks": build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT,
            suggestion_records=SUGGESTION_RECORDS,
            context_label="Plan \u2014 the requirements the tasks must cover",
            secondary_excerpt=TASKS_EXCERPT,
            secondary_label="Tasks file \u2014 the artifact under review",
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


def _unwrapped(text: str) -> str:
    """``text`` with every run of whitespace collapsed to one space.

    Prompt bodies are hard-wrapped, so a test that asserts on a whole sentence
    would otherwise have to re-encode the wrapping and would break on a reflow
    that changed no words.
    """
    return " ".join(text.split())


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

    def test_the_materialized_validation_task_under_the_limit_never_leaks_its_paths(self, tmp_path):
        """Same assertion at the producer level, where the paths actually exist.

        Under ``VALIDATION_CONTEXT_LIMIT`` the task is ``inline`` and names no
        path at all — the over-limit half is the next test (task 2.3).
        """
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
            # The prompt sidecar is the sharpest case of the rule: its whole
            # content *is* the prompt, so a prompt that named it would be
            # telling the subagent to go and read itself. The name stays out ...
            assert task["prompt_path"] not in task["prompt"]
            assert "prompt.txt" not in task["prompt"]
            # ... and the content is the prompt byte for byte. That identity is
            # what lets the read instruction live in host markdown instead of in
            # prompt text: a host can substitute the file for the string without
            # the subagent ever being told a path.
            with open(task["prompt_path"], "r", encoding="utf-8", newline="") as fh:
                assert fh.read() == task["prompt"]
            assert task["context_mode"] == "inline"
            assert "context_sources" not in task

    def test_the_over_limit_validation_task_names_exactly_one_snapshot_per_document(
        self, tmp_path
    ):
        """Over the limit (task 2.3): ``workspace-read``, and exactly one path per
        externalized document — the phase dir's hash-bound snapshot, never the
        live plan, ``grouped.json``, the output path or the sidecar."""
        from utils.validation import (
            PLAN_SNAPSHOT_NAME,
            VALIDATION_CONTEXT_LIMIT,
            bind_document_file,
            prepare_batched_validation_tasks,
        )

        plan = tmp_path / "plan.md"
        plan.write_text(
            "# Plan\n\n" + "".join(
                f"## S{i}\n" + "x" * 20000 + "\n" for i in range(5)
            ),
            encoding="utf-8",
        )
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase))
        assert len(document.text) > VALIDATION_CONTEXT_LIMIT
        groups = [
            {"theme": f"Theme {i}", "category": "x", "models": ["m"],
             "suggestions": [{"title": "t", "desc": "d", "importance": "MEDIUM",
                              "type": "addition"}]}
            for i in range(6)
        ]
        result = prepare_batched_validation_tasks(
            groups=groups, context=document.text, output_dir=str(phase),
            plan_file=str(plan), context_document=document,
        )
        snapshot = str(phase / PLAN_SNAPSHOT_NAME)
        for task in result["batches"]:
            assert task["context_mode"] == "workspace-read"
            prompt = task["prompt"]
            # The snapshot: in the `## Plan file` section and in the marker.
            assert "## Plan file" in prompt and snapshot in prompt
            assert prompt.count(snapshot) == 2
            assert [s["mode"] for s in task["context_sources"]] == ["path"]
            # Nothing else by path.
            assert str(plan) not in prompt
            assert task["grouped_file"] not in prompt
            assert task["output_path"] not in prompt
            assert task["prompt_path"] not in prompt

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

    def test_validation_judges_a_plan_omission_against_the_plan_not_the_tree(self):
        """"The plan does not specify X" is a claim about the plan text, so a
        working tree that already does X does not refute it.

        A 2026-09-13 run is the evidence: the same validator called "Car.reset()
        should clear boost state" *invalid* because "Car.reset() ... already
        resets boost state", while correctly calling its twin — "the plan should
        document that reset() grants maxCharges immediately" — valid. Both
        claims are about what the plan omits; only the verdicts differed. A
        partially implemented tree is the situation a plan review exists for, so
        the code already doing X makes the suggestion "document or decide it",
        not a false positive.
        """
        prompt = _unwrapped(build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        ))
        assert "A claim that the plan or the tasks omit something" in prompt
        assert (
            "is judged against the plan or tasks text above, not against the "
            "working tree"
        ) in prompt
        assert "Code that already does X does not make such a claim invalid" in prompt
        assert "the reason should say that the code already does X" in prompt

    def test_the_omission_rule_keeps_the_read_the_repository_rule(self):
        """The new rule narrows the old one, it does not replace it: a claim
        about existing code (a name, a line, current behaviour) is still checked
        by reading the repository."""
        prompt = _unwrapped(build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        ))
        assert "open it and check the claim before answering" in prompt
        assert "A claim that the plan or the tasks omit something" in prompt

    def test_a_second_context_is_its_own_labelled_section(self):
        """review-tasks judges claims about a task list, so the task list has to
        be in the prompt — and distinguishable from the plan beside it.

        Measured 2026-09-16 on a real review-tasks run: 28 of 30 suggestions
        named a bare `T###`, `'T001' in prompt` was False for every batch, and
        the prompt still told the validator to judge those claims "against the
        plan or tasks text above".
        """
        prompt = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT,
            suggestion_records=SUGGESTION_RECORDS,
            context_label="Plan \u2014 the requirements the tasks must cover",
            secondary_excerpt=TASKS_EXCERPT,
            secondary_label="Tasks file \u2014 the artifact under review",
        )
        assert PLAN_EXCERPT in prompt
        assert TASKS_EXCERPT in prompt
        assert "## Context (Plan \u2014 the requirements the tasks must cover)" in prompt
        assert "## Context (Tasks file \u2014 the artifact under review)" in prompt
        # Plan first, tasks second, suggestion records after both.
        assert prompt.index(PLAN_EXCERPT) < prompt.index(TASKS_EXCERPT)
        assert prompt.index(TASKS_EXCERPT) < prompt.index("aaaa000000000000")

    def test_one_context_renders_exactly_the_prompt_it_always_did(self):
        """The second block is opt-in: with no secondary excerpt the text is
        byte-identical to the single-context prompt every other phase sends."""
        prompt = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        )
        assert prompt.count("## Context (") == 1
        # No stray blank line where the second block would have gone.
        assert f"{PLAN_EXCERPT}\n\n## Suggestion groups to validate" in prompt

    def test_a_second_context_without_a_label_is_refused(self):
        """An unlabelled `## Context ()` block is the ambiguity two sections
        exist to remove, so it fails at render time rather than shipping."""
        with pytest.raises(ValueError, match="secondary_label"):
            build_validation_prompt(
                plan_excerpt=PLAN_EXCERPT,
                suggestion_records=SUGGESTION_RECORDS,
                secondary_excerpt=TASKS_EXCERPT,
            )

    def test_the_omission_rule_does_not_misstate_the_code_review_case(self):
        """Code review reuses this generator with CODE_REVIEW_VALIDATION_GUIDANCE
        and a plan excerpt as context, but there the artifact under review is the
        diff — so the rule names that case rather than telling a code reviewer to
        judge a diff against the plan."""
        prompt = _unwrapped(build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT,
            suggestion_records=SUGGESTION_RECORDS,
            extra_instructions=CODE_REVIEW_VALIDATION_GUIDANCE,
        ))
        assert (
            "Where the artifact under review is a diff or a code change rather "
            "than a plan, that change is the text such a claim is judged against."
        ) in prompt
        # And the code-review guidance still follows the rule, unchanged.
        assert _unwrapped(CODE_REVIEW_VALIDATION_GUIDANCE) in prompt

    def test_the_reply_template_offers_no_group_index_number_to_copy(self):
        """The reply schema shows `group_index` as an unfillable placeholder.

        A literal `"group_index": 0` there gets copied rather than filled in: 18
        of 32 verdicts in one run, and 41 of 46 in the run before it, came back
        numbered 0,1,2,3… The hash join absorbs the damage, but the template is
        where the habit starts.
        """
        prompt = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        )
        reply = prompt.split("## Reply", 1)[1]
        assert '"group_index": <the group\'s group_index value>' in reply
        assert '"group_index": 0' not in reply
        assert re.search(r'"group_index":\s*\d', reply) is None
        # The placeholder replaces a number, not the rule that explains it.
        assert (
            "The `group_index` you echo is advisory; the hash is what the "
            "results are joined on"
        ) in _unwrapped(reply)

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

    def test_judge_prompt_has_no_read_the_file_note_without_a_file(self):
        """JUDGE_ITEMS names no file (a plan-suggestion item) -> no note, and
        the golden fixture stays byte-identical."""
        prompt = build_human_decision_judge_prompt(JUDGE_ITEMS)
        assert "you may read" not in prompt.lower()

    def test_judge_prompt_embeds_file_and_line_range_and_permits_reading_it(self):
        """F9: a code-fix item (carrying file/line_range) gets both embedded
        and an explicit note that the judge may read the cited file."""
        item = {**JUDGE_ITEMS[0], "file": "index.html", "line_range": [93, 97]}
        prompt = build_human_decision_judge_prompt([item])

        assert '"file": "index.html"' in prompt
        assert '"line_range"' in prompt
        assert "you may read" in prompt.lower()
        # The rules and reply contract are unchanged either way.
        assert HUMAN_DECISION_JUDGE_RULES in prompt

    def test_judge_prompt_omits_file_key_when_item_has_none(self):
        """A plan-suggestion item (no file) never renders a null/empty `file`
        key — the note's presence is keyed on the field actually being
        there, not on a placeholder."""
        prompt = build_human_decision_judge_prompt(JUDGE_ITEMS)
        assert '"file"' not in prompt
        assert '"line_range"' not in prompt

    def test_task_generation_reply_is_the_document(self):
        prompt = build_task_generation_prompt(plan_text=PLAN_TEXT, repo_root="/r")
        assert '"plan_preamble"' in prompt
        assert '"executor_role": "agent|human"' in prompt
        assert "subagent_type" not in prompt


# ---------------------------------------------------------------------------
# Issues 6, 11 and 31 of the 2026-10-01 three-host run
# ---------------------------------------------------------------------------

class TestPlanReviewStaysOutOfTaskDecomposition:
    """Task breakdown belongs to generate-tasks; a plan review must not ask for it."""

    def test_the_reviewer_prompt_does_not_list_tasks_as_plan_content(self):
        text = (SKILL_ROOT / "prompts" / "plan_review.txt").read_text(encoding="utf-8")
        assert "Implementation tasks with dependencies" not in text
        assert "Out of Scope: Task Decomposition" in text
        assert "Do NOT report that the plan lacks a task breakdown" in text

    def test_a_plan_validation_tells_the_validator_the_scope_rule(self):
        prompt = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        )
        assert "no task breakdown" in prompt
        assert 'is "invalid"' in prompt

    def test_a_missing_ordering_constraint_stays_a_design_gap(self):
        """Adversarial review, 2026-10-02: "implementation order ... is invalid"
        read literally would also throw out "the plan never says the data
        migration must precede the schema switch", which is a design gap."""
        rule = "A missing ordering\nconstraint between parts of the design"
        prompt = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        )
        assert rule in prompt
        assert "is a design gap, judged on its merits" in prompt
        text = (SKILL_ROOT / "prompts" / "plan_review.txt").read_text(encoding="utf-8")
        assert ("A missing ordering constraint between parts of the design "
                "(a data migration that must precede the schema switch) is a "
                "design gap, not a task order: report it.") in text

    def test_code_review_and_review_tasks_validation_do_not_get_the_plan_rule(self):
        tasks = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT,
            suggestion_records=SUGGESTION_RECORDS,
            context_label="Tasks file",
        )
        code = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT,
            suggestion_records=SUGGESTION_RECORDS,
            extra_instructions=CODE_REVIEW_VALIDATION_GUIDANCE,
        )
        assert "no task breakdown" not in tasks
        assert "no task breakdown" not in code


class TestReplyContractIsRobustToAMalformedReply:
    def test_validation_and_judge_replies_name_the_bracket_rule(self):
        validation = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS
        )
        judge = build_human_decision_judge_prompt([{"group_id": "g", "title": "t"}])
        for prompt in (validation, judge):
            assert "a reply that does not parse is thrown away" in prompt
            assert "balanced brackets" in prompt


    @pytest.mark.parametrize("kind", ["validation", "revalidation"])
    def test_the_validation_reply_ends_on_the_groups_array(self, kind):
        """Claude Code racing runs (2026-10-01: 3 of 14 Sonnet batches; 10-04:
        3 of about 18; 10-07: in phases 01, 04 and 07) came back
        `{"groups":[{…},"metadata":{…}}` — the `]` dropped at the one junction
        where the array is followed by another key, on one-group batches too.
        The judge's reply, whose array is its last key, has not shown it. So
        `metadata` comes first, `groups` last, and the reply's ending is named:
        the last result's `}`, the `]`, the object's `}`. A re-ask stays the rule
        for a reply that still breaks (`util check-reply`)."""
        prompt = build_validation_prompt(
            plan_excerpt=PLAN_EXCERPT, suggestion_records=SUGGESTION_RECORDS,
            kind=kind,
        )
        reply = prompt[prompt.index("## Reply"):]
        template = reply[reply.index("\n{\n") + 1: reply.index("\n}\n") + 2]
        assert template.index('"metadata"') < template.index('"groups"')
        assert "`}]}`" in reply
        assert "`metadata` comes first and `groups` last" in reply
        # The template itself, placeholders filled, is one object ending `}]}`.
        filled = (
            template.replace("<the group's group_index value>", "3")
            .replace("valid|invalid|needs-human-decision", "valid")
        )
        compact = json.dumps(json.loads(filled), separators=(",", ":"))
        assert compact.endswith("}]}")
        assert list(json.loads(filled)) == ["metadata", "groups"]


class TestJudgePrompt:
    def test_a_code_fix_item_may_not_be_salvaged_into_a_plan_edit(self):
        prompt = build_human_decision_judge_prompt(
            [{"group_id": "g", "title": "t", "file": "index.html", "line_range": [1, 2]}]
        )
        assert "Never salvage it into an edit" in prompt
        assert "approved deviation" in prompt

    def test_a_prose_item_carries_no_code_fix_rule(self):
        prompt = build_human_decision_judge_prompt([{"group_id": "g", "title": "t"}])
        assert "Never salvage it into an edit" not in prompt

    def test_the_reply_asks_the_judge_to_name_its_model(self):
        prompt = build_human_decision_judge_prompt([{"group_id": "g", "title": "t"}])
        assert '"judge": "<the model you are>"' in prompt


class TestSchemaRepairPrompt:
    """The one schema-repair re-ask, shaped so a weaker model can follow it
    (2026-10-05: opencode/big-pickle failed task generation twice, after a
    re-ask that quoted a fragment's errors back at it)."""

    @pytest.mark.parametrize("kind", sorted(subagent_prompts.SCHEMA_REPAIR_SHAPES))
    def test_every_example_passes_the_kinds_whole_validation(self, kind):
        import copy

        from utils.subagent_executor import normalize_result, validate_result

        example = copy.deepcopy(subagent_prompts.SCHEMA_REPAIR_EXAMPLES[kind])
        normalize_result(kind, example)
        assert validate_result(kind, example) == []

    @pytest.mark.parametrize("kind,schema", [
        ("task_generation", "generated_tasks.schema.json"),
        ("apply_suggestions", "write_batch_result.schema.json"),
        ("apply_task_suggestions", "write_batch_result.schema.json"),
        ("apply_code_fixes", "write_batch_result.schema.json"),
        ("implement", "write_batch_result.schema.json"),
    ])
    def test_the_named_top_level_keys_are_the_schemas_required_ones(self, kind, schema):
        document = json.loads(
            (SKILL_ROOT / "schemas" / schema).read_text(encoding="utf-8")
        )
        assert list(subagent_prompts.SCHEMA_REPAIR_TOP_KEYS[kind]) == document["required"]

    def test_the_read_form_names_the_keys_and_ends_on_a_valid_document(self):
        prompt = subagent_prompts.build_schema_repair_prompt(
            "task_generation", "ORIGINAL QUESTION", ["<root>: missing required field 'tasks'"]
        )
        assert prompt.startswith("ORIGINAL QUESTION")
        assert "- <root>: missing required field 'tasks'" in prompt
        assert 'top-level keys are exactly "schema_version", "plan_preamble", "tasks"' in prompt
        assert 'Every task goes inside the "tasks" array' in prompt
        tail = prompt[prompt.rindex("copy its format, not its content:"):]
        document = json.loads(tail.split(":", 1)[1])
        assert document == subagent_prompts.SCHEMA_REPAIR_EXAMPLES["task_generation"]

    def test_a_syntax_error_replaces_the_fragments_errors(self):
        """The validator's errors describe whatever fragment was salvaged from
        inside the broken document; quoting them told a model whose keys were
        right that its keys were wrong."""
        prompt = subagent_prompts.build_schema_repair_prompt(
            "task_generation", "Q",
            ["<root>: missing required field 'schema_version'",
             "<root>: unexpected field 'acceptance_criteria'"],
            syntax_error="line 182, column 341: Expecting ',' delimiter — the "
                         "text there reads: so \"<<HERE>>3\", \"2\"",
        )
        assert "Your reply is not valid JSON" in prompt
        assert "line 182, column 341: Expecting ',' delimiter" in prompt
        assert "missing required field 'schema_version'" not in prompt
        assert "unexpected field 'acceptance_criteria'" not in prompt
        assert 'write it as \\"' in prompt

    def test_the_write_form_restates_and_never_resends_the_instructions(self):
        prompt = subagent_prompts.build_schema_repair_prompt(
            "apply_suggestions", "Apply batch 6: rewrite the Race logic section",
            ["<root>: expected type object, got list"],
            previous_reply='Changes applied.\n{"status": "applied", "ids": ["a"]}',
        )
        sys.path.insert(0, str(SKILL_ROOT / "tests" / "mocks"))
        import mock_llm

        assert prompt.startswith(mock_llm.REPORT_REASK_MARKER)
        assert "rewrite the Race logic section" not in prompt
        assert "Do NOT open, edit, create or delete any file" in prompt
        assert 'Changes applied.\n{"status": "applied", "ids": ["a"]}' in prompt
        assert "never name an id or a\nfile it does not name" in prompt

    @pytest.mark.parametrize("kind,item", [
        ("apply_suggestions", "suggestion"),
        ("apply_task_suggestions", "suggestion"),
        ("apply_code_fixes", "fix"),
        ("implement", "task"),
    ])
    def test_the_write_form_lists_the_batchs_ids_in_prompt_order(self, kind, item):
        """Fix round 11 (OpenCode racing run 4): the re-ask never named the
        batch's ids, and a reply that numbered its items ("Suggestion 1: …",
        as the batch prompt's Return Format asks) was restated with those
        names instead of the ids."""
        prompt = subagent_prompts.build_schema_repair_prompt(
            kind, "Q", ["x"], previous_reply="Suggestion 1: done\nSuggestion 2: done",
            expected_ids=["ec8092961e62a151", "7f1f3c58776954ff"],
        )
        assert (
            f"- {item} 1 = `ec8092961e62a151`\n- {item} 2 = `7f1f3c58776954ff`"
        ) in prompt
        assert "`ids` must use these spellings" in prompt
        assert "write the id listed beside\nthat number" in prompt
        assert "Name no id outside this list." in prompt
        assert "never claim an id for an item it does\nnot say it applied" in prompt
        # The list comes after the quoted reply and before the reply rules.
        assert (
            prompt.index("PREVIOUS REPLY>>>")
            < prompt.index("## The batch's ids")
            < prompt.index("## Reply")
        )

    def test_the_implement_form_restates_a_check_and_never_reruns_it(self):
        """Fix round 14: implement gets the report re-ask; a check task's
        result rides in `check`, which the verdict reads, so the re-ask asks
        for it as given — and only implement's re-ask says so."""
        prompt = subagent_prompts.build_schema_repair_prompt(
            "implement", "Run node --check on index.html", ["x"],
            previous_reply="The check ran: index.html parses.",
            expected_ids=["T014"],
        )
        assert "Run node --check" not in prompt
        assert "- task 1 = `T014`" in prompt
        assert "restate that object as\nit gave it; do not run the check again" in prompt
        other = subagent_prompts.build_schema_repair_prompt(
            "apply_suggestions", "Q", ["x"], previous_reply="done",
            expected_ids=["a"],
        )
        assert "`check` object" not in other

    def test_the_write_form_without_ids_lists_none(self):
        prompt = subagent_prompts.build_schema_repair_prompt(
            "apply_suggestions", "Q", ["x"], previous_reply="done",
            expected_ids=[],
        )
        assert "## The batch's ids" not in prompt
        assert "never name an id or a\nfile it does not name" in prompt

    def test_the_write_form_needs_a_reply_and_keeps_its_tail(self):
        with pytest.raises(ValueError):
            subagent_prompts.build_schema_repair_prompt(
                "apply_code_fixes", "Q", ["x"], previous_reply="  "
            )
        long_reply = "x" * 9000 + '{"status": "applied"}'
        prompt = subagent_prompts.build_schema_repair_prompt(
            "apply_code_fixes", "Q", ["x"], previous_reply=long_reply
        )
        assert '{"status": "applied"}' in prompt
        assert "earlier characters omitted" in prompt
        assert "x" * 7000 not in prompt

    def test_the_report_keys_are_the_schemas_properties(self):
        document = json.loads(
            (SKILL_ROOT / "schemas" / "write_batch_result.schema.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(subagent_prompts.WRITE_REPORT_KEYS) == set(document["properties"])

    def test_a_report_longer_than_the_tail_is_quoted_from_its_start(self):
        """Fix round 8: long notes pushed the report's status and ids out of a
        fixed 6000-character tail."""
        report = json.dumps({
            "status": "applied", "ids": ["a1", "b2"], "files_touched": ["p.md"],
            "notes": "checked " * 1500,
        })
        before = "Narration before it. " * 50
        reply = before + report

        prompt = subagent_prompts.build_schema_repair_prompt(
            "apply_suggestions", "Q", ["x"], previous_reply=reply
        )

        assert report in prompt
        assert f"[… {len(before)} earlier characters omitted …]\n{report}" in prompt

    def test_narration_after_the_report_is_cut_after_the_report(self):
        report = json.dumps({"status": "applied", "ids": ["a1"]})
        reply = "Before. " + report + "\n" + "after " * 5000

        quote = subagent_prompts.quote_previous_reply(reply)

        assert quote.startswith("[… 8 earlier characters omitted …]\n" + report)
        assert len(quote) < subagent_prompts.WRITE_REPORT_QUOTE_MAX_CHARS + 100
        assert quote.endswith("later characters omitted …]")
