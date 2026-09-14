#!/usr/bin/env python3
"""The five materialized-subagent prompt generators (D2i).

One module owns prompt assembly for every subagent kind the orchestrators
delegate: ``validation``/``revalidation``, ``consolidation``, ``salvage``,
``human_decision_judge`` and ``task_generation``. Before this module the
prompts lived in instruction markdown, so only a Claude Code agent reading that
markdown could assemble them; materializing them here is what makes one task
file executable by a native-subagent host *and* by a Python executor with
byte-identical prompt text.

Two properties are load-bearing and enforced by tests:

* **Every generator is pure.** No file or network I/O, no clock, no config
  lookup, no environment read — callers read files and pass records in. Identical
  inputs therefore render byte-identical prompt text, which is what makes
  per-kind golden prompt fixtures and the result-contract equivalence diff
  possible at all.
* **No generator interpolates a path the runner may be unable to open.** Under
  ``context_mode: inline`` every input travels inside the prompt as text: the
  plan excerpt, the suggestion records, the salvage request's ``raw_output``, the
  judge's criteria. ``plan_file``, ``grouped_file`` and ``output_path`` are
  provenance and invoker plumbing, never subagent instructions — a headless
  ``claude -p`` / ``codex exec`` invocation is not guaranteed a workspace-read
  tool, and a prompt full of unreadable paths degrades into a confidently empty
  answer rather than an error. ``task_generation`` is the single standing
  exception (``context_mode: workspace-read``): its context is the repository,
  and it names exactly one path, the pinned repo root.

Every generator also ends the same way: the callee **prints** its result and the
invoker persists it (the D3a print/persist contract). No prompt tells a subagent
to write a file.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "HUMAN_DECISION_JUDGE_RULES",
    "TASK_AUTHORING_RULES",
    "REFERENCE_SECTIONS",
    "build_validation_prompt",
    "build_consolidation_prompt",
    "build_salvage_prompt",
    "build_human_decision_judge_prompt",
    "build_task_generation_prompt",
    "render_reference_section",
    "reference_section_targets",
]


def _json_block(value: Any) -> str:
    """Deterministic JSON rendering for embedding in a prompt.

    ``sort_keys`` is deliberate: two callers that build the same record with
    keys inserted in a different order must still render the same prompt.
    """
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


# ===========================================================================
# validation / revalidation
# ===========================================================================

#: The reply shape every validation/revalidation batch must print.
#:
#: ``group_index`` is shown as an unfillable placeholder rather than a literal
#: ``0`` on purpose: with a numeric example there, models copy the example
#: instead of echoing the group's own value (18 of 32 verdicts in one run, 41 of
#: 46 in another, echoed 0,1,2,3…). The hash join absorbs it —
#: ``merge_batched_validation_results`` routes by ``group_hash`` and only counts
#: the mismatch — but the cheaper fix is a template with no number to copy. A
#: model that copies the placeholder *text* is tolerated too: the merge coerces a
#: non-integer echo to "matches nothing" rather than crashing.
_VALIDATION_REPLY_CONTRACT = '''## Reply

Print ONE JSON object and nothing else — no prose, no markdown fences, no
explanation. Do not write any file; whoever invoked you persists your reply.

{
  "groups": [
    {"group_index": <the group's group_index value>, "group_hash": "<copy the group_hash of the group verbatim>", "status": "valid|invalid|needs-human-decision", "reason": "<one brief sentence>", "confidence": 0.0}
  ],
  "metadata": {"model": "<the model you are>", "timestamp": "<ISO 8601>", "schema_version": "2.1"}
}

Rules:
1. Output EXACTLY one result per requested group — no more, no fewer.
2. Each group above carries its own `group_index` — echo that group's
   `group_index` value back into its result, and copy its `group_hash` value
   verbatim too. The `group_index` you echo is advisory; the hash is what the
   results are joined on, so a wrong hash discards the result.
3. Confidence is a number between 0.0 and 1.0.
4. Be conservative: when you cannot decide from the context given, answer
   "needs-human-decision" rather than guessing.
'''


#: Extra guidance for code-review validation, phrased in **capability-neutral**
#: terms. It refers only to the `file`/`line_range` values already embedded in the
#: issue records — never to a task's `plan_file`, `grouped_file` or `output_path`
#: — and it conditions every workspace action on the runner actually having one, so
#: the same prompt is correct for an agentic runner and for a headless one.
CODE_REVIEW_VALIDATION_GUIDANCE = '''Each issue record carries `file` and `line_range` when the finding has a
location.

- If you can read the repository, open those locations and compare what the code
  actually does against what the issue claims. Keep the effort bounded: at most 5
  files per group (prefer explicitly named files, then the narrowest line ranges,
  breaking ties by importance HIGH > MEDIUM > LOW), and at most ~200 lines around
  each referenced range.
- If you cannot read the repository, judge from the records and the context above
  alone.
- When an issue has no `file`/`line_range`, try the description plus the context
  above; if the missing location makes the claim unverifiable, answer
  "needs-human-decision" and say that the location metadata was absent.'''


def build_validation_prompt(
    *,
    plan_excerpt: str,
    suggestion_records: Sequence[Dict[str, Any]],
    kind: str = "validation",
    context_label: str = "Plan content",
    extra_instructions: str = "",
) -> str:
    """Render the prompt for one validation or revalidation batch.

    Args:
        plan_excerpt: The one shared context excerpt for this phase, already
            truncated by the caller (excerpted once per phase and reused across
            batches, so batches never re-truncate and prompts stay
            deterministic).
        suggestion_records: This batch's suggestion records — the values, not a
            path to them. Each record carries at least ``group_hash``, ``theme``
            and its ``suggestions``.
        kind: ``"validation"`` or ``"revalidation"``; only the framing sentence
            differs.
        context_label: What the embedded context is (``"Plan content"``,
            ``"Tasks file"``, ``"Diff"``, …).
        extra_instructions: Optional phase-specific guidance appended before the
            reply contract (e.g. :data:`CODE_REVIEW_VALIDATION_GUIDANCE`). Must be
            capability-neutral and must not name a task's path fields.

    Returns:
        The complete prompt text.
    """
    if kind == "revalidation":
        opening = (
            "You are re-validating review suggestions that a previous "
            "validation pass could not classify."
        )
    else:
        opening = "You are validating review suggestions for accuracy."

    count = len(suggestion_records)
    guidance = f"\n{extra_instructions.strip()}\n" if extra_instructions.strip() else ""
    return f'''{opening}

## Context ({context_label})

{plan_excerpt}

## Suggestion groups to validate ({count})

{_json_block(list(suggestion_records))}

## Task

For each of the {count} suggestion groups above, decide:
- "valid": the issue is real and should be addressed
- "invalid": a false positive — incorrect or not applicable
- "needs-human-decision": cannot be determined from the context above and
  requires human judgment

Compare each group against the context above to verify it. Judge only the
groups listed above; ignore anything else.

When a suggestion claims something about existing code — a function name, a
line number, current behaviour — and you can read the repository you are
running in, open it and check the claim before answering; do not rely on the
context above alone when the code itself is one read away. Reserve
"needs-human-decision" for genuine judgment calls (a product or design
trade-off), never for a claim you could have checked but did not look at. If
you cannot read the repository, judge from the records and the context above
alone.

A claim that the plan or the tasks omit something — "the plan does not specify
X", "the tasks never say to reset X" — is judged against the plan or tasks text
above, not against the working tree: the plan is what an implementer follows,
and a partially implemented tree is exactly the situation this review exists
for. Code that already does X does not make such a claim invalid; it makes it
valid — document or decide it — and the reason should say that the code already
does X. Where the artifact under review is a diff or a code change rather than a
plan, that change is the text such a claim is judged against.
{guidance}
{_VALIDATION_REPLY_CONTRACT}'''


# ===========================================================================
# consolidation
# ===========================================================================

#: Filled into the ``{OUTPUT_SCHEMA}`` slot of
#: ``prompts/consolidate_suggestions.txt``. Mirrors
#: ``schemas/consolidation_batch.schema.json``, which the reaggregation step
#: validates each ``consolidation_batch_N.json`` against.
CONSOLIDATION_OUTPUT_SCHEMA = '''{
  "verification": "passed",
  "clusters": [
    {
      "title": "<new synthesized title, max 120 characters>",
      "description": "<synthesized description preserving nuance from every underlying suggestion>",
      "importance": "HIGH|MEDIUM|LOW",
      "type": "addition|modification|deletion|clarification",
      "underlying_group_ids": ["<group_id>", "<group_id>"],
      "underlying_group_indices": [0, 1],
      "reasoning": "<1-2 sentences on why these belong together>"
    }
  ],
  "singletons": [
    {
      "group_id": "<group_id>",
      "group_index": 0,
      "reasoning": "<1-2 sentences on why this group stands alone>"
    }
  ]
}'''

CONSOLIDATION_EXAMPLE_INPUT = '''[
  {"index": 0, "group_id": "aaaa000000000000", "theme": "Add retry to the upload call", "importance": "MEDIUM", "type": "addition", "suggestions": [{"title": "Retry uploads", "description": "The upload call has no retry, so a transient 503 loses the file."}]},
  {"index": 1, "group_id": "bbbb000000000000", "theme": "Uploads need backoff on transient failures", "importance": "HIGH", "type": "modification", "suggestions": [{"title": "Exponential backoff for uploads", "description": "Wrap the upload in exponential backoff; a bare retry loop will hammer the service."}]},
  {"index": 2, "group_id": "cccc000000000000", "theme": "Document the config file location", "importance": "LOW", "type": "clarification", "suggestions": [{"title": "Say where config lives", "description": "The plan never states which directory the config file is read from."}]}
]'''

CONSOLIDATION_EXAMPLE_OUTPUT = '''{
  "verification": "passed",
  "clusters": [
    {
      "title": "Make uploads resilient to transient failures with bounded exponential backoff",
      "description": "The upload call currently has no retry at all, so a transient 503 loses the file; and a bare retry loop would hammer the service, so the retry must use bounded exponential backoff rather than immediate repetition.",
      "importance": "HIGH",
      "type": "modification",
      "underlying_group_ids": ["aaaa000000000000", "bbbb000000000000"],
      "underlying_group_indices": [0, 1],
      "reasoning": "Both address the same underlying concern — upload failures on transient errors — one naming the missing retry and the other the backoff policy it needs."
    }
  ],
  "singletons": [
    {
      "group_id": "cccc000000000000",
      "group_index": 2,
      "reasoning": "Documentation of the config location is unrelated to upload reliability."
    }
  ]
}'''

CONSOLIDATION_NON_OVERLAP_CONSTRAINTS = '''- Every input group appears EXACTLY ONCE across `clusters[].underlying_group_ids` and `singletons[].group_id`. Never in two clusters, never in both a cluster and singletons, never omitted.
- `underlying_group_ids` and `underlying_group_indices` must have the same length and must correspond element-by-element.
- A cluster contains at least 2 groups. A single group is a singleton, not a one-element cluster.
- Copy `group_id` and `index` values verbatim from the input. Never invent, renumber or reformat them.
- Set `"verification": "passed"` only after you have checked the three rules above against your own output. If you cannot satisfy them, return every group as a singleton instead of guessing.'''


def build_consolidation_prompt(
    *,
    template: str,
    group_records: Sequence[Dict[str, Any]],
) -> str:
    """Render the prompt for one consolidation batch.

    Args:
        template: The text of ``prompts/consolidate_suggestions.txt``. Passed in
            rather than loaded so this function stays pure.
        group_records: This batch's group records — the values, not a path.

    Returns:
        The filled template plus the print-only reply contract.
    """
    filled = (
        template
        .replace("{GROUPS_JSON}", _json_block(list(group_records)))
        .replace("{GROUP_COUNT}", str(len(group_records)))
        .replace("{OUTPUT_SCHEMA}", CONSOLIDATION_OUTPUT_SCHEMA)
        .replace("{EXAMPLE_INPUT}", CONSOLIDATION_EXAMPLE_INPUT)
        .replace("{EXAMPLE_OUTPUT}", CONSOLIDATION_EXAMPLE_OUTPUT)
        .replace("{NON_OVERLAP_CONSTRAINTS}", CONSOLIDATION_NON_OVERLAP_CONSTRAINTS)
    )
    return filled.rstrip("\n") + '''

## Reply

Print ONLY the JSON object described above — no prose, no markdown fences, no
explanation. Do not write any file; whoever invoked you persists your reply.
'''


# ===========================================================================
# salvage
# ===========================================================================

def build_salvage_prompt(salvage_request: Dict[str, Any]) -> str:
    """Render the prompt for one salvage request.

    Salvage repairs an unparseable *model review* from the fan-out, never a
    failed subagent task, so it carries no batch context: its entire state is
    the salvage-request record the orchestrator already wrote. The record's
    ``raw_output``, ``expected_type`` and ``expected_schema`` travel as values,
    not as a path — and are never truncated, since truncation salvages half an
    array.

    Two steps that used to live in this prompt are invoker duties now: the
    pre-write backup runs on the invoker side immediately before it persists the
    output, and deleting the salvage request is the orchestrator's job after
    reaggregation. What is left is extraction alone, which is why salvage no
    longer needs workspace access.

    **"Nothing recoverable" is a sentinel object, never an empty array.** A
    salvage task's ``output_name`` *is* the model's own result path, so an empty
    array printed here would be persisted over that path and reaggregation would
    read it as "this model reviewed the input and found nothing" — a model that
    never produced a parseable word becomes a phantom participant, and finding
    priority is scored partly by how many models flagged something. So the
    fallback is ``{"salvage_failed": true, "reason": ...}``, which
    ``subagent_executor`` recognises as a failed, never-retried outcome that
    writes nothing and carries the subagent's own reason into
    ``subagent_batch_report.json``. A bare empty array stays legal *as a
    recovery* — it is what a model whose reply really was an empty review
    produced — which is exactly why the negative needs a shape of its own.

    Args:
        salvage_request: The salvage-request record (``model``, ``phase``,
            ``raw_output``, ``expected_type``, ``expected_schema``, …).

    Returns:
        The complete prompt text.
    """
    model = salvage_request.get("model", "unknown")
    phase = salvage_request.get("phase", "unknown")
    expected_type = salvage_request.get("expected_type", "array")
    expected_schema = salvage_request.get("expected_schema", {})
    raw_output = salvage_request.get("raw_output", "")

    return f'''You are a JSON extraction specialist. Recover the valid JSON {expected_type}
from one model's malformed output.

The output below came from model `{model}` during the `{phase}` phase. It was
supposed to be a JSON {expected_type} and could not be parsed.

## Expected shape

{_json_block(expected_schema)}

## Raw output to salvage

<<<RAW_OUTPUT
{raw_output}
RAW_OUTPUT

## Task

1. Find the JSON {expected_type} inside the raw output above. If several
   complete ones are present, use the last complete one.
2. Repair only mechanical syntax damage — trailing commas, unterminated
   strings, stray prose before or after the {expected_type}, markdown fences.
3. Drop any entry that is missing a required field from the expected shape
   rather than inventing values for it.
4. Never add, rewrite or summarize content. You are recovering what the model
   already produced, not authoring it.

## Reply

Print ONLY the extracted JSON {expected_type} and nothing else — no prose, no
markdown fences, no explanation. Do not write any file; whoever invoked you
persists your reply.

If the raw output holds nothing recoverable at all, print exactly this object
instead — never an empty {expected_type}, which is indistinguishable from a
review that ran and genuinely found nothing:

{{"salvage_failed": true, "reason": "<what the raw output actually was>"}}

The reason is one short factual clause naming what the raw output turned out to
be — an authentication error, a stack trace, a truncated log line, an empty
stream, prose with no JSON anywhere in it. Not an apology, and not an account of
what you tried.
'''


# ===========================================================================
# human-decision judge
# ===========================================================================

#: The judge's Evaluation Criteria and Approve-vs-Salvage-vs-Skip rules.
#:
#: This constant is the single source of truth. The prompt inlines it verbatim
#: (under ``context_mode: inline`` the reference file is a path a headless runner
#: may not be able to open, so the criteria have to travel inside the prompt
#: body), and the corresponding section of ``references/human-decision-batch.md``
#: is *rendered from* it — ``tests/test_subagent_prompts.py`` fails when the two
#: diverge.
HUMAN_DECISION_JUDGE_RULES = '''### Three outcomes

- **Approve (as-is)** — the suggestion is sound; apply it unchanged.
- **Salvage (partial)** — only *part* of the suggestion is valid or worthwhile; rewrite it down to that defensible core and apply the rewritten version. **This is the default whenever an item is partially valid — never skip a whole suggestion just because part of it is wrong.**
- **Skip** — there is nothing worthwhile to keep; drop the item entirely.

### Evaluation Criteria

Read each item's details (title, description, importance, validation_reason, confidence) and decide based on:

1. **Clear improvement vs. subjective preference**: Approve items that address genuine bugs, security issues, or correctness problems. Be cautious with purely stylistic or preference-based items.
2. **Validation reason signals**: If the validation reason indicates genuine ambiguity ("could go either way", "trade-off", "matter of preference"), lean toward skipping. If it indicates implementation uncertainty ("not sure about scope", "might affect other files"), lean toward approving.
3. **Importance as risk signal**: Be more conservative (lean skip) with HIGH importance items — they have bigger impact if wrong. Be more permissive (lean approve) with LOW importance items.
4. **Security items lean approve**: Security-type items (XSS, injection, auth bypass, data exposure) should lean toward approval.
5. **Deletion caution**: For items that involve removing code/content, require stronger confidence before approving. Deletions are harder to reverse.
6. **Model consensus**: If the `models` array shows multiple models flagged this as `needs-human-decision`, be more cautious. If only one model flagged it while others said `valid`, lean toward approving.

### Deciding Approve vs. Salvage vs. Skip

Apply the criteria above to each **distinct claim inside the suggestion**, not just the suggestion as a whole. Many `needs-human-decision` items are a mix — one solid point bundled with an overreach, a speculative extra, or an out-of-scope aside. Salvaging the worthwhile part is the **default** for such items.

1. **Identify the worthwhile core**: Which specific part(s) address a genuine bug, correctness, security, or clarity issue and survive the criteria above?
2. **Whole thing is sound** -> **Approve** as-is.
3. **Part is worthwhile, the rest is questionable / speculative / out-of-scope / low-confidence** -> **Salvage**: rewrite the description to keep only the worthwhile core and drop the rest. Salvage only ever **narrows** a suggestion — never add anything that was not in the original. The existing criteria still govern the kept part (e.g. deletion caution, security lean-approve).
4. **Nothing survives** — the item is purely subjective, speculative, redundant, or wrong -> **Skip**. Salvage only when a concrete, defensible improvement remains; do **not** manufacture value or apply a watered-down change just to avoid skipping. If there is nothing worthwhile, skip it entirely.

A salvaged item is applied like an approval, except the **rewritten** description is what gets applied — never the original.'''


def build_human_decision_judge_prompt(items: Sequence[Dict[str, Any]]) -> str:
    """Render the prompt for one human-decision judging task.

    Args:
        items: The items in scope. Each carries ``group_id``, ``title``,
            ``description``, ``importance``, ``validation_reason``,
            ``confidence`` and ``models`` — the values, not a path to the report.

    Returns:
        The complete prompt text, with the judging rules inlined verbatim.
    """
    embedded = [
        {
            "group_id": item.get("group_id", ""),
            "title": item.get("title", ""),
            "description": item.get("description", item.get("desc", "")),
            "importance": (item.get("importance") or "MEDIUM").upper(),
            "validation_reason": item.get("validation_reason", ""),
            "confidence": item.get("validation_confidence", item.get("confidence", 0.0)),
            "models": list(item.get("models", [])),
        }
        for item in items
    ]
    count = len(embedded)
    noun = "item" if count == 1 else "items"

    return f'''You are deciding what to do with {count} review {noun} that automated validation
could not settle. For each one, choose Approve, Salvage or Skip.

## Items to judge ({count})

{_json_block(embedded)}

{HUMAN_DECISION_JUDGE_RULES}

## Reply

Print ONE JSON object and nothing else — no prose, no markdown fences, no
explanation. Do not write any file; whoever invoked you persists your reply.

{{
  "schema_version": "1.0",
  "verdicts": [
    {{
      "group_id": "<copy the item's group_id verbatim>",
      "decision": "approve|salvage|skip",
      "reason": "<one brief sentence>",
      "salvaged_description": "<the rewritten description — only when decision is salvage, otherwise null>",
      "dropped": "<what was intentionally left out — only when decision is salvage, otherwise null>"
    }}
  ]
}}

Rules:
1. Output EXACTLY one verdict per item above, keyed by the item's `group_id`
   copied verbatim. Verdicts are matched by that id, never by position.
2. `salvaged_description` and `dropped` are non-empty strings when — and only
   when — `decision` is "salvage". For "approve" and "skip" both are null.
3. Never omit an item. An item you cannot decide is a "skip" with the reason
   saying so.
'''


# ===========================================================================
# task generation
# ===========================================================================

#: The task-authoring rules, lifted out of ``instructions/generate-tasks.md``.
#:
#: Single source of truth, exactly like ``HUMAN_DECISION_JUDGE_RULES``: the
#: ``task_generation`` prompt inlines it verbatim and the "Guidelines for Task
#: Generation" section of ``instructions/generate-tasks.md`` is rendered from it,
#: with a test that fails when the two diverge.
TASK_AUTHORING_RULES = '''**Task Granularity**
- Produce 5-15 discrete implementation tasks, each completable in 15-60 minutes
- The first task (T001) is setup/foundation and has NO dependencies
- Each task modifies 1-5 files maximum
- Complex features are broken into multiple tasks
- Tasks should be independently testable when possible
- Give a task `executor_role: human` when:
  - The plan explicitly states something must be done manually
  - The task requires access to external services/dashboards an agent cannot reach
  - The task requires physical or manual verification
- Human task descriptions include clear step-by-step instructions, since a person (not an agent) will read them
- Human task acceptance criteria describe how the user can verify completion

**Dependencies**
- Use explicit `depends_on` arrays referencing task ids
- Independent tasks can share the same dependencies (this enables parallel execution)
- Never create circular dependencies
- Tasks with no dependencies can run in parallel

**Codebase Exploration**
- Look for existing patterns to follow (search for similar features)
- Identify integration points with existing code
- Check for related tests that need updates
- Note any shared utilities or components to reuse

**Task Descriptions**

Each description covers three components in 2-5 sentences:

1. **What** to implement — the concrete deliverable
2. **How** — key implementation approach, patterns/functions to reuse, APIs to call
3. **Interface contracts** — if this task produces or consumes interfaces, types, files or data structures other tasks use, state them explicitly

*Weak:* "Add validation to the form component." — this says *what* at a surface level and nothing about *how* or which interfaces are involved, so the implementer must guess the approach, which helpers exist, and whether other tasks depend on its output.

*Strong:* "Add Zod schema validation to UserProfileForm using the existing `validateWithSchema()` helper from `lib/validation.ts`. The schema enforces email format, required fields and max-length constraints matching the DB column limits in `T001`'s migration. Export the `UserProfileSchema` type from this file — `T003` (API handler) imports it for request body validation." — this covers what, how, and the interface contract.

**Acceptance Criteria**

Acceptance criteria are **task-scoped, specific and verifiable**: each one describes an observable outcome checkable with a concrete test, command or inspection. Avoid "works correctly" or "handles errors properly".

**Test scope rule**: a criterion may run a **specific, narrowly-scoped test** (a single test file or test class) related to this task's changes, but must NOT include broad regression tests like "full test suite passes", "all tests pass" or "existing tests still pass". Full regression testing runs once after implementation; individual tasks verify only their own changes.

*Bad:* "Form validation works correctly" / "Errors are handled properly" / "Performance is acceptable" / "Full test suite passes" / "All existing tests still pass".

*Good:* "Submitting the form with an empty email field shows the error message 'Email is required' below the input" / "`validateWithSchema()` returns `{ valid: false, errors: [...] }` when any required field is missing" / "The exported `UserProfileSchema` type is importable from `lib/validation.ts` and matches the column types in the `user_profiles` table migration".

**File References**
- Use paths relative to the project root
- Clearly distinguish files to modify from files to create
- Only reference files that actually need changes

**Plan Preamble**

`plan_preamble` is 3-5 sentences (roughly 200 tokens) of shared context injected into every task prompt: the overall goal/outcome, key architectural decisions, technology/library choices and global conventions.'''


def build_task_generation_prompt(
    *,
    plan_text: str,
    repo_root: str,
    plan_name: str = "",
) -> str:
    """Render the ``task_generation`` prompt.

    This is the one kind that must see the repository, so it is the one
    materialized as ``context_mode: workspace-read``: the plan text is embedded,
    but the repository is *explored*. The prompt therefore names exactly one
    path — ``repo_root`` — and instructs exploration in capability-neutral terms
    rather than in Claude tool names, which exist on no other host. It requires a
    runner declaring ``supports_workspace_reads``; a non-agentic runner does not
    fail on this prompt, it confidently invents a task list from the plan's
    prose, which is the worst output the workflow can produce because every
    later phase treats it as ground truth.

    Args:
        plan_text: The full plan text, embedded.
        repo_root: The pinned repository root — the only path this prompt names.
        plan_name: Optional plan file name, used only in the framing sentence.

    Returns:
        The complete prompt text.
    """
    subject = f"the plan `{plan_name}`" if plan_name else "the plan below"
    return f'''You are decomposing {subject} into implementation tasks.

## Plan

{plan_text}

## Repository

The repository root is: {repo_root}

Explore it before authoring tasks: search the repository for the files and
patterns the plan names, then read the ones a task would create or modify. Base
every task on what the code actually contains — never on the plan's prose alone.

## Task-authoring rules

{TASK_AUTHORING_RULES}

## Reply

Print ONE JSON object and nothing else — no prose, no markdown fences, no
explanation. Do not write any file; whoever invoked you persists your reply.

{{
  "schema_version": "1.0",
  "plan_preamble": "<3-5 sentences of shared context>",
  "tasks": [
    {{
      "id": "T001",
      "title": "<5-10 words>",
      "description": "<what, how, interface contracts>",
      "depends_on": [],
      "files_to_create": ["path/from/repo/root"],
      "files_to_modify": ["path/from/repo/root"],
      "acceptance_criteria": ["<specific, verifiable outcome>"],
      "estimated_complexity": "low|medium|high",
      "executor_role": "agent|human"
    }}
  ]
}}

Rules:
1. Ids are `T001`, `T002`, … — unique, zero-padded, in order.
2. `depends_on` names only ids declared in this document, forms no cycle, and at
   least one task has an empty `depends_on`.
3. `executor_role` is "agent" for anything an agent can do and "human" only for
   the manual cases listed in the rules above.
4. `depends_on`, `files_to_create` and `files_to_modify` are always present —
   `[]` when a task has none. Never drop a key because its list is empty.
'''


# ===========================================================================
# Rendering the rule constants back into the narrative files
# ===========================================================================
#
# "Prompt bodies move into Python generators, narrative stays": the reference
# and instruction files keep the same text as human-readable narrative, but they
# no longer *own* it. Each generated section sits between a BEGIN/END marker
# pair and is rendered from the constant above, with a test that fails when the
# checked-in text and the constant diverge.

_BEGIN = "<!-- BEGIN GENERATED: {name} — rendered from utils/subagent_prompts.py {const}. Edit the constant, not this block. -->"
_END = "<!-- END GENERATED: {name} -->"

#: ``section name -> (relative file path, constant name, constant value)``.
REFERENCE_SECTIONS: Dict[str, Any] = {
    "judge-rules": (
        "references/human-decision-batch.md",
        "HUMAN_DECISION_JUDGE_RULES",
        HUMAN_DECISION_JUDGE_RULES,
    ),
    "task-authoring-rules": (
        "instructions/generate-tasks.md",
        "TASK_AUTHORING_RULES",
        TASK_AUTHORING_RULES,
    ),
}


def reference_section_targets() -> Dict[str, str]:
    """Return ``section name -> relative file path`` for every rendered section."""
    return {name: spec[0] for name, spec in REFERENCE_SECTIONS.items()}


def render_reference_section(name: str) -> str:
    """Render one generated markdown block, markers included.

    Args:
        name: A key of :data:`REFERENCE_SECTIONS`.

    Returns:
        The exact text that must appear in the target file.
    """
    _, const_name, text = REFERENCE_SECTIONS[name]
    begin = _BEGIN.format(name=name, const=const_name)
    end = _END.format(name=name)
    return f"{begin}\n\n{text}\n\n{end}"


def extract_reference_section(document: str, name: str) -> Optional[str]:
    """Return the generated block currently present in ``document``, or None.

    Args:
        document: The full text of a target file.
        name: A key of :data:`REFERENCE_SECTIONS`.
    """
    _, const_name, _ = REFERENCE_SECTIONS[name]
    begin = _BEGIN.format(name=name, const=const_name)
    end = _END.format(name=name)
    start = document.find(begin)
    if start == -1:
        return None
    stop = document.find(end, start)
    if stop == -1:
        return None
    return document[start:stop + len(end)]


def replace_reference_section(document: str, name: str) -> str:
    """Return ``document`` with its generated block replaced by a fresh render.

    Raises:
        ValueError: when the document carries no marker pair for ``name``.
    """
    current = extract_reference_section(document, name)
    if current is None:
        raise ValueError(f"no BEGIN/END GENERATED markers for section {name!r}")
    return document.replace(current, render_reference_section(name))


def main(argv: Optional[List[str]] = None) -> int:
    """Rewrite every generated markdown block from its constant.

    ``uv run --project <skill-dir> -- python <skill-dir>/utils/subagent_prompts.py``
    """
    import pathlib

    try:
        from .stream_bootstrap import bootstrap_streams
    except ImportError:  # direct script invocation
        from stream_bootstrap import bootstrap_streams
    bootstrap_streams()

    skill_dir = pathlib.Path(__file__).resolve().parent.parent
    changed = 0
    for name, (rel_path, _const, _text) in REFERENCE_SECTIONS.items():
        target = skill_dir / rel_path
        original = target.read_text(encoding="utf-8")
        updated = replace_reference_section(original, name)
        if updated != original:
            target.write_text(updated, encoding="utf-8")
            print(f"rendered {name} into {rel_path}")
            changed += 1
    if not changed:
        print("all generated sections already match their constants")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual regeneration entry point
    import sys

    sys.exit(main(sys.argv[1:]))
