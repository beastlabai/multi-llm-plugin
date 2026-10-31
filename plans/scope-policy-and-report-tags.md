# Scope expansion policy and visible report tags

Status: implemented and verified 2026-10-31, including the policy/resume fixes
and packaged dry-run reporting fix found during QA. The final ordinary CI run
passed 26,699 tests with 281 skips.

Related: [implementation test report](scope-policy-and-report-tags-test-report.md)
and its [test evidence](scope-policy-test-evidence/).

## Goal and user contract

Keep multi-llm's default willingness to expand scope when required or warranted.
Add `--scope-policy {expand,preserve}`, defaulting to `expand` for a new run,
and visibly tag every suggestion in Markdown and HTML reports. Classification
is independent of validity, importance, selection, and application outcome.

This supersedes the earlier restrictive-default proposal. The motivating
[discussion](https://github.com/shinpr/awesome-codex-workflows/issues/54#issuecomment-6525248452)
is background, not a reason to change the default engineering philosophy.

- `expand`: warranted extensions remain eligible through existing validation,
  priority, user-selection, and judge rules. Permission to expand is not a
  direction to invent extra work or to approve every extension.
- `preserve`: only remedies established to be within scope are eligible.
  Reviewer prompts actively avoid proposing scope extensions; validation
  independently checks every returned candidate and rejects extensions or
  uncertain remedies for this run. Returned findings stay visible and are
  deferred; policy alone does not make an otherwise real finding invalid.
- Explicit user exclusions remain binding under both policies. Neither bulk
  approval nor `expand` overrides a prohibition such as “do not change the API”.
- Native and executor routes use the same contract, including detach, resume,
  commit wrappers, composites, and full workflows.

The same option controls both prompt checks. Under `preserve`, prevent expansion
at suggestion generation, reject remaining expansion at validation, and retain
the deterministic pre-write gate as the final enforcement layer. Under default
`expand`, both prompt roles continue to permit warranted extensions and label
them. Do not add another opt-in flag or a separate model round for these checks.

"When required" means a demonstrated gap, dependency, or benefit tied to the
user's goal, with a reason validation can assess. Completing an existing
requirement is ordinarily **within scope**, even if it adds substantial work.
Adding a new capability or changing an agreed outcome is an **extension**, even
if necessary to make the overall goal achievable. An optional but justified
extension can still be warranted; do not introduce a new severity threshold or
mandatory “necessary” score that silently makes the default restrictive.

### Default compatibility is an acceptance requirement

Omitting the option on a new run, or explicitly selecting `expand`, must retain
the current review, validation, judge, selection, and application behavior.
The descriptions of warranted expansion above explain the existing philosophy;
they are not additional acceptance criteria to inject into default prompts.

- Keep existing substantive review instructions, checklists, examples, and
  correctness/usefulness criteria. Reviewers may discover the same classes of
  features, improvements, refactors, and style findings they can discover now,
  subject to existing phase-specific rules. Add no new requirement that a
  default-mode finding be necessary for, or confined to, the original goal.
- Default reviewer/validator prompt text will not be byte-for-byte identical:
  it needs descriptive scope-classification instructions and output fields for
  the report tags. State explicitly that these labels are informational under
  `expand` and must not affect which findings are proposed or deemed valid.
  Do not inject the `preserve` suppression/rejection block, even as an example,
  into an `expand` prompt.
- Under `expand`, `scope_extension`, `unclear`, missing classification, or a
  stale scope binding adds no rejection, deferral, confirmation, model round,
  or write prerequisite. Missing scope-only context degrades to `Scope unclear`
  while existing correctness-context checks retain their current behavior.
  The scope-policy apply gate is a pass-through.
- Scope annotation alone must not select, narrow, rewrite, reprioritize, or
  replace a default-mode remedy, or change the existing judge criteria. Remedy
  binding/narrowing machinery introduced to enforce `preserve` must not change
  the default write instructions. Existing judge salvage continues as before.
- Explicit exclusions and user skips retain their existing authority. This
  feature adds no separate default-mode conflict filter or broader permission
  to ignore user instructions. Keep scope labels separate from those rules.

Only an effective `preserve` policy enables the new scope-based reviewer
suppression, validation rejection, and application restriction. Exact LLM
wording/output cannot be guaranteed identical after adding annotation fields;
the compatibility contract concerns instructions and workflow decisions.

## 1. Verified ownership and boundaries

Except for paths beginning `docs/`, `packaging/`, or `plans/`, paths below are
relative to `skills/multi-llm/`. Follow its `AGENTS.md` and relevant development
contracts, especially review-results, delegation, unattended, workflows, hosts,
and write-scheduler.

| Concern | Actual owners / integration points |
| --- | --- |
| Reviewer prompts | `prompts/{plan_review,task_review,code_review,commit_review}.txt`; review orchestrators' prompt-context builders; `utils/prompt_loader.py` loads cached template text, not policy-specific rendered prompts |
| Classification/validation | `utils/subagent_prompts.py`, `utils/validation.py` (legacy and batched routes, snapshot binding, merge/revalidation); `utils/check_reply.py` and `utils/schema_validator.py` for reply contracts |
| Group identity and metadata | `utils/suggestion_processor.py:SuggestionGroup`, `utils/state_manager.py` stamping/hashes; `utils/apply_selection_helpers.py:merge_validation_with_groups` is a **separate** apply-time merge from `validation.apply_validation_to_groups` |
| Apply eligibility and remedies | `utils/filtering.py`, `utils/apply_orchestrator_base.py`, `utils/human_decisions.py`, the three apply orchestrators' formatters, `utils/suggestion_batcher.py`, `utils/code_fix_batcher.py` |
| Shared write handoff | `utils/subagent_tasks.py`, `utils/write_phase_bridge.py`, `utils/write_batch_scheduler.py`, `utils/write_prompt.py`, `utils/batch_done.py`; native `util write-prompt` binds the same materialized task used by the executor |
| Markdown | `utils/review_orchestrator_base.py` (plan/tasks), `code_review_orchestrator.py:generate_review_report` / `_format_issue` (code/commits), **`utils/consolidation.py:generate_consolidated_report`** (consolidated) |
| HTML | `utils/html_report_generator.py` (flat/PR, `report_data.json`, decision overlays); **`utils/consolidation.py:generate_consolidated_html`** (separate consolidated payload/template path) |
| CLI and continuations | `utils/cli_common.py`, `multi_llm.py`, `modes.py`, `utils/detach.py`, `utils/next_phase.py`; `utils/improve_workflow.py` / `utils/improve_steps.py`, `full_workflow_orchestrator.py`, `check_workflow_prerequisites.py` |
| Commit wrappers | `review_commits_orchestrator.py`, `apply_commit_fixes_orchestrator.py`, `improve_commits_orchestrator.py`, `utils/commit_review.py`, `utils/commit_brief.py`, `utils/commit_range.py`; these front review-code/apply-code-fixes, not a second review pipeline |
| Generated distribution | `packaging/common/surface.py`, Codex/OpenCode build scripts, `hosts/render_capabilities.py`; edit source, regenerate owned outputs |

There is no `improve_orchestrator.py`. The three composites are `improve-plan`,
`improve-code`, and `improve-commits`; there is no improve-tasks mode to add.
Do not duplicate classification, flag parsing, or enforcement in host shims.
Do not change grouping similarity, stable-ID algorithms, provider permissions,
or file-containment rules as part of this feature. A product-scope policy is
not a replacement for existing review-file scope or write safety.

## 2. Baseline and semantics

The baseline is the intended outcome plus explicit limits, not the changed-file
list, the amount of work, or a finding's importance/type.

- **Plan:** the reviewed plan's goals, requirements, and exclusions.
- **Tasks:** the source plan constrains the task list. A task that adds an
  unauthorized feature cannot make that feature in-scope merely by listing it.
- **Code:** the accepted plan/requirements and recorded decisions govern intent;
  the reviewed diff and task list are evidence. Implementation drift cannot
  authorize itself. Fixing an introduced regression is ordinarily within scope.
- **Commits:** the pinned brief/messages, focus, and frozen context plan when
  supplied. A focus limits review attention; it does not authorize new goals.
  Vague messages alone cannot justify confident product-scope classifications.

An additional-file prerequisite can be in-scope; an unrelated capability in an
already changed file can be an extension. Removing existing scope creep is an
in-scope correction even when its finding has `type: scope`. Unrelated style
cleanup can be a scope extension; under `expand`, it remains subject to the
current phase-specific usefulness/validity rules, not a new goal-relevance test.
Keep existing phase applicability rules: for example, plan review still does
not require a task breakdown simply because expansion is allowed.

Reuse `validation.bind_document_file`, bound context sources, pinned commit
records and existing snapshots. Plan validation already binds
`validation-snapshot.md`; task validation also binds the task document, reusing
`tasks-backup.md` where appropriate. Record a small **scope baseline record**
in the existing phase/task/report metadata: source roles, snapshot references,
content hashes, pinned endpoints where relevant, and a host-computed baseline
identity. Bind reviewer and validator context to the same review-round inputs;
do not hash one read while sending another. Do not create an LLM intent-summary
phase, require a new plan template, or add a global baseline registry.

Keep scope intent fixed through that review/apply round. Existing revalidation
can deliberately judge correctness against the **current** document after an
apply (`revalidation-snapshot.md`); that must not replace the original scope
baseline. Give revalidators both labelled contexts when they differ. A new
review may adopt a revised accepted plan as its baseline; old reports retain
theirs. A changed live file during an unfinished round must not silently rebase
its verdicts. Missing/truncated relevant intent produces `unclear`, not an
invented requirement. Use existing context coverage reporting and read-capability
checks; a source hash proves identity, not that a model read it correctly.

Hard constraints must reach all relevant prompts from the recorded inputs.
Constraints existing only in a chat cannot be inferred by a standalone CLI:
the invoking host must include them in its supplied plan/context. Amend a hard
constraint explicitly in the authoritative input and start a new review; this
feature does not need an additional “approve scope amendment” UI or flag.

## 3. Metadata, remedies, and grouping

Use a small shared `utils/scope_policy.py` for normalization, labels, baseline
binding, eligibility, and role-specific prompt guidance. Suggested classification
fields:

```json
{
  "scope_relation": "scope_extension",
  "scope_reason": "Adds PDF export beyond the requested CSV outcome.",
  "scope_reference": "Plan: Output formats"
}
```

The values are `in_scope`, `scope_extension`, and `unclear`. New classified
verdicts require a short reason; an exact source reference is optional. Missing,
malformed, unbound, or stale classification is `unclear`. Do not infer scope
from validity, consensus, severity, finding type, or model confidence.

Reviewers provide provisional classifications. The existing validation round
returns the authoritative **group** classification with the validity verdict.
Store the host-verified `scope_baseline_id` and a digest of the classified
proposal with the accepted verdict; do not trust a model echo of a digest as
proof of binding. Join authoritative metadata by the stamped group hash and
bound task inputs. The apply-time merge currently also falls back by index;
**never transfer scope authority from an unmatched hashed verdict by index**.
Hash-less legacy validity remains readable, but cannot establish scope authority.

Classify the proposed action, not just whether the reported problem exists.
Review prompts currently allow findings without a proposed fix; retain that
useful behavior. Validation may identify the bounded correction implied by the
finding, but must use `unclear` if materially different remedies have different
scope and none is selected. Do not require reviewer-written patches.

Make the proposal being classified and the proposal being written the same:

- Normally classify the complete group proposal that the apply formatter would
  send, including all retained member descriptions. Reuse a shared extraction
  helper so validator and writer do not implement different interpretations.
- If validation selects/narrows a remedy, persist it explicitly (for example,
  `selected_remedy`) alongside the verdict, retain the original evidence, and
  make the formatter send that remedy as the instruction. Original alternatives
  may appear as evidence clearly marked as not instructions. In particular,
  `apply_code_fixes_orchestrator.format_fix_for_output` currently concatenates
  **all** descriptions; that must not reintroduce a discarded extension.
- Any new selected-remedy enforcement/narrowing in this feature is for
  `preserve`. Under `expand`, scope annotation describes the existing proposed
  action without changing it; ambiguity may be labelled `unclear` without
  requiring a newly chosen remedy or additional validation call.
- Bind scope authority to the full relevant proposal, not the stable group hash
  alone: existing hashes use truncated description content. Markdown/HTML
  description edits, per-member skips, and judge salvage invalidate this binding
  when they change the proposal. Validity-only changes cannot establish scope.
- Reuse an existing validation/revalidation handoff for a changed proposal, or
  leave it `unclear` and defer under `preserve`. Initial delivery need not add
  automatic scope-only retries for every edited/legacy item. A judge may narrow
  a suggestion, but approval or salvage alone is not a fresh scope verdict.

Preserve each member's provisional classification with its own textual tag;
label the authoritative group verdict separately. For an unsplit proposal,
any definite included extension makes it `scope_extension`; unresolved remedy
alternatives make it `unclear`. Do not use majority vote or an in-scope member
to hide an extension. No regrouping algorithm change is needed.

Consolidation is presentation and selection mapping over underlying groups,
not new scope authority. Retain their classifications and IDs; an umbrella
containing an extension displays `Scope extension` (explain “contains extension”),
otherwise one with unknown members displays `Scope unclear`, otherwise
`Within scope`. Mixed groups retain member detail. An umbrella approval maps to
underlying groups and the policy gate evaluates each; it cannot approve a
synthesized expansion absent from the validated proposals.

Keep conflicts with explicit exclusions distinct from ordinary scope extension.
Preserve their existing treatment in validation, judge, user decisions, and
writer instructions under both policies; `expand` never grants permission to
ignore an explicit prohibition. Do not add a separate scope-conflict field/gate
that changes default eligibility as part of this feature. Under `preserve`, the
authoritative baseline and scope gate additionally restrict proposed remedies.

Extend only the affected schemas/serializers: reviewer schemas
`review_suggestions.schema.json` and `code_review_issues.schema.json`, validation
reply parsing/merges, `SuggestionGroup`, and report/consolidated mappings.
Extend `subagent_tasks.schema.json` and `state_file.schema.json` only where their
existing shapes need the new task/policy records; keep old records readable. Check
`utils/structured_output.py` strict-schema adaptation as well as fallback
transports. Validation's reply contract lives in code, not a currently existing
`validation.schema.json`. Normalize malformed scope independently so it cannot
discard otherwise usable legacy findings. Keep all scope/decision fields out
of stable IDs. Update formatted JSON examples with correctly escaped braces.

### 3.1. Two policy-aware prompt checks

The earlier plan classified scope at review and validation but did not explicitly
prevent reviewers from generating extensions. Add that prevention and an
independent validation-stage rejection under `preserve`. These are instructions
for existing model calls, backed by the existing planned application gate;
neither prompt compliance nor a reviewer-provided tag is an enforcement guarantee.

**Shared inputs and rendering:** resolve policy before constructing prompts.
Pass the effective `scope_policy`, the same frozen scope baseline, and the
role-specific guidance through every review and validation prompt builder.
Keep the common scope definition and policy semantics in `utils/scope_policy.py`,
with separate reviewer and validator guidance so their responsibilities are
explicit. Add a formatted policy-guidance block to all four reviewer templates
and all shared/batched, legacy, and revalidation prompt paths. Update callers,
context-key contracts, and prompt fixtures together. A file manifest describes
what to inspect, not permission to introduce new product behavior.

Cache only policy-independent template text. Render the role/policy block per
invocation and bind materialized prompt/task inputs to that policy; a previous
`expand` rendering must never be served as a `preserve` prompt. Do not rely only
on a host's surrounding instructions or on an apply-only CLI flag.

The `expand` block adds annotation guidance only. For example:

> Follow the existing review/validation instructions and criteria. For report
> labels, also classify the suggestion's scope relative to the supplied
> baseline. This classification is informational: do not suppress a finding,
> change its validity or importance, or narrow its remedy because it extends
> scope or because its scope is uncertain.

The suppression/rejection instructions below are rendered only for `preserve`,
not appended as conditional warnings to default prompts. Leave the existing
validation definition of `valid` unchanged under `expand`.

**Check 1 — reviewers prevent new scope creep:**

- With `preserve`, explicitly request only findings whose proposed correction
  stays within the agreed outcome. Do not propose additional features,
  unrelated refactors, or optional cleanup outside that outcome, even when
  beneficial. Broad existing review checklists are subordinate to this rule.
- Continue finding missing requirements, introduced bugs/security regressions,
  and existing scope creep in the artifact. A suggestion to remove unauthorized
  behavior brings the work back within scope and must not be suppressed by a
  blanket ban on findings whose type is `scope`.
- Do not hide a concrete failure to meet an agreed requirement merely because
  the remedy is uncertain or appears to require an extension. Report the
  problem as a blocker with the evidence and an honest `unclear` or
  `scope_extension` remedy classification; request a scope decision rather
  than prescribing an unauthorized expansion as an automatic fix. These
  exceptional blocker findings remain subject to validation and deferral.
- With `expand`, leave the existing broad review instructions and eligibility
  criteria intact. Add scope annotation only, explicitly described as
  informational. Do not introduce a new goal-connection/necessity requirement,
  suppression rule, or stronger burden of justification than the current
  prompt. Classify returned suggestions in either mode. Returning no findings
  is acceptable.

Example restricted reviewer guidance:

> Scope policy: preserve. Find problems in delivering the stated requirements
> and suggest corrections within their boundaries. Do not propose new product
> outcomes or unrelated cleanup. Still report existing scope creep and how to
> remove it. If an agreed requirement cannot be met without extending scope,
> report the blocker and required decision; do not silently expand the task.

**Check 2 — validators independently reject leaked scope creep:**

- Re-evaluate each proposed remedy against the frozen baseline, independently
  of the reviewer's `scope_relation`, confidence, consensus, and wording. Do
  not assume a reviewer obeyed its restricted instructions, and do not accept
  a description that merely asserts its new feature is "required".
- With `preserve`, explicitly reject a `scope_extension` remedy for this run
  and defer an `unclear` remedy. A real or high-severity issue does not make an
  expanding remedy eligible. Preserve unresolved blockers for reporting.
- Keep correctness and policy rejection separate: return the ordinary validity
  verdict plus authoritative scope classification/reason. A factually correct
  suggestion can be `valid` and rejected by scope policy. Do not change it to
  `invalid` solely to make filtering drop it; that would erase useful findings
  and prevent a later explicit policy override from working correctly.
- After normalizing validation output, derive the scope-policy rejection with
  the shared eligibility helper and record/display it in the validation/report
  results. This is a validation-stage decision, not just a badge waiting for
  the apply step. No redundant model-generated allow/deny field is necessary.
  The pre-write gate recomputes that decision from the current policy and
  bound authoritative metadata, so stale approval cannot authorize a write.
- With `expand`, judge extensions under existing validity rules and retain
  scope tags; do not reject them just because they extend the baseline. Known
  hard exclusions remain binding under both policies.

Example restricted validation guidance:

> Scope policy: preserve. Verify the suggestion's facts, then independently
> check the full proposed remedy against the original requirements and limits.
> Reject remedies that extend those limits for this run, even if their facts
> are correct or reviewers labelled them in scope. Return correctness and scope
> separately, with the evidence for the scope decision. Uncertain scope must
> not be approved for automatic application.

Under `preserve`, use the same scope check on changed/narrowed remedies when
revalidation is performed. Under `expand`, existing revalidation behavior is
unchanged and a missing or invalidated scope annotation simply becomes unclear.
Neither consolidation, judge salvage, nor an edited description can bypass
validation by repeating an earlier in-scope label. An explicit apply-policy
override can recompute eligibility from an unchanged, already-bound scope
verdict without another model call, because the underlying classification is
independent of whether extensions are currently allowed.

## 4. CLI, persistence, and propagation

Declare the option once in the shared argument helpers, with omission represented
as `None` until resolution. Support review/apply plan, tasks, code, commits,
the three improve composites, and `full` (legacy `--full` spelling included).
Generation/implementation must accept and carry the effective policy when driven
by full workflows or their follow-up commands; they follow the accepted plan/tasks
and must not silently invent extra outcomes under `preserve`. No new classification
system for every generated task is required. Document the flag's applicable modes.

Resolution order for the target review round:

1. Explicit option on this invocation.
2. The current apply/composite continuation's recorded effective override.
3. The originating review's recorded policy when starting its apply phase, or
   the driving composite/full workflow's explicitly forwarded effective policy.
4. `expand` for a genuinely new run without a choice, and for legacy records.

Store policy and its source with existing phase/workflow records, keyed to the
review round, not a sticky project-wide setting or whichever sibling run was
most recently active. Resolve a composite once and pass that effective value
to its children. A forced new review without a policy is a new default-expand
run unless driven by an existing workflow's explicit inherited policy; a
reaggregate/revalidation of the same round keeps its policy and baseline.

A read-only review resolves policy before fan-out: it controls reviewer and
validator prompts as well as intended apply policy. Its reports show every
returned candidate, including extensions caught and rejected by validation;
they do not invent entries for suggestions the restricted reviewer never
generated. Do not silently hide returned extension findings after generation.
An explicit apply override changes only eligibility for remaining work; record
it without rerunning reviewers, relabelling findings, or undoing completed work.
Moving a completed `preserve` review to an `expand` apply can release returned
deferred extensions, but cannot recover suggestions never generated. To solicit
broader suggestions, start a fresh review round explicitly with `expand`.
Record the originating review-prompt policy separately from the effective apply
policy so an override cannot misrepresent what that reviewer was asked to find.
Do not mix reviewer outputs requested under different policies in one unfinished
fan-out round: use the existing new-round mechanism for an explicit review-policy
change after dispatch; omission on resume inherits the recorded review policy.
Handle a previously completed bounded apply that still has scope deferrals:
`expand` must offer those remaining items again, rather than hit an unconditional
“already complete” guard. The same override must survive subsequent omission.

Forward through `improve_workflow` step/cross-cutting/continuation flags,
`util improve-steps`, full workflow `common_flags`/prerequisite apply runners,
commit-wrapper argument rebuilding, detach re-entry, replay flags, printed next
commands and tool argument arrays. Update `next_phase` completion logic where
scope deferrals affect which work remains. Keep argument lists, not reconstructed
shell strings. Do not confuse the new policy with existing `--files` scope flags.

Register the decision site in `utils/unattended.ROWS` and `DECISION_FLAGS`, with
its actual parser flag and recorded automatic resolution, following the guide.
No new interactive question or TTY inference. Invalid/empty values use the
existing exit-3 decision-flag contract. `--yes`, `--no-confirm`, `--agent-decide`,
bulk validity approval, `--force`, and `--accept-stale-consolidation` cannot bypass
an effective `preserve` gate or a known hard exclusion.

### 4.1. Usage examples to document

These are proposed arguments for the installed host skill, not commands the
current implementation already accepts. Prefix them with
`$multi-llm:multi-llm` in Codex, `/multi-llm:multi-llm` in Claude Code, or
`/multi-llm` in OpenCode. Standalone phase examples assume the normal preceding
artifacts exist; the full-workflow example is an alternative to running them
individually. Put detailed user-facing examples in `docs/workflows.md`.

```text
# New plan review: existing default behavior, scope tags are informational
--review-plan plans/feature.md

# Plan review and application with a preserved scope
--review-plan plans/feature.md --scope-policy preserve
--apply-suggestions plans/feature.md --scope-policy preserve

# Task generation, review, correction, and implementation
--generate-tasks plans/feature.md --scope-policy preserve
--review-tasks plans/feature.md --scope-policy preserve
--apply-task-suggestions plans/feature.md --scope-policy preserve
--implement plans/feature.md --scope-policy preserve

# Code review and application in a single unattended command
--improve-code plans/feature.md --scope-policy preserve

# Alternative: preserve scope throughout the full unattended pipeline
--full plans/feature.md --scope-policy preserve --yes

# Commit review and application, without requiring a plan
--improve-commits main..HEAD --scope-policy preserve

# Explicitly allow expansion when applying remaining code-review findings
--apply-code-fixes plans/feature.md --scope-policy expand
```

Explain that omission on a new run means `expand`, whereas a continuation
inherits its recorded choice. The explicit `preserve` on each standalone example
makes its intent clear; users need not repeat it on continuations that inherit
that choice. An apply override affects only returned remaining findings, not
new suggestions a restricted reviewer never generated. A fresh broader review
requires the existing force/new-round controls as well as `expand`; do not imply
that repeating a completed standalone review automatically starts a new round.
`improve-*` modes edit their target unattended, while review modes remain read-only.

## 5. Eligibility, writes, and resume

Retain existing priority/validity/user-skip/judge rules. Apply the independent
scope gate after all changes to the selected proposal and immediately before
batch formatting/materialization. Account for earlier no-items exits,
confirmation text, and dry-run output so they describe the same eligible set.

| Effective policy | Scope-only eligibility |
| --- | --- |
| `expand` | All relations continue through existing selection rules; uncertainty alone introduces no new default fail-closed gate. |
| `preserve` | Only an authoritative `in_scope` proposal bound to this baseline and exact selected remedy may proceed. Others defer. |

Existing explicit constraints continue to govern both policies; the new
scope-policy gate is a pass-through under `expand`. Under `preserve`, a severity
threshold or validation override cannot bypass scope restrictions. A LOW
in-scope fix remains eligible at the existing default `--min-priority low`;
explicitly selecting a higher minimum must still exclude it.

Materialized tasks and writer prompts carry effective policy, baseline references,
selected remedy and its binding, and scope rationale. Under `preserve`, a writer
that discovers a necessary expansion must decline/defer the item with its reason.
Use existing declined/partial-write recovery; edits already made cannot be
reported as a clean decline. Keep semantic scope checks separate from existing
filesystem containment and commit guards.

Persist scope deferrals separately from invalid, user-skipped, applied, and
failed work, using stable group IDs, baseline identity, and reasons. They must
not enter processed/applied IDs merely to suppress repeated dispatch. Surface
counts and details in dry runs, structured results, apply summaries, and reports.
Record policy rejection as soon as validation determines it, even for a
read-only review with no apply phase. Policy changes reconsider deferrals;
user skips remain skips. Pre-write enforcement must cover imports and old
artifacts too, not just findings produced by the new restricted prompts.

A bounded apply can finish successfully with optional extensions deferred.
Completion records must say “eligible work completed” and retain the deferrals,
so `check_workflow_prerequisites.py` does not repeatedly count all valid groups
as unapplied work. Preserve its existing bounded one-auto-apply/recheck behavior.
When an actual dependency or writer/validator explanation establishes that the
accepted goal cannot be completed within the boundary, surface that concrete
blocker and exit 3 rather than claim the full goal complete. Do not infer a
blocker from HIGH priority or build a general LLM dependency planner. Record
blocker evidence in existing phase outcomes; dependent phases must respect it.

Resume safety is required, but avoid a general batch-replanning subsystem:

- Persist policy/baseline/proposal binding on write tasks and check it both in
  executor dispatch and native `util write-prompt` when enforcing `preserve`.
  Missing scope-only metadata must not block a default `expand` continuation
  that the existing task lifecycle permits. Cached prompts from before an
  explicit policy change must not enforce the wrong policy; update the native
  instructions too and retain existing stale-task checks unrelated to scope.
- Retain completed task/result/member IDs and partial-write evidence. A changed
  policy cannot renumber pending work onto completed batch IDs or make completed
  groups run again. Existing executor resume relies on batch result files while
  native resume also uses processed records; test both explicitly.
- With no started work, obsolete prepared tasks may be retired and rematerialized.
  With completed work, preserve its identities and rebuild only safely separable,
  never-started work. If existing batching cannot do that safely, exit 3 with a
  concrete recovery path (settle partial work or start a fresh review of the
  current artifact). Do not silently discard evidence or replay an old prompt.
- Old reports default to `expand` without migration/model calls. Under explicit
  `preserve`, missing classification defers and explains “fresh classified review
  or explicit expand”. A legacy pending task file cannot acquire authority from
  a newly supplied flag alone.

## 6. Markdown, HTML, selections, and regeneration

Use these identical textual labels everywhere: `Within scope`, `Scope extension`,
`Scope unclear`. Scope tags describe proposals; keep validity, selection,
deferred/applied outcome, and hard-conflict notices separate. Include a legend
and effective-policy line. Counts use distinct underlying groups, not code
issue rows or duplicated file/section cards. Avoid a new dashboard/filter.

Also show the originating reviewer-prompt policy when it differs from the
effective apply policy. A `preserve` review states that additional scope was
not solicited; its extension counts cover returned findings, not every possible
improvement. For returned candidates rejected by validation, show both the scope
tag and the policy outcome, such as `Scope extension` plus “Deferred: scope
preservation requested”. A real finding must not be presented as a false
positive just because this run disallows its proposed extension.

**Markdown:** inline-code labels and brief reasons, on both individual findings
and group/umbrella summaries. Preserve issue numbers, hashes, heading text,
numbering footer, and the checkbox immediately after `### N.`. Put scope metadata
**before** the existing `**Validation:**` / `**Importance:**` description boundary,
or extend the parser narrowly to exclude a precisely delimited generated block.
`report_parser.parse_suggestion_descriptions` and `parse_issue_descriptions`
currently take everything after that boundary as editable description: simply
adding a metadata line there would create a false user edit on every apply.
Flatten/escape generated reason/reference text so it cannot create headings or
checkboxes. Do not globally strip user text containing the word “scope”.

**HTML:** update flat, PR, and consolidated templates, all alternate views/cards,
individual suggestions, and umbrella summaries. Use text badges with accessible
contrast and concise reasons in existing details. Reviewer tags are labelled
provisional; group remedy tags are labelled validated. An edited description
shows that the existing classification may be stale, rather than a confidently
applicable in-scope badge. Under `preserve`, show deferred eligibility even when
validity is approved; keep selection controls usable for future policy overrides.
The server-side gate remains authoritative.

Persist baseline, classifications, selected remedies/bindings, and policy/outcome
overlays in `report_data.json` for flat/PR reports. **Consolidated HTML currently
embeds its payload directly and has no equivalent persisted sidecar or decision
regeneration.** Add a dedicated `consolidated_report_data.json` and extend the
existing regeneration entrypoint to overlay underlying group IDs in both files;
reuse normalization/embedding helpers rather than inventing another report
pipeline. Exclude the new sidecar from every reaggregation input scan (existing
`consolidated` exclusions may already cover it; verify with a fixture).

Regeneration uses persisted review payloads plus recorded decisions/outcomes,
never a changed live plan or new model inference. Recompute scope eligibility
when the effective policy changes, keeping historical classification intact.
Old payloads render unknown tags; a missing legacy consolidated sidecar is a
reported limitation/no-op rather than a fabricated reconstruction.

Existing HTML “Export” buttons export **selection JSON**, not an HTML report.
Keep their ID/selection schema compatible; imported tags or policy-like fields
must never override authoritative metadata or the CLI policy. Exercise exports,
localStorage restoration, edited descriptions, and consolidated selection
mapping. Static/saved HTML retains its embedded tags; no new HTML export feature
is needed. Preserve existing JSON/script escaping and escape rationale text.

## 7. Implementation sequence and verification

1. Add the shared contract and baseline/proposal bindings; extend serialization,
   reviewer prevention and independent validation-rejection prompt blocks,
   prompt outputs, strict/fallback transports, and **both** validation merge
   paths. Reuse existing round snapshots and task manifests. Both prompt roles
   consume the same resolved policy and baseline, without adding model rounds.
2. Thread policy resolution through parsers, persisted workflows, continuation
   builders, commit wrappers, prerequisites, and generated surfaces.
3. Add eligibility/deferral handling, selected-remedy formatting, writer context,
   native/executor stale-task checks, and completion/resume behavior.
4. Update all three Markdown generators, HTML mappings/templates, persisted
   consolidated payloads, selections, decision regeneration, and summaries.
5. Document the expansion default, restriction, legacy uncertainty, and recovery
   in README, `docs/workflows.md`, `docs/commit-reviews.md`, relevant instructions,
   source shims, and affected development contracts, following the README budget
   below. Regenerate distribution.

### Documentation placement and README budget

- **README.md:** target zero net growth in both words and lines by tightening
  existing prose and reusing the Common options table and existing workflow-doc
  links. Mention only `--scope-policy expand|preserve`, the expansion default,
  and a link to the detailed explanation. No new README section, command block,
  screenshots, or repeated phase-by-phase examples. If zero net growth cannot
  be achieved without removing useful existing information, allow at most one
  compact option-table row of about 30 words; offset it by editing redundant
  wording where practical.
- **docs/workflows.md:** own the complete scope-policy explanation and usage
  examples under a `Scope policy` subsection: preserved default behavior,
  reviewer/validator checks under `preserve`, report tags, phase examples,
  policy inheritance, explicit overrides, and legacy/unclear findings. This is
  the single user-facing source for shared scope semantics.
- **docs/commit-reviews.md:** add a short commit-specific example/explanation
  and link to the workflow subsection; explain pinned intent/context behavior
  without copying the entire policy guide.
- **Instructions, source shims, help, and development docs:** update the actual
  argument/routing and prompt/report contracts in the relevant existing files.
  Regenerate owned mirrors/surfaces; do not create another top-level guide or
  duplicate the full workflow explanation in each host document.

Before implementation edits, record README word and line counts. Compare them
afterward and report any net increase and its justification, targeting zero.
Check documentation examples against the supported parser/tool arguments and
mode applicability once implemented, and verify the new documentation links.
The current planning task edits this plan only; do not document an unimplemented
option as already supported in the actual README or runtime user guide.

Tests should exercise observable contracts, not mirror helper implementations:

- **Default vs override:** mocked end-to-end apply with no option applies a valid
  extension; preserve applies an in-scope item and defers extension/unknown;
  invalid/user-skipped items stay excluded under both. Repeat via unattended
  composite and native materialization. An explicit higher priority still works.
- **Default compatibility:** keep pre-feature fixture expectations for ordinary
  reviewer/validator instructions and selected/applied IDs, priorities, judge
  decisions, confirmations, and model-call counts. With identical recorded model
  results and user choices, no-option and explicit `expand` produce those same
  decisions after adding scope metadata. Exercise `in_scope`, `scope_extension`,
  `unclear`, malformed/missing tags and stale scope bindings, including existing
  LOW style findings and optional improvements. The added metadata/report tags
  may differ; the substantive default decisions and remedy instructions must
  not. Inspect default rendered prompts to confirm existing criteria remain
  intact, annotation is informational, and `preserve` prevention/rejection text
  is absent. Fixtures cannot promise identical free-running model outputs.
- **Two prompt checks:** inspect the actual rendered prompts for plan, task,
  code, and commit review plus shared/batched, legacy, and revalidation routes.
  `preserve` includes reviewer suppression and independent validator rejection;
  default/explicit `expand` permits warranted extensions. Both roles receive
  the same frozen baseline and policy, including native and executor task
  manifests. Render alternating policies in one process to catch cached or
  reused policy-specific prompts. Assertions must check the operative
  instructions and examples, not merely the presence of the option's name.
- **Leaked extension:** inject a reviewer result that mislabels an expanding
  remedy `in_scope`. The validation fixture returns an independent
  `scope_extension` verdict, and the `preserve` review itself records policy
  rejection while retaining the factually `valid` finding and visible tag.
  Applying with bulk approvals or judge approval still dispatches no write for
  it. With `expand`, the same correct extension remains eligible. This proves
  plumbing and enforcement, not that a model always identifies extensions.
- **Coverage and overrides:** required work, introduced regressions, and
  removal of existing scope creep remain solicited under `preserve`; a real
  blocked requirement is reported rather than hidden. A later apply override
  can release returned deferrals but does not fabricate omitted reviewer
  candidates. Broader solicitation requires a fresh `expand` review. Existing
  calls gain prompt instructions, not an extra validation round per finding.
- **Semantics:** prompt fixtures cover justified optional expansion, required
  existing work, added-file prerequisite, removing scope creep, unrelated cleanup,
  hard exclusion, vague commit intent, task/implementation drift, and mixed
  remedies. Missing context remains unclear. Fixture verdicts prove plumbing,
  not semantic model accuracy; review the rubric/examples manually.
- **Authority:** shuffled/mismatched group hashes cannot transfer scope verdicts;
  metadata-only changes leave stable IDs unchanged. Description edits beyond
  the ID hash's truncated prefix, HTML edits, member skips, and judge salvage
  invalidate the proposal binding. A narrowed remedy's write prompt excludes
  the discarded extension. Known hard conflicts survive validity/bulk overrides.
- **Lifecycle:** cover omission vs explicit expand, review-to-apply inheritance,
  consecutive resumes after an override, forced new review, reaggregate,
  current-document revalidation with original scope baseline, commit wrappers,
  full prerequisite apply, detach argv and tool/native next commands. A completed
  preserve run can later apply its deferred items without repeating landed work.
- **Resume:** test completed and pending batches together, partial writes,
  refusal of old task files in both dispatch paths, and explicit recovery where
  safe reconstruction is impossible. Optional deferrals finish without loops;
  a concrete blocker prevents dependent phases and a false completion claim.
- **Reports:** new/legacy data in plan/task/code/commit/consolidated Markdown and
  all three HTML templates; unchanged Markdown round-trips with **zero detected
  description edits**, while real edits and all checkboxes still round-trip by
  correct IDs. Test duplicate issues per group and consolidated mixed members.
- **Persistence/UI:** grouping, validation, consolidation, reaggregation, report
  payloads, and decision overlays retain metadata. Regenerate with source plans
  unavailable. Run executable template JavaScript/DOM fixtures for badge rendering,
  edited-description staleness, JSON export/import, and restored local selections;
  substring checks alone do not prove the UI works. Test HTML/script/Markdown
  escaping. No live provider spending or manual browser session is required.

Start with `test_filtering.py`, `test_subagent_prompts.py`,
`test_suggestion_processor.py`, `test_report_parser.py`, validation binding/merge
and context tests, consolidation/HTML tests, apply/write-prompt/resume tests,
`test_improve_orchestrator.py`, `test_full_workflow_orchestrator.py`, commit and
unattended/CLI-surface tests. Add focused scope-policy fixtures where needed.
Run the ordinary suite after focused checks pass:

```bash
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests -m "not live and not container and not package and not codex_jail"
uv run --project skills/multi-llm -- python skills/multi-llm/hosts/render_capabilities.py --check
```

For implementation, regenerate help goldens using the normalization documented
in `tests/golden/README.md`; do not indiscriminately recapture unrelated outputs.
Update prompt goldens through `MULTI_LLM_UPDATE_GOLDEN=1` only for intentional
changes. Regenerate changed capability docs using `hosts/render_capabilities.py`.
`packaging/common/surface.py` derives tool schemas from parsers; verify the flag
is accepted through both packaged tool routes. Use `make codex-regen` for the
tracked Codex mirror and `python packaging/opencode/build.py --allow-dirty`
(under the same `uv run --project skills/multi-llm --` prefix) for OpenCode staging.
The builders stage tracked inputs: new source files must be included in the
implementation's version-control inputs before package verification; an untracked
helper is not packaged by `--allow-dirty`. Do not hand-edit generated mirrors,
publish, or perform release/version work just to deliver this feature.

At the user's request, this implementation plan, its related QA report and its
evidence are kept together under `plans/`. The [QA report](scope-policy-and-report-tags-test-report.md)
records verification, discovered defects and their fixes.

## Review note — 2026-10-30

Reviewed directly against the shared checkout. Preserved default `expand` and
clarified required work vs warranted extensions and hard exclusions. Corrected
composite/merge/report ownership; closed gaps in proposal binding, narrowed
remedies, original-baseline revalidation, policy inheritance and deferred-work
resume, Markdown description parsing, and consolidated HTML persistence/export
semantics. Reduced machinery by reusing existing snapshots/validation, deriving
consolidated labels, and allowing an explicit recovery stop when rebatching is
unsafe rather than requiring a general replanner.

Remaining limitations: semantic classification and writer compliance still depend
on model judgment; hashes do not prove semantic correctness. Chat-only exclusions
need to be supplied as context. Legacy data stays usable but unclassified, and
legacy consolidated pages without saved payloads cannot be fully regenerated.
The specified resume refusal is an intentional limitation. These are documented
boundaries, not reasons to change the default. Runtime implementation and tests
remain future work; this review edited only this plan and launched no models or
delegates.

### Design amendment — 2026-10-31

Added the user's two prompt checks explicitly: restricted reviewers avoid
proposing new scope, and validators independently reject any leaked extension
for a `preserve` run. The pre-write gate remains final enforcement. Both checks
use the existing scope-policy option and model calls; default `expand` is
unchanged. Clarified correctness versus policy rejection, existing-scope-creep
corrections and blocker exceptions, read-only review rejection/reporting,
prompt-policy provenance, and the need for a fresh review to solicit previously
omitted extensions. Added corresponding rendered-prompt, leaked-extension,
override, and call-count verification requirements. This amendment was made
directly; it has not received another delegated review.

### Default-compatibility clarification — 2026-10-31

Made the user's requirement explicit: default/explicit `expand` retains current
substantive prompts, validity criteria, judge behavior, and apply eligibility.
New prompt instructions only request descriptive scope tags. Removed wording
that could add a default goal-relevance/necessity test or a new conflict gate,
limited new remedy and scope-binding enforcement to `preserve`, and required
before/after fixture checks for default decision and call-count compatibility.
Only the opt-in policy adds suppression and rejection instructions. Prompt text
necessarily changes for annotations, so exact model-output identity is not
promised. This clarification changes the plan only.

### Documentation clarification — 2026-10-31

Added phase-by-phase proposed usage examples and an explicit README budget:
prefer zero net word/line growth, otherwise at most a short option-table row.
Detailed examples and shared semantics belong in `docs/workflows.md`, with
commit-specific guidance and links in the associated existing docs. The actual
README and runtime documentation stay unchanged until implementation.

## Implementation Notes — 2026-10-31

Implemented in its own worktree, `/home/marc/code/mllm-wt/scope-policy`, on branch
`feat/scope-policy` cut from `origin/master` (`c91e8d29`). Everything is **staged, not
committed**; nothing was pushed or published. Paths below are relative to
`skills/multi-llm/` unless they start with `docs/`, `packaging/` or `README.md`.

### How it was built

1. A shared contract, `utils/scope_policy.py`, written first (with 82 unit tests in
   `tests/test_scope_policy.py`). It defines the vocabulary (policies, relations, the three
   labels, policy sources), normalization that never raises, the scope baseline record and
   id, the proposal payload/digest, host stamping of validation records
   (`stamp_validation_record`), hash-only verdict attachment (`attach_verdict`), authority
   (`authority()`: `bound|missing|stale|other-baseline|unbound`), the gate
   (`scope_decision`, a pass-through under `expand`), the role- and policy-specific prompt
   blocks (rendered per call), policy resolution, and the per-review-round `state.json`
   record (`scope_policy[review-plan|review-tasks|code-review]`).
2. Four Opus agents in parallel, one worktree each, by area: A prompts/validation
   (`utils/scope_round.py`), B CLI/propagation (`utils/scope_policy_runs.py`), C apply
   gate/writes/resume (`utils/scope_apply.py`), D reports (`utils/scope_report.py`). Merged
   by patch; two conflict sites were resolved by hand (`utils/improve_workflow.py`,
   `docs/development/workflows.md`, `docs/development/review-results.md` contents list).
3. A docs agent (README, `docs/workflows.md`, `docs/commit-reviews.md`, instructions,
   `SKILL.md` argument hint, `references/resume-detection.md`).
4. Three read-only Fable reviewers (validation/authority, apply/resume, CLI/reports), then
   two Opus fix agents and coordinator fixes for every confirmed finding (below).

### Plan items — status

- **Goal / default compatibility — done.** Omitted or explicit `expand` keeps every
  review, validation, judge, selection and application decision. Prompts gain only the
  informational "Scope Classification (labels only)" block and the three output fields;
  no `preserve` text appears in an `expand` prompt (asserted on rendered prompts for all
  four reviewer phases and the batched, legacy and revalidation validation routes,
  alternating policies in one process). `tests/test_scope_policy_default_compat.py`
  compares against a golden captured from the **pre-feature code** (independently
  re-captured by a Fable reviewer from `c91e8d29`: byte-identical): confirmations,
  selected/applied ids, priorities, judge decisions and judge-task prompts, batches,
  `write_tasks.json` prompts and every `util write-prompt` rendering, for no option on an
  expand round, no option on a legacy round, explicit expand over a preserve review and
  explicit expand on an expand round.
- **§1 ownership — done** as listed; no host-shim duplication; ID algorithms, grouping,
  permissions and file-containment rules unchanged.
- **§2 baseline — done.** Per-round baseline from the inputs each round already binds
  (plan: `validation-snapshot.md`; tasks: plan + task doc/`tasks-backup.md`; code: plan +
  reviewed diff/code + tasks; commits: pinned brief, endpoints, frozen context plan).
  First binding wins; reaggregate/revalidation never rebase; revalidation embeds the
  round's hash-checked original snapshot as a second labelled context
  (`[SCOPE_BASELINE_MOVED]` when validators see changed documents).
- **§3 metadata/remedies — done.** Members keep provisional `scope_relation/
  scope_reason/scope_reference`; the authoritative group verdict is `scope_verdict`
  (`relation, reason, reference, baseline_id, proposal_digest, source, selected_remedy?,
  missing?`), attached only through a stable-hash join (index fallbacks keep validity, never
  scope). The proposal digest covers every byte of every retained member, so description
  edits past the 200-char ID prefix, per-member skips and judge salvage make a verdict
  `stale`. `selected_remedy` is preserve-only; under preserve it becomes the sole write
  instruction (`format_fix_for_output` no longer concatenates other members), the original
  text is kept only as evidence in `orchestrator_output.json`. Consolidated umbrellas derive
  their label (`combined_relation`).
- **§3.1 the two prompt checks — done.** Reviewer suppression and independent validator
  rejection under `preserve` only; validity kept separate (a correct extension stays
  `valid`); validation-stage policy rejection recorded as `scope_policy_decision` plus
  `[SCOPE_POLICY]` counts in `validation.json` metadata and `.status.json`. No extra model
  calls (call counts asserted equal under both policies).
- **§4 CLI/propagation — done.** `--scope-policy {expand,preserve}` declared once
  (`cli_common.add_scope_policy_argument`, default `None`), on review/apply plan, tasks,
  code, commits, generate-tasks, implement, the three improve composites, `util
  improve-steps` and `full` (incl. `--full`); not on ask/status/init. Invalid/empty → exit 3
  (D6 row 16 + `DECISION_FLAGS`). Forwarded as one argv value through composites, full's
  `common_flags` and prerequisite auto-run, commit doors, detach re-entry, replay/retry
  tokens and resume arrays. Resolution order as specified; an explicit apply override is
  recorded (`apply_policy`) and survives omission; composites/full/generate-tasks/implement
  keep a typed choice in `state.json` `scope_policy_runs[<mode>]`.
- **§5 eligibility/writes/resume — done.** Gate after all proposal changes and again
  immediately before batching; earlier exits, confirmation text and dry runs describe the
  same set; no flag bypasses it; `--min-priority` still applies. Deferrals are their own
  outcome (`scope_policy.<review>.deferrals`), never processed/applied/skipped, surfaced in
  confirmation, dry run, summaries, `summary.md`, `orchestrator_output.json`, the result
  line `notice` and `[SCOPE_DEFERRED]`. Write tasks carry a schema-validated `scope` block
  under preserve; write prompts append `writer_guidance` (empty under expand). Stale-task
  refusal in executor dispatch and native `util write-prompt`; policy change with no started
  work retires/rematerializes, with landed work keeps it and numbers new batches after
  every earlier index, with an unsettled batch exits 3 naming the recovery. Completion
  "eligible work completed" keeps deferrals so prerequisites do not loop.
- **§6 reports — done.** All three Markdown generators and all three HTML templates;
  legend + policy line (review policy shown when an apply override differs); distinct-group
  counts; scope lines sit above the `**Validation:**`/`**Importance:**` boundary (unchanged
  reports round-trip with zero detected description edits); `report_data.json` and a new
  `consolidated_report_data.json` sidecar (excluded from reaggregation scans) persist the
  scope data; regeneration recomputes eligibility from persisted payloads + `state.json`
  only; a legacy consolidated page without a sidecar is a reported no-op; selection
  import/export drop scope/policy-like keys. Template JS exercised under node.
- **§7 docs — done.** See README budget below.

### Deviations and decisions (with reasons)

- **Composite-forwarded policy reads as explicit.** Only a user's choice is ever forwarded,
  so a child cannot (and need not) tell typed from forwarded; the `workflow` source word
  exists in the vocabulary but is not produced.
- **Next-phase commands do not repeat the flag**: continuations inherit from the records;
  a standalone generate-tasks/review-tasks/implement starts its own run, so the docs tell
  users to type the flag on each standalone phase (full/composites forward it).
- **Plain `--force` (no `--rerun-all`) over a `preserve` round with reviewer results**, no
  flag typed: keeps the reused results' policy and prints one notice naming `--force
  --rerun-all` for a fresh `expand` round (permissive, run completes; avoids mixing reviewer
  outputs requested under different policies). A typed different policy there exits 3
  (`[SCOPE_POLICY_CONFLICT]`). Plan §4 said a forced review without a policy is a new
  default-expand run; this is the one exception, for reused results.
- **Blocker signal**: a `preserve` writer decline whose reason starts `SCOPE BLOCKER:`. The
  phase records `completion.outcome: "blocked"` + `blockers`, prints `[SCOPE_BLOCKER]`, sets
  the result `error`, exits 3; dependent phases (incl. implement's own gate) refuse; not
  skippable with `--skip-prerequisites`; an `expand` apply (typed or a live recorded choice)
  re-offers it. Not inferred from HIGH priority.
- **No new `MULTI_LLM_RESULT` field**: deferrals ride `notice`, blockers `error`; structured
  data is in `orchestrator_output.json` (`scope_policy` block).
- **D6 row 16 is recorded under `--yes` for every non-typed resolution**, including the
  `expand` default (one `[AUTO_RESOLVED] scope-policy: …` line, like row 2's
  `defaults.models`). This is the one intended stdout addition on a default unattended run;
  the default-compat test drops exactly that line before comparing with the golden.
- **Pre-existing bug fixed in `utils/report_parser.py`**: `find_edited_descriptions`
  matched only the 16-char suggestion hash while v2 report headers carry the 8-char display
  hash, so plan/task `report.md` description edits were silently dropped. Now matched; a
  description containing its own `---`/`###` line is not counted as an edit when its
  readable part is unchanged.
- `SKILL.md` argument hint and `modes.ARGUMENT_HINT_TRAILER` gained
  `[--scope-policy expand|preserve]`. Three apply help goldens were already stale at HEAD
  and were regenerated (they also pick up `--human-task`, `--mark-completed`, py3.13 usage
  wrapping).

### Review findings fixed after the first merge

Fable (3 reviewers) found, all fixed with regression tests: the legacy
`--internal-validation` reply contract never asked for `group_id` (no verdict could bind;
preserve deferred everything); `validation_failed` groups counted as deferrals; salvage
dropped provisional tags; native `--resume` after a scope blocker cleared it; the blocker
was read for the review phase too (deadlocking the apply that clears it, with a wrong
recovery command); `--skip-prerequisites` papered over a blocker; an expand→preserve flip
recorded already-applied items as deferrals; a refused override was persisted before the
refusal; implement read a blocked apply-task-suggestions as met; the composites' and
full's release of deferrals ignored the live recorded choice (an override did not survive
omission) and never re-offered a blocked apply; `generate-tasks --force` / `implement
--fresh` inherited an old recorded choice; full's release check sat in `_recorded`, which
also judges whether a step that ran recorded itself; revalidation received the raw flag
instead of the apply's resolved policy; review/apply phases did not record row 16.

### Known limitations / deferred

- Reviewers still read the live plan path; the baseline is the one read that becomes the
  validators' snapshot (same pre-existing exposure as before for a plan edited mid-fan-out).
- `--internal-revalidation` gets no baseline (preserve defers; expand unaffected). The dead
  builders `build_validation_subagent_prompt` / `prepare_validation_task` are untouched.
- `report_parser` cannot see past a `---`/`###` line inside a description, so deleting such
  a tail reads as "no edit" (no regression: such edits were dropped before). The same
  boundary risk exists in `apply_code_fixes_orchestrator.find_edited_issue_descriptions`.
- Reviewer-policy mixing after a disjoint reviewer selection was confirmed and fixed
  during QA; see the [test report](scope-policy-and-report-tags-test-report.md#findings).
  Remaining unverified concerns: `status` can still suggest `generate-tasks` beside a
  blocker notice; the attended blocked-prerequisite prompt still offers "skip and
  continue" (now refused with exit 3).
- Pre-existing, noticed, unchanged: native `--resume` restores a judge-salvaged group with
  its original description; the human-decision overlay writes only `report_data["groups"]`.
- Not run: container recordings, `package`/`codex_jail`/`live` markers, any live provider
  run (the plan requires none). No release/version work.

### README budget

Before 311 lines / 2,971 words; after 310 / 2,963 (−1 line, −8 words). One Common-options
row (21 words) for `--scope-policy expand|preserve`, linking `docs/workflows.md#scope-policy`,
paid for by removing a sentence the diagram legend already states and folding a one-line
sentence into §3.4's closing line.

### Verification (all from the worktree root `/home/marc/code/mllm-wt/scope-policy`)

- Baseline before any edit: `uv run --project skills/multi-llm -- pytest skills/multi-llm/tests -m "not live and not container and not package and not codex_jail"` → 26,070 passed, 281 skipped, 0 failed.
- First merged run (before the review fixes): 26,634 passed, 4 failed (the three row-16
  expectations and the missing `full-workflow.md` row; all since fixed).
- Final run (same command, after every fix, Codex mirror and OpenCode staging regenerated): 26,666 passed, 281 skipped, 2,298 deselected, 0 failed (1,438 s).
- `uv run --project skills/multi-llm -- python skills/multi-llm/hosts/render_capabilities.py --check` → up to date.
- `make codex-regen` (tracked mirror regenerated, 211 payload files) and
  `uv run --project skills/multi-llm -- python packaging/opencode/build.py --allow-dirty`
  (staged 211 payload files); new source files were intent-to-add before both.
- Acceptance criteria not verified by a live run: real model compliance with the
  `preserve` reviewer/validator instructions (fixtures prove plumbing, not model accuracy).

## Final QA and integration handoff — 2026-10-31

The [related QA report](scope-policy-and-report-tags-test-report.md) supersedes the
initial handoff's verification limits above. It records real Haiku reviews,
validation and application, an application regression test, detached interruption
and resume checks, browser selection round-trips, and fixes for the two policy
defects and the pre-existing packaged dry-run reporting issue. The final ordinary
suite passed **26,699 tests**, with **281 skipped**; both package payloads were
regenerated. No release or version publication was performed.
