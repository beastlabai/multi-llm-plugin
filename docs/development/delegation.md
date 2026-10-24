# Delegation and validation

Read when changing task-file schemas, prompt context, executor dispatch, validation, or runner selection on retry.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Materialized task contract](#materialized-task-contract)
- [Code-review validation context](#code-review-validation-context)
- [Read and analyze executor](#read-and-analyze-executor)
- [In-process continuations](#in-process-continuations)
- [Validation model capability](#validation-model-capability)
- [Runner pinning after failures](#runner-pinning-after-failures)

## Materialized task contract

**Materialized subagent tasks are the one delegation contract.** Every task file an orchestrator
emits (`validation_tasks.json`, `revalidation_tasks.json`, `consolidation_tasks.json`,
`salvage_task_<model>.json`, `human_decision_tasks.json`, `task_generation_tasks.json`) carries
`tasks_schema_version: 2` and, per task, a complete `prompt` plus
`output_name`/`output_path`, `result_channel: file`, `context_mode`,
`capability_tier`, `executor_role`, `task_id` and an `inputs`
hash block — validated against `schemas/subagent_tasks.schema.json` **on write and on read**
(`utils/subagent_tasks.py`). The invoker's whole job is: run `task.prompt` as a subagent,
write the JSON it returns to `task.output_path`; the callee only prints.

Prompt bodies live in `utils/subagent_prompts.py` as five **pure** generators (no I/O, no clock, no
config — identical inputs render byte-identical text, pinned by `tests/golden/prompts/`), never in
instruction markdown. **No generator interpolates `plan_file`, `grouped_file` or
`output_path` into prompt text**: under `context_mode: inline` every input travels inside
`prompt`, because a headless runner is not guaranteed a workspace-read tool.

**Two named exceptions**, each declared as `context_mode: workspace-read` so `refuse_reason` stays
the single gate: `task_generation` names the repo root, and **a validation/revalidation task
names at most one path per document whose context exceeded `VALIDATION_CONTEXT_LIMIT`, each a hash-bound
snapshot** (plans/validation-context-by-path-and-codex-0156-fallout.md §2.3). Under the limit
nothing changes (`inline`, no path, goldens byte-identical). Over it, per document,
the prompt keeps the 60 000-char head excerpt, ends it with a marker naming the dropped sections and
telling the validator to read the file before judging anything those sections, or an
omission/absence claim, depends on, and names the file once under `## Plan file` /
`## Tasks file`.

The path is always the round's snapshot, written from the one in-memory byte read of the document:
`<phase_dir>/validation-snapshot.md` for the plan in every validating phase (never `backup.md`, never
the live plan, except the live file when the snapshot cannot be written and it still hash-matches),
`tasks-backup.md` for review-tasks' task list when its sha256 matches, else
`validation-tasks-snapshot.md`, and `revalidation-snapshot.md` for revalidation.

Each task records `context_sources[]` (`label`
`plan`|`tasks`|`reviewed-code`|`diff`,
`kind`, phase-dir-relative `path`, `abs_path`, raw-byte
`sha256`, `bytes`, `chars_shown`, `truncated`,
`mode` `inline`|`path`|`truncated`); the
snapshot hash is re-verified before dispatch, on resume and on `--reaggregate`
(`validation snapshot changed/missing; re-run validation`, never a silent re-point), and a moved or copied phase dir rebinds the
prompt to the copy's snapshot only after its hash verifies.

`mode: truncated` — no readable file, or a runner that cannot read files — keeps the bare cut
and is surfaced, never decided at dispatch. A validator that cannot read the file answers
`needs-human-decision` with a reason starting `insufficient-context:`, counted in
`validation_context.insufficient`. review-code's reviewed-code block and a code-fix diff are never named by
path. The schema's validation branch is `{"enum": ["inline", "workspace-read"]}`; consolidation stays
`const inline`. The premise holds because every adapter declaring `agent-readonly`
also declares `supports_workspace_reads` (a test pins it), and claude-code gets `--add-dir <dir>`
(its `read_scope_flag`) when a path source lies outside its cwd; codex reads anywhere
(`reads_outside_cwd`).

Every other adapter, opencode included, reads only inside its cwd, so an out-of-cwd path source is
truncated at materialization and refused at dispatch. Task files and `validation.json` are
stamped `context_contract: 2`; an unstamped round whose stored prompts carry the old bare
`[... truncated ...]` cut is renamed `.stale` and rematerialized
(`validation built from truncated context (contract 1); rematerializing`).

A v1 file, or one whose `inputs` hashes have moved, is refused with **exit 3** naming
the phase dir and, for a moved hash, the artifact behind the key (`tasks_sha256` reads
“tasks.md changed since validation — apply-task-suggestions rewrites it”, which is why a
`review-tasks --reaggregate` after that phase is a routine exit 3 — F8, 2026-09-16) — prompts are derived
data, so the fix is always re-materializing (`--force`), never patching.
`executor_role` (`agent`|`human`) is the routing field every
reader branches on; `subagent_type` is optional Claude-only metadata that no reader consults
and the schema rejects when it disagrees.

Judge verdicts are files (`human_decision_batch_<n>.json`, `schemas/human_decision_verdicts.schema.json`) ingested by
`utils/human_decisions.py`, never chat text. **Validation records name their index
`group_index`** — the key the reply echoes (advisory; `group_hash` is the join) —
and the merge prints one summary line, `[merge] N of M verdicts echoed a different group_index; all joined by hash`, instead of one warning per verdict (a
correct 46-group run once printed 41).

`validation.json`'s `metadata.model` is derived from the batch files' own metadata (one
value, or a sorted comma-joined list when they differ) ONLY on the host-native path; when
`--exec-subagents` dispatched the batch itself, `merge_validation_batches` (`review_orchestrator_base.py`)
reads the executor's own `subagent_batch_report.json` back (`_authoritative_validator_runner`) and that outranks the
self-report, which a captured run once spelled five different ways across 11 batches — one of them
another model's name copied out of the suggestions it was validating — and stores the same value
under `metadata.runner` too.

Since §7.3 (2026-09-24) that decision is **per batch and by content hash**: the executor records
`batch_provenance` (`{task_id, kind, round, runner, sha256}` over the bytes it persisted) in the batch report, and
the merge credits the recorded runner only while the batch's bytes and the task file's
`kind`/round still match — a batch rewritten by hand or by another host since is
attributed to its own `metadata.model`; `metadata.runners` keeps the per-batch map and
`metadata.model` is the single value or the sorted comma-joined list.

The merge also binds `validation.json` to the batch bytes it consumed (`metadata.batch_sha256` +
`metadata.batch_manifest`, §7.1), so `--reaggregate` re-merges when any listed batch changed,
vanished or belongs to another round; only a file without `batch_sha256` keeps the legacy
mtime test. And `consolidation_recommended` is recomputed up or down on both completion paths
(`recompute_consolidation_recommendation`), removed with `status_file.remove_status_keys` below the threshold, and once the
consolidation it recommends has run on the current `grouped.json` (`consolidation.consolidation_is_current`:
the `--reaggregate-consolidation` merge removes it, `detach.classify_run` and a resumed
`full` ignore it — live run 8) — never written as `null`.

An absent `confidence` is stored as `null` and never
`0.0`, a bare model name that is the model half of a prefixed sibling collapses
into it (one runner reported itself as both `nemotron-3-ultra-free` and `opencode/nemotron-3-ultra-free`), and
the prompt tells the validator to read the repository before answering `needs-human-decision` on a
claim about existing code: `agent-readonly` can read, and a batch that made zero tool calls
once declined a claim its sibling batch verified at 0.98.

**An omission claim is judged against the plan, not the working tree**: "the plan does not specify
X" is valid when the plan text omits X even if the code already does X — the plan is what an
implementer follows, and a partially implemented tree is the situation the review exists for — and
the reason says the code already does it (a diff under code review is the text such a claim is
judged against). Measured 2026-09-13: the validator marked "Task 2 does not require resetting boost"
invalid because `Car.reset()` already reset it, while applying the opposite rule to its
sibling.

**And the artifact under review has to be *in* the prompt for that rule to be answerable**:
`review-tasks` embeds two labelled `## Context (…)` blocks — the plan, and
`tasks/tasks.md`, each excerpted on its own budget so a long plan cannot squeeze the task list
out — and hashes the task list into `inputs.tasks_sha256` so an edited tasks.md invalidates the
file the way an edited plan does (`review_orchestrator_base.validation_context_kwargs`, both the fan-out and the reaggregate site,
which had to start resolving the tasks file too).

Measured 2026-09-16: 28 of 30 suggestions named a bare `T###`,
`'T001' in prompt` was False for all eight batches, and two verdicts came back citing text — a
phrase that exists only in tasks.md, and `state.json`'s `plan_preamble` field —
that no validator had been given. `apply-task-suggestions`' own `human_decision_tasks.json` and
`write_tasks.json` stamp the same `inputs.tasks_sha256`
(`build_judge_tasks`/`materialize_write_tasks`/`write_task_file_for_phase` now forward
`tasks_file=`), so a hand-edited tasks.md invalidates a stored `--resume` there
too (F8, 2026-09-16).

**`review-code` embeds the same pair, for the same reason**: the plan — no longer sliced by
the caller before `excerpt_context` re-sliced it, and `VALIDATION_CONTEXT_LIMIT` is 60 000 now
because 30 000 was under the size of a real plan (measured: a 44 374-char plan cut mid-§4.6, so
§4.7–§11 reached no validator while four of nine verdicts cited exactly those sections) — beside a
second block carrying the implement phase's own record and the file lines each issue cites
(`build_validation_secondary_context`), hashed into `inputs.reviewed_code_sha256` so a file edited between fan-out and
validation invalidates the task file, which `diff_base` alone cannot see.

Line windows rather than the diff, because a file the implementation *created* has a diff the size
of the file (61 KB on that run, twice the per-context budget).

**Each half of that block is excerpted on its own budget** — the record at `IMPLEMENTATION_RECORD_CHARS`,
the code windows given the rest of `VALIDATION_CONTEXT_LIMIT`, and whole files dropped and named rather
than one window cut mid-file — because under a single shared budget with the record first the *code*
was the half that went (measured: a 150-entry record plus 20 cited files left 11 of the 20 with no
window at all), and every cut prints one stderr line per source (F3, 2026-09-16):
`[VALIDATION_CONTEXT_TRUNCATED] <block> <n> chars over` plus a reason line for a `mode: truncated` source (the reviewed-code
block is always this when cut, since it is never named by path), `[VALIDATION_CONTEXT_EXCERPTED] plan <n> chars over; <cut>; full plan at <path>` for a
`mode: path` source, and nothing for an inline one.

The same facts are the top-level `.status.json` `validation_context` object
(`phase`, `limit`, `mode`, `insufficient`,
`sources[]`, one writer: `publish_validation_context`), repeated on `MULTI_LLM_RESULT`,
carried through `detach.classify_run` and re-recorded by `status`; a truncated source
sets the result/poll `notice` ("Validation saw 8 of 20 sections of the tasks file")
and a `report.md` header line. `mode: path` reads as excerpt plus path, never
"fully inlined".

The cited path is **model output**, so `build_reviewed_code_excerpt` reads only what resolves inside the
project root: an absolute path, a `../` escape or an in-tree symlink pointing out
renders `(outside the repository; not shown)`, where before those lines reached the validation provider and
`validation_tasks.json` (F1, 2026-09-16). `reviewed_code_sha256` still hashes the **untrimmed**
excerpt, which is what the reaggregate site recomputes.

**The reviewers are shown the record too** — `prompts/code_review.txt`'s `## Implementation Record`, fed
`summary.md`'s path, the implement-phase `prior_changes.jsonl` entries and the task ids
state records a *decision* for (`skipped`/`deferred`, or a
`pending` carrying a `task_status_reasons` entry) — implement seeds every task
`pending` at start, so a `failed` or crashed `in_progress` one
is listed on its own line as a gap to report normally rather than under “must not be reported as
missing” (F7, 2026-09-16); without it a deliberate, logged tune (`accel tuned 28->32 for <4s`, made to
satisfy the plan's own milestone) came back as a *valid* "deviates from plan" finding for
apply-code-fixes to revert (F7/F8, 2026-09-16) — which it then did, because the fix batcher's own
prompt showed the finding and the prior-*batch* log but never the prior-*phase* record either:
`code_fix_batcher.format_fix_batch_prompt` now takes the phase's `output_dir` and, when it has one, renders a
`## Implementation record` section from that same `implement`-phase `prior_changes.jsonl`
(entries naming the batch's own file sorted first) with an explicit "do not silently revert a
recorded decision" instruction (F3, 2026-09-16).

The reply template's `group_index` example is a placeholder, never a number (models copied
`0`: 41 of 46 and 18 of 32 verdicts echoed batch-local ordinals), and
`_reported_group_index` coerces whatever comes back — a copied placeholder joins by hash and counts
in the summary line; with no hash it falls through as `-1` instead of raising.

## Code-review validation context

**A code-review validation prompt carries no diff, and its first block is the plan, head-kept.**
Both `review-code` sites hand `prepare_batched_validation_tasks` `context="Plan file: …\n\nPlan content:\n" + plan` and the second
block is the implement record plus the cited files' line windows (`build_validation_secondary_context`); the
144-file diff a reviewer fan-out reads never reaches a validator, so an over-limit first block is
always the *plan* going over `VALIDATION_CONTEXT_LIMIT` (`excerpt_context` keeps the first 60 000
chars inline; the rest is reached through the plan snapshot by path, `[VALIDATION_CONTEXT_EXCERPTED]`, or is
cut and surfaced as `[VALIDATION_CONTEXT_TRUNCATED]` only when no readable snapshot or no file-reading runner
exists).

Measured 2026-09-17 on a 336 026-char plan: `276143 chars over`, 2 of 13 `##`
sections kept whole, the cut inside Phase 1, Phases 2–7, §8–10 and the `Implementation Notes` tail
dropped — while every verdict cited §2, §4, §6 or those notes, located by opening the file, which
`context_mode: inline` did not promise; the task now names the round's `validation-snapshot.md` and is
`workspace-read` (see [Materialized task contract](#materialized-task-contract)). Two further things changed. The line now names
the cut (`describe_context_cut`: sections kept whole, the one cut mid-way, the first three dropped
and `(+n more)`, headings inside code fences ignored), and `reviewed-code` names
the files whose windows went rather than a char count.

And `build_validation_batch_records` now carries each suggestion's
`file`/`line_range` **when present**: `CODE_REVIEW_VALIDATION_GUIDANCE` has said
"each issue record carries `file` and `line_range`" since D2 lifted it out
of `review-code.md` — true there, because the markdown pointed the validator at
`grouped.json` — but the builder kept the plan-review four-field shape, so every code-review
validator was told to open locations it was never given beside a code block naming five files and
saying nothing about which finding each belongs to. Keyed on presence, so plan-review records render
byte-identically.

Both context blocks are still **global** — excerpted once per phase, embedded verbatim in every
batch (validation.py's `shared_context`/`shared_secondary`) — so a batch's prompt carries
windows for files only *other* batches' groups cite (measured: 3–14% of each prompt, 71–80% of the
code block in the single-HIGH batches). That, and choosing plan sections by relevance instead of by
position, are the open design questions; the budget itself was not the problem, the cut's
*placement* was.

## Read and analyze executor

**`utils/subagent_executor.py` is the second invoker of those task files (`--exec-subagents`, D3
read/analyze half).** One dispatch is: refuse before spawning (the D2 refusals plus containment) →
pick a **profile from the task kind**, never from the task JSON (`agent-readonly` for every
read/analyze kind, `cwd` pinned to the repo) → capture stdout → the adapter's
**`parse_output` first, then `json_extractor`** (provider CLIs emit event streams; raw
extraction would match an envelope or a reasoning item and write a garbage result reaggregation
would accept) → validate against the kind's result shape → atomic 0600 write under the phase dir.

`persist_result` is the **shared** invoker-persists leg: `simulate_native_invoker` (which
`SkillRunner.simulate_subagents` delegates to) calls the same function, which is what makes "either invoker,
byte-identical `output_path`" a property of the code.

Every failure lands on exactly one **D3k rung**
(`spawn_failed`/`auth_failed`/`invalid_model`/`sandbox_unavailable` 1,
`timeout` 2, `rate_limited` 3, `runner_error` 4, `unparseable_result`
5, `schema_invalid` 6, `rejected` 7); rungs 2/4/5 consume `driver.subagent_retries`,
rate limiting backs off **without** consuming it, rungs 6/7 are never retried (one exception since
2026-10-05, the ladder's rule 7: a `task_generation` or write-kind reply that fails rung 6
*with a document in hand* gets **one** schema-repair re-ask on the same runner — the original prompt
plus the validator's own errors and the reply shape, `subagent_prompts.build_schema_repair_prompt` — granted by
`SCHEMA_REPAIRS_BY_KIND`; a write kind's re-ask (fix round 7; implement's since fix round 14, OpenCode
racing run 5's T013) never re-sends the batch's instructions — the runner restates its own previous
reply as the report (quoted from the report's own start when it begins before the reply's last 6000
characters, `subagent_prompts.quote_previous_reply`, fix round 8), told the batch's `expected_ids` in prompt
order so a reply that numbered its items ("Suggestion 1: …", as the batch prompt's Return Format
asks) is restated by id (fix round 11: OpenCode racing run 4 restated `ids: ["Suggestion 1", "Suggestion 2"]` and a
fully applied batch stopped at exit 3), and touches no file, a file that moves during it
(`tree_fingerprint`) makes the batch a partial application, and the snapshot comparison still
decides what was applied; a reply whose JSON did not parse is told the parser's line and column (the
`enclosing_syntax_error` that `json_extractor.extract_json_from_text` returns beside a piece of a broken document it
served — read off the extraction that served it, never by searching the reply again; a stray bracket
in prose is not a document, fix round 8), never the errors of a fragment salvaged from inside it;
charged to the same deadline, spending neither counter, recorded as its own attempt with
`repair: schema_invalid` in `failed/` and the batch report; a CLI's own exhausted
structured-output budget is not re-asked, nor a valid document `persist_result` refuses over
`MAX_RESULT_BYTES` (still `schema_invalid`, a size and not a shape), every other kind's
rung 6 is unchanged, and a second failure is the usual exit 3 — measured 2026-10-04,
`opencode/big-pickle`, which carries no structured output, answered the wrong shape twice and
generate-tasks dead-ended), and rung 1 fails the whole batch — nothing already dispatched is
cancelled and nothing further is dispatched whatever spec it would have used, with each undispatched
sibling recorded `incomplete` under a reason that either names the shared failing spec or
says its own runner was never tried (D4).

`invalid_model` is the classifier's own category given a rung: a rejected model id is a
property of the spec, so the fix is a config edit and `_recorded_runners` drops every pin on a
spec any row recorded with a fatal status, which is what lets that edit take effect on
re-invocation.

**The rung decision itself is not this module's** — it is `utils/retry_ladder.RetryLadder`, the one pure
policy object both invocation lanes obey (§7.1.b): one deadline per slot taken once (every attempt
spawned with what is *left* of it, backoff sleeps charged to the same deadline), the two separate
counters above, and §7.5's two composition tests against the adapter's **declared**
`internal_retries` envelope — no attempt is started when what remains cannot hold one worst-case
`max_wall_s` (recorded `budget_exhausted`, a rung-less non-fatal status meaning
**nothing was spawned**, and a config refusal no re-run changes; every grant — retry, backoff,
re-ask — takes the same test in `record()`, so a slot whose attempt ran keeps that
attempt's status with the refusal as `Action.note`, 2026-10-05), and an adapter that
declares `retries_rate_limit: true` gets no identical rate-limit re-dispatch, because that would repeat
a chain its own CLI already ran.

The ladder is provider-free by construction (the envelope is read structurally) and each lane keeps
only its own three-line loop: `subagent_executor._dispatch_task` blocks, `review_orchestrator_base.run_single_model` awaits — the
latter because it runs inside `asyncio.gather` under two semaphores, where a blocking retry
loop would serialize the whole fan-out. Each failed attempt writes `{phase}/failed/{task_id}.attemptN.json` with the
prompt elided to its sha256; attempt numbering restarts per invocation, so an existing file of that
name is first retired to `{task_id}.attemptN.<its own mtime, UTC>.json` (`_retire_capture`) rather than overwritten — a
retry never destroys the evidence of the attempt it retries.

A batch with any failed or incomplete task writes `{phase}/subagent_batch_report.json`, refuses to reaggregate and
exits **3 even under `--yes`** — and **never** falls back to salvage. Resume is
defined by artifacts: a task is done iff its `output_name` exists, parses and validates, so
re-invoking a phase re-dispatches exactly the failed set. `run_tasks_concurrent`
(`utils/review_orchestrator_base.py`) is `run_models_concurrent`'s mechanics keyed by `task_id`
instead of `model_spec` — same global gate, same per-provider semaphores, same stagger —
because a subagent batch normally shares one runner and a model-keyed map would overwrite completed
tasks.

**`{phase}/subagent_batch_report.json`'s filename is kind-aware** (`_batch_report_filename`, F6, 2026-09-16): every
read kind but `human_decision_judge` keeps the bare name, because `human_decision_judge` is the one
read kind whose dispatch can share a phase dir with a later write dispatch — an apply phase's
unattended judging pass runs on this executor before its own write batches ever touch the dir, and
both used to write the identical bare-named file, so the write phase silently overwrote the judge's
report (argv, prompt hash, duration, usage — gone).

It now writes `subagent_batch_report_human_decision_judge.json`, `_recorded_runners` reads back the same kind-qualified
name, and the judging dispatch is also the one read kind routed through `subagent_audit.jsonl`
(`AuditLog.dispatched`/`.completed`, the write path's own control), since a judged
decision is the one subagent output a human is most likely to re-examine.

## In-process continuations

**`--exec-subagents` continues past markers in-process (D3b)**, via `review_orchestrator_base.dispatch_subagent_tasks`:
markers are still printed (logs, `.status.json`) but are informational, and the phase
finishes in one invocation — validation and revalidation into their merges, judging into
`record_human_decision` and the apply set, consolidation into `reaggregate_consolidation_mode`, salvage into
the repaired model-result file. With the flag **off** every one of those sites returns
`False` and behaves byte-for-byte as before; salvage dispatch is deliberately
non-fatal (a failed salvage leaves that model failed, which the D6 row-3 partial-results path
already covers).

**A salvage that recovered nothing is one of those failures.** `build_salvage_prompt` asks for
`{"salvage_failed": true, "reason": ...}` rather than an empty array when the raw output holds nothing recoverable, and
both invokers write **nothing** for it: a salvage task's `output_name` *is* the model's own
result file, so an empty array there reaggregates as "this model reviewed the input and found
nothing" and a model that never produced a parseable word joins the aggregate as a phantom
participant. The outcome is recorded `salvage_unrecoverable` — rung-less (nothing about the dispatch
failed) and in none of the three retry sets, so never retried — carrying the subagent's own
`reason` into `subagent_batch_report.json`, which is the diagnostic the empty array threw
away.

An empty array genuinely *recovered* from the raw output stays a valid result, here and on the
fan-out path. **The sentinel is not the only shape that recovered nothing**, so
`_salvage_gave_up` asks a second question of what came back, and it is `json_extractor.is_answer_shaped` —
public now, and the *same* predicate `has_salvageable_json` asks of the raw output at the gate (an
array, `[]` included; an object with an array-valued key; an object naming a field
the answer schemas require, `_ANSWER_FIELD_NAMES`).

A subagent handed a truncated stream that turns out to hold a relayed `{"error": {"code": 401, …}}` body,
or the `{"type": "error", …}` record an `opencode`/`codex` run prints as
an event, can recover that object perfectly faithfully and never print the sentinel — as far as it
can tell it found something — and the gate is no help, because the gate judged the *raw output* and
this arrives from the subagent. One rule asked at both ends: *is there an answer in here to ask a
repair pass for?*, then *is what the repair pass returned an answer?*.

On the executor's lane what it adds is the **diagnostic**, not a write it stops:
`_salvage_result_errors` already refused a top-level object for not being an array, so an envelope
never reached disk — it was reported `schema_invalid`, "expected a JSON array of suggestions,
got dict", which reads as a botched repair pass and sends the operator back for another one when the
fix is the provider's login. It also moves the rule to where the phantom-participant argument lives
instead of resting it on an array check written for the consumer, which would silently re-open the
door the day a salvage result grew a wrapper-object form the way `_validation_result_errors` already
tolerates `{"groups": […]}`.

A near-miss stays a near-miss — a single record where an array was asked for names the answer's own
fields, so it is answer-shaped, falls through to `validate_result` and is reported as the
mis-packaged recovery it is. On the **native** lane the same rule is not a diagnostic but the write
itself: `references/salvage-handling.md` *is* the invoker there, and an agent following its earlier prose
would have persisted the envelope over the model's own result file — so that document states the
rule in the imperative, and self-enforces what the executor enforces in code.

**A salvage pair belongs to the attempt that produced it, and every fresh attempt retires the last
one.** `review_orchestrator_base.retire_salvage_artifacts` renames both `salvage_<stem>.json` **and** `salvage_task_<stem>.json` to
`retired_salvage_<stem>.<reason>.<stamp>.json`, called quietly at the top of `code_review_orchestrator.run_single_review` and of
`run_all_models`'s `per_model` — after the phase dir is made, before the provider
is invoked, **with `--exec-subagents` or without it**.

It closes the one route neither salvage rule can watch: the phase dir is stable per plan,
`--force` clears the state file's completion marker and deletes nothing from disk, and
`dispatch_salvage_tasks` globs every `salvage_task_*.json` it finds — so a rerun dispatched the
*previous* run's salvage task, against raw output that no longer exists, advertised it, and wrote
the result over that model's freshly written result path. `has_salvageable_json` judges this
attempt's raw output and the sentinel judges this dispatch's reply; a superseded pair is neither. A
rename and never a delete, because the request holds the raw output that says why the model failed,
and non-fatal but loud (a failed rename warns on stderr naming the file that survived).

The `retired_` prefix is the load-bearing half: it sits outside all three
prefix-anchored "still outstanding" globs (`dispatch_salvage_tasks`'s, `code_review_orchestrator`'s
end-of-run `salvage_*.json` scan, and that scan's `salvage_task_` filter) while still
matching the `'salvage' in basename` exclusion every reaggregation glob applies, so a retired pair is
neither advertised as work nor parsed as a model result — where `salvage_<model>.retired.json` would have
been both. Retiring at the *attempt* is what leaves a model **skipped as already complete** holding
its pair, which is right: it began no attempt and its request is still outstanding.

The residual is the mirror image — a model **dropped from the selection** between runs also begins
no attempt, so nothing retires its pair, exactly as nothing clears its stale `<model>.json`;
that one belongs to whatever eventually scopes a phase dir by run. **A verdict the executor already
called terminal is not re-advertised.** When, and only when, an outcome is `STATUS_SALVAGE_UNRECOVERABLE`,
`dispatch_salvage_tasks` retires that pair (`RETIRE_UNRECOVERABLE`) and prints `[SALVAGE_UNRECOVERABLE] <stem>: <reason>`
on **stdout**, in the place the `[SALVAGE_NEEDED]` marker held — the operator's next move is the
provider, and the reason is the only thing on screen that says so.

Every other salvage failure has a ladder rung, is re-runnable, and keeps its request on disk so the
end-of-run block goes on advertising it. The consequence is structural rather than declarative — no
`SALVAGE NEEDED` block, no `[REAGGREGATE_AFTER_SALVAGE]`, `record_next` never called —
because the artifacts those blocks glob are no longer there, not because a second reporting site
learned to re-derive the verdict out of `subagent_batch_report.json`. **With the flag off none of it fires
and the documented native flow is byte-for-byte unchanged**: no dispatch, so no outcome, so the
markers, the salvage-then-reaggregate loop and every printed command are what they were.

**And the gate that decides all of this now reads stdout, not stderr alone.** `llm_client._attach_stdout_excerpt`
attaches `details["stdout_excerpt"]` — redacted *then* bounded, and the key omitted rather than set empty
so `details.get("stdout_excerpt")` means "the provider said something worth reading" — on all three losing
paths (non-zero exit, timeout kill, zero-exit parse failure), and the code-review salvage site reads
it **ahead of** `details["stderr"]`.

Both halves are real failures: a provider can write a truncated array to stdout and *then* exit
non-zero, which a stderr-only gate could not see as salvageable at all; and the providers that
serialize their failures (`opencode`'s `{"type": "error", …}` event) put them on stdout
with stderr empty, so the gate was deciding on an empty string and reaching the right verdict for
the wrong reason. That excerpt is also what a salvage request's `raw_output` carries,
which is why `references/salvage-handling.md` tells its invoker to judge the content and never the channel,
and to read an elision in it as the harness's bound rather than the model's truncation.

## Validation model capability

**Full model for validation**: Validation subagents use full model capability — never downgrade to
haiku. Only the coordinator (simple orchestration) uses haiku.

## Runner pinning after failures

**A reply-shape failure does not pin its runner** (fix round 8): `_recorded_runners` drops, row
by row, the pin of a batch-report row whose status is `schema_invalid` or
`unparseable_result` (`UNPINNED_ROW_STATUSES`), beside the spec-wide drop for a fatal status, so a
re-run after a reply-shape failure takes the session model or `driver.subagent_runner` as they are now
(OpenCode run 3: generate-tasks stayed on `opencode/big-pickle` after the host switched model). A
transport failure (`timeout`, `rate_limited`, `runner_error`) or an
`incomplete` row keeps its pin, and the dispatch line's `kept from <report>` names the
kind's own report file (`_batch_report_filename`).
