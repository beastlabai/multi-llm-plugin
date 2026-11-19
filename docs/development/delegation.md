# Delegation and validation

Read when changing task-file schemas, prompt context, executor dispatch, validation, or runner selection on retry.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Materialized task contract](#materialized-task-contract)
- [The host index](#the-host-index)
- [Code-review validation context](#code-review-validation-context)
- [Cited plan sections](#cited-plan-sections)
- [Where the executor's helpers live](#where-the-executors-helpers-live)
- [Read and analyze executor](#read-and-analyze-executor)
- [In-process continuations](#in-process-continuations)
- [An idle writer](#an-idle-writer)
- [Tool-use counts on the executor leg](#tool-use-counts-on-the-executor-leg)
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
An apply write task materialized under `--scope-policy preserve` also carries a `scope` block
(policy, baseline id, binding, per-item relation and selected remedy; absent under `expand`),
which both dispatch paths check against the round's effective policy before binding a prompt
([write scheduling](write-scheduler.md#scope-policy-at-the-write-seam)).

Prompt bodies live in `utils/subagent_prompts.py` as five **pure** generators (no I/O, no clock, no
config — identical inputs render byte-identical text, pinned by `tests/golden/prompts/`), never in
instruction markdown. **No generator interpolates `plan_file`, `grouped_file` or
`output_path` into prompt text**: under `context_mode: inline` every input travels inside
`prompt`, because a headless runner is not guaranteed a workspace-read tool.

**Two named exceptions**, each declared as `context_mode: workspace-read` so `refuse_reason` stays
the single gate: `task_generation` names the repo root, and **a validation/revalidation task
names at most one path per document whose context exceeded `VALIDATION_CONTEXT_LIMIT` or whose
plan block carries only the batch's cited sections ([Cited plan sections](#cited-plan-sections)),
each a hash-bound snapshot** (plans/validation-context-by-path-and-codex-0156-fallout.md §2.3).
Under the limit and uncited nothing changes (`inline`, no path, goldens byte-identical). Over it, per document,
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
`mode` `inline`|`path`|`truncated`, and `selection: cited` on a `path` source whose excerpt is the
batch's cited plan sections); the
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
consolidation it recommends has run on the current `grouped.json` (`output_handler.consolidation_is_current`:
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
file the way an edited plan does (`review_orchestrator_base.validation_context_blocks`, read by
`review_validation_round`, the one round builder of both the fan-out and the reaggregate site).

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

**The validation task file carries the round's scope block** (`scope`: policy, prompt phase, the
baseline these validators are shown and the round's own baseline id;
plans/scope-policy-and-report-tags.md §3.1). `materialize_validation_round` renders `ValidationRound.scope` as
`scope_policy.validator_guidance` into every batch prompt per call, as its own argument after the
phase guidance (so `PLAN_VALIDATION_SCOPE_GUIDANCE` still fills in when `extra_instructions` is
empty), and both invokers run those bytes. The reply contract gains the optional
`scope_relation`/`scope_reason`/`scope_reference`, policy-independent; `util check-reply` and the
executor's result check never ask for them, so a reply without them is still a reply and reads
`unclear`. A revalidation resolves the policy as its apply phase does (explicit, recorded apply
override, recorded review policy, `expand`) and, when the current document is not the round's
baseline document, embeds the round's hash-verified `validation-snapshot.md` as a second labelled
context so correctness is judged against the current text and scope against the original
(`revalidation_scope_binding`). The in-process validation lane is retired: `--internal-validation`
and `--internal-revalidation` still parse, print one warning and change nothing, so validation and
revalidation always run through these task files (`cli_common.RETIRED_FLAGS`).

The reply template's `group_index` example is a placeholder, never a number (models copied
`0`: 41 of 46 and 18 of 32 verdicts echoed batch-local ordinals), and
`_reported_group_index` coerces whatever comes back — a copied placeholder joins by hash and counts
in the summary line; with no hash it falls through as `-1` instead of raising.

## The host index

`validation_tasks.json`, `revalidation_tasks.json` and `consolidation_tasks.json` each have a host
index beside them, `<stem>.index.json` (`refresh_host_index` in `utils/validation.py`). It copies
`total_batches`, `reaggregate_mode` and `plan_file` when the file has them, and per task `task_id`,
`batch_index`, `prompt_path`, `prompt_lines`, `output_path`, `subagent_type` and `context_mode`:
what a native host dispatches from, without the `prompt` strings, which are about 90% of the full
file and which a host that hands its subagent the sidecar never reads. The instructions name the
index on their "Read" lines, and the full file when there is no index.

- **A view, not a contract.** The executor, `util check-reply`, the audit and the container digest
  read the full file, which stays byte-identical, and so do the sidecars. Nothing but a host reads
  the index.
- **Every derived view follows one rule** (this index, and the apply phases' summary and batch
  details, [write-scheduler.md](write-scheduler.md#the-host-summary)): it is published only after
  its source is written, removed before its source or the sidecars it names change, and rebuilt
  from the source at every hand-back to the host. A host then never reads a view newer or older
  than the file beside it, and finding none always means reading the file.
- **Removed before the file changes.** `materialize_validation_round` removes the round's index
  before it writes a sidecar, `prepare_consolidation_tasks` before it writes anything,
  `ensure_prompt_sidecars` before it rewrites a sidecar or a pointer its caller then persists (and
  not at all when nothing needs repair), and `retire_stale_validation_round` before it moves any
  of the round aside: an index it cannot remove stops it with nothing retired. An interrupted
  rematerialisation or retirement therefore leaves no index beside what is left of the round.
- **Rebuilt from the file on disk, never from a document in memory.** `dispatch_subagent_tasks`
  rebuilds it on every call, the hand-back of a fresh round, a `--reaggregate` and the fan-out's
  continuation alike, and `prepare_consolidation_tasks` right after its write. A task file that is
  missing or unreadable leaves no index. An index whose bytes are already the ones it would write
  is left in place, neither replaced nor removed, so a healthy hand-back never stops on a file a
  host holds open (a Windows sharing violation).
- **One views mutex per phase dir.** `host_views_lock` (`{phase}/.views.lock`, a
  `process_control.FileMutex`, reentrant within a thread) is held across each sequence that
  removes an index, changes its task file or sidecars, and publishes it again:
  `materialize_validation_round` together with its caller's write of the full file (the
  revalidation round's as well), `ensure_prompt_sidecars` together with the commit of the repaired
  file, `prepare_consolidation_tasks`, and the refresh's read of the file and publication of its
  index. A refresh that read the file before a repair therefore publishes before that repair
  removes the index, never after it. A lock file that cannot be opened or locked at all lets the
  sequence run, removing as before, but publishes no index, so a host reads the full file; one
  another command holds for `HOST_VIEWS_LOCK_TIMEOUT_S` (60 s) stops the run with exit 3 naming
  it. A phase lock that serialises whole runs over one phase dir, once there is one, orders these
  sequences too; the views mutex does not depend on it.
- **Fail closed.** An index that cannot be written leaves none (`publish_host_view`), and one that
  cannot be removed stops the run with exit 3 naming it (`remove_host_view`): a phase dir made
  read-only after a round was written would otherwise keep an index of that round beside a newer
  full file.
- **The sidecar fallback carries over.** A task whose sidecar was skipped (`[PROMPT_SIDECAR_SKIPPED]`)
  has no `prompt_path` in the full file, so its index entry has none either, and the host runs that
  task's `prompt` from the full file.
- `RESERVED_OUTPUT_SUFFIXES` keeps a subagent result off it, and its name contains `validation`
  or `consolidation`, so no reaggregation glob reads it as a model's result.
- `task_generation_tasks.json` has no index: it holds one task.

## Code-review validation context

**A code-review validation prompt carries no diff, and its first block is the plan, head-kept.**
Both `review-code` sites build their round with `code_review_validation_round`: the first block is the
bound plan snapshot, with no `Plan file:` line, and the second
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

The code block is no longer global: each batch carries the windows its own issues cite, and the
plan block carries only the sections they cite when the opt-in is on
([Cited plan sections](#cited-plan-sections)). Before that, every batch repeated windows for files
only *other* batches' groups cite (measured 2026-11-07: 72% of the code block and a third of all
prompt chars on one round), and the round-wide budget dropped whole files from every batch.

## Cited plan sections

**Review-code batches carry the code windows their own issues cite. A batch's plan block can
carry only the sections its suggestions cite, but that is opt-in (`MULTI_LLM_VALIDATION_CONTEXT=cited`)
and the whole plan stays the default.** A round embeds one plan excerpt and used to embed one
reviewed-code block in every batch: over 55 recorded rounds 84–98% of all validation prompt text
repeated between batches, and the groups themselves were 3.7–13% of it
(plans/2026-11-07-code-health-04-review-pipeline.md REVIEW-8, -08-performance.md PERF-2). Batching is
unchanged; only what each batch embeds is.

**Why plan scoping is opt-in.** Smaller prompts do not show that verdicts are as good. A batch that
sees only its cited sections can uphold a suggestion that a section it was not shown forbids,
without ever making the omission or disagreement claim that would send it to the full file. The
code-health roadmap (plans/2026-11-07-code-health-13-roadmap.md, the REVIEW-8/PERF-2 row and owner
decision 6, live spend) asks for a live check before the default changes: verdicts on the same
recorded groups under cited and under full plan context. **That comparison is still owed.**
Until it is done, `cited` is for measuring and trying the mode, and the default round is the whole
plan in every batch, byte for byte what it was before cited sections existed (checked on all 622
recorded review-plan batches). The code windows stay on by default: a validator was never promised
the windows other batches' issues cite, the round-wide budget used to drop whole files from every
batch, and a replay of 24 recorded code batches lost no code block that overlapped the batch's own
citations.

**The switch.** Unset: whole plan, per-batch code windows. `cited`: cited plan sections and
per-batch code windows. `full`: the round-wide plan and code blocks in every batch. Any other value
is reported once and read as `full`. The Codex plugin forwards it (`packaging/codex/mcp.json`
`env_vars`); a name missing there never reaches the server.

**The plan block** (`utils/validation_sections.py`, called from `materialize_validation_round`).
A round qualifies (`cited_plan_for_round`) only when every batch can reach the rest of the plan:
the switch is `cited`, the block is the bound plan with a hash-bound snapshot, the runner may read
it (no `runner_read_refusal`; on a native host the snapshot is inside the project root), and the
plan is at least `CITED_MIN_CHARS` (12 000) long and has section headings. Then, per batch:

- *Headings.* ATX headings outside code fences; a lone `#` title belongs to the preamble. A fence
  closes only on a line of its own character (backtick or tilde) at least as long as its opener,
  so a four-backtick example showing a three-backtick block stays one fence. Setext headings (a
  line of text over `===` or `---`) are not parsed: a plan with one, or with a line that looks like
  one, is never cited (none of the 41 recorded plans has one).
- *What counts as cited.* Each suggestion's `reference` is split on ` / `, `; ` and a comma that
  starts another reference, and each part resolves against those headings. First rule that matches
  wins, and every heading it matches is taken: the heading text or its start at a word boundary
  (`## 2` → `## 2. Uploads`, never `## 20`), and for a part with no section key also every heading
  whose title without its key it starts (`Testing` → both `## Testing` and `## 3. Testing`); a
  numeric range (`§2.3-2.5`); the same section key (`9.2`, `stage a`, `t4.2`, `d5`); the title
  without its key; a heading title of six or more characters that starts the part; a key in
  parentheses (`### Fix (D5)` → `## D5 — …`).
- *Global sections.* A heading at the plan's shallowest section level whose title (without its
  number) has one of the words goal, non-goal, objective, scope, constraint, requirement,
  definition, glossary, terminology, assumption, invariant, principle or decision (singular or
  plural) is global: it rides in every cited batch with its subtree, whatever the batch cites.
  These are the sections any suggestion can conflict with. A nested `### Constraints` belongs to
  its parent and comes along only when that is cited.
- *What the batch embeds.* The preamble (text before the first section heading), the global
  sections, each cited heading's whole subtree, and the intro of every heading above it, merged in
  plan order; each omitted stretch becomes one `[... not cited by this batch, not shown: <first
  three headings> (+n more) ...]` line. Only a cited batch's records carry each suggestion's
  `reference`, so the validator sees which section each one names; every other batch's records
  render as before.
- *When it embeds the whole plan instead* (the default block, byte for byte): any suggestion in
  the batch has no `reference`, or any part of any reference names no heading (one resolved part
  never vouches for another, so the prompt never claims a section it does not show); the plan is
  unscoped (setext); the selection is 60% of the plan or more (`CITED_MAX_SHARE`); or it is over
  `VALIDATION_CONTEXT_LIMIT`. A batch is the unit, so one unresolved part gives its whole batch
  the whole plan.

A cited batch's plan source is `mode: path` with `selection: cited`: the prompt names the snapshot
under `## Plan file` and the task is `workspace-read`, so dispatch refusal, the executor's read
dirs, the resume re-hash and phase-dir rebinding (`verify_context_sources`) treat it exactly like
an over-limit excerpt. The prompt's `## Task` (`subagent_prompts._cited_task_section`) says the
block holds the opening, the global sections and the section each reference names as matched to
the plan's headings; it judges each group against those, tells the validator to read the file
before judging a group whose verdict needs a section not shown or whose reference seems to mean
another section, and judges every omission, "never defined" or "sections disagree" claim against
the full file, never the selection. A round whose only partial block is cited publishes no
`validation_context` (every cited section is whole in its prompt and the rest is one read away)
unless a verdict answered `insufficient-context:`; when another source publishes the round, the
plan entry reads "cited sections inline + full snapshot by path". One stderr line per cited round:
`[VALIDATION_CONTEXT_CITED] plan: 8 of 13 batches carry the plan sections their groups cite (…);
whole plan in 5 (a group cites a part that names no section of the plan); 52/90 reference parts
resolved`.

**The code block** (review-code). `build_validation_secondary_context(per_batch=…)` reads the cited
files once, as before, and hands back `for_batch`; `reviewed_code_for_batch` keeps, for each file
the batch's own issues cite, the windows that overlap those issues' windows, fitted to the same
budget by whole files. `reviewed_code_batch_view` turns it into the per-batch block, its own
`reviewed-code` source and its own dropped-file instruction, at both validation sites.
`reviewed_code_sha256` still hashes the round's whole untrimmed excerpt, so a round materialized
before this change resumes and reaggregates unchanged; `task_round` never hashed prompts.

**Determinism.** A selection is a pure function of the round's inputs and the batch's records, so
the same plan, `grouped.json` and files render byte-identical prompts per batch; batches now
differ from each other, which nothing compares.

**Measured** (2026-11-08, read-only replay of the recorded rounds in `plans/`, prompt chars).
review-plan, re-materialized from each round's `grouped.json` and snapshot (41 rounds, 622
batches): the default is identical to the code before cited sections; with `cited`, 321 batches
are cited and the round total goes 28.1 M → 21.0 M chars (−25.3%), median round −17.7% for the
34 plans of 12 000 chars or more. 494 of 3 617 reference parts name no heading (a task id or a
step inside a section, a quoted phrase, a heading renamed since): 244 batches fall back for that,
18 for the 60% share and 4 for the limit. Global sections are in 33 of the 41 plans (median 6.3%
of the plan) and cost 10 cited batches. An earlier rule, which let one resolved part stand for a
whole group, cited 503 batches for −42.6%, but kept 28 unresolved parts in cited batches.
review-code, approximated by re-selecting each recorded batch's code block from the round-wide
block (15 rounds whose prompts are inline, 105 batches): 12.5 M → 7.9 M chars (−36.6%, median
round −35.0%) from the code windows alone, by default. `cited` changes nothing there, because code
issues cite files rather than plan sections. These are prompt sizes, not spend: a validator that
reads the snapshot pays for that read, and per-batch prompts share a shorter prefix for provider
caching.

## Where the executor's helpers live

**`utils/subagent_executor.py` is a facade; the executor's code lives in thirteen modules below
it.** Lowest first, each imports only the ones before it, and none imports the executor:

| Module | What it owns |
|--------|--------------|
| `utils/batch_reply.py` | How a write reply is read, with no I/O: a check task's `check` (`read_check_reply`, `check_findings_from`), a code batch's own check and what it ran (`batch_check_findings`, `ran_check_outcomes`, `ran_no_check`), the items it declined (`declined_items`), the `WRITE_*` outcome words and labels, and the `row_*` readers of a recorded row. |
| `utils/failure_capture.py` | The redaction every executor write goes through (`redact.redact_text`, imported here, and `_redact_json`, the probe boundary `_redact_probe_tail`), the owner-only atomic JSON writer (`_atomic_write_json`) and the per-attempt captures under `failed/`. |
| `utils/task_outcome.py` | The records: `Attempt`, `TaskOutcome`, `BatchReport` and its file name (`batch_report_path`), and `ExecutorError` with `RunnerRejected`, `ResultRejected` and `BatchFailed`. |
| `utils/task_result.py` | A task's result file (D3f): `resolve_output_target`, `persist_result`, `extract_result`, `validate_result`, the salvage reads, `normalize_result` and `task_is_complete`. |
| `utils/runner_selection.py` | `profile_for_kind` and the jail grant's kind table beside it (`WRITE_SCOPE_BY_KIND`), `resolve_task_runner` and its source, the roster check (`eligible_runners`) and the refusal text a preflight leg raises. |
| `utils/probe_state.py` | What the process learned about its runners: the preflight caches, the sandbox decisions and the shared sandbox probe cache with their locks, the jail summary lines, the `reset_*` functions, and `sandbox_rung`. |
| `utils/sandbox_walk.py` | The sandbox gate: `sandbox_verdict` and `cached_sandbox_verdict`, the fallback ladder walk, the policy, the preflight's sandbox leg (`_detect_sandbox`, `repo_wide_write_refusal`) and the lines it prints. |
| `utils/dispatch_spawn.py` | One attempt's spawn (`_spawn`, returning a `_Capture`): the schema leg, the child environment (`dispatch_child_env`, which the auth probe reads too) and cached version, the process tree, the jail and what it reported (its `[JAIL_DENIED]` line from `landlock_dispatch.jail_denied_text`), the argv an audit record shows (`render_audit_argv`, `audit_argv`), the timeout kill (`_kill_timed_out`, which words its verdict by `process_control.unconfirmed_kill`), the idle stop and how it is read (`_idle_turn_finished`, `idle_stopped`, `idle_stop_case`, `runner_exit_code`, `idle_stop_note`), and the grant a write dispatch gets (`_dispatch_grant`). |
| `utils/attempt_classifier.py` | Which D3k rung an attempt lands on: `_classify_pre_parse` (rungs 1 and 2), `_parse_and_validate` and `_parse_with_schema_policy` (rungs 3 to 6), and `_classify_failed_run` with its redacted quote (`_quote_verdict`). Not `providers/failure_classifier.py`, the adapter-free stream rule it applies. |
| `utils/runner_preflight.py` | `preflight_runner` and its version, auth, model and nested-exec legs (`_probe_nested_exec` spawns one dispatch and classifies it), and `probe_setup_metadata`. |
| `utils/task_dispatch.py` | `dispatch_task` and its typed steps (below): the attempt loop, the state it carries (`AttemptState`), each hook's context (`hook_context`, `HOOK_CONTEXT_KEYS`), the re-asks and write hooks, the prompt budget, each attempt's record, the kind tables it reads (`SCHEMA_PHASE_BY_KIND`, `SCHEMA_REPAIRS_BY_KIND`, `RESENDS_BY_KIND`, `RECLASSIFIABLE_STATUSES`) and the D3g verification statuses (`STATUS_PARTIAL`, `STATUS_POLICY_VIOLATION`, `STATUS_BATCH_FAILED`). |
| `utils/task_batch.py` | `run_tasks`: the batch preflight, the D3i ceiling (`effective_ceiling`), the dispatch order, the concurrent fan-out (`_gather`) with provider backpressure and the judge audit, the recorded runners and provenance, and the `.status.json` mirror. |
| `utils/task_file_runner.py` | The entry points: `run_task_file`, the dry run (`dry_run_report`, D3l) and the continuation (`continue_past_marker`, D3b). |

`CHECK_FAILED_WITHOUT_FINDINGS` lives in `utils/known_checks.py`, which reads it and imports nothing
from the executor or these modules.

**The executor re-exports every moved name for reading only** (`subagent_executor._MOVED`, the
contract `write_batch_scheduler`, `providers/base`, `git_utils` and `run_status` already follow),
and the `retry_ladder` vocabulary: `subagent_executor.X` returns the owner's current object, and a
patch of `subagent_executor.X` reaches no code, because none is left there. The modules read a
moved name through its owner at each call (`_dispatch_spawn._spawn`), never a copy, so one patch of
the owner reaches them all. Patch the owner: `tests/test_subagent_executor_moved_names.py` fails a
test that writes any name on the executor, and holds the layering. A name the executor's modules
import from elsewhere (`get_provider`, `_resolve_executable`) is bound in each module that reads it,
so a test patching one patches it in the module whose code it drives, or in each of them.
`tests/test_import_layers.py` puts all thirteen at L3. New code imports from the owner, not the
facade.

**`dispatch_task` is a loop over typed steps.** `plan_dispatch` resolves once what every attempt
sends (`DispatchPlan`), or refuses with nothing spawned (`Refusal`). Each attempt is admitted
(`admit_attempt`: the provider's launch turn and the ladder's admission), passes the write path's
`before_attempt`, is spawned (`spawn_attempt`) and read onto its rung (`read_attempt`, returning an
`AttemptReading`). A schema-valid reply is persisted and asked about (`ask_about_reply`: rule 9's
re-send, rule 7's report re-asks), then accepted (`finish_ok`); any other is checked for moved bytes
on the write path (`check_moved_before_retry`), recorded with its `failed/` capture
(`record_failed_attempt`) and decided (`decide_next`: the ladder's one `record` call, the
schema-repair prompt, the backoff sleep), returning a `Next`; a stop ends in `close_outcome`. What one
attempt carries into the next is `AttemptState`, changed only by its `start_*` methods; every hook's
context comes from one table, `HOOK_CONTEXT_KEYS`. A step is tested alone from a hand-built
`_Capture` in `tests/test_dispatch_steps.py`.

## Read and analyze executor

**`utils/subagent_executor.py` is the second invoker of those task files (`--exec-subagents`, D3
read/analyze half).** One dispatch is: refuse before spawning (the D2 refusals plus containment) →
pick a **profile from the task kind**, never from the task JSON (`agent-readonly` for every
read/analyze kind, `cwd` pinned to the repo) → capture stdout → the adapter's
**`parse_output` first, then `json_extractor`** (provider CLIs emit event streams; raw
extraction would match an envelope or a reasoning item and write a garbage result reaggregation
would accept) → validate against the kind's result shape → atomic 0600 write under the phase dir.

`persist_result` is the **shared** invoker-persists leg: the test harness's
`simulate_native_invoker` (`tests/harness/native_invoker.py`, which
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
in prose is not a document, fix round 8; a document that never closes is one when nothing but JSON
lies between its opener and the piece, and the error says so), never the errors of a fragment
salvaged from inside it;
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
a chain its own CLI already ran. For the same reason an attempt the adapter's stdout watch stopped
keeps its `runner_error` but gets no identical re-dispatch either (rule 10, `stream_stopped`; see
[a CLI that loops](providers.md#a-cli-that-loops)).

The ladder is provider-free by construction (the envelope is read structurally) and each lane keeps
only its own three-line loop: `task_dispatch.dispatch_task` blocks, `review_orchestrator_base.dispatch_with_ladder` awaits — the
latter because it runs inside `asyncio.gather` under two semaphores, where a blocking retry
loop would serialize the whole fan-out. Each failed attempt writes `{phase}/failed/{task_id}.attemptN.json` with the
prompt elided to its sha256; attempt numbering restarts per invocation, so an existing file of that
name is first retired to `{task_id}.attemptN.<its own mtime, UTC>.json` (`_retire_capture`) rather than overwritten — a
retry never destroys the evidence of the attempt it retries.

**What each lane hands the ladder: the adapter's verdict.** The adapter's `classify_failure` is the
one classifier that knows its CLI's error shapes, so both lanes ask it rather than re-reading
a sentence composed for a human. The executor asks it for a run with no usable result
(`_classify_failed_run`) and lets the verdict replace the parse failure's status when it is one
that changes the slot's fate (auth, invalid model, sandbox, rate limit, context length, budget, or
a structural `runner_error`). Any other verdict, `None` included, leaves the parse failure's own
status, rung 4 or 5: one transport retry. The review fan-out (`dispatch_with_ladder`, the one
loop behind review-plan's and review-tasks' `run_single_model` and review-code's
`run_single_review`) takes it from `details["failure_category"]`, which
`llm_client.invoke_with_provider` sets on every failed child (a non-zero exit, and an exit 0 that
left no answer, below) while it still holds the streams (its reading is
`provider_outcome.outcome_from_capture`); only the category leaves, never the streams. The fan-out applies it the same way: `None` (the adapter recognised
nothing) is `runner_error`, connection and server errors are rung 4 through
`review_orchestrator_base._LADDER_STATUS_BY_CATEGORY`, and every other category reaches the ladder
under its own name. `spawn_failed` (an install or argv refusal, such as omp's unknown `--tools`
name) stops both lanes on a non-zero exit and fails the executor's batch, so it is read only off
exit 127, stderr (`failure_classifier.STDERR_LAUNCH_FAILURE_MARKERS`: a shell's `command not found`, omp's
refusals) and an adapter's own refusal shape (vibe's config refusal opening stdout), never off a
model's answer or a tool's output on stdout. A shell's phrase there may name a hook's, a wrapper's
or an MCP server's command rather than the CLI, so both lanes keep it only for a run that made no
progress: nothing on stdout and nothing else the adapter recognises on stderr
(`failure_classifier.launch_refusal`, asked by `provider_outcome` and `_classify_failed_run`).
Otherwise the verdict is the adapter's reading with only those phrases blanked, so a 401 on the
same line still decides (`None`, one transport retry, unless the rest names a rate limit or the
like); exit 127, omp's own refusals (`failure_classifier.CLI_REFUSAL_MARKERS`) and an adapter's
refusal shape stay fatal. `sandbox_unavailable` fails the batch too, so its
bubblewrap refusals (`failure_classifier.SANDBOX_MARKERS`) are whole measured lines read off stderr alone: an
answer that explains bubblewrap is not a broken sandbox; any other `bwrap: ` line on a non-zero
exit is `runner_error`, one transport retry, with a reason that names `multi_llm.py doctor`
(`attempt_classifier._sandbox_hinted`). Both read only the stderr of an adapter
whose CLI writes it alone, never one that relays a model's or a tool's words there
([launch diagnostics](providers.md#launch-diagnostics)). One verdict stops on the
fan-out and is retried once by the executor, which does not forward it: `permission_denied` (a 403
or a denied tool), which can fail one task and not its neighbour, so the difference is deliberate.
On both lanes a run killed at its deadline is `timeout` whatever
its partial streams say (the executor's `_classify_pre_parse`), and a jail denial is never retried
whatever the category. The fan-out's error line on a non-zero exit ends with the category
(`<cli> exited with code 1: <reason> [rate_limited]`), so `.status.json` names why a slot stopped
or retried. Read by `classify_streams`, that tag can only ever name its own category. An adapter
whose `classify_failure` raises, or answers something other than a category name or `None`,
sets no key, and that slot falls back to its error line, below.

On an **exit-0 failure** (an answer that did not parse, and no result file) stdout is usually the
model's own answer, prose quoting whatever the plan says about 429s and 401s, and **no answer may
decide a retry, a backoff or a stop on either lane**. Both lanes read such a run through one gate,
`providers/failure_classifier.exit_zero_failure_category`. It gives a verdict only for an adapter that declares
`EXIT0_CLASSIFIABLE = True`, a reviewed claim that its `classify_exit_zero` never reads the answer,
in a complete envelope or an interrupted one: claude-code, cursor-agent, gemini and grok read their
envelope without the answer field (and a cut or ambiguous stdout not at all), the one the parser
reads, through one reduction on every exit code ([reading a capture](providers.md#reading-a-capture));
the executor's failure reason quotes the same reduced streams, so an answer that quotes a diagnostic
is never the quote of the CLI's own; and codex, opencode, omp, pi,
cline, kimi and fx through `vendor_text` reductions that keep only the CLI's own events. Each
adapter's declaration carries its reason. agy, aider, goose, kilocode, copilot, muse, qwen-code and
vibe declare `False`: their answer cannot be told apart from what the CLI wrote (aider's answer cut
before its accounting line can quote its own `litellm.RateLimitError: …` line verbatim; goose's
model turn can open with its `Ran into this error` wording and lacks the `metadata.inference` stamp
whenever goose could not resolve the model), so the run is left unclassified, the fan-out stops on
its error line and the executor keeps its parse failure's one retry; their exit-0 rows are marked
`open`. A line's shape cannot say who wrote it, so only a structural marker the model cannot write
earns `True`. `spawn_failed` from the gate is read as nothing (a CLI that refused its own invocation
never exits 0). A stdout answer that parses but is not one (`_is_valid_parsed_data`, a bare `raw`
wrapper) takes one verdict on both lanes: the gate's when it names a category (a real diagnosis on
stderr, say), else `unparseable_result`. The executor's `_classify_failed_run` asks the gate after
its own rejection; `invoke_with_file_output` asks it the same way, through
`provider_outcome.rejected_answer_category`, over the capture the exit 0 handed back, and carries the
verdict on `details`. A failure with no verdict at all (a launch failure, an adapter that
raised, an adapter that is not `EXIT0_CLASSIFIABLE`) is classified by its error line alone with
`classify_streams`, retrying only what that line names (a rate limit, a connection or server
error); an unrecognised line is never retried. `tests/test_exit_zero_prose_matrix.py` feeds every
adapter rate-limit, auth, context-length, numeric and launch prose, and quoted diagnostic lines, in
its complete, unfinished and cut envelopes and asserts neither lane moves off its status for a
neutral answer; it also gives every adapter a rejected `raw` answer with and without a stderr
diagnosis and asserts both lanes agree.
`tests/test_lane_a_ladder.py` replays every committed failing fixture row, exit 0 included, and
every row as a deadline kill, through both lanes' real paths (`invoke_with_file_output` into
`run_single_model`, and `dispatch_task` with the spawn stubbed and the profile gate lifted) and
asserts the same number of dispatches, apart from the deliberate verdict and the `open` exit-0
rows above.
`tests/test_code_review_ladder.py` drives the same verdicts through review-code's slot and asserts it
dispatches each as often as review-plan's. On both review lanes the slot's deadline starts
(`slot_ladder`) before its prompt and result baselines are prepared, which it pays for. A slot that
raises is a failed result carrying `Exception: <type>: <message>`, redacted (`slot_exception_error`),
never a deadline loss. When a dispatch raises, the attempts before it keep their spend, audits and
home checks, the raising attempt counts as unaudited, and the write, mode and home checks run as
after a return, so a write in the window still fails the slot by path. The raising attempt's home
check is the ladder loop's own read-back of the jailed children it armed (see
[providers](providers.md)), and a write it fails is recorded with `source: exception`. Cancellation
(`CancelledError`, `KeyboardInterrupt`) is not caught there: the fan-out's teardown retires the slot.

**Launch spacing is between dispatches, per provider.** An adapter with a measured concurrent-start
fault declares `LAUNCH_STAGGER_S` (cursor-agent: 2 s); every other adapter declares 0 and is never
throttled. Each fan-out (`run_models_concurrent`, used by review-plan, review-tasks and ask;
`run_tasks_concurrent`; review-code's and review-plan's own loops) shares one
`utils/launch_limiter.py` limiter across its slots and their retries. A slot takes its provider's
turn only once it holds both semaphores (it reads `queued` while parked, `staggering` while it waits
for the turn), so slots queued behind other providers cannot dispatch together. Every ladder attempt
then takes a turn right before it dispatches (`dispatch_with_ladder`, and `dispatch_task` on its
worker thread): the first attempt confirms the slot's turn or waits again if another slot of the
provider took one meanwhile, a retry always takes a fresh one, and the wait is charged to the slot's
deadline. A lane with no ladder (ask, review-plan without an output dir) takes the turn the same
way at its one dispatch, after its pre-dispatch probe and snapshots, so slow preparation cannot eat
the spacing. Turns are served in arrival order and taken when claimed, never reserved: a waiter that
leaves (cancelled, or its deadline passed) takes no turn and wakes the waiters behind it to work
their turns out again, and no turn is granted once the waiter's deadline has passed. A retry waiting
for its turn reads `staggering` but has dispatched, so its row keeps counting run time. Lane B's
blocking waiter is not stopped when its asyncio slot is cancelled, like the dispatch it precedes; no
adapter with a launch interval serves the executor's profiles today. The overall deadline adds the
largest provider's slot count times its stagger.

A batch with any failed or incomplete task writes `{phase}/subagent_batch_report.json`, refuses to reaggregate and
exits **3 even under `--yes`** — and **never** falls back to salvage. Resume is
defined by artifacts: a task is done iff its `output_name` exists, parses and validates, so
re-invoking a phase re-dispatches exactly the failed set. `run_tasks_concurrent`
(`utils/review_orchestrator_base.py`) is `run_models_concurrent`'s mechanics keyed by `task_id`
instead of `model_spec` — same global gate, same per-provider semaphores, same launch spacing —
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

**And the gate that decides all of this now reads stdout, not stderr alone.** `provider_outcome._attach_stdout_excerpt`
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

## An idle writer

**An executor-leg write subagent that prints nothing for `driver.subagent_idle_timeout_s`
seconds (default 900) with no tool call in flight is stopped. What it printed is read as after an
exit only when it had finished speaking; a writer that went quiet in mid-turn is the timeout it
would have been at the deadline, only sooner.** The incident: an improve-plan batch on
`codex:gpt-6-luna:high` edited the plan, printed an `agent_message` in prose, and then printed
nothing for 25 minutes, with no `turn.completed` and no exit. Killing the codex process by hand
ended the attempt with exit -15; the prose did not parse (`unparseable_result`, rung 5), the
pre-retry check saw the moved bytes, and the report re-ask restated the report in 17 s. The run
finished normally. Without that kill the 1800 s slot deadline would have ended it as `timeout`,
and a timed-out batch that moved bytes is a partial application: a run stop.

- **The stop is the timeout's own; where the runner stopped decides the reading.**
  `process_capture.PipeCapture` keeps an idle clock (`idle_s`, `in_flight`): the time of the last
  byte on either pipe, and the adapter's tracker's word on whether a tool call is running. Once it
  expires, `communicate` raises `IdleStopped`, a `TimeoutExpired` but never a `StreamStopped`, and
  `dispatch_spawn._spawn_in_context` stops the tree through `_kill_timed_out` and drains it. With
  the kill confirmed, the adapter's `LLMProvider.turn_finished` reads the captured stdout
  (`dispatch_spawn._idle_turn_finished`, which fails closed: a capped stdout, an adapter with no
  answer and a raise all read as mid-turn), told whether the attempt was a report re-ask
  (`reask`, which `task_dispatch.spawn_attempt` sets from `AttemptState.repair`: rule 7's two
  re-asks, never rule 9's re-send, which is the batch again). codex's (`codex.finished_speaking`)
  reads its events only: finished when the last event is `turn.completed`, or an `item.completed`
  of an `agent_message` with a tool item before it; on a report re-ask, an `agent_message` last is
  finished with no tool item before it, since a re-ask runs no tool by design.
  - *Finished speaking:* the capture is not `timed_out`, `exit_code` is the signal's, `idle_stop`
    carries the reason, and `read_attempt` parses the stream like any other exit. That is the
    manual kill's path, so:
    - edits, then prose, then silence (the incident): rung 5, moved bytes, the report re-ask
      (rule 7), then the snapshot comparison decides;
    - edits, then a schema-valid report, then silence: accepted. The write lane judges the reply
      with `dispatch_spawn.runner_exit_code`, which is `None` after this stop, so
      `WriteFacts.crashed` does not send it to the verdict table's row 5. The runner was silent
      for the whole timeout with no tool running and its reply as its last word, so no edit was
      under way, and its whole tree was gone before the snapshot comparison, which sees every
      byte it moved. The attempt's row keeps the real exit code, and its reason names the stop;
    - silence with nothing moved: the ordinary transport retry (`driver.subagent_retries`).
  - *In mid-turn:* a tool item last, a message with no tool item before it on a dispatch of the
    batch (only the opening preamble: gpt-6-luna says what it will do before its first tool
    call), a `reasoning` item, an `error` event or a `turn.failed` after the message, or no event
    at all. The capture stays `timed_out` with no exit code, and `_classify_pre_parse` records
    `timeout` ("… in mid-turn: …"): nothing it printed is parsed, and it is never re-asked. The
    pre-retry check then makes a batch that moved bytes a partial application
    (`ran_out_of_time`: its edits are kept, and the resume sends it again on top of them), and
    one that moved nothing gets the rung-2 retry. The stop's words name the idle timeout and suggest another model, not a longer
    `subagent_timeout_s` (`StopAccount.stalled`, `stop["idle_stalled"]`). Before this rule a
    re-ask after such a stall quoted the preamble, and a runner that attested every id was
    `applied` with the edits after the stall missing (Fable review 2026-11-19, finding 3).
  - Each attempt's row in the batch report says which case applied: `idle_stop` is
    `finished_turn`, `mid_turn` or `unconfirmed` (`task_outcome.IDLE_STOP_*`,
    `dispatch_spawn.idle_stop_case`), and the run's log line says so too.
- **The limits.**
  - *Commentary between tool calls reads as finished.* gpt-6-luna also speaks between tool
    calls, and the event stream marks no message as the last one before `turn.completed`. A
    writer that stalls right after an intermediate `agent_message` that follows a tool call is
    read as an exit: its prose draws the report re-ask, and a runner that attests every id is
    applied on the moved bytes, with any edit it meant to make after that message missing. The
    incident was of this kind: its last message ended "I'm checking the resulting sections now".
    Reading that shape as mid-turn would have stopped the incident's run at the deadline instead
    of recovering it.
  - *A lone message on a dispatch of the batch reads as mid-turn.* A writer whose only output
    is one message with no tool call before it (a report that declines everything, then
    silence) is retried as the batch, as at the deadline. Rule 9's re-send, which is the batch
    again, is read the same way.
  - *A report re-ask's lone message reads as finished.* The re-ask asks for the report alone and
    runs no tool, so its report is one message with nothing before it. A re-ask that prints its
    report and then hangs is read as an exit, and its report is judged like any other: applied
    when it is schema-valid and the re-ask left the files as the guard found them
    (`repair_baseline`; a re-ask that moves a file is failed). Until Fable review 2 of
    2026-11-19 (N4) it was read as mid-turn: it waited out the idle timeout on every re-send (up
    to 900 s each) and then stopped the batch as a partial application with a valid report in
    its capture. The cost: a re-ask that prints only a preamble and then hangs is read as
    finished too. The preamble is no report, so the re-ask ends as one that gave no readable
    answer does (`task_dispatch.close_outcome`): the schema-repair re-ask's unparseable reading
    stands, and an incomplete-report re-ask leaves the first report to be classified as it
    stands. The preamble is never read as the report.
  - *A late hang still meets the deadline.* A writer that goes quiet less than the idle timeout
    before its slot deadline (after about 900 s of codex's 1800 s) reaches the deadline first, and a
    timed-out batch that moved bytes is a partial application, as before.
- **An unconfirmed kill fails closed.** A survivor may still write, so the attempt stays
  `timed_out` and is recorded `runner_error` with the kill verdict (`_classify_pre_parse`): never
  parsed, never re-asked, a partial application if bytes moved. It is not rung 2, because it did not
  run out of time. Its row's `idle_stop` is `unconfirmed`.
- **Only a live event stream has a clock.** `LLMProvider.idle_watch()` returns a fresh in-flight
  tracker, or `None` (the default). codex declares one (`codex.ToolsInFlight`): an `item.started`
  of any type but `agent_message`, `reasoning`, `todo_list` and `error` opens a call, its
  `item.completed` closes it, and `turn.completed` or `turn.failed` closes all. A test suite run
  in codex's shell prints nothing between its `item.started` and `item.completed`, so the clock
  waits for it, and an exec session still running at its yield time keeps the clock off until it
  ends (the deadline then decides). A line the tracker cannot read changes nothing, so a missed
  completion keeps the clock off rather than a missed start letting it run under a working tool;
  a tracker that raises turns the clock off. claude-code prints its answer at exit
  (`--output-format json`), so a timer on its stdout would kill every healthy writer that works
  longer than the timeout: it declares none, and has no idle timeout. Giving it one needs
  `--output-format stream-json --verbose` and a tracker over its `tool_use` and `tool_result`
  blocks.
- **Write kinds only, executor leg only.** `plan_dispatch` reads the key for an `agent-write`
  profile (the report re-ask included) and hands it down; read kinds, the preflight probe and the
  review fan-out never arm it. On the native leg the host's own Agent subagents write, and the
  plugin cannot stop them: the key does nothing there.
- **Why 900 s.** The clock only has to outlast the model's own silences, because a running tool
  stops it. In 1,848 healthy codex write attempts (apply-suggestions and apply-code-fixes on
  `gpt-6-luna:high`, from the improve-mode benchmark and test runs), the longest whole attempt
  took 463 s, so no silence in them was longer. codex's session rollouts stamp every item with
  its start and end times; rebuilding what `exec --json` prints from 3,185 finished turns on one
  developer machine, the longest silence with no tool in flight was 130 s in 159 `codex exec`
  turns on `gpt-6-luna`, 228 s over all 288 `gpt-6-luna` turns, and 554 s over every model on the
  app-server sessions (`gpt-6.1-sol` at `xhigh`, writing a large patch). Only the interactive
  TUI's sessions went longer, in gaps with no model activity inside them: a person's wait.
  Counting a running tool as silence instead, a flat 900 s timer would have stopped 12 of those
  healthy turns (a single tool ran 880 s in a `codex exec` turn), which is why the clock waits for
  tools. A model stream codex itself gives up on is no silence either: each reconnect prints an
  `error` event (`Reconnecting... 1/5`, `tests/fixtures/codex/rows/error-envelope.stdout.txt`),
  which restarts the clock. 900 s is 1.6 times the longest silence seen, and it still stops a hang
  like the incident's well before the 1800 s slot deadline, on the path that recovers.
- **Configuration.** `0` turns the clock off. A blank value keeps the base default, as for every
  `driver.*` key (`_deep_merge` skips a null), so `0` is the one way to disable it. Validated as a
  whole number of seconds (`DRIVER_NONNEGATIVE_INT_KEYS`).

Tests: `tests/test_idle_writer.py` (the capture's clock against real child processes, the codex
tracker over every committed codex stream, `finished_speaking` over the incident's event shape and
every committed stream with its `turn.completed` cut, the stop through the executor's real kill
path against stand-in CLIs, the scheduler's outcomes against the mock codex's
`MOCK_LLM_HANG_AFTER_REPLY` with `MOCK_LLM_CODEX_TOOL_ITEMS` (the incident's shape re-asked and
applied; a mid-turn stall and a lone message stopped as partials with no re-ask, a re-ask that
would attest every id included; a re-ask that hangs after its report, with
`MOCK_LLM_HANG_AFTER_REASK`, applied on that report), the `reask` reading of a lone message at
each layer, and the key's validation).

## Tool-use counts on the executor leg

**A codex writer's tool calls are counted from its stream and recorded where the native leg
records the host's count.** The native leg passes the host's per-subagent tool count to
`util metrics record --tool-uses`, which lands in `state.json` as `metrics[<phase>][].tool_uses`
and in the metrics report's Tool Uses column. The executor records the same field through
`WriteEta.record`, from `usage["tool_uses"]`, and the batch reports (`write_batch_report.json`,
`subagent_batch_report.json`) carry it in each task's and each attempt's `usage` and in the
totals. codex's usage held only tokens, so the field was missing on the codex executor leg.

`CodexProvider.parse_usage` now counts the distinct ids of `command_execution`, `file_change`,
`mcp_tool_call`, `web_search` and `collab_tool_call` items the stream started or completed
(`codex.CODEX_TOOL_ITEM_TYPES`), so each call counts once, one a stop cut short included. A
code-mode `exec` cell is no item of its own; the tools it calls are. `todo_list` is left out:
codex folds every plan update into one item, so those calls cannot be counted one by one, while
the host's count includes its plan tool's calls. Writer streams run `--ephemeral` and leave no
rollout, so the stream is the only record; the executor's drain captures it line by line, and
`parse_usage` reads that same capture on every attempt, so the count is the live one.
`_accumulate_usage` sums it over a batch's attempts, the report re-ask included. A stream a stop
cut short (no `turn.completed`, so no tokens) still reports its count, a finished turn with no tool
call reports `0`, and a stream with neither reports nothing (absent, never a made-up zero). A count
is no token figure: `_accumulate_usage` marks a task's usage reported only once an attempt reports a
`*_tokens` figure, so a task whose attempts left only a tool count keeps it in its `usage` beside
`usage_reported: false`, and the batch report's totals sum the count while still counting the task
under `tasks_without_usage` (the bench ledger reads that as "tokens incomplete"). The
review fan-out's codex slots gain the same count, which their usage records already carry for
opencode and pi. Tested end to end, to `state.json` and the batch report, in
`tests/test_idle_writer.py` with the mock codex's `MOCK_LLM_CODEX_TOOL_ITEMS`.

claude-code's `--output-format json` envelope holds no tool blocks (`num_turns` counts model
turns, several tool calls can share one, so it is not substituted), so a claude-code writer on
the executor leg still records no `tool_uses`; counting them needs the same `stream-json` argv
change as its idle clock.

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
