# Review artifacts and reports

Read when changing aggregation, stable group IDs, review snapshots, result recovery, report decisions, or report regeneration.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Snapshots and empty review outcomes](#snapshots-and-empty-review-outcomes)
- [Code-review scope and diff files](#code-review-scope-and-diff-files)
- [Stable group IDs](#stable-group-ids)
- [Output paths](#output-paths)
- [Validation statuses](#validation-statuses)
- [Importance levels](#importance-levels)
- [Grouping and section references](#grouping-and-section-references)
- [Consolidation batches](#consolidation-batches)
- [The shared review lane](#the-shared-review-lane)
- [Reaggregation and result identity](#reaggregation-and-result-identity)
- [Round identity](#round-identity)
- [Malformed result entries](#malformed-result-entries)
- [Report payload](#report-payload)
- [Report regeneration](#report-regeneration)
- [Scope tags in reports](#scope-tags-in-reports)
- [Issue numbers and group decisions](#issue-numbers-and-group-decisions)
- [Report grammar](#report-grammar)
- [Scope tags and the review round](#scope-tags-and-the-review-round)

## Snapshots and empty review outcomes

**A review phase snapshots the artifact it reviewed, named after what it holds, and hands back every
file it wrote.** `review-plan` writes `review-plan/backup.md` (the plan);
`review-tasks` writes `review-tasks/tasks-backup.md` (`tasks/tasks.md`) — it used to write a
44 KB `backup.md` copy of the one file it never touches, which nothing reads
(`output_handler` pins the `backup` output type to review-plan) and which
`apply-task-suggestions` duplicates under its own `tasks-backup.md` anyway. Neither review phase
overwrites anything, so these are records of what was reviewed, not rollback points.

`aggregate_results` records `validation.json` through `detach.record_output_files` as well as
`record_artifacts`: the host reads the *poll* result, whose artifacts are rebuilt from
`.status.json.output_files`, and `apply_task_suggestions_orchestrator.load_data` hard-fails without that file.

The facts every review lane reports the same way — the per-model `progress.models` map and the
`models_done` / `models_succeeded` counters derived from it, the deadline back-fill of
`failed_models`, the artifact hand-back, and the distinct-HIGH-group count
(`high_priority_groups` in `.status.json` and on the result line, for review-plan and
review-tasks as well as review-code since 2026-09-17) — are one helper each (`progress_tracker.FanoutProgress`
— its own module since the map outgrew the bare counter `fanout_slot_counter` used to be, and
because ask and review-code reach it without going through the review base at all; then
`backfill_deadline_losses`, `record_review_artifacts` and `record_high_priority_groups` in
`review_orchestrator_base`), and review-code's on-disk batch merge goes through the shared
`merge_validation_batches`; none of them is copied into `code_review_orchestrator.py` any more.

review-code's two completion sites, the fan-out and `--reaggregate`, end in one
`finish_code_review` (the summary, `record_high_priority_groups` and the `--fail-on-high`
gate), reached after the phase is recorded complete, so the host-driven route records the
count and holds the gate too; a `--skip-validation` reaggregate counts the groups on disk,
as the base report does. Both lanes' `--reaggregate` resolve a validation round through
one function ([The shared review lane](#the-shared-review-lane)): stale prompt sidecars
are rebound and persisted, a whole round of `validation_batch_<n>.json` is merged, a
partial one is dispatched under `--exec-subagents` or answered `Only n/m batch files
found.` with the phase left open. The merge reads a missing batch as `validation_failed`,
so it is never handed a partial round. No broad `except` surrounds the merge or the
dispatch: one that raises ends the command with the phase open, never completed with no
`validation.json`.

Both lanes' `--reaggregate` reopen the phase wherever they hand a round out, a partial one
or a newly built one, through the base's `reopen_phase_for_validation`: `state:
validation_pending` (and the progress step on the host hand-off), and a completion an
earlier `--skip-validation` run recorded in `state.json` is dropped, as `--force` drops it.
Otherwise the command asks for validation while `status` and the next-step builder still
read the phase as complete. A dispatched round records the phase complete again when it
merges. The drop is one read-modify-write under `metrics.state_file_lock` that removes only
that phase's entry, so a completion or `commit_range` another command records at the same
moment is kept.

**An empty review outcome reaches its apply phase as a completion** (2026-10-09). review-plan's
fan-out with zero groups (the `else` of `run_review_orchestration`'s
`if all_suggestions:`) and the base `--reaggregate` that finds none retire the previous
validation round and write a zero-group `grouped.json` and `validation.json` as
`[]`, as code-review's own empty branch already did — before, the round wrote
neither, apply-suggestions' `load_data` exited 1 on the missing file and a clean plan
stopped `full` at exit 3, and an empty `--force` round left the
previous round's files to be applied again.

A model whose parsed answer is `[]` counts as a completed review
(`_answer_text` serializes it whatever its length; both success returns of
`run_single_model` tested its truthiness and recorded the model failed, `Unknown error`).

And apply-code-fixes' no-items exit (`handle_no_items_early_exit`) records the phase completed, never
skipped, except under `--dry-run` or `--dry-run-subagents` (a preflight records nothing;
apply-task-suggestions' own no-items skip follows the same rule), so an all-invalid code review no
longer reads as an unfinished apply; reached by a native `--resume` after every batch
landed, it prints `--mark-completed`'s `[TESTS_NOT_RUN]` line too (`print_native_tests_not_run`),
and apply-suggestions' completion at emit under `--resume` re-stamps
`plan_hash` as `--mark-completed` would (Fable review 2026-10-09).

## Code-review scope and diff files

**A code reviewer gets its scope and its diff as files, and its prompt says what it can do with
them** (plan `review-run-scope-salvage-and-failure-visibility`).

`review-code` resolves one scope (`resolve_review_scope`:
`--files`/`--files-from`, then `tracked_files`, then the loud
`git diff` fallback with `[SCOPE_FALLBACK]`/`[SCOPE_LARGE]`), orders it
source → tests → fixtures → generated with the single classifier `review_diff.classify_review_paths`
(`code_review.collapse_paths` in `providers.yaml` configures the fixture/generated globs), and
before dispatch writes `code-review/review-scope.md` plus one whole, uncapped `code-review/diff/<n>-<slug>.diff` per
source/test file (`write_review_diff`; git failures degrade coverage to `partial`
with `[DIFF_FAILED]`/`[DIFF_PARTIAL]`, never the phase).

`.review-scope.json` records which scope id each result reviewed, so a resume re-runs a result
from another scope (`[SCOPE_CHANGED]`, old file kept as `.stale-<id>`). The prompt's
`{diff_access}` block is rendered by `prompt_channels.render_diff_access` from two
`generate`-profile declarations on the adapter, never from a provider name:
`runs_shell_in_generate` (`True`/`False`/`None`,
measured sources only, pinned by `TestShellDeclarationMatchesTheEvidence`) and `generate_read_scope`
(`none`/`cwd`/`any`).

A lane that cannot open the phase dir gets plan, manifest and diffs inline under a 96 KB cap (argv
`MAX_ARG_STRLEN`), or is recorded failed with `[REVIEW_UNSUPPORTED]` without being dispatched. On a
commit review with `--context-plan`, the frozen snapshot is one more inline block (`context-plan.md`)
under the same cap, and an unreadable snapshot is unsupported too: no lane judges the range without it.
Every Lane A review template also carries `{working_limits}` (`render_working_limits`: the
attempt's real minutes, no suites/builds, no unobserved test claims), which names no shell command
because `{diff_access}` alone decides that.

## Stable group IDs

Group IDs are 16-char hex content hashes — use `state_manager.py` hashing, don't generate random
IDs

**Records join on the stamped id, never on a recomputed one.** `generate_group_id` reads neither
`file` nor `line_range`, so one finding in twelve files is twelve groups with one content hash.
`stamp_stable_ids` gives the n-th of them discriminator n-1, with no cap (PERF-11: the loop used to
give up after 10), and stamps the result as `group_hash`. Every writer files under that stamp:
validation task ids and the verdicts that echo them (`stamped_group_id`), `validation.json`,
`processed_<phase>` (a batch's `expected_ids`) and human decisions (`group_id_of`). Readers join
through `group_join_ids`, which gives each group its stamp plus the plain hash where that differs
(the key records had before every writer used the stamp, such as a resume record for an edited group),
each only where it names one group. `records_by_unique_id` leaves out an id two records carry. A
shared plain hash therefore stays with the group whose stamp it is: a `validation.json` from before
the stamp leaves its same-content groups undecided on the review side and joins them by index on
the apply side, and a validation batch whose task file gave two groups one id discards the verdicts
that echo it, so those groups are revalidated. Recomputing the plain hash gave every same-content
group the first one's verdict, resume record and decision (2026-11-08: eleven invalid findings read
valid and were applied, and a `--resume` dropped eleven unapplied groups).
`tests/test_same_content_group_identity.py` drives the real merge, apply and resume paths.

## Output paths

All output paths are plan-relative for portability

## Validation statuses

Validation statuses: `valid`, `invalid`, `needs-human-decision`,
`validation_failed`

## Importance levels

Importance levels: HIGH > MEDIUM > LOW. `utils/importance.py` owns the spelling (`LEVELS`), the
order (`RANK`) and the score table (`WEIGHT`). Every reader goes through it (`normalize`, `rank`,
`weight`, `is_high`, `highest`, `group_importance`) and never compares the string itself.

- **Read either case, write upper case.** Reviewers answer in the schemas' lower-case enum
  (`MODEL_VALUES`), and artifacts from before 2026-11-08 carry either case: the plan lane wrote
  upper case and the code lane lower case. Reads accept both. The plugin stores upper case: the
  base lane's `save_model_result`, and on the code lane `run_single_review`, the fan-out's
  flattening and the reaggregate loader (`canonicalize`).
- **A value that names no level reads as MEDIUM** unless the caller passes another default. The
  plan lane drops such a suggestion; the code lane only warns and keeps it as written, so the
  report lists it under MEDIUM instead of in no section.
- **A group has no importance field.** `SuggestionGroup.to_dict` writes none; derive it from the
  members with `group_importance` (`members_importance` for a bare member list): the highest
  member, a member naming no level counting as MEDIUM, as its own badge does. Only a group with no
  members has no level; the HTML report shows that as `""` (no badge, sorted after LOW). The HTML
  report's `maxImportance`, its priority sorts and apply filtering share this one rule, and the
  report templates take a group's level and the rank table from the generator's data
  (`maxImportance`, `sortConfig`) rather than deriving their own.
- **Ids.** Importance enters neither `generate_group_id` nor `generate_suggestion_id`, so a
  spelling change moves no id. The rank does order a group's members and the groups themselves,
  so a regrouping can give a mixed-importance group a different theme (and so a different
  `group_hash`) than a grouping made before the case fix. An in-flight round still reuses its
  bound `grouped.json`, and a regroup retires the verdicts it orphans
  (`retire_validation_for_regroup`).

## Grouping and section references

**Grouping tracks meaning, not word count (2026-09-11).** `suggestion_processor._similarity` is
`0.33·title + 0.53·body + 0.08·reference + 0.02·type + 0.04·title-rarity` over normalized tokens (inline code spans tokenized like prose — a backticked
identifier is the most specific thing a reviewer names — camelCase/dotted identifiers split,
stopwords and review boilerplate removed, suffix-stemmed) weighted by IDF over the set being grouped
— which is what demotes a plan's own subject word without a per-project stopword list; the reference
channel is scaled by section rarity (a flat bonus handed every pair inside the review's biggest
section the full 0.08, exactly where the near-threshold false pairs live), the title-rarity channel
is the IDF mass of shared title tokens above `TITLE_RARITY_FLOOR` and is inert on small sets by
construction, and two findings naming different files never merge (path-suffix aware; a file is
evidence of difference only).

`group_similar_suggestions` is agglomerative average-link (merge the closest pair, average over all cross
pairs, stop at `DEFAULT_SIMILARITY_THRESHOLD` 0.20 — the safe band on the 2026-09-13 review is 0.196–0.205;
below it a third finding joins a coherent pair, above it one comes apart), so it is
order-independent by construction and cannot chain. The misses that remain on that review are
structural, not tuning: a pair whose members each prefer a different neighbour cannot land in one
group under average-link without the chaining the design rejects.

A group's `theme` is `summarize_theme` (60-char budget, word-boundary cut,
`…` only when cut) but `state_manager.generate_group_id` hashes the theme at the legacy
50-character cut, so a display change is not an identity change and the container goldens' baked ids
hold. The old `0.3·title + 0.5·desc + 0.2·(type match)` bag-of-words with greedy single-link at 0.4 was uncorrelated
with meaning: on a real 56-suggestion review it filed "activation semantics undefined" with two
opponent-parity findings (the 0.2 type bonus was half the threshold) while nine obvious clusters
stayed split across 23 groups; the new rules give 28 coherent groups where it gave 46. Every e2e
fixture groups to byte-identical `group_hash` lists under both, so no golden moved.

`html_report_generator.split_section_refs` files a group under every section a member cites — a composite
`reference` such as `## Design Requirements / ### 7. Lap Integration` is split on a delimited ` / `
or `, ` (a bare slash is part of a heading), so per-section counts no longer
under-report.

## Consolidation batches

`utils/consolidation.py` clusters the valid groups of a finished round (`--consolidate`, and
`full-workflow`/`improve-plan` on `[CONSOLIDATION_RECOMMENDED]`). It reads the group record
`SuggestionGroup.to_dict` writes, which has no `reference`, `importance` or `type`.

- **Importance comes from the members.** Group records in the prompt, singleton entries and the
  consolidated reports use `importance.group_importance`; a merged cluster is never below its most
  important group (the level the prompt asks for). Both report pages show the level Python sends:
  a value naming no level reads as MEDIUM, as on the flat and PR pages.
- **A suggestion's text is `desc`** (`description` only in older records). The content measure and
  the truncation of an oversized solo group read it.
- **Batches are cut by count alone**, in priority order, in one `_uncategorized` section: no group
  carries a `reference`, and `CONSOLIDATION_BATCH_CHAR_BUDGET` is `None`. This is how every recorded
  consolidation ran. Both alternatives give more, smaller batches, and a cluster forms only inside
  one batch, so duplicates in different batches stop merging. On 29 recorded rounds a 12,000-char
  content budget split 8 of them into one to three more batches, some holding one group. Sections
  derived from the members, or a content budget, wait on a check against a recorded `grouped.json`.
- **Ids.** A group's consolidation id is the 12-char `generate_group_id` (theme and titles),
  assigned through `consolidation_group_ids`: the first group with a given hash keeps it, so an
  ordinary group's id is unchanged; each later same-content group gets one derived from its stamped
  `group_hash`, so it stays its own entry. `consolidated.json` says so with `group_id_format:
  "distinct"`; a file without it is from before, when every copy was filed under the first one's
  id, so a decision on that id covers every copy, as it did when it was made. A
  `consolidation_tasks.json` from before listed every copy under that id too: its `group_indices`
  say which copies each batch was given, and that batch's reply is read through them, so a copy
  no batch was given (validation excluded it) stays out.
- **A reply names only its own batch's groups.** Each reply is read through its batch's entry in
  the task file: an id the batch was not given (a guess, another batch's group, an excluded group,
  a non-string) is dropped before anything is claimed, and every index in an entry comes from an
  accepted id, never from the reply's `underlying_group_indices` or `group_index`, so the decision
  join (ids) and the pages (indices) show the same groups. What is left of a cluster sets its
  shape: nothing, a singleton, or a cluster. An error reply's `groups` list is not read: every
  group of the batch becomes a singleton. In 9 recorded rounds whose task files predate embedded
  prompts, models invented ids ("group_3", slugs); the merge showed those clusters by the model's
  indices and the same groups again as missing singletons, and decisions on them reached no group.

## The shared review lane

The review base runs review-plan and review-tasks; `code_review_orchestrator.py` runs
review-code. `utils/review_lane.py` holds what the two lanes share, so a fix made once
reaches both, and `tests/test_review_lane_parity.py` runs each scenario below on both
lanes offline (real phase dirs, fan-out mechanics, grouping, rounds and reports; only the
model slots stubbed). A scenario that passes on one lane and fails on the other is drift.

- **`ReviewLane`** is what a lane supplies where the lanes legitimately differ:
  `retry_command` and `round_follow_up` (review-code spells a commit review's commands in
  its commit mode; the base resolves the task file's durable mode and plan),
  `orchestrator_script` (what the durable follow-up fields name), `save_slot` (the base
  writes a losing slot's salvage request there; review-code's slot already did),
  `adopt_salvaged` (review-code folds a repaired model's issues back into the round it
  aggregates from its slots), and `reuse_checks_inputs` (whether reused verdicts are held
  to the task file's inputs, below). The base's lane is `BaseReviewLane`, configured by
  the review-plan and review-tasks orchestrators; review-code's is `CodeLane`. The review
  base's own functions reach the shared stages as `FanoutSteps` and `RoundSteps`
  (`fanout_steps`, `round_steps`), built by each lane from the names its module binds
  (`base_round_steps`, `code_round_steps`), so a test that patches one there reaches the
  shared code, and `review_lane` never imports the base.
- **`select_review_models` and `open_round`** start a fan-out. The selection refuses
  `--quick` with `--interactive` (exit 1) and stops at `defaults.models` under `--yes`
  (D6 row 2, exit 3 when nothing resolves). The round's opening checkpoint reads the
  previous attempt's `usage` and sandbox words before its write erases them, records the
  requested specs (the only record a reaggregate inverts a result-file stem with) and the
  round id, drops the previous `progress`, records the `--yes` resolutions (rows 2 and
  16) and publishes the `fanout` step. review-code reads `models_requested` straight back
  (`after_opening`) and warns when the best-effort write lost it.
- **`SlotResult`** is one model slot's outcome: `model_spec`, `success`, `payload`,
  `error`, `source`, `details` and the salvage `failure` kind. `payload` is only ever the
  answer of a completed run; a losing run's text stays in `details` for salvage.
  `slot_results` reads review-plan's record dict and review-code's `(success, issues,
  error)` triple alike; `run_slot` and review-code's slot wrapper return one, and
  `FanoutProgress.record_result` reads it as it reads a record dict.
- **`run_slot`** is one model slot on both lanes: the guard's gate and before-snapshots
  (`utils/slot_guard.py`), the lane's preparation, one deadline per slot
  (`slot_ladder`), the misnamed-result and requested-result baselines taken once, the
  ladder loop (`dispatch_with_ladder`) and the guard's checks over every attempt before
  the lane's ending. `run_single_model` and `run_single_review` stay the entry points and
  return their own shapes. A lane's slot (`BaseSlot`, `CodeSlot`, a `ReviewSlot`)
  supplies the hooks where the lanes differ, and the core reaches the lane's module names
  through `SlotSteps` (`base_slot_steps`, `code_slot_steps`), as the round's stages do.
  The differences `tests/test_review_lane_parity.py` pins (`TestTheSlotCore`):
  review-code's slot retires another scope's result on a refusal, can end
  undispatched as `[REVIEW_UNSUPPORTED]` (the diff set out of reach, or an inline prompt
  over the argv cap), records its provenance in the run's collectors, puts a failure's
  spend on its error string, takes its own terminal tracker rows, writes its salvage
  request itself and leaves an exception past its deadline to the fan-out's wrapper; the
  base's slot records spend and provenance on its record, leaves its terminal rows to
  `run_models_concurrent`, has its salvage request written when the record is saved,
  records an exception past its deadline itself (with the prompt), and, on a direct call
  without an output folder, makes one stdout dispatch without the ladder
  (`SingleShot`) under the same guard.
- **The fan-out's mechanics** are `run_models_concurrent` on both lanes (review-code's
  `run_all_reviews` stays the entry point and still returns its triples): the semaphores,
  the launch spacing, the seeded map, the deadline that harvests finished slots before
  cancelling the rest, and the sweep. A slot that raises is a failed slot with its own
  redacted text (`raised_slot_record`, source `exception`), never a deadline loss; one
  whose own coroutine was cancelled is left to the sweep and the deadline backfill.
- **`close_fanout`** is what follows the fan-out up to grouping: each slot saved, the
  deadline's losses named in `failed_models` and the map, the round checked, the slots
  that ran unsandboxed mirrored for the poller, salvage run in-process under
  `--exec-subagents` (only a file the dispatch names as repaired is adopted), then either
  the exit 3 of a run where no model answered, which still records `models_failed` and
  `models_served`, or the partial results and `models_complete` with the served models
  and usage.
- **`group_findings`** is the fan-out's grouping: the findings of every model that
  answered (review-plan's read from their files, review-code's from its slots) grouped,
  given their stable ids and passed to the lane's `annotate` (review-code's commit
  attribution) before `grouped.json` is written, with the previous verdicts retired in the
  same step (`retire_validation_for_regroup`: the whole round when the grouping changed,
  `validation.json` alone for the same hashes), and the size recorded as
  `grouping_complete` (`total_groups`, `total_suggestions`). Zero findings retire the
  whole previous round and write the empty `grouped.json` and `validation.json`.
- **`materialize_fanout_round`** is the round the fan-out builds over those groups: the
  plan bound at load, the lane's round (`build_round`) materialized under
  `host_views_lock`, stamped with `format_version` and published with `state:
  validation_pending` while the round is this run's, the hand-off's lines (the marker, the
  batches and their efficiency gain, the group counts), then the dispatch and merge under
  `--exec-subagents`, or the host hand-off: the follow-up printed and put on the result
  line, and `stdout_markers`, `validation_tasks_path` and the follow-up's durable fields
  (`reaggregate_mode`, `plan_file`, `replay_flags`) recorded, so `status` names the
  `--reaggregate` that advances the phase.
- **`reaggregate_result_files` and `load_reaggregate_results`** are a reaggregate's
  loading half. The result files are the shared rule's (`result_files`), with any file of
  an unrecorded model set aside; none at all is exit 3 when every requested reviewer
  failed, else exit 0. Each model's findings are read once per logical model (W3),
  non-record entries dropped before the lane's `stamp` (review-code stamps `model` before
  `source_model`, since `grouped_sha256` hashes bytes), a model with a file reconciled as
  completed and the failures left named. No readable result at all clears the recorded
  `validation_counts` (no report rebuild reaches them there) and the lane's own recount
  (review-plan's consolidation recommendation), then ends like the empty-dir case.
- **`bind_grouping`** is the grouping a `--reaggregate` reports. A `grouped.json` that a
  `validation_tasks.json` beside it binds (whatever the flags), or a `validation.json`
  this run will use, is reused rather than rewritten, since the task file hashes its
  bytes; zero findings bind nothing. A result that landed after grouping retires the
  round and regroups (`grouped_coverage_gap`). review-code's commit attribution is its
  `annotate` pass: in memory on a reused grouping, into the file on a regroup.
- **`resolve_validation_round`** is what a `--reaggregate` does with the validation round
  beside its grouping. A round built from truncated context (contract 1) is retired
  whole. A `validation.json` bound to the grouping and to the batch bytes it merged
  (`validation_source_decision`) is reused: its round's snapshots are re-hashed (exit 3
  on a changed one), its verdicts applied and their scope decisions re-derived, and
  nothing is merged again. review-code's verdicts are also held to the inputs its prompts
  embed and no snapshot covers (`reuse_checks_inputs`): the task file is first read
  against `current_inputs` as a resume reads it, so a reviewed line edited since
  validation (`reviewed_code_sha256`), an edited plan or a commit review's changed frozen
  context plan (`context_plan_sha256`) refuses the reaggregate with the resume's exit 3.
  A review pinned to a commit cuts that excerpt at its recorded head, so a working-tree
  edit moves nothing. Verdicts that no task file binds to every one of those inputs (the
  task file is gone, or records fewer) are not reused, and the round goes on as if there
  were no `validation.json`. The base lane's reuse compares no live input: its verdicts
  are bound by the round's snapshots, which an edit to the live plan leaves as they were.
  A `validation.json` that does not bind is retired (renamed) and the batch files
  decide. Zero groups retire the previous round and write `validation.json` as `[]`.
  `--skip-validation` reads and writes nothing, and the report then shows the groups in
  `grouped.json`. A
  `validation_tasks.json` on disk is refused (exit 3) when its inputs or snapshots moved,
  its prompt sidecars are rebound and persisted, and a whole round is merged while a
  partial one is dispatched under `--exec-subagents` or handed to the host with the phase
  reopened. With no task file a new round is built, stamped with `format_version`,
  published, the phase reopened and `progress.subagents` cleared before the dispatch or
  the hand-off. Each lane passes what is its own: the task file's current inputs
  (`current_inputs`) and its round builder (`build_round`).
- **`record_phase_complete` and `complete_reaggregate`** are the completion every ending
  that records the phase runs: the round checked (`claim.check`), the phase marked in
  `state.json`, the phase machine's `state: complete` and the last `progress.step`; a
  reaggregate first reports `[PARTIAL_RESULTS]` for the models still missing
  (`unattended=False`, no `next`). The report before it is the lane's own
  (`aggregate_results` records the HIGH groups and verdict counts on the base;
  review-code's `generate_review_report`, `publish_report` and `finish_code_review`, whose
  `--fail-on-high` gate follows the completion). Every site that publishes a report, and
  review-code's heal of a lost completion, hands its files back through
  `hand_back_report`: `.status.json`'s `output_files` and the result line's `artifacts`
  name the same files (report, HTML report, grouping, `validation.json` when it exists,
  review-code's `issues.json`).

Both lanes also record these the same way:

- the host hand-off that ends a fan-out records `stdout_markers`, `validation_tasks_path`
  and the follow-up in its durable form (`reaggregate_mode`, `plan_file`,
  `replay_flags`, `cli_common.durable_reaggregate_fields`), so `status` names the
  `--reaggregate` that advances the phase;
- a `--detach` launcher writes nothing before its boundary: neither a first run's
  `state.json`, nor a `--force` clear of the phase's completion, nor the copy of the
  reviewed document (review-plan's `backup.md`, review-tasks' `tasks-backup.md`). A launch
  refused at the boundary leaves them as it found them, and the worker runs the same
  pre-flight and writes them.

**A plain re-run over a round on disk** never records the phase complete on a round that
did not finish, and records it complete only through the completion above. The lanes get
there differently, by decision (code health REVIEW-7): review-code's partial-completion
guard refuses with exit 3, naming `--reaggregate` (validation on disk) or `--force` (files
of another round), and when the round's verdicts and a whole report sealed with them are on
disk and only the record is missing, it finishes that completion
(`heal_unrecorded_completion`, which records the HIGH groups and applies `--fail-on-high`,
or refuses with exit 3 when `grouped.json` is not the grouping the sealed verdicts bind);
the base has no guard and runs the review again, which builds a new round. The parity test
holds both to the shared rule (`TestAPlainRerunOverARoundOnDisk`).

What stays different between the lanes, as the lane's own code: the code-review scope,
diff set and pinned ranges; commit templates; the scope-ledger resume rule (review-code's
skip predicate); the `model` then `source_model` stamp order on reaggregated issues
(`grouped_sha256` hashes the file's bytes); the per-issue report with its numbering
footer, and `issues.json`; the stale-implement guard; the partial-completion guard above;
consolidation (review-plan) and the tasks document (review-tasks). Still written once per
lane: the slot functions (`run_single_model`, `run_single_review`) and the report each
lane renders from its groups.

**review-code's run, stage by stage.** `code_review_orchestrator._main` is a sequence of
named stages over one frozen `RunScope`: the base ref, the resolved scope and its review
order, the pinned pair, a commit brief's frozen context plan, the git and project roots,
and the `--files` record. The pre-flight runs on the caller's terminal:
`_guard_phase_state` (the pinned pair resolved before anything is written, then the
completed-phase and partial-completion guards), the model selection,
`_resolve_base_and_scope` and `_resolve_round_policy_and_plan_bytes`.
`_cross_detach_boundary` calls `maybe_detach`, and nothing after it runs in the invoker.
Past it, a front door's landed stage reloads `state`, and `RunScope.refresh` re-derives
the pair, its drift list and the context plan (the base ref and the paths are kept).
`_open_fanout_round` saves the `--files` record, binds the plan and opens the round.
`_write_diff_set`, `_run_reviews_and_save` and `_group_validate_and_report` run inside
`intent_to_add_untracked`, and the last stops its phase ticker only after the terminal
`progress.step`. `--reaggregate` is `_reaggregate_entry`: its own pre-flight, the same
boundary, and the phase-scoped ticker around `reaggregate_from_existing_files`.

## Reaggregation and result identity

Reaggregation reads `{phase_dir}/*.json` and treats each file as a per-model result list. Which
names can be a model's result is decided in one place, `utils/result_files.model_result_paths`
(the rule is `is_model_result_name`; the slot's `snapshot_candidates` uses it too): a non-model JSON
file written into a phase dir (e.g. `report_data.json`, `consolidated_report_data.json`,
`subagent_batch_report.json`) MUST be covered by `NON_MODEL_JSON_MARKERS` or `NON_MODEL_JSON_NAMES`
there, or it gets parsed as suggestions and crashes reaggregate. Both reaggregates, the
partial-completion guard (`leftover_model_results`) and `retire_unselected_results` read that list.
The executor's per-attempt diagnostics escape this by living in a **directory** (`{phase}/failed/`),
which the rule never matches.

Detach's artifacts are invisible to it by construction — `.status.json` because
`glob` skips leading-dot names, `orchestrator-run.log` and its rotated
`orchestrator-run.<run_id>.log` siblings by extension — and a regression test in `tests/test_status_file.py` pins
that rather than assuming it.

**Only the recorded reviewers are read** (fix round 24, live run 10: pi wrote `…deepseek-v4-flash.json`
for `…deepseek-v4_1-flash.json`, the slot recorded it failed, and `--reaggregate` then loaded the
file as a third reviewer by its stem): both reaggregates pass the glob through
`utils/result_files.set_aside_unrecorded_results`, which keeps a file only when its stem is the anchored or pre-anchor stem of
a model `.status.json` records (`models_requested`, else `models_completed`; with
neither, every file is read as before) and renames any other JSON array `<name>.set-aside`
(never deleted), named in a `[RESULT_SET_ASIDE]` line, `.status.json`'s
`result_notes`, `progress.notes` (so `status` and a terminal poll's
`notice` carry it) and report.md's `**Result files:**` line; and a slot whose
requested file is missing reads the one fresh JSON array its provider prefix wrote elsewhere in the
phase dir since the slot started (one baseline per slot, taken before its first attempt and handed
to every retry, fix round 25), when no other model of the run shares that prefix, and renames it to
the requested name (read through one `O_NOFOLLOW` descriptor `fstat`-ed as
the regular file it was, `result_files.read_regular_text`, as the set-aside check reads a stray, fix round 25;
`invoke_with_file_output`'s `run_model_specs`, `[RESULT_FILE_MISNAMED]`, same surfaces).

The requested file has a per-slot baseline too (`llm_client.requested_result_baseline`: device, inode, size and
mtime, taken before the slot's first attempt and handed to every retry): one that was there when the
slot started and has not moved is a previous round's answer, never read as this slot's, and every
later step runs as if it were missing (measured 2026-10-09: a stdout-channel reviewer re-run under
`--force --rerun-all` re-saved the previous round's findings, logged `source: file`).

The moment the slot finds it so, the file is renamed `<stem>.json.stale-earlier-run` (`result_files.set_aside_stale_result`,
`[RESULT_STALE]`, never deleted), outside every reaggregate glob and the salvage loop's
`exists()`, and the review base's salvage recovery adopts only a file this attempt's
`dispatch_salvage_tasks` repaired (Fable review 2026-10-09: left in place, `--exec-subagents`'
salvage loop and `--reaggregate` both read it back as this round's).

## Round identity

**A review round's files say which round wrote them** (`utils/review_round.py`, code health U81).
A fan-out of review-plan, review-tasks or review-code stamps a random `round_id` into
`.status.json` (`review_round.start_round`) right past its detach boundary, before it replaces or
retires any file of the previous round, and its opening `write_status` carries the same id. The
round's `validation_tasks.json` records it as a top-level key (`RoundProvenance.round_id`, written
last, so the file is the one from before round ids plus that key), `validation.json` as
`metadata.round_id`, copied from the task file the merge read (`merge_validation_batches`), and
`report.md` on the line right after `ESCAPED_TEXT_MARKER` (`report_markdown.round_line`, read only
there by `report_markdown.report_round`, never from model text). A `--reaggregate` continues the
round on record and draws no new id. `report_data.json` gets no key, and a zero-group round's
`validation.json` stays the bare `[]` its readers expect, so it records none.

Against the checkpoint a file is the round's own (same id), another round's (`stale`, a different
id) or `legacy` (no id on either side: written before round ids, or by a writer that records none).
The id is a random token, never a time: a clock set back, a copy that kept its dates or a coarse
file system cannot make an earlier round's file pass for this one's. Batch files carry none; a
host's subagents write them as the validator replied, and they follow the task file beside them
(a regroup that changes the grouping retires them, and the merge joins verdicts by group hash).

The stamp writes `state: models_running` in the same update, so an earlier `validation_pending`
never stands beside the new id: an interruption between the stamp and the opening write would
otherwise leave an id-less task file passing the legacy proof below as the new round's.

**Two runs never write one phase dir at once**: every review run holds `{phase}/.lock` from its
entry to its exit, a second is refused there, and a live holder's lock is never broken (runtime.md,
"One writer per phase dir"). The claims below are defence in depth behind that lock, should another
round be stamped regardless (a lock file removed by hand): nothing a run publishes passes for it, and
a result its batch dispatch persists late is dropped (`review_round.results_bound_to`).

**Each invocation writes for its own round, captured once** (`review_round.Claim`):
`start_round` returns the fan-out's, `continue_round` a `--reaggregate`'s (the round on record when
it starts, `None` on a checkpoint without one). Writers take the id from the claim, never from
`.status.json` again, and every publication (the task file, `validation.json`, `grouped.json` with
the retirement it replaces, `report.md`) and completion decision runs under `Claim.held`, which
re-reads the checkpoint under its `.status.lock` mutex and stops the run at exit 3
(`RoundTakenOver`, a `SystemExit`) unless the checkpoint still records the claim's round. It fails
closed: a mutex it cannot take (`checkpoint_lock(required=True)`), or a checkpoint that lost an id
the claim established, stops the run as surely as another round's id. `start_round` establishes a
round only when it reads its stamp back under the mutex, and the fan-out's opening write runs under
`Claim.opening`, which puts the id back if that best-effort write landed without it (the run then
carries on, as on every lost checkpoint write) and never restores a superseded one. A fan-out also
checks once its model results are saved, before it records, groups or retires anything, so its
spend stays on disk for the `--force` the message names. Before this, two runs in one phase dir
each read the other's id off the checkpoint, and a run whose round another run had replaced wrote
that run's id into its own task file. The files recovery reads (the task file, `validation.json`,
`grouped.json`, `report.md`, `report.html`, `report_data.json`) are written to a temp file and
renamed into place (`review_round.replace_file`, `html_report_generator._write_atomic`), so no
reader sees part of one.

**`report.md`'s round line seals it**: `<!-- multi-llm:round <id> validation <digest> report
<digest> -->`, inserted at publication (`review_round.publish_report`). `validation` is the digest
of the exact `validation.json` bytes the report's verdicts came from (`review_round.Verdicts`,
`Claim.verdicts`): the bytes the merge wrote (`Claim.publish_verdicts`) or the reuse branch read
and parsed (`Claim.read_verdicts`), carried to the report and never computed by reading the path
again, which let a report built from one revision certify another of the same round. It is recorded
only when those bytes record the claim's round (a report never claims a round its verdicts did not
come from: with id-less verdicts it gets no line), and absent for a report built without them
(`--skip-validation`, zero groups). `report` is the digest of the text below the line, newlines
read as `\n`. `review_round.report_seal` checks both against the disk.

Two readers decide on it. review-code's partial-completion guard records the phase complete from
the files only when `validation.json` and `report.md` both record the round on record and the
report's seal holds (the whole report, built from this very `validation.json`), and then finishes
the completion that run stopped short of (`heal_unrecorded_completion`): the artifacts handed back,
the HIGH groups recorded from `grouped.json` and `--fail-on-high` applied, so a healed review
never passes a gate that did not run. The groups are counted only from the `grouped.json` the
sealed `validation.json` binds (its `metadata.grouped_sha256`): a grouping that is missing,
unreadable or of other bytes is never counted as zero groups, and the guard records nothing and
sends the run to `--force` (exit 3); a stale task
file or `validation.json`, or a task file with no id that no `validation_pending` checkpoint ties
to the run (`review_round.tasks_file_is_this_rounds`), sends the run to `--force`, which keeps
every reusable result and validates again, and this round's verdicts beside another round's
report, a report cut short, or one built before the verdicts were merged again send it to
`--reaggregate`. Before round ids a forced review stopped after its fan-out started left the
previous round's three files beside a cleared completion, and the next plain review recorded the
phase complete on them. `util resume-state` answers from the same decision (case c with
`ownership=stale`, or `uncertain` for a legacy file; references/resume-detection.md). review-plan
and review-tasks have no such guard: a plain run of either starts a new round.

**A `--reaggregate` reuses or merges only its round's files** (`review_round.refuse_foreign_round`,
one check for all three orchestrators, before the detach boundary and again on the claim it
captures): a task file or `validation.json` recording another round is refused at exit 3 naming the
`--force` re-run, and nothing is changed. Before it, a reaggregate after a `--force` stopped right
past its stamp merged the previous round's files and wrote a report claiming the new round.
Files with no id on either side keep the reaggregate's content checks, as before round ids.

## Malformed result entries

**A per-model result list is not guaranteed to hold objects, and every aggregation site stamps
one.** `issue["model"] = model` / `s["source_model"] = model` are item assignments, so a single non-dict
element raises `TypeError` and loses every OTHER model's findings with it. Measured
2026-09-20: the stdout fallback scraped the `[0]` out of a clean reviewer's prose
sentence "`unattended.ROWS[0].site` correctly left as-is" instead of the `[]` it
ended on, `validate_code_review_issues` warned `'[0]: expected object, got int'` and nothing acted on it, and a
five-model `review-code` phase died in aggregation with the other four models' eleven
findings unwritten — `--reaggregate`, the documented recovery, died at the same statement one
function over.

Both halves are now closed: every stamping site (`code_review_orchestrator`'s three,
`review_orchestrator_base`'s two, `suggestion_processor.extract_suggestions_from_review`) filters through `schema_validator.drop_non_record_entries`, which
warns on stderr naming the model and the element and lets the run continue; and clause 5 of
`json_extractor.extract_json_from_text` ranks candidates as records → empty document → array of bare scalars, so a
bracketed index in prose can never outrank the `[]` a reviewer that found nothing
really wrote. Candidates parse with `strict=False` (`_loads_reply`, 2026-10-05): a
raw control character inside a string no longer fails the document and serves a fragment from inside
it.

**Saving a model's result never ends the round's save loop.** On the plan and tasks lane,
`validate_suggestion` treats an entry that is not an object, or a `type` that is not a string, as an
invalid suggestion: `save_model_result` drops it with a warning and saves the rest, `[]` when nothing
is left. `extract_json_array` unwraps `{"result": ...}` only when the value is text (a CLI envelope);
any other value is searched like an array under any other key. Both lanes write through
`review_orchestrator_base.write_model_result`. It refuses a result nested deeper than
`MAX_RESULT_DEPTH` (100) levels, because the round re-encodes the records later with `json.dump`'s
pure-Python encoder (`grouped.json`, the issues file), which stops near the recursion limit. It also
encodes before opening `{model}.json`, so an encoder failure leaves no partial file. Either way the
model fails, nothing is salvaged (the reply parsed), and `error_{model}.log` names the encoder.
A `{model}.json` already in the phase dir (a file-capable provider writes it before the reply is
saved) is renamed `{model}.json.rejected` (`result_files.set_aside_rejected_result`, never
deleted, noted on the `**Result files:**` line), outside every `*.json` reader, or `--reaggregate`
would read it back, move the model to `models_completed` and stop writing `grouped.json`. Both
reaggregates refuse such a file too (`resolve_result_files_by_model`: deeper than
`MAX_RESULT_DEPTH`, or than the decoder's stack), and set it aside the same way.
review-code aggregates from memory, so its loop also marks such a model failed in `results`. A
result file nested deeper than the decoder's stack is read as one that does not parse
(`json_extractor.parse_file_content`).

## Report payload

`html_report_generator.generate_html_report` is the view, its embedding and its sidecar. The view is
`report_view.build_report_view(ReportInputs)`: `ReportInputs` holds the function's parameters one
field each, and `ReportView` (a `TypedDict`) lists the payload keys the templates read, marking the
conditional ones. A conditional key is absent, never empty, when it does not apply (`servedModels`,
`usageTotal`, `requestedModels`, `commitRange`, ...), so a default run's `report_data.json` gains no
key. The builders take their inputs as arguments: `build_file_contexts` reads code only through the
reader it is handed, and `pinned_file_reader` never falls back to the disk on a pinned review (a path
missing from the `head` snapshots has no file context). The consolidated page builds its badges with
the same `build_model_metadata` (fallback, sandbox rung and usage labels; `reaggregate_consolidation_mode`
passes `recorded_served_models`) and its title with the same `plan_title`. Every page loads its
template through `load_report_template` and embeds through `embed_report_data`.

**Each group is stored once.** The `fileView`, `sectionView` and `taskView` buckets hold
`groupIndexes` (positions in `groups`), and `globalSuggestions` is `{"groupIndexes": [...]}`
(`report_view.compact_views`); the PR template's `hydrateViews` rebuilds `suggestions` from them
before anything renders, and `report_view.hydrate_views` is its Python twin, used by readers and
tests. The views then share the group objects, so a decision or guard stamp the overlay puts on
`groups` shows in every PR-view block (the per-view copies never carried it). A bucket that still
holds `suggestions` (a sidecar written before 2026-11-08) is left as it is by both, and a
regeneration rewrites it as it found it. Sidecars are compact JSON (`html_report_generator.sidecar_json`).
On a real 262 KB plan's review the page went from 6.5 MB to 1.7 MB and the sidecar from 6.2 MB to
1.3 MB. Tests: `tests/test_html_report_view.py` (runs `hydrateViews` under Node and the PR page in
headless Chrome; each skips without its tool).

**A full section is a heading's section.** `extract_section_contexts(..., full_section=True)` matches
only headings (`_heading_levels`: one to six `#` and a space, outside fenced code, the same rule as
`extract_plan_section_order`) and ends at the next heading of the same or a higher level. A reference
that matches only body text (`status`, `next step`) gets no `fullSectionContexts` entry: its block
still renders its suggestions, with the hover context around the words. Matching body text there gave
the section no level, so it ran to the end of the plan.

## Report regeneration

`generate_html_report` persists `{phase_dir}/report_data.json` (the assembled report data) so the apply
phase can overlay human/judge decisions onto it and re-embed via `html_report_generator.py regenerate-decisions` — no
reconstruction of runtime inputs (diff_data, base_ref, models). `regenerate-decisions --apply-phase` accepts
either an apply phase's mode word or its `phase_name` (`html_report_generator._APPLY_PHASE_STATE_KEYS` normalizes
`apply-code-fixes` → `apply-fixes`, the one apply phase where they differ), matching
`state.json`'s `human_decisions_{phase_name}` key the orchestrator actually writes.

`regenerate_report_with_human_decisions` returns `(report_path, matched)`, not just the path (F2, 2026-09-16): a report
whose groups carry no `groupHash` (the code-review main path used to hand the validators
and the merge step an un-stamped copy of `grouped` instead of the
`stamp_stable_ids`-stamped `grouped_list`) used to overlay decisions onto nothing and
report success anyway; both callers now print matched-vs-read and warn when they differ.

The consolidated page has its own sidecar, `consolidated_report_data.json`, written by
`consolidation.generate_consolidated_html`; the same entrypoint (and the in-process twin, through
`regenerate_report_with_human_decisions(include_consolidated=True)`) re-embeds it with decisions
joined to each umbrella's **underlying** groups by `groupHash`
(`regenerate_consolidated_with_human_decisions`). A consolidated page with no sidecar (written before
it existed) is a reported no-op, never reconstructed. Both payloads embed through
`html_report_generator.script_json`.

`output_handler.PHASE_DIRECTORIES` and `utils/prior_changes.PHASE_ALIASES` carry the same `apply-code-fixes` →
`apply-fixes` mapping so
`status --phase`/`--abort`/`--sweep` and `prior_changes.jsonl`'s
`append`/`render` resolve either spelling to the one phase (F1/F6,
2026-09-16) — `full_workflow_orchestrator.WRITE_PHASE_DIRS` was previously the only place that held it.

## Scope tags in reports

**Every report shows scope tags through `utils/scope_report.py`** (plan
`scope-policy-and-report-tags` §6), over the foundation's vocabulary in `utils/scope_policy.py`:
each finding's own tag is the reviewer's provisional classification, each group's tag is
`scope_policy.authority` (``Scope unclear`` when missing, unbound or stale — never a confident badge
for an edited proposal), an umbrella's is derived from its members (`combined_relation`), and under an
effective `preserve` a group the gate would not pass shows `Deferred: scope preservation requested`
whatever its validity. Counts are per distinct group. The round's policy comes from `state.json`'s
`scope_policy` record beside the phase dir (or a passed `scope_state`); a legacy round reads `expand`.

In `report.md` the scope lines sit between the checkboxes and the `**Validation:**`/`**Importance:**`
line, so `report_parser`'s description boundary is unchanged and an untouched report shows no
edit; model text goes through `scope_report.md_text` (flattened, Markdown punctuation escaped), so a
reason cannot forge a header, a checkbox, a metadata line or the issue-digest footer.
`find_edited_descriptions` also matches the header's display hash (it used to compare only the
16-character `suggestion_hash`, so a v2 report's edits were dropped).

The HTML payloads persist each `scope` entry (classification, authority state, baseline id, proposal
digest, selected remedy, decision) and a `scopePolicy` block; regeneration recomputes only the
decision for the recorded effective policy and deferrals (`scope_report.replay_decision`, the one
gate), keeping the classification, and refreshes a report whose policy moved even with no decision
recorded (`scope_regeneration_needed`). The three pages share one JS block between
`BEGIN SCOPE TAGS`/`END SCOPE TAGS`, in `templates/shared/report_core.js`
([shared report code](../development.md#shared-report-code)); an edited description in the page marks its tags stale, and
scope- or policy-like keys in a stored or imported selection are dropped (`report_parser` drops them
from `user_selections.json` too). Tests: `tests/test_scope_report_tags.py` (runs the templates under
Node; skips without it).

## Issue numbers and group decisions

**`code-review`'s `report.md` numbers ISSUES; every checkbox on it acts on a
GROUP** (F1/F5, 2026-09-16). `generate_review_report` is the one report generator that iterates raw
issues, so a consensus finding is rendered once per model — and `apply_code_fixes_orchestrator` used to index
`self.groups` with those `### N.` numbers: measured on a 12-issue / 9-group
review, 11 of 12 `- [x] Skip` / `Mark valid` / `Mark invalid` /
`Let the agent decide` marks landed on a different finding and the last three fell off the end of
the group list unreported — the user suppresses one fix and silently suppresses another while the
one they meant still runs.

Both sites (`parse_skips_from_report`, `apply_group_validation_overrides`) now resolve through
`_group_hash_for_issue` → the `issue_to_group_map` `post_load_hook` already builds for the
edited-description path, match `self.merged` by `group_hash` rather than by
position (so `--resume`'s filtered list cannot re-key them either), warn instead of
dropping a number the map cannot place, and take `invalid > needs-human-decision > valid > judge_decide` when two members of one
group are marked differently. Without issues.json the legacy 1:1 reading survives, which is correct
only when no two models agreed.

The report says so rather than leaving it to be inferred: a `**Groups:** N` header line
beside `**Total Issues:**`, `(of N groups)` on the validation line, `_Same finding as #7, #11 (3 models agree); marking any marks all._`
on every member of a multi-member group, and `(group verdict)` on the badge those members share
— counting the rendered badges used to give 10 Valid / 1 Invalid / 1 Needs Review over a 7/1/1
validation line and a 12-issue header, three sets of numbers none of which reconciled without
`grouped.json`.

The numbering itself does not move: `build_issue_to_group_map` and the parser's
`### N.`-plus-next-line contract both depend on it — and that contract is now enforced
from both ends (F6, 2026-09-16): `_format_issue` flattens newlines out of the title (a
`\n` in one pushed the Skip box off the header line and lost the tick silently),
and `report_parser` accepts a header only at column 0 after a blank line or the
`---` rule, so a rendered description containing its own `### 2.`
plus a ticked box cannot forge a skip on an issue the user never touched.

## Report grammar

**`utils/report_markdown.py` owns the structure `report.md` and `consolidated-report.md` carry, for
the writers and for `report_parser` alike** (REVIEW-4, 2026-11-08). A ticked box is a decision the
apply phases act on, and the same file holds model text that can quote a hostile plan or repository,
so model text must never read as structure:

- **Writers** emit headers (`group_header`, `suggestion_header`, `issue_header`,
  `consolidated_header`) and boxes (`checkbox_line`, labels in `CHECKBOX_LABELS`) through the module.
  A field on a header or `**Field:**` line goes through `inline_text` (line breaks become spaces), a
  text block (description, validation reason, failure message, a "Needs Human Review" reason)
  through `neutralize_model_text`, which puts a backslash before a line's `##` or `- [` marker. The
  line renders as the same characters and no parser reads it. `report.md` (plan, task and code
  review) starts with `ESCAPED_TEXT_MARKER`, an HTML comment that says so.
- **Parsers** read a header only at column 0 and a ticked box only at a line start, one line at a
  time (`is_checked`, `first_checked` in precedence order). The old patterns matched mid-line and
  across line breaks (`-\s*\[`), so a one-line description could tick its own group.
- **One structural pass for a plan or task report.** `plan_sections` finds every `## G{N}` and
  `### G{N}S{M}` header, refused ones included, and gives each the region up to the next one (or an
  earlier `---` or `##` line). All four decision readers (both Skip readers, both override readers)
  take their boxes from it, so a box counts only for the header right above it: a group's own boxes
  stop at its first suggestion (a suggestion's tick is not its group's, as `user_selections.json`
  keeps `G1S1` keys out of the group overrides), a group's last suggestion stops at the group's
  closing `---` (the next group's tick is not its), and a refused header keeps its box, which every
  reader ignores. Per-reader blocks let a header one reader refused feed another (Codex review of
  REVIEW-4: a forged `## G2:` inside a suggestion gave that suggestion a `valid`).
- **Descriptions read back** go through `restore_model_text`, the exact inverse of the escape, so an
  untouched report shows no edit and a user's edit reaches the apply prompt without backslashes;
  only in a report that starts with the marker (`has_escaped_text`). In an older report a backslash
  before `##` or `- [` is the model's own and stays. The writers before the marker put the title on
  the first line and model text never comes first, so no report can claim it. `_is_block_cut` checks
  both renderings, each read back as the parser reads it.
- **Mixed headers are refused.** Every group and suggestion header carries a bracket hash or none
  does, so an unhashed `## G{N}:` or `### G{N}S{M}:` in a report that also has hashed ones is not
  the writer's: `hashed_headers` tells the parsers to ignore it (with a warning) instead of resolving
  it through `grouped.json`'s index map. This also covers reports written before the escape.
- The issue-digest footer is read from the last line that holds it (`issue_digest`).

Tests: `tests/test_report_markdown.py` drives each lane's real writer with forged headers and boxes
(after a blank line, mid-line, after a bare CR, indented, split over two lines) and the real parsers,
puts each kind of refused header in each region of an older report, and reads older reports'
literal backslashes back unchanged.

## Scope tags and the review round

**One review round has one scope policy and one scope baseline, given to its reviewers and its
validators alike** (plans/scope-policy-and-report-tags.md §2–§4; `utils/scope_policy.py` owns the
vocabulary and prompt blocks, `utils/scope_round.py` the round). The policy is resolved before
fan-out: the explicit `--scope-policy`, else — for a same-round resume, `--reaggregate` or
revalidation — the round's recorded `review_policy`, else `expand`. An explicit policy that differs
from the one an unfinished round's reviewers were asked under exits 3 with
`[SCOPE_POLICY_CONFLICT]` (the new-round control is `--force --rerun-all`), and so does one typed
for a `--force` round that would reuse reviewer results requested under another policy; with none
typed, that round keeps the reused results' policy (source `review`, one `Scope policy notice:`
line). `state.json`'s `scope_policy[<review phase>]` records the round (`record_review_round` for a
new round only, `update_round_baseline` — first binding wins — otherwise), its baseline built by
host hash from the inputs the round already binds: the plan's `validation-snapshot.md` bytes,
review-tasks' `tasks-backup.md`, review-code's task list and range endpoints, a commit review's
brief, pinned endpoints, focus and frozen context plan.

The conflict check includes every cached reviewer in the round, even when the invocation
selects only different reviewers. An authorized new round that changes policy retires all
old-policy result files under the existing `.stale-earlier-run` convention, after the detach
boundary and before recording the new policy or dispatching reviewers. A failed retirement
exits 3; reselecting an earlier reviewer cannot reuse an answer from another policy.

All four reviewer templates carry a `{scope_policy_guidance}` slot, rendered per invocation
(`reviewer_guidance`) — the template cache never holds policy text — and the three output fields
`scope_relation`/`scope_reason`/`scope_reference`. Under `expand` the slot is the informational
annotation block alone and every existing checklist stands; only `preserve` renders the
suppression block. Every aggregation site normalizes a member's tag through
`schema_validator.drop_non_record_entries` (malformed reads `unclear`, never drops the finding),
both reviewer schemas accept the fields optional and untyped, and a salvage request lists them as
`expected_schema.optional_fields` beside the unchanged required `fields`.

**Scope authority moves only by stable hash.** The validation merge stamps a verdict
(`scope_round.stamp` → `scope_policy.stamp_validation_record`) only when it joined its group by
hash, against the host's own copy of that group and the baseline the task file's `scope` block
names; an index-strategy verdict, a placeholder and a hash-less legacy record keep their validity
and lose their stamps. `apply_validation_to_groups` attaches `scope_verdict` through
`attach_verdict(..., hash_matched=)` — never through the `group_index` fallback — and
`save_validation_results` carries the stamped keys plus `metadata.scope_policy`,
`scope_baseline_id` and `scope_round_baseline_id`. A validators' baseline that moved from the
round's (`[SCOPE_BASELINE_MOVED]`) binds its verdicts to the moved text, which reads unclear for the
round. Under an effective `preserve`, `merge_validation_batches` derives the validation-stage
decision (`scope_policy_decision` on every group validation judged and its record — a
`validation_failed` group gets none, an `invalid` one is shown but not counted — counts in
`metadata.scope_policy_decisions` and `.status.json`, one `[SCOPE_POLICY]` line); validity never
changes. Under `expand` nothing is decided.
