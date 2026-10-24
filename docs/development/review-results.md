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
- [Reaggregation and result identity](#reaggregation-and-result-identity)
- [Malformed result entries](#malformed-result-entries)
- [Report regeneration](#report-regeneration)
- [Issue numbers and group decisions](#issue-numbers-and-group-decisions)

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
source → tests → fixtures → generated with the single classifier `git_utils.classify_review_paths`
(`code_review.collapse_paths` in `providers.yaml` configures the fixture/generated globs), and
before dispatch writes `code-review/review-scope.md` plus one whole, uncapped `code-review/diff/<n>-<slug>.diff` per
source/test file (`write_review_diff`; git failures degrade coverage to `partial`
with `[DIFF_FAILED]`/`[DIFF_PARTIAL]`, never the phase).

`.review-scope.json` records which scope id each result reviewed, so a resume re-runs a result
from another scope (`[SCOPE_CHANGED]`, old file kept as `.stale-<id>`). The prompt's
`{diff_access}` block is rendered by `llm_client.render_diff_access` from two
`generate`-profile declarations on the adapter, never from a provider name:
`runs_shell_in_generate` (`True`/`False`/`None`,
measured sources only, pinned by `TestShellDeclarationMatchesTheEvidence`) and `generate_read_scope`
(`none`/`cwd`/`any`).

A lane that cannot open the phase dir gets plan, manifest and diffs inline under a 96 KB cap (argv
`MAX_ARG_STRLEN`), or is recorded failed with `[REVIEW_UNSUPPORTED]` without being dispatched.
Every Lane A review template also carries `{working_limits}` (`render_working_limits`: the
attempt's real minutes, no suites/builds, no unobserved test claims), which names no shell command
because `{diff_access}` alone decides that.

## Stable group IDs

Group IDs are 16-char hex content hashes — use `state_manager.py` hashing, don't generate random
IDs

## Output paths

All output paths are plan-relative for portability

## Validation statuses

Validation statuses: `valid`, `invalid`, `needs-human-decision`,
`validation_failed`

## Importance levels

Importance levels: HIGH > MEDIUM > LOW

## Grouping and section references

**Grouping tracks meaning, not word count (2026-09-11).** `suggestion_processor.compute_similarity` is
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

## Reaggregation and result identity

Reaggregation globs `{phase_dir}/*.json` and treats each file as a per-model result list.
Non-model JSON files written into a phase dir (e.g. `report_data.json`, `subagent_batch_report.json`,
`state`, `.status`) MUST be added to every `exclude_patterns` list
(`review_orchestrator_base.py` — two sites — and `code_review_orchestrator.py`) or they get parsed as suggestions
and crash reaggregate. The executor's per-attempt diagnostics escape this by living in a
**directory** (`{phase}/failed/`), which the glob never matches.

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

`output_handler.PHASE_DIRECTORIES` and `utils/prior_changes.PHASE_ALIASES` carry the same `apply-code-fixes` →
`apply-fixes` mapping so
`status --phase`/`--abort`/`--sweep` and `prior_changes.jsonl`'s
`append`/`render` resolve either spelling to the one phase (F1/F6,
2026-09-16) — `full_workflow_orchestrator.WRITE_PHASE_DIRS` was previously the only place that held it.

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
