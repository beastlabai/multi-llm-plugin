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
as the base report does. Its `--reaggregate` resolves a validation round the way the base
lane does: stale prompt sidecars are rebound and persisted, a whole round of
`validation_batch_<n>.json` is merged, a partial one is dispatched under `--exec-subagents`
or answered `Only n/m batch files found.` with the phase left open. The merge reads a
missing batch as `validation_failed`, so it is never handed a partial round. On neither
lane does a broad `except` surround the merge or the dispatch: one that raises ends the
command with the phase open, never completed with no `validation.json`.

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
Non-model JSON files written into a phase dir (e.g. `report_data.json`, `consolidated_report_data.json`,
`subagent_batch_report.json`, `state`, `.status`) MUST be added to every `exclude_patterns` list
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
recorded (`scope_regeneration_needed`). The three templates carry one identical JS block between
`BEGIN SCOPE TAGS`/`END SCOPE TAGS`; an edited description in the page marks its tags stale, and
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
