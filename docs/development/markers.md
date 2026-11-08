<!-- GENERATED FILE — DO NOT EDIT.
     Rendered by skills/multi-llm/tests/tools/render_markers.py from the skill's
     Python sources, the hand-written packaging/ sources and the skill's SKILL.md,
     instructions/ and references/. Regenerate with
     `uv run --project skills/multi-llm -- python skills/multi-llm/tests/tools/render_markers.py`;
     `tests/test_markers_registry.py` fails when this file differs from a fresh render. -->

# Bracket markers

The orchestrators print lines that start with a bracket marker such as `[BATCH_DONE]`.
Hosts and the opencode and codex servers parse some of them, and the instructions tell
the driving agent what to do on others, so **a marker is a contract: never rename one**
without changing every file in its row.

Four tests keep this page honest (`tests/test_markers_registry.py`):

- the page matches a fresh render of the sources;
- every marker `SKILL.md`, `instructions/` or `references/` cites is emitted by a
  source;
- every emitted marker is either cited by one of those documents or listed as
  not-for-the-host in `NOT_FOR_HOST` in the generator, with a reason. The page cannot
  tell a line the host must act on from one it only relays, so every marker counts as
  host-facing until it is listed;
- no printed marker is built from upper-case text and a run-time value
  (`f"[BATCH_{kind}]"`): write each one whole, as a literal or a constant, so the scan
  can list it.

**What counts as emitted.** The sources are the skill's non-test `.py` files and the
hand-written `packaging/` sources, never the generated `packaging/*/multi-llm/` copies.
A marker is emitted where a string the code prints, writes, appends, returns or stores
opens a line with it. A string the code only tests for the marker (`in`, `startswith`,
a regex) is a parser, and a sentence that names a marker mid-line mentions it; neither
emits it. Strings are followed through f-strings, `+`, constants, imports, local names
and a helper's parameters (`sweep_live_clis("DEADLINE_SWEEP")`). The generator's
docstring has the full rules.

**How to read a row.** *Emitted by* names a file (under `skills/multi-llm/`, or
`packaging/…` from the repository root) and, where the marker's text comes from a
module constant or a helper's argument, the constant or `helper()` to grep for; there
are no line numbers because they would make the page stale after any edit above them.
*Consumed by* is where an instruction, reference or `SKILL.md` cites the marker; where
none does, it shows the reason key from the legend below. *Meaning* is the first
sentence of the comment or docstring on the constant, and `—` where the marker is an
inline literal or has none. Cells list a few entries and then `+N more`; grep for the
rest. A marker that only a comment, a docstring or a parser mentions is not emitted
and is not listed.

To add a marker: emit it, then either cite it in the instruction that handles it or add
it to `NOT_FOR_HOST` with a reason, then re-render this page.

## Counts

- Emitted markers: 181
- Cited by an instruction, reference or `SKILL.md`: 109
- Declared not-for-the-host: 72
- Neither (must be zero): 0

## Registry

| Marker | Emitted by | Consumed by | Meaning |
| --- | --- | --- | --- |
| `[ABORT_SWEEP]` | `utils/detach.py` | not for the host: runner-diagnostic | — |
| `[ACCEPTED]` | `ask_orchestrator.py` | not for the host: phase-progress | — |
| `[ALREADY_COMPLETE]` | `utils/improve_steps.py` `ALREADY_COMPLETE_MARKER`<br>`utils/improve_workflow.py` | `instructions/improve-code.md`<br>`instructions/improve-commits.md`<br>`instructions/improve-plan.md` | — |
| `[AUTO_RESOLVED]` | `utils/unattended.py` `RESOLVED_MARKER` | `instructions/full-workflow.md`<br>`instructions/implement.md`<br>`instructions/improve-code.md`<br>`instructions/improve-commits.md`<br>`instructions/improve-plan.md`<br>`instructions/review-code.md`<br>+4 more | Prefix of the one stdout line every auto-resolution prints |
| `[BATCH_CHECK_ACCEPTED]` | `utils/write_batch_scheduler.py` `BATCH_CHECK_ACCEPTED_MARKER` | `instructions/implement.md` | The line a code task prints when its own check reported only findings the user accepted for a check (`--human-task <check>=done`): both… |
| `[BATCH_CHECK_FAILED]` | `utils/write_batch_scheduler.py` `BATCH_CHECK_MARKER` | `instructions/apply-code-fixes.md`<br>`instructions/implement.md` | The line a code phase prints per batch whose own check failed after its edits (`WriteBatchScheduler._batch_check_lines`, `util batch-done… |
| `[BATCH_CHECK_REPEATED]` | `utils/write_batch_scheduler.py` `BATCH_CHECK_REPEATED_MARKER` | `instructions/implement.md` | The line a code task prints when its own check reported only the same findings a check task of the run reported (fix round 28): both legs… |
| `[BATCH_DECLINED]` | `utils/batch_done.py` `BATCH_DECLINED_MARKER` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md` | Fix round 18: one line per item an apply batch declined with a reason (`--declined`), and per item the user chose to skip when the batch… |
| `[BATCH_DONE]` | `utils/batch_done.py` `MARKER` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md`<br>`instructions/implement.md` | The one stdout line a successful record prints |
| `[BATCH_SKIPPED]` | `utils/batch_done.py` `BATCH_SKIPPED_MARKER` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md` | — |
| `[CHECK_BLOCKED]` | `utils/batch_done.py` `CHECK_BLOCKED_MARKER` | `instructions/implement.md` | The line naming the tasks a check with findings holds back (stdout) |
| `[CHECK_FINDINGS]` | `utils/write_batch_scheduler.py` `CHECK_FINDINGS_MARKER` | not for the host: unnamed-handler | The line an implement run prints per check task that reported problems (`WriteBatchScheduler._check_findings_lines`, racing re-run 4) |
| `[CHECK_NOT_RECORDED]` | `utils/batch_done.py` `CHECK_NOT_RECORDED_MARKER` | `instructions/implement.md` | — |
| `[CHECK_NOT_RERUN]` | `implement_orchestrator.py` | `instructions/implement.md` | — |
| `[CODEX_HOME_CHECK]` | `utils/providers/base.py` `CODEX_HOME_CHECK_MARKER` | not for the host: runner-diagnostic | — |
| `[COMMIT_BRIEF]` | `utils/commit_front_door.py` `COMMIT_BRIEF`<br>`utils/improve_steps.py` | `SKILL.md`<br>`instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | Markers this core prints (the commit block's own are commit_range's) |
| `[COMMIT_DECISIONS_DROPPED]` | `utils/commit_front_door.py` `COMMIT_DECISIONS_DROPPED` | `instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[COMMIT_FIXES]` | `utils/commit_range.py` `COMMIT_FIXES` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md` | — |
| `[COMMIT_FIXES_END]` | `utils/commit_range.py` `COMMIT_FIXES_END` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md` | — |
| `[COMMIT_FIXES_STALE]` | `utils/write_batch_scheduler.py`<br>`utils/write_prompt.py` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md` | — |
| `[COMMIT_FLAG_CONFLICT]` | `multi_llm.py` `record_error()`<br>`utils/improve_steps.py` `COMMIT_FLAG_CONFLICT` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | The markers of the two refusals only a commit brief meets (a third, `[RANGE_UNRESOLVED]`, is `commit_range`'s own) |
| `[COMMIT_PROTECTED_WRITE]` | `utils/batch_done.py` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md` | — |
| `[COMMIT_RANGE]` | `utils/commit_range.py` `COMMIT_RANGE` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[COMMIT_RANGE_END]` | `utils/commit_range.py` `COMMIT_RANGE_END` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[COMMIT_RANGE_PINNED]` | `utils/commit_front_door.py` `COMMIT_RANGE_PINNED` | `instructions/improve-commits.md` | — |
| `[COMMIT_REMAINING_FLAGS]` | `utils/commit_front_door.py` `COMMIT_REMAINING_FLAGS` | `instructions/improve-commits.md` | — |
| `[COMMIT_REVIEWS_NOT_IGNORED]` | `utils/commit_paths.py` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[COMMIT_REVIEW_ARCHIVED]` | `utils/commit_front_door.py` `COMMIT_REVIEW_ARCHIVED` | `instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[COMMIT_REVIEW_KEPT]` | `utils/commit_front_door.py` `COMMIT_REVIEW_KEPT` | `instructions/apply-commit-fixes.md` | — |
| `[COMMIT_REVIEW_NOT_FOUND]` | `utils/commit_apply.py` `NOT_FOUND_MARKER`<br>`utils/commit_front_door.py` `COMMIT_REVIEW_NOT_FOUND`<br>`utils/improve_steps.py` `COMMIT_REVIEW_NOT_FOUND`<br>`utils/run_status.py` | `instructions/apply-commit-fixes.md`<br>`instructions/review-commits.md` | The exit-3 refusal when no completed review fits |
| `[COMMIT_REVIEW_RESTARTED]` | `utils/commit_front_door.py` `COMMIT_REVIEW_RESTARTED` | `instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[COMMIT_REVIEW_SELECTED]` | `utils/commit_apply.py` `SELECTED_MARKER` | `instructions/apply-commit-fixes.md` | The commit block's line naming the review a no-argument apply picked |
| `[CONSISTENCY_WARNING]` | `utils/apply_orchestrator_base.py` | not for the host: run-notice | — |
| `[CONSOLIDATION_BATCHES_PENDING]` | `review_plan_orchestrator.py` | `instructions/full-workflow.md`<br>`instructions/improve-plan.md`<br>`instructions/review-plan.md`<br>`references/consolidation.md` | — |
| `[CONSOLIDATION_PENDING]` | `review_plan_orchestrator.py` | `instructions/full-workflow.md`<br>`instructions/improve-plan.md`<br>`instructions/review-plan.md`<br>`references/consolidation.md` | — |
| `[CONSOLIDATION_RECOMMENDED]` | `utils/review_orchestrator_base.py` | `instructions/full-workflow.md`<br>`instructions/review-plan.md`<br>`references/consolidation.md` | — |
| `[CONSOLIDATION_SKIPPED]` | `review_plan_orchestrator.py`<br>`utils/consolidation.py` | not for the host: phase-progress | — |
| `[CONTEXT_PLAN_CHANGED]` | `utils/commit_front_door.py` `CONTEXT_PLAN_CHANGED` | `instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[CONTEXT_PLAN_SNAPSHOT_CHANGED]` | `code_review_orchestrator.py`<br>`utils/commit_front_door.py` | not for the host: run-notice | — |
| `[DEADLINE]` | `ask_orchestrator.py` | not for the host: phase-progress | — |
| `[DEADLINE_SWEEP]` | `ask_orchestrator.py` `sweep_live_clis()` | not for the host: runner-diagnostic | — |
| `[DEPRECATED]` | `utils/apply_orchestrator_base.py` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md` | — |
| `[DETACHED_RUN]` | `utils/detach.py` | not for the host: unnamed-handler | — |
| `[DIFF_FAILED]` | `utils/git_utils.py` | `instructions/review-code.md` | — |
| `[DIFF_PARTIAL]` | `utils/git_utils.py` | `instructions/review-code.md` | — |
| `[DRY_RUN]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[DRY_RUN_COMPLETE]` | `review_plan_orchestrator.py` | not for the host: phase-progress | — |
| `[ETA]` | `utils/metrics.py` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md`<br>`instructions/implement.md`<br>`instructions/review-code.md`<br>`instructions/review-tasks.md`<br>+2 more | — |
| `[FINDING_RESTATE]` | `utils/batch_done.py` `FINDING_RESTATE_MARKER` | `instructions/implement.md`<br>`references/own-check.md` | The line `util batch-done` prints for a finding that holds a finding the user accepted, word for word, with words added (fix round 28) |
| `[GIT_CONFIG_CHANGED]` | `utils/git_utils.py` `GIT_CONFIG_CHANGED_MARKER` | not for the host: run-notice | The name a changed-metadata line carries |
| `[HEAD_REF_MISMATCH]` | `code_review_orchestrator.py` `HEAD_REF_MISMATCH` | not for the host: run-notice | The refusal for a typed pair that is not the pair on record |
| `[HIGH_PRIORITY_ISSUES]` | `code_review_orchestrator.py` | not for the host: phase-progress | — |
| `[HUMAN_DECISION_JUDGING_PENDING]` | `utils/apply_orchestrator_base.py` | not for the host: phase-progress | — |
| `[HUMAN_TASK_ANSWERED]` | `implement_orchestrator.py` | `instructions/implement.md` | — |
| `[IMPROVE_RERUN]` | `utils/improve_steps.py` `RERUN_MARKER` | `SKILL.md`<br>`instructions/improve-code.md`<br>`instructions/improve-commits.md`<br>`instructions/improve-plan.md`<br>`references/host-capabilities.md` | — |
| `[IMPROVE_SCOPE_CHANGED]` | `utils/improve_workflow.py` | `instructions/improve-code.md`<br>`instructions/improve-commits.md` | — |
| `[IMPROVE_STEP]` | `utils/improve_steps.py` `STEP_MARKER` | `instructions/improve-code.md`<br>`instructions/improve-commits.md`<br>`instructions/improve-plan.md` | — |
| `[IMPROVE_STEP_FLAGS]` | `utils/improve_steps.py` `FLAGS_MARKER` | `instructions/improve-code.md`<br>`instructions/improve-commits.md`<br>`instructions/improve-plan.md` | — |
| `[IMPROVE_STEP_FRESH]` | `utils/improve_steps.py` `FRESH_MARKER` | `instructions/improve-code.md`<br>`instructions/improve-commits.md`<br>`instructions/improve-plan.md` | — |
| `[INSTALL_CHECK]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[INSTALL_CHECK_FAILED]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[INTERRUPT_SWEEP]` | `multi_llm.py` `sweep_live_clis()`<br>`utils/process_control.py` `sweep_live_clis()` | not for the host: runner-diagnostic | — |
| `[JAIL_DENIED]` | `utils/providers/base.py` `JAIL_DENIED_MARKER` | not for the host: runner-diagnostic | — |
| `[LEGACY_INSTRUCTIONS]` | `multi_llm.py` | not for the host: operator-cli | — |
| `[MANUAL_GATE]` | `implement_orchestrator.py` | `instructions/implement.md` | — |
| `[MANUAL_TASK]` | `utils/next_phase.py` `MANUAL_TASK_MARKER` | `instructions/implement.md`<br>`references/summary-format.md` | One line per deferred implement task under `[WORKFLOW_COMPLETE]`: the id, then the title when `state.json` recorded one |
| `[MANUAL_TASK_COMMAND]` | `utils/next_phase.py` `MANUAL_TASK_COMMAND_MARKER` | `instructions/implement.md`<br>`references/summary-format.md` | The line after `[MANUAL_TASK_NOTE]`: the command that records the manual tasks done, alone (fix round 22, live run 9: the note carried it… |
| `[MANUAL_TASK_NOTE]` | `implement_orchestrator.py`<br>`utils/next_phase.py` `MANUAL_TASK_NOTE_MARKER` | `instructions/full-workflow.md`<br>`instructions/implement.md`<br>`references/summary-format.md` | One line after the `[MANUAL_TASK]` lines: how to record them done (`manual_close_note`, fix round 14) |
| `[METRICS_CONFLICT]` | `utils/metrics.py` | not for the host: metrics-note | — |
| `[METRICS_DUPLICATE]` | `utils/metrics.py` | not for the host: metrics-note | — |
| `[METRICS_INDEX_MISMATCH]` | `utils/metrics.py` | not for the host: metrics-note | — |
| `[METRICS_NO_TASK_ID]` | `utils/metrics.py` `NO_TASK_ID_MARKER` | not for the host: metrics-note | — |
| `[MODEL_FAILED]` | `utils/review_orchestrator_base.py` `MODEL_FAILED_MARKER` | `instructions/review-code.md`<br>`instructions/review-plan.md`<br>`instructions/review-tasks.md`<br>`references/summary-format.md` | The stdout line a reaggregate prints per model that is still failed, right after `record_models_present`: the model and its stored reason… |
| `[MODE_CHECK]` | `utils/providers/base.py` `MODE_CHECK_MARKER` | not for the host: runner-diagnostic | The after-the-fact checks of a slot on the incidental rung (F1 "What it fixes", Limits): a flipped executable bit on a tracked file… |
| `[NEXT_COMMAND]` | `utils/next_phase.py` `NEXT_COMMAND_MARKER` | `SKILL.md`<br>`instructions/improve-code.md`<br>`instructions/improve-commits.md`<br>`instructions/improve-plan.md`<br>`instructions/review-code.md`<br>`instructions/review-plan.md`<br>+2 more | The four line kinds of the block (§2.2(2)) |
| `[NEXT_COMMAND_ANSWERS]` | `utils/next_phase.py` `NEXT_COMMAND_ANSWERS_MARKER` | `references/summary-format.md` | — |
| `[NEXT_COMMAND_ANSWERS_WHY]` | `utils/next_phase.py` `NEXT_COMMAND_ANSWERS_WHY_MARKER` | `references/summary-format.md` | The answers line pair `util next-command` prints under a resume of a stopped native implement (fix round 14): the why on its own line, then… |
| `[NEXT_COMMAND_OPTIONAL]` | `utils/next_phase.py` `NEXT_COMMAND_OPTIONAL_MARKER` | not for the host: unnamed-handler | — |
| `[NEXT_COMMAND_OPTIONAL_WHY]` | `utils/next_phase.py` `NEXT_COMMAND_OPTIONAL_WHY_MARKER` | not for the host: unnamed-handler | — |
| `[NEXT_COMMAND_WHY]` | `utils/next_phase.py` `NEXT_COMMAND_WHY_MARKER` | `references/summary-format.md` | The line `util next-command` prints directly above the `--resume` of a stopped native implement (fix round 22): what that plain resume does… |
| `[NEXT_STEP_MARKDOWN]` | `utils/next_phase.py` `NEXT_STEP_MARKDOWN_MARKER` | `SKILL.md`<br>`instructions/implement.md`<br>`references/summary-format.md` | The lines `util next-command` prints around its answer's `markdown` (fix round 24, live run 10: Claude Code racing 06 put the manual tasks… |
| `[NEXT_STEP_MARKDOWN_END]` | `utils/next_phase.py` `NEXT_STEP_MARKDOWN_END_MARKER` | `SKILL.md`<br>`instructions/implement.md`<br>`references/summary-format.md` | — |
| `[NOTHING_TO_REVIEW]` | `utils/commit_range.py` `NOTHING_TO_REVIEW` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | Markers |
| `[NOT_YET_LOADED]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[OUTPUT_FILE]` | `utils/apply_orchestrator_base.py`<br>`utils/apply_output_helpers.py` | `references/summary-format.md` | — |
| `[OWN_CHECK_NOT_RECORDED]` | `utils/batch_done.py` `OWN_CHECK_NOT_RECORDED_MARKER` | `instructions/apply-code-fixes.md`<br>`instructions/implement.md` | Fix round 24 (live run 10, Claude Code racing 06: all ten implement records lacked `--own-check passed` while every reply said its check… |
| `[PIPE_HELD]` | `utils/process_capture.py` | not for the host: runner-diagnostic | — |
| `[PORCELAIN_CHECK]` | `utils/review_orchestrator_base.py` | not for the host: runner-diagnostic | — |
| `[PREREQUISITE_CHECK]` | `implement_orchestrator.py` | `instructions/implement.md`<br>`references/implement-prerequisites.md` | — |
| `[PREREQUISITE_SKIPPED]` | `check_workflow_prerequisites.py` `SKIPPED_MARKER` | `instructions/generate-tasks.md`<br>`instructions/implement.md`<br>`references/implement-prerequisites.md` | — |
| `[PREREQUISITE_UNMET]` | `check_workflow_prerequisites.py` `UNMET_MARKER`<br>`full_workflow_orchestrator.py` | `instructions/generate-tasks.md`<br>`instructions/implement.md`<br>`instructions/review-tasks.md`<br>`references/implement-prerequisites.md` | — |
| `[PROGRESS]` | `utils/progress_tracker.py` `PROGRESS_PREFIX` | `instructions/review-code.md`<br>`instructions/review-plan.md`<br>`instructions/review-tasks.md` | The greppable marker, in the style of the existing `[SALVAGE_NEEDED]` lines |
| `[PROMPT_SIDECAR_REWRITTEN]` | `utils/validation.py` | not for the host: run-notice | — |
| `[PROMPT_SIDECAR_SKIPPED]` | `utils/validation.py` | not for the host: run-notice | — |
| `[RANGE_AFTER_FLAGS]` | `utils/commit_front_door.py` `RANGE_AFTER_FLAGS` | `SKILL.md`<br>`instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[RANGE_BAD_REF]` | `code_review_orchestrator.py`<br>`utils/commit_range.py` `RANGE_BAD_REF` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[RANGE_DIVERGED]` | `utils/commit_range.py` `RANGE_DIVERGED` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[RANGE_LARGE]` | `utils/commit_range.py` `RANGE_LARGE` | `instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[RANGE_SHALLOW]` | `utils/commit_range.py` `RANGE_SHALLOW` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[RANGE_UNRECOGNIZED]` | `utils/commit_range.py` `RANGE_UNRECOGNIZED`<br>`utils/run_status.py` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[RANGE_UNRESOLVED]` | `utils/commit_range.py` `RANGE_UNRESOLVED` | `instructions/apply-commit-fixes.md`<br>`instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[REAGGREGATE_AFTER_SALVAGE]` | `code_review_orchestrator.py` | `instructions/improve-commits.md`<br>`instructions/review-commits.md`<br>`references/salvage-handling.md` | — |
| `[REPLY_INVALID]` | `utils/check_reply.py` | `instructions/review-code.md`<br>`instructions/review-plan.md`<br>`instructions/review-tasks.md`<br>`references/revalidation.md` | — |
| `[REPLY_MISSING]` | `utils/check_reply.py` `MISSING_MARKER` | `instructions/review-code.md`<br>`instructions/review-plan.md`<br>`instructions/review-tasks.md`<br>`references/revalidation.md`<br>`references/wave-batching.md` | What a task with no reply on disk prints (fix round 28): the reply is not there yet, which is no failure and nothing to re-ask |
| `[REPLY_OK]` | `utils/check_reply.py` | not for the host: phase-progress | — |
| `[RESULT_FILE_MISNAMED]` | `utils/review_orchestrator_base.py` | `instructions/improve-code.md`<br>`instructions/improve-commits.md`<br>`instructions/improve-plan.md`<br>`instructions/review-code.md`<br>`instructions/review-commits.md`<br>`instructions/review-plan.md`<br>+1 more | — |
| `[RESULT_NOT_READ]` | `utils/result_files.py` | not for the host: run-notice | — |
| `[RESULT_SET_ASIDE]` | `utils/result_files.py` | `instructions/improve-code.md`<br>`instructions/improve-commits.md`<br>`instructions/improve-plan.md`<br>`instructions/review-code.md`<br>`instructions/review-commits.md`<br>`instructions/review-plan.md`<br>+1 more | — |
| `[RESULT_STALE]` | `utils/result_files.py` | not for the host: run-notice | — |
| `[RESUME_COMMAND]` | `utils/next_phase.py` `RESUME_COMMAND_MARKER` | `instructions/apply-commit-fixes.md` | The one line `emit_resume` prints: the command that resumes a stopped write phase, and nothing else on the line (same rule as… |
| `[RESUME_REFUSED]` | `utils/batch_done.py` `RESUME_REFUSED_MARKER` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md`<br>`instructions/implement.md` | The first line of a refused native `--resume` (stderr) |
| `[REVALIDATION_BATCHES_PENDING]` | `utils/apply_orchestrator_base.py` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`references/revalidation.md` | — |
| `[REVALIDATION_PENDING]` | `utils/apply_orchestrator_base.py` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`references/revalidation.md` | — |
| `[REVIEW_SCOPE_REGENERATED]` | `code_review_orchestrator.py` | not for the host: run-notice | — |
| `[REVIEW_UNSUPPORTED]` | `code_review_orchestrator.py` | `instructions/review-code.md` | — |
| `[RUNNER_WARNING]` | `utils/next_phase.py` `RUNNER_WARNING_MARKER` | `SKILL.md`<br>`references/summary-format.md` | One line above the block when the command it names is certain to be refused (`check_workflow_prerequisites.repo_wide_sandbox_warning`) |
| `[SALVAGE_NEEDED]` | `code_review_orchestrator.py`<br>`utils/review_orchestrator_base.py` | `instructions/review-code.md`<br>`instructions/review-plan.md`<br>`instructions/review-tasks.md`<br>`references/salvage-handling.md` | — |
| `[SALVAGE_RETIRED]` | `utils/review_orchestrator_base.py` | not for the host: run-notice | — |
| `[SALVAGE_TASK]` | `code_review_orchestrator.py`<br>`utils/review_orchestrator_base.py` | `references/salvage-handling.md` | — |
| `[SALVAGE_UNRECOVERABLE]` | `utils/review_orchestrator_base.py` | `references/salvage-handling.md` | — |
| `[SANDBOX_BYPASSED]` | `utils/providers/base.py` `SANDBOX_BYPASSED_MARKER` | not for the host: runner-diagnostic | What every artifact of a bypassed slot says, verbatim — the fan-out's stdout line, the executor's preflight line, `models_served` in… |
| `[SANDBOX_FALLBACK]` | `utils/providers/base.py` `SANDBOX_FALLBACK_MARKER` | not for the host: runner-diagnostic | The marker for a slot that ran on a **contained** fallback rung — a real OS boundary that is not the adapter's default one |
| `[SANDBOX_INCIDENTAL]` | `utils/providers/base.py` `SANDBOX_INCIDENTAL_MARKER` | not for the host: runner-diagnostic | The marker for a slot on an `incidental`-grade rung |
| `[SANDBOX_RUNG_CHANGED]` | `utils/review_orchestrator_base.py` `SANDBOX_RUNG_CHANGED_MARKER` | not for the host: runner-diagnostic | The once-per-run line for specs that ran on the plugin's jail last run and without any sandbox this one |
| `[SANDBOX_RUNG_RETIRED]` | `utils/providers/base.py` `SANDBOX_RUNG_RETIRED_MARKER` | not for the host: runner-diagnostic | The marker for the once-only line saying a contained rung the previous run took is no longer usable (`sandbox_rung_retired_notice`) |
| `[SANDBOX_UNAVAILABLE]` | `utils/review_orchestrator_base.py` | not for the host: runner-diagnostic | — |
| `[SCOPE_BASELINE_MOVED]` | `utils/review_orchestrator_base.py` | not for the host: run-notice | — |
| `[SCOPE_BLOCKER]` | `utils/scope_apply.py` `BLOCKER_MARKER` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md` | — |
| `[SCOPE_CHANGED]` | `code_review_orchestrator.py` | `instructions/review-code.md` | — |
| `[SCOPE_DEFERRED]` | `utils/scope_apply.py` `DEFERRED_MARKER` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md` | Marker lines |
| `[SCOPE_DRIFT]` | `utils/commit_range.py` `SCOPE_DRIFT` | `instructions/improve-commits.md`<br>`instructions/review-commits.md` | — |
| `[SCOPE_FALLBACK]` | `code_review_orchestrator.py` | `instructions/improve-code.md`<br>`instructions/review-code.md` | — |
| `[SCOPE_FLAGS_IGNORED]` | `code_review_orchestrator.py` | not for the host: run-notice | — |
| `[SCOPE_LARGE]` | `code_review_orchestrator.py` | `instructions/review-code.md` | — |
| `[SCOPE_POLICY]` | `utils/scope_round.py` | `instructions/review-code.md`<br>`instructions/review-plan.md`<br>`instructions/review-tasks.md` | — |
| `[SCOPE_POLICY_CHANGED]` | `utils/scope_apply.py` `RETIRED_MARKER` | not for the host: run-notice | — |
| `[SCOPE_POLICY_CONFLICT]` | `utils/scope_round.py` | `instructions/review-code.md`<br>`instructions/review-plan.md`<br>`instructions/review-tasks.md`<br>`references/resume-detection.md` | — |
| `[SELECTION_CHANGED]` | `utils/review_orchestrator_base.py` | not for the host: run-notice | — |
| `[SHIM_ALSO_FOUND]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[SHIM_DIR_REMOVED]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[SHIM_EDIT_PRESERVED]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[SHIM_INSTALLED]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[SHIM_KEPT]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[SHIM_NOT_FOUND]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[SHIM_REMOVED]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[SHIM_SHADOWED]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[SHIM_UNCHANGED]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[SHIM_WOULD_REMOVE]` | `utils/host_install.py` | not for the host: operator-cli | — |
| `[SKIPPED]` | `full_workflow_orchestrator.py` | not for the host: phase-progress | — |
| `[SKIP]` | `ask_orchestrator.py`<br>`code_review_orchestrator.py`<br>`review_plan_orchestrator.py`<br>`utils/review_orchestrator_base.py` | `SKILL.md`<br>`instructions/review-code.md`<br>`instructions/review-plan.md`<br>`instructions/review-tasks.md`<br>`references/host-capabilities.md`<br>`references/interrupted-run-recovery.md`<br>+1 more | — |
| `[STATE_REPLACED]` | `utils/commit_front_door.py` `STATE_REPLACED` | not for the host: run-notice | Step 6 replaced a `state.json` that did not parse (§2.1 allows it; the apply door refuses the same file loudly, so this one says it too) |
| `[STREAM_CAP]` | `utils/process_capture.py` | not for the host: runner-diagnostic | — |
| `[TASKS_GRAPH_OK]` | `utils/tasks_graph_check.py` | `instructions/apply-task-suggestions.md` | — |
| `[TASKS_GRAPH_WARNING]` | `utils/tasks_graph_check.py` `WARNING_PREFIX` | `instructions/apply-task-suggestions.md` | — |
| `[TASKS_MISSING]` | `implement_orchestrator.py` | `instructions/implement.md`<br>`references/implement-prerequisites.md` | — |
| `[TASK_GENERATION_PENDING]` | `generate_tasks_orchestrator.py` `PENDING_MARKER` | not for the host: unnamed-handler | Printed when a native-subagent host must run the prompt itself |
| `[TASK_SUGGESTIONS_ADVISORY]` | `implement_orchestrator.py` | `instructions/implement.md`<br>`references/implement-prerequisites.md` | — |
| `[TESTS_NOT_RUN]` | `utils/apply_orchestrator_base.py`<br>`utils/next_phase.py` `TESTS_NOT_RUN_MARKER` | `SKILL.md`<br>`instructions/apply-code-fixes.md`<br>`instructions/implement.md`<br>`instructions/improve-code.md`<br>`instructions/improve-commits.md`<br>`references/summary-format.md` | The line `util next-command` prints above a stopped native implement's commands when nothing ran the repository's tests… |
| `[USAGE_LIMIT]` | `utils/providers/claude_code.py` `USAGE_LIMIT_TAG` | not for the host: runner-diagnostic | The tag `ClaudeCodeProvider.describe_failure` puts in front of a 429 whose sentence says the *account* is out of budget rather than the… |
| `[UV_MISSING]` | `packaging/opencode/src/index.mjs`<br>`utils/unattended.py` | not for the host: operator-cli | — |
| `[VALIDATION_BATCHES_PENDING]` | `code_review_orchestrator.py`<br>`utils/review_orchestrator_base.py` | `instructions/review-code.md`<br>`instructions/review-plan.md`<br>`instructions/review-tasks.md` | — |
| `[VALIDATION_CONTEXT_CITED]` | `utils/validation_sections.py` | not for the host: run-notice | — |
| `[VALIDATION_CONTEXT_EXCERPTED]` | `utils/validation.py` | not for the host: run-notice | — |
| `[VALIDATION_CONTEXT_OUTSIDE_ROOT]` | `utils/review_orchestrator_base.py` | not for the host: run-notice | — |
| `[VALIDATION_CONTEXT_REBOUND]` | `utils/validation.py` | not for the host: run-notice | — |
| `[VALIDATION_CONTEXT_TRUNCATED]` | `code_review_orchestrator.py`<br>`utils/validation.py` | not for the host: run-notice | — |
| `[VALIDATION_PENDING]` | `code_review_orchestrator.py`<br>`utils/review_orchestrator_base.py` | `instructions/review-code.md`<br>`instructions/review-plan.md`<br>`instructions/review-tasks.md` | — |
| `[VALIDATION_RETIRED]` | `utils/review_orchestrator_base.py` | not for the host: run-notice | — |
| `[VALIDATION_TASKS_OVERSIZE]` | `utils/validation.py` | not for the host: run-notice | — |
| `[WARN]` | `ask_orchestrator.py`<br>`utils/provider_outcome.py`<br>`utils/providers/claude_code.py` `warn_gate_once()`<br>`utils/providers/pi.py` `warn_gate_once()`<br>+2 more | `instructions/ask.md`<br>`references/summary-format.md` | — |
| `[WORKFLOW_COMPLETE]` | `utils/next_phase.py` `WORKFLOW_COMPLETE_MARKER` | `SKILL.md`<br>`references/summary-format.md` | — |
| `[WRITE_ARTIFACT]` | `utils/write_prompt.py` `MARKER_ARTIFACT` | `instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md` | — |
| `[WRITE_PHASE_NOT_EXECUTED]` | `full_workflow_orchestrator.py` | not for the host: phase-progress | — |
| `[WRITE_PROMPT]` | `utils/write_prompt.py` `MARKER_PATH` | `instructions/apply-code-fixes.md`<br>`instructions/apply-commit-fixes.md`<br>`instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md`<br>`instructions/implement.md`<br>`instructions/improve-commits.md` | The marker lines a host reads (`--stdout` prints none of them) |
| `[WRITE_PROMPT_BLOCKED]` | `utils/write_prompt.py` `MARKER_BLOCKED` | `instructions/implement.md` | The refusal for a task a check with findings holds back (stderr, exit 3) |
| `[WRITE_PROMPT_CONTEXT]` | `utils/write_prompt.py` `MARKER_CONTEXT` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md`<br>`instructions/implement.md` | — |
| `[WRITE_PROMPT_LINES]` | `utils/write_prompt.py` `MARKER_LINES` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md`<br>`instructions/implement.md` | — |
| `[WRITE_PROMPT_ORDER]` | `utils/write_prompt.py` `MARKER_ORDER` | `instructions/apply-code-fixes.md`<br>`instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md`<br>`instructions/implement.md` | The warning a prompt bound out of order prints (stderr; never a refusal) |
| `[WRITE_PROMPT_REFUSED]` | `utils/write_prompt.py` `MARKER_REFUSED` | `instructions/implement.md` | The refusal for a task the write policy forbids (stderr, exit 3) |
| `[WRITE_SNAPSHOT]` | `utils/write_prompt.py` `MARKER_SNAPSHOT` | `instructions/apply-suggestions.md`<br>`instructions/apply-task-suggestions.md` | The marker lines naming the snapshot and the file it copies |
| `[WRITE_TASKS_FILE]` | `implement_orchestrator.py`<br>`utils/apply_orchestrator_base.py` | `instructions/apply-suggestions.md`<br>`instructions/implement.md` | — |

## Not-for-the-host reasons

| Key | Why no instruction names the marker |
| --- | --- |
| `operator-cli` | Output of `install-host`, `doctor`, the uv bootstrap or `util instructions`, read by the person running that command. No workflow phase drives them. |
| `runner-diagnostic` | A sandbox, provider-runner or process-capture diagnostic printed into the run's output and relayed with it. No instruction branches on it. |
| `metrics-note` | A `utils/metrics.py` warning about the run's timing record. It changes no decision the host makes. |
| `run-notice` | A one-line notice that an orchestrator adjusted, set aside or refused an input. The line explains itself; the instruction acts on the exit code and `MULTI_LLM_RESULT`, not on the marker. |
| `phase-progress` | Progress or outcome text of one orchestrator that the phase's generic 'relay the report' rule already covers. |
| `unnamed-handler` | Printed to the host with a phase report, and reached only through a generic rule (relay the report, the detached-run text) rather than by name. The likeliest candidates for a named handler; recorded rather than assumed. |
