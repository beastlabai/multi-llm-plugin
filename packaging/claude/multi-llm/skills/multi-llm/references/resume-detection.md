# Resume Detection

Check for existing output BEFORE running the orchestrator to avoid expensive duplicate runs.

The review cores' resume step runs `util resume-state <mode> <plan>`, which makes these checks
itself, read-only, and prints the case and the action as one `[RESUME_STATE]` line. This file is
what that line means; with the command unavailable, make the same checks by hand.

Two rules hold throughout. Only `state.json` completes a phase: files on disk never do. And no file
date says which round wrote a file (a clock set back, a copy that kept its dates or a coarse file
system all mislead): only `.status.json`'s `validation_pending` beside the task file does. When
nothing proves whose validation files are on disk, the answer is a `--force` re-run (case c), which
may re-run work but never reports an earlier round's verdicts as this one's.

## Detection Cases

<!-- BEGIN GENERATED: resume cases (utils/resume_state.py) -->
Check in this order; the first case that matches is the answer. `util resume-state` prints it as `[RESUME_STATE] case=<a-f> action=<word> models=<…> batches=<…> flags=<…> ownership=<…>`.

a. If `state.json` records the phase complete: the orchestrator's own completed-phase guard. Files on disk never make a phase complete: a `--force` re-run leaves the previous round's `validation.json` and `report.md` in place until it regroups:
   -> The phase is complete. Report results from `{plan}/{phase}/report.md` (or `report.html`) and skip (`report`).

b. If `{plan}/{phase}/.status.json` reads `state: "validation_pending"`, `validation_tasks.json` beside it lists its batches, and `validation_batch_*.json` files exist (a retired round's `*.stale.json` does not count). That checkpoint word is what proves the task file is this round's: a run writes it right after the task file, and the next fan-out's first write replaces it:
   -> Validation ran partially or fully. If every batch the task file lists has its `validation_batch_<N>.json`: run `--reaggregate` to finalize (`reaggregate`). If some are missing: run only the validation tasks of the missing batches (`batches=` names them), then reaggregate (`validate`).

c. If `validation.json`, `validation_tasks.json` or a `validation_batch_*.json` exists in `{plan}/{phase}/`, but the checkpoint does not prove them this round's as in b and f: `.status.json` reads another state (a re-run's `models_running`, `models_complete` or `grouping_complete`, or a `complete` that `state.json` no longer records) or is missing, or the task file cannot be read. No file date settles it:
   -> Their round is uncertain: reporting, reaggregating or validating them could finish the phase on an earlier round's verdicts. Run the orchestrator again with `--force` (every phase) and the models `models_requested` records (`rerun`, `ownership=uncertain`; `models=-` when none is recorded: launch with the selection the core would make). It keeps each reusable model result (`[SKIP]`), regroups, retires the verdicts its grouping does not bind and validates again. Never `--rerun-all`.

d. If no validation file (as in c), but per-model `*.json` result files (or a `grouped.json`) exist in `{plan}/{phase}/`, or `.status.json` reads a fan-out stage (`models_running`, `models_complete`, `grouping_complete`) and lists `models_requested`:
   -> The fan-out stopped before validation. Compare the files with `models_requested` in `{plan}/{phase}/.status.json`; a file counts when the fan-out would keep it (non-empty, a JSON list, and on review-code tied by the scope ledger `.review-scope.json` to the scope `review-scope.md` records). If every requested model has its file: run `--reaggregate`. If some have none: run the orchestrator again with the same models. With no file at all that is still the original run's whole selection. It keeps each file (`[SKIP]`) and runs only the missing ones; a `--reaggregate` would report without them. On review-code add `--force`, which a dir holding results or an earlier round's files needs (exit 3 otherwise); a composite's flags line already carries it. Never `--rerun-all`. Left out, `--scope-policy` stays the one the kept files were requested under (one `Scope policy notice:` line says so); a different one typed stops at exit 3 with `[SCOPE_POLICY_CONFLICT]`. The first answer is `reaggregate`; the second is `rerun`, its `models=` listing every requested model.

e. If none of the other cases; or `state.json` records a code review completed before the latest implement (checked first: its files are out of date):
   -> Nothing to resume: run the orchestrator as planned (`run`). A code review older than the latest implement is re-run as `--force` by the orchestrator itself.

f. If `.status.json` reads `state: "validation_pending"`, `validation_tasks.json` lists its batches (as in b), and NO `validation_batch_*.json` exists:
   -> The orchestrator handed validation over and none has run. Skip to Validation Handling: run every validation task, then reaggregate (`validate`).

The action words:

- `report`: the phase is complete: report its results and run no orchestrator.
- `reaggregate`: run the orchestrator's `--reaggregate`; no model runs.
- `validate`: go to the validation step: run the validation tasks (only the batches `batches=` names, when it names any), then `--reaggregate`.
- `rerun`: launch the orchestrator again with `--models` exactly as `models=` lists them (`-`: none recorded, so the selection the core would make) and the `flags=` added; models with a result keep it (`[SKIP]`).
- `run`: nothing to resume: launch the orchestrator as planned.

`ownership=uncertain` (case c) says why a `rerun` was chosen over the files on disk: nothing recorded ties them to the run `.status.json` describes. Every other answer prints `ownership=-`.
<!-- END GENERATED: resume cases -->

## Orchestrator Guards

The orchestrator has its own re-run protection as defense-in-depth:
- **Primary**: Exits with code 2 if phase is already marked complete in state.json. Use `--force` to override.
- **Secondary** (code-review only): Exits with code 3 if partial completion artifacts exist without the phase being complete.

This instruction-level resume detection is the FIRST line of defense; the orchestrator guards are backup protection.
