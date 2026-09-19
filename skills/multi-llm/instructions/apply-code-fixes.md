# Apply Code Fixes Mode Instructions

Applies validated fixes from a code review to the codebase, using smart batching by file location to minimize subagent calls while maintaining edit safety.

## Usage

```bash
/multi-llm:multi-llm --apply-code-fixes <plan_path> [options]
```

## Prerequisites

This mode requires a completed code review with validation. The following files must exist in `{plan}/code-review/`:
- `grouped.json` - Grouped issues from review
- `validation.json` - Validation results

If these files don't exist, run `--review-code` first.

### User Selections (Auto-Detected)

If available, the following file is auto-detected from `{plan}/code-review/`:
- `user_selections.json` - User decisions from the HTML report (skips, validation overrides, edited descriptions)

HTML selections take precedence over markdown checkbox selections.

### Consolidated Decisions (Auto-Detected)

If the code-review phase included a consolidation step, the following files are auto-detected from `{plan}/code-review/`:
- `consolidated-report.md` - Consolidated issue report
- `consolidated_user_selections.json` - User decisions on consolidated groups

No additional flags are needed — consolidation integration is automatic.

## Smart Batching

The orchestrator groups related fixes by file for efficient processing:

### Batching Heuristics

1. **File Proximity**: Fixes targeting the same file are grouped together
2. **Safety Isolation**:
   - Security fixes are ALWAYS isolated (one per batch)
   - HIGH importance fixes are ALWAYS isolated
   - Other fixes can be batched together
3. **Size Limits**: Max 3 fixes per batch, max 3000 chars total description
4. **Line Ordering**: Fixes within a file are processed in line order (top to bottom)
5. **Priority Ordering**: Batches are sorted by priority score (higher importance first)

### Subagent Routing

All fix batches use `executor_role: "agent"` — a code review never emits a manual step. (Batches also carry an optional host-specific subagent-type hint for hosts whose own subagent mechanism requires one; no reader branches on it.)

### Efficiency Gains

With batching, 10 fixes across 4 files might become 5-6 batches:
- **Before**: 10 subagent calls (one per fix)
- **After**: 5-6 subagent calls (grouped by file, security isolated)
- **Savings**: 40-50% reduction in API calls

## Process

1. **Load Code Review Results**: Read the grouped issues and validation JSON
2. **Filter Valid Fixes**:
   - Only process fixes with `status: "valid"`
   - Exclude issues marked `[x] Skip` in `report.md`
3. **Smart Batching**: Group related fixes by file location
4. **Handle Human Decisions**: For `needs-human-decision` items, **ask the user** — each rendered question also offers a per-item **Let Claude decide** option that offloads just that one fix's judgment to a subagent. Or, if `--claude-decide` was passed, let Claude judge every fix up front — or pre-marked in the report's *Let Claude decide* dropdown/checkbox, which routes just those fixes to the judge without prompting. See step 6. **Under `--yes` (or `--non-interactive`) this site is never rendered** (D6 row 10): `--yes` sets `--claude-decide` itself, so every item goes through the judged/deferred path up front and no item — and no batched approve-all/skip-all UI — can block.
5. **Batch Application**: Process each batch with a specialist subagent:
   - Run the batch as a subagent (every batch is `executor_role: "agent"`)
   - Wait for completion before moving to the next batch
   - This prevents conflicts from concurrent edits
6. **Run Verification**: Run the repository's `driver.verify_command`, if one is configured, after all fixes
7. **Generate Report**: Create `{plan}/apply-fixes/summary.md` with results
8. **Update Plan**: Add "Code Fixes Applied" section to plan file
9. **Mark Phase Completed**: Record `apply-code-fixes` in `state.json` — **only** when every batch was applied and verified. See step 10.

## Output Files

```
plans/todo/my-feature/
├── state.json                           # Session state (plan-local)
├── code-review/                         # Input files from review phase
│   ├── report.md                        # Original review
│   ├── grouped.json                     # Grouped issues
│   └── validation.json                  # Validation data
└── apply-fixes/                         # Output files from this phase
    └── summary.md                       # Summary of applied fixes
```

## Options

### Basic Options
- `--dry-run`: Show what batches would be processed without making changes
- `--min-priority {low,medium,high}`: Minimum importance level to include (default: low = all valid fixes)
- `--include-low`: [DEPRECATED] No longer needed, low is now default. Use `--min-priority` to filter.
- `--no-batch`: Disable smart batching (process each fix separately)
- `--max-batch-size N`: Maximum fixes per batch (default: 3)
- `--base-ref REF`: Git ref to use for diffs (default: from state file)

### Confirmation Behavior
- `--no-confirm`: Skip confirmation when no user selections are found (for unattended/silent operation)
- Also bypassed by: `--yes`, `--force`, `--approve-all`, `--skip-all-human`, `--approve-all-low`, `--approve-importance`, `--approve-validation-failed`, `--dry-run`

### Bulk Approval Options

These options reduce manual review for items with `needs-human-decision` status:

- `--claude-decide` (alias `--let-claude-decide`): Let Claude evaluate each `needs-human-decision` fix with its own judgment instead of prompting you. This is the non-interactive equivalent of choosing "Let Claude decide" in the review prompt. Unlike the blanket flags below, it is a **per-item** judgment that **salvages partially-valid fixes** (trims them to their worthwhile core and applies that) rather than skipping them wholesale; only fixes with nothing worthwhile are skipped (see "Let Claude Decide" Mode in `references/human-decision-batch.md`). It does **not** bypass the no-selection confirmation for valid fixes — combine with `--no-confirm` for fully unattended runs.
- `--approve-all-low`: Auto-approve all LOW importance items
- `--approve-all`: Auto-approve ALL items (requires `--yes` or `--force`)
- `--skip-all-human`: Skip all `needs-human-decision` and `validation_failed` items
- `--approve-importance LOW MEDIUM`: Auto-approve items at specified importance levels
- `--approve-validation-failed`: Auto-approve items that had recoverable validation failures (parsing errors, timeouts)

### Safety Guardrails

- `--yes` / `-y`: Confirm bulk approval operations (required with `--approve-all`)
- `--force` / `-f`: Alias for `--yes`
- `--include-high`: Allow `--approve-all` to include HIGH importance items

### Revalidation Options

When validation failed due to parsing errors or timeouts, you can retry:

- `--revalidate`: Re-run validation only on items with `validation_failed` status
- `--revalidate-model MODEL`: Use a different model for retry (e.g., `cursor-agent:opus`)
- `--revalidate-all-human`: Also re-validate all `needs-human-decision` items
- `--internal-revalidation`: Run revalidation inside the orchestrator (legacy mode)

**Note**: By default, revalidation is delegated to a subagent. Use `--internal-revalidation` to run revalidation inside the orchestrator (legacy mode).

### Resume/State Options

- `--resume`: Resume from last checkpoint, skip already-processed items
- `--fresh`: Clear previous progress and start fresh
- `--skip-human-review`: [DEPRECATED] Use `--skip-all-human` instead

### Batch Review Mode

- `--batch-review-mode {individual,by-importance,summary-only}`: How to present items for human review (default: by-importance)

### User Skip Filtering

If users have marked issues to skip in `report.md`, the orchestrator will:
1. Parse `{plan}/code-review/report.md` for `[x] Skip` markers
2. Filter out matching issues before processing
3. Report: `User skipped N issues: 1, 3, ...`

This happens automatically - no additional flags needed.

---

## Step-by-Step Execution

### 1. Validate Prerequisites

Check that required files exist:
```
{plan}/code-review/grouped.json
{plan}/code-review/validation.json
```

If not found, inform user to run `--review-code` first.

### 2. Run the Orchestrator

```bash
uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" apply-code-fixes --plan-file "$PLAN_PATH" [options]
```

**IMPORTANT**: Pass `$PLAN_PATH` as given — the orchestrator resolves it to an OS-native absolute path itself. Do NOT wrap it in `$(realpath ...)`: on Git for Windows, `realpath` emits a POSIX `/c/...` path that a native Windows Python process cannot use.

**IMPORTANT**: Run this command in the foreground — do **not** run it detached. The orchestrator runs quickly — it only generates batch JSON, it does not apply changes — so 10 minutes is far more than enough.

The orchestrator will:
1. Load validation results and grouped issues
2. Filter to only `valid` fixes (skip `invalid`, prompt for `needs-human-decision`)
3. Group fixes into efficient batches by file
4. Determine appropriate subagent type for each batch
5. Write output to `{plan}/apply-fixes/orchestrator_output.json`
6. Exit — the driving agent handles the actual application

### 2a. Handle No-Selections Confirmation

After the orchestrator completes, check if its stdout contains `"status": "confirmation_needed"`.

**Under `--yes` / `--no-confirm` this marker is never emitted** (D6 row 11): the
orchestrator treats either flag as deliberate intent and applies the valid set
without asking, so there is nothing here to render. The prompt below is a
Python-emitted status, not a markdown judgment — if you see it, `--yes` was not
passed.

If present (attended runs only):
1. Display the warning message to the user
2. **Ask the user**:
   > "No user selections were found (no user_selections.json, consolidated_user_selections.json, or report.md edits detected). This will apply ALL {item_count} valid fixes. Continue?"
   > Options: "Yes, apply all" / "No, I need to make my selections first"
3. If user confirms: Re-run the orchestrator with `--no-confirm` appended
4. If user declines: Stop and inform the user to review the HTML report and export selections

If NOT present, proceed to step 3.

### 3. Display Decision Summary

After the orchestrator completes, run the display script to show the user which fixes will be applied, skipped, need review, etc.:

```bash
uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util decisions --plan-file "$PLAN_PATH" --phase apply-code-fixes
```

This prints a formatted summary with each fix's title and status. Display this output to the user before proceeding.

### 4. Read the Orchestrator Output

Read `{plan}/apply-fixes/orchestrator_output.json`.

Extract the following key fields directly from the Read output (do NOT use Bash/python to parse — just read the JSON and understand it):
- `batches`: Array of batch objects, each with `prompt`, `file_key`, `prior_changes_id`, `executor_role`, `fixes`
- `needs_human_review`: Items requiring human decision
- `human_review_config`: Batch mode config for human review
- `summary`: Counts of valid, skipped, batches, etc.

### 5. Process Batches Sequentially

Use the batch data already loaded in step 4. Do NOT use Bash/python to re-parse or extract fields from the JSON file.

**Before the loop**, mark the phase start so ETA tracking has a wall-clock baseline. Run:
```bash
uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics start \
  --state-file "$STATE_FILE" --phase "apply-code-fixes" --total-batches {total_batches}
```
Where `{total_batches}` is the length of the `batches` array.

### The Python write executor: `--exec-subagents` (an alternative to this loop)

Everything from here to the end of the batch loop is the **native-subagent** path:
this agent substitutes `{prior_changes_context}`, snapshots the artifact, spawns a
subagent, diffs against the snapshot and appends to the prior-changes log.

A host with no native subagent tool runs the *same* loop in Python instead. Add
`--exec-subagents` to the orchestrator invocation above and the phase schedules
itself in one process: batches in ascending `batch_index`, each one's snapshot/diff
classification completed and recorded **before** the next is dispatched, the
prior-batch context bound late per dispatch (the literal `{prior_changes_context}`
token never reaches a runner), `prior_changes.py append` and `metrics.py record`
called per batch, and the phase marked completed only after every batch verified.
The runner comes from `driver.subagent_runner` in `providers.yaml`; write
concurrency comes from `driver.max_concurrent_write_subagents` (default **1**, and
raise it above 1 only for this phase, whose batches are partitioned by file — and even then two batches co-schedule only when their declared file sets are provably disjoint).

Do **not** combine the two: with `--exec-subagents` the phase finishes in that one
invocation and there is no batch left for this agent to run. It also writes this
phase's own summary report itself: the batch loop's outcome block names every item
the judge decided, `{phase}/summary.md` is written with the same sections the
*Generate Summary Report* step below describes, the recorded decisions are overlaid
onto the review report, and the phase is marked completed only once every batch
verified. Those steps below belong to the native path; do not run them a second time.

Three flags belong to that path and are worth knowing even on this one:

- `--dry-run-subagents` — full preflight, **zero dispatch**. Renders the exact
  command per batch, applies the invocation profile, the co-scheduling admission
  check and the `driver.write_paths` policy, writes
  `{phase}/subagent_dry_run.json` and exits 0 having started no child and changed
  nothing else. The recommended first unattended run on a new host.
- `--detach` — the phase self-daemonizes and returns a log path plus a `status`
  command to poll. Per-batch progress lands in `{phase}/.status.json` under
  `progress.write_subagents`, so an unattended write run is observable with no chat
  context at all. Never combine it with a host's own background primitive: on Claude
  host resolves *run detached* to its own background mechanism uses that
  mechanism and never passes `--detach`.
- Rollback, when a batch half-applies —
  `multi_llm.py util restore-batch --phase-dir <dir> --task-id <id>` (also
  `--all-failed`), invoked through the dispatcher the same way every other mode is.
  It restores exactly the files `{phase}/subagent_audit.jsonl` recorded as touched,
  from the pre-batch snapshot under `{phase}/pre_batch/{task_id}/`, and **refuses**
  a file whose current hash no longer matches what the audit recorded at completion
  (someone edited it since) unless `--force` is passed. A partial application never
  reverts itself: the run exits 3 naming this command, because an automatic revert
  would destroy the half-applied state a human needs to inspect.

For each batch (index N, starting at 1) returned by the orchestrator, you MUST:

1. **Read prior changes** — Run:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util prior-changes read \
     --output-dir "{plan}/apply-fixes"
   ```
   Capture stdout as `prior_changes_text`.

   **Error handling**: If `read` fails (corrupted JSONL, permission error, unexpected exception), use the default text `(none — this is the first batch)` and log a warning. Do not abort the batch loop.

2. **Substitute placeholder** — In `batch.prompt`, replace the literal `{prior_changes_context}` with `prior_changes_text`.

3. **Run the batch as a subagent** (`executor_role: "agent"` on every fix batch):
   - `general-purpose` for all fixes
4. **Wait for completion** - Do NOT parallelize batch processing

5. **Verify batch outcome** — After the subagent completes (or fails), determine the batch outcome by inspecting actual file/diff state rather than relying solely on the subagent's exit status. Classify the result:
   - **Success**: The expected changes are present in the target files (verified via diff or content check).
   - **Failure (clean)**: The subagent failed and no file modifications were made (verified by checking that target files are unchanged from their pre-batch state).
   - **Partial application**: Some but not all expected changes are present, or the subagent failed but file modifications were detected.

   **Partial application handling**: If verification detects a partial application, **stop the batch loop and surface the issue for human review**. Do not continue to subsequent batches. Log the partial application details and prompt the human operator to resolve the state before resuming.

6. **Extract summary** — Generate a one-line summary by compressing the subagent's response into a single sentence. Do **not** require the subagent to output a special summary line.

   **Normalize** the summary: collapse any newlines or multiple spaces to a single space, trim leading/trailing whitespace, and truncate to 200 characters if needed.

7. **Append to prior changes (only after verification)** — This step runs **only** after step 5 has confirmed success or clean failure. It must **never** run for partially-applied batches. Pass the summary via stdin:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util prior-changes append \
     --output-dir "{plan}/apply-fixes" \
     --id "{batch.prior_changes_id}" --phase "apply-fixes" --summary-stdin <<'SUMMARY_EOF'
   {normalized_summary}
   SUMMARY_EOF
   ```
   The `--id` is `batch.prior_changes_id` — read it off the batch, never construct one.
   The batcher emits **many** batches for the same `file_key`, and `append` is idempotent
   on `(id, phase)`: an id two batches share records the first and silently drops the
   rest (exit 0, a `Notice:` on stderr), so every later batch reads a prior-changes
   context missing its predecessors. `prior_changes_id` is that key plus a digest of the
   batch's own items — unique per batch, and identical if the same batch is re-dispatched.
   Do **not** use the batch loop index. The `--phase` is `"apply-fixes"` — the
   `--exec-subagents` executor appends under `"apply-code-fixes"` (its mode word)
   instead, and `util prior-changes` accepts both: they are normalized to the same
   phase (`utils/prior_changes.PHASE_ALIASES`), so a batch dispatched on one leg
   and resumed on the other is recognised as the same entry rather than
   re-appended under a phase the log now treats as "other".

   For **clean failures**, use summary: `(batch failed — {compact reason}, verified no file changes)`

   **Error handling**: If `append` fails, log a warning but do not abort the batch loop.

8. **Log the result** - Track success/failure for the summary
9. **Record metrics** — Run:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics record \
     --state-file "$STATE_FILE" --phase "apply-code-fixes" \
     --label "Batch {N} ({file_key})" --subagent-type "{executor_role}" \
     --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
     --total-batches {total_batches} --batch-index {N}
   ```
   Where `token_count`, `tool_uses`, `duration_ms` come from whatever your subagent mechanism reports. Omit any flag it does not report. The script prints an `[ETA]` line to stderr after each call.

**CRITICAL**: Process batches ONE AT A TIME. Never spawn multiple subagents simultaneously.

**After the loop completes**, mark the phase finish:
```bash
uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics finish \
  --state-file "$STATE_FILE" --phase "apply-code-fixes"
```

### 6. Handle Human Decision Items

Follow the human decision batch mode process in `references/human-decision-batch.md`.

**Auto-route report-pre-marked fixes first**: before prompting for anything, partition `needs_human_review` into (a) the **pre-marked Claude set** — every fix whose `group_id` is in `human_review_config.claude_decide_item_ids` or that carries `decision_mode == "claude_auto_decide"` — and (b) the remaining **interactive set**. Run **"Let Claude Decide" Mode** (per the criteria in `references/human-decision-batch.md`) on the pre-marked Claude set **without any prompt**, then handle only the interactive set with the flow below. These fixes were chosen in the report's "Let Claude decide" dropdown/checkbox; they are the per-item equivalent of `--claude-decide` scoped to just those fixes.

**Check `human_review_config.decision_mode` first**: if it is `"claude_auto_decide"` (the user passed `--claude-decide`), do NOT prompt the user — go straight to "Let Claude Decide" Mode and evaluate every `needs-human-decision` fix autonomously per the criteria in that reference. By default, **partially-valid fixes are salvaged** (rewritten to their worthwhile core and applied) rather than skipped; only fixes with nothing worthwhile are skipped entirely.

Context line for individual review display: `File: {file}, Type: {type}`

### 7. Run Verification

After all fixes are applied, run **the repository's `driver.verify_command`, if one
is configured** — the check this repo declares for itself in `.multi-llm/providers.yaml`
(`pnpm typecheck && pnpm lint:fix` in a pnpm/TypeScript repo, `cargo check`, `make test`,
whatever it says). Run it once, from the repository root, exactly as written. Do **not**
substitute a command of your own: a repo that declares none has declared none, and a
guessed `pnpm typecheck` in a repo with no `package.json` reports a failure the fixes did
not cause.

If no `verify_command` is configured, say so in the summary — "not configured" is the
honest verification status, and it is not the same as "passed".

If the command fails:
- Report which fix may have caused the issue
- Record the command, its exit code and the tail of its output in the summary
- Do **not** revert anything — every batch was already verified against its own
  pre-batch snapshot, and the half-applied state is what a human needs to read
- The phase is **not** complete: stop and hand the failure to a human

On the `--exec-subagents` leg the executor does all of this itself, from the same
`driver.verify_command` — it runs the command once after the last batch, records the
result in `summary.md` and `.status.json`, and ends the phase at exit 3 when it fails.

### 8. Generate Summary Report

After all batches are processed, create `{plan}/apply-fixes/summary.md` — the
same path the *Output Files* tree above lists, and the same one the
`--exec-subagents` leg writes for itself:

```markdown
# Applied Code Fixes Report

**Plan**: /path/to/plan.md
**Generated**: YYYY-MM-DD HH:MM:SS
**Base ref**: abc1234

## Summary
- Batches processed: N
- Fixes applied: N
- Fixes skipped: N
- Human decisions made: N
- Decided by the judge subagent: N (A approved, V salvaged, S skipped) — includes `--claude-decide`, the batch "Let Claude decide" option, and per-item "Let Claude decide" selections
- Verification: `{driver.verify_command}` pass/fail, or `not configured`

## Batching Efficiency
- Total fixes: N
- Total batches: N
- Subagent calls saved: N (X% efficiency gain)

## Subagent Distribution
- general-purpose: N batches

## Applied Fixes

### Batch 1 (File: src/api.ts)
**Subagent**: general-purpose
**Fixes**: 2
**Type**: bug

1. **[Fix Title]** - Applied
   - File: src/api.ts:42-50
   - Changes: Added null check before accessing user.profile

2. **[Fix Title]** - Applied
   ...

### Batch 2 (File: apps/web/supabase/migrations/xxx.sql)
**Subagent**: general-purpose
...

## Decided by Claude

Fixes whose decision was offloaded to Claude — via `--claude-decide`, the batch "Let Claude decide" option, or a per-item "Let Claude decide" selection. Read these from `state.json`: the `human_decisions_{phase}` entries whose `batch_context.decision_source` is `claude_auto_decide` or `claude_auto_decide_salvage`. Omit this section if there are none.

### [Fix Title] ({importance}) — Approved
**Reason**: {reason}

### [Fix Title] ({importance}) — Salvaged
**Kept**: {salvaged_description}
**Dropped**: {dropped}

### [Fix Title] ({importance}) — Skipped
**Reason**: {reason}

## Skipped Fixes

### [Fix Title]
**Reason**: Invalid / User skipped / Error
...
```

   Generate resource usage:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics report \
     --state-file "$STATE_FILE" --phase "apply-code-fixes"
   ```
   Include the output (if non-empty) at the end of the summary report.

   **Then update the HTML review report** so the decisions show up there too. This overlays every human/Claude decision recorded in `state.json` onto the existing `{plan}/code-review/report.html`, giving each finding an **Approved / Salvaged / Skipped** badge plus an expandable detail with the kept/dropped notes:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util report \
     regenerate-decisions \
     --phase-dir "$(dirname "$STATE_FILE")/code-review" \
     --state-file "$STATE_FILE" \
     --apply-phase "apply-code-fixes"
   ```
   Best-effort: if `report_data.json` is missing from the review dir (the review predates this feature), the command prints a notice and exits 0 without changing the report — just continue.

### 9. Update Plan File

Add a "Code Fixes Applied" section to the plan file:

```markdown
<!-- CODE_FIXES_APPLIED_START -->
## Code Fixes Applied

*Applied: YYYY-MM-DD HH:MM:SS*

**Summary:** N fixes applied, N skipped

### Successfully Fixed
- **Missing null check** (src/api.ts:42)
- **RLS policy missing** (migrations/xxx.sql)

### Skipped
- **Style improvement** - User skipped (LOW importance)

See: [Applied Fixes Report](./my-feature/apply-fixes/summary.md)
<!-- CODE_FIXES_APPLIED_END -->
```

### 10. Mark Phase Completed

**Run this only if step 5's batch loop reached the end** — every batch dispatched, every batch verified in step 5.5, and step 7's verification run. Then mark the apply-code-fixes phase as completed in state.json:

```bash
uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" apply-code-fixes --plan-file "$PLAN_PATH" --mark-completed
```

**Do NOT run it if the loop stopped early** — a clean failure you aborted on, a partial application you surfaced for human review (step 5.5), or an interrupted run. The phase must stay *incomplete* in those cases, because that record is what tells a later `--resume`, `--full` run or grader that the fixes were actually applied. A stopped loop simply never reaches this step; there is nothing else to do to leave the phase open.

Two notes on the word this writes:

- It records the **mode** name `apply-code-fixes` in `phases_completed`, not the phase-**directory** name `apply-fixes` that steps 5.1 and 5.7 pass to `util prior-changes`. These are two different fields with two different owners (`state.json.phases_completed` vs. `prior_changes.jsonl`'s per-entry `phase`) and both spellings are deliberate; do not "unify" them here. `util prior-changes` itself is more forgiving of the *other* field: `append`/`render` treat `apply-fixes` and `apply-code-fixes` as one phase (`utils/prior_changes.PHASE_ALIASES`), because the `--exec-subagents` executor writes `prior_changes.jsonl` entries under the mode word — so passing either spelling to `util prior-changes` lands on the same entries; step 5.7 still names `"apply-fixes"` as the one to type.
- It is the same word `--exec-subagents` records when its executor finalizes, so the native loop and the Python executor cannot drift apart. Under `--exec-subagents` the executor has already done this and there is no separate command to run.

This ensures subsequent phases (and `check_workflow_prerequisites`) recognize that the code fixes have been applied.

**Then report to the user** following `references/summary-format.md`, naming the summary report and every other file this phase wrote.

---

## JSON Output Structure

The orchestrator outputs JSON with batches:

```json
{
  "plan_file": "/path/to/plan.md",
  "prefix": "my-feature",
  "output_dir": "/path/to/my-feature/",
  "state_file": "/path/to/state.json",
  "base_ref": "abc1234",
  "timestamp": "2026-01-21T12:00:00",
  "batches": [
    {
      "fixes": [
        {
          "index": 0,
          "title": "Missing null check",
          "description": "...",
          "type": "bug",
          "file": "src/api.ts",
          "line_range": [42, 50],
          "anchor_text": "user.profile.name",
          "importance": "HIGH",
          "executor_role": "agent"
        }
      ],
      "file_key": "src/api.ts",
      "prior_changes_id": "src/api.ts#4b1d9f2a",
      "batch_type": "bug",
      "executor_role": "agent",
      "fix_count": 2,
      "priority_score": 6.0,
      "prompt": "Fix the following issues in src/api.ts..."
    }
  ],
  "to_apply": [...],
  "needs_human_review": [...],
  "batching_stats": {
    "total_fixes": 8,
    "total_batches": 4,
    "subagent_calls_saved": 4,
    "efficiency_gain_percent": 50.0,
    "subagent_distribution": {
      "general-purpose": 2,
      "general-purpose": 2
    }
  },
  "human_review_config": {
    "mode": "by-importance",
    "batch_enabled": true,
    "decision_mode": "interactive",
    "prompt_strategy": "single_batch",
    "by_importance": {
      "HIGH": [...],
      "MEDIUM": [...],
      "LOW": [...]
    },
    "total_count": 2,
    "batch_prompt_template": "batch_approval_v1"
  },
  "resume_info": {
    "previously_processed": [],
    "previous_decisions": {},
    "can_resume": false
  },
  "summary": {
    "total_issues": 12,
    "valid_count": 8,
    "needs_human_count": 2,
    "skipped_count": 2,
    "batch_count": 4,
    "validation_failed_count": 0,
    "auto_approved_count": 0
  }
}
```

---

## Example Execution

```
User: /multi-llm:multi-llm --apply-code-fixes plans/my-feature.md

Agent:
1. Checks for prerequisite files - found
2. Runs orchestrator to get batched fixes
3. Gets 8 valid fixes in 4 batches, 2 needs-human-decision

   Batch 1: 1 security fix in src/auth.ts -> general-purpose (isolated)
   Batch 2: 2 bug fixes in src/api.ts -> general-purpose
   Batch 3: 1 RLS fix in migrations/xxx.sql -> general-purpose
   Batch 4: 2 fixes in src/components/Form.tsx -> general-purpose

   Efficiency: 4 subagent calls saved (50% gain)

For each batch (sequentially):
- Runs the batch as a subagent (batch.executor_role)
- Uses pre-generated batch.prompt
- Waits for completion
- Logs results

For needs-human-decision items:
- Asks the user for a decision
- Applies or skips based on response

4. Runs verification: the repository's `driver.verify_command`, if configured
5. Generates summary report
6. Updates plan file with results
7. Marks the phase completed (every batch was applied and verified)
8. Reports completion with file paths
```

---

## Batch Prompt Format

Each batch includes a pre-generated prompt. For single-fix batches:

```
Fix the following code issue:

**Plan file**: {plan_path}
**File**: {file_path}
**Lines**: {line_start}-{line_end}
**Issue**: {title}
**Type**: {type}
**Importance**: {importance}

**Description**:
{description}

**Anchor text** (to help locate): `{anchor_text}`

## Instructions
1. If base_ref is non-empty, use `git diff {base_ref} -- {file_path}` to see recent changes (skip this step when base_ref is empty to avoid misleading unstaged diffs)
2. Read the file and locate the issue using line numbers or anchor text
3. Make the necessary fix
4. Verify your fix doesn't break anything (run typecheck if applicable)
5. Do NOT make any other changes beyond this specific fix

Return: Brief summary of what you changed.
```

For multi-fix batches:

```
Fix the following N related issues in `{file_key}`:

**Plan file**: {plan_path}
**Primary file**: {file_key}
**Batch type**: {batch_type}
**Fixes in this batch**: N

---
### Fix 1: {title}
- **Type**: {type}
- **Lines**: {line_range}
- **Importance**: {importance}
- **Anchor text**: `{anchor_text}`

**Description**:
{description}

---
### Fix 2: {title}
...

---

## Instructions

1. If base_ref is non-empty, use `git diff {base_ref} -- {file_key}` to see recent changes (skip this step when base_ref is empty to avoid misleading unstaged diffs)
2. Read the file first to understand the current state
3. Apply ALL fixes in this batch, processing them in line order
4. For each fix:
   - Locate using line numbers or anchor text
   - Make the necessary change
   - Ensure it integrates smoothly
5. Do NOT make any changes beyond these specific fixes
6. Run typecheck if applicable to verify no regressions

## Return Format

Return a brief summary for EACH fix applied:
```
Fix 1: [What was changed]
Fix 2: [What was changed]
...
```
```

---

## Critical Rules

1. **SEQUENTIAL BATCHES**: Never parallelize batch processing - one batch at a time
2. **DELEGATE EVERY BATCH**: Every batch is `executor_role: "agent"` — run it as a subagent, never inline
3. **USE PRE-GENERATED PROMPTS**: Use the `prompt` field from each batch
4. **NEVER IMPLEMENT MANUALLY**: always **spawn subagents** to do the work
5. **TRACK EVERYTHING**: Log every applied/skipped fix for the report
6. **SECURITY IS ISOLATED**: Security and HIGH importance fixes are always in their own batch
7. **VERIFY AFTER FIXES**: Run the repository's `driver.verify_command` after applying all fixes — never a command of your own, and never nothing: a repo that configured none gets `not configured` in the report, and a non-zero exit stops the phase for a human without reverting anything
8. **UPDATE PLAN**: Always add the results section to the plan file

---

## Disabling Batching

To process fixes one-at-a-time (legacy mode):

```bash
uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" apply-code-fixes --plan-file "$PLAN_PATH" --no-batch
```

This is useful for:
- Debugging issues with batched edits
- Very sensitive code where each edit needs verification
- When batching produces unexpected results

---

## Workflow Comparison

### Combined Mode (existing)
```bash
/multi-llm:multi-llm --review-code plans/my-feature.md --apply-fixes
# Reviews code AND applies fixes in one session
```

### Separate Mode (new)
```bash
# Step 1: Code review only
/multi-llm:multi-llm --review-code plans/my-feature.md

# Step 2: User manually reviews the issues in the generated report

# Step 3: Apply fixes later
/multi-llm:multi-llm --apply-code-fixes plans/my-feature.md
```

The separate mode gives users time to:
- Review issues before deciding to fix them
- Skip certain issues manually
- Apply fixes in a separate session
- Verify the validation results are accurate

---

## Edge Cases

| Edge Case | Handling |
|-----------|----------|
| No valid fixes | Exit early with message |
| All needs-human-decision | Prompt user for each |
| Lines changed since review | Use anchor_text fallback |
| Fix breaks typecheck | Report as failed, continue |
| Same file in multiple batches | Process sequentially (isolated fixes first) |
| Missing validation file | Treat all as needs-human-decision |

---

## Recovery Workflows

### When Validation Fails

If validation produces many `validation_failed` items due to parsing errors or timeouts:

```bash
# Option 1: Re-run validation with a different model
uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" apply-code-fixes \
  --plan-file "$PLAN_PATH" \
  --revalidate \
  --revalidate-model cursor-agent:opus

# Option 2: Bulk approve validation failures (if you trust the fixes)
uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" apply-code-fixes \
  --plan-file "$PLAN_PATH" \
  --approve-validation-failed
```

### Handling Revalidation Subagent

When running with `--revalidate` (without `--internal-revalidation`), the orchestrator outputs a revalidation marker.

#### The task file

`revalidation_tasks.json` carries `tasks_schema_version: 2` and one complete task
per batch under `batches` — each with `prompt`, `output_name`, the derived
`output_path`, `prompt_path` and `prompt_lines`, `result_channel: file`,
`context_mode: inline`, `capability_tier`, `executor_role` and `task_id`. The
sidecar at `prompt_path` holds exactly the bytes of `prompt` (`prompt_lines` is
its line count), so the two are interchangeable and a host that would otherwise
re-emit the whole prompt can point the subagent at the file instead. The plan
excerpt and this batch's issue records travel **inside** `prompt`;
`item_indices` and `original_validation` stay on the file for the merge step.

**For each task in the file: when the task carries `prompt_path`, run a subagent
that reads that file in full — through line `prompt_lines`, paging if a single
read stops short — and follows it verbatim; otherwise, or if the file is absent,
run `task.prompt` as a subagent. Either way, write the JSON it returns to
`task.output_path`.** That is the whole contract — do not assemble a prompt
beyond that one read instruction, and do not tell the subagent to write the file
itself. The file's bytes are exactly `task.prompt`; the two are interchangeable.

A file whose `tasks_schema_version` is missing or not `2`, or whose recorded
`inputs` hashes no longer match their sources, is refused with exit 3: re-run the
phase with `--force` to re-materialize it.

#### Single Batch: `[REVALIDATION_PENDING]`

If `[REVALIDATION_PENDING]` marker is present:
1. Read `revalidation_tasks.json` from `{plan}/code-review/`
2. Run the single task as the contract above says — the sidecar at `prompt_path`
   when it has one, `task.prompt` otherwise — and write its JSON reply to that
   task's `output_path`
3. After the result file is written, run the orchestrator again (without `--revalidate`)

#### Multiple Batches: `[REVALIDATION_BATCHES_PENDING]`

If `[REVALIDATION_BATCHES_PENDING]` marker is present:
1. Read `revalidation_tasks.json` from `{plan}/code-review/`
2. Count the number of batches and choose a strategy:

**Strategy A: Direct parallel spawning (≤ 4 batches)**

For each task in the `batches` array, run the task exactly as the contract above says — the sidecar at `prompt_path` when it has one, `task.prompt` otherwise — and write the JSON it returns to that task's `output_path`.

**Run all batches in parallel** (single message with multiple subagent calls).

**Strategy B: Wave-based parallel spawning (> 4 batches)**

Follow the wave-based spawning strategy in `references/wave-batching.md`.

3. After ALL batches complete, run the orchestrator again (without `--revalidate`) to merge and continue

**Legacy Mode**: Add `--internal-revalidation` to run revalidation inside the orchestrator instead of delegating to a subagent.

### Resuming Interrupted Sessions

If a session is interrupted, progress is saved automatically:

```bash
# Resume from where you left off
uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" apply-code-fixes \
  --plan-file "$PLAN_PATH" \
  --resume

# Or start fresh, clearing all previous progress
uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" apply-code-fixes \
  --plan-file "$PLAN_PATH" \
  --fresh
```

### Bulk Approval Examples

```bash
# Approve all LOW importance items automatically
--approve-all-low

# Approve all MEDIUM and LOW importance items
--approve-importance MEDIUM LOW

# Approve everything (DANGEROUS - requires confirmation)
--approve-all --yes

# Skip all items requiring human review
--skip-all-human

# Combine: approve validation failures, skip the rest
--approve-validation-failed --skip-all-human
```

---

## Error Types and Recovery

The orchestrator now distinguishes between different error types:

| Error Type | Status | Recoverable | Recovery Action |
|------------|--------|-------------|-----------------|
| Parsing error | `validation_failed` | Yes | Use `--revalidate` or `--approve-validation-failed` |
| Timeout | `validation_failed` | Yes | Use `--revalidate --revalidate-model faster-model` |
| Rate limited | `validation_failed` | Yes | Wait and use `--revalidate` |
| Model failure | `needs-human-decision` | No | Review manually |
| Genuine ambiguity | `needs-human-decision` | No | Review manually |

The `validation_failed` status is different from `needs-human-decision`:
- `validation_failed`: Technical error during validation, can be retried
- `needs-human-decision`: LLM determined item genuinely requires human judgment
