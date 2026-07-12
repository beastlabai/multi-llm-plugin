# Apply Suggestions Mode Instructions

Applies validated suggestions from a plan review to the original plan file, using smart batching to minimize subagent calls while maintaining edit safety.

## Usage

```bash
/multi-llm:multi-llm --apply-suggestions <plan_path> [options]
```

## Prerequisites

This mode requires a completed plan review with validation. The following files must exist in `{plan}/review-plan/`:
- `validation.json` - Validation results
- `grouped.json` - Grouped suggestions

If these files don't exist, run `--review-plan` first.

### User Selections (Auto-Detected)

If available, the following file is auto-detected from `{plan}/review-plan/`:
- `user_selections.json` - User decisions from the HTML report (skips, validation overrides, edited descriptions)

HTML selections take precedence over markdown checkbox selections.

### Consolidated Decisions (Auto-Detected)

If the review-plan phase included a consolidation step, the following files are auto-detected from `{plan}/review-plan/`:
- `consolidated-report.md` - Consolidated suggestion report
- `consolidated_user_selections.json` - User decisions on consolidated groups

No additional flags are needed — consolidation integration is automatic. Specifically:
- **C-level** (consolidated) decisions are merged with **G-level** (per-group) decisions automatically
- **C-level skips** are unioned with **G-level skips** (both sources contribute to the final skip set)
- **C-level validation overrides** cascade to all underlying groups, but **G-level overrides take precedence** (more specific wins)
- If `grouped.json` has changed since consolidation was run, the orchestrator will warn about staleness. Use `--accept-stale-consolidation` to proceed anyway.

## Smart Batching

The orchestrator now groups related suggestions into batches for efficient processing:

### Batching Heuristics

1. **Section Proximity**: Suggestions targeting the same section are grouped together
2. **Type Compatibility**:
   - Deletions are ALWAYS isolated (one per batch) to prevent conflicts
   - Additions, modifications, and clarifications can be batched together
3. **Size Limits**: Max 4 suggestions per batch, max 2500 chars total description
4. **Priority Ordering**: Batches are sorted by priority score (higher importance first)

### Efficiency Gains

With batching, 10 suggestions might become 3-4 batches:
- **Before**: 10 subagent calls (one per suggestion)
- **After**: 3-4 subagent calls (grouped by section)
- **Savings**: 60-70% reduction in API calls

## Process

1. **Load Validation Results**: Read the validation JSON to get validated suggestions
2. **Filter Suggestions**:
   - Only process suggestions with `status: "valid"`
   - Exclude suggestions marked `[x] Skip` in `report.md`
3. **Smart Batching**: Group related suggestions into batches
4. **Handle Human Decisions**: For `needs-human-decision` items, use AskUserQuestion — each interactive prompt also offers a per-item **Let Claude decide** option that offloads just that one item's judgment to a subagent. Or, if `--claude-decide` was passed, let Claude judge every item up front — or pre-marked in the report's *Let Claude decide* dropdown/checkbox, which routes just those suggestions to the judge without prompting. See step 6.
5. **Batch Application**: Process each batch with a single subagent:
   - Spawn a Task subagent with the batch prompt
   - Wait for completion before moving to the next batch
   - This prevents conflicts from concurrent edits
6. **Generate Report**: Create `{prefix}_applied_suggestions.md` with results

## Output Files

```
plans/todo/my-feature/
├── state.json                           # Session state (plan-local)
├── review-plan/                         # Input files from review phase
│   ├── grouped.json
│   ├── validation.json
│   └── backup.md
└── apply-suggestions/                   # Output files from this phase
    ├── summary.md                       # Summary of applied changes
    └── results.json                     # Detailed application results
```

## Options

### Basic Options
- `--dry-run`: Show what batches would be processed without making changes
- `--min-priority {low,medium,high}`: Minimum importance level to include (default: low = all valid suggestions)
- `--include-low`: [DEPRECATED] No longer needed, low is now default. Use `--min-priority` to filter.
- `--no-batch`: Disable smart batching (process each suggestion separately)
- `--max-batch-size N`: Maximum suggestions per batch (default: 4)

### Confirmation Behavior
- `--no-confirm`: Skip confirmation when no user selections are found (for unattended/silent operation)
- Also bypassed by: `--yes`, `--force`, `--approve-all`, `--skip-all-human`, `--approve-all-low`, `--approve-importance`, `--approve-validation-failed`, `--dry-run`

### Bulk Approval Options

These options reduce manual review for items with `needs-human-decision` status:

- `--claude-decide` (alias `--let-claude-decide`): Let Claude evaluate each `needs-human-decision` item with its own judgment instead of prompting you. This is the non-interactive equivalent of choosing "Let Claude decide" in the review prompt. Unlike the blanket flags below, it is a **per-item** judgment that **salvages partially-valid items** (trims them to their worthwhile core and applies that) rather than skipping them wholesale; only items with nothing worthwhile are skipped (see "Let Claude Decide" Mode in `references/human-decision-batch.md`). It does **not** bypass the no-selection confirmation for valid items — combine with `--no-confirm` for fully unattended runs.
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

**Note**: By default, revalidation is delegated to a Claude Code subagent. Use `--internal-revalidation` to run revalidation inside the orchestrator (legacy mode).

### Resume/State Options

- `--resume`: Resume from last checkpoint, skip already-processed items
- `--fresh`: Clear previous progress and start fresh
- `--skip-human-review`: [DEPRECATED] Use `--skip-all-human` instead

### Batch Review Mode

- `--batch-review-mode {individual,by-importance,summary-only}`: How to present items for human review (default: by-importance)

### User Skip Filtering

If users have marked suggestions to skip in `report.md`, the orchestrator will:
1. Parse `{plan}/review-plan/report.md` for `[x] Skip` markers
2. Filter out matching suggestions before processing
3. Report: `User skipped N suggestions: S001, S003, ...`

This happens automatically - no additional flags needed.

---

## Step-by-Step Execution

### 1. Validate Prerequisites

Check that required files exist:
```
{plan}/review-plan/validation.json
{plan}/review-plan/grouped.json
```

If not found, inform user to run `--review-plan` first.

### 2. Run the Orchestrator

```bash
uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_suggestions_orchestrator.py" --plan-file "$PLAN_PATH" [options]
```

**IMPORTANT**: Pass `$PLAN_PATH` as given — the orchestrator resolves it to an OS-native absolute path itself. Do NOT wrap it in `$(realpath ...)`: on Git for Windows, `realpath` emits a POSIX `/c/...` path that a native Windows Python process cannot use.

**IMPORTANT**: Run this command in the FOREGROUND (do NOT use `run_in_background`). Use `timeout: 600000` (10 minutes — the Bash tool caps `timeout` at 600000 ms; larger values are silently clamped). The orchestrator runs quickly — it only generates batch JSON, it does not apply changes — so 10 minutes is far more than enough.

The orchestrator will:
1. Load validation results and grouped suggestions
2. Filter to only `valid` suggestions
3. Group suggestions into efficient batches
4. Write output to `{plan}/apply-suggestions/orchestrator_output.json`
5. Exit - Claude Code handles the actual application

### 2a. Handle No-Selections Confirmation

After the orchestrator completes, check if its stdout contains `"status": "confirmation_needed"`.

If present:
1. Display the warning message to the user
2. Use AskUserQuestion:
   > "No user selections were found (no user_selections.json, consolidated_user_selections.json, or report.md edits detected). This will apply ALL {item_count} valid items. Continue?"
   > Options: "Yes, apply all" / "No, I need to make my selections first"
3. If user confirms: Re-run the orchestrator with `--no-confirm` appended
4. If user declines: Stop and inform the user to review the HTML report and export selections

If NOT present, proceed to step 3.

### 3. Display Decision Summary

After the orchestrator completes, run the display script to show the user which suggestions will be applied, skipped, need review, etc.:

```bash
uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/display_decisions.py" --plan-file "$PLAN_PATH" --phase apply-suggestions
```

This prints a formatted summary with each suggestion's title and status. Display this output to the user before proceeding.

### 4. Read the Orchestrator Output

Use the **Read tool** to load `{plan}/apply-suggestions/orchestrator_output.json`.

Extract the following key fields directly from the Read output (do NOT use Bash/python to parse — just read the JSON and understand it):
- `batches`: Array of batch objects, each with `prompt`, `section_key`, `suggestions`
- `needs_human_review`: Items requiring human decision
- `human_review_config`: Batch mode config for human review
- `summary`: Counts of valid, skipped, batches, etc.

### 5. Process Batches Sequentially

Use the batch data already loaded from the Read tool in step 4. Do NOT use Bash/python to re-parse or extract fields from the JSON file.

**Before the loop**, mark the phase start so ETA tracking has a wall-clock baseline. Run:
```bash
uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/metrics.py" start \
  --state-file "$STATE_FILE" --phase "apply-suggestions" --total-batches {total_batches}
```
Where `{total_batches}` is the length of the `batches` array.

For each batch (index N, starting at 1) returned by the orchestrator, Claude MUST:

1. **Read prior changes** — Run:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/prior_changes.py" read \
     --output-dir "{plan}/apply-suggestions"
   ```
   Capture stdout as `prior_changes_text`.

   **Error handling**: If `read` fails (corrupted JSONL, permission error, unexpected exception), use the default text `(none — this is the first batch)` and log a warning. Do not abort the batch loop.

2. **Substitute placeholder** — In `batch.prompt`, replace the literal `{prior_changes_context}` with `prior_changes_text`.

3. **Read current plan state and snapshot it** — Always read the latest plan file version before each batch, then capture a pre-batch snapshot so the batch's effect can be verified in step 6 **without** relying on git. (Plan files are frequently untracked — freshly created — and every `git diff` variant silently ignores fully-untracked files, so git CANNOT detect whether a batch changed the plan. Snapshot the file and diff against the snapshot instead.)

   The snapshot lives in the **workspace temp dir** at a deterministic path (each batch overwrites it; no cleanup step; the dir is self-gitignored). Resolve the project root first — if the command fails or prints nothing, STOP with the error: "multi-llm requires running inside a git repository":
   ```bash
   git rev-parse --show-toplevel
   ```
   The snapshot path is `{project_root}/.multi-llm/tmp/batch_snapshot_plan_{plan_stem}.md`, where `{project_root}` is the output of the command above and `{plan_stem}` is the plan filename without extension.

   Also resolve the **plan file's absolute path** now — call it `{plan_abs}`. The plan argument is commonly relative (e.g. `plans/my-feature.md`), but harness Read/Write tools require OS-native absolute paths, so a relative path cannot be handed to them directly. If the plan path is already absolute, use it as-is; otherwise join `{project_root}` with the relative plan path (e.g. `{project_root}/plans/my-feature.md`).

   **Shell variables do NOT survive between Bash calls** — each Bash invocation is a fresh process, so do not stash these values in variables like `PROJECT_ROOT` or `BATCH_SNAPSHOT` for later steps. Note the concrete absolute snapshot path and `{plan_abs}` now and substitute them literally wherever later steps reference them.

   Capture the snapshot with harness tools, not Bash: **Read** `{plan_abs}` with the Read tool, then **Write** its exact content to the snapshot path with the Write tool (use the absolute path — join the resolved project root with `.multi-llm/tmp/batch_snapshot_plan_{plan_stem}.md`; the Write tool creates parent directories automatically, so no mkdir step and no Bash copy command is needed). If `{project_root}/.multi-llm/tmp/.gitignore` does not exist yet, also Write it with content `*` so the temp dir ignores itself.

   Keep the concrete absolute snapshot path and `{plan_abs}` for step 6.

4. **Spawn a Task subagent** using the substituted prompt
   - Use `general-purpose` subagent type
   - The prompt already contains all suggestions and instructions
5. **Wait for completion** - Do NOT parallelize batch processing

6. **Verify batch outcome** — After the subagent completes (or fails), determine the batch outcome by inspecting actual file state rather than relying solely on the subagent's exit status.

   **Do NOT use `git diff` / `git diff --stat` to detect whether the plan changed.** Plan files are frequently untracked (freshly created), and every `git diff` variant silently ignores fully-untracked files — it will report "no changes" even when the subagent edited the plan, producing a false clean-failure. Detect changes by diffing against the step-3 snapshot instead:
   ```bash
   diff -u "{snapshot_path}" "{plan_path}"   # empty output = byte-identical (unchanged); non-empty = file changed
   ```
   Substitute `{snapshot_path}` and `{plan_path}` with the **concrete absolute paths** noted in step 3 (the snapshot path and `{plan_abs}` — the same resolved absolute plan path used for the step-3 Read, not the possibly-relative plan argument). Do NOT write `$BATCH_SNAPSHOT` or `$PLAN_PATH` here — shell variables assigned in earlier Bash calls do not exist in this fresh Bash process, and an unset variable would make `diff` compare the wrong files or fail. If the snapshot path was not carried forward, recompute and diff in one single invocation, embedding the root as an inline command substitution (not as a `PROJECT_ROOT=...` assignment prefix, which the permission allowlist does not cover):
   ```bash
   diff -u "$(git rev-parse --show-toplevel)/.multi-llm/tmp/batch_snapshot_plan_{plan_stem}.md" "{plan_path}"
   ```
   Classify the result:
   - **Success**: The snapshot diff is non-empty AND the expected changes are present (spot-check by `grep`-ing the plan for distinctive text from this batch's suggestions).
   - **Failure (clean)**: The subagent reported failure AND the snapshot diff is empty (plan byte-identical to the pre-batch snapshot).
   - **Partial application**: The snapshot diff shows some but not all expected changes, or the subagent failed but the snapshot diff is non-empty.

   The snapshot diff is the authoritative, fully git-independent check — no git intent-to-add step is needed here (flows that genuinely need untracked files visible to git diffs, like code review, handle that inside their Python orchestrator via `utils/git_utils.intent_to_add_untracked`).

   There is **no snapshot cleanup step**: the snapshot lives in the self-gitignored workspace temp dir and is simply overwritten by the next batch. On **partial application** the loop stops (below), so the snapshot file still holds the pre-batch content for the human reviewer to inspect.

   **Partial application handling**: If verification detects a partial application, **stop the batch loop and surface the issue for human review**. Do not continue to subsequent batches. Log the partial application details and prompt the human operator to resolve the state before resuming.

7. **Extract summary** — Generate a one-line summary by compressing the subagent's response into a single sentence. Do **not** require the subagent to output a special summary line.

   **Normalize** the summary: collapse any newlines or multiple spaces to a single space, trim leading/trailing whitespace, and truncate to 200 characters if needed.

8. **Append to prior changes (only after verification)** — This step runs **only** after step 6 has confirmed success or clean failure. It must **never** run for partially-applied batches. Pass the summary via stdin:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/prior_changes.py" append \
     --output-dir "{plan}/apply-suggestions" \
     --id "{suggestion_id}" --phase "apply-suggestions" --summary-stdin <<'SUMMARY_EOF'
   {normalized_summary}
   SUMMARY_EOF
   ```
   The `--id` must be a **stable identifier** for this batch's content (the suggestion ID, group key, or section key). Do **not** use the batch loop index.

   For **clean failures**, use summary: `(batch failed — {compact reason}, verified no file changes)`

   **Error handling**: If `append` fails, log a warning but do not abort the batch loop.

9. **Log the result** - Track success/failure for the summary
10. **Record metrics** — Run:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/metrics.py" record \
     --state-file "$STATE_FILE" --phase "apply-suggestions" \
     --label "Batch {N} ({section_key})" --subagent-type "general-purpose" \
     --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
     --total-batches {total_batches} --batch-index {N}
   ```
   Where `token_count`, `tool_uses`, `duration_ms` come from the Task tool result. Omit any flags the Task tool didn't return. The script prints an `[ETA]` line to stderr after each call.

**CRITICAL**: Process batches ONE AT A TIME. Never spawn multiple Task subagents simultaneously.

**After the loop completes**, mark the phase finish:
```bash
uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/metrics.py" finish \
  --state-file "$STATE_FILE" --phase "apply-suggestions"
```

### 6. Handle Human Decision Items

Follow the human decision batch mode process in `references/human-decision-batch.md`.

**Auto-route report-pre-marked items first**: before prompting for anything, partition `needs_human_review` into (a) the **pre-marked Claude set** — every suggestion whose `group_id` is in `human_review_config.claude_decide_item_ids` or that carries `decision_mode == "claude_auto_decide"` — and (b) the remaining **interactive set**. Run **"Let Claude Decide" Mode** (per the criteria in `references/human-decision-batch.md`) on the pre-marked Claude set **without any prompt**, then handle only the interactive set with the flow below. These suggestions were chosen in the report's "Let Claude decide" dropdown/checkbox; they are the per-item equivalent of `--claude-decide` scoped to just those suggestions.

**Check `human_review_config.decision_mode` first**: if it is `"claude_auto_decide"` (the user passed `--claude-decide`), do NOT prompt the user — go straight to "Let Claude Decide" Mode and evaluate every `needs-human-decision` item autonomously per the criteria in that reference. By default, **partially-valid items are salvaged** (rewritten to their worthwhile core and applied) rather than skipped; only items with nothing worthwhile are skipped entirely.

Context line for individual review display: `Type: {type}, Section: {reference}`

### 7. Generate Summary Report

After all batches are processed, create `{prefix}_applied_suggestions.md`:

```markdown
# Applied Suggestions Report

**Plan**: /path/to/plan.md
**Generated**: YYYY-MM-DD HH:MM:SS

## Summary
- Batches processed: N
- Suggestions applied: N
- Suggestions skipped: N
- Human decisions made: N
- Decided by Claude: N (A approved, V salvaged, S skipped) — includes `--claude-decide`, the batch "Let Claude decide" option, and per-item "Let Claude decide" selections

## Batching Efficiency
- Total suggestions: N
- Total batches: N
- Subagent calls saved: N (X% efficiency gain)

## Applied Changes

### Batch 1 (Section: step_3)
**Suggestions**: 3
**Type**: addition

1. **[Suggestion Title]** - Applied
   Description: [Brief description of what was changed]

2. **[Suggestion Title]** - Applied
   ...

### Batch 2 (Section: step_4)
...

## Decided by Claude

Items whose decision was offloaded to Claude — via `--claude-decide`, the batch "Let Claude decide" option, or a per-item "Let Claude decide" selection. Read these from `state.json`: the `human_decisions_{phase}` entries whose `batch_context.decision_source` is `claude_auto_decide` or `claude_auto_decide_salvage`. Omit this section if there are none.

### [Suggestion Title] ({importance}) — Approved
**Reason**: {reason}

### [Suggestion Title] ({importance}) — Salvaged
**Kept**: {salvaged_description}
**Dropped**: {dropped}

### [Suggestion Title] ({importance}) — Skipped
**Reason**: {reason}

## Skipped Suggestions

### [Suggestion Title]
**Reason**: Invalid / User skipped / Error
...
```

   Generate resource usage:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/metrics.py" report \
     --state-file "$STATE_FILE" --phase "apply-suggestions"
   ```
   Include the output (if non-empty) at the end of the summary report.

   **Then update the HTML review report** so the decisions show up there too. This overlays every human/Claude decision recorded in `state.json` onto the existing `{plan}/review-plan/report.html`, giving each finding an **Approved / Salvaged / Skipped** badge plus an expandable detail with the kept/dropped notes:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/html_report_generator.py" \
     regenerate-decisions \
     --phase-dir "$(dirname "$STATE_FILE")/review-plan" \
     --state-file "$STATE_FILE" \
     --apply-phase "apply-suggestions"
   ```
   Best-effort: if `report_data.json` is missing from the review dir (the review predates this feature), the command prints a notice and exits 0 without changing the report — just continue.

---

## JSON Output Structure

The orchestrator outputs JSON with batches:

```json
{
  "plan_file": "/path/to/plan.md",
  "prefix": "my-feature",
  "output_dir": "/path/to/my-feature/",
  "timestamp": "2024-01-21T12:00:00",
  "batches": [
    {
      "suggestions": [
        {
          "index": 0,
          "title": "Add error handling",
          "description": "...",
          "type": "addition",
          "reference": "### Step 3",
          "importance": "HIGH"
        },
        {
          "index": 1,
          "title": "Add validation",
          "description": "...",
          "type": "addition",
          "reference": "### Step 3",
          "importance": "MEDIUM"
        }
      ],
      "section_key": "step_3",
      "batch_type": "addition",
      "suggestion_count": 2,
      "priority_score": 5.0,
      "prompt": "Apply the following 2 related suggestions..."
    }
  ],
  "to_apply": [...],  // Flat list for reference
  "needs_human_review": [...],
  "batching_stats": {
    "total_suggestions": 8,
    "total_batches": 3,
    "subagent_calls_saved": 5,
    "efficiency_gain_percent": 62.5,
    "batching_enabled": true
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
    "total_groups": 10,
    "valid_count": 8,
    "needs_human_count": 2,
    "skipped_count": 0,
    "batch_count": 3,
    "validation_failed_count": 0,
    "auto_approved_count": 0
  }
}
```

---

## Example Execution

```
User: /multi-llm:multi-llm --apply-suggestions plans/my-feature.md

Claude:
1. Checks for prerequisite files - found
2. Runs orchestrator to get batched suggestions
3. Gets 8 valid suggestions in 3 batches, 2 needs-human-decision

   Batch 1: 3 additions in step_3 (priority: 7.0)
   Batch 2: 2 modifications in step_4 (priority: 5.0)
   Batch 3: 3 additions in step_5 (priority: 4.0)

   Efficiency: 5 subagent calls saved (62.5% gain)

For each batch (sequentially):
- Reads current plan
- Spawns Task subagent with batch.prompt
- Waits for completion
- Logs results

For needs-human-decision items:
- Uses AskUserQuestion to get user decision
- Applies or skips based on response

4. Generates summary report
5. Reports completion with file paths
```

---

## Batch Prompt Format

Each batch includes a pre-generated prompt. For single-suggestion batches:

```
Apply the following suggestion to the plan file:

**Plan file**: {plan_path}
**Suggestion**: {title}
**Type**: {type}
**Section**: {reference}
**Importance**: {importance}

**Details**:
{description}

Instructions:
1. Read the current plan file
2. Locate the section mentioned in the reference
3. Apply the suggested change appropriately
4. Ensure the change integrates smoothly
5. Do NOT make any other changes beyond this suggestion

Return a brief summary of what was changed.
```

For multi-suggestion batches:

```
Apply the following N related suggestions to the plan file (Section: step_3):

**Plan file**: {plan_path}
**Batch type**: addition
**Suggestions in this batch**: N

---
### Suggestion 1: {title}
- **Type**: addition
- **Section**: ### Step 3
- **Importance**: HIGH

**Details**:
{description}

---
### Suggestion 2: {title}
...

---

## Instructions

1. Read the current plan file first
2. Apply ALL suggestions in this batch, processing them in order
3. For each suggestion:
   - Locate the referenced section
   - Apply the change according to its type
   - Ensure changes integrate smoothly
4. Do NOT make any changes beyond these specific suggestions
5. If suggestions affect the same area, apply them intelligently
6. Before applying each suggestion, check if the plan already reflects a similar change (from a previously applied batch). If a suggestion is already addressed, note it as "already applied" in the summary and move on

## Return Format

Return a brief summary for EACH suggestion applied:
```
Suggestion 1: [What was changed]
Suggestion 2: [What was changed]
Suggestion N: Already addressed by prior changes - skipped
...
```
```

---

## Critical Rules

1. **SEQUENTIAL BATCHES**: Never parallelize batch processing - one batch at a time
2. **ALWAYS READ FIRST**: Read the plan before each batch to get latest state
3. **USE PRE-GENERATED PROMPTS**: Use the `prompt` field from each batch
4. **USE SUBAGENTS**: Always use Task tool with `general-purpose` type
5. **PRESERVE BACKUP**: The original plan backup from review-plan should not be overwritten
6. **TRACK EVERYTHING**: Log every applied/skipped suggestion for the report
7. **DELETIONS ARE ISOLATED**: Deletion suggestions are always in their own batch

---

## Disabling Batching

To process suggestions one-at-a-time (legacy mode):

```bash
uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_suggestions_orchestrator.py" --plan-file "$PLAN_PATH" --no-batch
```

This is useful for:
- Debugging issues with batched edits
- Very sensitive plans where each edit needs verification
- When batching produces unexpected results

---

## Recovery Workflows

### When Validation Fails

If validation produces many `validation_failed` items due to parsing errors or timeouts:

```bash
# Option 1: Re-run validation with a different model
uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_suggestions_orchestrator.py" \
  --plan-file "$PLAN_PATH" \
  --revalidate \
  --revalidate-model cursor-agent:opus

# Option 2: Bulk approve validation failures (if you trust the suggestions)
uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_suggestions_orchestrator.py" \
  --plan-file "$PLAN_PATH" \
  --approve-validation-failed
```

### Handling Revalidation Subagent

When running with `--revalidate` (without `--internal-revalidation`), the orchestrator outputs a revalidation marker.

#### Reference-Based Revalidation

The `revalidation_tasks.json` file uses a reference-based format (small metadata file):
```json
{
  "batches": [
    {
      "batch_index": 0,
      "group_indices": [0, 2, 5],
      "output_path": "{plan}/review-plan/validation_batch_0.json"
    }
  ],
  "grouped_file": "{plan}/review-plan/grouped.json",
  "plan_file": "/absolute/path/to/plan.md",
  "item_indices": [0, 2, 5],
  "reaggregate_command": "uv run ... apply_suggestions_orchestrator.py ..."
}
```

#### Single Batch: `[REVALIDATION_PENDING]`

If `[REVALIDATION_PENDING]` marker is present:
1. Read `revalidation_tasks.json` from `{plan}/review-plan/` (small metadata file)
2. Spawn a single revalidation subagent that reads source files directly:

   ```
   Task tool call:
     subagent_type: general-purpose
     description: "Revalidate failed suggestions"
     prompt: |
       Revalidate suggestion groups that previously failed validation.

       Read these files:
       - Grouped suggestions: {grouped_file from revalidation_tasks.json}
       - Plan context: {plan_file from revalidation_tasks.json}

       Revalidate ONLY these groups (use group_hash from grouped.json to identify each):
       {for each i: "  index {group_indices[i]} (group_hash: {group_ids[i]})" from batches[N]}
       You MUST output EXACTLY {len(group_indices)} results — one per requested group.
       Do NOT revalidate or include results for any other groups.

       For each group, determine:
       - "valid": Issue is real, should be addressed
       - "invalid": False positive, not applicable
       - "needs-human-decision": Requires human judgment

       Use the Write tool (not Bash) to save results to: {output_path from batches[N]}

       Output format:
       {
         "groups": [
           {"group_index": 0, "group_hash": "<copy from group's group_hash field>", "status": "valid|invalid|needs-human-decision", "reason": "...", "confidence": 0.0-1.0}
         ],
         "metadata": {"model": "claude", "timestamp": "ISO", "schema_version": "2.1"}
       }

       IMPORTANT: Copy the exact group_hash value from each group in grouped.json into your output.

       CRITICAL: Your entire response MUST be a single brief line, e.g.:
       "Done: wrote 3 revalidation results to /path/to/output.json"
       Do NOT include reasoning, analysis, or explanation in your response.
   ```

3. After the subagent writes the result file, run the orchestrator again (without `--revalidate`)

#### Multiple Batches: `[REVALIDATION_BATCHES_PENDING]`

If `[REVALIDATION_BATCHES_PENDING]` marker is present:
1. Read `revalidation_tasks.json` from `{plan}/review-plan/`
2. Count the number of batches and choose a strategy:

**Strategy A: Direct parallel spawning (≤ 4 batches)**

For each batch in `batches` array, spawn a Task agent using the same prompt template above, but with that batch's `group_indices` and `output_path`.

**Run all batches in parallel** (single message with multiple Task tool calls).

**Strategy B: Wave-based parallel spawning (> 4 batches)**

Follow the wave-based spawning strategy in `references/wave-batching.md`.

3. After ALL batches complete, run the orchestrator again (without `--revalidate`) to merge and continue

**Legacy Mode**: Add `--internal-revalidation` to run revalidation inside the orchestrator instead of delegating to a subagent.

### Resuming Interrupted Sessions

If a session is interrupted, progress is saved automatically:

```bash
# Resume from where you left off
uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_suggestions_orchestrator.py" \
  --plan-file "$PLAN_PATH" \
  --resume

# Or start fresh, clearing all previous progress
uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_suggestions_orchestrator.py" \
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
