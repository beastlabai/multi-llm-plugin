# Apply Task Suggestions Mode Instructions

Applies validated suggestions from a task review to the tasks.md file, using smart batching to minimize subagent calls while maintaining edit safety.

## Usage

```bash
/multi-llm:multi-llm --apply-task-suggestions <plan_path> [options]
```

## Prerequisites

This mode requires a completed task review with validation. The following files must exist in `{plan}/review-tasks/`:
- `validation.json` - Validation results
- `grouped.json` - Grouped suggestions

If these files don't exist, run `--review-tasks` first.

### User Selections (Auto-Detected)

If available, the following file is auto-detected from `{plan}/review-tasks/`:
- `user_selections.json` - User decisions from the HTML report (skips, validation overrides, edited descriptions)

HTML selections take precedence over markdown checkbox selections.

### Consolidated Decisions (Auto-Detected)

If the review-tasks phase included a consolidation step, the following files are auto-detected from `{plan}/review-tasks/`:
- `consolidated-report.md` - Consolidated suggestion report
- `consolidated_user_selections.json` - User decisions on consolidated groups

No additional flags are needed — consolidation integration is automatic. Specifically:
- **C-level** (consolidated) decisions are merged with **G-level** (per-group) decisions automatically
- **C-level skips** are unioned with **G-level skips** (both sources contribute to the final skip set)
- **C-level validation overrides** cascade to all underlying groups, but **G-level overrides take precedence** (more specific wins)
- If `grouped.json` has changed since consolidation was run, the orchestrator will warn about staleness. Use `--accept-stale-consolidation` to proceed anyway.

## Smart Batching

The orchestrator groups related task suggestions into batches for efficient processing:

### Batching Heuristics

1. **Task Reference Proximity**: Suggestions targeting the same task (e.g., "Task T001") are grouped together
2. **Type Compatibility**:
   - Deletions are ALWAYS isolated (one per batch) to prevent conflicts
   - Additions, modifications, and clarifications can be batched together
3. **Size Limits**: Max 4 suggestions per batch, max 2500 chars total description
4. **Priority Ordering**: Batches are sorted by priority score (higher importance first)

### Efficiency Gains

With batching, 10 suggestions might become 3-4 batches:
- **Before**: 10 subagent calls (one per suggestion)
- **After**: 3-4 subagent calls (grouped by task reference)
- **Savings**: 60-70% reduction in API calls

## Process

1. **Load Validation Results**: Read the validation JSON to get validated suggestions
2. **Filter Suggestions**:
   - Only process suggestions with `status: "valid"`
   - Exclude suggestions marked `[x] Skip` in `report.md`
3. **Smart Batching**: Group related suggestions into batches by task reference
4. **Handle Human Decisions**: For `needs-human-decision` items, use AskUserQuestion — each interactive prompt also offers a per-item **Let Claude decide** option that offloads just that one suggestion's judgment to a subagent. Or, if `--claude-decide` was passed, let Claude judge every suggestion up front. See step 6.
5. **Batch Application**: Process each batch with a single subagent:
   - Spawn a Task subagent with the batch prompt
   - Wait for completion before moving to the next batch
   - This prevents conflicts from concurrent edits to tasks.md
6. **Generate Report**: Create `{prefix}_applied_task_suggestions.md` with results

## Output Files

```
plans/todo/my-feature/
├── state.json                           # Session state (plan-local)
├── review-tasks/                        # Input files from review phase
│   ├── grouped.json
│   ├── validation.json
│   └── backup.md
└── apply-task-suggestions/              # Output files from this phase
    ├── orchestrator_output.json         # Batch instructions for Claude Code
    ├── tasks-backup.md                  # Pre-modification backup of tasks.md
    └── summary.md                       # Summary of applied changes
```

## Options

### Basic Options
- `--dry-run`: Show what batches would be processed without making changes
- `--min-priority {low,medium,high}`: Minimum importance level to include (default: low = all valid suggestions)
- `--no-batch`: Disable smart batching (process each suggestion separately)
- `--max-batch-size N`: Maximum suggestions per batch (default: 4)

### Confirmation Behavior
- `--no-confirm`: Skip confirmation when no user selections are found (for unattended/silent operation)
- Also bypassed by: `--yes`, `--force`, `--approve-all`, `--skip-all-human`, `--approve-all-low`, `--approve-importance`, `--dry-run`

### Bulk Approval Options

These options reduce manual review for items with `needs-human-decision` status:

- `--claude-decide` (alias `--let-claude-decide`): Let Claude evaluate each `needs-human-decision` suggestion with its own judgment instead of prompting you. This is the non-interactive equivalent of choosing "Let Claude decide" in the review prompt. Unlike the blanket flags below, it is a **per-item** judgment that **salvages partially-valid suggestions** (trims them to their worthwhile core and applies that) rather than skipping them wholesale; only suggestions with nothing worthwhile are skipped (see "Let Claude Decide" Mode in `references/human-decision-batch.md`). It does **not** bypass the no-selection confirmation for valid suggestions — combine with `--no-confirm` for fully unattended runs.
- `--approve-all-low`: Auto-approve all LOW importance items
- `--approve-all`: Auto-approve ALL items (requires `--yes` or `--force`)
- `--skip-all-human`: Skip all `needs-human-decision` and `validation_failed` items
- `--approve-importance LOW MEDIUM`: Auto-approve items at specified importance levels

### Safety Guardrails

- `--yes` / `-y`: Confirm bulk approval operations (required with `--approve-all`)
- `--force` / `-f`: Alias for `--yes`
- `--include-high`: Allow `--approve-all` to include HIGH importance items

### Resume/State Options

- `--resume`: Resume from last checkpoint, skip already-processed items
- `--fresh`: Clear previous progress and start fresh

### Batch Review Mode

- `--batch-review-mode {individual,by-importance,summary-only}`: How to present items for human review (default: by-importance)

### User Skip Filtering

If users have marked suggestions to skip in `report.md`, the orchestrator will:
1. Parse `{plan}/review-tasks/report.md` for `[x] Skip` markers
2. Filter out matching suggestions before processing
3. Report: `User skipped N suggestions: S001, S003, ...`

This happens automatically - no additional flags needed.

---

## Step-by-Step Execution

### 1. Validate Prerequisites

Check that required files exist:
```
{plan}/review-tasks/validation.json
{plan}/review-tasks/grouped.json
```

If not found, inform user to run `--review-tasks` first.

### 2. Run the Orchestrator

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" [options]
```

**IMPORTANT**: Always use `$(realpath "$PLAN_PATH")` to convert to absolute path.

**IMPORTANT**: Run this command in the FOREGROUND (do NOT use `run_in_background`). Use `timeout: 1200000` (20 minutes). The orchestrator runs quickly — it only generates batch JSON, it does not apply changes.

The orchestrator will:
1. Load validation results and grouped suggestions
2. Filter to only `valid` suggestions
3. Group suggestions into efficient batches
4. Write output to `{plan}/apply-task-suggestions/orchestrator_output.json`
5. Exit - Claude Code handles the actual application

#### Zero-Finding and Zero-Selection Paths

The orchestrator handles four distinct no-op/early-exit scenarios:

1. **No valid findings** (`review-tasks/grouped.json` does not exist or contains zero suggestions): The orchestrator writes an empty `orchestrator_output.json` (with `batches: []`), marks the phase as **skipped** via `state_manager.mark_phase_skipped("apply-task-suggestions", reason="no actionable findings from review-tasks")`, calls `state.save()`, prints a user-facing message (`"No actionable task suggestions found — skipping apply-task-suggestions phase."`), and exits with code 0. No summary report is generated. `--status` shows the phase as `"skipped (no actionable findings from review-tasks)"`.

2. **All items user-skipped** (findings exist but every suggestion is marked as skipped in `user_selections.json` and/or `report.md`): The orchestrator writes an empty `orchestrator_output.json` (with `batches: []`), marks the phase as **skipped** via `state_manager.mark_phase_skipped("apply-task-suggestions", reason="all suggestions skipped by user")`, calls `state.save()`, prints `"All task suggestions were skipped by user — no changes applied."`, and exits with code 0. A minimal summary report is still generated (documenting what was skipped and why), so the user has an audit trail. `--status` shows `"skipped (all suggestions skipped by user)"`.

3. **No selection file exists** (`review-tasks/user_selections.json` missing AND no skip/approve marks in `report.md`): This is NOT a no-op — it means the user did not make any explicit selections. The orchestrator proceeds normally, treating all suggestions as approved (the default behavior, matching `apply-suggestions` semantics). However, unless `--no-confirm` or `--yes` is passed, the orchestrator triggers the no-selections confirmation prompt (step 2a) asking the user to confirm they want to apply all suggestions. If the user declines at the prompt, the phase is marked **skipped** via `state_manager.mark_phase_skipped("apply-task-suggestions", reason="user declined at no-selections prompt")`, `state.save()` is called, and the orchestrator exits with code 0.

4. **Suggestions exist but all are filtered out by CLI flags** (e.g., `--min-priority` filters everything): Same behavior as case 2 — write empty output, mark phase **skipped** with reason `"all suggestions filtered out by CLI flags"`, generate minimal summary report, exit 0. `--status` shows `"skipped (all suggestions filtered out by CLI flags)"`.

### 2a. Handle No-Selections Confirmation

After the orchestrator completes, check if its stdout contains `"status": "confirmation_needed"`.

If present:
1. Display the warning message to the user
2. Use AskUserQuestion:
   > "No user selections were found (no user_selections.json, consolidated_user_selections.json, or report.md edits detected). This will apply ALL {item_count} valid task suggestions. Continue?"
   > Options: "Yes, apply all" / "No, I need to make my selections first"
3. If user confirms: Re-run the orchestrator with `--no-confirm` appended
4. If user declines: Stop and inform the user to review the HTML report and export selections

If NOT present, proceed to step 3.

### 3. Display Decision Summary

After the orchestrator completes, run the display script to show the user which suggestions will be applied, skipped, need review, etc.:

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/display_decisions.py --plan-file "$(realpath "$PLAN_PATH")" --phase apply-task-suggestions
```

This prints a formatted summary with each suggestion's title and status. Display this output to the user before proceeding.

### 4. Read the Orchestrator Output

Use the **Read tool** to load `{plan}/apply-task-suggestions/orchestrator_output.json`.

Extract the following key fields directly from the Read output (do NOT use Bash/python to parse — just read the JSON and understand it):
- `batches`: Array of batch objects, each with `prompt`, `section_key`, `suggestions`
- `needs_human_review`: Items requiring human decision
- `human_review_config`: Batch mode config for human review
- `summary`: Counts of valid, skipped, batches, etc.

### 5. Process Batches Sequentially

Use the batch data already loaded from the Read tool in step 4. Do NOT use Bash/python to re-parse or extract fields from the JSON file.

**Before the loop**, mark the phase start so ETA tracking has a wall-clock baseline. Run:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py start \
  --state-file "$STATE_FILE" --phase "apply-task-suggestions" --total-batches {total_batches}
```
Where `{total_batches}` is the length of the `batches` array.

For each batch (index N, starting at 1) returned by the orchestrator, Claude MUST:

1. **Read prior changes** — Run:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/prior_changes.py read \
     --output-dir "{plan}/apply-task-suggestions"
   ```
   Capture stdout as `prior_changes_text`.

   **Error handling**: If `read` fails (corrupted JSONL, permission error, unexpected exception), use the default text `(none — this is the first batch)` and log a warning. Do not abort the batch loop.

2. **Substitute placeholder** — In `batch.prompt`, replace the literal `{prior_changes_context}` with `prior_changes_text`.

3. **Read current tasks.md state** — Always read the latest version of tasks.md before each batch.

4. **Spawn a Task subagent** using the substituted prompt
   - Use `general-purpose` subagent type
   - The prompt already contains all suggestions and instructions
5. **Wait for completion** - Do NOT parallelize batch processing

6. **Verify batch outcome** — After the subagent completes (or fails), determine the batch outcome by inspecting actual file/diff state rather than relying solely on the subagent's exit status. Classify the result:
   - **Success**: The expected changes are present in the target files (verified via diff or content check).
   - **Failure (clean)**: The subagent failed and no file modifications were made (verified by checking that target files are unchanged from their pre-batch state).
   - **Partial application**: Some but not all expected changes are present, or the subagent failed but file modifications were detected.

   **Partial application handling**: If verification detects a partial application, **stop the batch loop and surface the issue for human review**. Do not continue to subsequent batches. Log the partial application details and prompt the human operator to resolve the state before resuming.

7. **Extract summary** — Generate a one-line summary by compressing the subagent's response into a single sentence. Do **not** require the subagent to output a special summary line.

   **Normalize** the summary: collapse any newlines or multiple spaces to a single space, trim leading/trailing whitespace, and truncate to 200 characters if needed.

8. **Append to prior changes (only after verification)** — This step runs **only** after step 6 has confirmed success or clean failure. It must **never** run for partially-applied batches. Pass the summary via stdin:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/prior_changes.py append \
     --output-dir "{plan}/apply-task-suggestions" \
     --id "{section_key}" --phase "apply-task-suggestions" --summary-stdin <<'SUMMARY_EOF'
   {normalized_summary}
   SUMMARY_EOF
   ```
   The `--id` uses `batch.section_key` as the stable identifier. The `--phase` is `"apply-task-suggestions"`.

   For **clean failures**, use summary: `(batch failed — {compact reason}, verified no file changes)`

   **Error handling**: If `append` fails, log a warning but do not abort the batch loop.

9. **Log the result** - Track success/failure for the summary
10. **Record metrics** — Run:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py record \
     --state-file "$STATE_FILE" --phase "apply-task-suggestions" \
     --label "Batch {N} ({section_key})" --subagent-type "general-purpose" \
     --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
     --total-batches {total_batches} --batch-index {N}
   ```
   Where `token_count`, `tool_uses`, `duration_ms` come from the Task tool result. Omit any flags the Task tool didn't return. The script prints an `[ETA]` line to stderr after each call.

**CRITICAL**: Process batches ONE AT A TIME. Never spawn multiple Task subagents simultaneously.

**After the loop completes**, mark the phase finish:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py finish \
  --state-file "$STATE_FILE" --phase "apply-task-suggestions"
```

### 6. Handle Human Decision Items

Follow the human decision batch mode process in `references/human-decision-batch.md`.

**Check `human_review_config.decision_mode` first**: if it is `"claude_auto_decide"` (the user passed `--claude-decide`), do NOT prompt the user — go straight to "Let Claude Decide" Mode and evaluate every `needs-human-decision` suggestion autonomously per the criteria in that reference. By default, **partially-valid suggestions are salvaged** (rewritten to their worthwhile core and applied) rather than skipped; only suggestions with nothing worthwhile are skipped entirely.

Context line for individual review display: `Type: {type}, Task: {task_reference}`

### 7. Generate Summary Report

After all batches are processed, create `{prefix}_applied_task_suggestions.md`:

```markdown
# Applied Task Suggestions Report

**Plan**: /path/to/plan.md
**Tasks file**: /path/to/tasks.md
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

### Batch 1 (Task: T001)
**Suggestions**: 3
**Type**: modification

1. **[Suggestion Title]** - Applied
   Description: [Brief description of what was changed in tasks.md]

2. **[Suggestion Title]** - Applied
   ...

### Batch 2 (Task: T003)
...

## Decided by Claude

Suggestions whose decision was offloaded to Claude — via `--claude-decide`, the batch "Let Claude decide" option, or a per-item "Let Claude decide" selection. Read these from `state.json`: the `human_decisions_{phase}` entries whose `batch_context.decision_source` is `claude_auto_decide` or `claude_auto_decide_salvage`. Omit this section if there are none.

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
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py report \
     --state-file "$STATE_FILE" --phase "apply-task-suggestions"
   ```
   Include the output (if non-empty) at the end of the summary report.

   **Then update the HTML review report** so the decisions show up there too. This overlays every human/Claude decision recorded in `state.json` onto the existing `{plan}/review-tasks/report.html`, giving each finding an **Approved / Salvaged / Skipped** badge plus an expandable detail with the kept/dropped notes:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/html_report_generator.py \
     regenerate-decisions \
     --phase-dir "$(dirname "$STATE_FILE")/review-tasks" \
     --state-file "$STATE_FILE" \
     --apply-phase "apply-task-suggestions"
   ```
   Best-effort: if `report_data.json` is missing from the review dir (the review predates this feature), the command prints a notice and exits 0 without changing the report — just continue.

### 8. Mark Phase Completed

After the summary report is generated, mark the apply-task-suggestions phase as completed in state.json:

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --mark-completed
```

This ensures subsequent phases (e.g., `--implement`) recognize that task suggestions have been applied.

---

## JSON Output Structure

The orchestrator outputs JSON with batches:

```json
{
  "plan_file": "/path/to/plan.md",
  "tasks_file": "/path/to/tasks.md",
  "tasks_file_backup": "/path/to/my-feature/apply-task-suggestions/tasks-backup.md",
  "prefix": "my-feature",
  "output_dir": "/path/to/my-feature/",
  "timestamp": "2024-01-21T12:00:00",
  "batches": [
    {
      "suggestions": [
        {
          "index": 0,
          "title": "Add integration test task",
          "description": "...",
          "type": "addition",
          "reference": "Task T001: Setup DB schema",
          "task_reference": "Task T001",
          "importance": "HIGH"
        },
        {
          "index": 1,
          "title": "Clarify acceptance criteria",
          "description": "...",
          "type": "clarification",
          "reference": "Task T001: Setup DB schema",
          "task_reference": "Task T001",
          "importance": "MEDIUM"
        }
      ],
      "section_key": "task_t001_setup_db_schema",
      "batch_type": "mixed",
      "suggestion_count": 2,
      "priority_score": 5.0,
      "prompt": "Apply the following 2 related task suggestions..."
    }
  ],
  "to_apply": [...],
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
User: /multi-llm:multi-llm --apply-task-suggestions plans/my-feature.md

Claude:
1. Checks for prerequisite files - found
2. Runs orchestrator to get batched task suggestions
3. Gets 8 valid suggestions in 3 batches, 2 needs-human-decision

   Batch 1: 3 modifications for Task T001 (priority: 7.0)
   Batch 2: 2 additions for Plan Coverage (priority: 5.0)
   Batch 3: 1 deletion for Task T005 (isolated, priority: 4.0)

   Efficiency: 5 subagent calls saved (62.5% gain)

For each batch (sequentially):
- Reads current tasks.md
- Spawns Task subagent (general-purpose) with batch.prompt
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
Apply the following task suggestion to the tasks file:

**Plan file** (context only — do NOT modify): {plan_path}
**Tasks file** (target — apply changes HERE): {tasks_path}
**Suggestion**: {title}
**Type**: {type}
**Task reference**: {task_reference}
**Importance**: {importance}

**Details**:
{description}

## Type-Specific Instructions

### For "addition" type:
- Add new task(s) using `### Task T0XX:` format
- Assign the next available sequential task ID
- New tasks MUST include ALL canonical fields:
  - `files_to_modify`, `files_to_create`
  - `estimated_complexity`
  - `subagent_type` (or manual type specification)
  - `depends_on`
  - Acceptance criteria
- After adding, regenerate the dependency graph and header metadata (task count, complexity summary)

### For "modification" type:
- Update the specified task fields (description, dependencies, acceptance criteria, etc.)
- Modified tasks MUST preserve or explicitly update ALL canonical fields — never silently drop fields
- If modifying `depends_on`, regenerate the dependency graph

### For "deletion" type:
- Remove the specified task entirely
- Scan ALL tasks globally for `depends_on` references to the deleted task ID(s) and remove them
- NEVER renumber surviving task IDs — gaps in ID sequence are acceptable and expected
- After deletion, regenerate the dependency graph and header metadata (task count, complexity summary)

### For "clarification" type:
- Rewrite the task description and/or acceptance criteria for clarity
- ALL canonical fields must be preserved — do not drop any fields during clarification

## General Instructions
1. Read the current tasks.md file first
2. Read the plan file for context (but do NOT modify it)
3. Apply the suggested change appropriately
4. Ensure the change integrates smoothly with existing tasks
5. Do NOT make any other changes beyond this suggestion

Return a brief summary of what was changed.
```

For multi-suggestion batches:

```
Apply the following N related task suggestions to the tasks file (Task reference: {task_reference}):

**Plan file** (context only — do NOT modify): {plan_path}
**Tasks file** (target — apply changes HERE): {tasks_path}
**Batch type**: {batch_type}
**Suggestions in this batch**: N

---
### Suggestion 1: {title}
- **Type**: {type}
- **Task reference**: {task_reference}
- **Importance**: {importance}

**Details**:
{description}

---
### Suggestion 2: {title}
...

---

## Type-Specific Instructions

### For "addition" type:
- Add new task(s) using `### Task T0XX:` format, assign next available ID
- New tasks MUST include ALL canonical fields: `files_to_modify`, `files_to_create`, `estimated_complexity`, `subagent_type`, `depends_on`, and acceptance criteria
- After adding, regenerate the dependency graph and header metadata

### For "modification" type:
- Update the specified task fields
- MUST preserve or explicitly update ALL canonical fields — never silently drop fields
- If modifying `depends_on`, regenerate the dependency graph

### For "deletion" type:
- Remove the task and scan ALL tasks globally for `depends_on` references to the deleted ID(s)
- NEVER renumber surviving task IDs — gaps are acceptable
- After deletion, regenerate the dependency graph and header metadata

### For "clarification" type:
- Rewrite task description/criteria for clarity
- ALL canonical fields must be preserved

## General Instructions

1. Read the current tasks.md file first
2. Read the plan file for context (but do NOT modify it)
3. Apply ALL suggestions in this batch, processing them in order
4. For each suggestion:
   - Locate the referenced task
   - Apply the change according to its type
   - Ensure changes integrate smoothly
5. Do NOT make any changes beyond these specific suggestions
6. If suggestions affect the same task, apply them intelligently
7. Before applying each suggestion, check if tasks.md already reflects a similar change (from a previously applied batch). If a suggestion is already addressed, note it as "already applied" in the summary and move on

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
2. **ALWAYS READ FIRST**: Read tasks.md before each batch to get latest state
3. **USE PRE-GENERATED PROMPTS**: Use the `prompt` field from each batch
4. **USE SUBAGENTS**: Always use Task tool with `general-purpose` type
5. **PRESERVE BACKUP**: The orchestrator automatically backs up tasks.md to `apply-task-suggestions/tasks-backup.md` before any modifications. Do not overwrite this backup. The `tasks_file_backup` field in orchestrator_output.json contains the path.
6. **TRACK EVERYTHING**: Log every applied/skipped suggestion for the report
7. **DELETIONS ARE ISOLATED**: Deletion suggestions are always in their own batch
8. **PLAN IS READ-ONLY**: The plan file is used for context only — never modify it during this phase
9. **STABLE TASK IDS**: Never renumber surviving task IDs after deletions — IDs are stable identifiers

---

## Disabling Batching

To process suggestions one-at-a-time (legacy mode):

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --no-batch
```

This is useful for:
- Debugging issues with batched edits
- Very sensitive task definitions where each edit needs verification
- When batching produces unexpected results

---

## Recovery Workflows

### Resuming Interrupted Sessions

If a session is interrupted, progress is saved automatically:

```bash
# Resume from where you left off
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py \
  --plan-file "$(realpath "$PLAN_PATH")" \
  --resume

# Or start fresh, clearing all previous progress
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py \
  --plan-file "$(realpath "$PLAN_PATH")" \
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
```

---

## Workflow Comparison

### Full Workflow Mode
```bash
# Task review and apply happen as sequential phases in the full workflow
/multi-llm:multi-llm --full-workflow plans/my-feature.md
# Phase 3b reviews tasks, Phase 3c applies task suggestions automatically
```

### Separate Mode
```bash
# Step 1: Task review only
/multi-llm:multi-llm --review-tasks plans/my-feature.md

# Step 2: User manually reviews the task suggestions in the generated report

# Step 3: Apply task suggestions later
/multi-llm:multi-llm --apply-task-suggestions plans/my-feature.md
```

The separate mode gives users time to:
- Review task suggestions before deciding to apply them
- Skip certain suggestions manually via the HTML report
- Apply suggestions in a separate session
- Verify the validation results are accurate

---

## Edge Cases

| Edge Case | Handling |
|-----------|----------|
| No valid suggestions | Exit early with message, mark phase skipped |
| All needs-human-decision | Prompt user for each |
| tasks.md changed since review | Read latest version before each batch |
| Suggestion references deleted task | Note as "already addressed" or skip |
| Same task in multiple batches | Process sequentially, re-read tasks.md between batches |
| Missing validation file | Treat all as needs-human-decision |
| Task ID conflicts on addition | Assign next available ID based on current tasks.md state |

---

## Error Types and Recovery

The orchestrator distinguishes between different error types:

| Error Type | Status | Recoverable | Recovery Action |
|------------|--------|-------------|-----------------|
| Parsing error | `validation_failed` | Yes | Manual review or re-run review-tasks |
| Timeout | `validation_failed` | Yes | Re-run with increased timeout |
| Rate limited | `validation_failed` | Yes | Wait and re-run |
| Model failure | `needs-human-decision` | No | Review manually |
| Genuine ambiguity | `needs-human-decision` | No | Review manually |

The `validation_failed` status is different from `needs-human-decision`:
- `validation_failed`: Technical error during validation, can be retried
- `needs-human-decision`: LLM determined item genuinely requires human judgment
