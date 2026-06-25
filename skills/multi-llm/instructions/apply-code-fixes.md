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

All fix batches use `subagent_type: "general-purpose"` — the only implementation subagent type available in Claude Code.

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
4. **Handle Human Decisions**: For `needs-human-decision` items, use AskUserQuestion — each interactive prompt also offers a per-item **Let Claude decide** option that offloads just that one fix's judgment to a subagent. Or, if `--claude-decide` was passed, let Claude judge every fix up front — or pre-marked in the report's *Let Claude decide* dropdown/checkbox, which routes just those fixes to the judge without prompting. See step 6.
5. **Batch Application**: Process each batch with a specialist subagent:
   - Spawn a Task subagent with the appropriate `subagent_type`
   - Wait for completion before moving to the next batch
   - This prevents conflicts from concurrent edits
6. **Run Verification**: Execute `pnpm typecheck` after all fixes
7. **Generate Report**: Create `{prefix}_applied_fixes.md` with results
8. **Update Plan**: Add "Code Fixes Applied" section to plan file

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

**Note**: By default, revalidation is delegated to a Claude Code subagent. Use `--internal-revalidation` to run revalidation inside the orchestrator (legacy mode).

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
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" [options]
```

**IMPORTANT**: Always use `$(realpath "$PLAN_PATH")` to convert to absolute path.

**IMPORTANT**: Run this command in the FOREGROUND (do NOT use `run_in_background`). Use `timeout: 600000` (10 minutes — the Bash tool caps `timeout` at 600000 ms; larger values are silently clamped). The orchestrator runs quickly — it only generates batch JSON, it does not apply changes — so 10 minutes is far more than enough.

The orchestrator will:
1. Load validation results and grouped issues
2. Filter to only `valid` fixes (skip `invalid`, prompt for `needs-human-decision`)
3. Group fixes into efficient batches by file
4. Determine appropriate subagent type for each batch
5. Write output to `{plan}/apply-fixes/orchestrator_output.json`
6. Exit - Claude Code handles the actual application

### 2a. Handle No-Selections Confirmation

After the orchestrator completes, check if its stdout contains `"status": "confirmation_needed"`.

If present:
1. Display the warning message to the user
2. Use AskUserQuestion:
   > "No user selections were found (no user_selections.json, consolidated_user_selections.json, or report.md edits detected). This will apply ALL {item_count} valid fixes. Continue?"
   > Options: "Yes, apply all" / "No, I need to make my selections first"
3. If user confirms: Re-run the orchestrator with `--no-confirm` appended
4. If user declines: Stop and inform the user to review the HTML report and export selections

If NOT present, proceed to step 3.

### 3. Display Decision Summary

After the orchestrator completes, run the display script to show the user which fixes will be applied, skipped, need review, etc.:

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/display_decisions.py --plan-file "$(realpath "$PLAN_PATH")" --phase apply-code-fixes
```

This prints a formatted summary with each fix's title and status. Display this output to the user before proceeding.

### 4. Read the Orchestrator Output

Use the **Read tool** to load `{plan}/apply-fixes/orchestrator_output.json`.

Extract the following key fields directly from the Read output (do NOT use Bash/python to parse — just read the JSON and understand it):
- `batches`: Array of batch objects, each with `prompt`, `file_key`, `subagent_type`, `fixes`
- `needs_human_review`: Items requiring human decision
- `human_review_config`: Batch mode config for human review
- `summary`: Counts of valid, skipped, batches, etc.

### 5. Process Batches Sequentially

Use the batch data already loaded from the Read tool in step 4. Do NOT use Bash/python to re-parse or extract fields from the JSON file.

**Before the loop**, mark the phase start so ETA tracking has a wall-clock baseline. Run:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py start \
  --state-file "$STATE_FILE" --phase "apply-code-fixes" --total-batches {total_batches}
```
Where `{total_batches}` is the length of the `batches` array.

For each batch (index N, starting at 1) returned by the orchestrator, Claude MUST:

1. **Read prior changes** — Run:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/prior_changes.py read \
     --output-dir "{plan}/apply-fixes"
   ```
   Capture stdout as `prior_changes_text`.

   **Error handling**: If `read` fails (corrupted JSONL, permission error, unexpected exception), use the default text `(none — this is the first batch)` and log a warning. Do not abort the batch loop.

2. **Substitute placeholder** — In `batch.prompt`, replace the literal `{prior_changes_context}` with `prior_changes_text`.

3. **Spawn a Task subagent** using the `subagent_type` from the batch (not always `general-purpose` — use the batch-specific type):
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
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/prior_changes.py append \
     --output-dir "{plan}/apply-fixes" \
     --id "{file_key}" --phase "apply-fixes" --summary-stdin <<'SUMMARY_EOF'
   {normalized_summary}
   SUMMARY_EOF
   ```
   The `--id` uses `batch.file_key` as the stable identifier. The `--phase` is `"apply-fixes"`.

   For **clean failures**, use summary: `(batch failed — {compact reason}, verified no file changes)`

   **Error handling**: If `append` fails, log a warning but do not abort the batch loop.

8. **Log the result** - Track success/failure for the summary
9. **Record metrics** — Run:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py record \
     --state-file "$STATE_FILE" --phase "apply-code-fixes" \
     --label "Batch {N} ({file_key})" --subagent-type "{subagent_type}" \
     --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
     --total-batches {total_batches} --batch-index {N}
   ```
   Where `token_count`, `tool_uses`, `duration_ms` come from the Task tool result. Omit any flags the Task tool didn't return. The script prints an `[ETA]` line to stderr after each call.

**CRITICAL**: Process batches ONE AT A TIME. Never spawn multiple Task subagents simultaneously.

**After the loop completes**, mark the phase finish:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py finish \
  --state-file "$STATE_FILE" --phase "apply-code-fixes"
```

### 6. Handle Human Decision Items

Follow the human decision batch mode process in `references/human-decision-batch.md`.

**Auto-route report-pre-marked fixes first**: before prompting for anything, partition `needs_human_review` into (a) the **pre-marked Claude set** — every fix whose `group_id` is in `human_review_config.claude_decide_item_ids` or that carries `decision_mode == "claude_auto_decide"` — and (b) the remaining **interactive set**. Run **"Let Claude Decide" Mode** (per the criteria in `references/human-decision-batch.md`) on the pre-marked Claude set **without any prompt**, then handle only the interactive set with the flow below. These fixes were chosen in the report's "Let Claude decide" dropdown/checkbox; they are the per-item equivalent of `--claude-decide` scoped to just those fixes.

**Check `human_review_config.decision_mode` first**: if it is `"claude_auto_decide"` (the user passed `--claude-decide`), do NOT prompt the user — go straight to "Let Claude Decide" Mode and evaluate every `needs-human-decision` fix autonomously per the criteria in that reference. By default, **partially-valid fixes are salvaged** (rewritten to their worthwhile core and applied) rather than skipped; only fixes with nothing worthwhile are skipped entirely.

Context line for individual review display: `File: {file}, Type: {type}`

### 7. Run Verification

After all fixes are applied:

```bash
pnpm typecheck && pnpm lint:fix
```

If typecheck fails:
- Report which fix may have caused the issue
- Mark that fix as "failed" in the summary
- Continue with remaining fixes if possible

### 8. Generate Summary Report

After all batches are processed, create `{prefix}_applied_fixes.md`:

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
- Decided by Claude: N (A approved, V salvaged, S skipped) — includes `--claude-decide`, the batch "Let Claude decide" option, and per-item "Let Claude decide" selections
- Typecheck status: pass/fail

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
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py report \
     --state-file "$STATE_FILE" --phase "apply-code-fixes"
   ```
   Include the output (if non-empty) at the end of the summary report.

   **Then update the HTML review report** so the decisions show up there too. This overlays every human/Claude decision recorded in `state.json` onto the existing `{plan}/code-review/report.html`, giving each finding an **Approved / Salvaged / Skipped** badge plus an expandable detail with the kept/dropped notes:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/html_report_generator.py \
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

See: [Applied Fixes Report](./my-feature/my-feature_applied_fixes.md)
<!-- CODE_FIXES_APPLIED_END -->
```

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
          "subagent_type": "general-purpose"
        }
      ],
      "file_key": "src/api.ts",
      "batch_type": "bug",
      "subagent_type": "general-purpose",
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

Claude:
1. Checks for prerequisite files - found
2. Runs orchestrator to get batched fixes
3. Gets 8 valid fixes in 4 batches, 2 needs-human-decision

   Batch 1: 1 security fix in src/auth.ts -> general-purpose (isolated)
   Batch 2: 2 bug fixes in src/api.ts -> general-purpose
   Batch 3: 1 RLS fix in migrations/xxx.sql -> general-purpose
   Batch 4: 2 fixes in src/components/Form.tsx -> general-purpose

   Efficiency: 4 subagent calls saved (50% gain)

For each batch (sequentially):
- Spawns Task subagent with batch.subagent_type
- Uses pre-generated batch.prompt
- Waits for completion
- Logs results

For needs-human-decision items:
- Uses AskUserQuestion to get user decision
- Applies or skips based on response

4. Runs verification: pnpm typecheck && pnpm lint:fix
5. Generates summary report
6. Updates plan file with results
7. Reports completion with file paths
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
2. **USE SUBAGENT TYPES**: Always use the `subagent_type` from each batch
3. **USE PRE-GENERATED PROMPTS**: Use the `prompt` field from each batch
4. **NEVER IMPLEMENT MANUALLY**: Always use Task tool with the appropriate subagent type
5. **TRACK EVERYTHING**: Log every applied/skipped fix for the report
6. **SECURITY IS ISOLATED**: Security and HIGH importance fixes are always in their own batch
7. **VERIFY AFTER FIXES**: Run typecheck after applying all fixes
8. **UPDATE PLAN**: Always add the results section to the plan file

---

## Disabling Batching

To process fixes one-at-a-time (legacy mode):

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --no-batch
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
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py \
  --plan-file "$(realpath "$PLAN_PATH")" \
  --revalidate \
  --revalidate-model cursor-agent:opus

# Option 2: Bulk approve validation failures (if you trust the fixes)
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py \
  --plan-file "$(realpath "$PLAN_PATH")" \
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
      "output_path": "{plan}/code-review/validation_batch_0.json"
    }
  ],
  "grouped_file": "{plan}/code-review/grouped.json",
  "plan_file": "/absolute/path/to/plan.md",
  "item_indices": [0, 2, 5],
  "reaggregate_command": "uv run ... apply_code_fixes_orchestrator.py ..."
}
```

#### Single Batch: `[REVALIDATION_PENDING]`

If `[REVALIDATION_PENDING]` marker is present:
1. Read `revalidation_tasks.json` from `{plan}/code-review/` (small metadata file)
2. Spawn a single revalidation subagent that reads source files directly:

   ```
   Task tool call:
     subagent_type: general-purpose
     description: "Revalidate failed code fixes"
     prompt: |
       Revalidate code issue groups that previously failed validation.

       Read these files:
       - Grouped issues: {grouped_file from revalidation_tasks.json}
       - Plan context: {plan_file from revalidation_tasks.json}
       - Base git ref: {base_ref from revalidation_tasks.json}

       For each group, examine the actual code before making your determination:
       - Read the file(s) referenced in the group's issues (use the `file` and `line_range` fields)
       - If base_ref is non-empty, optionally run `git diff {base_ref} -- {file}` to see what changed (skip this step when base_ref is empty to avoid misleading unstaged diffs)
       - Compare what the code actually does against what the issue claims

       If an issue is missing `file` or `line_range` metadata (e.g., the finding refers to a deleted/renamed file, or describes a cross-cutting concern without a specific location):
       - Attempt validation using whatever context is available: the issue description, the plan file, and any other files in the group that do have location data
       - If you can determine validity from the description and plan alone, return `valid` or `invalid` as normal
       - If the missing location context makes it impossible to verify the claim with confidence, return `needs-human-decision` with a note explaining that file/line metadata was absent

       Code context collection limits:
       - Max files per group: 5 files
       - Max lines per file: 200 lines around referenced line range (if no line range is specified, read the first 200 lines)
       - Max diff chunks per file: 3 hunks with --unified=5
       - If diff output exceeds 3 hunks, note "diff truncated" in your reasoning and make your determination from the available context
       - Prioritize HIGH importance issues when limits are reached
       - When selecting which files to examine under limits, prefer files referenced by HIGH importance issues first, then MEDIUM, then LOW

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
1. Read `revalidation_tasks.json` from `{plan}/code-review/`
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
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py \
  --plan-file "$(realpath "$PLAN_PATH")" \
  --resume

# Or start fresh, clearing all previous progress
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py \
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
