# Review Code Mode Instructions

Reviews code changes made during implementation against the original plan.

## Usage

```bash
/multi-llm:multi-llm --review-code <plan_path> [--models model1 model2] [--base-ref REF] [--quick]
```

## Options

- `--models`: Models to use for review (default: claude-sonnet)
- `--base-ref`: Git ref to compare against (default: from state file)
- `--skip-validation`: Skip the LLM validation step (faster but no false-positive filtering)
- `--validation-model <model>`: Model to use for validation (default: "auto")
- `--apply-fixes`: Prepare fix tasks for Claude Code subagents to apply
- `--quick` / `-q`: Use quick_models from providers.yaml for lightweight reviews (2 models)

## Process

1. **Get Diff**: Compare current code against base ref
2. **Run Reviews**: Each model reviews the diff against the plan
3. **Aggregate Issues**: Collect and prioritize issues by severity
4. **Validate Issues**: Filter out false positives and invalid issues
5. **Prepare Fix Tasks** (if `--apply-fixes`): Generate fix tasks JSON for subagents
6. **Apply Fixes** (if `--apply-fixes`): Claude spawns Task subagents to fix each valid issue
7. **Update Plan**: Document what was fixed in the original plan
8. **Generate Report**: Create review report with all findings

## Output Files

All output files are organized into phase-based subdirectories:

```
plans/todo/my-feature.md                      # Original plan (updated with fixes if --apply-fixes)
plans/todo/my-feature/                        # Output folder
├── state.json                                # Session state (plan-local)
└── code-review/                              # Code review phase outputs
    ├── report.md                             # Review report
    ├── issues.json                           # Raw issues
    ├── grouped.json                          # Grouped issues
    ├── validation.json                       # Validation results
    ├── fix_tasks.json                        # Fix tasks for subagents (if --apply-fixes)
    ├── {model}.json                          # Per-model results
    ├── log_{model}.txt                       # Agent output logs (for debugging)
    ├── error_{model}.log                     # Error logs (if failed)
    └── salvage_{model}.json                  # Salvage requests (if JSON parse failed)
```

## Report Format and User Skip Functionality

The generated `report.md` uses a checkbox-based format that allows users to mark issues they want to skip:

### Issue Format

```markdown
### 1. Missing null check in user validation
- [ ] Skip
**Validation:** ✓ Valid | **File:** `src/auth.ts:42-50` | **Type:** bug | **Model:** gpt-4

Description of the issue...

---
```

### Validation Status Display

Each issue shows explicit validation status:
- `**Validation:** ✓ Valid` - LLM confirmed this is a real issue
- `**Validation:** ✗ Invalid` - LLM determined this is a false positive
- `**Validation:** ? Needs Review` - Requires human judgment
- `**Validation:** ? Unknown` - Validation status not available

### Marking Issues to Skip

To skip an issue when applying fixes later:
1. Open `{plan}/code-review/report.md` in any editor
2. Change `- [ ] Skip` to `- [x] Skip` for issues you want to exclude
3. Save the file
4. Run `--apply-code-fixes` - marked issues will be filtered out

This is useful for:
- Skipping valid issues that are intentional or out of scope
- Excluding issues you've already fixed manually
- Filtering out low-priority items you don't want to address

## Debug Logs

Each model invocation generates a log file (`log_{model}.txt`) in the `code-review/` directory containing:
- Timestamp and duration
- The prompt sent to the agent
- Full stdout/stderr output
- Success/failure status and error details

These logs are useful for debugging why an agent failed or what thinking/tools it used.

## Validation Statuses

Each suggestion/issue group receives a validation status:

- **valid** (checkmark): Issue is real and should be addressed
- **invalid** (x): False positive, can be ignored
- **needs-human-decision** (?): Requires human judgment

Items with `needs-human-decision` status are included in the report for the user to review. Human decisions are handled later in the **apply-code-fixes** phase — do NOT prompt the user for decisions during review-code.

---

## Critical Subagent Delegation Rules

**MANDATORY**: These rules MUST be followed when applying fixes:

1. **ALWAYS use the Task tool**: When applying fixes, ALWAYS use the Task tool with `subagent_type: "general-purpose"`. NEVER implement fixes manually.

2. **OUTPUT TRACKING**: After applying fixes (with `--apply-fixes`):
   - Update `{plan}_code_review.md` to mark issues as fixed with summaries
   - Update the original plan file to reference the code review file

---

## Step-by-Step Execution

0. **Check `uv` is available** (BEFORE the first `uv run` command)

   Every command below runs through `uv run`. Follow
   `references/uv-check.md`: run `command -v uv`; if missing, check
   `~/.local/bin/uv` / `~/.cargo/bin/uv` and use the absolute path, otherwise
   use AskUserQuestion to offer installing uv (official installer or package
   manager). If the user declines, abort this mode — do not fall back to bare
   `python`.

1. **Validate plan file exists**

2. **Model selection:**
   - If `--models` provided in args: use those directly
   - If NO models provided: Claude MUST use AskUserQuestion to prompt user for model selection
     - Present available models from `providers.yaml` (multi-select)
     - Then run orchestrator with `--models <selected>`

3. **Determine base-ref (handled automatically by orchestrator):**
   - If `--base-ref` provided in CLI: it will be used
   - Otherwise: the orchestrator automatically loads the state file from `{plan}/state.json` and retrieves `head_at_start`
   - If neither exists: falls back to HEAD~1

   **IMPORTANT**: Do NOT manually search for state files. The orchestrator handles state loading automatically via `get_or_create_state(plan_path)`.

4. **Resume Detection** (check BEFORE running orchestrator)

   Check for existing output to avoid expensive duplicate runs. Follow the detection cases in `references/resume-detection.md`, using `{plan}/code-review/` as the phase directory.

   **Note**: The orchestrator has TWO guards as defense-in-depth:
   - **Primary**: Exits with code 2 if phase is already marked complete in state.json. Use `--force` to override.
   - **Secondary**: Exits with code 3 if partial completion artifacts exist (validation_tasks.json, grouped.json, or per-model .json files) without the phase being complete. This prevents expensive re-runs when the previous run was interrupted.

   This instruction-level resume detection is the FIRST line of defense. The orchestrator guards are backup protection.

5. **Run the code review orchestrator (DETACHED):**

   A fan-out review over many models routinely exceeds the Claude Code Bash tool's hard 10-min `timeout` cap (600000 ms; larger values are silently clamped), so run it **DETACHED** with `run_in_background: true`, redirecting stdout+stderr to a log file in the phase dir and setting `PYTHONUNBUFFERED=1`:
   ```bash
   PYTHONUNBUFFERED=1 uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/code_review_orchestrator.py" \
     --plan-file "$PLAN_PATH" --models <selected> [--base-ref REF] [--apply-fixes] \
     > "{plan}/code-review/orchestrator-run.log" 2>&1
   ```

   **IMPORTANT**: Pass `$PLAN_PATH` as given — the orchestrator resolves it to an OS-native absolute path itself (do NOT wrap it in `$(realpath ...)`; on Git for Windows that emits a POSIX `/c/...` path a native process cannot use). Use `--project` (not `--directory`).

   Launch with `run_in_background: true`. Detached runs are NOT subject to the 10-min Bash cap, so the orchestrator survives long multi-model runs. `PYTHONUNBUFFERED=1` is required so Python streams output to the log instead of block-buffering it (block-buffering leaves the log empty for minutes when stdout is a non-TTY pipe). When the background task completes, read all markers and output paths **from `{plan}/code-review/orchestrator-run.log`**, not from terminal stdout.

   **Resume**: re-invoking with `--force` RESUMES (keeps already-completed per-model result files, runs only the missing models); `--rerun-all` forces a full re-run discarding existing per-model results.

   The orchestrator will output (to the log):
   - `Git base reference: {commit}` - showing which base ref is being used
   - `Tracked files: N files` - files that were modified during implementation
   - If no state file exists, it will show: "No tracked files in state, falling back to git diff from {base_ref}"

6. **Salvage Handling (Post-Orchestrator):**

   After the background task completes, check the log file `{plan}/code-review/orchestrator-run.log` for `[SALVAGE_NEEDED]` markers. Follow the salvage process in `references/salvage-handling.md`.

   The reaggregation command for this mode is:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/code_review_orchestrator.py" \
     --plan-file "$PLAN_PATH" \
     --reaggregate
   ```

7. **Validation Handling (Post-Orchestrator):**

   After the background task completes, check the log file `{plan}/code-review/orchestrator-run.log` for validation markers.

   #### Reference-Based Validation

   The `validation_tasks.json` file uses a reference-based format (small metadata file):
   ```json
   {
     "batches": [
       {
         "batch_index": 0,
         "group_indices": [0, 1, 2],
         "output_path": "{plan}/code-review/validation_batch_0.json"
       }
     ],
     "grouped_file": "{plan}/code-review/grouped.json",
     "plan_file": "/absolute/path/to/plan.md",
     "reaggregate_command": "uv run ... code_review_orchestrator.py ... --reaggregate"
   }
   ```

   #### Single Batch: `[VALIDATION_PENDING]`

   If `[VALIDATION_PENDING]` marker is present:
   1. Read `validation_tasks.json` from `{plan}/code-review/` (small metadata file)
   2. Spawn one validation subagent that reads the source files directly:

      ```
      Task tool call:
        subagent_type: general-purpose
        description: "Validate code review issues"
        prompt: |
          Validate issue groups from a code review.

          Read these files:
          - Grouped issues: {grouped_file from validation_tasks.json}
          - Plan context: {plan_file from validation_tasks.json}
          - Base git ref: {base_ref from validation_tasks.json}

          For each group, examine the actual code before making your determination:
          - Read the file(s) referenced in the group's issues (use the `file` and `line_range` fields)
          - If base_ref is non-empty, optionally run `git diff {base_ref} -- {file}` to see what changed (skip this step when base_ref is empty to avoid misleading unstaged diffs)
          - Compare what the code actually does against what the issue claims

          If an issue is missing `file` or `line_range` metadata (e.g., the finding refers to a deleted/renamed file, or describes a cross-cutting concern without a specific location):
          - Attempt validation using whatever context is available: the issue description, the plan file, and any other files in the group that do have location data
          - If you can determine validity from the description and plan alone, return `valid` or `invalid` as normal
          - If the missing location context makes it impossible to verify the claim with confidence, return `needs-human-decision` with a note explaining that file/line metadata was absent

          Code context collection limits:
          - Max files per group: 5 files. If a group references more files, prioritize by: (1) files explicitly named in `file` fields, (2) files with the smallest line ranges (most specific references). Break ties using importance (HIGH > MEDIUM > LOW).
          - Max lines per file: 200 lines around referenced line range (100 before, 100 after). If no line range is specified, read the first 200 lines of the file.
          - Max diff chunks per file: 3 hunks with --unified=5
          - If diff output exceeds 3 hunks, note "diff truncated" in your reasoning and make your determination from the available context
          - Prioritize HIGH importance issues when limits are reached

          Validate ONLY these groups (use group_hash from grouped.json to identify each):
          {for each i: "  index {group_indices[i]} (group_hash: {group_ids[i]})" from batches[N]}
          You MUST output EXACTLY {len(group_indices)} results — one per requested group.
          Do NOT validate or include results for any other groups.

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
          "Done: wrote 3 validation results to /path/to/output.json"
          Do NOT include reasoning, analysis, or explanation in your response.
      ```

   3. Wait for the subagent to complete

   **Record metrics** for each validation subagent. When the validation phase used multiple batches, include the cumulative `--batch-index` (1..N across waves) and `--total-batches` so the script prints an `[ETA]` line:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/metrics.py" record \
     --state-file "$STATE_FILE" --phase "code-review" \
     --label "Validation batch {N}" --subagent-type "general-purpose" \
     --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
     --total-batches {total_batches} --batch-index {N}
   ```
   Where `token_count`, `tool_uses`, `duration_ms` come from the Task tool result. Omit any flags the Task tool didn't return. Single-batch validation runs may omit `--total-batches`/`--batch-index`.

   4. Run reaggregation using the command from `reaggregate_command` in validation_tasks.json

   #### Multiple Batches: `[VALIDATION_BATCHES_PENDING]`

   If `[VALIDATION_BATCHES_PENDING]` marker is present (for 5+ issue groups):
   1. Read `validation_tasks.json` once from `{plan}/code-review/`
   2. Count the number of batches as `{total_batches}` and mark the phase start:
      ```bash
      uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/metrics.py" start \
        --state-file "$STATE_FILE" --phase "review-code-batches" --total-batches {total_batches}
      ```
   3. Choose a strategy:

   **Strategy A: Direct parallel spawning (≤ 4 batches)**

   For each batch in the `batches` array, spawn a Task agent using the same prompt template as single-batch validation above, but with that batch's `group_indices` and `output_path`.

   **Run all batches in parallel** (single message with multiple Task tool calls).

   **Strategy B: Wave-based parallel spawning (> 4 batches)**

   Follow the wave-based spawning strategy in `references/wave-batching.md`.

   3. After ALL batches complete, run reaggregation using `reaggregate_command`

   #### Reaggregation (After Validation)

   After validation completes, run the `reaggregate_command` from validation_tasks.json:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/code_review_orchestrator.py" \
     --plan-file "$PLAN_PATH" --reaggregate
   ```

   After reaggregation, mark the validation phase finish (only if `start` was called above):
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/metrics.py" finish \
     --state-file "$STATE_FILE" --phase "review-code-batches"
   ```

   **CRITICAL**: Do NOT report results to the user until reaggregation is complete.

8. **If `--apply-fixes` was used, apply fixes using Task subagents:**

   After the orchestrator outputs `{prefix}_fix_tasks.json`, Claude MUST use the Task tool to spawn subagents to fix each issue.

   Before the fix loop, mark the phase start where `{total_fixes}` is the number of fix tasks:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/metrics.py" start \
     --state-file "$STATE_FILE" --phase "review-code-fixes" --total-batches {total_fixes}
   ```

   For each fix task in the JSON file:

   a. Read the fix tasks JSON file to get the list of issues to fix
   b. For each task, spawn a Task subagent:
      ```
      Task tool call:
      - subagent_type: "general-purpose"
      - description: "Fix: {task.title}"
      - prompt: |
          Fix the following code issue:

          **Issue**: {task.title}
          **Type**: {task.type}
          **Importance**: {task.importance}
          **Description**: {task.description}
          **Location hints**: {task.location_hints}

          Plan file for context: {task.plan_path}

          1. Read the relevant files and understand the issue
          2. Make the necessary code changes to fix it
          3. Verify your changes don't break anything

          After fixing, report what you changed.
      ```
   c. Track fix results (success/failure) for each task

   **Record metrics** for each fix subagent:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/metrics.py" record \
     --state-file "$STATE_FILE" --phase "code-review" \
     --label "Fix: {task_title}" --subagent-type "{subagent_type}" \
     --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
     --total-batches {total_fixes} --batch-index {N}
   ```
   Where `token_count`, `tool_uses`, `duration_ms` come from the Task tool result, and `{N}` is the 1-based fix index. Omit any flags the Task tool didn't return. The script prints an `[ETA]` line to stderr after each call.

   d. After all fixes are attempted, mark the phase finish:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/metrics.py" finish \
     --state-file "$STATE_FILE" --phase "review-code-fixes"
   ```

   e. Update the plan file with a "Review Fixes Applied" section documenting what was fixed

## Post-Fix Workflow

After spawning subagents to apply fixes:

1. **Wait for all subagents to complete**
   - Track success/failure of each fix

2. **Run verification**
   ```bash
   pnpm typecheck
   pnpm lint:fix
   ```

3. **Update `{plan}_code_review.md`** with implementation status for each issue:

   For each issue in the code review file, add an "Implementation" subsection:

   ```markdown
   ### Issue N: [Original Title]
   **Type**: security | **Importance**: HIGH
   **Description**: [original description]
   **Location Hints**: [original hints]

   #### Implementation Status: FIXED
   **Files Modified**:
   - `path/to/file1.ts` - Added input validation
   - `path/to/file2.ts` - Updated error handling

   **Summary**: Added proper input sanitization to prevent SQL injection...
   ```

   Or if not fixed:
   ```markdown
   #### Implementation Status: NOT FIXED
   **Reason**: Requires architectural changes beyond scope of this review
   ```

4. **Update the original plan file** by appending a reference section:

   ```markdown
   ---
   ## Code Review Applied

   A code review was performed on this implementation. See the full review and implementation details at:
   **[{plan}_code_review.md](./{plan}/{plan}_code_review.md)**

   **Summary**: X of Y issues were fixed.
   **Date**: [timestamp]
   ```

6. **Report results** including files in `{plan}/code-review/`:

   Generate resource usage:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/metrics.py" report \
     --state-file "$STATE_FILE" --phase "code-review"
   ```
   Include the output (if non-empty) in the results report.

   - `report.md` - Review report
   - `issues.json` - Raw issues
   - `grouped.json` - Grouped issues
   - `validation.json` - Validation results
   - `fix_tasks.json` (if --apply-fixes)
   - Summary of issues by severity
   - Summary of fixes applied (if --apply-fixes)

---

## Example Executions

### Example 1: Review Code (no models provided)

```
User: /multi-llm:multi-llm --review-code plans/my-feature.md

Claude:
1. No models in args, so Claude uses AskUserQuestion:
   "Which models would you like to use for the code review?" (multi-select)
   Options: auto, gpt-5.2, gemini-3-pro, grok, gemini-3-flash, kimi-k2
2. User selects: gpt-5.2
3. Runs: uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/code_review_orchestrator.py" --plan-file plans/my-feature.md --models gpt-5.2
4. Claude reports review findings and output files
```

### Example 2: Review Code with --apply-fixes

```
User: /multi-llm:multi-llm --review-code --apply-fixes plans/my-feature.md

Claude:
1. No models in args, so Claude uses AskUserQuestion to prompt for model selection
2. User selects: gemini-3-pro
3. Runs: uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/code_review_orchestrator.py" --plan-file plans/my-feature.md --models gemini-3-pro --apply-fixes
4. Orchestrator outputs:
   - my-feature_code_review.md (report)
   - my-feature_fix_tasks.json (3 valid issues to fix)
5. Claude reads fix_tasks.json and spawns Task subagents for each fix:
   - Task 1: "Fix: Missing null check in user validation" -> Subagent fixes it
   - Task 2: "Fix: SQL injection vulnerability" -> Subagent fixes it
   - Task 3: "Fix: Incorrect error message" -> Subagent fixes it
6. Claude updates the plan file with "Review Fixes Applied" section:
   - Successfully Fixed: 3 issues
   - Files modified: auth.ts, db.ts, errors.ts
7. Reports final summary including fixes applied
```
