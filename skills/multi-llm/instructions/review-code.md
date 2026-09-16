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
- `--apply-fixes`: Prepare fix tasks for subagents to apply
- `--quick` / `-q`: Use quick_models from providers.yaml for lightweight reviews (2 models)

## Process

1. **Get Diff**: Compare current code against base ref
2. **Run Reviews**: Each model reviews the diff against the plan
3. **Aggregate Issues**: Collect and prioritize issues by severity
4. **Validate Issues**: Filter out false positives and invalid issues
5. **Prepare Fix Tasks** (if `--apply-fixes`): Generate fix tasks JSON for subagents
6. **Apply Fixes** (if `--apply-fixes`): the agent spawns one subagent per valid issue
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

The report numbers **issues** (one block per model finding) while a fix is
applied per **group**, so a block carrying `_Same finding as #7, #11_` is one of
several views of a single finding: marking `Skip`, `Mark valid` or `Mark
invalid` on any of them marks the whole group. The header's `**Groups:**` line
says how many distinct findings the issue count collapsed into, and the
validation counts are per group.

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

1. **ALWAYS delegate**: When applying fixes, ALWAYS run the fix task as a subagent (`executor_role: "agent"`). NEVER implement fixes manually.

2. **OUTPUT TRACKING**: After applying fixes (with `--apply-fixes`):
   - Update `{plan}_code_review.md` to mark issues as fixed with summaries
   - Update the original plan file to reference the code review file

---

## Step-by-Step Execution

0. **Check `uv` is available** (BEFORE the first `uv run` command)

   Every command below runs through `uv run`. Follow
   `references/uv-check.md`: run `command -v uv`; if missing, check the seven
   documented install locations and use the absolute path. If uv is truly
   absent, print `hosts/uv-missing.txt` verbatim and **stop with exit 3** — never
   offer to install it, never ask, and never fall back to bare `python`
   (D6 row 1).

1. **Validate plan file exists**

2. **Model selection:**
   - If `--models` provided in args: use those directly
   - **With `--yes` (or `--non-interactive`)**: never ask. Pass no model flag and
     let the orchestrator resolve `defaults.models` from the config layering
     (it prints `[AUTO_RESOLVED] model-selection: …`). If nothing resolves, it
     exits **3** naming the flags that answer the gate — do not substitute a
     question (D6 row 2). `--interactive` with `--yes` is exit 3 at parse time.
   - If NO models provided and NOT `--yes`: **ask the user** to select models
     - Present available models from `providers.yaml` (multi-select)
     - Then run orchestrator with `--models <selected>`
   - **Partial results** (D6 row 3): under `--yes`, a model that produced nothing
     is re-run once via `--force` and never with `--rerun-all`; a second failure
     is reported as `[PARTIAL_RESULTS: <model>,…]` with
     `MULTI_LLM_RESULT.state: "partial"` and exit **0**. Zero results is exit
     **3** naming the retry command.

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

   A fan-out review over many models routinely outlives any single shell command a host is willing to wait for, so **run detached**, redirecting stdout+stderr to a log file in the phase dir and setting `PYTHONUNBUFFERED=1`:
   ```bash
   PYTHONUNBUFFERED=1 uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-code \
     --plan-file "$PLAN_PATH" --models <selected> [--base-ref REF] [--apply-fixes] \
     > "{plan}/code-review/orchestrator-run.log" 2>&1
   ```

   **IMPORTANT**: Pass `$PLAN_PATH` as given — the orchestrator resolves it to an OS-native absolute path itself (do NOT wrap it in `$(realpath ...)`; on Git for Windows that emits a POSIX `/c/...` path a native process cannot use). Use `--project` (not `--directory`).

   **Run detached.** A detached run is not subject to your host's shell-command cap, so the orchestrator survives long multi-model runs. `PYTHONUNBUFFERED=1` is required so Python streams output to the log instead of block-buffering it (block-buffering leaves the log empty for minutes when stdout is a non-TTY pipe). When the background task completes, read all markers and output paths **from `{plan}/code-review/orchestrator-run.log`**, not from terminal stdout.

   **Resume**: re-invoking with `--force` RESUMES (keeps already-completed per-model result files, runs only the missing models); `--rerun-all` forces a full re-run discarding existing per-model results.

   The orchestrator will output (to the log):
   - `Git base reference: {commit}` - showing which base ref is being used
   - `Tracked files: N files` - files that were modified during implementation
   - If no state file exists, it will show: "No tracked files in state, falling back to git diff from {base_ref}"

6. **Salvage Handling (Post-Orchestrator):**

   After the background task completes, check the log file `{plan}/code-review/orchestrator-run.log` for `[SALVAGE_NEEDED]` markers. Follow the salvage process in `references/salvage-handling.md`.

   The reaggregation command for this mode is:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-code \
     --plan-file "$PLAN_PATH" \
     --reaggregate
   ```

7. **Validation Handling (Post-Orchestrator):**

   After the background task completes, check the log file `{plan}/code-review/orchestrator-run.log` for validation markers.

   #### The task file

   `validation_tasks.json` carries `tasks_schema_version: 2` and one complete
   task per batch under `batches` — each with `prompt`, `output_name`, the
   derived `output_path`, `result_channel: file`, `context_mode: inline`,
   `capability_tier`, `executor_role` and `task_id`. The plan excerpt, this
   batch's issue records (including each issue's `file` and `line_range`) and the
   code-inspection guidance all travel **inside** `prompt`.

   **For each task in the file, run `task.prompt` as a subagent, then write the
   JSON it returns to `task.output_path`.** That is the whole contract — do not
   assemble a prompt, and do not tell the subagent to write the file itself.

   A file whose `tasks_schema_version` is missing or not `2`, or whose recorded
   `inputs` hashes no longer match the plan and grouped.json, is refused with
   exit 3: re-run the phase with `--force` to re-materialize it.

   #### Single Batch: `[VALIDATION_PENDING]`

   If `[VALIDATION_PENDING]` marker is present:
   1. Read `validation_tasks.json` from `{plan}/code-review/`
   2. Run the single task's `prompt` as a subagent and write its JSON reply to
      that task's `output_path`

   **Record metrics** for each validation subagent. When the validation phase used multiple batches, include the cumulative `--batch-index` (1..N across waves) and `--total-batches` so the script prints an `[ETA]` line:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics record \
     --state-file "$STATE_FILE" --phase "code-review" \
     --label "Validation batch {N}" --subagent-type "agent" \
     --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
     --total-batches {total_batches} --batch-index {N}
   ```
   Where `token_count`, `tool_uses`, `duration_ms` come from whatever your subagent mechanism reports. Omit any flag it does not report. Single-batch validation runs may omit `--total-batches`/`--batch-index`.

   3. Run the reaggregation command under **Reaggregation (After Validation)** below. `validation_tasks.json` records `reaggregate_mode` + `plan_file`, never a resolved command, so the command is rebuilt from those two values

   #### Multiple Batches: `[VALIDATION_BATCHES_PENDING]`

   If `[VALIDATION_BATCHES_PENDING]` marker is present (for 5+ issue groups):
   1. Read `validation_tasks.json` once from `{plan}/code-review/`
   2. Count the number of batches as `{total_batches}` and mark the phase start:
      ```bash
      uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics start \
        --state-file "$STATE_FILE" --phase "review-code-batches" --total-batches {total_batches}
      ```
   3. Choose a strategy:

   **Strategy A: Direct parallel spawning (≤ 4 batches)**

   For each task in the `batches` array, run `task.prompt` as a subagent and write the JSON it returns to that task's `output_path`.

   **Run all batches in parallel** (one batch of parallel subagent spawns).

   **Strategy B: Wave-based parallel spawning (> 4 batches)**

   Follow the wave-based spawning strategy in `references/wave-batching.md`.

   3. After ALL batches complete, run the reaggregation command below (rebuilt from `reaggregate_mode` + `plan_file`)

   #### Reaggregation (After Validation)

   After validation completes, rebuild the reaggregation command from `reaggregate_mode` + `plan_file` in validation_tasks.json — the file stores that durable pair rather than a resolved command, so a skill dir that moved with a plugin update cannot leave a dead path behind:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-code \
     --plan-file "$PLAN_PATH" --reaggregate
   ```

   After reaggregation, mark the validation phase finish (only if `start` was called above):
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics finish \
     --state-file "$STATE_FILE" --phase "review-code-batches"
   ```

   **CRITICAL**: Do NOT report results to the user until reaggregation is complete.

8. **If `--apply-fixes` was used, apply fixes by spawning subagents:**

   After the orchestrator outputs `{prefix}_fix_tasks.json`, **spawn one subagent per issue**.

   Before the fix loop, mark the phase start where `{total_fixes}` is the number of fix tasks:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics start \
     --state-file "$STATE_FILE" --phase "review-code-fixes" --total-batches {total_fixes}
   ```

   For each fix task in the JSON file:

   a. Read the fix tasks JSON file to get the list of issues to fix
   b. For each task, spawn a subagent:
      ```
      Spawn one subagent with:
      - executor_role: "agent"
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
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics record \
     --state-file "$STATE_FILE" --phase "code-review" \
     --label "Fix: {task_title}" --subagent-type "{executor_role}" \
     --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
     --total-batches {total_fixes} --batch-index {N}
   ```
   Where `token_count`, `tool_uses`, `duration_ms` come from whatever your subagent mechanism reports, and `{N}` is the 1-based fix index. Omit any flag it does not report. The script prints an `[ETA]` line to stderr after each call.

   d. After all fixes are attempted, mark the phase finish:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics finish \
     --state-file "$STATE_FILE" --phase "review-code-fixes"
   ```

   e. Update the plan file with a "Review Fixes Applied" section documenting what was fixed

## Post-Fix Workflow

After spawning subagents to apply fixes:

1. **Wait for all subagents to complete**
   - Track success/failure of each fix

2. **Run verification**
   Run the repository's own typecheck/lint commands, if any (for example
   `pnpm typecheck` and `pnpm lint:fix` in a pnpm project) — this plan may
   have neither.

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

6. **Report results** following `references/summary-format.md`, including files in `{plan}/code-review/`:

   Generate resource usage:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics report \
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

Agent:
1. No models in args, so the agent asks the user:
   "Which models would you like to use for the code review?" (multi-select)
   Options: auto, gpt-5.2, gemini-3-pro, grok, gemini-3-flash, kimi-k2
2. User selects: gpt-5.2
3. Runs: uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-code --plan-file plans/my-feature.md --models gpt-5.2
4. Claude reports review findings and output files
```

### Example 2: Review Code with --apply-fixes

```
User: /multi-llm:multi-llm --review-code --apply-fixes plans/my-feature.md

Agent:
1. No models in args, so the agent asks the user to select models
2. User selects: gemini-3-pro
3. Runs: uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-code --plan-file plans/my-feature.md --models gemini-3-pro --apply-fixes
4. Orchestrator outputs:
   - my-feature_code_review.md (report)
   - my-feature_fix_tasks.json (3 valid issues to fix)
5. Claude reads fix_tasks.json and spawns subagents for each fix:
   - Task 1: "Fix: Missing null check in user validation" -> Subagent fixes it
   - Task 2: "Fix: SQL injection vulnerability" -> Subagent fixes it
   - Task 3: "Fix: Incorrect error message" -> Subagent fixes it
6. Claude updates the plan file with "Review Fixes Applied" section:
   - Successfully Fixed: 3 issues
   - Files modified: auth.ts, db.ts, errors.ts
7. Reports final summary including fixes applied
```
