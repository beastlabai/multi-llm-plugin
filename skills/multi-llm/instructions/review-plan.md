# Review Plan Mode Instructions

Reviews an implementation plan using multiple LLMs and aggregates their suggestions.

## Usage

```bash
/multi-llm:multi-llm --review-plan <plan_path> [--models provider:model ...] [--quick] [--interactive]
```

## Process

1. **Model Selection**: A fixed ladder, resolved in this order. Nothing is
   sniffed from the environment — interactivity is *requested*, never detected
   (D6b), so omitting models does **not** trigger a picker:
   - `--models provider:model ...` — use exactly those
   - `--interactive` — render the picker now (gum → fzf → numbered prompt); this
     is the only way a picker is reached
   - `--quick` — use `quick_models` from `providers.yaml`
   - otherwise **`defaults.models`** from the resolved config layering — the
     normal path, with no prompt
   - if none of those resolve, the run **fails closed with exit 3** naming the
     flags that answer the gate (D6 row 2); a picker is never substituted for it
2. **Run Reviews**: Execute the orchestrator with selected models
3. **Post-processing**:
   - Group similar suggestions across models
   - Validate each suggestion group (valid/invalid/needs-human-decision)
   - Generate reports (HTML + Markdown) for user review
4. **Generate Report**: Create `{prefix}_review_plan_report.md` with all findings

## Output Files

All output files are organized into phase-based subdirectories:

```
plans/todo/my-feature.md                         # Original plan (unchanged)
plans/todo/my-feature/                           # Output folder
├── state.json                                   # Session state (plan-local)
└── review-plan/                                 # Review plan phase outputs
    ├── report.md                                # Consolidated review report
    ├── grouped.json                             # Grouped suggestions
    ├── validation.json                          # Validation results
    ├── backup.md                                # Original plan backup
    ├── {model}.json                             # Per-model results
    ├── log_{model}.txt                          # Agent output logs (for debugging)
    ├── error_{model}.log                        # Error logs (if failed)
    └── salvage_{model}.json                     # Salvage requests (if JSON parse failed)
```

## Report Format and User Skip Functionality

The generated `report.md` uses a checkbox-based format that allows users to mark suggestions they want to skip:

### Suggestion Format

```markdown
### S001: Add error handling for API calls
- [ ] Skip
**Validation:** ✓ Valid | **Model:** cursor-agent:auto | **Type:** addition | **Section:** Step 3

Description of the suggestion...

---
```

### Validation Status Display

Each suggestion shows explicit validation status:
- `**Validation:** ✓ Valid` - LLM confirmed this is a real issue
- `**Validation:** ✗ Invalid` - LLM determined this is a false positive
- `**Validation:** ? Needs Review` - Requires human judgment
- `**Validation:** ? Unknown` - Validation status not available

### Marking Suggestions to Skip

To skip a suggestion when applying changes later:
1. Open `{plan}/review-plan/report.md` in any editor
2. Change `- [ ] Skip` to `- [x] Skip` for suggestions you want to exclude
3. Save the file
4. Run `--apply-suggestions` - marked suggestions will be filtered out

This is useful for:
- Skipping valid suggestions that don't fit your use case
- Excluding suggestions you've already addressed manually
- Filtering out low-priority items you don't want to apply

## Debug Logs

Each model invocation generates a log file (`log_{model}.txt`) in the `review-plan/` directory containing:
- Timestamp and duration
- The prompt sent to the agent
- Full stdout/stderr output
- Success/failure status and error details

These logs are useful for debugging why an agent failed or what thinking/tools it used.

## Options

- `--skip-validation`: Skip the LLM validation step (faster but no false-positive filtering)
- `--validation-model <model>`: Model to use for validation (default: "auto")
- `--quick` / `-q`: Use quick_models from providers.yaml for lightweight reviews (2 models)

## Validation Statuses

Each suggestion/issue group receives a validation status:

- **valid** (checkmark): Issue is real and should be addressed
- **invalid** (x): False positive, can be ignored
- **needs-human-decision** (?): Requires human judgment

Items with `needs-human-decision` status are included in the report for the user to review. Human decisions are handled later in the **apply-suggestions** phase — do NOT prompt the user for decisions during review-plan.

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
   - If models provided in args (e.g., `/multi-llm:multi-llm plans/foo.md --models provider:model-a provider:model-b`): use those directly
   - **With `--yes` (or `--non-interactive`)**: never ask. Pass no model flag at
     all and let the orchestrator resolve `defaults.models` from the config
     layering; it prints the resolved set as
     `[AUTO_RESOLVED] model-selection: defaults.models: …`. If nothing resolves
     (no `--models`, no `--quick`, no `defaults.models`, or every default
     provider unavailable) the orchestrator exits **3** naming the flags that
     answer it — that is the whole resolution, so do not substitute a question
     for it (D6 row 2).
   - If NO models provided and NOT `--yes`: **ask the user** to select models
     - Present available models from `providers.yaml` (multi-select)
     - Then run orchestrator with `--models <selected>`

3. **Resume Detection** (check BEFORE running orchestrator)

   Check for existing output to avoid expensive duplicate runs. Follow the detection cases in `references/resume-detection.md`, using `{plan}/review-plan/` as the phase directory.

   **Note**: The orchestrator guards against re-runs (exits with code 2 if phase is already marked complete in state.json). Use `--force` to override. This instruction-level resume detection is the FIRST line of defense; the orchestrator guard is backup protection.

4. **Run the orchestrator (DETACHED):**

   A fan-out review over many models routinely outlives any single shell command a host is willing to wait for, so **run detached**, redirecting stdout+stderr to a log file in the phase dir and setting `PYTHONUNBUFFERED=1`:
   ```bash
   PYTHONUNBUFFERED=1 uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-plan \
     --plan-file "$PLAN_PATH" --models <selected> \
     > "{plan}/review-plan/orchestrator-run.log" 2>&1
   ```

   **IMPORTANT**: Pass `$PLAN_PATH` as given — the orchestrator resolves it to an OS-native absolute path itself (do NOT wrap it in `$(realpath ...)`; on Git for Windows that emits a POSIX `/c/...` path a native process cannot use). Use `--project` (not `--directory`).

   **Run detached.** A detached run is not subject to your host's shell-command cap, so the orchestrator survives long multi-model runs. `PYTHONUNBUFFERED=1` is required so Python streams output to the log instead of block-buffering it (block-buffering leaves the log empty for minutes when stdout is a non-TTY pipe). When the background task completes, read all markers and output paths **from `{plan}/review-plan/orchestrator-run.log`**, not from terminal stdout.

   **Resume**: re-invoking with `--force` RESUMES (keeps already-completed per-model result files, runs only the missing models); `--rerun-all` forces a full re-run discarding existing per-model results.

5. **Interrupted-run Recovery** (conditional — only if the background run was interrupted or the log shows no completion markers)

   If the detached orchestrator was interrupted (e.g. the session ended) or `{plan}/review-plan/orchestrator-run.log` shows neither a validation marker nor a clean completion:

   1. Read `{plan}/review-plan/.status.json`
   2. Based on `state` field, determine progress:
      - `models_running` → check which `{model}.json` files exist in `{plan}/review-plan/` (those models completed)
      - `models_complete` / `grouping_complete` → check for `grouped.json`, `validation_tasks.json`
      - `validation_pending` → orchestrator completed normally; proceed to validation using `validation_tasks_path` from the status file, then reaggregate. Both the status file and the tasks file record the mode + plan path (`reaggregate_mode`, `plan_file`) rather than a resolved command, so a plugin update cannot leave a stale skill-dir path behind; `multi_llm.py status <plan> --view run` prints the command resolved against the running skill dir.
      - `complete` → report results
   3. If no `.status.json` → fall back to existing file-based resume detection (step 3)
   4. **With `--yes` (or `--non-interactive`): always resume, never ask** (D6
      row 3). Re-run only the missing models with `--force`; never `--rerun-all`
      (it discards work that succeeded). If a model fails a second time in the
      same run, reaggregate with what exists: the orchestrator prints
      `[PARTIAL_RESULTS: <model>,…]`, lists the failed models in the report and
      `.status.json`, sets `MULTI_LLM_RESULT.state: "partial"` and exits **0** —
      a partial aggregate is usable output. The one exception: if *no* model
      produced a result there is nothing to aggregate, so it exits **3** naming
      the retry command.
   5. Without `--yes`, **ask the user**, showing:
      - Which models completed successfully
      - Which models failed or were interrupted
      - Options: (a) Proceed with partial results (`--reaggregate`), (b) Resume — re-run only the missing models (`--force`), (c) Re-run everything from scratch (`--force --rerun-all`)

6. **Salvage Handling (Post-Orchestrator):**

   After the background task completes, check the log file `{plan}/review-plan/orchestrator-run.log` for `[SALVAGE_NEEDED]` markers. Follow the salvage process in `references/salvage-handling.md`.

   The reaggregation command for this mode is:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-plan \
     --plan-file "$PLAN_PATH" \
     --reaggregate
   ```

7. **Validation Handling (Post-Orchestrator):**

   After the background task completes, check the log file `{plan}/review-plan/orchestrator-run.log` for validation markers.

   #### The task file

   `validation_tasks.json` carries `tasks_schema_version: 2` and one complete
   task per batch under `batches` — each with `prompt`, `output_name`, the
   derived `output_path`, `result_channel: file`, `context_mode: inline`,
   `capability_tier`, `executor_role` and `task_id`. The plan excerpt and this
   batch's suggestion records travel **inside** `prompt`, so no subagent has to
   open a file.

   **For each task in the file, run `task.prompt` as a subagent, then write the
   JSON it returns to `task.output_path`.** That is the whole contract — do not
   assemble a prompt, and do not tell the subagent to write the file itself.

   A file whose `tasks_schema_version` is missing or not `2`, or whose recorded
   `inputs` hashes no longer match the plan and grouped.json, is refused with
   exit 3: re-run the phase with `--force` to re-materialize it.

   #### Single Batch: `[VALIDATION_PENDING]`

   If `[VALIDATION_PENDING]` marker is present:
   1. Read `validation_tasks.json` from `{plan}/review-plan/`
   2. Run the single task's `prompt` as a subagent and write its JSON reply to
      that task's `output_path`
   3. Run the reaggregation command under **Reaggregation (After Validation)** below. `validation_tasks.json` records `reaggregate_mode` + `plan_file`, never a resolved command, so the command is rebuilt from those two values

   #### Multiple Batches: `[VALIDATION_BATCHES_PENDING]`

   If `[VALIDATION_BATCHES_PENDING]` marker is present (for 5+ suggestion groups):
   1. Read `validation_tasks.json` once from `{plan}/review-plan/`
   2. Count the number of batches as `{total_batches}` and mark the phase start before spawning the first wave:
      ```bash
      uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics start \
        --state-file "$STATE_FILE" --phase "review-plan" --total-batches {total_batches}
      ```
      Validation runs in parallel waves — wall-clock ETA reflects actual elapsed time, while per-batch duration would overstate it. The `record` step below picks `min(per-item, wall-clock)` automatically.
   3. Choose a strategy:

   **Strategy A: Direct parallel spawning (≤ 4 batches)**

   For each task in the `batches` array, run `task.prompt` as a subagent and write the JSON it returns to that task's `output_path`.

   **Run all batches in parallel** (single message with multiple subagent calls).

   **Strategy B: Wave-based parallel spawning (> 4 batches)**

   Follow the wave-based spawning strategy in `references/wave-batching.md`.

   3. After ALL batches complete, run the reaggregation command below (rebuilt from `reaggregate_mode` + `plan_file`)

   #### Reaggregation (After Validation)

   After validation completes, rebuild the reaggregation command from `reaggregate_mode` + `plan_file` in validation_tasks.json — the file stores that durable pair rather than a resolved command, so a skill dir that moved with a plugin update cannot leave a dead path behind:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-plan \
     --plan-file "$PLAN_PATH" --reaggregate
   ```

   **CRITICAL**: Do NOT report results to the user until reaggregation is complete.

   **Legacy Mode**: To run validation inside the orchestrator (bypassing subagent delegation), add `--internal-validation` flag to the orchestrator command.

8. **Report results** following `references/summary-format.md`, including all generated files in `{plan}/review-plan/`:
   - `report.md` - Consolidated review report
   - `grouped.json` - Grouped suggestions
   - `validation.json` - Validation results
   - `backup.md` - Original plan backup

   **Metrics (optional):** After validation/salvage subagents complete, record their metrics. When the validation phase used multiple batches, include `--batch-index` (cumulative across waves: 1..N) and `--total-batches`:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics record \
     --state-file "$STATE_FILE" --phase "review-plan" \
     --label "Validation batch {N}" --subagent-type "general-purpose" \
     --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
     --total-batches {total_batches} --batch-index {N}
   ```
   Where `token_count`, `tool_uses`, `duration_ms` come from whatever your subagent mechanism reports. Omit any flag it does not report. Single-batch validation runs may omit `--total-batches`/`--batch-index`.

   After all batches and reaggregation complete, mark the phase finish (only if `start` was called):
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics finish \
     --state-file "$STATE_FILE" --phase "review-plan"
   ```

9. **Suggestion Consolidation (Optional):**

   After reporting results, check the log file `{plan}/review-plan/orchestrator-run.log` for a `[CONSOLIDATION_RECOMMENDED]` marker. The same fact rides the result contract for a caller that never sees the log — `MULTI_LLM_RESULT.consolidation_recommended` (`{valid_groups, threshold, command}`, present only when recommended, and carried by `status --view run` too) — so report it from there rather than inferring it from the marker's absence.

   If **not present**, skip this step entirely.

   If **present** with `--yes` (or `--non-interactive`): **auto-run consolidation
   once, no prompt** (D6 row 4). There is no second threshold to apply here — the
   orchestrator only emits the marker at `CONSOLIDATION_RECOMMENDED_THRESHOLD`
   (15 valid groups, tunable with `--consolidation-threshold`), so the marker's
   presence *is* the decision. If consolidation itself fails, keep the
   unconsolidated results, warn, and continue — a failed optional pass is not a
   failed phase.

   If **present** without `--yes`, inform the user of the group count and **ask the user** whether to run consolidation. If the user declines, skip this step.

   If consolidation is to run (auto or agreed):

   **a) Run the consolidation orchestrator:**

   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-plan --plan-file "$PLAN_PATH" --consolidate
   ```

   **b) Handle subagent spawning for consolidation batches:**

   Check the orchestrator output for markers:
   - `[CONSOLIDATION_PENDING]` — single batch (1 batch)
   - `[CONSOLIDATION_BATCHES_PENDING]` — multiple batches

   Then:
   1. Load `consolidation_tasks.json` from the phase directory (`{plan}/review-plan/`) — every task in its `batches` array carries a complete `prompt`
   2. Count the number of batches as `{total_batches}` and mark the consolidation phase start:
      ```bash
      uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics start \
        --state-file "$STATE_FILE" --phase "review-plan-consolidation" --total-batches {total_batches}
      ```
   3. For each task in the file, run `task.prompt` as a subagent — the filled `consolidate_suggestions.txt` template and that batch's group records are already inside it
   4. Write the JSON each subagent returns to that task's `output_path` (`consolidation_batch_N.json` in the phase directory)
   5. After each batch completes, record metrics with cumulative `--batch-index`:
      ```bash
      uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics record \
        --state-file "$STATE_FILE" --phase "review-plan-consolidation" \
        --label "Consolidation batch {N}" --subagent-type "general-purpose" \
        --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
        --total-batches {total_batches} --batch-index {N}
      ```

   **Strategy A (≤ 4 batches):** Run all batch subagents from the main agent in parallel (single message with multiple subagent calls).

   **Strategy B (> 4 batches):** Split batches into waves of up to 4 (last wave may absorb remaining batches, up to 6). For each wave, run all batch subagents in parallel (single message), wait for completion, then proceed to the next wave.

   **c) Run reaggregation after all batches complete:**

   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-plan --plan-file "$PLAN_PATH" --reaggregate-consolidation
   ```

   After reaggregation, mark the consolidation phase finish:
   ```bash
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util metrics finish \
     --state-file "$STATE_FILE" --phase "review-plan-consolidation"
   ```

   **CRITICAL**: Do NOT report consolidation results until reaggregation is complete.

   **d) Report all consolidation output paths:**
   - `consolidated.json` - Consolidated suggestion groups
   - `consolidated-report.md` - Human-readable consolidation report
   - `consolidated-report.html` - HTML version of the consolidation report

---

## Example Execution

```
User: /multi-llm:multi-llm plans/my-feature.md

Agent:
1. No models in args, so the agent asks the user:
   "Which models would you like to use for the review?" (multi-select)
   Options: auto, gpt-5.2, gemini-3-pro, grok, gemini-3-flash, kimi-k2
2. User selects: gpt-5.2, gemini-3-pro
3. Runs: uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-plan --plan-file plans/my-feature.md --models gpt-5.2 gemini-3-pro
4. Orchestrator outputs [VALIDATION_PENDING] marker
5. Claude reads plans/my-feature/review-plan/validation_tasks.json
6. For each task: Claude runs task.prompt as a subagent
7. Claude writes each subagent's JSON reply to that task's output_path
8. Claude runs reaggregation:
   uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-plan \
     --plan-file plans/my-feature.md --reaggregate
9. Claude reports all output files when complete
```
