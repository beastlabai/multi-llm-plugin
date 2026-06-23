# Review Plan Mode Instructions

Reviews an implementation plan using multiple LLMs and aggregates their suggestions.

## Usage

```bash
/multi-llm:multi-llm --review-plan <plan_path> [model1 model2 ...] [--quick]
```

## Process

1. **Model Selection**: Models can be provided as command-line arguments. If not provided, interactive selection is triggered:
   - Tries **gum** first (provides interactive menu with multi-select)
   - Falls back to **fzf** (fuzzy finder with TAB for multi-select)
   - Falls back to **numeric prompt** (displays numbered list, accepts space-separated numbers)
   - In non-interactive environments (no TTY), selection fails with an error
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

1. **Validate plan file exists**

2. **Model selection:**
   - If models provided in args (e.g., `/multi-llm:multi-llm plans/foo.md gpt-5.2 gemini-3-pro`): use those directly
   - If NO models provided: Claude MUST use AskUserQuestion to prompt user for model selection
     - Present available models from `providers.yaml` (multi-select)
     - Then run orchestrator with `--models <selected>`

3. **Resume Detection** (check BEFORE running orchestrator)

   Check for existing output to avoid expensive duplicate runs. Follow the detection cases in `references/resume-detection.md`, using `{plan}/review-plan/` as the phase directory.

   **Note**: The orchestrator guards against re-runs (exits with code 2 if phase is already marked complete in state.json). Use `--force` to override. This instruction-level resume detection is the FIRST line of defense; the orchestrator guard is backup protection.

4. **Run the orchestrator:**
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_plan_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --models <selected>
   ```

   **IMPORTANT**: Always use `$(realpath "$PLAN_PATH")` to convert to absolute path.

   **IMPORTANT**: Run this command in the FOREGROUND (do NOT use `run_in_background`). Use `timeout: 1200000` (20 minutes) to allow enough time for all models to complete. The orchestrator manages parallelism internally.

   **TIMEOUT RECOVERY**: If this command times out or returns "(No output)", do NOT re-run. Go to step 5 (Timeout Recovery).

5. **Timeout Recovery** (conditional — only if Bash returns "(No output)" or timeout)

   If the orchestrator command from step 4 times out or returns "(No output)":

   1. Read `{plan}/review-plan/.status.json`
   2. Based on `state` field, determine progress:
      - `models_running` → check which `{model}.json` files exist in `{plan}/review-plan/` (those models completed)
      - `models_complete` / `grouping_complete` → check for `grouped.json`, `validation_tasks.json`
      - `validation_pending` → orchestrator completed normally; proceed to validation using `validation_tasks_path` and `reaggregate_command` from the status file
      - `complete` → report results
   3. If no `.status.json` → fall back to existing file-based resume detection (step 3)
   4. **Use AskUserQuestion** to inform the user:
      - Which models completed successfully
      - Which models failed or were interrupted
      - Options: (a) Proceed with partial results (`--reaggregate`), (b) Re-run missing models only, (c) Re-run everything with longer timeout

6. **Salvage Handling (Post-Orchestrator):**

   After the orchestrator completes, check for `[SALVAGE_NEEDED]` markers in the output. Follow the salvage process in `references/salvage-handling.md`.

   The reaggregation command for this mode is:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_plan_orchestrator.py \
     --plan-file "$(realpath "$PLAN_PATH")" \
     --reaggregate
   ```

7. **Validation Handling (Post-Orchestrator):**

   After the orchestrator completes, check for validation markers in the output.

   #### Reference-Based Validation

   The `validation_tasks.json` file uses a reference-based format (small metadata file):
   ```json
   {
     "batches": [
       {
         "batch_index": 0,
         "group_indices": [0, 1, 2],
         "output_path": "{plan}/review-plan/validation_batch_0.json"
       }
     ],
     "grouped_file": "{plan}/review-plan/grouped.json",
     "plan_file": "/absolute/path/to/plan.md",
     "reaggregate_command": "uv run ... --reaggregate"
   }
   ```

   #### Single Batch: `[VALIDATION_PENDING]`

   If `[VALIDATION_PENDING]` marker is present:
   1. Read `validation_tasks.json` from `{plan}/review-plan/` (small metadata file)
   2. Spawn one validation subagent that reads the source files directly:

      ```
      Task tool call:
        subagent_type: general-purpose
        description: "Validate plan review suggestions"
        prompt: |
          Validate suggestion groups from a plan review.

          Read these files:
          - Grouped suggestions: {grouped_file from validation_tasks.json}
          - Plan context: {plan_file from validation_tasks.json}

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
   4. Run reaggregation using the command from `reaggregate_command` in validation_tasks.json

   #### Multiple Batches: `[VALIDATION_BATCHES_PENDING]`

   If `[VALIDATION_BATCHES_PENDING]` marker is present (for 5+ suggestion groups):
   1. Read `validation_tasks.json` once from `{plan}/review-plan/`
   2. Count the number of batches as `{total_batches}` and mark the phase start before spawning the first wave:
      ```bash
      uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py start \
        --state-file "$STATE_FILE" --phase "review-plan" --total-batches {total_batches}
      ```
      Validation runs in parallel waves — wall-clock ETA reflects actual elapsed time, while per-batch duration would overstate it. The `record` step below picks `min(per-item, wall-clock)` automatically.
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
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_plan_orchestrator.py \
     --plan-file "$(realpath "$PLAN_PATH")" --reaggregate
   ```

   **CRITICAL**: Do NOT report results to the user until reaggregation is complete.

   **Legacy Mode**: To run validation inside the orchestrator (bypassing subagent delegation), add `--internal-validation` flag to the orchestrator command.

8. **Report results** including all generated files in `{plan}/review-plan/`:
   - `report.md` - Consolidated review report
   - `grouped.json` - Grouped suggestions
   - `validation.json` - Validation results
   - `backup.md` - Original plan backup

   **Metrics (optional):** After validation/salvage subagents complete, record their metrics. When the validation phase used multiple batches, include `--batch-index` (cumulative across waves: 1..N) and `--total-batches`:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py record \
     --state-file "$STATE_FILE" --phase "review-plan" \
     --label "Validation batch {N}" --subagent-type "general-purpose" \
     --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
     --total-batches {total_batches} --batch-index {N}
   ```
   Where `token_count`, `tool_uses`, `duration_ms` come from the Task tool result. Omit any flags the Task tool didn't return. Single-batch validation runs may omit `--total-batches`/`--batch-index`.

   After all batches and reaggregation complete, mark the phase finish (only if `start` was called):
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py finish \
     --state-file "$STATE_FILE" --phase "review-plan"
   ```

9. **Suggestion Consolidation (Optional):**

   After reporting results, check the orchestrator output for a `[CONSOLIDATION_RECOMMENDED]` marker.

   If **not present**, skip this step entirely.

   If **present**, inform the user of the group count and ask whether they want to run consolidation (use AskUserQuestion). If the user declines, skip this step.

   If the user agrees:

   **a) Run the consolidation orchestrator:**

   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_plan_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --consolidate
   ```

   **b) Handle subagent spawning for consolidation batches:**

   Check the orchestrator output for markers:
   - `[CONSOLIDATION_PENDING]` — single batch (1 batch)
   - `[CONSOLIDATION_BATCHES_PENDING]` — multiple batches

   Then:
   1. Load `consolidation_tasks.json` from the phase directory (`{plan}/review-plan/`)
   2. Read the `consolidate_suggestions.txt` prompt template from `${CLAUDE_SKILL_DIR}/prompts/`
   3. Count the number of batches as `{total_batches}` and mark the consolidation phase start:
      ```bash
      uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py start \
        --state-file "$STATE_FILE" --phase "review-plan-consolidation" --total-batches {total_batches}
      ```
   4. For each batch, spawn a Task subagent (`subagent_type: general-purpose`) with the prompt template filled with that batch's groups data
   5. Instruct each subagent: **"Respond with ONLY the JSON output. No explanation."**
   6. Save each subagent's output to `consolidation_batch_N.json` in the phase directory
   7. After each batch completes, record metrics with cumulative `--batch-index`:
      ```bash
      uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py record \
        --state-file "$STATE_FILE" --phase "review-plan-consolidation" \
        --label "Consolidation batch {N}" --subagent-type "general-purpose" \
        --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
        --total-batches {total_batches} --batch-index {N}
      ```

   **Strategy A (≤ 4 batches):** Spawn all batch subagents directly from the main agent in parallel (single message with multiple Task tool calls).

   **Strategy B (> 4 batches):** Split batches into waves of up to 4 (last wave may absorb remaining batches, up to 6). For each wave, spawn all batch subagents in parallel (single message with multiple Task calls), wait for completion, then proceed to the next wave.

   **c) Run reaggregation after all batches complete:**

   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_plan_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --reaggregate-consolidation
   ```

   After reaggregation, mark the consolidation phase finish:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py finish \
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

Claude:
1. No models in args, so Claude uses AskUserQuestion:
   "Which models would you like to use for the review?" (multi-select)
   Options: auto, gpt-5.2, gemini-3-pro, grok, gemini-3-flash, kimi-k2
2. User selects: gpt-5.2, gemini-3-pro
3. Runs: uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_plan_orchestrator.py --plan-file "$(realpath plans/my-feature.md)" --models gpt-5.2 gemini-3-pro
4. Orchestrator outputs [VALIDATION_PENDING] marker
5. Claude reads plans/my-feature/review-plan/validation_task.json
6. Claude spawns validation subagent:
   Task tool call:
     subagent_type: general-purpose
     description: "Validate plan review suggestions"
     prompt: <prompt from validation_task.json>
7. Subagent completes and writes validation.json
8. Claude runs reaggregation:
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_plan_orchestrator.py \
     --plan-file "$(realpath plans/my-feature.md)" --reaggregate
9. Claude reports all output files when complete
```
