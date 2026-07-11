# Salvage Handling

After the orchestrator completes, check for `[SALVAGE_NEEDED]` markers in the output.

For each salvage request, use the Task tool to spawn a subagent:

```
Task tool call:
  subagent_type: general-purpose
  model: haiku  # Fast model for simple extraction
  description: "Salvage JSON from {model} output"
  prompt: |
    You are a JSON extraction specialist. Read the salvage request file and extract valid JSON.

    Salvage request file: {salvage_request_path}

    Steps:
    1. Read the salvage request JSON file
    2. From the `raw_output` field, extract the valid JSON array
       - Look for JSON arrays `[{...}, {...}]` in the text
       - If multiple arrays exist, use the last complete one
       - Fix common syntax errors (trailing commas, etc.)
    3. Validate each item has the required fields from `expected_schema`
    4. Backup any existing file at `output_path` before overwriting:
       uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/utils/backup.py" "{output_path}"
       - If the file exists, it will be copied to: `{filename}-BEFORE-SALVAGE-{timestamp}.json`
       - Report "Backed up existing file to: {backup_path}" if backup was created
    5. Write the valid JSON array to `output_path`
    6. Delete the salvage request file
    7. Report: number of items salvaged, backup path (if any), or error if failed
```

After ALL salvage subagents complete, run reaggregation. The orchestrator prints a `[REAGGREGATE_AFTER_SALVAGE]` marker with the exact command — copy and run it. This is mandatory: reaggregation regenerates grouped.json, validation.json, and report files to include the salvaged data. Do NOT report results to the user until reaggregation is complete.
