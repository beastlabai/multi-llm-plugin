# Golden Baseline Snapshots

Pre-refactor baseline snapshots for the three apply orchestrators, captured
for artifact parity validation during and after the ApplyOrchestratorBase
refactor.

## Capture Details

- **Capture date**: 2026-03-14
- **Commit hash**: `b50260c71d204c9f67a948168f78cb8591f18879`
- **Branch**: `master` (pre-refactor state)

## Orchestrators Covered

1. `apply_suggestions_orchestrator.py` — applies validated plan review suggestions
2. `apply_code_fixes_orchestrator.py` — applies validated code review fixes
3. `apply_task_suggestions_orchestrator.py` — applies validated task review suggestions

## Artifact Types

### `--help` output
CLI argument parser output for each orchestrator. Verifies that CLI interfaces
remain identical after refactor.

- `apply_suggestions_help.txt`
- `apply_code_fixes_help.txt`
- `apply_task_suggestions_help.txt`

**These are normalized, not raw captures.** Since D1 each parser's `epilog`
prints its example commands with the skill dir *resolved from `__file__`* (no
`${CLAUDE_SKILL_DIR}` — no shell but Claude Code's expands it, so on Codex or
OpenCode a copied example would die before Python started). A raw capture would
therefore bake the capturing machine's absolute path into a checked-in fixture,
so `capture_baselines.normalize_skill_dir` rewrites it back to the `{SKILL_DIR}`
template token. Compare against live `--help` by rendering the fixture forward
with the skill's own renderer — one round-trip, no second normalization rule:

```python
cli_common.render_instruction_text(golden_text, cli_common.skill_dir())
```

### `--dry-run` output
Console output (stdout + stderr) from running each orchestrator with `--dry-run`
against representative test fixtures. Captures batch summaries, filtering
decisions, and item counts.

- `apply_suggestions_dry_run_stdout.txt` / `_stderr.txt` / `_exit_code.txt`
- `apply_code_fixes_dry_run_stdout.txt` / `_stderr.txt` / `_exit_code.txt`
- `apply_task_suggestions_dry_run_stdout.txt` / `_stderr.txt` / `_exit_code.txt`

### JSON output (`orchestrator_output.json`)
Full output JSON from each orchestrator run, capturing schema structure, keys,
value types, nesting, batch count, batch contents, and ordering.

- `apply_suggestions_output.json` — stdout JSON from `--output-format json`
- `apply_suggestions_orchestrator_output.json` — file written to disk
- `apply_code_fixes_output.json` — stdout JSON from `--output-format json`
- `apply_code_fixes_orchestrator_output.json` — file written to disk
- `apply_task_suggestions_orchestrator_output.json` — file written to disk
- `apply_task_suggestions_stdout.txt` — stdout from normal run

### Report-parsing intermediates
Parsed user edits, skips, and validation overrides from report.md for each
orchestrator, serialized as JSON snapshots.

- `apply_suggestions_report_parsing.json`
- `apply_code_fixes_report_parsing.json`
- `apply_task_suggestions_report_parsing.json`

### Metadata
- `capture_metadata.json` — capture date, commit hash, file manifest

## Regenerating Baselines

To regenerate all golden files from the current code state:

```bash
uv run --project skills/multi-llm -- python \
    skills/multi-llm/tests/golden/capture_baselines.py
```

The script is idempotent and will overwrite existing golden files.

## Usage in Parity Validation

During the refactor, compare post-refactor outputs against these golden files
to ensure behavioral parity:

1. **CLI interface**: Diff `--help` output against golden help files
2. **Dry-run output**: Compare stderr markers, counts, and batch summaries
3. **JSON schema**: Validate that output JSON has identical keys, types, and nesting
4. **Report parsing**: Ensure skipped items, overrides, and edits are parsed identically

Note: Some fields in the JSON outputs are environment-specific (e.g., `plan_file`
paths contain `/tmp/` prefixes, `timestamp` values). Parity checks should
normalize or ignore these fields.
