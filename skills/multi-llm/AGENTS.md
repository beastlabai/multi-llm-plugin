# Multi-LLM Skill — Development Guide

## Architecture

- Orchestrators produce JSON instructions for Claude Code to execute — they never modify code directly
- `generate-tasks`, `--status`, and `--init` have no Python orchestrator — handled via instruction files in `instructions/`
- **Config layering** (`provider_registry.load_config`): base `providers.yaml` → auto-discovered `<git-root>/.multi-llm/providers.yaml` → `MULTI_LLM_PROVIDERS_CONFIG` env override. The auto-discovered layer deep-merges its **full** contents — including a `providers:` block — over base, identical to the env layer (no longer filtered to selection keys). Deep-merge, lists replace, `None` skipped. Discovery anchors on the optional `anchor` (plan path threaded from `resolve_models`) then CWD; cache keyed on the resolved anchor. Present-but-invalid explicit overrides fail fast (`ConfigError`); `MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE=1` restores warn-and-skip. Config `command` is metadata-only and NEVER executed (binaries are hardcoded in `utils/providers/`); a merged provider name with no hardcoded adapter is ignored at runtime. Scaffold via `init_config.py` (`--init`, auto-detects installed CLIs).
- `ask` mode (`ask_orchestrator.py`) is a read-only free-text Q&A side-channel: markdown output, no JSON schema / grouping / validation, output under `{plan}/ask/<question-slug>-<hash8>/`. It shares concurrency mechanics with the review modes via `run_models_concurrent` in `review_orchestrator_base.py`.

## Developer Notes

- **Full model for validation**: Validation subagents use full model capability — never downgrade to haiku. Only the coordinator (simple orchestration) uses haiku.
- **Backup before modification**: Always backup files before any salvage or modification operation.

## Common Pitfalls

- Use `--project` not `--directory` with `uv run` (path duplication bug)
- Group IDs are 16-char hex content hashes — use `state_manager.py` hashing, don't generate random IDs
- All output paths are plan-relative for portability
- Validation statuses: `valid`, `invalid`, `needs-human-decision`, `validation_failed`
- Importance levels: HIGH > MEDIUM > LOW
- Reaggregation globs `{phase_dir}/*.json` and treats each file as a per-model result list. Non-model JSON files written into a phase dir (e.g. `report_data.json`, `state`, `.status`) MUST be added to every `exclude_patterns` list (`review_orchestrator_base.py`, `code_review_orchestrator.py`) or they get parsed as suggestions and crash reaggregate.
- `generate_html_report` persists `{phase_dir}/report_data.json` (the assembled report data) so the apply phase can overlay human/Claude decisions onto it and re-embed via `html_report_generator.py regenerate-decisions` — no reconstruction of runtime inputs (diff_data, base_ref, models).

## Testing

```bash
uv run --project skills/multi-llm -- pytest
uv run --project skills/multi-llm -- pytest tests/test_filtering.py -v
uv run --project skills/multi-llm -- pytest -m robustness
```

Markers (defined in `pyproject.toml`): `robustness` (error handling), `live` (requires CLI+API, skipped in CI), `timeout`
