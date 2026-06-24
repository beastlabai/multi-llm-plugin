# Init Config Mode Instructions

Scaffolds a commented **per-project provider config override** at
`<git-root>/.multi-llm/providers.yaml`. This is the opt-in way to give a single
repository its own *selection* defaults (`default_provider`, `defaults.models`,
`defaults.quick_models`, `defaults.modes`) without editing the installed plugin.

Like `--status` and `--generate-tasks`, `--init` has **no Python orchestrator** —
it is handled entirely by this instruction file routing to the standalone
`init_config.py` script.

## Usage

```bash
/multi-llm:multi-llm --init [--dir PATH] [--force] [--gitignore]
```

`--init` does **not** take a plan path. Optional arguments:

| Arg | Meaning |
| --- | --- |
| `--dir PATH` | Target directory (defaults to the git root; falls back to CWD outside a repo, with a printed notice). |
| `--force` | Overwrite an existing `.multi-llm/providers.yaml` (the script refuses by default). |
| `--gitignore` | Also append `.multi-llm/` to the repo's `.gitignore` (idempotent) so the override stays developer-local and untracked. Default: leave it trackable (commit it for a team-wide standard). |

## Process (Claude Code executes these steps)

1. **Map the user's request to flags.** Pass through any `--dir`, `--force`,
   `--gitignore` the user supplied. If they ask to "set up a project config" with
   no extra detail, run with no flags (writes a committable stub at the git root).

2. **Run the scaffolder.** Substitute `${CLAUDE_SKILL_DIR}` with the absolute
   skill-directory path (per the SKILL.md path-resolution rules), then run:

   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/init_config.py [--dir PATH] [--force] [--gitignore]
   ```

3. **Report the result.** Print the path the script wrote and its git-tracking
   state (the script prints both). If it refused to overwrite an existing file,
   relay that and offer `--force`.

## Notes

- The override is **optional and auto-discovered** from the git root at run time;
  absent → built-in defaults, identical to today. Editing the generated file
  changes only which providers/models a run selects.
- **Lists replace, they do not append.** A blank/omitted key inherits the base;
  an explicit empty list `[]` is the only way to deliberately empty a list.
- Provider *definitions* are intentionally **not** overridable in this
  auto-discovered file — a `providers:` block there is ignored with a warning.
- One override per repository (discovery anchors at the git root). For per-run or
  out-of-tree overrides use `MULTI_LLM_PROVIDERS_CONFIG=/path.yaml`.

See the "Per-project configuration" section of the README for the full semantics.
