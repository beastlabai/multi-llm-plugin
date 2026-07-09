# Init Config Mode Instructions

Scaffolds a **per-project provider config override** at
`<git-root>/.multi-llm/providers.yaml`. This is the opt-in way to give a single
repository its own provider/model defaults without editing the installed plugin.

Like `--status` and `--generate-tasks`, `--init` has **no Python orchestrator** —
it is handled entirely by this instruction file routing to the standalone
`init_config.py` script.

`--init` is **fully automatic and zero-prompt**. It scans `PATH` for the
supported provider CLIs (claude-code→`claude`, cursor-agent→`cursor-agent`,
gemini→`gemini`, grok→`grok`, opencode→`opencode`, codex→`codex`,
kilocode→`kilocode`, cline→`cline`, goose→`goose`, aider→`aider`),
copies an **inert template** to the target path, then **uncomments** the lines
that belong to the detected providers — their full `providers:` sub-blocks
(command, timeouts, concurrency, `models:` list), the `defaults.models` /
`defaults.quick_models` candidate entries whose provider is detected, and
`default_provider`. Undetected providers stay commented. There are **no prompts**
and **no model-listing subprocess calls**, so it behaves identically inside
Claude Code, a plain terminal, or CI.

## Usage

```bash
/multi-llm:multi-llm --init [--dir PATH] [--force] [--gitignore] [--template-only]
```

`--init` does **not** take a plan path. Optional arguments:

| Arg | Meaning |
| --- | --- |
| `--dir PATH` | Target directory (defaults to the git root; falls back to CWD outside a repo, with a printed notice). |
| `--force` | Overwrite an existing `.multi-llm/providers.yaml` (the script refuses by default). |
| `--gitignore` | Also append `.multi-llm/` to the repo's `.gitignore` (idempotent) so the override stays developer-local and untracked. Default: leave it trackable (commit it for a team-wide standard). |
| `--template-only` | Skip detection entirely; write the pristine inert template verbatim (every provider left commented) for hand-editing. |

## Process (Claude Code executes these steps)

1. **Map flags.** Pass through any `--dir`, `--force`, `--gitignore` the user
   supplied. Use `--template-only` only when the user explicitly wants the
   untouched stub ("just write the template / give me the stub to edit / don't
   detect anything"). A bare `--init` runs auto-detection — no routing question
   is needed because nothing prompts.

2. **Run the scaffolder.** Substitute `${CLAUDE_SKILL_DIR}` with the absolute
   skill-directory path (per the SKILL.md path-resolution rules), then run:

   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/init_config.py [--dir PATH] [--force] [--gitignore] [--template-only]
   ```

3. **Report the result.** Print the path the script wrote and its git-tracking
   state (the script prints both), plus any notice it emitted (see below). If it
   refused to overwrite an existing file, relay that and offer `--force`.

## What gets written

- **`default_provider`** is set to the **first detected provider in base
  declaration order** (claude-code, cursor-agent, gemini, grok, opencode,
  codex, kilocode, cline, goose, aider). So with both claude-code and cursor-agent installed it becomes
  `claude-code`. When **nothing** is detected it is left commented and inherits
  the base default.
- **Nothing detected → no error.** With no provider CLIs on `PATH` the script
  writes the inert template (which inherits the built-in defaults) and exits 0
  with an install-guidance notice. It never fails.
- **Off-panel-only detection.** If the only detected providers are ones absent
  from the base default model panel (gemini / opencode), the script injects their
  first catalog model into `defaults.models` / `quick_models` so the file is
  runnable, and prints a notice saying so.
- **Re-running.** Re-run `--init --force` after installing a new CLI (to pick it
  up) or after a plugin update (to refresh the pinned provider metadata and prune
  providers the new base dropped) — see the DRIFT note below.

## Notes

- The override is **optional and auto-discovered** from the git root at run time;
  absent → built-in defaults, identical to today. Editing the generated file
  changes which providers/models a run selects.
- **Lists replace, they do not append.** A blank/omitted key inherits the base;
  an explicit empty list `[]` is the only way to deliberately empty a list.
- The generated file now carries a full **`providers:` block** for each detected
  provider. That block **deep-merges over the base**, identical to the
  `MULTI_LLM_PROVIDERS_CONFIG` env layer (lists replace, nested dicts deep-merge).
  Invariants that still hold:
  - `command:` is **documentation-only and is NEVER executed** — provider
    binaries are hardcoded in `utils/providers/`, so a `command:` value in any
    layer can never make the tool run a different program.
  - A merged provider name with **no hardcoded adapter** (e.g. one a newer plugin
    removed or renamed) is ignored at runtime — it lists no models and can never
    be selected.
  - **DRIFT:** a copied full block pins base's scalar/model values **at init
    time**; re-run `--init --force` after a plugin update to refresh the pinned
    metadata and prune dropped providers.
- One override per repository (discovery anchors at the git root). For per-run or
  out-of-tree overrides use `MULTI_LLM_PROVIDERS_CONFIG=/path.yaml`.

See the "Per-project configuration" section of the README for the full semantics.
