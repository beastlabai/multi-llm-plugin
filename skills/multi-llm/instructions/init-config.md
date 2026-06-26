# Init Config Mode Instructions

Scaffolds a commented **per-project provider config override** at
`<git-root>/.multi-llm/providers.yaml`. This is the opt-in way to give a single
repository its own *selection* defaults (`default_provider`, `defaults.models`,
`defaults.quick_models`, `defaults.modes`) without editing the installed plugin.

Like `--status` and `--generate-tasks`, `--init` has **no Python orchestrator** —
it is handled entirely by this instruction file routing to the standalone
`init_config.py` script.

`--init` has **three** outcomes. On a **real** TTY it is **interactive** by
default: it detects installed provider CLIs, lets the user pick the default +
`--quick` model panels (curated first, full CLI catalog opt-in behind "Show
all…"), and writes the chosen *selection* keys. With **no TTY** (inside Claude
Code) the **Claude-driven picker** below takes over as the canonical way to
choose models. Only on explicit stub intent (`--template-only` /
`--non-interactive`) does it write the commented template stub verbatim instead.

> **Inside Claude Code there is never a TTY.** Both the Bash tool and the `!`
> prompt prefix pipe stdout, so the script's `is_tty()` is always `False` — the
> `!`-prefix does **not** grant a TTY, so never advise it as a way to reach the
> interactive picker. To actually *choose* models from inside Claude Code, use the
> **Claude-driven picker** below (the canonical flow); the script's own TTY picker
> is reachable only from an external/standalone terminal.

## Usage

```bash
/multi-llm:multi-llm --init [--dir PATH] [--force] [--gitignore] \
    [--template-only] [--non-interactive] [--timeout SECONDS]
```

`--init` does **not** take a plan path. Optional arguments:

| Arg | Meaning |
| --- | --- |
| `--dir PATH` | Target directory (defaults to the git root; falls back to CWD outside a repo, with a printed notice). |
| `--force` | Overwrite an existing `.multi-llm/providers.yaml` (the script refuses by default). |
| `--gitignore` | Also append `.multi-llm/` to the repo's `.gitignore` (idempotent) so the override stays developer-local and untracked. Default: leave it trackable (commit it for a team-wide standard). |
| `--template-only` | Skip the interactive picker; write the commented template stub verbatim (today's behavior). |
| `--non-interactive` | Never prompt (CI/unattended). Implies `--template-only`; also implied automatically when not attached to a TTY. |
| `--timeout SECONDS` | Seconds to wait for a provider's `models` listing command during the interactive picker / a `--show-all` listing (default 10). |
| `--emit-catalog [--show-all PROVIDER] [--json]` | Detection/catalog **out**: print the providers/curated/catalog JSON to stdout and exit 0 without writing (read-only). Drives the Claude picker below. |
| `--from-selections FILE [--force] [--gitignore]` | Selection **in**: read a `{default_provider, models, quick_models}` JSON file and write the config through the standard writer. Drives the Claude picker below. |

## Process (Claude Code executes these steps)

1. **Route the user's intent (Decision D4), then map flags.** Pass through any
   `--dir`, `--force`, `--gitignore` the user supplied. Because **inside Claude
   Code there is never a TTY** (see the note above), the script's own picker is
   unreachable here, so route by *intent* rather than by TTY:
   - **Customize / interactive intent** — a bare `--init`, or "pick / choose / set
     up models / interactive / help me set up" → **drive the Claude picker**
     instead of the stub: skip step 2 and follow "Driving the picker from Claude
     Code (no TTY)" below.
   - **Stub / accept-defaults / CI intent** — "just write the template / don't
     prompt me / scaffold only / give me the stub to edit" → `--template-only`;
     "non-interactive / CI / unattended / headless" → `--non-interactive`. Only
     these explicit stub/CI intents take the scaffolder path in step 2.
   - "wait up to N seconds for the model list / the listing is slow" → `--timeout N`.
   - When genuinely ambiguous, ask **one** routing question first (via
     AskUserQuestion) before doing anything else.

2. **Run the scaffolder (stub/CI intent only).** For the picker path, skip this and
   use "Driving the picker from Claude Code (no TTY)" below (`--emit-catalog` /
   `--from-selections`). For an explicit stub/CI intent, substitute
   `${CLAUDE_SKILL_DIR}` with the absolute skill-directory path (per the SKILL.md
   path-resolution rules), then run:

   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/init_config.py [--dir PATH] [--force] [--gitignore] [--template-only] [--non-interactive] [--timeout SECONDS]
   ```

3. **Report the result.** Print the path the script wrote and its git-tracking
   state (the script prints both). If it refused to overwrite an existing file,
   relay that and offer `--force`.

## Driving the picker from Claude Code (no TTY)

Inside Claude Code the script's own picker can never run (no TTY, above). Instead
**Claude is the picker front-end**: collect the user's choices with the
AskUserQuestion tool, then materialize them through the script's existing, hardened
writer via two non-interactive JSON seams. Claude never hand-writes YAML — every
byte comes from `init_config.py`. This is the canonical, reproducible flow.

### 1. Route the intent (Decision D4)

- **Drive the picker** for a bare `--init`, or phrasings like "pick / choose / set
  up models / interactive / help me set up."
- **Keep the stub** (`--template-only`) for "just the stub / scaffold only / write
  the template / don't prompt me."
- When genuinely ambiguous, ask **one** routing question first (via
  AskUserQuestion) before doing anything else.

### 2. Emit the catalog (read-only, never writes)

Run `--emit-catalog --json` (pass through any `--dir`). It is a pure read — safe
even if the config already exists. The `--json` flag only silences a human note on
stderr; stdout is pure JSON either way.

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/init_config.py --emit-catalog --json [--dir PATH]
```

```json
{
  "target_dir": "<abs path>",
  "config_path": "<target_dir>/.multi-llm/providers.yaml",
  "config_exists": false,
  "default_provider_base": "cursor-agent",
  "git_tracked_default": true,
  "providers": [
    {"name": "claude-code", "available": true, "can_list_models": false, "curated": ["sonnet","opus","fable","haiku"]}
  ],
  "base_catalog": ["claude-code:fable", "..."],
  "shadowed_modes": []
}
```

- `providers[]` is in base-config declaration order. Offer **only** providers with
  `available: true` (their CLI is installed). `curated` is that provider's
  recommended bare model ids (show these first). `can_list_models: true` means a
  full CLI listing exists, so a "Show all…" escalation is possible.
- `base_catalog` is every known `provider:model` spec; use it to annotate any pick
  **not** in it as `(unverified id)` in the confirmation summary.
- `shadowed_modes` lists modes whose per-mode list still **wins** over the globals
  you are writing; if non-empty, surface that note before writing (see §4).

### 3. The progressive AskUserQuestion script (the crux)

AskUserQuestion shows **at most 4 questions per call** and **2–4 options per
question**, plus a free-text "Other". A curated list (4–15 entries) overflows one
question and a full catalog (hundreds) is hopeless as raw options — so the flow must
be **progressive**, not a flat enumeration.

- **Per-provider headline choice (≤4 options).** For each available provider ask
  e.g. *"Use the recommended models for `<provider>`?"* with options: **Use
  recommended** (the first 1–2 curated specs) / **Let me pick specific models** /
  **Skip this provider** / free-text "Other" (= type model ids). Batch up to 4
  providers across the 4 questions in a single AskUserQuestion call.
- **Drill-down only on "Let me pick".** Present that provider's curated models in
  batches of ≤4 options with `multiSelect: true`, paginating across multiple
  questions when curated > 4. **Soft-cap the pagination** (Decision D3: ~2 batches),
  then fall back to offering "use recommended" + free-text to avoid question
  fatigue. The "Other" free-text captures a manually typed id.
- **"Show all…" is a deliberate escalation.** Only when the user explicitly wants to
  browse beyond curated, run:

  ```bash
  uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/init_config.py --emit-catalog --show-all <provider> --json [--timeout SECONDS]
  ```

  This re-runs emit-catalog and attaches `"full": [...]` (all CLI model ids, curated
  floated to top) and `"note": <str|null>` to **only** that provider's entry. It is
  the only path that runs a listing command; it is timeout-bounded and never crashes
  on failure/timeout/auth (falls back to curated + a note). Offer the returned ids
  through the **"Other" free-text** field (fuzzy-match the user's typed substring
  against `full`) — never enumerate hundreds of options.

Collect across providers into `default_provider` / `models` (ordered
`provider:model` specs) / `quick_models`.

### 4. Parity requirements you must honor (protect the user from a broken config)

- **Zero-selection brick guard.** If the user selects nothing, **re-ask** — never
  produce an empty `models`. (`--from-selections` also hard-fails on empty `models`
  as a backstop, exit 1.)
- **`--quick` default.** Default `quick_models` to the first ~2 of `models`. Writing
  `quick_models: []` (disable `--quick`) requires an explicit confirmation question.
- **Unverified-id flag.** In the confirmation summary, annotate any pick that is
  manual / from "Show all…" **or** absent from `base_catalog` as `(unverified id)`.
- **Mode-shadowing note.** If `shadowed_modes` is non-empty, surface the "those
  per-mode lists still **win** over these globals" note **before** writing.
- **default_provider.** Default it to the provider of the first picked model.

### 5. Confirm + write

Show a confirmation summary (with the `(unverified id)` annotations and any
shadowed-modes note), then write the selection JSON to your **scratchpad** and run
`--from-selections <file> --force` (add `--gitignore` if the user wants the override
developer-local/untracked).

```json
{"default_provider": "claude-code", "models": ["claude-code:opus","cursor-agent:composer-2.5"], "quick_models": ["claude-code:opus"]}
```

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/init_config.py --from-selections <scratchpad-file> --force [--gitignore] [--dir PATH]
```

`--from-selections` routes through the **same** writer the TTY picker uses
(managed-block markers, outside-marker preservation on re-init, parse-gate,
post-write recheck, gitignore report). `models` must be **non-empty**; `quick_models`
defaults to `[]` if omitted (writing `[]` deliberately disables `--quick`). Each spec
must be well-formed `provider:model` and `default_provider` must be a known provider
— it exits 1 on malformed JSON / unreadable file / empty `models` / malformed spec /
unknown `default_provider`, and honors the shared "file exists and no `--force`"
refusal. Report the written path + git-tracking state from the script's own output.

### 6. v1 scope (Decision D5)

The Claude picker writes only the **global** selection keys (`default_provider` /
`defaults.models` / `defaults.quick_models`), matching the TTY picker. Per-mode
`defaults.modes.<mode>` overrides are a follow-up the user can hand-edit.

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
