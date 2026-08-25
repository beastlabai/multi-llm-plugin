<!-- The section below is GENERATED from skills/multi-llm/hosts/capabilities.yaml
     by skills/multi-llm/hosts/render_capabilities.py, between the two markers.
     Edit the YAML and re-render; CI byte-compares this file, so a hand-edit
     inside the markers cannot ship. The short version of this page is the
     README's "Codex and OpenCode" section. -->

<!-- BEGIN GENERATED: host-install (hosts/render_capabilities.py) -->

# Hosting multi-llm in other harnesses

multi-llm ships as a Claude Code plugin, and it runs unchanged inside `codex` and `opencode` as well. The workflow itself is host-neutral: the instruction core names three capability *verbs* (*spawn subagents*, *ask the user*, *run detached*) and no tool, and one dispatcher command runs every mode. What a host needs is therefore just an **entry shim** — a small file, in that host's own command/skill format, with this skill's absolute path and its `--host` id baked in. `install-host` renders and writes it:

```bash
# from anywhere; SKILL is this skill tree on disk
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host <host> [--global|--project]
```

`SKILL` is either a git checkout of this repository or the plugin install `claude plugin list --json` reports; the README's [Codex and OpenCode](../README.md#codex-and-opencode) block resolves it, with a `jq`-free fallback for machines without a JSON parser.

It writes a **rendered copy** — never a symlink or junction, so nothing here needs symlink privileges on Windows — atomically, then verifies what it wrote and rolls back if the verification fails. `--check` re-runs that verification on its own: shim present (every copy found, in the host's discovery order, naming the one that loads), baked skill-dir prefix still current, rendered shape valid, shim generation current, and `uv` plus the host's runner CLI resolvable.

## Per-host quickstart

**`codex`**

```bash
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host codex            # machine-wide
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host codex --project  # just this repository
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host codex --check    # verify an install
```

Writes `$HOME/.codex/skills/multi-llm/SKILL.md` (`--project`: `.codex/skills/multi-llm/SKILL.md`), then start a new codex session afterwards (it enumerates its commands/skills at session start). Invoke it by name rather than as a slash command — `Use the multi-llm skill with exactly these arguments: <the same arguments the Claude Code skill takes>`. Codex installs this shim as a **skill**, which is model-invoked rather than typed: its composer answers `/multi-llm` with `Unrecognized command`. Unattended is the default on every host but Claude Code, so `--yes` is baked in.

**`opencode`**

```bash
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host opencode            # machine-wide
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host opencode --project  # just this repository
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host opencode --check    # verify an install
```

Writes `$XDG_CONFIG_HOME/opencode/commands/multi-llm.md` (whose `$XDG_CONFIG_HOME` falls back to `~/.config` when unset) (`--project`: `.opencode/commands/multi-llm.md`), then start a new opencode session afterwards (it enumerates its commands/skills at session start). Invoke it as `/multi-llm <the same arguments the Claude Code skill takes>`. Unattended is the default on every host but Claude Code, so `--yes` is baked in.

Headless, the argument list needs the host's own separator: `opencode run --command multi-llm -- "<args>"`.

## Capability matrix

Generated from `skills/multi-llm/hosts/capabilities.yaml` — the same data `install-host` renders a shim from. Every value carries a **provenance**: `verified` means a recorded live run observed it, `documented` means host docs only, `unknown` is the shipping default, and a `documented`-only value never selects the non-conservative path (which is why hosts whose native subagent dispatch is merely documented ship as `executor`).

| Host | Installable | Command format | Args | *spawn subagents* | *ask the user* | *run detached* | Default runner | Reload |
|---|---|---|---|---|---|---|---|---|
| `claude-code` | no (ships with the plugin) | `markdown+frontmatter` | `placeholder` | `native` (`Task`) — verified | `structured` (`AskUserQuestion`) — verified | `native` (`run_in_background`) — verified | `claude-code:sonnet` | `none` |
| `codex` | yes | `markdown+frontmatter` | `placeholder` | `executor` (`agent`) — documented | `free-text` — documented | `python` — documented | `codex:gpt-5.2-codex` | `new-session` |
| `opencode` | yes | `markdown+frontmatter` | `placeholder` | `executor` (`task`) — documented | `free-text` — documented | `required` — verified | `opencode:opencode/sonnet` | `new-session` |

## Where the shim goes

One row per host **and scope**, not one path per host. `install-host` writes only the preferred path of the scope you asked for; it *detects and reports* the rest, so an existing install is upgraded rather than silently shadowed. Compatibility locations stay valid inputs and are never written: migrating is re-run `install-host`, then delete the file it names. `~` is `%USERPROFILE%` on Windows and the XDG roots honour `$XDG_CONFIG_HOME`; the tokens below are what the data stores.

| Host | Global (preferred) | Project (preferred) | Compatibility (detected, never written) |
|---|---|---|---|
| `claude-code` | *n/a* — the plugin's own `SKILL.md` | *n/a* | `.claude/skills/multi-llm`, `$HOME/.claude/skills/multi-llm` (non-plugin install) |
| `codex` | `$HOME/.codex/skills/multi-llm` | `.codex/skills/multi-llm` | `$HOME/.codex/prompts/multi-llm.md` |
| `opencode` | `$XDG_CONFIG_HOME/opencode/commands`, `$HOME/.config/opencode/commands` | `.opencode/commands` | — |

## After a plugin update

A plugin update **changes the skill directory path** — the Claude plugin cache is keyed by commit hash — so every installed shim's baked prefix goes stale. That is the common case, not an edge case, and it is one command:

```bash
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host --check --all   # what moved
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host --all           # re-render every installed shim
```

Repair is always re-render, never in-place patching, and a re-install with nothing changed is byte-identical and does nothing. A shim you edited by hand is **never** overwritten silently: the install stops and names the file, and `--force` keeps your version as `<shim>.local.bak`. The inverse is `uninstall-host <host> [--global|--project|--all] [--dry-run]`, which removes only files carrying our `multi_llm_shim` block — identified by that block, never by filename.

<!-- END GENERATED: host-install -->

---

Back to the [README](../README.md) · [Providers](providers.md) · [Per-project configuration](configuration.md)
