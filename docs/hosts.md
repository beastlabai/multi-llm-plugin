<!-- The section below is GENERATED from skills/multi-llm/hosts/capabilities.yaml
     by skills/multi-llm/hosts/render_capabilities.py, between the two markers.
     Edit the YAML and re-render; CI byte-compares this file, so a hand-edit
     inside the markers cannot ship. The README points here; it does not keep a
     second copy of the install recipes. -->

<!-- BEGIN GENERATED: host-install (hosts/render_capabilities.py) -->

# Hosting multi-llm in other harnesses

multi-llm ships as a Claude Code plugin, and it runs unchanged inside `codex` and `opencode` as well. The workflow itself is host-neutral: the instruction core names three capability *verbs* (*spawn subagents*, *ask the user*, *run detached*) and no tool, and one dispatcher command runs every mode. What a host needs is therefore just an **entry shim** — a small file, in that host's own command/skill format, with this skill's absolute path and its `--host` id baked in. `install-host` renders and writes it:

```bash
# from anywhere; SKILL is this skill tree on disk
uv run --frozen --no-dev --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host <host> [--global|--project]
```

`SKILL` is either a git checkout of this repository or the plugin install `claude plugin list --json` reports. Set it once — the first of these three that applies to you, not all three. On Windows the shell must be **Git Bash** (see [Prerequisites](../README.md#prerequisites)):

```bash
# Set SKILL once - the first of these three that applies to you, not all three.

# 1. You cloned this repository:
SKILL=<checkout>/skills/multi-llm

# 2. You installed the plugin: ask Claude Code where it put it (needs `jq`). Installs of the
#    same plugin coexist per project and per version, and versions older than the dispatcher
#    are still in the cache - so prefer enabled and most-recently-updated, then take the first
#    that actually carries multi_llm.py:
SKILL=$(claude plugin list --json \
  | jq -r 'map(select(.id == "multi-llm@beastlabai")) | sort_by([.enabled, .lastUpdated]) | reverse | .[].installPath' \
  | while IFS= read -r p; do [ -f "$p/skills/multi-llm/multi_llm.py" ] && { echo "$p/skills/multi-llm"; break; }; done)

# 3. Same, but no `jq` on this machine: glob the cache, whose path changes on every update.
#    The marketplace segment is pinned to the one `/plugin install` used in the README - change it if
#    you installed from a fork:
SKILL=$(dirname "$(ls -t ~/.claude/plugins/cache/beastlabai/multi-llm/*/skills/multi-llm/multi_llm.py | head -1)")

# Then install, gated on SKILL: an unresolved SKILL is empty (2) or the current directory (3),
# and either would run install-host against the wrong tree.
if [ -f "$SKILL/multi_llm.py" ]; then
  uv run --frozen --no-dev --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host codex      # add --project to scope it to one repo
  uv run --frozen --no-dev --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host opencode   # add --check to verify an install
else
  echo "no multi-llm skill tree at '$SKILL' - install the plugin, or set SKILL=<checkout>/skills/multi-llm" >&2
fi
```

It writes a **rendered copy** — never a symlink or junction, so nothing here needs symlink privileges on Windows — atomically, then verifies what it wrote and rolls back if the verification fails. `--check` re-runs that verification on its own: shim present (every copy found, in the host's discovery order, naming the one that loads), baked skill-dir prefix still current, rendered shape valid, shim generation current, and `uv` plus the host's runner CLI resolvable.

## Per-host quickstart

**`codex`**

```bash
uv run --frozen --no-dev --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host codex            # machine-wide
uv run --frozen --no-dev --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host codex --project  # just this repository
uv run --frozen --no-dev --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host codex --check    # verify an install
```

Writes `$HOME/.codex/skills/multi-llm/SKILL.md` (`--project`: `.codex/skills/multi-llm/SKILL.md`), then start a new codex session afterwards (it enumerates its commands/skills at session start). Invoke it by name rather than as a slash command — `Use the multi-llm skill with exactly these arguments: <the same arguments the Claude Code skill takes>`. Codex installs this shim as a **skill**, which is model-invoked rather than typed: its composer answers `/multi-llm` with `Unrecognized command`. Unattended is the default on every host but Claude Code, so `--yes` is baked in.

```text
Use the multi-llm skill with exactly these arguments: plans/my-feature.md
Use the multi-llm skill with exactly these arguments: --review-code plans/my-feature.md
```

**`opencode`**

```bash
uv run --frozen --no-dev --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host opencode            # machine-wide
uv run --frozen --no-dev --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host opencode --project  # just this repository
uv run --frozen --no-dev --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host opencode --check    # verify an install
```

Writes `$XDG_CONFIG_HOME/opencode/commands/multi-llm.md` (whose `$XDG_CONFIG_HOME` falls back to `~/.config` when unset) (`--project`: `.opencode/commands/multi-llm.md`), then start a new opencode session afterwards (it enumerates its commands/skills at session start). Invoke it as `/multi-llm <the same arguments the Claude Code skill takes>`. Unattended is the default on every host but Claude Code, so `--yes` is baked in.

Headless, the argument list needs the host's own separator: `opencode run --command multi-llm -- "<args>"`.

```text
/multi-llm plans/my-feature.md
/multi-llm --review-code plans/my-feature.md
```

**`opencode` — install via npm (not yet published)** — documented

A second route on `opencode`, and the one this documentation will recommend once it is published. **No published install has been verified yet**, so a config entry naming this spec resolves to nothing today — and on this host that failure is silent: the plugin stays listed, the exit code is 0, and the debug log is empty. Until then, use the `install-host` route above, or build and install the tarball locally.

Add it to the host config (`~/.config/opencode/opencode.json`) and restart:

```json
{ "plugin": ["opencode-multi-llm@0.1.0"] }
```

Needs opencode 1.18.18 or newer. Upgrading is bumping the pin and restarting: the cache key is the literal spec string and the installer early-returns when the package directory already exists, so an **unpinned** entry resolves once and never again. It provides a slash command and a model-callable tool, and you invoke it as `/multi-llm <the same arguments the Claude Code skill takes>`, or by calling the `multi_llm` tool directly.

## What differs from Claude Code

Three differences worth knowing on these hosts:

- **Unattended is the default.** Claude Code is the only host with a structured question tool, so the shim bakes in `--yes`: every gate resolves from config and flags instead of prompting you. On the `install-host` file route, `--interactive` opts back into a rendered gate. The package route on `opencode` refuses `--interactive` and `-i` instead — it spawns the dispatcher with nothing on its stdin, so the model picker they force could not prompt — and asks you to choose non-interactively: `--models "provider:model,…"` or `--quick`, or `defaults.models` in `providers.yaml`.
- **Subagents are dispatched through a provider CLI** rather than a native task tool, so the default runner is `codex:gpt-5.2-codex` on `codex` and `opencode:opencode/big-pickle` on `opencode` — not a Claude subagent.
- **Headless `opencode` needs its own separator:** `opencode run --command multi-llm -- "<args>"`.

## Capability matrix

Generated from `skills/multi-llm/hosts/capabilities.yaml` — the same data `install-host` renders a shim from. Every value carries a **provenance**: `verified` means a recorded live run observed it, `documented` means host docs only, `unknown` is the shipping default, and a `documented`-only value never selects the non-conservative path (which is why hosts whose native subagent dispatch is merely documented ship as `executor`).

| Host | Installable | Command format | Args | *spawn subagents* | *ask the user* | *run detached* | Default runner | Reload |
|---|---|---|---|---|---|---|---|---|
| `claude-code` | no (ships with the plugin) | `markdown+frontmatter` | `placeholder` | `native` (`Task`) — verified | `structured` (`AskUserQuestion`) — verified | `native` (`run_in_background`) — verified | `claude-code:sonnet` | `none` |
| `codex` | yes | `markdown+frontmatter` | `placeholder` | `executor` (`agent`) — documented | `free-text` — documented | `python` — documented | `codex:gpt-5.2-codex` | `new-session` |
| `opencode` | yes | `markdown+frontmatter` | `placeholder` | `executor` (`task`) — documented | `free-text` — documented | `required` — verified | `opencode:opencode/big-pickle` | `new-session` |

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
uv run --frozen --no-dev --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host --check --all   # what moved
uv run --frozen --no-dev --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host --all           # re-render every installed shim
```

Repair is always re-render, never in-place patching, and a re-install with nothing changed is byte-identical and does nothing. A shim you edited by hand is **never** overwritten silently: the install stops and names the file, and `--force` keeps your version as `<shim>.local.bak`. The inverse is `uninstall-host <host> [--global|--project|--all] [--dry-run]`, which removes only files carrying our `multi_llm_shim` block — identified by that block, never by filename.

<!-- END GENERATED: host-install -->

## Linux sandboxed CLIs

Two provider CLIs put an OS sandbox between the model and the filesystem, and both use bubblewrap: `codex` (`--sandbox workspace-write`) and `muse` (its shell sandbox, which multi-llm now turns off with `--disable-sandbox`). On a stock Ubuntu 24.04 (`kernel.apparmor_restrict_unprivileged_userns = 1`, AppArmor 4.0) bubblewrap cannot create the user namespace it needs, so every sandboxed write dies before the model is reached — this is a property of the **machine**, not of any provider, and it is the same on every host that runs the CLI. Measured from a plain shell:

```
$ bwrap --ro-bind / / --dev /dev --unshare-net -- /bin/true
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
$ bwrap --ro-bind / / --dev /dev -- /bin/true
bwrap: setting up uid map: Permission denied
$ unshare -rn /bin/true
unshare: write failed /proc/self/uid_map: Operation not permitted
```

`multi_llm.py doctor` detects it (exit 3, naming the `bwrap:` line) and the executor refuses an edit-capable `codex` dispatch before it starts (`sandbox_unavailable`, rung 1). The fix is one AppArmor profile of the shape Ubuntu already ships for chrome/Discord/1password, installed with sudo — the plugin prints it and never applies it:

```
# /etc/apparmor.d/bwrap
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,

  # Site-specific additions and overrides. See local/README for details.
  include if exists <local/bwrap>
}
```

then `sudo apparmor_parser -r /etc/apparmor.d/bwrap`. That grants the right to `/usr/bin/bwrap` alone; `sysctl kernel.apparmor_restrict_unprivileged_userns=0` also works but loosens every binary on the machine and is not recommended. Reviews are unaffected either way — the fan-out asks codex for its array on stdout rather than for a sandboxed file write. `grok`'s workspace-trust gate (`grok inspect` → `Project trusted: no`, dispatched with `--always-approve` and no trust flag) is an open question: whether it blocks a headless write is unmeasured, and `doctor` prints `trust: untested` for it until it is.

---

Back to the [README](../README.md) · [Providers](providers.md) · [Per-project configuration](configuration.md)
