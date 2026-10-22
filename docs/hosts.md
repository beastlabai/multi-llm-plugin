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

`SKILL` is either a git checkout of this repository or the plugin install `claude plugin list --json` reports. Set it once — the first of these three that applies to you, not all three. On Windows the shell must be **Git Bash** (see [Prerequisites](../README.md#15-prerequisites)):

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

**`codex` — install via codex-marketplace (not yet published)** — documented

A second route on `codex`, and the one this documentation will recommend once it is published. **It is not yet published**: the `release` ref the first command names does not exist until the publish workflow first moves it, so today `codex plugin marketplace add … --ref release` fails **loudly**, with a git error naming the missing ref — nothing is half-installed. Until then, the `install-host codex` route above is what works.

**Read this before installing.** The plugin's tool runs the orchestrator from a server codex starts *outside* its own exec sandbox, so the session's `--sandbox` posture does not govern it: installing the plugin gives a codex session a write path it did not have. So under a read-only posture, or one it cannot read, the tool **refuses write-capable modes** before it spawns anything, unless `MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1` is in the environment that starts codex. Despite its name, `MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1` lifts only this tool's own refusal of write-capable modes under a read-only or unreadable session posture. It never lets the codex processes the orchestrator starts write without a sandbox, and it does not change which sandbox they run under: a codex editing job refused because codex's own sandbox cannot start on this host stays refused with it set. Provider CLIs the orchestrator dispatches stay under their own sandboxes.

Install it from the marketplace — two commands, in this order, then start a **new** codex session:

```bash
codex plugin marketplace add beastlabai/multi-llm-plugin --ref release --sparse .agents --sparse packaging/codex
codex plugin add multi-llm@beastlabai
```

Needs codex 0.159.2 or newer. It provides a skill and a model-callable tool, and you invoke it with a `$` mention of the namespaced skill — `$multi-llm:multi-llm <the same arguments the Claude Code skill takes>`; plugins add no slash command, so `/multi-llm` stays unrecognized. If the model ever reworks the arguments, spell it out — `Use the multi-llm:multi-llm skill with exactly these arguments: <args>` — or let the model call the `multi_llm` tool directly.

```text
$multi-llm:multi-llm plans/my-feature.md
$multi-llm:multi-llm --review-code plans/my-feature.md
```

Then configure providers once per repository, in the new session: `$multi-llm:multi-llm --init` writes `<git-root>/.multi-llm/providers.yaml`, outside `$CODEX_HOME`, where no reinstall or upgrade touches it.

Upgrading is a different sequence from the install, because repeating the install is a no-op on a marketplace that is already registered: refresh the marketplace, re-add the plugin from it, then start a **new** codex session:

```bash
codex plugin marketplace upgrade
codex plugin add multi-llm@beastlabai
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

A second route on `opencode`, and the one this documentation will recommend once it is published. **No published install has been verified yet**, so a config entry naming this spec resolves to nothing today — and on this host that failure is silent: the plugin stays listed, the exit code is 0, and the debug log is empty. Until then, use the `install-host` route above, or build the package and name its **directory** — `npm pack`'s `.tgz` is not an installable spec on this host: MEASURED 2026-09-19 against opencode 1.18.31, a config entry naming a tarball resolves to the archive's *containing directory* and never opens the archive, so it loads nothing unless that directory happens to be an unpacked package itself.

Add it to the host config (`~/.config/opencode/opencode.json`) and restart:

```json
{ "plugin": ["opencode-multi-llm@0.1.0"] }
```

Needs opencode 1.19.3 or newer. Upgrading is bumping the pin and restarting: the cache key is the literal spec string and the installer early-returns when the package directory already exists, so an **unpinned** entry resolves once and never again. It provides a slash command and a model-callable tool, and you invoke it as `/multi-llm <the same arguments the Claude Code skill takes>`, or by calling the `multi_llm` tool directly.

## What differs from Claude Code

Three differences worth knowing on these hosts:

- **Unattended is the default.** Claude Code is the only host with a structured question tool, so the shim bakes in `--yes`: every gate resolves from config and flags instead of prompting you. On the `install-host` file route, `--interactive` opts back into a rendered gate. The package route on `codex` and `opencode` refuses `--interactive` and `-i` instead — it spawns the dispatcher with nothing on its stdin, so the model picker they force could not prompt — and asks you to choose non-interactively: `--models "provider:model,…"` or `--quick`, or `defaults.models` in `providers.yaml`.
- **Subagents are dispatched through a provider CLI** rather than a native task tool, so the default runner is `codex:gpt-6-astra` on `codex` and `opencode:opencode/big-pickle` on `opencode` — not a Claude subagent.
- **Headless `opencode` needs its own separator:** `opencode run --command multi-llm -- "<args>"`.

## Capability matrix

Generated from `skills/multi-llm/hosts/capabilities.yaml` — the same data `install-host` renders a shim from. Every value carries a **provenance**: `verified` means a recorded live run observed it, `documented` means host docs only, `unknown` is the shipping default, and a `documented`-only value never selects the non-conservative path (which is why hosts whose native subagent dispatch is merely documented ship as `executor`).

| Host | Installable | Command format | Args | *spawn subagents* | *ask the user* | *run detached* | Default runner | Reload |
|---|---|---|---|---|---|---|---|---|
| `claude-code` | no (ships with the plugin) | `markdown+frontmatter` | `placeholder` | `native` (`Task`) — verified | `structured` (`AskUserQuestion`) — verified | `native` (`run_in_background`) — verified | `claude-code:sonnet` | `none` |
| `codex` | yes | `markdown+frontmatter` | `placeholder` | `executor` (`agent`) — documented | `free-text` — documented | `python` — documented | `codex:gpt-6-astra` | `new-session` |
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

## Codex plugin route

The README has the two install commands. This section has what they don't cover. **The route is not yet published**: the `release` ref the install names does not exist until the first release is promoted, so `--ref release` does not resolve yet and `codex plugin marketplace add` fails with a git error. **What works today is `install-host codex`** (the file route above).

**Floors.** These are published but not enforced, since nothing in the plugin checks them: **codex >= 0.159.2**, **python3 >= 3.9** on `PATH` under that exact name, and **uv** at any version. Neither install command installs dependencies. The first invocation builds the plugin's venv (`pyyaml` alone), which needs the network or a warm uv cache.

```bash
codex --version                     # 0.159.2 or newer
python3 --version && uv --version   # python3 3.9 or newer, under these exact names
```

**Configuration.** Once per repository, in the new session, run *"Use the `multi-llm:multi-llm` skill with exactly these arguments: `--init`"*. It writes `<git-root>/.multi-llm/providers.yaml`, outside `$CODEX_HOME`, where no plugin reinstall or upgrade touches it. Configure through that file, never the copy installed inside the plugin.

**What it runs and what it may write.** The tool starts the orchestrator from a server that Codex launches *outside* its own exec sandbox, so the session's `--sandbox` posture does not govern it. Installing the plugin therefore gives a Codex session a write path it did not have. Under a read-only posture, or one it cannot read, the tool **refuses write-capable modes** before it spawns anything, unless `MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1` is in the environment that starts Codex. Despite its name, `MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1` lifts only this tool's own refusal of write-capable modes under a read-only or unreadable session posture. It never lets the Codex processes the orchestrator starts write without a sandbox, and it does not change which sandbox they run under: a Codex editing job refused because Codex's own sandbox cannot start on this host stays refused with it set. Provider CLIs the orchestrator dispatches stay under their own sandboxes. One session posture does reach them (since 2026-09-30): when the Codex chat itself runs with `danger-full-access`, the tool hands that posture to the orchestrator on each call, and a Codex editing job the plugin's own jail cannot take then runs without a sandbox instead of being refused — never ahead of that jail, and only under `sandbox_fallback: auto` (see [configuration](configuration.md#safety)). **The plugin pre-approves its own tool** (`"default_tools_approval_mode": "approve"` in its `.mcp.json`), so Codex calls it without asking. Without that, a headless `codex exec` under a sandbox refuses the call outright. To be asked before every call, set `default_tools_approval_mode = "prompt"` under `[plugins."multi-llm@beastlabai".mcp_servers.multi_llm]` in `$CODEX_HOME/config.toml` (measured on codex 0.156.1: the call then needs approval again). The install itself is a Git clone and a directory copy with no signature, so you are trusting whoever can move the `release` ref and the workflow that moves it.

| Platform | Status | Prerequisites |
| --- | --- | --- |
| Linux | `verified` — the only platform this route claims | codex >= 0.159.2; `uv` and python3 >= 3.9 on `PATH` |
| macOS | `unmeasured` | codex >= 0.159.2; `uv` and python3 >= 3.9 on `PATH` (the command-line-tools `python3` is 3.9). Caveat: the run starts through a `bash -lc` login shell, which reads `~/.bash_profile` / `~/.profile` and **not** `~/.zprofile` or `~/.zshrc` — keys and `PATH` edits that live only in the zsh files never reach it |
| Windows | `unsupported for the tool route` — `python3` is usually absent under that name, so the plugin's server never starts and the tool never lists. Use `install-host codex` (the file route) instead | codex >= 0.159.2; `uv`; python3 >= 3.9 **under that exact name**; **plus Git Bash (or another POSIX `bash`) on `PATH` — a hard prerequisite**: without it the tool refuses with a missing-`bash` error naming the `PATH` it searched, rather than running without a login shell |

**Provider keys go in the file your login shell reads.** Codex starts the plugin's server with a stripped environment, so a key exported only in the terminal you launched Codex from never reaches the run. Put `export OPENROUTER_API_KEY=…` (and whatever else your providers read) in `~/.profile` or `~/.bash_profile`, not `~/.bashrc`, and on macOS not `~/.zprofile`. Do not edit the installed `.mcp.json` to pass keys: the next `codex plugin add` replaces it wholesale. When a key does not arrive, the `environment` block in the tool result (`login_shell`, `profile_shell`) says which startup file was read.

**When it does nothing.** Suppose the new session does not know the skill, or knows the skill but has no tool. Run these:

```bash
codex plugin list             # multi-llm@beastlabai — look at `enabled`; no entry means removed
codex mcp get multi_llm       # the server codex tried to start; python3 must be on PATH
python3 --version             # if this fails the server never started, and that is the whole story
```

`codex doctor` does not see plugin servers, so it cannot diagnose any of this.

**Upgrade:** run `codex plugin marketplace upgrade`, then `codex plugin add multi-llm@beastlabai`, then start a new session. Re-running the install's first command does nothing on a marketplace that is already added. **Coming from the file route:** remove the shim with `uninstall-host codex --global`, the counterpart of `install-host`. Otherwise both skills stay advertised (`install-host codex --check` warns when they are). **Uninstall:** `codex plugin remove multi-llm@beastlabai`. Your `.multi-llm/providers.yaml` is not touched. The plugin's own page is [`packaging/codex/README.md`](../packaging/codex/README.md).

## OpenCode npm route

The README has the one config line. **No published install has been verified yet.** Until `opencode-multi-llm` is on npm, an `opencode.json` entry naming it resolves to nothing, and on OpenCode that failure is silent: the plugin stays listed, the exit code is 0, and the debug log is empty. Use `install-host opencode` until then, or run a local build.

**Local build: name the directory, never the `.tgz`.** `make opencode-package` builds, packs and secret-sweeps the package into `packaging/opencode/`. Point `plugin[]` at that **directory** (or a `file://` URL of it) and restart:

```json
{ "plugin": ["/abs/path/to/packaging/opencode"] }
```

OpenCode never opens a tarball. A `.tgz` spec resolves to the directory that contains it, so it loads only when that directory happens to be an unpacked package.

**Checking that it loaded.** Every failure on this route exits 0, so neither the `plugin[]` entry nor an exit code proves anything. Check for the key the plugin adds:

```bash
opencode debug config | jq '.command["multi-llm"] != null'   # true = loaded
```

**First run.** The first `/multi-llm` after an install or upgrade needs the network or a warm `~/.cache/uv`, because it builds the package's venv (`pyyaml` alone).

**Upgrade and uninstall.** To upgrade, bump the pinned version and restart. There is no `opencode plugin update`, and an unpinned entry stays on the version it first resolved to. To uninstall, delete the `plugin` entry. The cache directory `~/.cache/opencode/packages/<spec>` stays on disk, and a later reinstall of the same spec silently reuses it, so `rm -rf` it first, while online. Coming from the file route, remove the old command with `uninstall-host opencode --global`. The package's own page, with cache and `MULTI_LLM_UV_RUN_ARGS` details, is [`packaging/opencode/README.md`](../packaging/opencode/README.md).

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

`multi_llm.py doctor` detects it (exit 3, naming the `bwrap:` line) and the executor refuses an edit-capable `codex` dispatch that no fallback rung takes before it starts (`sandbox_unavailable`, rung 1), leading with the way out that needs no change to the machine: a `driver.subagent_runner` that is not Codex. The system-level fix is one AppArmor profile of the shape Ubuntu already ships for chrome/Discord/1password, installed with sudo — the plugin prints it and never applies it:

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

then `sudo apparmor_parser -r /etc/apparmor.d/bwrap`. That grants the right to `/usr/bin/bwrap` alone; `sysctl kernel.apparmor_restrict_unprivileged_userns=0` also works but loosens every binary on the machine and is not recommended. Reviews are unaffected either way — the fan-out asks codex for its array on stdout rather than for a sandboxed file write, and without the repair codex's read-only phases take the `providers.codex.sandbox_fallback` ladder (codex's own landlock rung first, retired from codex 0.156.1; then, since 2026-09-30, the plugin's own Landlock jail, which stops stray writes, not the model; `danger-full-access` last; see [configuration](configuration.md#the-plugins-landlock-jail)). Of the write phases, only `apply-suggestions` and `apply-task-suggestions` run without this profile — on that jail, under `sandbox_fallback: auto` alone, writing only the plan file or the tasks directory, with the network not confined. For `apply-code-fixes` and `implement`, and for codex's own `workspace-write` boundary with its confined network, this profile remains the fix (a runner that is not Codex is the other way out); where the plugin's jail probes ok, `doctor` heads it as a system-level alternative. `grok`'s workspace-trust gate (`grok inspect` → `Project trusted: no`, dispatched with `--always-approve` and no trust flag) is an open question: whether it blocks a headless write is unmeasured, and `doctor` prints `trust: untested` for it until it is.

---

Back to the [README](../README.md) · [Providers](providers.md) · [Per-project configuration](configuration.md)
