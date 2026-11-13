# Installation details

What the README's install steps leave out: what the Claude Code registration does,
updating the plugin, moving an existing registration to sparse paths, the boundary
with Codex, the prerequisites in full, and the plugin's own `providers.yaml`. The
Codex plugin and the OpenCode package have the rest of theirs in
[Hosts](hosts.md#codex-plugin-route) (and [the OpenCode npm
route](hosts.md#opencode-npm-route)). Back to the [README](../README.md).

---

## Registering the Claude Code marketplace

Run this in a shell, then reload and run `--init` in Claude Code as the README shows:

```bash
claude plugin marketplace add beastlabai/multi-llm-plugin \
  --sparse .claude-plugin packaging/claude/multi-llm
claude plugin install multi-llm@beastlabai
```

Two separate improvements keep this small:

- **A small plugin directory limits the installed copy.** The marketplace catalog, `.claude-plugin/marketplace.json`, selects `packaging/claude/multi-llm`, a generated directory holding only the files the plugin runs. Installing copies that directory and nothing else, however you registered.
- **Sparse and partial fetching limit the initial repository download.** Registering a marketplace clones its repository into a marketplace cache. `--sparse` makes that a *sparse checkout*, which writes only the listed paths and a few small files from their parent directories. Over `https` and `ssh` it is also a *partial clone*, which does not fetch the contents of other files, so the test files stay on the server.

| | With `--sparse` | Without `--sparse` |
| --- | --- | --- |
| Installed plugin | 234 files, about 10.4 MB | The same |
| Downloaded by registering and installing | About 65% less (about 3.2 MB when measured) | About 9.2 MB when measured |
| Marketplace cache | The plugin, plus small files such as `README.md` and `Makefile`. No test files. | The whole repository, about 60 MB |

Before the plugin had its own directory, an install copied the whole repository: 2,363 files, 53.1 MB.

A registration without sparse options still works and installs the same plugin, but it fetches development files that never reach the installed copy. The in-session `/plugin marketplace add beastlabai/multi-llm-plugin` is that kind of registration.

`--sparse` takes both paths after one flag and must come after the source. Written before it, the command stops with `error: missing required argument 'source'`. Only the shell command takes `--sparse`: going by the CLI's own text, the in-session command reads everything after `add` as one source.

## Updating the Claude Code plugin

> **Already added the marketplace before (e.g. in another project)?** `/plugin marketplace add` does **not** refetch an existing marketplace - it silently reuses the locally cached copy, so `/plugin install` will give you a stale version. Run the update steps below instead.

These steps work for registrations with or without sparse paths. In a shell:

```bash
claude plugin marketplace update beastlabai
claude plugin update multi-llm@beastlabai                  # one scope only
cd /path/to/project && claude plugin update multi-llm@beastlabai --scope project   # each other installation, or --scope local
```

Then restart Claude Code, or run this in the session:

```text
/reload-plugins
/reload-skills
```

- **`marketplace update` comes first.** It is the step that pulls the latest commits. Without it, `plugin update` says the plugin is already at the latest version even after `master` has moved.
- **Each installation updates separately.** With no `--scope`, `plugin update` moves one installation. Every other one needs its own command with an explicit `--scope`, run from its own directory. `claude plugin list --json` lists them with their scope and project path.
- **Refresh each repository's config.** In each repository where you use multi-llm, run `/multi-llm:multi-llm --init --force`. It refreshes the provider metadata pinned in `.multi-llm/providers.yaml` and drops providers the new version removed. It also regenerates the selection block between the file's markers, so hand-edits there are replaced. The previous file is saved beside it as `providers.yaml.bak.<timestamp>`.

**What happens to the old cached copy.** Updating from a version that installed the whole repository leaves that copy, about 53 MB, in `plugins/cache/beastlabai/multi-llm/<old-version>/`. Claude Code marks it with an `.orphaned_at` file once the last installation has left it. No `claude plugin` command deleted it in testing. The CLI's own text suggests a session removes it after 14 days, but that has not been measured.

Updating does not change the marketplace cache. A registration without sparse paths gets the small plugin but keeps its whole-repository checkout until you move it, as described next. Updating and moving work in either order.

## Moving an existing registration to sparse paths

This is optional: updating already gets you the small plugin. Moving also shrinks the marketplace cache and later fetches. It happens in place, and the plugin stays installed. It was tested across 4 declarations and 7 installations (user, project and local scope, in several projects, enabled and disabled), and every installation kept its saved options, secrets and data.

A *declaration* is the `extraKnownMarketplaces.beastlabai` entry in a settings file. Change every declaration: a stale project declaration may undo the move, and that case could not be measured without a live session.

First find what needs changing. Run this from each project:

```bash
jq -r 'select(.extraKnownMarketplaces.beastlabai) | input_filename' \
  "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/settings.json" .claude/settings.json .claude/settings.local.json 2>/dev/null
claude plugin list --json | jq -r '.[] | select(.id=="multi-llm@beastlabai") | "\(.scope)\t\(.projectPath // "-")"'
```

Then change them:

```bash
# 1. The user declaration, from any directory. Skip this step if your user settings do not declare it.
settings="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/settings.json"
jq '.extraKnownMarketplaces.beastlabai.source.sparsePaths = [".claude-plugin","packaging/claude/multi-llm"]' \
  "$settings" > "$settings.tmp" && mv "$settings.tmp" "$settings"
claude plugin marketplace add beastlabai/multi-llm-plugin --sparse .claude-plugin packaging/claude/multi-llm

# 2. Each project or local declaration, from that exact directory. No edit needed.
cd /path/to/project
claude plugin marketplace add beastlabai/multi-llm-plugin --sparse .claude-plugin packaging/claude/multi-llm --scope project   # or --scope local
```

Things that go wrong:

- **`marketplace update` is not enough.** A plain `claude plugin marketplace update` does not add sparse settings.
- **Adding again without the edit does nothing.** While your user settings still declare the old source, the add is refused at every scope and changes nothing. The error says the source `doesn't match its extraKnownMarketplaces entry`.
- **The plugin looks broken between the two commands.** Between the edit and the add, or if the add fails, the marketplace disappears from `list`, `update` says `not found`, and every installation shows `failed to load`. Re-running the add, or undoing the edit, recovers it.
- **Project scope means the current directory, not the Git root.** `--scope project` from the wrong directory exits 0, writes a new `.claude/settings.json` there, and leaves the real entry stale. Delete a stray entry by hand.

> **Do not use `claude plugin marketplace remove` for this.** When it removes the last declaration visible from the current directory, with or without `--scope`, it also uninstalls the plugin. In testing, one command from a neutral directory removed all 7 installations, the registration, the checkout, and saved options, secrets and data, while other projects kept stale declarations. The same applies to a stray entry: running `marketplace remove` there can wipe everything.

## Codex uses a different catalog

Each host installs its own plugin directory, selected by its own catalog:

| Host | Catalog | Plugin directory |
| --- | --- | --- |
| Claude Code | `.claude-plugin/marketplace.json` | `packaging/claude/multi-llm` |
| Codex | `.agents/plugins/marketplace.json` | `packaging/codex/multi-llm` |

Install on Codex with both of its sparse paths:

```bash
codex plugin marketplace add beastlabai/multi-llm-plugin --ref release \
  --sparse .agents --sparse packaging/codex
codex plugin add multi-llm@beastlabai
```

Codex reads both catalogs and prefers `.agents`. When a checkout has no `.agents` catalog, it falls back to `.claude-plugin/marketplace.json`, and **that fallback fails cleanly**: `codex plugin add` stops with ``plugin `multi-llm` is not available for install in marketplace `beastlabai` `` and installs nothing. That catalog's entry is for Claude Code only and carries a Codex policy saying so, which Claude Code ignores.

What other sparse paths do on codex 0.167.0:

| `--sparse` paths | Result |
| --- | --- |
| `packaging/codex` alone (the incomplete example: no catalog in the checkout) | `marketplace add` fails with `marketplace root does not contain a supported manifest`. Nothing is registered or installed. |
| `.claude-plugin` and `packaging/claude/multi-llm` | `marketplace add` succeeds. `plugin add` fails with `is not available for install`. The marketplace stays registered, and nothing is installed. |
| `.claude-plugin` and `packaging/codex` | `plugin add` fails with `plugin source path is not a directory`. The marketplace stays registered, and nothing is installed. |
| `.agents` alone | The same failure, for `packaging/codex/multi-llm`. |
| `.claude-plugin packaging/claude/multi-llm` after one flag | `error: unexpected argument`. Codex takes one path per `--sparse`. Nothing is registered. |

None of these copies the repository root into the plugin cache.

**To recover:**

- If `marketplace add` failed, nothing was registered, so run the two install commands above. `codex plugin marketplace remove beastlabai` would only report that the marketplace `is not configured or installed`.
- If `marketplace add` succeeded with other paths, adding it again is refused: `marketplace 'beastlabai' is already added from a different source; remove it before adding this source`. `codex plugin marketplace upgrade` does not change the paths. Remove the registration first:

  ```bash
  codex plugin marketplace remove beastlabai
  codex plugin marketplace add beastlabai/multi-llm-plugin --ref release \
    --sparse .agents --sparse packaging/codex
  codex plugin add multi-llm@beastlabai
  # then start a NEW Codex session
  ```

Unlike the Claude Code command, `codex plugin marketplace remove` removes only the registration and its checkout. An installed copy stays in the plugin cache, and `plugin list` stops showing it until the marketplace is back. The `plugin add` above then replaces it, including a Claude Code package installed by the fallback.

## Prerequisites in full

| Requirement | Why |
| --- | --- |
| **Claude Code plugin only:** Claude Code 2.1.200 or newer | The oldest build this plugin was run on with subagents and `SendMessage` both working. Of the older builds tried, 2.1.150 and 2.1.100 can start a subagent but cannot send a batch back to it for more work; builds between 2.1.150 and 2.1.200 were not tried. |
| [`uv`](https://docs.astral.sh/uv/) on your `PATH` | The orchestrators run as `uv run` Python scripts. uv installs their dependencies (just `pyyaml`) on first use. If `uv` is missing when a mode starts (`--init`, `--review-plan`, `--review-code`, ...), the skill prints the installation instructions from `skills/multi-llm/hosts/uv-missing.txt` and stops (exit 3) — it never installs uv and never offers to. Install location and installer differ per OS: on **Linux/macOS** the official installer (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) puts `uv` in `~/.local/bin/uv`; on **macOS** Homebrew installs to `/opt/homebrew/bin/uv` (Apple Silicon) or `/usr/local/bin/uv` (Intel), and MacPorts to `/opt/local/bin/uv`; on **Windows** the official installer is PowerShell (`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"`) and creates `%USERPROFILE%\.local\bin\uv.exe` — which Git Bash sees as `~/.local/bin/uv.exe` (note the `.exe`); `winget install --id=astral-sh.uv -e` and `pipx install uv` also work. |
| Python >= 3.10 (usually nothing to install) | The orchestrators need Python 3.10 or newer, and `uv` supplies it: if no suitable interpreter is installed, `uv` downloads a managed one on first use, which needs the network once. **Codex plugin only:** the plugin's tool server starts on the system `python3` before `uv` is involved, so python3 >= 3.9 must be on `PATH` under exactly that name. |
| [Git](https://git-scm.com/), and a Git repository to run in | `--review-code`, `--implement` and the apply modes read diffs and stage files with `git`. `--init` writes its config at the Git root, and the per-project config is only discovered inside a Git repository. |
| **Windows only:** [Git for Windows](https://git-scm.com/download/win) (Git Bash) | **Hard prerequisite on Windows.** The skill's entire instruction set and its permission (`allowed-tools`) rules are Bash-only; Claude Code's PowerShell-tool fallback (used when Git for Windows is absent) is explicitly **unsupported** — the skill fails fast with an install pointer rather than attempting PowerShell equivalents. Git for Windows also supplies the bundled coreutils the command blocks depend on (`ls`, `grep`, `diff`, `tail`, `dirname`, ...), which are absent from PowerShell/cmd. Note that `realpath` is *not* one of them: plan paths are passed to the orchestrators as-is (never wrapped in `$(realpath ...)`), and the orchestrators resolve them to OS-native absolute paths themselves. |
| One or more provider CLIs | You only need the CLIs for the models you actually run. See [Providers](providers.md). |

The plugin ships its own `pyproject.toml`/`uv.lock`, so no manual Python setup is required beyond the Codex plugin's `python3` - `uv` resolves the environment the first time a mode runs.

## Headless Claude Code runs in default permission mode

The skill's own permission rules (`allowed-tools`) last only for the turn that loaded it, and in default permission mode Claude Code also asks before a shell command creates a file in the working directory. An interactive session shows you those questions. A headless `claude -p` run refuses them, so a phase can stop part-way: at a fan-out's launch (its redirect into `orchestrator-run.log`, or the `mkdir` of a first run's phase directory), or after a batch is sent back to its subagent, because that resume runs in the background and ends the turn. The plugin grants nothing extra on your behalf. To run headless in default mode, opt in yourself. The two stops need different fixes, so a phase that both launches a fan-out and sends a batch back needs both:

- **A fan-out's launch** (`--review-plan`, `--review-tasks`, `--review-code`, `--review-commits`, `--ask`): run with `--permission-mode acceptEdits`, which lets the launch's redirect and a first run's `mkdir` go ahead. On 2.1.319 allow rules did not clear it. The mode is wider than these two steps and lasts the whole session: Claude Code then accepts every file edit, and common filesystem commands such as `mkdir`, in the working directory and any added directories without asking, so the narrower `Edit` rules below do not limit it.
- **A later turn** (after a batch is sent back): allow the skill's dispatcher command, and its edits to the plan and to the run's snapshots under `.multi-llm/tmp/`. Adjust `plans/**` to wherever your plans live. `acceptEdits` alone does not cover this: on 2.1.319 a later turn's dispatcher command still needed approval.

  ```json
  {
    "permissions": {
      "allow": [
        "Bash(uv run --frozen --no-dev --project \"*/skills/multi-llm\" -- python \"*/skills/multi-llm/multi_llm.py\" *)",
        "Edit(plans/**)",
        "Edit(.multi-llm/tmp/**)"
      ]
    }
  }
  ```

  With exactly these three rules, Claude Code 2.1.322 finished an `--apply-suggestions` phase with no denials in default mode, including a batch sent back to its subagent. With the `Bash` rule alone, it stopped at the parent's next edit.

  **What the `Bash` rule allows.** In a Claude Code rule `*` matches any text, spaces and `..` included, so this rule is not limited to the plugin's files: it allows any command that starts with `uv run --frozen --no-dev --project "` and has that shape. Writing the plugin's installed path in full instead of the first two `*` is narrower, but on 2.1.322 it refused one of the commands a phase runs (one with a `$(…)` substitution), and the path changes with every plugin update. For comparison, the skill grants itself `Bash(uv:*)` for the turn that loads it.

  Put the rules where a headless run reads them: your user settings (`~/.claude/settings.json`), or a file passed with `--settings <file>`. A headless run ignores a project's `.claude/settings.json` until you have trusted that folder in an interactive session; it says so on stderr ("this workspace has not been trusted").

## The plugin's own providers.yaml

The README's way to choose providers is `--init` and the per-project
`<git-root>/.multi-llm/providers.yaml` it writes ([Per-project
configuration](configuration.md)). The file that one is layered over is the
plugin's own [`skills/multi-llm/providers.yaml`](../skills/multi-llm/providers.yaml):
the base catalog and defaults every repository inherits.

To change the base itself, edit it to match the code harnesses and models you want to use. The shipped file is a starting point only - keep the providers whose CLIs you have installed, remove the rest, and set `defaults.models` (and optionally `quick_models`) to the `provider:model` pairs you actually run. See [Providers](providers.md) for the format and available keys.

> **Where is the file after installing the plugin?** If you installed the plugin (rather than cloning this repo), `providers.yaml` lives inside the installed plugin, under your Claude Code plugin cache - `~/.claude/plugins/cache/beastlabai/multi-llm/<version>/skills/multi-llm/providers.yaml`, where `<version>` changes on every update. Each update installs into a new directory there, so an edit to the old copy does not carry over to the new one: keep anything you want to survive an update in the per-project file. The marketplace checkout under `plugins/marketplaces/` holds another copy, which is not the one that runs. The quickest way to open it is to ask Claude Code to "open the multi-llm `providers.yaml`": the skill resolves its own install location (via `${CLAUDE_SKILL_DIR}`), so Claude can read and edit the file in place.

> **On the Codex plugin or the OpenCode package**, leave the installed copy alone - the next upgrade replaces it. Run `--init` in each repository instead and edit the `<git-root>/.multi-llm/providers.yaml` it writes ([Per-project configuration](configuration.md)).

---

Back to the [README](../README.md) · [Hosts](hosts.md) · [Providers](providers.md) · [Per-project configuration](configuration.md)
