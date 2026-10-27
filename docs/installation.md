# Installation details

What the README's install steps leave out: updating the Claude Code plugin, the
prerequisites in full, and the plugin's own `providers.yaml`. The Codex plugin
and the OpenCode package have theirs in [Hosts](hosts.md#codex-plugin-route)
(and [the OpenCode npm route](hosts.md#opencode-npm-route)). Back to the
[README](../README.md).

---

## Updating the Claude Code plugin

> **Already added the marketplace before (e.g. in another project)?** `/plugin marketplace add` does **not** refetch an existing marketplace - it silently reuses the locally cached copy, so `/plugin install` will give you a stale version. Run the update steps below instead.

**Update.** The marketplace cache only refreshes when you explicitly ask it to - `/plugin marketplace update` is the step that actually pulls the latest commits from GitHub. Then update (or install) the plugin and reload:

```text
/plugin marketplace update beastlabai
# remove "--scope local" if you installed it as user scope
! claude plugin update multi-llm@beastlabai --scope local
/reload-plugins
/reload-skills
/multi-llm:multi-llm --init --force
```

## Prerequisites in full

| Requirement | Why |
| --- | --- |
| [`uv`](https://docs.astral.sh/uv/) on your `PATH` | The orchestrators run as `uv run` Python scripts. uv installs their dependencies (just `pyyaml`) on first use. If `uv` is missing when a mode starts (`--init`, `--review-plan`, `--review-code`, ...), the skill prints the installation instructions from `skills/multi-llm/hosts/uv-missing.txt` and stops (exit 3) — it never installs uv and never offers to. Install location and installer differ per OS: on **Linux/macOS** the official installer (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) puts `uv` in `~/.local/bin/uv`; on **macOS** Homebrew installs to `/opt/homebrew/bin/uv` (Apple Silicon) or `/usr/local/bin/uv` (Intel), and MacPorts to `/opt/local/bin/uv`; on **Windows** the official installer is PowerShell (`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"`) and creates `%USERPROFILE%\.local\bin\uv.exe` — which Git Bash sees as `~/.local/bin/uv.exe` (note the `.exe`); `winget install --id=astral-sh.uv -e` and `pipx install uv` also work. |
| Python >= 3.10 (usually nothing to install) | The orchestrators need Python 3.10 or newer, and `uv` supplies it: if no suitable interpreter is installed, `uv` downloads a managed one on first use, which needs the network once. **Codex plugin only:** the plugin's tool server starts on the system `python3` before `uv` is involved, so python3 >= 3.9 must be on `PATH` under exactly that name. |
| [Git](https://git-scm.com/), and a Git repository to run in | `--review-code`, `--implement` and the apply modes read diffs and stage files with `git`. `--init` writes its config at the Git root, and the per-project config is only discovered inside a Git repository. |
| **Windows only:** [Git for Windows](https://git-scm.com/download/win) (Git Bash) | **Hard prerequisite on Windows.** The skill's entire instruction set and its permission (`allowed-tools`) rules are Bash-only; Claude Code's PowerShell-tool fallback (used when Git for Windows is absent) is explicitly **unsupported** — the skill fails fast with an install pointer rather than attempting PowerShell equivalents. Git for Windows also supplies the bundled coreutils the command blocks depend on (`ls`, `grep`, `diff`, `tail`, `dirname`, ...), which are absent from PowerShell/cmd. Note that `realpath` is *not* one of them: plan paths are passed to the orchestrators as-is (never wrapped in `$(realpath ...)`), and the orchestrators resolve them to OS-native absolute paths themselves. |
| One or more provider CLIs | You only need the CLIs for the models you actually run. See [Providers](providers.md). |

The plugin ships its own `pyproject.toml`/`uv.lock`, so no manual Python setup is required beyond the Codex plugin's `python3` - `uv` resolves the environment the first time a mode runs.

## The plugin's own providers.yaml

The README's way to choose providers is `--init` and the per-project
`<git-root>/.multi-llm/providers.yaml` it writes ([Per-project
configuration](configuration.md)). The file that one is layered over is the
plugin's own [`skills/multi-llm/providers.yaml`](../skills/multi-llm/providers.yaml):
the base catalog and defaults every repository inherits.

To change the base itself, edit it to match the code harnesses and models you want to use. The shipped file is a starting point only - keep the providers whose CLIs you have installed, remove the rest, and set `defaults.models` (and optionally `quick_models`) to the `provider:model` pairs you actually run. See [Providers](providers.md) for the format and available keys.

> **Where is the file after installing via `/plugin`?** If you installed the plugin (rather than cloning this repo), `providers.yaml` lives inside the installed plugin, under your Claude Code plugin cache - `~/.claude/plugins/cache/<marketplace>/multi-llm/<version-or-commit>/skills/multi-llm/providers.yaml`, where the last segment changes on every update. Each update installs into a new directory there, so an edit to the old copy does not carry over to the new one: keep anything you want to survive an update in the per-project file. The quickest way to open it is to ask Claude Code to "open the multi-llm `providers.yaml`": the skill resolves its own install location (via `${CLAUDE_SKILL_DIR}`), so Claude can read and edit the file in place.

> **On the Codex plugin or the OpenCode package**, leave the installed copy alone - the next upgrade replaces it. Run `--init` in each repository instead and edit the `<git-root>/.multi-llm/providers.yaml` it writes ([Per-project configuration](configuration.md)).

---

Back to the [README](../README.md) · [Hosts](hosts.md) · [Providers](providers.md) · [Per-project configuration](configuration.md)
