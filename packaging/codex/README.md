# multi-llm — the Codex plugin

Runs the multi-llm workflow — plan reviews, task generation, implementation and code
review fanned out across several LLM CLIs in parallel — inside a Codex session. Installed,
the plugin contributes two things:

- a **tool**, `multi_llm`, served by a small stdio MCP server bundled with the plugin. It
  composes the dispatcher's flags in code, runs the run to a result and hands the model
  one JSON answer. This is the entry point.
- a **skill**, advertised as `multi-llm:multi-llm`, whose whole body points the model at
  that tool. A Codex plugin has no slash command, so you name the skill in prose:
  *"Use the `multi-llm:multi-llm` skill with exactly these arguments:
  `--review-plan plans/x.md`"*.

The install below pins `--ref release`, which the publish workflow moves to each promoted
`codex-v*` tag (first release: `codex-v0.1.0`). The file route — `install-host codex`, see
the repository's `docs/hosts.md` — remains the alternative.

## Floors

Published here because nothing enforces them — the plugin manifest has no host-version
field, so an older host does not refuse the install, it quietly lacks the skill or the
tool:

- **codex >= 0.159.2**
- **python3 >= 3.9** on `PATH`, under exactly that name (the tool's server runs on it
  before `uv` is involved)
- **uv**, any version, on `PATH`

Before a release is published, its install, skill and tool are checked offline on three
codex builds: 0.159.2 (the floor above), the version the repository's container pins, and
npm's newest `@openai/codex` that day. The release notes name the versions and any
`codex features list` change the newest one showed.

## Install

```bash
codex --version                     # 0.159.2 or newer
python3 --version && uv --version   # python3 3.9 or newer, under these exact names
codex plugin marketplace add beastlabai/multi-llm-plugin --ref release \
  --sparse .agents --sparse packaging/codex
codex plugin add multi-llm@beastlabai
# then start a NEW session
```

Both `--sparse` paths are required. Neither command installs dependencies: the first
invocation builds the plugin's venv (`pyyaml` alone), which needs the network or a warm uv
cache.

Then, once per repository, in the new session: *"Use the `multi-llm:multi-llm` skill with
exactly these arguments: `--init`"*. It writes `<git-root>/.multi-llm/providers.yaml`,
outside `$CODEX_HOME`, where no reinstall or upgrade touches it. Configure providers
there — never in the copy installed inside the plugin, which the next `codex plugin add`
replaces.

**Provider keys** go in the file your login shell reads — `~/.profile` or
`~/.bash_profile`, not `~/.bashrc` (and on macOS not `~/.zprofile`). Codex starts the
plugin's server with a stripped environment, and the run re-derives your profile through a
`bash -lc` login shell, so a key exported only in the terminal never arrives. Do not edit
the installed `.mcp.json`: the next `codex plugin add` replaces it wholesale.

## What it runs, and what it may write

The tool starts the orchestrator from a server Codex launches **outside its own exec
sandbox**, so the session's `--sandbox` posture does not govern it: installing this plugin
gives a Codex session a write path it did not have. So under a read-only posture, or one it
cannot read, the tool **refuses write-capable modes** before it spawns anything, unless
`MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1` is in the environment that starts Codex. Despite
its name, `MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1` lifts only this tool's own refusal of
write-capable modes under a read-only or unreadable session posture. It never lets the
Codex processes the orchestrator starts write without a sandbox, and it does not change
which sandbox they run under: a Codex editing job refused because Codex's own sandbox
cannot start on this host stays refused with it set. Provider CLIs the orchestrator
dispatches stay under their own sandboxes.

**The plugin pre-approves its own tool** (`"default_tools_approval_mode": "approve"` in its `.mcp.json`), so Codex calls it without asking — without that, a headless `codex exec` under a sandbox refuses the call outright. To be asked before every call, set `default_tools_approval_mode = "prompt"` under `[plugins."multi-llm@beastlabai".mcp_servers.multi_llm]` in `$CODEX_HOME/config.toml` (measured on codex 0.156.1: the call then needs approval again).

The install is a Git clone and a directory copy with no signature: you are trusting
whoever can move the `release` ref and the workflow that moves it.

## Platforms

| Platform | Status |
| --- | --- |
| Linux | `verified` — the only platform this route claims |
| macOS | `unmeasured`. The run's `bash -lc` login shell reads `~/.bash_profile` / `~/.profile`, not `~/.zprofile` or `~/.zshrc` |
| Windows | `unsupported for the tool route` — `python3` is usually absent under that name, so the server never starts. Git Bash (or another POSIX `bash`) on `PATH`, or beside a `git` that is, is also a hard prerequisite (WSL's `System32\bash.exe` does not count): without it the tool refuses with a missing-`bash` error. Use the file route (`install-host codex`) |

## Upgrade

```bash
codex plugin marketplace upgrade
codex plugin add multi-llm@beastlabai
# then start a NEW session
```

Re-running the install's first command does **not** upgrade: the same command again
reports the marketplace as already added and changes nothing, and the same command with a
different `--ref` or different `--sparse` paths is refused ("already added from a
different source") — remove the marketplace first (`codex plugin marketplace remove
beastlabai`) to switch refs. The upgrade fetches whatever the registered ref points at; when
the version moved it replaces the installed version directory itself, and the `plugin add`
after it is the explicit step (measured on codex 0.164.0).

## Uninstall

```bash
codex plugin remove multi-llm@beastlabai
```

It removes the installed version and the `[plugins."multi-llm@beastlabai"]` entry in
`config.toml`, and leaves the marketplace registered, so a later reinstall needs only
`codex plugin add`. Your `.multi-llm/providers.yaml` is not touched. Coming from the file
route, remove its shim with `uninstall-host codex --global`, or both skills stay advertised.

## When it silently does nothing

Every failure on this route is silent: exit 0, nothing printed. What each one looks like:

- **The skill is not advertised.** `codex plugin list` has no `multi-llm@beastlabai`
  entry (removed), or shows `enabled: false` (disabled). An entry is not proof the plugin
  loads — check `enabled`, then start a new session.
- **The skill is there, the tool is not.** The server did not start. `codex mcp get
  multi_llm` names the command Codex tried; `python3 --version` failing, or printing a
  version below 3.9, is the whole story.
- **The tool answers, but models fail with provider errors.** A key did not reach the
  run: the result's `environment` block says whether a login shell ran and which startup
  file it read.
- **The install command "succeeds" on an old codex.** Below 0.159.2 the plugin surface is
  unmeasured; check `codex --version` first.

`codex doctor` does not see plugin servers, so it cannot diagnose any of these.
