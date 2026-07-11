# uv Availability Check

Every orchestrator and helper script in this skill is executed through
`uv run --project "${CLAUDE_SKILL_DIR}" -- python ...`. If `uv` is not installed,
that first command fails with a confusing `command not found` deep inside the
mode. Run this check **once per mode, before the first `uv run` command**.

## Supported shells (fail fast otherwise)

This check — and every other command in this skill — is POSIX-shell syntax and
runs under the Bash tool (Git Bash on Windows, the native shell on macOS/Linux).
If the shell environment is **not** a POSIX shell (e.g. Claude Code fell back to
its PowerShell tool because Git for Windows is not installed), **stop
immediately** and tell the user:

> Windows support requires Git for Windows (Git Bash); PowerShell-only
> environments are not supported — install it from
> https://git-scm.com/download/win and restart Claude Code.

Do NOT attempt PowerShell equivalents of any command in this skill — there is no
PowerShell instruction path.

## Detection

Run detection as a **single standalone command** — do NOT chain it with `||`,
`&&`, `;`, or a pipe. The permission allowlist contains the exact-match rule
`Bash(command -v uv)`, and exact-match rules only apply when the command is run
exactly as written; chaining anything onto it triggers a permission prompt.

```bash
command -v uv
```

- **Exit 0 (path printed)** → `uv` is available. Proceed with the mode; nothing
  to report to the user.
- **Non-zero exit** → `uv` is not on `PATH`. Before offering to install, check
  the common install locations that a fresh shell may not have on `PATH` yet
  (quoted `"$HOME"`, never a quoted tilde — `"~"` does not expand; the `.exe`
  variants are what the Windows installer creates under Git Bash):

  ```bash
  ls "$HOME/.local/bin/uv" "$HOME/.local/bin/uv.exe" "$HOME/.cargo/bin/uv" "$HOME/.cargo/bin/uv.exe" /opt/homebrew/bin/uv /usr/local/bin/uv /opt/local/bin/uv 2>/dev/null
  ```

  **Read the result from stdout, NOT from the exit status.** This single `ls`
  exits non-zero whenever *any* candidate is absent — which is the normal case
  even when uv IS installed (at most one of the seven paths exists on a given
  machine). Any path printed on stdout means **FOUND**, regardless of exit
  status; only an empty stdout means uv is truly absent.

  Candidate coverage: `~/.local/bin` (official installer on Linux/macOS; on
  Windows the installer writes `uv.exe` there), `~/.cargo/bin` (cargo installs,
  again with a Windows `.exe` variant), `/opt/homebrew/bin` (macOS Homebrew on
  Apple Silicon), `/usr/local/bin` (Homebrew on Intel macs and generic Unix),
  and `/opt/local/bin` (macOS MacPorts).

  If a candidate is printed, verify it is actually executable before using it —
  run `test -x` on it, or equivalently run `"$FOUND" --version` and treat
  success as the verification:

  ```bash
  test -x "$HOME/.local/bin/uv"   # substitute the printed path
  ```

  If one exists and is executable, **do not reinstall**. Note, however, that
  launching uv by absolute path (e.g. `"$HOME/.local/bin/uv" run --project ...`)
  is NOT covered by the `Bash(uv:*)` allow rule, so **every subsequent command
  will raise a permission prompt**. The primary remedy is to get `uv` onto
  `PATH` instead: tell the user to add the directory to their `PATH` (or simply
  restart the shell / Claude Code — installers update the profile, so a fresh
  shell usually picks uv up automatically), then re-run the standalone
  `command -v uv` detection. Only fall back to the absolute path for the rest
  of the session if the user prefers to proceed immediately and accepts the
  per-command prompts.

## Offer to install (only when truly absent)

Use AskUserQuestion to offer installation — never install without consent. The
installer differs per OS:

- **Linux / macOS — official installer (recommended)**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Installs to `~/.local/bin/uv`.
- **Windows (Git Bash) — official installer** (the installer itself is
  PowerShell; this is the only PowerShell content in this skill, and it is an
  install offer, not a detection or execution path):
  ```bash
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  Installs to `%USERPROFILE%\.local\bin\uv.exe` — which Git Bash sees as
  `~/.local/bin/uv.exe` (note the `.exe`).
- **Package manager**: whichever fits the platform, e.g. `brew install uv`
  (macOS), `pipx install uv`, `pip install --user uv`,
  `winget install --id=astral-sh.uv -e` (Windows), or the distro package.
- **Decline**: abort the mode with a clear message that `uv` is required, and
  point to https://docs.astral.sh/uv/getting-started/installation/ — do NOT
  fall back to running the scripts with bare `python` (the skill's dependencies
  are resolved by uv from its `pyproject.toml`).

## After installing

Re-run the detection above: first the standalone `command -v uv`, then (if
still not on `PATH`) the seven-candidate `ls` fallback — the current shell's
`PATH` usually does not pick up a fresh install. On Windows the re-detection
must include the `.exe` variants (`"$HOME/.local/bin/uv.exe"`,
`"$HOME/.cargo/bin/uv.exe"`) — the installer creates `uv.exe`, not `uv`. If the
binary exists but only by absolute path, prefer restarting the shell (or
Claude Code) so `uv` resolves on `PATH`, per the remedy above. If detection
still fails, stop and report; never proceed to a `uv run` command while `uv`
cannot be resolved.
