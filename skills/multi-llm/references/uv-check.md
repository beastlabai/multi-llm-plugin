# uv Availability Check

Every orchestrator and helper script in this skill is executed through
`uv run --project ${CLAUDE_SKILL_DIR} -- python ...`. If `uv` is not installed,
that first command fails with a confusing `command not found` deep inside the
mode. Run this check **once per mode, before the first `uv run` command**.

## Detection

```bash
command -v uv
```

- **Exit 0 (path printed)** → `uv` is available. Proceed with the mode; nothing
  to report to the user.
- **Non-zero exit** → `uv` is not on `PATH`. Before offering to install, check
  the common install locations that a fresh shell may not have on `PATH` yet:

  ```bash
  ls "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" 2>/dev/null
  ```

  If one exists, **do not reinstall**. Use that absolute path in place of the
  bare `uv` in every command for the rest of the session (e.g.
  `$HOME/.local/bin/uv run --project ...`), and tell the user to add the
  directory to their `PATH` to make it permanent.

## Offer to install (only when truly absent)

Use AskUserQuestion to offer installation — never install without consent:

- **Official installer (recommended)**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Installs to `~/.local/bin`.
- **Package manager**: whichever fits the platform, e.g. `pipx install uv`,
  `pip install --user uv`, `brew install uv` (macOS), or the distro package.
- **Decline**: abort the mode with a clear message that `uv` is required, and
  point to https://docs.astral.sh/uv/getting-started/installation/ — do NOT
  fall back to running the scripts with bare `python` (the skill's dependencies
  are resolved by uv from its `pyproject.toml`).

## After installing

Re-run the detection above, including the `~/.local/bin` / `~/.cargo/bin`
fallback — the current shell's `PATH` usually does not pick up a fresh install.
If detection still fails, stop and report; never proceed to a `uv run` command
while `uv` cannot be resolved.
