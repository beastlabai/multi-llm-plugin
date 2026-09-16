# Per-project configuration

Giving one repository its own provider/model defaults, what each config layer
overrides, and why an auto-loaded override is safe. The base layer itself is
documented in [Providers](providers.md). Back to the [README](../README.md).

---

The shipped [`providers.yaml`](../skills/multi-llm/providers.yaml) is the
**base** layer shared by every repo. To give one repository its own
provider/model defaults — without editing the installed plugin — add an
optional, auto-discovered override file at
`<git-root>/.multi-llm/providers.yaml`. When it's absent, behavior is unchanged.

It deep-merges its full contents over the base and inherits everything you omit,
so it can stay tiny — change just the *selection* keys (`default_provider`,
`defaults.models`, `defaults.quick_models`, `defaults.modes`) and leave the rest:

```yaml
# .multi-llm/providers.yaml — only what this repo changes
default_provider: claude-code
defaults:
  models:
    - claude-code:fable
    - cursor-agent:composer-2.5
  quick_models:
    - claude-code:fable
```

Create one with `/multi-llm:multi-llm --init`. This is **fully automatic and
zero-prompt** — it works the same in a terminal, inside Claude Code, or in CI. It
scans your `PATH` for the supported provider CLIs and writes a preconfigured
override: starting from an inert template, it **uncomments** the full `providers:`
block, `defaults.models` / `quick_models` entries, and `default_provider` for each
CLI it detects, leaving the rest commented. `default_provider` is set to the
**first detected provider** in base order (claude-code, cursor-agent, gemini,
grok, opencode, codex, kilocode, kimi, qwen, copilot, cline, goose, aider, agy, pi, muse, vibe). With **no CLIs installed** it writes the inert
template (which inherits the built-in defaults) and exits 0 with an install
notice — it never errors. Pass `--template-only` to skip detection and write the
pristine commented stub for hand-editing. Add `--gitignore` to keep the file
developer-local, and `--force` to overwrite an existing one — overwriting first
backs up the previous file alongside as `providers.yaml.bak.<timestamp>` (and
prints the path), so you can port any custom settings into the regenerated file.

## Good to know

- **Precedence (low → high):** base `providers.yaml` → project-local
  `.multi-llm/providers.yaml` → `MULTI_LLM_PROVIDERS_CONFIG=/path.yaml` env override.
  Each layer is deep-merged over the one below.
- **Lists replace, they don't append.** Set `defaults.models` and you get exactly
  that list. Omitting or blanking a key inherits the base; an explicit empty list
  `[]` is the only way to clear one — and an empty `defaults.models`/`quick_models`
  falls back to interactive selection (or fails in unattended runs like `--full`).
- **Per-mode lists win.** Modes with a base `defaults.modes.<mode>` entry (e.g.
  `--review-plan`, `--review-code`) ignore a project-wide `defaults.models` —
  override `defaults.modes.<mode>` to change those.
- **Requires a git repo.** Discovery resolves the git root, so there's one override
  per repo, shared repo-wide (no per-subdirectory configs). Non-git projects must
  use `MULTI_LLM_PROVIDERS_CONFIG` instead.
- **Commit or ignore?** Commit it for a team-wide standard; ignore it (`--gitignore`)
  or use the env override for personal preferences.
- **`driver.verify_command` is this repo's own check**, and the one `driver.*` key
  worth setting per project: a shell string (`pnpm typecheck && pnpm lint:fix`,
  `cargo check`, `make test`) that the code-editing phases — `implement` and
  `apply-code-fixes` — run **once**, from the repository root, after their last
  batch applied. Left unset, those phases say so in one line rather than completing
  unchecked; a non-zero exit reverts nothing (every batch was already verified
  against its own snapshot) but ends the phase at exit 3 with the command and the
  tail of its output. `driver.verify_timeout_s` bounds it (default 600 s), and
  the bound is on the command's whole process group rather than on the
  `/bin/sh` that started it, so a hung `tsc` is killed rather than orphaned.
  The markdown apply phases never run it.

## Safety

A repo's override is auto-loaded, so a cloned repo can change which models a run
selects — but never *what executes* on the model path: provider binaries are
hardcoded and the config `command` field is never run, in any layer. The one key
that *is* executed is `driver.verify_command`, and only by a write phase you asked
for, after it edited that same repo with an agent CLI — the run prints the command
and the config layer it came from before it starts, and again with its exit code
once it ends. Treat an untrusted repo's override the way you would its `Makefile`
— and the blast radius is not only the override file: the *scripts the command
invokes* (`package.json`'s `typecheck`, a `Makefile` target) are files the write
subagent was just allowed to edit, and they run outside whatever sandbox the
agent CLI had. The check runs with the host's own session variables stripped and,
when `env_allowlist` is on, with the same deny-by-default environment a runner
gets; otherwise it inherits the environment you launched the run with.

The override now deep-merges its full contents (including a `providers:` block)
over base, exactly like the `MULTI_LLM_PROVIDERS_CONFIG` env layer — but since
`command` is documentation-only, a merged block can only retune metadata
(timeouts, model lists), and a provider name with no hardcoded adapter is
ignored at runtime. A present-but-malformed explicit override fails fast with a
clear error naming the file; set `MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE=1` to
warn-and-skip instead.
