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

One provider key does change *how* a binary executes:
`providers.codex.sandbox_fallback` (`auto` by default, `off` to disable). codex
is the only reviewer that insists on its own OS sandbox — `codex exec` runs the
model's shell through bubblewrap, and on a stock Ubuntu 24.04 bwrap cannot
start (`multi_llm.py doctor` prints the AppArmor repair). Under `auto`, once
the sandbox probe has failed, codex's **read-only** work (the review/ask
fan-out and the executor's validation/consolidation/salvage kinds) runs
`--sandbox danger-full-access` in the real repo root, i.e. with the same rights
as your shell — which is the footing cursor-agent, pi, claude-code and muse
have always had in those phases — and every artifact of such a slot says so
(`[SANDBOX_BYPASSED]` on stdout, `ran unsandboxed: sandbox_fallback=auto` on
the report's model labels, a `(fallback: auto …)` note on doctor's BROKEN row,
a `git status` check that fails the slot by path if the tree changed). `off`
restores the refusal: codex is left out on such a machine until the repair is
installed. The write phases are refused under a broken sandbox whatever the
key says. Because the default is already `auto`, a cloned repository's
`.multi-llm/providers.yaml` can only turn the fallback *off* — there is nothing
for an untrusted override to widen here. See the key's comment in
`providers.yaml` for what `auto` gives up.
