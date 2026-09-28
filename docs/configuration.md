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
grok, opencode, codex, kilocode, kimi, qwen, copilot, cline, goose, aider, agy, pi, muse, vibe, omp). With **no CLIs installed** it writes the inert
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
- **`progress.interval_s` / `progress.poll_s` price two different cadences.**
  `interval_s` (default 30 s) is how often a running fan-out prints a
  `[PROGRESS]` ticker line — a stdout cadence only, so `interval_s: 0` is the
  documented **disable form**: the lines stop, while the run's status file still
  records every transition and `--wait` is untouched. `poll_s` (default 90 s) is
  how often the agent host polls that run, and the resolved `--wait` default — a
  poll costs a **tool turn**, which is why it is a separate knob rather than the
  same number. It is **clamped under the host's shell cap** (the tightest
  surveyed ceiling is OpenCode's 120 s, less a 30 s margin — which is where 90
  comes from), so raising it past that cap buys nothing. Both floors are 5 s;
  `poll_s` has **no** disable form (`0` is rejected, since it would mean a poll
  that returns instantly rather than one switched off), and an unknown key under
  `progress:` is rejected at load rather than silently keeping the default.
- **`code_review.collapse_paths` decides which changed files reviewers skim.**
  `review-code` sorts its scope into source, tests, fixtures and generated
  files, in that order, and writes a whole per-file diff for every source and
  test file to its phase dir (`code-review/review-scope.md` and
  `code-review/diff/`). Fixtures and generated files are listed one line per
  directory, with no diff. A path is generated when `.gitattributes` gives it
  `linguist-generated`, or when it matches a `generated` glob here; it is a
  fixture when it matches a `fixtures` glob. The defaults are `generated: []`
  and `fixtures: ["**/tests/fixtures/**", "**/tests/golden/**"]`; a project can
  add, say, `**/*.lock` or `dist/**` so a huge machine-written file is never
  handed to reviewers as a diff. A malformed value is rejected at load.

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
One credential makes that filter's shape load-bearing for a user's own setup:
`omp` is the first adapter whose key is admitted by neither its own `PI_`/`OMP_`
prefix nor the shared base list, so the adapter passes `OPENROUTER_API_KEY`
through **explicitly** — a filtered run still carries it — and omp then resolves
it through its own chain: the process environment, then a `.env` in the directory
the run starts from (a checked-out repo's own key therefore outranks the one you
exported), then `~/.omp/agent/.env`, then `~/.omp/.env` or `~/.env`, of which
`~/.omp/agent/.env` is the source that depends on nothing in the parent
environment and the one to use when the key is not exported in the shell the run
starts in.

The override now deep-merges its full contents (including a `providers:` block)
over base, exactly like the `MULTI_LLM_PROVIDERS_CONFIG` env layer — but since
`command` is documentation-only, a merged block can only retune metadata
(timeouts, model lists), and a provider name with no hardcoded adapter is
ignored at runtime. A present-but-malformed explicit override fails fast with a
clear error naming the file; set `MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE=1` to
warn-and-skip instead.

One provider key does change *how* a binary executes:
`providers.codex.sandbox_fallback` (`auto` by default; `contained` or `off` to
narrow it). codex is the only reviewer that insists on its own OS sandbox —
`codex exec` runs the model's shell through bubblewrap, and on a stock Ubuntu
24.04 bwrap cannot start (`multi_llm.py doctor` prints the AppArmor repair).
Once the sandbox probe has failed, codex's **read-only** work (the review/ask
fan-out and the executor's validation/consolidation/salvage kinds) walks a
two-rung ladder. Rung 1, `landlock`, runs `--sandbox read-only` with a
per-dispatch `-c features.use_legacy_landlock=true` — codex's second Linux
backend, which needs no user namespace — and is taken only when its probe's
canary write is denied by the kernel; the slot stays sandboxed and says
so (`[SANDBOX_FALLBACK]` on stdout, `sandboxed by fallback rung: landlock` on
the report's model labels, `sandbox-fallback landlock <spec>` in `status`). Rung 2 runs `--sandbox danger-full-access` in the real
repo root, i.e. with the same rights as your shell — the footing cursor-agent,
pi, claude-code and muse have always had in those phases — and every artifact
of such a slot says so (`[SANDBOX_BYPASSED]` on stdout,
`ran unsandboxed: sandbox_fallback=auto` on the report's model labels, a
`git status` check that fails the slot by path if the tree changed). `auto`
takes rung 1, then rung 2; `contained` takes rung 1 only and otherwise refuses;
`off` restores the refusal: codex is left out on such a machine until the
repair is installed. A run that falls to rung 2 where the previous run of the
phase took rung 1 says so once, on its own `[SANDBOX_RUNG_RETIRED]` line —
never a second `[SANDBOX_BYPASSED]`, which stays one line per slot. doctor's BROKEN row names the value in force and the rung
it selected. The write phases are refused under a broken sandbox whatever the
key says — no rung serves them, so the AppArmor repair is still their fix.

**Never enable landlock globally.** `codex features enable
use_legacy_landlock` makes every `workspace-write` dispatch panic (exit 101),
write phases included; the flag is only safe as the per-dispatch `-c` rung 1
adds, and `doctor` warns if your codex config sets it. Rung 1 is read-only only
and rides a feature codex has marked deprecated (measured at 0.155.1); when it
is removed, `auto` falls back to rung 2 and `contained` to refusal.

Because the default is already the widest value, a cloned repository's
`.multi-llm/providers.yaml` can only narrow the fallback — there is nothing for
an untrusted override to widen here. See the key's comment in `providers.yaml`
for what each rung gives up.
